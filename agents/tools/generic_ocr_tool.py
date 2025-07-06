"""
Generic OCR Tool - Domain-agnostic document processing and text extraction.

This tool provides core OCR capabilities without any domain-specific logic:
- PDF and image processing (JPEG, PNG)
- Multi-language OCR (English, Hindi, Bengali, Tamil)
- Table extraction and detection
- Image preprocessing and optimization
- Raw text and structured data output

The tool outputs raw extracted data that can be consumed by any domain-specific agent.
"""

import asyncio
import io
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from decimal import Decimal
import re

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import httpx
from pydantic import BaseModel, Field
import camelot
import pandas as pd
import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class OCRProcessingError(Exception):
    """Base exception for OCR processing errors."""
    pass


class FileValidationError(OCRProcessingError):
    """Raised when file validation fails."""
    pass


class ExtractedTable(BaseModel):
    """A table extracted from a document."""
    page_number: int
    table_index: int
    headers: List[str]
    rows: List[List[str]]
    confidence: float
    extraction_method: str  # 'camelot' or 'regex'
    bounding_box: Optional[Dict[str, float]] = None


class OCRProcessingStats(BaseModel):
    """Statistics about OCR processing."""
    pages_processed: int
    ocr_confidence: float
    text_extracted_chars: int
    tables_found: int
    tables_extracted: int = 0
    processing_time_ms: int
    errors_encountered: List[str] = []


class ExtractedDocument(BaseModel):
    """Complete extracted document with raw OCR data."""
    raw_text: str
    tables: List[ExtractedTable] = []
    pages: List[str] = []  # Text per page
    language_detected: str
    stats: OCRProcessingStats
    metadata: Dict[str, Any] = {}


class GenericOCRTool:
    """
    Generic OCR tool for document processing without domain-specific logic.
    
    Features:
    - Multi-format support (PDF, JPG, PNG)
    - Advanced image preprocessing with multiple strategies
    - Multi-language OCR with automatic best-strategy selection
    - Table detection and extraction
    - High-quality text extraction
    - Comprehensive error handling
    """
    
    # Supported file formats with their magic bytes
    FORMAT_SIGNATURES = {
        'pdf': [b'%PDF-'],
        'jpg': [b'\xff\xd8\xff\xe0', b'\xff\xd8\xff\xe1', b'\xff\xd8\xff\xdb', 
                b'\xff\xd8\xff\xc0', b'\xff\xd8\xff\xc2', b'\xff\xd8\xff\xc4'],
        'jpeg': [b'\xff\xd8\xff\xe0', b'\xff\xd8\xff\xe1', b'\xff\xd8\xff\xdb',
                 b'\xff\xd8\xff\xc0', b'\xff\xd8\xff\xc2', b'\xff\xd8\xff\xc4'],
        'png': [b'\x89PNG\r\n\x1a\n']
    }
    
    # Image quality thresholds
    MIN_IMAGE_WIDTH = 800
    MIN_IMAGE_HEIGHT = 600
    MAX_IMAGE_WIDTH = 4000
    MAX_IMAGE_HEIGHT = 4000
    
    def __init__(
        self,
        supported_formats: List[str] = None,
        max_file_size: int = 15 * 1024 * 1024,
        confidence_threshold: float = 60,
        enable_camelot: bool = True
    ):
        """Initialize the generic OCR tool."""
        self.supported_formats = supported_formats or ['pdf', 'jpg', 'jpeg', 'png']
        self.max_file_size = max_file_size
        self.confidence_threshold = confidence_threshold
        self.enable_camelot = enable_camelot
        
        # Validate supported formats
        for fmt in self.supported_formats:
            if fmt not in self.FORMAT_SIGNATURES:
                raise ValueError(f"Unsupported format: {fmt}")
        
        logger.info("Initialized GenericOCRTool")
    
    async def __call__(
        self,
        file_content: bytes,
        doc_id: str,
        language: str = "english",
        file_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a document and extract raw OCR data.
        
        Args:
            file_content: Binary content of the document file
            doc_id: Unique document identifier
            language: Document language for OCR processing
            file_format: Optional file format hint
            
        Returns:
            Dict containing raw extracted document information
        """
        try:
            logger.info(
                "Starting OCR processing",
                doc_id=doc_id,
                language=language,
                file_size=len(file_content)
            )
            
            # Convert language string to enum
            lang_enum = self._convert_language(language)
            
            # Process the document
            extracted_doc = await self.process_document(
                file_content=file_content,
                doc_id=doc_id,
                language=lang_enum,
                file_format=file_format
            )
            
            # Return structured result
            result = {
                "success": True,
                "doc_id": doc_id,
                "raw_text": extracted_doc.raw_text,
                "pages": extracted_doc.pages,
                "tables": [table.dict() for table in extracted_doc.tables],
                "language_detected": extracted_doc.language_detected,
                "processing_stats": extracted_doc.stats.dict(),
                "metadata": extracted_doc.metadata
            }
            
            logger.info(
                "OCR processing completed",
                doc_id=doc_id,
                text_chars=len(extracted_doc.raw_text),
                tables_found=len(extracted_doc.tables),
                processing_time_ms=extracted_doc.stats.processing_time_ms
            )
            
            return result
                    
        except Exception as e:
            error_msg = f"OCR processing failed: {str(e)}"
            logger.error(error_msg, doc_id=doc_id, exc_info=True)
            
            return {
                "success": False,
                "doc_id": doc_id,
                "error": error_msg,
                "raw_text": "",
                "pages": [],
                "tables": [],
                "language_detected": "unknown",
                "processing_stats": {
                    "pages_processed": 0,
                    "ocr_confidence": 0.0,
                    "text_extracted_chars": 0,
                    "tables_found": 0,
                    "tables_extracted": 0,
                    "processing_time_ms": 0,
                    "errors_encountered": [error_msg]
                },
                "metadata": {}
            }
    
    def _convert_language(self, language: str) -> str:
        """Convert language string to tesseract language code."""
        lang_map = {
            "english": "eng",
            "hindi": "hin", 
            "bengali": "ben",
            "tamil": "tam"
        }
        return lang_map.get(language.lower(), "eng")
    
    async def process_document(
        self,
        file_content: bytes,
        doc_id: str,
        language: str = "eng",
        file_format: Optional[str] = None
    ) -> ExtractedDocument:
        """
        Process a document and extract all OCR data.
        
        Args:
            file_content: Binary content of the document
            doc_id: Unique document identifier  
            language: Tesseract language code
            file_format: Optional file format hint
            
        Returns:
            ExtractedDocument with all OCR data
        """
        start_time = asyncio.get_event_loop().time()
        errors_encountered = []
        
        try:
            # Validate file
            validated_format = await self._validate_file(file_content, file_format)
            
            # Convert to images
            images = await self._convert_to_images(file_content, validated_format)
            
            # Extract text with OCR
            pages_text = []
            total_confidence = 0.0
            
            for i, image in enumerate(images):
                try:
                    page_text, confidence = await self._extract_text_ocr(image, language)
                    pages_text.append(page_text)
                    total_confidence += confidence
                except Exception as e:
                    error_msg = f"OCR failed for page {i+1}: {str(e)}"
                    logger.warning(error_msg)
                    errors_encountered.append(error_msg)
                    pages_text.append("")
            
            # Combine all text
            raw_text = '\n\n'.join(pages_text)
            avg_confidence = total_confidence / len(images) if images else 0.0
            
            # Extract tables
            tables = []
            try:
                if validated_format == 'pdf':
                    tables = await self._extract_tables_camelot(file_content)
            except Exception as e:
                error_msg = f"Table extraction failed: {str(e)}"
                logger.warning(error_msg)
                errors_encountered.append(error_msg)
            
            # Calculate processing time
            end_time = asyncio.get_event_loop().time()
            processing_time_ms = int((end_time - start_time) * 1000)
            
            # Create processing stats
            stats = OCRProcessingStats(
                pages_processed=len(images),
                ocr_confidence=avg_confidence,
                text_extracted_chars=len(raw_text),
                tables_found=len(tables),
                tables_extracted=len(tables),
                processing_time_ms=processing_time_ms,
                errors_encountered=errors_encountered
            )
            
            # Create extracted document
            extracted_doc = ExtractedDocument(
                raw_text=raw_text,
                tables=tables,
                pages=pages_text,
                language_detected=language,
                stats=stats,
                metadata={
                    "file_format": validated_format,
                    "total_pages": len(images),
                    "avg_confidence": avg_confidence
                }
            )
            
            return extracted_doc
            
        except Exception as e:
            error_msg = f"Document processing failed: {str(e)}"
            logger.error(error_msg, doc_id=doc_id, exc_info=True)
            raise OCRProcessingError(error_msg) from e
    
    def _detect_format(self, data: bytes) -> str:
        """Detect file format from magic bytes."""
        for fmt, signatures in self.FORMAT_SIGNATURES.items():
            for signature in signatures:
                if data.startswith(signature):
                    return fmt
        
        # Fallback detection
        if b'PDF' in data[:100]:
            return 'pdf'
        elif data.startswith(b'\xff\xd8'):
            return 'jpg'
        elif data.startswith(b'\x89PNG'):
            return 'png'
        
        raise ValueError("Unknown file format")
    
    async def _validate_file(self, data: bytes, format_hint: str = None) -> str:
        """Validate file format and size."""
        # Detect format
        if format_hint and format_hint in self.supported_formats:
            file_format = format_hint
        else:
            file_format = self._detect_format(data)
        
        # Validate format is supported
        if file_format not in self.supported_formats:
            raise FileValidationError(f"Unsupported file format: {file_format}")
        
        # Validate file size
        if len(data) > self.max_file_size:
            size_mb = len(data) / (1024 * 1024)
            max_mb = self.max_file_size / (1024 * 1024)
            raise FileValidationError(f"File too large: {size_mb:.1f}MB > {max_mb:.1f}MB")
        
        return file_format
    
    async def _convert_to_images(self, data: bytes, file_format: str) -> List[Image.Image]:
        """Convert file data to list of PIL Images."""
        images = []
        
        if file_format == 'pdf':
            # Convert PDF pages to images
            try:
                with fitz.open(stream=data, filetype="pdf") as doc:
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        # Render at 2x resolution for better OCR
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        images.append(image)
                        
            except Exception as e:
                raise OCRProcessingError(f"Failed to convert PDF to images: {e}")
                
        elif file_format in ['jpg', 'jpeg', 'png']:
            # Single image file
            try:
                image = Image.open(io.BytesIO(data))
                images.append(image)
                
            except Exception as e:
                raise OCRProcessingError(f"Failed to load {file_format.upper()} image: {e}")
        
        else:
            raise ValueError(f"Cannot convert format '{file_format}' to images")
        
        logger.debug(f"Converted {file_format.upper()} to {len(images)} image(s)")
        return images
    
    async def _extract_text_ocr(self, image: Image.Image, language: str = "eng") -> Tuple[str, float]:
        """Extract text using OCR with advanced preprocessing."""
        if image is None:
            raise OCRProcessingError("Image is None")
        
        # Preprocess image for better OCR
        processed_image = self._preprocess_image(image)
        
        # OCR configuration optimized for documents
        ocr_configs = [
            r'--oem 3 --psm 6',  # Uniform block of text
            r'--oem 3 --psm 4',  # Single column of text
            r'--oem 3 --psm 1',  # Automatic page segmentation with OSD
            r'--oem 3 --psm 3'   # Fully automatic page segmentation
        ]
        
        best_text = ""
        best_confidence = 0.0
        
        for config in ocr_configs:
            try:
                data = pytesseract.image_to_data(
                    processed_image,
                    lang=language,
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Calculate confidence and extract text
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                if not confidences:
                    continue
                    
                avg_conf = sum(confidences) / len(confidences)
                
                # Extract text above confidence threshold
                text_parts = []
                for j in range(len(data['text'])):
                    text = data['text'][j].strip()
                    conf = int(data['conf'][j])
                    
                    if text and conf > self.confidence_threshold:
                        text_parts.append(text)
                
                if avg_conf > best_confidence and text_parts:
                    best_confidence = avg_conf
                    best_text = ' '.join(text_parts)
                    
            except Exception as e:
                logger.debug(f"OCR config failed: {e}")
                continue
        
        if not best_text:
            raise OCRProcessingError("No text extracted above confidence threshold")
        
        return best_text, best_confidence / 100.0
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Advanced image preprocessing for better OCR."""
        try:
            # Convert PIL to OpenCV format
            opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply multiple preprocessing strategies and select best
            strategies = [
                self._preprocess_strategy_basic(opencv_img),
                self._preprocess_strategy_adaptive(opencv_img),
                self._preprocess_strategy_morphological(opencv_img),
                self._preprocess_strategy_denoising(opencv_img)
            ]
            
            # Test each strategy with quick OCR check
            best_image = strategies[0]  # Default fallback
            best_confidence = 0
            
            for processed_img in strategies:
                try:
                    # Convert back to PIL for quick test
                    pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                    
                    # Quick confidence check on small sample
                    width, height = pil_img.size
                    crop_box = (width // 4, height // 4, 3 * width // 4, 3 * height // 4)
                    sample_img = pil_img.crop(crop_box)
                    
                    data = pytesseract.image_to_data(
                        sample_img,
                        config='--psm 6',
                        output_type=pytesseract.Output.DICT
                    )
                    
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        best_image = processed_img
                        
                except Exception:
                    continue
            
            # Convert back to PIL
            final_image = Image.fromarray(cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB))
            return final_image
            
        except Exception as e:
            logger.warning(f"Advanced preprocessing failed: {e}, using basic fallback")
            return self._preprocess_basic_fallback(image)
    
    def _preprocess_strategy_basic(self, img: np.ndarray) -> np.ndarray:
        """Basic preprocessing strategy."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Resize if too small
        height, width = gray.shape
        if width < self.MIN_IMAGE_WIDTH or height < self.MIN_IMAGE_HEIGHT:
            scale_factor = max(
                self.MIN_IMAGE_WIDTH / width,
                self.MIN_IMAGE_HEIGHT / height
            )
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Denoise
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_strategy_adaptive(self, img: np.ndarray) -> np.ndarray:
        """Adaptive thresholding strategy."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Resize if needed
        height, width = gray.shape
        if width < self.MIN_IMAGE_WIDTH or height < self.MIN_IMAGE_HEIGHT:
            scale_factor = max(
                self.MIN_IMAGE_WIDTH / width,
                self.MIN_IMAGE_HEIGHT / height
            )
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        
        return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_strategy_morphological(self, img: np.ndarray) -> np.ndarray:
        """Morphological operations strategy."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Resize if needed
        height, width = gray.shape
        if width < self.MIN_IMAGE_WIDTH or height < self.MIN_IMAGE_HEIGHT:
            scale_factor = max(
                self.MIN_IMAGE_WIDTH / width,
                self.MIN_IMAGE_HEIGHT / height
            )
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # Otsu's thresholding
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)
        
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_strategy_denoising(self, img: np.ndarray) -> np.ndarray:
        """Advanced denoising strategy."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Resize if needed
        height, width = gray.shape
        if width < self.MIN_IMAGE_WIDTH or height < self.MIN_IMAGE_HEIGHT:
            scale_factor = max(
                self.MIN_IMAGE_WIDTH / width,
                self.MIN_IMAGE_HEIGHT / height
            )
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Unsharp masking
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        unsharp = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        
        return cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_basic_fallback(self, image: Image.Image) -> Image.Image:
        """Basic fallback preprocessing."""
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize if too small
        width, height = image.size
        if width < self.MIN_IMAGE_WIDTH or height < self.MIN_IMAGE_HEIGHT:
            scale_factor = max(
                self.MIN_IMAGE_WIDTH / width,
                self.MIN_IMAGE_HEIGHT / height
            )
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Enhance contrast and sharpness
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Reduce noise
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return image
    
    async def _extract_tables_camelot(self, pdf_data: bytes) -> List[ExtractedTable]:
        """Extract tables from PDF using Camelot."""
        if not self.enable_camelot:
            return []
        
        tables = []
        
        try:
            # Save PDF to temporary file for Camelot
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_data)
                temp_path = temp_file.name
            
            try:
                # Extract tables with Camelot
                camelot_tables = camelot.read_pdf(temp_path, pages='all', flavor='lattice')
                
                for i, table in enumerate(camelot_tables):
                    if table.df.empty:
                        continue
                    
                    # Convert DataFrame to our format
                    headers = table.df.iloc[0].tolist()
                    rows = table.df.iloc[1:].values.tolist()
                    
                    # Clean headers and rows
                    headers = [str(h).strip() for h in headers]
                    rows = [[str(cell).strip() for cell in row] for row in rows]
                    
                    extracted_table = ExtractedTable(
                        page_number=table.page,
                        table_index=i,
                        headers=headers,
                        rows=rows,
                        confidence=table.accuracy / 100.0,
                        extraction_method='camelot'
                    )
                    
                    tables.append(extracted_table)
                    
            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.warning(f"Camelot table extraction failed: {e}")
        
        return tables 