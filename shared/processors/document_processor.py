"""
Document Processing & OCR Pipeline for MedBillGuardAgent.

This module handles:
- PDF and image processing (JPEG, PNG)
- Multi-language OCR (English, Hindi, Bengali, Tamil)
- Table extraction and line-item parsing
- Document type detection
- Robust error handling for corrupt files and OCR failures
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
from pydantic import BaseModel
import camelot
import pandas as pd
import cv2
import numpy as np

from shared.schemas.schemas import DocumentType, Language, LineItemType

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass


class FileValidationError(DocumentProcessingError):
    """Raised when file validation fails."""
    pass


class OCRError(DocumentProcessingError):
    """Raised when OCR processing fails."""
    pass


class TableExtractionError(DocumentProcessingError):
    """Raised when table extraction fails."""
    pass


class ExtractedLineItem(BaseModel):
    """A single line item extracted from a document."""
    description: str
    quantity: int = 1
    unit_price: Optional[Decimal] = None
    total_amount: Decimal
    item_type: LineItemType = LineItemType.OTHER
    confidence: float = 0.0
    source_table: Optional[int] = None
    source_method: str = "regex"


class ExtractedTable(BaseModel):
    """A table extracted from a document."""
    page_number: int
    table_index: int
    headers: List[str]
    rows: List[List[str]]
    confidence: float
    extraction_method: str  # 'camelot' or 'regex'
    bounding_box: Optional[Dict[str, float]] = None


class ProcessingStats(BaseModel):
    """Statistics about document processing."""
    pages_processed: int
    ocr_confidence: float
    text_extracted_chars: int
    tables_found: int
    tables_extracted: int = 0
    line_items_found: int
    processing_time_ms: int
    errors_encountered: List[str] = []  # Track errors for debugging


class ExtractedDocument(BaseModel):
    """Complete extracted document with all information."""
    raw_text: str
    tables: List[ExtractedTable] = []
    line_items: List[ExtractedLineItem]
    document_type: DocumentType
    language: Language
    stats: ProcessingStats
    metadata: Dict[str, Any] = {}


class DocumentProcessor:
    """Main document processing pipeline."""
    
    # Supported file formats with their magic bytes
    FORMAT_SIGNATURES = {
        'pdf': [b'%PDF-'],
        'jpg': [b'\xff\xd8\xff\xe0', b'\xff\xd8\xff\xe1', b'\xff\xd8\xff\xdb', 
                b'\xff\xd8\xff\xc0', b'\xff\xd8\xff\xc2', b'\xff\xd8\xff\xc4'],
        'jpeg': [b'\xff\xd8\xff\xe0', b'\xff\xd8\xff\xe1', b'\xff\xd8\xff\xdb',
                 b'\xff\xd8\xff\xc0', b'\xff\xd8\xff\xc2', b'\xff\xd8\xff\xc4'],
        'png': [b'\x89PNG\r\n\x1a\n']
    }
    
    # Maximum file sizes by format (in bytes)
    MAX_FILE_SIZES = {
        'pdf': 15 * 1024 * 1024,  # 15 MB
        'jpg': 15 * 1024 * 1024,  # 15 MB
        'jpeg': 15 * 1024 * 1024, # 15 MB
        'png': 15 * 1024 * 1024   # 15 MB
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
        """Initialize the document processor.
        
        Args:
            supported_formats: List of supported file formats
            max_file_size: Maximum file size in bytes
            confidence_threshold: Minimum OCR confidence threshold
            enable_camelot: Whether to enable Camelot table extraction
        """
        self.supported_formats = supported_formats or ['pdf', 'jpg', 'jpeg', 'png']
        self.max_file_size = max_file_size
        self.confidence_threshold = confidence_threshold
        # Add camera-specific thresholds
        self.camera_confidence_threshold = 25  # Lower threshold for camera images
        self.scanned_confidence_threshold = 60  # Higher threshold for scanned docs
        self.enable_camelot = enable_camelot
        
        # Initialize language support
        self.language_support = {
            Language.ENGLISH: 'eng',
            Language.HINDI: 'hin',
            Language.BENGALI: 'ben',
            Language.TAMIL: 'tam'
        }
        
        logger.info(f"DocumentProcessor initialized with formats: {self.supported_formats}, max_size: {self.max_file_size}MB, confidence_threshold: {self.confidence_threshold}%")
        
        # Validate supported formats
        for fmt in self.supported_formats:
            if fmt not in self.FORMAT_SIGNATURES:
                raise ValueError(f"Unsupported format: {fmt}")
        
        # Line item extraction patterns
        self.line_item_patterns = [
            # Standard bill format: Description Qty Rate Amount
            re.compile(r'(.+?)\s+(\d+)\s+(?:Rs\.?\s*)?(\d+(?:\.\d{2})?)\s+(?:Rs\.?\s*)?(\d+(?:\.\d{2})?)', re.IGNORECASE),
            # Simple format: Description Amount
            re.compile(r'(.+?)\s+(?:Rs\.?\s*)?(\d+(?:,\d{3})*(?:\.\d{2})?)', re.IGNORECASE),
            # With currency symbols
            re.compile(r'(.+?)\s+[₹Rs\.]+\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', re.IGNORECASE),
            # Pharmacy format with MRP
            re.compile(r'(.+?)\s+MRP\s*[₹Rs\.]*\s*(\d+(?:\.\d{2})?)', re.IGNORECASE)
        ]
        
        # Table detection patterns
        self.table_patterns = [
            re.compile(r'^\s*(?:Description|Item|Service|Particulars)\s+(?:Qty|Quantity)?\s*(?:Rate|Price|Amount)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*(?:Sr\.?\s*No\.?|S\.?\s*No\.?)\s+(?:Description|Item)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*(?:Medicine|Drug|Test)\s+(?:Qty|Quantity)\s+(?:Rate|Price)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*(?:Procedure|Treatment)\s+(?:Charges?|Amount|Fee)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*(?:Room|Ward)\s+(?:Charges?|Rent)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*(?:Total|Sub\s*Total|Grand\s*Total)\s*[:\-]?\s*(?:Rs\.?|₹)?\s*\d+', re.IGNORECASE | re.MULTILINE)
        ]
        
        # Document type patterns
        self.doc_type_patterns = {
            DocumentType.HOSPITAL_BILL: [
                'hospital', 'nursing home', 'medical center', 'clinic', 
                'admission', 'patient', 'consultation', 'doctor', 'physician',
                'room charges', 'bed charges', 'nursing charges', 'operation'
            ],
            DocumentType.PHARMACY_INVOICE: [
                'pharmacy', 'chemist', 'medical store', 'drugstore',
                'tablet', 'capsule', 'syrup', 'medicine', 'drug',
                'mrp', 'batch', 'expiry', 'prescription'
            ],
            DocumentType.DIAGNOSTIC_REPORT: [
                'laboratory', 'pathology', 'radiology', 'diagnostic',
                'blood test', 'urine test', 'scan', 'x-ray', 'ct scan',
                'mri', 'ultrasound', 'report', 'specimen', 'sample'
            ]
        }
        
        # Line item classification patterns
        self.line_item_classification = {
            LineItemType.CONSULTATION: [
                'consultation', 'doctor', 'physician', 'specialist', 'visit',
                'opd', 'checkup', 'examination', 'consultation fee'
            ],
            LineItemType.DIAGNOSTIC: [
                'test', 'blood', 'urine', 'scan', 'x-ray', 'ct', 'mri',
                'ultrasound', 'ecg', 'echo', 'pathology', 'lab', 'specimen'
            ],
            LineItemType.MEDICATION: [
                'tablet', 'capsule', 'syrup', 'injection', 'medicine',
                'drug', 'antibiotic', 'painkiller', 'supplement'
            ],
            LineItemType.PROCEDURE: [
                'surgery', 'operation', 'procedure', 'treatment', 'therapy',
                'dressing', 'suture', 'biopsy', 'endoscopy'
            ],
            LineItemType.ROOM_CHARGE: [
                'room', 'bed', 'ward', 'icu', 'accommodation', 'stay',
                'nursing', 'attendant', 'charges'
            ]
        }
    
    def _detect_format(self, data: bytes) -> str:
        """Detect file format from binary data using magic bytes.
        
        Args:
            data: Binary file data
            
        Returns:
            Detected format or 'unknown'
        """
        if not data or len(data) < 8:
            return 'unknown'
            
        # Check first 8 bytes against known signatures
        header = data[:8]
        
        for fmt, signatures in self.FORMAT_SIGNATURES.items():
            for sig in signatures:
                if header.startswith(sig) or data.startswith(sig):
                    return fmt
        
        # Additional checks for edge cases
        try:
            # Try to detect JPEG variants
            if data.startswith(b'\xff\xd8') and b'JFIF' in data[:100]:
                return 'jpg'
            if data.startswith(b'\xff\xd8') and b'Exif' in data[:100]:
                return 'jpg'
                
            # Check for PNG with different chunk ordering
            if b'PNG' in data[:16] and b'IHDR' in data[:32]:
                return 'png'
                
            # Check for PDF with whitespace variations
            if b'%PDF' in data[:32]:
                return 'pdf'
                
        except Exception as e:
            logger.warning(f"Error in format detection: {e}")
        
        return 'unknown'
    
    def _validate_file_format(self, data: bytes, format_hint: str = None) -> str:
        """Validate and return the actual file format.
        
        Args:
            data: Binary file data
            format_hint: Hint about expected format from filename
            
        Returns:
            Validated format
            
        Raises:
            ValueError: If format is unsupported or invalid
        """
        detected_format = self._detect_format(data)
        
        if detected_format == 'unknown':
            raise ValueError(f"Unable to detect file format. Data starts with: {data[:16].hex()}")
        
        if detected_format not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {detected_format}. Supported: {self.supported_formats}")
        
        # Warn if format hint doesn't match detection
        if format_hint and format_hint != detected_format:
            logger.warning(f"Format hint '{format_hint}' doesn't match detected format '{detected_format}'")
        
        return detected_format
    
    def _validate_file_size(self, data: bytes, file_format: str, filename: str = "unknown") -> None:
        """Validate file size against format-specific limits.
        
        Args:
            data: Binary file data
            file_format: Detected file format
            filename: Original filename for error messages
            
        Raises:
            ValueError: If file is too large
        """
        file_size = len(data)
        max_size = self.MAX_FILE_SIZES.get(file_format, self.max_file_size)
        
        if file_size > max_size:
            raise ValueError(
                f"File '{filename}' size ({file_size:,} bytes) exceeds "
                f"limit for {file_format.upper()} files ({max_size:,} bytes)"
            )
        
        # Warn for very small files that might be corrupted
        min_size = 1024  # 1 KB
        if file_size < min_size:
            logger.warning(f"File '{filename}' is very small ({file_size} bytes), might be corrupted")
    
    def _validate_image_quality(self, image: Image.Image, filename: str = "unknown") -> None:
        """Validate image quality and dimensions.
        
        Args:
            image: PIL Image object
            filename: Original filename for error messages
            
        Raises:
            ValueError: If image quality is insufficient
        """
        width, height = image.size
        
        # Check minimum dimensions
        if width < self.MIN_IMAGE_WIDTH or height < self.MIN_IMAGE_HEIGHT:
            logger.warning(
                f"Image '{filename}' has low resolution ({width}x{height}). "
                f"Minimum recommended: {self.MIN_IMAGE_WIDTH}x{self.MIN_IMAGE_HEIGHT}"
            )
        
        # Check maximum dimensions
        if width > self.MAX_IMAGE_WIDTH or height > self.MAX_IMAGE_HEIGHT:
            logger.warning(
                f"Image '{filename}' has very high resolution ({width}x{height}). "
                f"This may slow down processing."
            )
        
        # Check aspect ratio (very wide or tall images might be problematic)
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 10:
            logger.warning(
                f"Image '{filename}' has extreme aspect ratio ({aspect_ratio:.1f}:1). "
                f"This might affect OCR accuracy."
            )
    
    def _validate_pdf_structure(self, data: bytes, filename: str = "unknown") -> None:
        """Validate PDF structure and readability.
        
        Args:
            data: PDF binary data
            filename: Original filename for error messages
            
        Raises:
            ValueError: If PDF is corrupted or unreadable
        """
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                page_count = len(doc)
                
                if page_count == 0:
                    raise ValueError(f"PDF '{filename}' contains no pages")
                
                if page_count > 50:
                    logger.warning(f"PDF '{filename}' has many pages ({page_count}). Processing may be slow.")
                
                # Try to access first page to check for corruption
                try:
                    first_page = doc[0]
                    # Try to get page dimensions
                    rect = first_page.rect
                    if rect.width <= 0 or rect.height <= 0:
                        raise ValueError(f"PDF '{filename}' has invalid page dimensions")
                        
                except Exception as e:
                    raise ValueError(f"PDF '{filename}' appears to be corrupted: {e}")
                    
        except fitz.FileDataError as e:
            raise ValueError(f"PDF '{filename}' is corrupted or not a valid PDF: {e}")
        except Exception as e:
            raise ValueError(f"Error validating PDF '{filename}': {e}")
    
    async def _validate_file(self, data: bytes, filename: str = "unknown") -> str:
        """Comprehensive file validation with robust error handling.
        
        Args:
            data: Binary file data
            filename: Original filename for error messages
            
        Returns:
            Validated file format
            
        Raises:
            FileValidationError: If file is invalid
        """
        try:
            # Detect and validate format
            file_format = self._validate_file_format(data, self._get_format_hint(filename))
            
            # Validate file size
            self._validate_file_size(data, file_format, filename)
            
            # Format-specific validation
            if file_format == 'pdf':
                self._validate_pdf_structure(data, filename)
            elif file_format in ['jpg', 'jpeg', 'png']:
                try:
                    image = Image.open(io.BytesIO(data))
                    self._validate_image_quality(image, filename)
                except Exception as e:
                    raise FileValidationError(f"Invalid {file_format.upper()} image '{filename}': {e}")
            
            logger.info(f"File '{filename}' validated successfully as {file_format.upper()} ({len(data):,} bytes)")
            return file_format
            
        except (ValueError, FileValidationError) as e:
            logger.error(f"File validation failed for '{filename}': {e}")
            raise FileValidationError(f"File validation failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during file validation for '{filename}': {e}")
            raise FileValidationError(f"Unexpected validation error: {e}")
    
    def _get_format_hint(self, filename: str) -> Optional[str]:
        """Get format hint from filename extension.
        
        Args:
            filename: Original filename
            
        Returns:
            Format hint or None
        """
        if not filename or '.' not in filename:
            return None
            
        ext = filename.lower().split('.')[-1]
        
        # Map common extensions to formats
        ext_map = {
            'pdf': 'pdf',
            'jpg': 'jpg',
            'jpeg': 'jpg',
            'png': 'png'
        }
        
        return ext_map.get(ext)
    
    async def _download_file(self, file_path: str) -> bytes:
        """Download file from various sources.
        
        Args:
            file_path: Local path, HTTP URL, or S3 URL
            
        Returns:
            Binary file data
            
        Raises:
            Exception: If download fails
        """
        if file_path.startswith(('http://', 'https://')):
            # HTTP/HTTPS URL
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(file_path)
                response.raise_for_status()
                return response.content
                
        elif file_path.startswith('s3://'):
            # S3 URL (placeholder for future implementation)
            raise NotImplementedError("S3 downloads not yet implemented")
            
        else:
            # Local file path
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
                
            return path.read_bytes()
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Advanced image preprocessing for medical bills with multiple strategies.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Best preprocessed PIL Image based on OCR confidence
        """
        try:
            # Convert PIL to OpenCV format
            opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply multiple preprocessing strategies
            strategies = [
                self._preprocess_strategy_1_basic(opencv_img),
                self._preprocess_strategy_2_adaptive(opencv_img),
                self._preprocess_strategy_3_morphological(opencv_img),
                self._preprocess_strategy_4_denoising(opencv_img)
            ]
            
            # Test each strategy with a quick OCR check
            best_image = None
            best_confidence = 0
            
            for i, processed_img in enumerate(strategies):
                try:
                    # Convert back to PIL for OCR test
                    pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                    
                    # Quick OCR confidence check (sample small area)
                    width, height = pil_img.size
                    # Test center region (usually contains important text)
                    crop_box = (
                        width // 4, height // 4,
                        3 * width // 4, 3 * height // 4
                    )
                    sample_img = pil_img.crop(crop_box)
                    
                    # Get OCR confidence for this strategy
                    data = pytesseract.image_to_data(
                        sample_img,
                        config='--psm 6',
                        output_type=pytesseract.Output.DICT
                    )
                    
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    logger.debug(f"Strategy {i+1} confidence: {avg_confidence:.1f}")
                    
                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        best_image = processed_img
                        
                except Exception as e:
                    logger.debug(f"Strategy {i+1} failed: {e}")
                    continue
            
            # If no strategy worked well, use the first one as fallback
            if best_image is None:
                best_image = strategies[0]
                logger.warning("All preprocessing strategies had low confidence, using basic strategy")
            
            # Convert back to PIL
            final_image = Image.fromarray(cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB))
            
            logger.debug(f"Best preprocessing strategy achieved {best_confidence:.1f}% confidence")
            return final_image
            
        except Exception as e:
            logger.error(f"Advanced preprocessing failed: {e}, falling back to basic preprocessing")
            return self._preprocess_basic_fallback(image)
    
    def _preprocess_strategy_1_basic(self, img: np.ndarray) -> np.ndarray:
        """Basic preprocessing strategy - contrast + denoising."""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Resize if too small (upscale for better OCR)
        height, width = gray.shape
        if width < self.MIN_IMAGE_WIDTH or height < self.MIN_IMAGE_HEIGHT:
            scale_factor = max(
                self.MIN_IMAGE_WIDTH / width,
                self.MIN_IMAGE_HEIGHT / height
            )
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Denoise
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_strategy_2_adaptive(self, img: np.ndarray) -> np.ndarray:
        """Adaptive thresholding strategy for medical documents."""
        # Convert to grayscale
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
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_strategy_3_morphological(self, img: np.ndarray) -> np.ndarray:
        """Morphological operations strategy for text enhancement."""
        # Convert to grayscale
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
        
        # Apply sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # Otsu's thresholding
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text components
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)
        
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_strategy_4_denoising(self, img: np.ndarray) -> np.ndarray:
        """Advanced denoising strategy for noisy WhatsApp images."""
        # Convert to grayscale
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
        
        # Non-local means denoising (good for preserving text edges)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Enhance contrast with histogram equalization
        equalized = cv2.equalizeHist(denoised)
        
        # Apply unsharp masking for text sharpening
        gaussian = cv2.GaussianBlur(equalized, (9, 9), 10.0)
        unsharp = cv2.addWeighted(equalized, 1.5, gaussian, -0.5, 0)
        
        return cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_basic_fallback(self, image: Image.Image) -> Image.Image:
        """Basic fallback preprocessing if advanced methods fail."""
        # Convert to grayscale for better OCR
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
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Sharpen
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Reduce noise
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return image
    
    async def _convert_to_images(self, data: bytes, file_format: str = None) -> List[Image.Image]:
        """Convert file data to list of PIL Images.
        
        Args:
            data: Binary file data
            file_format: File format (auto-detected if None)
            
        Returns:
            List of PIL Images (one per page for PDFs)
            
        Raises:
            Exception: If conversion fails
        """
        if file_format is None:
            file_format = self._detect_format(data)
        
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
                raise Exception(f"Failed to convert PDF to images: {e}")
                
        elif file_format in ['jpg', 'jpeg', 'png']:
            # Single image file
            try:
                image = Image.open(io.BytesIO(data))
                images.append(image)
                
            except Exception as e:
                raise Exception(f"Failed to load {file_format.upper()} image: {e}")
        
        else:
            raise ValueError(f"Cannot convert format '{file_format}' to images")
        
        logger.info(f"Converted {file_format.upper()} to {len(images)} image(s)")
        return images

    async def _extract_text_ocr(
        self, 
        images: List[Image.Image], 
        language: Language = Language.ENGLISH
    ) -> Tuple[str, float]:
        """Extract text using OCR with multi-language support and robust error handling.
        
        Args:
            images: List of PIL Images
            language: Target language for OCR
            
        Returns:
            Tuple of (extracted_text, confidence_score)
            
        Raises:
            OCRError: If OCR completely fails
        """
        if not images:
            raise OCRError("No images provided for OCR processing")
        
        # First, let's check if Tesseract is available
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {tesseract_version}")
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")
            raise OCRError(f"Tesseract OCR engine not available: {e}")
        
        # Check available languages
        try:
            available_langs = pytesseract.get_languages()
            logger.info(f"Available Tesseract languages: {available_langs}")
        except Exception as e:
            logger.warning(f"Could not check available languages: {e}")
        
        # Language mapping for Tesseract
        lang_map = {
            Language.ENGLISH: 'eng',
            Language.HINDI: 'hin',
            Language.BENGALI: 'ben',
            Language.TAMIL: 'tam'
        }
        
        tesseract_lang = lang_map.get(language, 'eng')
        
        # Check if requested language is available
        try:
            available_langs = pytesseract.get_languages()
            if tesseract_lang not in available_langs:
                logger.warning(f"Language '{tesseract_lang}' not available. Available: {available_langs}")
                if 'eng' in available_langs:
                    tesseract_lang = 'eng'
                    logger.info("Falling back to English")
                else:
                    raise OCRError(f"No suitable language pack available. Available: {available_langs}")
        except Exception as e:
            logger.warning(f"Could not verify language availability: {e}")
            # Continue with requested language
        
        # OCR configuration optimized for medical documents with multiple PSM modes
        ocr_configs = [
            r'--oem 3 --psm 6',  # Uniform block of text (good for medical bills)
            r'--oem 3 --psm 4',  # Single column of text
            r'--oem 3 --psm 1',  # Automatic page segmentation with OSD
            r'--oem 3 --psm 3'   # Fully automatic page segmentation
        ]
        
        all_text = []
        total_confidence = 0.0
        total_words = 0
        failed_pages = 0
        
        for i, image in enumerate(images):
            try:
                # Validate image
                if image is None:
                    logger.warning(f"Page {i+1}: image is None, skipping")
                    failed_pages += 1
                    continue
                
                # Log image details for debugging
                logger.info(f"Page {i+1}: Processing image {image.size} pixels, mode: {image.mode}")
                
                # Detect image type and get appropriate confidence threshold
                image_type = self._detect_image_type(image)
                dynamic_threshold = self._get_dynamic_confidence_threshold(image)
                logger.info(f"Page {i+1}: Image type detected as '{image_type}', using confidence threshold: {dynamic_threshold}%")
                
                # Preprocess image for better OCR
                try:
                    if image_type == 'camera':
                        # Use camera-specific preprocessing
                        processed_image = self._preprocess_camera_image(image)
                        logger.debug(f"Page {i+1}: Applied camera-specific preprocessing")
                    else:
                        # Use standard preprocessing for scanned documents
                        processed_image = self._preprocess_image(image)
                        logger.debug(f"Page {i+1}: Applied standard preprocessing")
                except Exception as e:
                    logger.error(f"Page {i+1}: image preprocessing failed: {e}")
                    # Try with original image
                    processed_image = image
                    # Fall back to original confidence threshold
                    dynamic_threshold = self.confidence_threshold
                
                # Try multiple OCR configurations and pick the best result
                best_data = None
                best_avg_conf = 0
                config_errors = []
                
                for config_idx, config in enumerate(ocr_configs):
                    try:
                        logger.debug(f"Page {i+1}: Trying OCR config {config_idx+1}: {config}")
                        
                        data = pytesseract.image_to_data(
                            processed_image,
                            lang=tesseract_lang,
                            config=config,
                            output_type=pytesseract.Output.DICT
                        )
                        
                        # Calculate average confidence for this config
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        avg_conf = sum(confidences) / len(confidences) if confidences else 0
                        
                        logger.debug(f"Page {i+1}, Config {config_idx+1}: avg confidence {avg_conf:.1f}% ({len(confidences)} words)")
                        
                        if avg_conf > best_avg_conf:
                            best_avg_conf = avg_conf
                            best_data = data
                            
                    except pytesseract.TesseractError as e:
                        error_msg = f"Page {i+1}, Config {config_idx+1}: Tesseract failed: {e}"
                        logger.debug(error_msg)
                        config_errors.append(error_msg)
                        continue
                    except Exception as e:
                        error_msg = f"Page {i+1}, Config {config_idx+1}: OCR failed: {e}"
                        logger.debug(error_msg)
                        config_errors.append(error_msg)
                        continue
                
                # Use the best OCR result
                if best_data is None:
                    logger.error(f"Page {i+1}: All OCR configurations failed")
                    for error in config_errors:
                        logger.error(error)
                    failed_pages += 1
                    continue
                
                data = best_data
                logger.debug(f"Page {i+1}: Using best config with {best_avg_conf:.1f}% confidence")
                
                # Filter out low-confidence words using dynamic threshold
                page_text = []
                page_confidence = []
                
                try:
                    for j in range(len(data['text'])):
                        text = data['text'][j].strip()
                        conf = int(data['conf'][j])
                        
                        # Use dynamic confidence threshold instead of fixed threshold
                        if text and conf > dynamic_threshold:
                            page_text.append(text)
                            page_confidence.append(conf)
                except (KeyError, ValueError, IndexError) as e:
                    logger.error(f"Page {i+1}: error processing OCR data: {e}")
                    failed_pages += 1
                    continue
                
                if page_text:
                    all_text.append(' '.join(page_text))
                    total_confidence += sum(page_confidence)
                    total_words += len(page_confidence)
                    
                    logger.info(f"Page {i+1}: extracted {len(page_text)} words with avg confidence {sum(page_confidence)/len(page_confidence):.1f}% (threshold: {dynamic_threshold}%)")
                else:
                    logger.warning(f"Page {i+1}: no text extracted above confidence threshold ({dynamic_threshold}%)")
                    
                    # For camera images, try with even lower threshold as fallback
                    if image_type == 'camera' and dynamic_threshold > 15:
                        logger.info(f"Page {i+1}: Trying emergency fallback with 15% threshold for camera image")
                        fallback_text = []
                        fallback_confidence = []
                        
                        try:
                            for j in range(len(data['text'])):
                                text = data['text'][j].strip()
                                conf = int(data['conf'][j])
                                
                                if text and conf > 15:  # Emergency threshold
                                    fallback_text.append(text)
                                    fallback_confidence.append(conf)
                        except (KeyError, ValueError, IndexError):
                            pass
                        
                        if fallback_text:
                            all_text.append(' '.join(fallback_text))
                            total_confidence += sum(fallback_confidence)
                            total_words += len(fallback_confidence)
                            logger.info(f"Page {i+1}: emergency fallback extracted {len(fallback_text)} words with avg confidence {sum(fallback_confidence)/len(fallback_confidence):.1f}%")
                        else:
                            # Ultimate fallback: try different preprocessing strategies
                            logger.info(f"Page {i+1}: Trying multiple preprocessing strategies for very difficult camera image")
                            
                            # Try aggressive preprocessing strategies
                            strategies = [
                                ("aggressive_contrast", self._preprocess_camera_image_aggressive),
                                ("edge_preserving", self._preprocess_camera_image_edge_preserving),
                                ("morphological", self._preprocess_camera_image_morphological)
                            ]
                            
                            for strategy_name, strategy_func in strategies:
                                try:
                                    logger.debug(f"Page {i+1}: Trying {strategy_name} preprocessing strategy")
                                    alt_processed = strategy_func(image)
                                    
                                    # Quick OCR test with this strategy
                                    alt_data = pytesseract.image_to_data(
                                        alt_processed,
                                        lang=tesseract_lang,
                                        config=r'--oem 3 --psm 6',
                                        output_type=pytesseract.Output.DICT
                                    )
                                    
                                    alt_text = []
                                    alt_conf = []
                                    
                                    for j in range(len(alt_data['text'])):
                                        text = alt_data['text'][j].strip()
                                        conf = int(alt_data['conf'][j])
                                        
                                        if text and conf > 10:  # Very low threshold
                                            alt_text.append(text)
                                            alt_conf.append(conf)
                                    
                                    if alt_text and len(alt_text) > len(fallback_text):
                                        logger.info(f"Page {i+1}: {strategy_name} strategy extracted {len(alt_text)} words with avg confidence {sum(alt_conf)/len(alt_conf):.1f}%")
                                        all_text.append(' '.join(alt_text))
                                        total_confidence += sum(alt_conf)
                                        total_words += len(alt_conf)
                                        break
                                        
                                except Exception as e:
                                    logger.debug(f"Page {i+1}: {strategy_name} strategy failed: {e}")
                                    continue
                            else:
                                failed_pages += 1
                                logger.warning(f"Page {i+1}: All preprocessing strategies failed - image may be too poor quality")
                    else:
                        failed_pages += 1
            
            except Exception as e:
                logger.error(f"Page {i+1}: unexpected OCR error: {e}")
                failed_pages += 1
                continue
        
        # Check if OCR completely failed
        if not all_text:
            if failed_pages == len(images):
                raise OCRError(f"OCR failed for all {len(images)} pages")
            else:
                raise OCRError("No text extracted from any page")
        
        # Warn if many pages failed
        if failed_pages > 0:
            failure_rate = failed_pages / len(images)
            if failure_rate > 0.5:
                logger.warning(f"High OCR failure rate: {failed_pages}/{len(images)} pages failed ({failure_rate:.1%})")
        
        # Combine all text
        combined_text = '\n\n'.join(all_text)
        
        # Calculate overall confidence
        avg_confidence = (total_confidence / total_words) / 100.0 if total_words > 0 else 0.0
        
        # Try fallback languages if confidence is low
        if avg_confidence < 0.6 and language != Language.ENGLISH:
            try:
                logger.info(f"Low confidence ({avg_confidence:.2f}) with {language.value}, trying English fallback")
                fallback_text, fallback_conf = await self._extract_text_ocr(images, Language.ENGLISH)
                
                if fallback_conf > avg_confidence:
                    logger.info(f"English fallback performed better ({fallback_conf:.2f} vs {avg_confidence:.2f})")
                    return fallback_text, fallback_conf
            except OCRError as e:
                logger.warning(f"English fallback also failed: {e}")
                # Continue with original result
        
        logger.info(f"OCR completed: {len(combined_text)} characters, confidence {avg_confidence:.2f}")
        return combined_text, avg_confidence
    
    def _detect_document_type(self, text: str) -> DocumentType:
        """Detect document type from extracted text.
        
        Args:
            text: Extracted text from document
            
        Returns:
            Detected document type
        """
        text_lower = text.lower()
        
        # Score each document type
        scores = {}
        
        for doc_type, keywords in self.doc_type_patterns.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of each keyword
                count = text_lower.count(keyword.lower())
                score += count
                
                # Bonus for keywords in first 500 characters (header area)
                if keyword.lower() in text_lower[:500]:
                    score += 2
            
            scores[doc_type] = score
        
        # Return the type with highest score
        if scores:
            best_type = max(scores, key=scores.get)
            logger.info(f"Document type detected: {best_type.value} (score: {scores[best_type]})")
            return best_type
        
        # Default fallback
        logger.warning("Could not determine document type, defaulting to hospital bill")
        return DocumentType.HOSPITAL_BILL
    
    def _classify_line_item(self, description: str, doc_type: DocumentType) -> LineItemType:
        """Classify a line item based on its description.
        
        Args:
            description: Item description text
            doc_type: Document type for context
            
        Returns:
            Classified line item type
        """
        desc_lower = description.lower()
        
        # Score each line item type
        scores = {}
        
        for item_type, keywords in self.line_item_classification.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in desc_lower:
                    score += 1
                    # Bonus for exact matches
                    if keyword.lower() == desc_lower:
                        score += 2
            
            scores[item_type] = score
        
        # Return the type with highest score
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        # Document type specific defaults
        if doc_type == DocumentType.PHARMACY_INVOICE:
            return LineItemType.MEDICATION
        elif doc_type == DocumentType.DIAGNOSTIC_REPORT:
            return LineItemType.DIAGNOSTIC
        
        return LineItemType.OTHER
    
    def _is_valid_line_item_text(self, text: str) -> bool:
        """Validate if extracted text looks like a real medical line item.
        
        Args:
            text: Extracted text to validate
            
        Returns:
            True if text appears to be valid line item, False if likely OCR garbage
        """
        if not text or len(text.strip()) < 2:
            return False
            
        text = text.strip()
        
        # Reject single digits or very short fragments
        if len(text) <= 2 and text.isdigit():
            return False
            
        # Reject pure punctuation
        if all(c in '.,()-₹Rs/ ' for c in text):
            return False
            
        # Reject repetitive patterns (common OCR artifacts)
        if len(set(text)) == 1 and len(text) > 2:  # Same character repeated
            return False
            
        # Reject if mostly numbers with no context
        digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
        if digit_ratio > 0.8 and len(text) < 10:
            return False
            
        return True

    def _extract_line_items(self, text: str, doc_type: DocumentType) -> List[ExtractedLineItem]:
        """Extract line items from text using strict patterns for medical bills.
        
        Args:
            text: Extracted text from document
            doc_type: Document type for context
            
        Returns:
            List of validated medical line items only
        """
        items = []
        
        # First, try to find the total bill amount
        total_patterns = [
            r'(?:total|grand\s*total|bill\s*amount|amount\s*payable|net\s*amount)\s*[:\-]?\s*(?:rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]{1,2})?)',
            r'(?:rs\.?|₹)\s*([0-9,]+(?:\.[0-9]{1,2})?)\s*(?:only|total|grand\s*total)',
        ]
        
        total_amount = None
        for pattern in total_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount = self._parse_currency_value(match.group(1))
                if amount and amount > 100:  # Reasonable total amount
                    total_amount = amount
                    logger.info(f"Found total bill amount: ₹{total_amount}")
                    break
            if total_amount:
                break
        
        # Choose patterns based on document type
        if doc_type == DocumentType.PHARMACY_INVOICE:
            # Pharmacy-specific patterns for medicines and pharmacy items
            item_patterns = [
                # Medicine with quantity and price: "Paracetamol 500mg x 10 = ₹150"
                r'([a-zA-Z][a-zA-Z\s\d\-\.]+(?:mg|ml|gm|mcg|tablet|capsule|syrup|drops)[a-zA-Z\s\d\-\.]*)\s*[x×]?\s*(\d+)?\s*[=\-]?\s*(?:rs\.?|₹)\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                # Simple medicine with price: "Crocin Tablet ₹25"
                r'([a-zA-Z][a-zA-Z\s\d\-\.]+(?:tablet|capsule|syrup|drops|cream|ointment|injection)[a-zA-Z\s\d\-\.]*)\s*[:\-]?\s*(?:rs\.?|₹)\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                # Medicine name with dosage and price: "Azithromycin 250mg ₹180"
                r'([a-zA-Z][a-zA-Z\s\d\-\.]+\s+\d+(?:mg|ml|gm|mcg))\s*[:\-]?\s*(?:rs\.?|₹)\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                # Generic medicine pattern: "Medicine Name Price"
                r'([a-zA-Z][a-zA-Z\s\d\-\.]{4,30})\s+(?:rs\.?|₹)\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                # Medicine with MRP: "Item Name MRP ₹200"
                r'([a-zA-Z][a-zA-Z\s\d\-\.]{4,30})\s+(?:mrp|price)\s*[:\-]?\s*(?:rs\.?|₹)\s*([0-9,]+(?:\.[0-9]{1,2})?)',
            ]
        else:
            # Hospital/medical service patterns
            item_patterns = [
                # Consultation fees with specific keywords
                r'((?:consultation|doctor|physician|specialist|opd|checkup|visit)\s*(?:fee|charges?)?)\s*[:\-]?\s*(?:rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                # Room/bed charges  
                r'((?:room|bed|ward|icu|cabin)\s*(?:charges?|rent|fee))\s*[:\-]?\s*(?:rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                # Medical procedures with specific keywords
                r'((?:surgery|operation|procedure|treatment|therapy)\s*(?:charges?|fee)?)\s*[:\-]?\s*(?:rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                # Nursing and medical services
                r'((?:nursing|medical|ambulance|oxygen|injection)\s*(?:charges?|fee|service)?)\s*[:\-]?\s*(?:rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                # Tests and diagnostics with clear amounts
                r'((?:blood|urine|ct|mri|x-ray|scan|test|pathology|lab)\s*(?:test|charges?|fee)?)\s*[:\-]?\s*(?:rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]{1,2})?)',
            ]
        
        for pattern in item_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    description = match.group(1).strip()
                    
                    # Handle different pattern structures
                    if doc_type == DocumentType.PHARMACY_INVOICE:
                        # Pharmacy patterns might have quantity and amount in different groups
                        groups = match.groups()
                        if len(groups) >= 3 and groups[1] and groups[2]:
                            # Pattern with quantity: "Medicine x 10 = ₹150"
                            quantity = self._parse_numeric_value(groups[1]) or 1
                            amount_str = groups[2]
                        elif len(groups) >= 2:
                            # Pattern without quantity: "Medicine ₹150"
                            quantity = 1
                            amount_str = groups[-1]  # Last group is always the amount
                        else:
                            continue
                        
                        # Validate this looks like a pharmacy item
                        if not self._is_valid_pharmacy_item(description):
                            continue
                        
                        source_method = 'pharmacy_regex'
                    else:
                        # Medical service patterns always have description and amount
                        quantity = 1
                        amount_str = match.group(2)
                        
                        # Validate this looks like a medical service
                        if not self._is_valid_medical_service(description):
                            continue
                        
                        source_method = 'medical_regex'
                    
                    amount = self._parse_currency_value(amount_str)
                    if not amount or amount <= 0 or amount > 50000:  # Reasonable range
                        continue
                    
                    # Calculate unit price and total
                    unit_price = amount / quantity if quantity > 1 else amount
                    total_amount = amount
                    
                    item = ExtractedLineItem(
                        description=description,
                        quantity=quantity,
                        unit_price=unit_price,
                        total_amount=total_amount,
                        item_type=self._classify_line_item(description, doc_type),
                        confidence=0.9,  # High confidence for patterns
                        source_method=source_method
                    )
                    
                    items.append(item)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse item from match {match.group()}: {e}")
                    continue
        
        # If we found a total but no individual items, create a summary item
        if total_amount and not items:
            items.append(ExtractedLineItem(
                description=f"{doc_type.value.replace('_', ' ').title()} - Total Amount",
                quantity=1,
                unit_price=total_amount,
                total_amount=total_amount,
                item_type=LineItemType.OTHER,
                confidence=0.8,
                source_method='total_extraction'
            ))
            logger.info(f"Created summary item for total amount: ₹{total_amount}")
        
        # Remove duplicates and validate
        validated_items = self._validate_extracted_items(items, doc_type)
        
        logger.info(f"Extracted {len(validated_items)} validated line items from {len(items)} candidates")
        return validated_items
    
    def _validate_extracted_items(self, items: List[ExtractedLineItem], doc_type: DocumentType) -> List[ExtractedLineItem]:
        """Validation for extracted line items based on document type.
        
        Args:
            items: List of extracted items
            doc_type: Document type for context-specific validation
            
        Returns:
            Validated items only
        """
        validated = []
        
        for item in items:
            # Skip very short descriptions
            if len(item.description) < 3:
                continue
                
            # Skip unreasonable amounts based on document type
            if doc_type == DocumentType.PHARMACY_INVOICE:
                # Pharmacy items can be cheaper (₹5 for basic medicines)
                if item.total_amount < Decimal('5') or item.total_amount > Decimal('10000'):
                    continue
            else:
                # Medical services are typically more expensive
                if item.total_amount < Decimal('10') or item.total_amount > Decimal('50000'):
                    continue
                
            # Skip if description is mostly numbers
            if sum(1 for c in item.description if c.isdigit()) > len(item.description) * 0.5:
                continue
                
            validated.append(item)
        
        return validated
    
    def _is_valid_medical_service(self, description: str) -> bool:
        """Check if description represents a valid medical service.
        
        Args:
            description: Service description
            
        Returns:
            True if this looks like a medical service
        """
        desc_lower = description.lower().strip()
        
        # Must contain medical keywords
        medical_keywords = [
            'consultation', 'doctor', 'physician', 'specialist', 'opd', 'checkup', 'visit',
            'room', 'bed', 'ward', 'icu', 'cabin', 'charges', 'fee', 'rent',
            'surgery', 'operation', 'procedure', 'treatment', 'therapy',
            'nursing', 'medical', 'ambulance', 'oxygen', 'injection',
            'blood', 'urine', 'ct', 'mri', 'x-ray', 'scan', 'test', 'pathology', 'lab'
        ]
        
        if not any(keyword in desc_lower for keyword in medical_keywords):
            return False
        
        # Reject if it looks like address, medicine name, or metadata
        reject_patterns = [
            r'^[a-z]+\s*\d+$',  # Like "plot 123"
            r'^\d+[a-z]*$',     # Like "123a"
            r'^[a-z]{2,6}$',    # Short codes like "akppd", "phco"
            r'bhubaneswar|khordha|state|city|pin|pincode',  # Location names
            r'gstin|gst|cgst|sgst|tax',  # Tax related
            r'syrup|tablet|capsule|mg|ml',  # Medicine forms
            r'date|time|bill|invoice|payment|round',  # Bill metadata
        ]
        
        for pattern in reject_patterns:
            if re.search(pattern, desc_lower):
                return False
        
        return True
    
    def _is_valid_pharmacy_item(self, description: str) -> bool:
        """Check if description represents a valid pharmacy/medicine item.
        
        Args:
            description: Item description
            
        Returns:
            True if this looks like a medicine or pharmacy item
        """
        desc_lower = description.lower().strip()
        
        # Pharmacy-specific keywords
        pharmacy_keywords = [
            # Medicine forms
            'tablet', 'capsule', 'syrup', 'injection', 'drops', 'cream', 'ointment',
            'powder', 'liquid', 'gel', 'suspension', 'lotion', 'spray',
            # Dosage indicators
            'mg', 'ml', 'gm', 'mcg', 'iu', 'unit', 'dose',
            # Common medicine prefixes/suffixes
            'tab', 'cap', 'inj', 'susp', 'sol', 'ext', 'er', 'sr', 'xr',
            # Medicine categories
            'antibiotic', 'analgesic', 'antacid', 'vitamin', 'supplement',
            'expectorant', 'cough', 'fever', 'pain', 'allergy'
        ]
        
        # Check for pharmacy keywords
        has_pharmacy_keyword = any(keyword in desc_lower for keyword in pharmacy_keywords)
        
        # Also accept items that look like medicine names (alphanumeric with spaces)
        medicine_name_pattern = r'^[a-z][a-z\s\d\-\.]+[a-z\d]$'
        looks_like_medicine = bool(re.match(medicine_name_pattern, desc_lower)) and len(desc_lower) > 3
        
        if not (has_pharmacy_keyword or looks_like_medicine):
            return False
        
        # Reject if it looks like address or metadata
        reject_patterns = [
            r'^[a-z]+\s*\d+$',  # Like "plot 123"
            r'^\d+[a-z]*$',     # Like "123a"
            r'^[a-z]{2,6}$',    # Short codes like "akppd", "phco"
            r'bhubaneswar|khordha|state|city|pin|pincode',  # Location names
            r'gstin|gst|cgst|sgst|tax',  # Tax related
            r'date|time|bill|invoice|payment|round',  # Bill metadata
            r'phone|mobile|email|address',  # Contact info
        ]
        
        for pattern in reject_patterns:
            if re.search(pattern, desc_lower):
                return False
        
        return True
    
    def _validate_medical_items(self, items: List[ExtractedLineItem]) -> List[ExtractedLineItem]:
        """Additional validation for medical line items.
        
        Args:
            items: List of extracted items
            
        Returns:
            Validated items only
        """
        validated = []
        
        for item in items:
            # Skip very short descriptions
            if len(item.description) < 5:
                continue
                
            # Skip unreasonable amounts
            if item.total_amount < Decimal('10') or item.total_amount > Decimal('50000'):
                continue
                
            # Skip if description is mostly numbers
            if sum(1 for c in item.description if c.isdigit()) > len(item.description) * 0.5:
                continue
                
            validated.append(item)
        
        return validated
    
    def _count_tables(self, text: str) -> int:
        """Count the number of tables detected in text.
        
        Args:
            text: Extracted text from document
            
        Returns:
            Number of tables detected
        """
        table_count = 0
        
        for pattern in self.table_patterns:
            matches = pattern.findall(text)
            table_count += len(matches)
        
        # Remove duplicates by looking for unique table headers
        unique_tables = set()
        for pattern in self.table_patterns:
            matches = pattern.findall(text)
            for match in matches:
                # Use first 50 characters as unique identifier
                identifier = match[:50] if isinstance(match, str) else str(match)[:50]
                unique_tables.add(identifier)
        
        return len(unique_tables)
    
    # Table extraction methods
    
    async def _extract_tables_camelot(self, pdf_data: bytes) -> List[ExtractedTable]:
        """Extract tables from PDF using Camelot.
        
        Args:
            pdf_data: PDF binary data
            
        Returns:
            List of extracted tables
            
        Raises:
            TableExtractionError: If Camelot extraction fails critically
        """
        if not self.enable_camelot:
            return []
        
        if not pdf_data:
            raise TableExtractionError("No PDF data provided for Camelot extraction")
        
        tables = []
        temp_path = None
        
        try:
            # Save PDF to temporary file for Camelot
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_data)
                temp_path = temp_file.name
            
            try:
                # Extract tables using Camelot
                camelot_tables = camelot.read_pdf(temp_path, pages='all', flavor='lattice')
                
                if not camelot_tables:
                    logger.info("Camelot found no tables in PDF")
                    return []
                
                for i, table in enumerate(camelot_tables):
                    try:
                        # Convert to our format
                        df = table.df.dropna(how='all').dropna(axis=1, how='all')
                        
                        if len(df) < 2:  # Need at least header + 1 row
                            logger.debug(f"Camelot table {i}: insufficient rows ({len(df)})")
                            continue
                        
                        # Extract headers and rows
                        headers = df.iloc[0].tolist()
                        rows = df.iloc[1:].values.tolist()
                        
                        # Clean up headers and rows
                        headers = [str(h).strip() for h in headers if str(h).strip()]
                        rows = [[str(cell).strip() for cell in row if str(cell).strip()] for row in rows]
                        rows = [row for row in rows if len(row) >= 2]  # Keep rows with at least 2 columns
                        
                        if not headers or not rows:
                            logger.debug(f"Camelot table {i}: no valid headers or rows after cleanup")
                            continue
                        
                        # Validate table accuracy
                        accuracy = getattr(table, 'accuracy', 0)
                        if accuracy < 30:  # Very low accuracy threshold
                            logger.debug(f"Camelot table {i}: low accuracy ({accuracy}%), skipping")
                            continue
                        
                        # Create ExtractedTable
                        extracted_table = ExtractedTable(
                            page_number=getattr(table, 'page', 1),
                            table_index=i,
                            headers=headers,
                            rows=rows,
                            confidence=accuracy / 100.0,
                            extraction_method='camelot',
                            bounding_box={
                                'x1': table._bbox[0],
                                'y1': table._bbox[1], 
                                'x2': table._bbox[2],
                                'y2': table._bbox[3]
                            } if hasattr(table, '_bbox') else None
                        )
                        
                        tables.append(extracted_table)
                        logger.debug(f"Camelot table {i}: extracted successfully with {len(rows)} rows")
                        
                    except Exception as e:
                        logger.warning(f"Failed to process Camelot table {i}: {e}")
                        continue
                        
            except ImportError as e:
                raise TableExtractionError(f"Camelot library not available: {e}") from e
            except Exception as e:
                # Log but don't raise - allow fallback to regex
                logger.warning(f"Camelot table extraction failed: {e}")
                return []
                
        except Exception as e:
            error_msg = f"Critical error in Camelot table extraction: {e}"
            logger.error(error_msg)
            raise TableExtractionError(error_msg) from e
            
        finally:
            # Clean up temporary file
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
        
        logger.info(f"Camelot extracted {len(tables)} tables")
        return tables
    
    def _extract_tables_regex(self, text: str) -> List[ExtractedTable]:
        """Extract tables from text using regex patterns.
        
        Args:
            text: Extracted text from document
            
        Returns:
            List of extracted tables
        """
        tables = []
        lines = text.split('\n')
        
        # Look for table patterns
        for i, line in enumerate(lines):
            for pattern in self.table_patterns:
                if pattern.search(line):
                    # Found potential table header
                    table = self._extract_table_from_lines(lines, i)
                    if table:
                        tables.append(table)
                        break
        
        logger.info(f"Regex extracted {len(tables)} tables")
        return tables
    
    def _extract_table_from_lines(self, lines: List[str], header_index: int) -> Optional[ExtractedTable]:
        """Extract a table starting from a header line.
        
        Args:
            lines: All text lines
            header_index: Index of the header line
            
        Returns:
            ExtractedTable or None if extraction fails
        """
        if header_index >= len(lines):
            return None
        
        header_line = lines[header_index].strip()
        if not header_line:
            return None
        
        # Parse headers (split by multiple spaces or tabs)
        headers = re.split(r'\s{2,}|\t+', header_line)
        headers = [h.strip() for h in headers if h.strip()]
        
        if len(headers) < 2:  # Need at least 2 columns
            return None
        
        # Look for data rows after header
        rows = []
        for i in range(header_index + 1, min(header_index + 20, len(lines))):  # Look ahead max 20 lines
            line = lines[i].strip()
            
            if not line:
                continue
            
            # Stop if we hit another table header or total line
            if any(pattern.search(line) for pattern in self.table_patterns):
                if i > header_index + 1:  # Don't stop immediately
                    break
            
            # Parse row data
            row_data = re.split(r'\s{2,}|\t+', line)
            row_data = [cell.strip() for cell in row_data if cell.strip()]
            
            if len(row_data) >= 2:  # At least 2 columns
                # Calculate confidence based on row quality
                confidence = self._calculate_row_confidence(row_data, headers)
                if confidence > 0.3:  # Minimum confidence threshold
                    rows.append(row_data)
        
        if len(rows) < 1:  # Need at least 1 data row
            return None
        
        # Calculate overall table confidence
        avg_confidence = sum(self._calculate_row_confidence(row, headers) for row in rows) / len(rows)
        
        return ExtractedTable(
            page_number=1,  # Assume page 1 for text extraction
            table_index=0,
            headers=headers,
            rows=rows,
            confidence=avg_confidence,
            extraction_method='regex'
        )
    
    def _calculate_row_confidence(self, row_data: List[str], headers: List[str]) -> float:
        """Calculate confidence score for a table row.
        
        Args:
            row_data: Row cell data
            headers: Table headers
            
        Returns:
            Confidence score between 0 and 1
        """
        if not row_data:
            return 0.0
        
        score = 0.0
        
        # Column count match
        if len(row_data) == len(headers):
            score += 0.3
        elif len(row_data) >= len(headers) - 1:  # Allow 1 missing column
            score += 0.2
        
        # Numeric values in expected positions (rate, amount columns)
        numeric_count = 0
        for cell in row_data:
            if re.search(r'\d+', cell):
                numeric_count += 1
        
        if numeric_count >= 2:
            score += 0.4
        elif numeric_count >= 1:
            score += 0.2
        
        # Non-empty cells
        non_empty = len([cell for cell in row_data if cell.strip()])
        if non_empty >= len(row_data) * 0.8:  # 80% non-empty
            score += 0.3
        
        return min(score, 1.0)
    
    def _tables_similar(self, table1: ExtractedTable, table2: ExtractedTable) -> bool:
        """Check if two tables are similar (for deduplication).
        
        Args:
            table1: First table
            table2: Second table
            
        Returns:
            True if tables are similar
        """
        # Check header similarity
        if len(table1.headers) != len(table2.headers):
            return False
        
        header_matches = 0
        for h1, h2 in zip(table1.headers, table2.headers):
            if h1.lower().strip() == h2.lower().strip():
                header_matches += 1
        
        header_similarity = header_matches / len(table1.headers)
        
        if header_similarity < 0.8:  # Headers must be 80% similar
            return False
        
        # Check row count similarity
        row_diff = abs(len(table1.rows) - len(table2.rows))
        if row_diff > max(len(table1.rows), len(table2.rows)) * 0.3:  # 30% difference threshold
            return False
        
        return True
    
    def _deduplicate_tables(self, tables: List[ExtractedTable]) -> List[ExtractedTable]:
        """Remove duplicate tables, keeping the one with highest confidence.
        
        Args:
            tables: List of extracted tables
            
        Returns:
            Deduplicated list of tables
        """
        if len(tables) <= 1:
            return tables
        
        unique_tables = []
        
        for table in tables:
            is_duplicate = False
            
            for existing in unique_tables:
                if self._tables_similar(table, existing):
                    # Found duplicate, keep the one with higher confidence
                    if table.confidence > existing.confidence:
                        unique_tables.remove(existing)
                        unique_tables.append(table)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
        
        return unique_tables
    
    def _map_table_headers(self, headers: List[str]) -> Dict[str, int]:
        """Map table headers to standard column types.
        
        Args:
            headers: List of header strings
            
        Returns:
            Mapping of column types to indices
        """
        mapping = {}
        
        for i, header in enumerate(headers):
            header_lower = header.lower().strip()
            
            # Description column
            if any(keyword in header_lower for keyword in ['description', 'item', 'service', 'particular', 'medicine', 'drug', 'test']):
                mapping['description'] = i
            
            # Quantity column
            elif any(keyword in header_lower for keyword in ['qty', 'quantity', 'no', 'count']):
                mapping['quantity'] = i
            
            # Rate/Price column
            elif any(keyword in header_lower for keyword in ['rate', 'price', 'unit', 'mrp', 'cost']):
                mapping['rate'] = i
            
            # Amount/Total column
            elif any(keyword in header_lower for keyword in ['amount', 'total', 'sum', 'value']):
                mapping['amount'] = i
        
        return mapping
    
    def _get_table_value(self, row: List[str], column_index: Optional[int]) -> Optional[str]:
        """Safely get value from table row.
        
        Args:
            row: Table row data
            column_index: Column index to extract
            
        Returns:
            Cell value or None
        """
        if column_index is None or column_index >= len(row):
            return None
        return row[column_index].strip() if row[column_index] else None
    
    def _parse_currency_value(self, value_str: str) -> Optional[Decimal]:
        """Parse currency value from string.
        
        Args:
            value_str: String containing currency value
            
        Returns:
            Decimal value or None if parsing fails
        """
        if not value_str:
            return None
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[₹Rs\.,\s]', '', value_str)
        
        # Extract numeric value
        match = re.search(r'\d+(?:\.\d{1,2})?', cleaned)
        if match:
            try:
                return Decimal(match.group())
            except:
                return None
        
        return None
    
    def _parse_numeric_value(self, value_str: str) -> Optional[int]:
        """Parse numeric value from string.
        
        Args:
            value_str: String containing numeric value
            
        Returns:
            Integer value or None if parsing fails
        """
        if not value_str:
            return None
        
        # Extract first number found
        match = re.search(r'\d+', value_str)
        if match:
            try:
                return int(match.group())
            except:
                return None
        
        return None
    
    def _extract_line_items_comprehensive(
        self, 
        text: str, 
        tables: List[ExtractedTable], 
        doc_type: DocumentType
    ) -> List[ExtractedLineItem]:
        """Extract line items from both text and tables, then deduplicate.
        
        Args:
            text: Raw extracted text
            tables: Extracted tables
            doc_type: Document type for context
            
        Returns:
            Deduplicated list of line items
        """
        all_items = []
        
        # Extract from text using regex
        text_items = self._extract_line_items(text, doc_type)
        all_items.extend(text_items)
        
        # Extract from tables
        for i, table in enumerate(tables):
            table_items = self._extract_line_items_from_table(table, doc_type, i)
            all_items.extend(table_items)
        
        # Deduplicate items
        unique_items = self._deduplicate_line_items(all_items)
        
        logger.info(f"Total line items: {len(all_items)}, unique: {len(unique_items)}")
        return unique_items
    
    def _line_items_similar(self, item1: ExtractedLineItem, item2: ExtractedLineItem) -> bool:
        """Check if two line items are similar (potential duplicates).
        
        Args:
            item1: First line item
            item2: Second line item
            
        Returns:
            True if items are similar, False otherwise
        """
        # Normalize descriptions for comparison
        desc1 = item1.description.lower().strip()
        desc2 = item2.description.lower().strip()
        
        # Check if descriptions are identical
        if desc1 == desc2:
            # If amounts are also the same, they're duplicates
            return item1.total_amount == item2.total_amount
        
        # Check for similar descriptions (fuzzy matching)
        # Use simple word-based similarity for now
        words1 = set(desc1.split())
        words2 = set(desc2.split())
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        similarity = intersection / union if union > 0 else 0
        
        # Consider items similar if >70% word overlap and amount within 10%
        if similarity > 0.7:
            if item1.total_amount and item2.total_amount:
                amount_diff = abs(item1.total_amount - item2.total_amount)
                avg_amount = (item1.total_amount + item2.total_amount) / 2
                amount_similarity = 1 - (amount_diff / avg_amount) if avg_amount > 0 else 0
                return amount_similarity > 0.9  # Within 10% difference
        
        return False
    
    def _deduplicate_line_items(self, items: List[ExtractedLineItem]) -> List[ExtractedLineItem]:
        """Remove duplicate line items, keeping the best version of each.
        
        Args:
            items: List of line items to deduplicate
            
        Returns:
            List of unique line items
        """
        if not items:
            return []
        
        unique_items = []
        
        for current_item in items:
            # Check if this item is similar to any existing unique item
            found_duplicate = False
            
            for i, existing_item in enumerate(unique_items):
                if self._line_items_similar(current_item, existing_item):
                    # Found a duplicate - keep the better one
                    found_duplicate = True
                    
                    # Determine which item to keep based on priority:
                    # 1. Higher confidence
                    # 2. Table source over regex
                    # 3. More complete information
                    
                    current_score = self._calculate_item_priority(current_item)
                    existing_score = self._calculate_item_priority(existing_item)
                    
                    if current_score > existing_score:
                        # Replace existing item with current item
                        unique_items[i] = current_item
                    
                    break
            
            if not found_duplicate:
                # No duplicate found, add as new unique item
                unique_items.append(current_item)
        
        return unique_items
    
    def _calculate_item_priority(self, item: ExtractedLineItem) -> float:
        """Calculate priority score for line item deduplication.
        
        Higher score = better item to keep
        
        Args:
            item: Line item to score
            
        Returns:
            Priority score
        """
        score = 0.0
        
        # Confidence score (0-1)
        score += item.confidence
        
        # Source method bonus
        if item.source_method == 'table':
            score += 0.5  # Tables are generally more reliable
        elif item.source_method == 'regex':
            score += 0.2
        
        # Completeness bonus
        if item.unit_price is not None:
            score += 0.1
        
        if item.quantity > 1:
            score += 0.05
        
        if item.item_type != LineItemType.OTHER:
            score += 0.1  # Classified items are better
        
        return score
    
    def _extract_line_items_from_table(
        self, 
        table: ExtractedTable, 
        doc_type: DocumentType, 
        table_index: int
    ) -> List[ExtractedLineItem]:
        """Extract line items from a table.
        
        Args:
            table: Extracted table to process
            doc_type: Document type for context
            table_index: Index of the table for tracking
            
        Returns:
            List of extracted line items
        """
        line_items = []
        
        if not table.rows:
            return line_items
        
        # Map headers to find relevant columns
        header_map = self._map_table_headers(table.headers)
        
        # Extract line items from each row
        for row in table.rows:
            if not row or len(row) < 2:  # Need at least description and amount
                continue
            
            try:
                # Get description (first column or description column)
                description_idx = header_map.get('description', 0)
                description = self._get_table_value(row, description_idx)
                
                if not description or not self._is_valid_line_item_text(description):
                    continue
                
                # Skip if this looks like a header or total row
                if any(keyword in description.lower() for keyword in 
                       ['total', 'subtotal', 'grand total', 'amount', 'description', 'item', 'service']):
                    continue
                
                # Get amount (last column or amount column)
                amount_idx = header_map.get('amount', len(row) - 1)
                amount_str = self._get_table_value(row, amount_idx)
                
                if not amount_str:
                    continue
                
                amount = self._parse_currency_value(amount_str)
                if not amount or amount <= 0:
                    continue
                
                # Get quantity if available
                quantity_idx = header_map.get('quantity')
                quantity = 1
                if quantity_idx is not None:
                    quantity_str = self._get_table_value(row, quantity_idx)
                    if quantity_str:
                        parsed_qty = self._parse_numeric_value(quantity_str)
                        if parsed_qty and parsed_qty > 0:
                            quantity = parsed_qty
                
                # Get unit price if available (or calculate from total/quantity)
                unit_price_idx = header_map.get('unit_price') or header_map.get('rate')
                unit_price = None
                if unit_price_idx is not None:
                    unit_price_str = self._get_table_value(row, unit_price_idx)
                    if unit_price_str:
                        unit_price = self._parse_currency_value(unit_price_str)
                
                # Calculate unit price if not provided
                if unit_price is None and quantity > 0:
                    unit_price = amount / quantity
                
                # Classify the line item
                item_type = self._classify_line_item(description, doc_type)
                
                # Check if it's a valid medical service for medical documents
                if doc_type in [DocumentType.HOSPITAL_BILL, DocumentType.DIAGNOSTIC_REPORT]:
                    if not self._is_valid_medical_service(description):
                        continue
                
                # Create line item
                line_item = ExtractedLineItem(
                    description=description.strip(),
                    quantity=quantity,
                    unit_price=unit_price,
                    total_amount=amount,
                    item_type=item_type,
                    confidence=table.confidence,
                    source_table=table_index,
                    source_method='table'
                )
                
                line_items.append(line_item)
                
            except Exception as e:
                logger.debug(f"Failed to extract line item from table row {row}: {e}")
                continue
        
        logger.debug(f"Extracted {len(line_items)} line items from table {table_index}")
        return line_items
    
    async def process_document(
        self,
        file_path: str,
        doc_id: str,
        language: Language = Language.ENGLISH
    ) -> ExtractedDocument:
        """Process a document and extract structured information.
        
        Args:
            file_path: Path to document file (local, HTTP, or S3)
            doc_id: Unique document identifier
            language: Primary language for OCR
            
        Returns:
            ExtractedDocument with all extracted information
            
        Raises:
            FileValidationError: If file validation fails
            OCRError: If OCR processing fails
            TableExtractionError: If table extraction fails
            DocumentProcessingError: For other processing errors
        """
        start_time = asyncio.get_event_loop().time()
        errors_encountered = []
        file_data = None
        file_format = None
        images = []
        raw_text = ""
        ocr_confidence = 0.0
        
        try:
            # Step 1: Download and validate file
            logger.info(f"Processing document {doc_id}: {file_path}")
            
            try:
                file_data = await self._download_file(file_path)
            except Exception as e:
                error_msg = f"Failed to download file: {str(e)}"
                logger.error(error_msg)
                errors_encountered.append(error_msg)
                raise FileValidationError(error_msg) from e
            
            try:
                file_format = await self._validate_file(file_data, file_path.split('/')[-1])
            except (FileValidationError, ValueError) as e:
                error_msg = f"File validation failed: {str(e)}"
                logger.error(error_msg)
                errors_encountered.append(error_msg)
                raise FileValidationError(error_msg) from e
            
            # Step 2: Convert to images
            try:
                images = await self._convert_to_images(file_data, file_format)
                if not images:
                    error_msg = "No images could be extracted from the document"
                    logger.error(error_msg)
                    errors_encountered.append(error_msg)
                    raise DocumentProcessingError(error_msg)
            except Exception as e:
                error_msg = f"Failed to convert document to images: {str(e)}"
                logger.error(error_msg)
                errors_encountered.append(error_msg)
                raise DocumentProcessingError(error_msg) from e
            
            # Step 3: Extract text using OCR
            try:
                raw_text, ocr_confidence = await self._extract_text_ocr(images, language)
                
                # Check if OCR produced meaningful text
                if len(raw_text.strip()) < 10:
                    error_msg = f"OCR produced insufficient text (only {len(raw_text.strip())} characters)"
                    logger.warning(error_msg)
                    errors_encountered.append(error_msg)
                    # Don't raise exception here - continue with empty text
                    
                if ocr_confidence < self.confidence_threshold:
                    warning_msg = f"OCR confidence ({ocr_confidence:.1f}%) below threshold ({self.confidence_threshold}%)"
                    logger.warning(warning_msg)
                    errors_encountered.append(warning_msg)
                    # Continue processing with low confidence
                    
            except Exception as e:
                error_msg = f"OCR processing failed: {str(e)}"
                logger.error(error_msg)
                errors_encountered.append(error_msg)
                raise OCRError(error_msg) from e
            
            # Step 4: Detect document type
            try:
                doc_type = self._detect_document_type(raw_text)
            except Exception as e:
                error_msg = f"Document type detection failed: {str(e)}"
                logger.warning(error_msg)
                errors_encountered.append(error_msg)
                doc_type = DocumentType.UNKNOWN  # Fallback to unknown type
            
            # Step 5: Extract tables (with graceful degradation)
            tables = []
            try:
                if file_data and file_format:
                    tables = await self._extract_tables_camelot(file_data)
            except TableExtractionError as e:
                error_msg = f"Table extraction failed: {str(e)}"
                logger.warning(error_msg)
                errors_encountered.append(error_msg)
                # Continue without tables
            except Exception as e:
                error_msg = f"Unexpected error during table extraction: {str(e)}"
                logger.warning(error_msg)
                errors_encountered.append(error_msg)
                # Continue without tables
            
            # Step 6: Extract line items from text and tables
            line_items = []
            try:
                line_items = self._extract_line_items_comprehensive(raw_text, tables, doc_type)
            except Exception as e:
                error_msg = f"Line item extraction failed: {str(e)}"
                logger.warning(error_msg)
                errors_encountered.append(error_msg)
                # Try fallback extraction from text only
                try:
                    line_items = self._extract_line_items(raw_text, doc_type)
                except Exception as fallback_e:
                    fallback_msg = f"Fallback line item extraction also failed: {str(fallback_e)}"
                    logger.warning(fallback_msg)
                    errors_encountered.append(fallback_msg)
                    # Continue with empty line items
            
            # Step 7: Count tables for statistics
            tables_found = 0
            try:
                tables_found = self._count_tables(raw_text)
            except Exception as e:
                error_msg = f"Table counting failed: {str(e)}"
                logger.warning(error_msg)
                errors_encountered.append(error_msg)
                # Use extracted tables count as fallback
                tables_found = len(tables)
            
            # Calculate processing time
            end_time = asyncio.get_event_loop().time()
            processing_time_ms = int((end_time - start_time) * 1000)
            
            # Create processing statistics
            stats = ProcessingStats(
                pages_processed=len(images),
                ocr_confidence=ocr_confidence,
                text_extracted_chars=len(raw_text),
                tables_found=tables_found,
                tables_extracted=len(tables),
                line_items_found=len(line_items),
                processing_time_ms=processing_time_ms,
                errors_encountered=errors_encountered
            )
            
            # Create metadata
            metadata = {
                'doc_id': doc_id,
                'file_format': file_format or 'unknown',
                'file_size_bytes': len(file_data) if file_data else 0,
                'ocr_language': language.value,
                'table_extraction_method': 'camelot' if self.enable_camelot else 'regex',
                'confidence_threshold': self.confidence_threshold,
                'processing_warnings': len(errors_encountered)
            }
            
            # Create final document
            document = ExtractedDocument(
                raw_text=raw_text,
                tables=tables,
                line_items=line_items,
                document_type=doc_type,
                language=language,
                stats=stats,
                metadata=metadata
            )
            
            if errors_encountered:
                logger.warning(f"Document {doc_id} processed with {len(errors_encountered)} warnings in {processing_time_ms}ms")
            else:
                logger.info(f"Document {doc_id} processed successfully in {processing_time_ms}ms")
            
            return document
            
        except (FileValidationError, OCRError, TableExtractionError) as e:
            # Re-raise specific exceptions
            logger.error(f"Failed to process document {doc_id}: {str(e)}")
            raise
            
        except Exception as e:
            # Wrap unexpected exceptions
            error_msg = f"Unexpected error processing document {doc_id}: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def _detect_image_type(self, image: Image.Image) -> str:
        """
        Detect if image is camera-captured or scanned based on characteristics.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            'camera' or 'scanned'
        """
        width, height = image.size
        
        # Camera images typically have:
        # - Higher resolution (usually > 2000px)
        # - Aspect ratios close to camera ratios (4:3, 16:9, etc.)
        # - More noise and compression artifacts
        
        # Scanned documents typically have:
        # - Lower resolution (usually < 1500px)
        # - Standard document ratios (A4: ~1.414, Letter: ~1.294)
        # - Clean, uniform backgrounds
        
        aspect_ratio = width / height
        total_pixels = width * height
        
        # High resolution suggests camera
        if total_pixels > 4_000_000:  # > 4MP
            return 'camera'
        
        # Very low resolution suggests scan
        if total_pixels < 1_000_000:  # < 1MP
            return 'scanned'
        
        # Check aspect ratios
        camera_ratios = [4/3, 16/9, 3/2, 1.85, 2.35]  # Common camera ratios
        document_ratios = [1.414, 1.294]  # A4, Letter
        
        # Find closest ratio
        min_camera_diff = min(abs(aspect_ratio - ratio) for ratio in camera_ratios)
        min_doc_diff = min(abs(aspect_ratio - ratio) for ratio in document_ratios)
        
        # If much closer to camera ratio, likely camera
        if min_camera_diff < 0.1 and min_camera_diff < min_doc_diff - 0.1:
            return 'camera'
        
        # If much closer to document ratio, likely scanned
        if min_doc_diff < 0.1 and min_doc_diff < min_camera_diff - 0.1:
            return 'scanned'
        
        # Default: use resolution as tie-breaker
        return 'camera' if total_pixels > 2_000_000 else 'scanned'

    def _get_dynamic_confidence_threshold(self, image: Image.Image) -> float:
        """
        Get appropriate confidence threshold based on image type.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Confidence threshold (0-100)
        """
        image_type = self._detect_image_type(image)
        
        if image_type == 'camera':
            # Camera images need lower threshold
            threshold = self.camera_confidence_threshold
            logger.debug(f"Camera image detected, using threshold: {threshold}%")
        else:
            # Scanned documents can use higher threshold
            threshold = self.scanned_confidence_threshold
            logger.debug(f"Scanned document detected, using threshold: {threshold}%")
        
        return threshold

    def _preprocess_camera_image(self, image: Image.Image) -> Image.Image:
        """
        Special preprocessing for camera-captured images.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image optimized for camera images
        """
        # Convert PIL to OpenCV format
        opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Try multiple preprocessing strategies and return the best one
        strategies = []
        
        # Camera-specific preprocessing pipeline
        try:
            # 1. Resize if too large (improves processing speed)
            height, width = opencv_img.shape[:2]
            if width > 2000 or height > 2000:
                scale = min(2000 / width, 2000 / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                opencv_img = cv2.resize(opencv_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.debug(f"Resized camera image from {width}x{height} to {new_width}x{new_height}")
            
            # Strategy 1: Standard camera preprocessing
            gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            strategies.append(("standard", binary))
            
            # Strategy 2: Aggressive contrast enhancement
            gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
            # Apply gamma correction to brighten dark images
            gamma = 1.5
            lookUpTable = np.empty((1, 256), np.uint8)
            for i in range(256):
                lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            gray = cv2.LUT(gray, lookUpTable)
            
            # Strong CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
            
            # Aggressive denoising
            gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
            
            # Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            strategies.append(("aggressive_contrast", binary))
            
            # Strategy 3: Edge-preserving smoothing
            gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
            
            # Edge-preserving filter
            filtered = cv2.edgePreservingFilter(opencv_img, flags=2, sigma_s=30, sigma_r=0.4)
            gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
            
            # Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
            gray = clahe.apply(gray)
            
            # Median blur to reduce noise
            gray = cv2.medianBlur(gray, 3)
            
            # Adaptive thresholding with different parameters
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 4
            )
            strategies.append(("edge_preserving", binary))
            
            # Strategy 4: Morphological preprocessing
            gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
            
            # Apply opening to remove noise
            kernel = np.ones((2, 2), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Apply closing to fill gaps
            kernel = np.ones((3, 3), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Gaussian blur
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3
            )
            strategies.append(("morphological", binary))
            
            # If we have multiple strategies, return the first one for now
            # In a more sophisticated implementation, we could test each with a quick OCR
            if strategies:
                chosen_strategy, result = strategies[0]  # Use standard strategy by default
                logger.debug(f"Using camera preprocessing strategy: {chosen_strategy}")
                return Image.fromarray(result)
            
        except Exception as e:
            logger.error(f"All camera preprocessing strategies failed: {e}")
            # Return original image if all preprocessing fails
            return image
            
        # Return original image if no strategies worked
        return image

    def _preprocess_camera_image_aggressive(self, image: Image.Image) -> Image.Image:
        """Aggressive contrast enhancement for very dark/poor quality images."""
        opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        
        # Gamma correction to brighten dark images
        gamma = 1.8
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        gray = cv2.LUT(gray, lookUpTable)
        
        # Strong CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        
        # Aggressive denoising
        gray = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return Image.fromarray(binary)

    def _preprocess_camera_image_edge_preserving(self, image: Image.Image) -> Image.Image:
        """Edge-preserving smoothing for noisy images."""
        opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Edge-preserving filter
        filtered = cv2.edgePreservingFilter(opencv_img, flags=2, sigma_s=50, sigma_r=0.4)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        gray = clahe.apply(gray)
        
        # Median blur to reduce noise
        gray = cv2.medianBlur(gray, 5)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 4
        )
        
        return Image.fromarray(binary)

    def _preprocess_camera_image_morphological(self, image: Image.Image) -> Image.Image:
        """Morphological operations for text cleanup."""
        opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        
        # Apply opening to remove noise
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Apply closing to fill gaps
        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Gaussian blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3
        )
        
        return Image.fromarray(binary)


# Convenience function for external use
async def process_document(
    file_path: str,
    doc_id: str,
    language: Language = Language.ENGLISH,
    **kwargs
) -> ExtractedDocument:
    """Convenience function to process a document.
    
    Args:
        file_path: Path to document file
        doc_id: Unique document identifier
        language: Primary language for OCR
        **kwargs: Additional arguments for DocumentProcessor
        
    Returns:
        ExtractedDocument with extracted information
    """
    processor = DocumentProcessor(**kwargs)
    return await processor.process_document(file_path, doc_id, language) 