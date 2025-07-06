"""
Document Processor Tool - Async wrapper for DocumentProcessor.

This tool wraps the existing DocumentProcessor component to make it compatible
with the OpenAI Agent SDK framework for medical bill analysis.
"""

import asyncio
import structlog
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from medbillguardagent.document_processor import DocumentProcessor
from medbillguardagent.schemas import Language, DocumentType

logger = structlog.get_logger(__name__)


class DocumentProcessorInput(BaseModel):
    """Input schema for document processor tool."""
    file_content: bytes = Field(..., description="Binary content of the document file")
    doc_id: str = Field(..., description="Unique document identifier")
    language: Language = Field(default=Language.ENGLISH, description="Document language for OCR")
    file_format: Optional[str] = Field(default=None, description="File format hint (pdf, jpg, png)")


class DocumentProcessorTool:
    """
    Async tool wrapper for DocumentProcessor.
    
    Provides document processing capabilities including:
    - OCR text extraction
    - Table detection and extraction  
    - Line item extraction
    - Document type classification
    """
    
    def __init__(
        self,
        supported_formats: Optional[list] = None,
        max_file_size: int = 15 * 1024 * 1024,
        confidence_threshold: float = 60,
        enable_camelot: bool = True
    ):
        """Initialize the document processor tool."""
        self.processor = DocumentProcessor(
            supported_formats=supported_formats,
            max_file_size=max_file_size,
            confidence_threshold=confidence_threshold,
            enable_camelot=enable_camelot
        )
        logger.info("Initialized DocumentProcessorTool")
    
    async def __call__(
        self,
        file_content: bytes,
        doc_id: str,
        language: str = "english",
        file_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a medical document and extract structured information.
        
        Args:
            file_content: Binary content of the document file
            doc_id: Unique document identifier
            language: Document language for OCR processing
            file_format: Optional file format hint
            
        Returns:
            Dict containing extracted document information
        """
        try:
            logger.info(
                "Processing document",
                doc_id=doc_id,
                language=language,
                file_size=len(file_content)
            )
            
            # Convert language string to enum
            lang_enum = Language.ENGLISH
            if language.lower() in ["hindi", "hin"]:
                lang_enum = Language.HINDI
            elif language.lower() in ["bengali", "ben"]:
                lang_enum = Language.BENGALI
            elif language.lower() in ["tamil", "tam"]:
                lang_enum = Language.TAMIL
            
            # Handle text files directly without going through document processor
            if file_format == 'txt' or (file_format is None and (file_content.startswith(b'APOLLO') or b'CHARGES:' in file_content)):
                # Handle as plain text medical bill
                try:
                    text_content = file_content.decode('utf-8')
                except UnicodeDecodeError:
                    text_content = file_content.decode('latin-1', errors='ignore')
                
                # Create a simplified extracted document for text files
                from medbillguardagent.document_processor import ExtractedDocument, ProcessingStats, ExtractedLineItem
                from medbillguardagent.schemas import DocumentType, LineItemType
                from decimal import Decimal
                import re
                
                # Simple line item extraction from text
                line_items = []
                lines = text_content.split('\n')
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    # Look for patterns like "1. Item Name: Rs 123.00"
                    pattern = r'(\d+\.?\s+)?(.+?):?\s+Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)'
                    match = re.search(pattern, line, re.IGNORECASE)
                    
                    if match:
                        description = match.group(2).strip()
                        amount_str = match.group(3).replace(',', '')
                        try:
                            amount = Decimal(amount_str)
                            line_items.append(ExtractedLineItem(
                                description=description,
                                quantity=1,
                                total_amount=amount,
                                item_type=LineItemType.CONSULTATION if 'consultation' in description.lower() else LineItemType.DIAGNOSTIC,
                                confidence=0.9,
                                source_method="text_regex"
                            ))
                        except (ValueError, TypeError):
                            continue
                
                extracted_doc = ExtractedDocument(
                    raw_text=text_content,
                    line_items=line_items,
                    document_type=DocumentType.HOSPITAL_BILL,
                    language=lang_enum,
                    stats=ProcessingStats(
                        pages_processed=1,
                        ocr_confidence=100.0,  # No OCR needed for text
                        text_extracted_chars=len(text_content),
                        tables_found=0,
                        tables_extracted=0,
                        line_items_found=len(line_items),
                        processing_time_ms=10,
                        errors_encountered=[]
                    ),
                    metadata={}
                )
                
            else:
                # Save content to temporary file for processing
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format or 'pdf'}") as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Process the document
                    extracted_doc = await self.processor.process_document(
                        tmp_file_path, doc_id, lang_enum
                    )
                finally:
                    # Cleanup temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
            
            # Convert to serializable format
            result = {
                "success": True,
                "doc_id": doc_id,
                "raw_text": extracted_doc.raw_text,
                "document_type": extracted_doc.document_type.value,
                "language": extracted_doc.language.value,
                "line_items": [
                    {
                        "description": item.description,
                        "quantity": item.quantity,
                        "unit_price": float(item.unit_price) if item.unit_price else None,
                        "total_amount": float(item.total_amount),
                        "item_type": item.item_type.value,
                        "confidence": item.confidence,
                        "source_method": item.source_method
                    }
                    for item in extracted_doc.line_items
                ],
                "tables": [
                    {
                        "page_number": table.page_number,
                        "table_index": table.table_index,
                        "headers": table.headers,
                        "rows": table.rows,
                        "confidence": table.confidence,
                        "extraction_method": table.extraction_method
                    }
                    for table in extracted_doc.tables
                ],
                "processing_stats": {
                    "pages_processed": extracted_doc.stats.pages_processed,
                    "ocr_confidence": extracted_doc.stats.ocr_confidence,
                    "text_extracted_chars": extracted_doc.stats.text_extracted_chars,
                    "tables_found": extracted_doc.stats.tables_found,
                    "tables_extracted": extracted_doc.stats.tables_extracted,
                    "line_items_found": extracted_doc.stats.line_items_found,
                    "processing_time_ms": extracted_doc.stats.processing_time_ms,
                    "errors_encountered": extracted_doc.stats.errors_encountered
                },
                "metadata": extracted_doc.metadata
            }
            
            logger.info(
                "Document processing completed",
                doc_id=doc_id,
                line_items_found=len(extracted_doc.line_items),
                tables_found=len(extracted_doc.tables),
                processing_time_ms=extracted_doc.stats.processing_time_ms
            )
            
            return result
                    
        except Exception as e:
            error_msg = f"Document processing failed: {str(e)}"
            logger.error(error_msg, doc_id=doc_id, exc_info=True)
            
            return {
                "success": False,
                "doc_id": doc_id,
                "error": error_msg,
                "line_items": [],
                "tables": [],
                "processing_stats": {
                    "pages_processed": 0,
                    "ocr_confidence": 0.0,
                    "text_extracted_chars": 0,
                    "tables_found": 0,
                    "tables_extracted": 0,
                    "line_items_found": 0,
                    "processing_time_ms": 0,
                    "errors_encountered": [error_msg]
                },
                "metadata": {}
            }


# Tool schema for OpenAI Agent SDK
DocumentProcessorTool._tool_schema = {
    "type": "function",
    "function": {
        "name": "process_document",
        "description": "Process a medical document (PDF, image, text) and extract structured data including line items, tables, and metadata",
        "parameters": {
            "type": "object",
            "properties": {
                "file_content": {
                    "type": "string",
                    "description": "Base64 encoded binary content of the document file"
                },
                "doc_id": {
                    "type": "string", 
                    "description": "Unique document identifier for tracking"
                },
                "language": {
                    "type": "string",
                    "enum": ["english", "hindi", "bengali", "tamil"],
                    "description": "Document language for OCR processing",
                    "default": "english"
                },
                "file_format": {
                    "type": "string",
                    "enum": ["pdf", "jpg", "jpeg", "png", "txt"],
                    "description": "File format hint to optimize processing"
                }
            },
            "required": ["file_content", "doc_id"]
        }
    }
} 