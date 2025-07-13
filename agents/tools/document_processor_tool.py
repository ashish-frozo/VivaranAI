"""
Document processor tool for agent system.
Provides OCR, table extraction, and document analysis capabilities.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from agents.base_agent import BaseTool
from shared.processors.document_processor import DocumentProcessor
from shared.schemas.schemas import Language, DocumentType

logger = logging.getLogger(__name__)

class DocumentProcessorTool(BaseTool):
    """Tool for processing documents with OCR, table extraction, and text analysis."""
    
    def __init__(self, confidence_threshold: float = 60, enable_camelot: bool = True):
        """
        Initialize the document processor tool.
        
        Args:
            confidence_threshold: Minimum OCR confidence threshold (0-100)
            enable_camelot: Whether to enable Camelot for table extraction
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.enable_camelot = enable_camelot
        self.processor = DocumentProcessor(
            confidence_threshold=confidence_threshold,
            enable_camelot=enable_camelot
        )
        logger.info("Initialized DocumentProcessorTool")
    
    async def process_document(
        self,
        file_path: str,
        doc_id: str,
        language: Language = Language.ENGLISH
    ) -> Dict[str, Any]:
        """
        Process a document and extract text, tables, and line items.
        
        Args:
            file_path: Path to the document file
            doc_id: Unique document identifier
            language: Language for OCR processing
            
        Returns:
            Dictionary containing extraction results
        """
        try:
            # Process document
            extracted_doc = await self.processor.process_document(
                file_path=file_path,
                doc_id=doc_id,
                language=language
            )
            
            # Convert to dictionary for tool response
            result = {
                "doc_id": doc_id,
                "raw_text": extracted_doc.raw_text,
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
                "document_type": extracted_doc.document_type.value,
                "language": extracted_doc.language.value,
                "stats": {
                    "pages_processed": extracted_doc.stats.pages_processed,
                    "ocr_confidence": extracted_doc.stats.ocr_confidence,
                    "text_extracted_chars": extracted_doc.stats.text_extracted_chars,
                    "tables_found": extracted_doc.stats.tables_found,
                    "tables_extracted": extracted_doc.stats.tables_extracted,
                    "line_items_found": extracted_doc.stats.line_items_found,
                    "processing_time_ms": extracted_doc.stats.processing_time_ms,
                    "errors_encountered": extracted_doc.stats.errors_encountered
                }
            }
            
            logger.info(
                f"Document processing completed",
                extra={
                    "doc_id": doc_id,
                    "line_items_found": len(result["line_items"]),
                    "tables_found": len(result["tables"]),
                    "processing_time_ms": result["stats"]["processing_time_ms"]
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}", extra={"doc_id": doc_id})
            raise

# Import classes for backward compatibility
from shared.processors.document_processor import ExtractedDocument, ProcessingStats, ExtractedLineItem
from shared.schemas.schemas import DocumentType, LineItemType 