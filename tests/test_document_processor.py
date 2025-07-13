"""
Unit tests for DocumentProcessor class.

This module contains comprehensive tests for the DocumentProcessor including:
- File format validation and detection
- OCR text extraction from images and PDFs
- Table extraction using Camelot and regex
- Line item extraction and validation
- Document type classification
- Error handling and edge cases
"""

import pytest
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from shared.processors.document_processor import (
    DocumentProcessor,
    DocumentProcessingError,
    FileValidationError,
    OCRError,
    TableExtractionError,
    ExtractedLineItem,
    ExtractedTable,
    ExtractedDocument,
    ProcessingStats,
    process_document
)
from shared.schemas.schemas import DocumentType, Language, LineItemType


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance for testing."""
        return DocumentProcessor(
            supported_formats=['pdf', 'jpg', 'jpeg', 'png'],
            max_file_size=15 * 1024 * 1024,
            confidence_threshold=60,
            enable_camelot=True
        )

    @pytest.fixture
    def sample_pdf_data(self):
        """Sample PDF file data with proper magic bytes."""
        return b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n174\n%%EOF'

    @pytest.fixture
    def sample_jpg_data(self):
        """Sample JPEG file data with proper magic bytes."""
        return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9'

    @pytest.fixture
    def sample_png_data(self):
        """Sample PNG file data with proper magic bytes."""
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'

    @pytest.fixture
    def sample_corrupted_data(self):
        """Sample corrupted file data."""
        return b'corrupted data that looks like nothing'

    @pytest.fixture
    def sample_oversized_data(self):
        """Sample oversized file data (>15MB)."""
        return b'%PDF-1.4\n' + b'x' * (16 * 1024 * 1024)  # 16MB

    @pytest.fixture
    def sample_image(self):
        """Create a sample PIL image for testing."""
        return Image.new('RGB', (800, 600), color='white')

    @pytest.fixture
    def sample_high_res_image(self):
        """Create a high resolution PIL image for testing."""
        return Image.new('RGB', (5000, 4000), color='white')

    @pytest.fixture
    def sample_low_res_image(self):
        """Create a low resolution PIL image for testing."""
        return Image.new('RGB', (400, 300), color='white')

    @pytest.fixture
    def sample_extreme_aspect_image(self):
        """Create an image with extreme aspect ratio for testing."""
        return Image.new('RGB', (5000, 200), color='white')

    @pytest.fixture
    def sample_extracted_text(self):
        """Sample extracted text from a medical bill."""
        return """
        ABC Hospital
        Patient: John Doe
        
        Description                  Qty    Rate     Amount
        Consultation Fee              1      500      500
        Blood Test                    2      200      400
        X-Ray Chest                   1      800      800
        Room Charges                  2      1000     2000
        
        Total Amount: Rs. 3700
        """

    @pytest.fixture
    def sample_table_text(self):
        """Sample text with table structure."""
        return """
        Medical Bill Summary
        
        Description          Qty   Rate    Amount
        Consultation Fee      1    500     500
        Blood Test CBC        1    300     300
        X-Ray Chest          1    800     800
        Medicine             2    150     300
        
        Total: Rs. 1900
        """

    @pytest.fixture
    def sample_extracted_table(self):
        """Sample extracted table for testing."""
        return ExtractedTable(
            page_number=1,
            table_index=0,
            headers=['Description', 'Qty', 'Rate', 'Amount'],
            rows=[
                ['Consultation Fee', '1', '500', '500'],
                ['Blood Test', '2', '200', '400'],
                ['X-Ray', '1', '800', '800']
            ],
            confidence=0.85,
            extraction_method='camelot'
        )

    # File Format Detection Tests

    def test_detect_format_pdf(self, processor, sample_pdf_data):
        """Test PDF format detection."""
        result = processor._detect_format(sample_pdf_data)
        assert result == 'pdf'

    def test_detect_format_jpg(self, processor, sample_jpg_data):
        """Test JPEG format detection."""
        result = processor._detect_format(sample_jpg_data)
        assert result == 'jpg'

    def test_detect_format_png(self, processor, sample_png_data):
        """Test PNG format detection."""
        result = processor._detect_format(sample_png_data)
        assert result == 'png'

    def test_detect_format_unknown(self, processor, sample_corrupted_data):
        """Test unknown format detection."""
        result = processor._detect_format(sample_corrupted_data)
        assert result == 'unknown'

    def test_detect_format_empty_data(self, processor):
        """Test format detection with empty data."""
        result = processor._detect_format(b'')
        assert result == 'unknown'

    def test_detect_format_short_data(self, processor):
        """Test format detection with very short data."""
        result = processor._detect_format(b'%PDF')  # Less than 8 bytes
        assert result == 'unknown'

    def test_detect_format_jpeg_variants(self, processor):
        """Test detection of various JPEG format variants."""
        # JFIF variant
        jfif_data = b'\xff\xd8\xff\xe0\x00\x10JFIF'
        assert processor._detect_format(jfif_data) == 'jpg'
        
        # Exif variant
        exif_data = b'\xff\xd8\xff\xe1\x00\x16Exif'
        assert processor._detect_format(exif_data) == 'jpg'

    # File Format Validation Tests

    def test_validate_file_format_success(self, processor, sample_pdf_data):
        """Test successful file format validation."""
        result = processor._validate_file_format(sample_pdf_data)
        assert result == 'pdf'

    def test_validate_file_format_unknown(self, processor, sample_corrupted_data):
        """Test file format validation with unknown format."""
        with pytest.raises(ValueError, match="Unable to detect file format"):
            processor._validate_file_format(sample_corrupted_data)

    def test_validate_file_format_unsupported(self, processor):
        """Test file format validation with unsupported format."""
        # Create processor that only supports PDF
        limited_processor = DocumentProcessor(supported_formats=['pdf'])
        
        with pytest.raises(ValueError, match="Unsupported file format: jpg"):
            limited_processor._validate_file_format(b'\xff\xd8\xff\xe0JFIF')

    def test_validate_file_format_with_hint(self, processor, sample_jpg_data):
        """Test file format validation with format hint."""
        result = processor._validate_file_format(sample_jpg_data, 'jpg')
        assert result == 'jpg'

    def test_validate_file_format_hint_mismatch(self, processor, sample_jpg_data):
        """Test file format validation with mismatched hint."""
        with patch('medbillguardagent.document_processor.logger') as mock_logger:
            result = processor._validate_file_format(sample_jpg_data, 'png')
            assert result == 'jpg'
            mock_logger.warning.assert_called_once()

    # File Size Validation Tests

    def test_validate_file_size_success(self, processor, sample_pdf_data):
        """Test successful file size validation."""
        # Should not raise any exception
        processor._validate_file_size(sample_pdf_data, 'pdf', 'test.pdf')

    def test_validate_file_size_too_large(self, processor, sample_oversized_data):
        """Test file size validation with oversized file."""
        with pytest.raises(ValueError, match="File .* size .* exceeds limit"):
            processor._validate_file_size(sample_oversized_data, 'pdf', 'large.pdf')

    def test_validate_file_size_very_small(self, processor):
        """Test file size validation with very small file."""
        small_data = b'%PDF-1.4\n'  # Less than 1KB
        
        with patch('medbillguardagent.document_processor.logger') as mock_logger:
            processor._validate_file_size(small_data, 'pdf', 'small.pdf')
            mock_logger.warning.assert_called_once()

    def test_validate_file_size_format_specific_limits(self, processor):
        """Test format-specific file size limits."""
        # Test with format-specific max size
        processor.MAX_FILE_SIZES['test_format'] = 1024  # 1KB limit
        
        large_data = b'x' * 2048  # 2KB
        with pytest.raises(ValueError, match="exceeds limit"):
            processor._validate_file_size(large_data, 'test_format', 'test.file')

    # Image Quality Validation Tests

    def test_validate_image_quality_success(self, processor, sample_image):
        """Test successful image quality validation."""
        # Should not raise any exception
        processor._validate_image_quality(sample_image, 'test.jpg')

    def test_validate_image_quality_low_resolution(self, processor, sample_low_res_image):
        """Test image quality validation with low resolution."""
        with patch('medbillguardagent.document_processor.logger') as mock_logger:
            processor._validate_image_quality(sample_low_res_image, 'lowres.jpg')
            mock_logger.warning.assert_called()

    def test_validate_image_quality_high_resolution(self, processor, sample_high_res_image):
        """Test image quality validation with high resolution."""
        with patch('medbillguardagent.document_processor.logger') as mock_logger:
            processor._validate_image_quality(sample_high_res_image, 'highres.jpg')
            mock_logger.warning.assert_called()

    def test_validate_image_quality_extreme_aspect_ratio(self, processor, sample_extreme_aspect_image):
        """Test image quality validation with extreme aspect ratio."""
        with patch('medbillguardagent.document_processor.logger') as mock_logger:
            processor._validate_image_quality(sample_extreme_aspect_image, 'extreme.jpg')
            mock_logger.warning.assert_called()

    # PDF Structure Validation Tests

    def test_validate_pdf_structure_success(self, processor):
        """Test successful PDF structure validation."""
        with patch('fitz.open') as mock_fitz:
            # Mock successful PDF opening
            mock_doc = Mock()
            mock_doc.__len__ = Mock(return_value=2)  # 2 pages
            mock_page = Mock()
            mock_page.rect.width = 612
            mock_page.rect.height = 792
            mock_doc.__getitem__ = Mock(return_value=mock_page)
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            # Should not raise any exception
            processor._validate_pdf_structure(b'%PDF-1.4', 'test.pdf')
            
            mock_fitz.assert_called_once()
            mock_doc.close.assert_called_once()

    def test_validate_pdf_structure_no_pages(self, processor):
        """Test PDF validation with no pages."""
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_doc.__len__ = Mock(return_value=0)  # No pages
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            with pytest.raises(ValueError, match="PDF has no pages"):
                processor._validate_pdf_structure(b'%PDF-1.4', 'test.pdf')

    def test_validate_pdf_structure_many_pages(self, processor):
        """Test PDF validation with too many pages."""
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_doc.__len__ = Mock(return_value=101)  # Too many pages
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            with pytest.raises(ValueError, match="PDF has too many pages"):
                processor._validate_pdf_structure(b'%PDF-1.4', 'test.pdf')

    def test_validate_pdf_structure_invalid_dimensions(self, processor):
        """Test PDF validation with invalid page dimensions."""
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_doc.__len__ = Mock(return_value=1)
            mock_page = Mock()
            mock_page.rect.width = 50  # Too small
            mock_page.rect.height = 50  # Too small
            mock_doc.__getitem__ = Mock(return_value=mock_page)
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            with pytest.raises(ValueError, match="Invalid page dimensions"):
                processor._validate_pdf_structure(b'%PDF-1.4', 'test.pdf')

    def test_validate_pdf_structure_corrupted(self, processor):
        """Test PDF structure validation with corrupted PDF."""
        with patch('fitz.open', side_effect=Exception("Corrupted PDF")):
            with pytest.raises(ValueError, match="Error validating PDF"):
                processor._validate_pdf_structure(b'corrupted data', 'corrupt.pdf')

    def test_validate_pdf_structure_file_data_error(self, processor):
        """Test PDF structure validation with file data error."""
        import fitz
        with patch('fitz.open', side_effect=fitz.FileDataError("Invalid PDF")):
            with pytest.raises(ValueError, match="corrupted or not a valid PDF"):
                processor._validate_pdf_structure(b'invalid data', 'invalid.pdf')

    # Comprehensive File Validation Tests

    @pytest.mark.asyncio
    async def test_validate_file_success_pdf(self, processor, sample_pdf_data):
        """Test comprehensive file validation for PDF."""
        with patch.object(processor, '_validate_pdf_structure'):
            result = await processor._validate_file(sample_pdf_data, 'test.pdf')
            assert result == 'pdf'

    @pytest.mark.asyncio
    async def test_validate_file_success_image(self, processor, sample_jpg_data):
        """Test comprehensive file validation for image."""
        with patch('PIL.Image.open') as mock_image_open:
            mock_image = Mock()
            mock_image.size = (1000, 800)
            mock_image_open.return_value = mock_image
            
            result = await processor._validate_file(sample_jpg_data, 'test.jpg')
            assert result == 'jpg'

    @pytest.mark.asyncio
    async def test_validate_file_invalid_image(self, processor):
        """Test comprehensive file validation with invalid image."""
        invalid_jpg_data = b'\xff\xd8\xff\xe0invalid'
        
        with patch('PIL.Image.open', side_effect=Exception("Invalid image")):
            with pytest.raises(ValueError, match="Invalid JPG image"):
                await processor._validate_file(invalid_jpg_data, 'invalid.jpg')

    # Format Hint Tests

    def test_get_format_hint_success(self, processor):
        """Test getting format hint from filename."""
        assert processor._get_format_hint('document.pdf') == 'pdf'
        assert processor._get_format_hint('image.jpg') == 'jpg'
        assert processor._get_format_hint('photo.jpeg') == 'jpg'
        assert processor._get_format_hint('screenshot.png') == 'png'

    def test_get_format_hint_no_extension(self, processor):
        """Test getting format hint from filename without extension."""
        assert processor._get_format_hint('document') is None
        assert processor._get_format_hint('') is None

    def test_get_format_hint_unknown_extension(self, processor):
        """Test getting format hint from unknown extension."""
        assert processor._get_format_hint('document.txt') is None
        assert processor._get_format_hint('archive.zip') is None

    def test_get_format_hint_case_insensitive(self, processor):
        """Test format hint detection is case insensitive."""
        assert processor._get_format_hint('Document.PDF') == 'pdf'
        assert processor._get_format_hint('IMAGE.JPG') == 'jpg'
        assert processor._get_format_hint('Photo.JPEG') == 'jpg'
        assert processor._get_format_hint('Screenshot.PNG') == 'png'

    # File Download Tests

    @pytest.mark.asyncio
    async def test_download_file_local_path(self, processor, tmp_path):
        """Test downloading file from local path."""
        test_file = tmp_path / "test.pdf"
        test_content = b'%PDF-1.4\ntest content'
        test_file.write_bytes(test_content)
        
        result = await processor._download_file(str(test_file))
        assert result == test_content

    @pytest.mark.asyncio
    async def test_download_file_local_not_found(self, processor):
        """Test downloading non-existent local file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            await processor._download_file('/nonexistent/file.pdf')

    @pytest.mark.asyncio
    async def test_download_file_local_not_a_file(self, processor, tmp_path):
        """Test downloading local path that's not a file."""
        directory = tmp_path / "not_a_file"
        directory.mkdir()
        
        with pytest.raises(ValueError, match="Path is not a file"):
            await processor._download_file(str(directory))

    @pytest.mark.asyncio
    async def test_download_file_http_url(self, processor):
        """Test downloading file from HTTP URL."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.content = b'http content'
            mock_response.raise_for_status = Mock()
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await processor._download_file('http://example.com/test.pdf')
            assert result == b'http content'

    @pytest.mark.asyncio
    async def test_download_file_https_url(self, processor):
        """Test downloading file from HTTPS URL."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.content = b'https content'
            mock_response.raise_for_status = Mock()
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await processor._download_file('https://example.com/test.pdf')
            assert result == b'https content'

    @pytest.mark.asyncio
    async def test_download_file_s3_url(self, processor):
        """Test downloading file from S3 URL (not implemented)."""
        with pytest.raises(NotImplementedError, match="S3 downloads not yet implemented"):
            await processor._download_file('s3://bucket/key.pdf')

    # Image Preprocessing Tests

    def test_preprocess_image_grayscale_conversion(self, processor):
        """Test image preprocessing converts to grayscale."""
        color_image = Image.new('RGB', (1000, 800), color='red')
        processed = processor._preprocess_image(color_image)
        assert processed.mode == 'L'

    def test_preprocess_image_already_grayscale(self, processor):
        """Test image preprocessing with already grayscale image."""
        gray_image = Image.new('L', (1000, 800), color=128)
        processed = processor._preprocess_image(gray_image)
        assert processed.mode == 'L'
        assert processed.size == (1000, 800)

    def test_preprocess_image_upscaling(self, processor):
        """Test image preprocessing upscales small images."""
        small_image = Image.new('RGB', (400, 300), color='white')
        processed = processor._preprocess_image(small_image)
        
        # Should be upscaled to meet minimum dimensions
        width, height = processed.size
        assert width >= processor.MIN_IMAGE_WIDTH or height >= processor.MIN_IMAGE_HEIGHT

    def test_preprocess_image_no_upscaling_needed(self, processor):
        """Test image preprocessing doesn't upscale large enough images."""
        good_image = Image.new('RGB', (1200, 900), color='white')
        processed = processor._preprocess_image(good_image)
        
        # Size should remain the same (or close due to processing)
        width, height = processed.size
        assert width >= 1200 and height >= 900

    # Image Conversion Tests

    @pytest.mark.asyncio
    async def test_convert_to_images_single_image(self, processor, sample_jpg_data):
        """Test converting single image file to images list."""
        with patch('PIL.Image.open') as mock_open:
            mock_image = Mock()
            mock_open.return_value = mock_image
            
            result = await processor._convert_to_images(sample_jpg_data, 'jpg')
            assert len(result) == 1
            assert result[0] == mock_image

    @pytest.mark.asyncio
    async def test_convert_to_images_pdf_multipage(self, processor, sample_pdf_data):
        """Test converting multipage PDF to images."""
        with patch('fitz.open') as mock_fitz:
            # Mock PDF document with 2 pages
            mock_doc = Mock()
            mock_doc.__len__ = Mock(return_value=2)
            
            # Mock pages with get_pixmap method
            mock_page1 = Mock()
            mock_page2 = Mock()
            
            # Mock pixmaps
            mock_pixmap1 = Mock()
            mock_pixmap1.tobytes.return_value = b'fake_image_data_1'
            mock_pixmap2 = Mock()
            mock_pixmap2.tobytes.return_value = b'fake_image_data_2'
            
            mock_page1.get_pixmap.return_value = mock_pixmap1
            mock_page2.get_pixmap.return_value = mock_pixmap2
            
            mock_doc.__getitem__ = Mock(side_effect=[mock_page1, mock_page2])
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            # Mock PIL Image.open to return valid images
            with patch('PIL.Image.open') as mock_image_open:
                mock_img1 = Mock()
                mock_img1.mode = 'RGB'
                mock_img1.size = (800, 600)
                mock_img2 = Mock() 
                mock_img2.mode = 'RGB'
                mock_img2.size = (800, 600)
                mock_image_open.side_effect = [mock_img1, mock_img2]
                
                images = await processor._convert_to_images(sample_pdf_data, 'pdf')
                
                assert len(images) == 2
                assert images[0] == mock_img1
                assert images[1] == mock_img2

    @pytest.mark.asyncio
    async def test_convert_to_images_pdf_error(self, processor, sample_pdf_data):
        """Test PDF to images conversion error handling."""
        with patch('fitz.open', side_effect=Exception("PDF error")):
            with pytest.raises(Exception, match="Failed to convert PDF to images"):
                await processor._convert_to_images(sample_pdf_data, 'pdf')

    @pytest.mark.asyncio
    async def test_convert_to_images_image_error(self, processor, sample_jpg_data):
        """Test image loading error handling."""
        with patch('PIL.Image.open', side_effect=Exception("Image error")):
            with pytest.raises(Exception, match="Failed to load JPG image"):
                await processor._convert_to_images(sample_jpg_data, 'jpg')

    @pytest.mark.asyncio
    async def test_convert_to_images_unsupported_format(self, processor):
        """Test conversion of unsupported format."""
        with pytest.raises(ValueError, match="Cannot convert format 'unknown' to images"):
            await processor._convert_to_images(b'unknown data', 'unknown')

    @pytest.mark.asyncio
    async def test_convert_to_images_auto_detect(self, processor, sample_png_data):
        """Test image conversion with auto-detection."""
        with patch('PIL.Image.open') as mock_open:
            mock_image = Mock()
            mock_open.return_value = mock_image
            
            result = await processor._convert_to_images(sample_png_data)  # No format specified
            assert len(result) == 1
            assert result[0] == mock_image

    # Initialization Tests

    def test_init_default_values(self):
        """Test DocumentProcessor initialization with default values."""
        processor = DocumentProcessor()
        
        assert processor.supported_formats == ['pdf', 'jpg', 'jpeg', 'png']
        assert processor.max_file_size == 15 * 1024 * 1024
        assert processor.confidence_threshold == 60
        assert processor.enable_camelot == True
        assert len(processor.line_item_patterns) == 4
        assert len(processor.table_patterns) == 6

    def test_init_custom_values(self):
        """Test DocumentProcessor initialization with custom values."""
        processor = DocumentProcessor(
            supported_formats=['pdf'],
            max_file_size=10 * 1024 * 1024,
            confidence_threshold=70,
            enable_camelot=False
        )
        
        assert processor.supported_formats == ['pdf']
        assert processor.max_file_size == 10 * 1024 * 1024
        assert processor.confidence_threshold == 70
        assert processor.enable_camelot == False

    def test_init_invalid_format(self):
        """Test DocumentProcessor initialization with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format: invalid"):
            DocumentProcessor(supported_formats=['invalid'])

    # Class Constants Tests

    def test_format_signatures_exist(self):
        """Test that format signatures are properly defined."""
        assert 'pdf' in DocumentProcessor.FORMAT_SIGNATURES
        assert 'jpg' in DocumentProcessor.FORMAT_SIGNATURES
        assert 'jpeg' in DocumentProcessor.FORMAT_SIGNATURES
        assert 'png' in DocumentProcessor.FORMAT_SIGNATURES
        
        # Check that signatures are lists of bytes
        for fmt, sigs in DocumentProcessor.FORMAT_SIGNATURES.items():
            assert isinstance(sigs, list)
            for sig in sigs:
                assert isinstance(sig, bytes)

    def test_max_file_sizes_exist(self):
        """Test that max file sizes are properly configured."""
        processor = DocumentProcessor()
        
        # Check that max file sizes are configured
        assert hasattr(processor, 'MAX_FILE_SIZES')
        assert 'pdf' in processor.MAX_FILE_SIZES
        assert 'jpg' in processor.MAX_FILE_SIZES
        
        # Check actual configured size (15MB)
        expected_size = 15 * 1024 * 1024
        assert processor.MAX_FILE_SIZES['pdf'] == expected_size
        assert processor.max_file_size == expected_size

    def test_image_quality_thresholds(self):
        """Test that image quality thresholds are reasonable."""
        assert DocumentProcessor.MIN_IMAGE_WIDTH == 800
        assert DocumentProcessor.MIN_IMAGE_HEIGHT == 600
        assert DocumentProcessor.MAX_IMAGE_WIDTH == 4000
        assert DocumentProcessor.MAX_IMAGE_HEIGHT == 4000
        
        # Min should be less than max
        assert DocumentProcessor.MIN_IMAGE_WIDTH < DocumentProcessor.MAX_IMAGE_WIDTH
        assert DocumentProcessor.MIN_IMAGE_HEIGHT < DocumentProcessor.MAX_IMAGE_HEIGHT

    def test_detect_document_type_hospital(self, processor):
        """Test hospital bill document type detection."""
        text = "ABC Hospital Nursing Home admission patient consultation doctor fee"
        result = processor._detect_document_type(text)
        assert result == DocumentType.HOSPITAL_BILL

    def test_detect_document_type_pharmacy(self, processor):
        """Test pharmacy invoice document type detection."""
        text = "XYZ Pharmacy Chemist tablet capsule medicine MRP batch"
        result = processor._detect_document_type(text)
        assert result == DocumentType.PHARMACY_INVOICE

    def test_detect_document_type_diagnostic(self, processor):
        """Test diagnostic report document type detection."""
        text = "Pathology Laboratory blood test scan report specimen"
        result = processor._detect_document_type(text)
        assert result == DocumentType.DIAGNOSTIC_REPORT

    def test_detect_document_type_default(self, processor):
        """Test default document type when no matches."""
        text = "Some random text without medical keywords"
        result = processor._detect_document_type(text)
        assert result == DocumentType.HOSPITAL_BILL

    def test_classify_line_item_consultation(self, processor):
        """Test consultation line item classification."""
        result = processor._classify_line_item("Doctor consultation fee", DocumentType.HOSPITAL_BILL)
        assert result == LineItemType.CONSULTATION

    def test_classify_line_item_diagnostic(self, processor):
        """Test diagnostic line item classification."""
        result = processor._classify_line_item("Blood test CBC", DocumentType.HOSPITAL_BILL)
        assert result == LineItemType.DIAGNOSTIC

    def test_classify_line_item_medication(self, processor):
        """Test medication line item classification."""
        result = processor._classify_line_item("Paracetamol tablets", DocumentType.PHARMACY_INVOICE)
        assert result == LineItemType.MEDICATION

    def test_classify_line_item_other(self, processor):
        """Test other line item classification."""
        result = processor._classify_line_item("Unknown item", DocumentType.HOSPITAL_BILL)
        assert result == LineItemType.OTHER

    def test_extract_line_items(self, processor, sample_extracted_text):
        """Test line item extraction from text."""
        result = processor._extract_line_items(sample_extracted_text, DocumentType.HOSPITAL_BILL)
        
        assert len(result) >= 2  # Should extract at least some items
        
        # Check if amounts are extracted correctly
        amounts = [float(item.total_amount) for item in result]
        assert any(amount > 0 for amount in amounts)
        
        # Check if descriptions are not empty
        descriptions = [item.description for item in result]
        assert all(len(desc) > 0 for desc in descriptions)

    def test_count_tables(self, processor, sample_extracted_text):
        """Test table counting in text."""
        result = processor._count_tables(sample_extracted_text)
        assert result >= 1  # Should detect at least one table

    @pytest.mark.asyncio
    async def test_extract_text_ocr(self, processor, sample_image):
        """Test OCR text extraction."""
        with patch('pytesseract.image_to_data') as mock_ocr:
            mock_ocr.return_value = {
                'text': ['Sample', 'text', 'from', 'OCR'],
                'conf': [85, 90, 80, 88]
            }
            
            text, confidence = await processor._extract_text_ocr([sample_image], Language.ENGLISH)
            
            assert isinstance(text, str)
            assert len(text) > 0
            assert 0 <= confidence <= 1

    @pytest.mark.asyncio
    async def test_extract_text_ocr_multi_language(self, processor, sample_image):
        """Test OCR with multi-language support."""
        with patch('pytesseract.image_to_data') as mock_ocr:
            mock_ocr.return_value = {
                'text': ['Sample', 'हिंदी', 'text'],
                'conf': [85, 90, 80]
            }
            
            text, confidence = await processor._extract_text_ocr([sample_image], Language.HINDI)
            
            assert isinstance(text, str)
            assert 0 <= confidence <= 1

    # Table Extraction Tests

    def test_extract_tables_regex(self, processor, sample_table_text):
        """Test regex-based table extraction."""
        result = processor._extract_tables_regex(sample_table_text)
        
        assert len(result) >= 1
        table = result[0]
        assert isinstance(table, ExtractedTable)
        assert table.extraction_method == 'regex'
        assert len(table.headers) >= 2
        assert len(table.rows) >= 1

    def test_extract_table_from_lines(self, processor):
        """Test extracting table from text lines."""
        lines = [
            "Medical Bill",
            "Description    Qty    Rate    Amount",
            "Consultation    1     500     500",
            "Blood Test      2     200     400",
            "X-Ray          1     800     800",
            "Total: Rs. 1700"
        ]
        
        result = processor._extract_table_from_lines(lines, 1)  # Header at index 1
        
        assert result is not None
        assert result.extraction_method == 'regex'
        assert len(result.headers) == 4
        assert len(result.rows) >= 2
        assert result.confidence > 0

    def test_calculate_row_confidence(self, processor):
        """Test row confidence calculation."""
        headers = ['Description', 'Qty', 'Rate', 'Amount']
        
        # Good row with all columns and numbers
        good_cells = ['Consultation Fee', '1', '500', '500']
        confidence = processor._calculate_row_confidence(good_cells, headers)
        assert confidence > 0.5
        
        # Poor row with missing data
        poor_cells = ['X', '']
        confidence = processor._calculate_row_confidence(poor_cells, headers)
        assert confidence < 0.5

    def test_tables_similar(self, processor):
        """Test table similarity detection."""
        table1 = ExtractedTable(
            page_number=1,
            table_index=0,
            headers=['Description', 'Amount'],
            rows=[['Item 1', '100'], ['Item 2', '200']],
            confidence=0.8,
            extraction_method='regex'
        )
        
        table2 = ExtractedTable(
            page_number=1,
            table_index=1,
            headers=['Description', 'Amount'],
            rows=[['Item 1', '100'], ['Item 2', '200']],
            confidence=0.9,
            extraction_method='camelot'
        )
        
        # Should be similar (same headers and similar rows)
        assert processor._tables_similar(table1, table2) == True
        
        # Different headers should not be similar
        table3 = ExtractedTable(
            page_number=1,
            table_index=2,
            headers=['Name', 'Price'],
            rows=[['Item 1', '100']],
            confidence=0.8,
            extraction_method='regex'
        )
        
        assert processor._tables_similar(table1, table3) == False

    def test_deduplicate_tables(self, processor):
        """Test table deduplication."""
        table1 = ExtractedTable(
            page_number=1,
            table_index=0,
            headers=['Description', 'Amount'],
            rows=[['Item 1', '100']],
            confidence=0.8,
            extraction_method='regex'
        )
        
        table2 = ExtractedTable(
            page_number=1,
            table_index=1,
            headers=['Description', 'Amount'],
            rows=[['Item 1', '100']],
            confidence=0.9,  # Higher confidence
            extraction_method='camelot'
        )
        
        table3 = ExtractedTable(
            page_number=2,
            table_index=0,
            headers=['Service', 'Cost'],
            rows=[['Service 1', '300']],
            confidence=0.85,
            extraction_method='regex'
        )
        
        tables = [table1, table2, table3]
        result = processor._deduplicate_tables(tables)
        
        # Should keep table2 (higher confidence) and table3 (different)
        assert len(result) == 2
        assert table2 in result
        assert table3 in result

    def test_map_table_headers(self, processor):
        """Test table header mapping."""
        headers = ['Item Description', 'Quantity', 'Unit Price', 'Total Amount']
        result = processor._map_table_headers(headers)
        
        assert result['description'] == 0
        assert result['quantity'] == 1
        assert result['rate'] == 2
        assert result['amount'] == 3

    def test_get_table_value(self, processor):
        """Test getting values from table rows."""
        row = ['Item 1', '2', '100', '200']
        
        assert processor._get_table_value(row, 0) == 'Item 1'
        assert processor._get_table_value(row, 3) == '200'
        assert processor._get_table_value(row, 10) is None  # Out of bounds
        assert processor._get_table_value(row, None) is None

    def test_parse_currency_value(self, processor):
        """Test currency value parsing."""
        assert processor._parse_currency_value("1500.50") == Decimal("1500.50")
        assert processor._parse_currency_value("Rs. 1500.50") == Decimal("1500.50")
        assert processor._parse_currency_value("₹1500.50") == Decimal("1500.50")
        assert processor._parse_currency_value("1,500.50") == Decimal("1500.50")
        assert processor._parse_currency_value("Rs 1,500.50") == Decimal("1500.50")
        assert processor._parse_currency_value("invalid") is None
        assert processor._parse_currency_value("") is None

    def test_parse_numeric_value(self, processor):
        """Test numeric value parsing."""
        assert processor._parse_numeric_value('5') == 5
        assert processor._parse_numeric_value('10 pieces') == 10
        assert processor._parse_numeric_value('no number') is None
        assert processor._parse_numeric_value('') is None

    def test_extract_line_items_from_table(self, processor, sample_extracted_table):
        """Test line item extraction from table."""
        result = processor._extract_line_items_from_table(
            sample_extracted_table, 
            DocumentType.HOSPITAL_BILL, 
            0
        )
        
        assert len(result) >= 2  # Should extract multiple items
        
        for item in result:
            assert isinstance(item, ExtractedLineItem)
            assert item.source_method == 'table'
            assert item.source_table == 0
            assert item.total_amount > 0
            assert len(item.description) > 0

    def test_line_items_similar(self, processor):
        """Test line item similarity detection."""
        item1 = ExtractedLineItem(
            description="Blood Test CBC",
            total_amount=Decimal('300'),
            source_method='table'
        )
        
        item2 = ExtractedLineItem(
            description="Blood Test CBC",
            total_amount=Decimal('300'),
            source_method='regex'
        )
        
        # Should be similar (same description and amount)
        assert processor._line_items_similar(item1, item2) == True
        
        item3 = ExtractedLineItem(
            description="X-Ray Chest",
            total_amount=Decimal('800'),
            source_method='regex'
        )
        
        # Should not be similar (different description and amount)
        assert processor._line_items_similar(item1, item3) == False

    def test_deduplicate_line_items(self, processor):
        """Test line item deduplication."""
        item1 = ExtractedLineItem(
            description="Blood Test",
            total_amount=Decimal('300'),
            confidence=0.8,
            source_method='regex'
        )
        
        item2 = ExtractedLineItem(
            description="Blood Test",
            total_amount=Decimal('300'),
            confidence=0.9,  # Higher confidence
            source_method='table'
        )
        
        item3 = ExtractedLineItem(
            description="X-Ray",
            total_amount=Decimal('800'),
            confidence=0.85,
            source_method='table'
        )
        
        items = [item1, item2, item3]
        result = processor._deduplicate_line_items(items)
        
        # Should keep item2 (higher confidence/table source) and item3 (different)
        assert len(result) == 2
        assert item2 in result
        assert item3 in result

    @pytest.mark.asyncio
    async def test_extract_tables_camelot_success(self, processor, sample_pdf_data):
        """Test successful table extraction using Camelot."""
        # Mock camelot.read_pdf
        with patch('camelot.read_pdf') as mock_camelot:
            # Mock camelot table
            mock_table = Mock()
            mock_table.df = Mock()
            mock_table.df.values.tolist.return_value = [
                ['Description', 'Amount'],
                ['Item 1', '100'],
                ['Item 2', '200']
            ]
            mock_table.df.columns.tolist.return_value = ['Description', 'Amount']
            mock_table.accuracy = 0.85
            mock_table.page = 1
            
            # Mock TableList
            mock_table_list = Mock()
            mock_table_list.__len__ = Mock(return_value=1)
            mock_table_list.__getitem__ = Mock(return_value=mock_table)
            mock_camelot.return_value = mock_table_list
            
            # Create a temporary file for camelot
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_temp_file = Mock()
                mock_temp_file.name = '/tmp/test.pdf'
                mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                mock_temp_file.__exit__ = Mock(return_value=None)
                mock_temp.return_value = mock_temp_file
                
                tables = await processor._extract_tables_camelot(sample_pdf_data)
                
                assert len(tables) == 1
                assert tables[0].headers == ['Description', 'Amount']
                assert len(tables[0].rows) == 2
                assert tables[0].confidence == 0.85

    @pytest.mark.asyncio
    async def test_extract_tables_camelot_failure(self, processor, sample_pdf_data):
        """Test Camelot table extraction failure handling."""
        with patch('camelot.read_pdf', side_effect=Exception("Camelot failed")):
            result = await processor._extract_tables_camelot(sample_pdf_data)
            assert result == []  # Should return empty list on failure

    @pytest.mark.asyncio
    async def test_extract_tables_comprehensive(self, processor, sample_pdf_data):
        """Test comprehensive table extraction (Camelot + regex)."""
        with patch.object(processor, '_extract_tables_camelot', return_value=[]), \
             patch.object(processor, '_extract_tables_regex', return_value=[]), \
             patch.object(processor, '_deduplicate_tables', return_value=[]):
            
            result = await processor._extract_tables(sample_pdf_data, 'test.pdf', 'sample text')
            
            assert isinstance(result, list)

    @pytest.mark.asyncio
    @patch('medbillguardagent.document_processor.asyncio.get_event_loop')
    async def test_process_document_with_tables(self, mock_loop, processor, sample_pdf_data):
        """Test document processing with table extraction."""
        # Mock the event loop time
        mock_loop.return_value.time.side_effect = [0.0, 1.0]  # 1 second processing
        
        sample_table = ExtractedTable(
            page_number=1,
            table_index=0,
            headers=['Description', 'Amount'],
            rows=[['Item 1', '100']],
            confidence=0.8,
            extraction_method='camelot'
        )
        
        with patch.object(processor, '_download_file', return_value=sample_pdf_data), \
             patch.object(processor, '_validate_file'), \
             patch.object(processor, '_convert_to_images', return_value=[Mock()]), \
             patch.object(processor, '_preprocess_image', return_value=Mock()), \
             patch.object(processor, '_extract_text_ocr', return_value=("Sample text", 0.85)), \
             patch.object(processor, '_detect_document_type', return_value=DocumentType.HOSPITAL_BILL), \
             patch.object(processor, '_extract_tables', return_value=[sample_table]), \
             patch.object(processor, '_extract_line_items_comprehensive', return_value=[]), \
             patch.object(processor, '_count_tables', return_value=1):
            
            result = await processor.process_document("test.pdf", "doc-123", Language.ENGLISH)
            
            assert isinstance(result, ExtractedDocument)
            assert len(result.tables) == 1
            assert result.stats.tables_extracted == 1
            assert 'table_extraction_method' in result.metadata

    @pytest.mark.asyncio
    async def test_process_document_error_handling(self, processor):
        """Test error handling in document processing."""
        with patch.object(processor, '_download_file', side_effect=Exception("Download failed")):
            with pytest.raises(Exception, match="Download failed"):
                await processor.process_document("test.pdf", "doc-123")

    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test the convenience process_document function."""
        with patch('medbillguardagent.document_processor.DocumentProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_result = Mock()
            # Make the process_document method return an awaitable
            mock_processor.process_document = AsyncMock(return_value=mock_result)
            mock_processor_class.return_value = mock_processor

            result = await process_document("test.pdf", "doc-123", Language.ENGLISH)

            assert result == mock_result
            mock_processor_class.assert_called_once()
            mock_processor.process_document.assert_called_once_with("test.pdf", "doc-123", Language.ENGLISH)


class TestExtractedTable:
    """Test cases for ExtractedTable model."""

    def test_extracted_table_creation(self):
        """Test creating an ExtractedTable."""
        table = ExtractedTable(
            page_number=1,
            table_index=0,
            headers=['Description', 'Amount'],
            rows=[['Item 1', '100'], ['Item 2', '200']],
            confidence=0.85,
            extraction_method='camelot',
            bounding_box={'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}
        )
        
        assert table.page_number == 1
        assert table.table_index == 0
        assert len(table.headers) == 2
        assert len(table.rows) == 2
        assert table.confidence == 0.85
        assert table.extraction_method == 'camelot'
        assert table.bounding_box is not None

    def test_extracted_table_defaults(self):
        """Test ExtractedTable with default values."""
        table = ExtractedTable(
            page_number=1,
            table_index=0,
            headers=['Description'],
            rows=[['Item 1']],
            confidence=0.8,
            extraction_method='regex'
        )
        
        assert table.bounding_box is None


class TestExtractedLineItem:
    """Test cases for ExtractedLineItem model."""

    def test_extracted_line_item_creation(self):
        """Test creating an ExtractedLineItem."""
        item = ExtractedLineItem(
            description="Blood Test",
            quantity=2,
            unit_price=Decimal("200.00"),
            total_amount=Decimal("400.00"),
            item_type=LineItemType.DIAGNOSTIC,
            confidence=0.9,
            source_table=0,
            source_method='table'
        )
        
        assert item.description == "Blood Test"
        assert item.quantity == 2
        assert item.unit_price == Decimal("200.00")
        assert item.total_amount == Decimal("400.00")
        assert item.item_type == LineItemType.DIAGNOSTIC
        assert item.confidence == 0.9
        assert item.source_table == 0
        assert item.source_method == 'table'

    def test_extracted_line_item_defaults(self):
        """Test ExtractedLineItem with default values."""
        item = ExtractedLineItem(
            description="Test Item",
            total_amount=Decimal("100.00")
        )
        
        assert item.quantity == 1
        assert item.unit_price is None
        assert item.item_type == LineItemType.OTHER
        assert item.confidence == 0.0
        assert item.source_table is None
        assert item.source_method == "regex"


class TestProcessingStats:
    """Test cases for ProcessingStats model."""

    def test_processing_stats_creation(self):
        """Test creating ProcessingStats."""
        stats = ProcessingStats(
            pages_processed=3,
            ocr_confidence=0.85,
            text_extracted_chars=1500,
            tables_found=2,
            tables_extracted=1,
            line_items_found=10,
            processing_time_ms=5000
        )
        
        assert stats.pages_processed == 3
        assert stats.ocr_confidence == 0.85
        assert stats.text_extracted_chars == 1500
        assert stats.tables_found == 2
        assert stats.tables_extracted == 1
        assert stats.line_items_found == 10
        assert stats.processing_time_ms == 5000

    def test_processing_stats_defaults(self):
        """Test ProcessingStats with default values."""
        stats = ProcessingStats(
            pages_processed=1,
            ocr_confidence=0.8,
            text_extracted_chars=1000,
            tables_found=1,
            line_items_found=5,
            processing_time_ms=3000
        )
        
        assert stats.tables_extracted == 0  # Default value


class TestExtractedDocument:
    """Test cases for ExtractedDocument model."""

    def test_extracted_document_creation(self):
        """Test creating an ExtractedDocument."""
        stats = ProcessingStats(
            pages_processed=1,
            ocr_confidence=0.8,
            text_extracted_chars=1000,
            tables_found=1,
            tables_extracted=1,
            line_items_found=5,
            processing_time_ms=3000
        )
        
        table = ExtractedTable(
            page_number=1,
            table_index=0,
            headers=['Description', 'Amount'],
            rows=[['Item 1', '100']],
            confidence=0.8,
            extraction_method='camelot'
        )
        
        line_item = ExtractedLineItem(
            description="Test Item",
            total_amount=Decimal("100.00")
        )
        
        doc = ExtractedDocument(
            raw_text="Sample extracted text",
            tables=[table],
            line_items=[line_item],
            document_type=DocumentType.HOSPITAL_BILL,
            language=Language.ENGLISH,
            stats=stats,
            metadata={
                "doc_id": "test-123",
                "file_format": "pdf",
                "file_size_bytes": 1024,
                "ocr_language": "en",
                "table_extraction_method": "camelot+regex",
                "confidence_threshold": 60,
                "processing_warnings": 0
            }
        )
        
        assert doc.raw_text == "Sample extracted text"
        assert len(doc.tables) == 1
        assert len(doc.line_items) == 1
        assert doc.document_type == DocumentType.HOSPITAL_BILL
        assert doc.language == Language.ENGLISH
        assert doc.stats == stats
        assert doc.metadata["doc_id"] == "test-123"

    def test_extracted_document_defaults(self):
        """Test ExtractedDocument with default values."""
        stats = ProcessingStats(
            pages_processed=1,
            ocr_confidence=0.8,
            text_extracted_chars=1000,
            tables_found=1,
            line_items_found=0,
            processing_time_ms=3000
        )
        
        doc = ExtractedDocument(
            raw_text="Sample text",
            line_items=[],
            document_type=DocumentType.HOSPITAL_BILL,
            language=Language.ENGLISH,
            stats=stats,
            metadata={}
        )
        
        assert len(doc.tables) == 0  # Default empty list 