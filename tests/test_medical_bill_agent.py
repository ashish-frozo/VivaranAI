"""
Unit tests for MedicalBillAgent and async tool wrappers.

Tests the complete medical bill analysis workflow including tool integration,
error handling, and result aggregation.
"""

import asyncio
import base64
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from agents.medical_bill_agent import MedicalBillAgent
from agents.base_agent import AgentContext, ModelHint
from agents.tools import (
    DocumentProcessorTool,
    RateValidatorTool,
    DuplicateDetectorTool,
    ProhibitedDetectorTool,
    ConfidenceScorerTool
)


@pytest.fixture
def sample_file_content():
    """Sample medical bill file content (base64 encoded PDF)."""
    # This would be actual PDF content in practice
    sample_content = b"Sample medical bill PDF content"
    return base64.b64encode(sample_content).decode('utf-8')


@pytest.fixture
def sample_line_items():
    """Sample line items extracted from document."""
    return [
        {
            "description": "Specialist Consultation",
            "quantity": 1,
            "unit_price": 800.0,
            "total_amount": 800.0,
            "item_type": "consultation",
            "confidence": 0.95,
            "source_method": "table"
        },
        {
            "description": "Complete Blood Count (CBC)",
            "quantity": 1,
            "unit_price": 450.0,
            "total_amount": 450.0,
            "item_type": "diagnostic",
            "confidence": 0.92,
            "source_method": "table"
        },
        {
            "description": "X-Ray Chest",
            "quantity": 1,
            "unit_price": 600.0,
            "total_amount": 600.0,
            "item_type": "diagnostic",
            "confidence": 0.89,
            "source_method": "regex"
        }
    ]


@pytest.fixture
def sample_processing_stats():
    """Sample document processing statistics."""
    return {
        "pages_processed": 2,
        "ocr_confidence": 94.5,
        "text_extracted_chars": 2847,
        "tables_found": 1,
        "tables_extracted": 1,
        "line_items_found": 3,
        "processing_time_ms": 1250,
        "errors_encountered": []
    }


@pytest.fixture
def agent_context():
    """Sample agent context for testing."""
    return AgentContext(
        doc_id="test_bill_001",
        user_id="user_123",
        correlation_id="corr_456",
        model_hint=ModelHint.STANDARD,
        start_time=time.time(),
        metadata={"task_type": "medical_bill_analysis"}
    )


class TestDocumentProcessorTool:
    """Test DocumentProcessorTool functionality."""
    
    @pytest.mark.asyncio
    async def test_document_processing_success(self, sample_file_content):
        """Test successful document processing."""
        tool = DocumentProcessorTool()
        
        with patch.object(tool.processor, 'process_document') as mock_process:
            # Mock successful processing
            mock_result = MagicMock()
            mock_result.raw_text = "Sample medical bill text"
            mock_result.document_type.value = "hospital_bill"
            mock_result.language.value = "english"
            mock_result.line_items = [
                MagicMock(
                    description="Test Item",
                    quantity=1,
                    unit_price=100.0,
                    total_amount=100.0,
                    item_type=MagicMock(value="consultation"),
                    confidence=0.9,
                    source_method="table"
                )
            ]
            mock_result.tables = []
            mock_result.processing_stats = MagicMock(
                pages_processed=1,
                ocr_confidence=95.0,
                text_extracted_chars=1000,
                tables_found=0,
                tables_extracted=0,
                line_items_found=1,
                processing_time_ms=500,
                errors_encountered=[]
            )
            mock_result.metadata = {"test": "metadata"}
            
            mock_process.return_value = mock_result
            
            # Test the tool
            file_content = base64.b64decode(sample_file_content)
            result = await tool(
                file_content=file_content,
                doc_id="test_doc",
                language="english",
                file_format="pdf"
            )
            
            assert result["success"] is True
            assert result["doc_id"] == "test_doc"
            assert len(result["line_items"]) == 1
            assert result["line_items"][0]["description"] == "Test Item"
            assert result["processing_stats"]["pages_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_document_processing_failure(self, sample_file_content):
        """Test document processing failure handling."""
        tool = DocumentProcessorTool()
        
        with patch.object(tool.processor, 'process_document') as mock_process:
            mock_process.side_effect = Exception("OCR failed")
            
            file_content = base64.b64decode(sample_file_content)
            result = await tool(
                file_content=file_content,
                doc_id="test_doc",
                language="english"
            )
            
            assert result["success"] is False
            assert "OCR failed" in result["error"]
            assert result["line_items"] == []


class TestRateValidatorTool:
    """Test RateValidatorTool functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_validation_success(self, sample_line_items):
        """Test successful rate validation."""
        tool = RateValidatorTool()
        
        with patch.object(tool.validator, 'validate_item_rates') as mock_validate:
            with patch.object(tool.validator, 'generate_red_flags') as mock_flags:
                # Mock rate matches
                mock_rate_match = MagicMock()
                mock_rate_match.bill_item = "Specialist Consultation"
                mock_rate_match.reference_item = "Specialist Consultation - CGHS"
                mock_rate_match.billed_amount = 800.0
                mock_rate_match.reference_rate = 500.0
                mock_rate_match.overcharge_amount = 300.0
                mock_rate_match.overcharge_percentage = 60.0
                mock_rate_match.source.value = "cghs"
                mock_rate_match.confidence = 0.95
                mock_rate_match.item_type.value = "consultation"
                mock_rate_match.match_method = "exact"
                mock_rate_match.state_code = None
                
                mock_validate.return_value = [mock_rate_match]
                mock_flags.return_value = [
                    {
                        "type": "overcharge",
                        "severity": "high",
                        "item": "Specialist Consultation",
                        "reason": "Charged 60% above CGHS rate",
                        "overcharge_amount": 300.0,
                        "confidence": 0.95
                    }
                ]
                
                result = await tool(
                    line_items=sample_line_items,
                    state_code="DL",
                    validation_sources=["cghs", "esi"]
                )
                
                assert result["success"] is True
                assert len(result["rate_matches"]) == 1
                assert len(result["red_flags"]) == 1
                assert result["validation_summary"]["total_overcharge"] == 300.0
    
    @pytest.mark.asyncio
    async def test_rate_validation_empty_items(self):
        """Test rate validation with empty line items."""
        tool = RateValidatorTool()
        
        result = await tool(line_items=[], state_code=None)
        
        assert result["success"] is True
        assert result["rate_matches"] == []
        assert result["red_flags"] == []
        assert result["validation_summary"]["total_items"] == 0


class TestMedicalBillAgent:
    """Test MedicalBillAgent complete workflow."""
    
    @pytest.mark.asyncio
    async def test_medical_bill_agent_initialization(self):
        """Test agent initialization with all tools."""
        agent = MedicalBillAgent()
        
        assert agent.agent_id == "medical_bill_agent"
        assert agent.name == "Medical Bill Analysis Agent"
        assert len(agent.tools) == 5
        assert isinstance(agent.document_processor_tool, DocumentProcessorTool)
        assert isinstance(agent.rate_validator_tool, RateValidatorTool)
        assert isinstance(agent.duplicate_detector_tool, DuplicateDetectorTool)
        assert isinstance(agent.prohibited_detector_tool, ProhibitedDetectorTool)
        assert isinstance(agent.confidence_scorer_tool, ConfidenceScorerTool)
    
    @pytest.mark.asyncio
    async def test_process_task_success(self, agent_context, sample_file_content, sample_line_items, sample_processing_stats):
        """Test successful medical bill analysis workflow."""
        agent = MedicalBillAgent()
        
        # Mock all tool responses
        with patch.object(agent.document_processor_tool, '__call__') as mock_doc:
            with patch.object(agent.rate_validator_tool, '__call__') as mock_rate:
                with patch.object(agent.duplicate_detector_tool, '__call__') as mock_duplicate:
                    with patch.object(agent.prohibited_detector_tool, '__call__') as mock_prohibited:
                        with patch.object(agent.confidence_scorer_tool, '__call__') as mock_confidence:
                            
                            # Mock document processing
                            mock_doc.return_value = {
                                "success": True,
                                "doc_id": "test_bill_001",
                                "line_items": sample_line_items,
                                "processing_stats": sample_processing_stats,
                                "raw_text": "Sample text",
                                "document_type": "hospital_bill"
                            }
                            
                            # Mock rate validation
                            mock_rate.return_value = {
                                "success": True,
                                "rate_matches": [
                                    {
                                        "bill_item": "Specialist Consultation",
                                        "overcharge_amount": 300.0,
                                        "confidence": 0.95
                                    }
                                ],
                                "red_flags": [
                                    {
                                        "type": "overcharge",
                                        "severity": "high",
                                        "item": "Specialist Consultation",
                                        "overcharge_amount": 300.0,
                                        "confidence": 0.95
                                    }
                                ],
                                "validation_summary": {
                                    "total_overcharge": 300.0
                                }
                            }
                            
                            # Mock duplicate detection
                            mock_duplicate.return_value = {
                                "success": True,
                                "duplicate_groups": [],
                                "red_flags": [],
                                "duplicate_summary": {
                                    "total_duplicate_items": 0
                                }
                            }
                            
                            # Mock prohibited detection
                            mock_prohibited.return_value = {
                                "success": True,
                                "prohibited_items": [],
                                "red_flags": [],
                                "prohibited_summary": {
                                    "prohibited_items_found": 0
                                }
                            }
                            
                            # Mock confidence scoring
                            mock_confidence.return_value = {
                                "success": True,
                                "overall_confidence": {
                                    "score": 0.89,
                                    "source": "hybrid",
                                    "reasoning": "High confidence analysis"
                                },
                                "verdict": "warning",
                                "recommendations": [
                                    "Review flagged overcharges with hospital"
                                ]
                            }
                            
                            # Test the complete workflow
                            task_data = {
                                "file_content": sample_file_content,
                                "doc_id": "test_bill_001",
                                "language": "english",
                                "state_code": "DL",
                                "insurance_type": "cghs",
                                "file_format": "pdf"
                            }
                            
                            result = await agent.process_task(agent_context, task_data)
                            
                            assert result["success"] is True
                            assert result["analysis_complete"] is True
                            assert result["doc_id"] == "test_bill_001"
                            assert result["verdict"] == "warning"
                            assert result["total_bill_amount"] == 1850.0  # Sum of all line items
                            assert result["total_overcharge"] == 300.0
                            assert result["confidence_score"] == 0.89
                            assert len(result["red_flags"]) == 1
                            assert len(result["recommendations"]) == 1
                            
                            # Verify analysis summary
                            summary = result["analysis_summary"]
                            assert summary["items_analyzed"] == 3
                            assert summary["rate_matches_found"] == 1
                            assert summary["duplicates_detected"] == 0
                            assert summary["prohibited_items_found"] == 0
                            assert summary["total_red_flags"] == 1
                            assert summary["state_validation_used"] is True
                            assert summary["insurance_type"] == "cghs"
    
    @pytest.mark.asyncio
    async def test_process_task_document_processing_failure(self, agent_context, sample_file_content):
        """Test handling of document processing failure."""
        agent = MedicalBillAgent()
        
        with patch.object(agent.document_processor_tool, '__call__') as mock_doc:
            mock_doc.return_value = {
                "success": False,
                "error": "OCR processing failed"
            }
            
            task_data = {
                "file_content": sample_file_content,
                "doc_id": "test_bill_001"
            }
            
            result = await agent.process_task(agent_context, task_data)
            
            assert result["success"] is False
            assert "Document processing failed" in result["error"]
            assert result["step"] == "document_processing"
    
    @pytest.mark.asyncio
    async def test_process_task_no_line_items(self, agent_context, sample_file_content, sample_processing_stats):
        """Test handling when no line items are found."""
        agent = MedicalBillAgent()
        
        with patch.object(agent.document_processor_tool, '__call__') as mock_doc:
            mock_doc.return_value = {
                "success": True,
                "doc_id": "test_bill_001",
                "line_items": [],  # No line items found
                "processing_stats": sample_processing_stats
            }
            
            task_data = {
                "file_content": sample_file_content,
                "doc_id": "test_bill_001"
            }
            
            result = await agent.process_task(agent_context, task_data)
            
            assert result["success"] is True
            assert result["analysis_complete"] is True
            assert result["verdict"] == "ok"
            assert result["total_bill_amount"] == 0.0
            assert result["total_overcharge"] == 0.0
            assert result["confidence_score"] == 0.95
            assert "no billable items" in result["message"].lower()
    
    @pytest.mark.asyncio
    async def test_process_task_missing_file_content(self, agent_context):
        """Test handling of missing file content."""
        agent = MedicalBillAgent()
        
        task_data = {
            "doc_id": "test_bill_001"
            # Missing file_content
        }
        
        result = await agent.process_task(agent_context, task_data)
        
        assert result["success"] is False
        assert "file_content is required" in result["error"]
    
    @pytest.mark.asyncio
    async def test_process_task_invalid_base64(self, agent_context):
        """Test handling of invalid base64 content."""
        agent = MedicalBillAgent()
        
        task_data = {
            "file_content": "invalid_base64_content!!!",
            "doc_id": "test_bill_001"
        }
        
        result = await agent.process_task(agent_context, task_data)
        
        assert result["success"] is False
        assert "Invalid base64 file content" in result["error"]
    
    @pytest.mark.asyncio
    async def test_analyze_medical_bill_convenience_method(self, sample_file_content):
        """Test the convenience method for direct analysis."""
        agent = MedicalBillAgent()
        
        # Mock the process_task method
        with patch.object(agent, 'process_task') as mock_process:
            mock_process.return_value = {
                "success": True,
                "analysis_complete": True,
                "verdict": "ok"
            }
            
            file_content = base64.b64decode(sample_file_content)
            result = await agent.analyze_medical_bill(
                file_content=file_content,
                doc_id="test_bill_001",
                user_id="user_123",
                language="english",
                state_code="DL",
                insurance_type="cghs",
                file_format="pdf"
            )
            
            assert result["success"] is True
            assert result["analysis_complete"] is True
            assert result["verdict"] == "ok"
            
            # Verify process_task was called with correct parameters
            mock_process.assert_called_once()
            call_args = mock_process.call_args
            context, task_data = call_args[0]
            
            assert context.doc_id == "test_bill_001"
            assert context.user_id == "user_123"
            assert task_data["language"] == "english"
            assert task_data["state_code"] == "DL"
            assert task_data["insurance_type"] == "cghs"
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test agent health check."""
        agent = MedicalBillAgent()
        
        # Mock Redis ping for base health check
        with patch.object(agent, 'redis_client') as mock_redis:
            mock_redis.ping = AsyncMock()
            
            health = await agent.health_check()
            
            assert health["agent_id"] == "medical_bill_agent"
            assert health["name"] == "Medical Bill Analysis Agent"
            assert "tools" in health
            assert len(health["analysis_capabilities"]) == 6
            assert "document_processing" in health["analysis_capabilities"]
            assert "rate_validation" in health["analysis_capabilities"]


class TestToolIntegration:
    """Test integration between tools and error propagation."""
    
    @pytest.mark.asyncio
    async def test_tool_error_propagation(self, agent_context, sample_file_content):
        """Test that tool errors are properly handled and propagated."""
        agent = MedicalBillAgent()
        
        with patch.object(agent.document_processor_tool, '__call__') as mock_doc:
            # Simulate tool raising an exception
            mock_doc.side_effect = Exception("Tool execution failed")
            
            task_data = {
                "file_content": sample_file_content,
                "doc_id": "test_bill_001"
            }
            
            result = await agent.process_task(agent_context, task_data)
            
            assert result["success"] is False
            assert "Medical bill analysis failed" in result["error"]
            assert result["analysis_complete"] is False
    
    @pytest.mark.asyncio
    async def test_partial_tool_failures(self, agent_context, sample_file_content, sample_line_items, sample_processing_stats):
        """Test handling when some tools fail but others succeed."""
        agent = MedicalBillAgent()
        
        with patch.object(agent.document_processor_tool, '__call__') as mock_doc:
            with patch.object(agent.rate_validator_tool, '__call__') as mock_rate:
                with patch.object(agent.duplicate_detector_tool, '__call__') as mock_duplicate:
                    with patch.object(agent.prohibited_detector_tool, '__call__') as mock_prohibited:
                        with patch.object(agent.confidence_scorer_tool, '__call__') as mock_confidence:
                            
                            # Document processing succeeds
                            mock_doc.return_value = {
                                "success": True,
                                "line_items": sample_line_items,
                                "processing_stats": sample_processing_stats
                            }
                            
                            # Rate validation fails
                            mock_rate.return_value = {
                                "success": False,
                                "error": "Rate validation failed",
                                "red_flags": []
                            }
                            
                            # Other tools succeed with empty results
                            mock_duplicate.return_value = {
                                "success": True,
                                "red_flags": [],
                                "duplicate_summary": {"total_duplicate_items": 0}
                            }
                            
                            mock_prohibited.return_value = {
                                "success": True,
                                "red_flags": [],
                                "prohibited_summary": {"prohibited_items_found": 0}
                            }
                            
                            mock_confidence.return_value = {
                                "success": True,
                                "overall_confidence": {"score": 0.8},
                                "verdict": "ok",
                                "recommendations": []
                            }
                            
                            task_data = {
                                "file_content": sample_file_content,
                                "doc_id": "test_bill_001"
                            }
                            
                            result = await agent.process_task(agent_context, task_data)
                            
                            # Analysis should still complete successfully
                            assert result["success"] is True
                            assert result["analysis_complete"] is True
                            
                            # Should handle failed rate validation gracefully
                            assert result["analysis_summary"]["rate_matches_found"] == 0 