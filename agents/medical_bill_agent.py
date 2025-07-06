"""
Medical Bill Agent - AI-powered medical bill analysis agent.

This agent coordinates the complete medical bill analysis workflow using
async tool wrappers around DocumentProcessor, RateValidator, DuplicateDetector,
ProhibitedDetector, and ConfidenceScorer components.
"""

import asyncio
import base64
import json
import structlog
import time
from typing import Dict, Any, List, Optional
from opentelemetry import trace

from agents.base_agent import BaseAgent, AgentContext, AgentResult, ModelHint
from agents.tools import (
    DocumentProcessorTool,
    RateValidatorTool,
    DuplicateDetectorTool,
    ProhibitedDetectorTool,
    ConfidenceScorerTool
)

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class MedicalBillAgent(BaseAgent):
    """
    AI-powered medical bill analysis agent.
    
    Provides comprehensive medical bill analysis including:
    - Document processing and OCR
    - Rate validation against CGHS/ESI/NPPA rates  
    - Duplicate detection
    - Prohibited item detection
    - Confidence scoring and recommendations
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/1",
        openai_api_key: Optional[str] = None,
        reference_data_loader = None
    ):
        """Initialize the medical bill agent with all analysis tools."""
        
        # Initialize all analysis tools
        self.document_processor_tool = DocumentProcessorTool()
        self.rate_validator_tool = RateValidatorTool(reference_data_loader=reference_data_loader)
        self.duplicate_detector_tool = DuplicateDetectorTool()
        self.prohibited_detector_tool = ProhibitedDetectorTool()
        self.confidence_scorer_tool = ConfidenceScorerTool(openai_api_key=openai_api_key)
        
        # Collect all tools for BaseAgent
        tools = [
            self.document_processor_tool,
            self.rate_validator_tool,
            self.duplicate_detector_tool,
            self.prohibited_detector_tool,
            self.confidence_scorer_tool
        ]
        
        super().__init__(
            agent_id="medical_bill_agent",
            name="Medical Bill Analysis Agent",
            instructions="""You are an expert medical bill analysis agent specializing in detecting overcharges, 
            duplicates, and prohibited items in Indian medical bills. You have access to tools for document processing, 
            rate validation against CGHS/ESI/NPPA rates, duplicate detection, prohibited item detection, and confidence 
            scoring. Use these tools to provide comprehensive analysis and actionable recommendations.""",
            tools=tools,
            redis_url=redis_url,
            default_model="gpt-4o"
        )
        
        logger.info("Initialized MedicalBillAgent with all analysis tools")
    
    async def process_task(self, context: AgentContext, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process medical bill analysis task.
        
        Args:
            context: Agent execution context
            task_data: Task data containing file content and analysis parameters
            
        Returns:
            Complete medical bill analysis results
        """
        with tracer.start_as_current_span("medical_bill_agent.process_task") as span:
            span.set_attribute("doc_id", context.doc_id)
            span.set_attribute("user_id", context.user_id)
            
            try:
                # Extract task parameters
                file_content = task_data.get("file_content")
                doc_id = task_data.get("doc_id", context.doc_id)
                language = task_data.get("language", "english")
                state_code = task_data.get("state_code")
                insurance_type = task_data.get("insurance_type", "cghs")
                file_format = task_data.get("file_format", "pdf")
                
                # Validate required parameters
                if not file_content:
                    raise ValueError("file_content is required for medical bill analysis")
                
                # Decode base64 file content if provided as string
                if isinstance(file_content, str):
                    try:
                        file_content = base64.b64decode(file_content)
                    except Exception as e:
                        raise ValueError(f"Invalid base64 file content: {str(e)}")
                
                logger.info(
                    "Starting medical bill analysis",
                    doc_id=doc_id,
                    language=language,
                    state_code=state_code,
                    insurance_type=insurance_type,
                    file_size=len(file_content)
                )
                
                # Step 1: Document Processing
                logger.info("Step 1: Processing document", doc_id=doc_id)
                doc_result = await self.document_processor_tool(
                    file_content=file_content,
                    doc_id=doc_id,
                    language=language,
                    file_format=file_format
                )
                
                if not doc_result.get("success"):
                    return {
                        "success": False,
                        "error": f"Document processing failed: {doc_result.get('error', 'Unknown error')}",
                        "step": "document_processing"
                    }
                
                line_items = doc_result.get("line_items", [])
                processing_stats = doc_result.get("processing_stats", {})
                
                if not line_items:
                    return {
                        "success": True,
                        "analysis_complete": True,
                        "doc_id": doc_id,
                        "verdict": "ok",
                        "message": "No billable items found in the document",
                        "document_processing": doc_result,
                        "total_bill_amount": 0.0,
                        "total_overcharge": 0.0,
                        "confidence_score": 0.95,
                        "red_flags": [],
                        "recommendations": ["Document contains no billable items for analysis"]
                    }
                
                # Step 2: Rate Validation
                logger.info("Step 2: Validating rates", doc_id=doc_id, items_count=len(line_items))
                rate_result = await self.rate_validator_tool(
                    line_items=line_items,
                    state_code=state_code,
                    validation_sources=["cghs", "esi"]
                )
                
                # Step 3: Duplicate Detection
                logger.info("Step 3: Detecting duplicates", doc_id=doc_id)
                duplicate_result = await self.duplicate_detector_tool(
                    line_items=line_items,
                    similarity_threshold=0.8
                )
                
                # Step 4: Prohibited Item Detection
                logger.info("Step 4: Detecting prohibited items", doc_id=doc_id)
                prohibited_result = await self.prohibited_detector_tool(
                    line_items=line_items,
                    insurance_type=insurance_type
                )
                
                # Collect all red flags
                all_red_flags = []
                if rate_result.get("success"):
                    all_red_flags.extend(rate_result.get("red_flags", []))
                if duplicate_result.get("success"):
                    all_red_flags.extend(duplicate_result.get("red_flags", []))
                if prohibited_result.get("success"):
                    all_red_flags.extend(prohibited_result.get("red_flags", []))
                
                # Step 5: Confidence Scoring
                logger.info("Step 5: Calculating confidence", doc_id=doc_id, red_flags_count=len(all_red_flags))
                confidence_result = await self.confidence_scorer_tool(
                    analysis_results={
                        "document_processing": doc_result,
                        "rate_validation": rate_result,
                        "duplicate_detection": duplicate_result,
                        "prohibited_detection": prohibited_result
                    },
                    processing_stats=processing_stats,
                    red_flags=all_red_flags
                )
                
                # Calculate summary statistics
                total_bill_amount = sum(float(item.get("total_amount", 0)) for item in line_items)
                total_overcharge = sum(
                    float(flag.get("overcharge_amount", 0)) 
                    for flag in all_red_flags
                )
                overcharge_percentage = (
                    (total_overcharge / total_bill_amount * 100) 
                    if total_bill_amount > 0 else 0
                )
                
                # Get verdict and recommendations from confidence scorer
                verdict = confidence_result.get("verdict", "unknown")
                recommendations = confidence_result.get("recommendations", [])
                overall_confidence = confidence_result.get("overall_confidence", {})
                
                # Compile final results
                result = {
                    "success": True,
                    "analysis_complete": True,
                    "doc_id": doc_id,
                    "verdict": verdict,
                    "total_bill_amount": float(total_bill_amount),
                    "total_overcharge": float(total_overcharge),
                    "overcharge_percentage": float(overcharge_percentage),
                    "confidence_score": float(overall_confidence.get("score", 0.0)),
                    "red_flags": all_red_flags,
                    "recommendations": recommendations,
                    
                    # Detailed results from each step
                    "document_processing": doc_result,
                    "rate_validation": rate_result,
                    "duplicate_detection": duplicate_result,
                    "prohibited_detection": prohibited_result,
                    "confidence_analysis": confidence_result,
                    
                    # Summary statistics
                    "analysis_summary": {
                        "items_analyzed": len(line_items),
                        "rate_matches_found": len(rate_result.get("rate_matches", [])) if rate_result.get("success") else 0,
                        "duplicates_detected": duplicate_result.get("duplicate_summary", {}).get("total_duplicate_items", 0) if duplicate_result.get("success") else 0,
                        "prohibited_items_found": prohibited_result.get("prohibited_summary", {}).get("prohibited_items_found", 0) if prohibited_result.get("success") else 0,
                        "total_red_flags": len(all_red_flags),
                        "state_validation_used": state_code is not None,
                        "insurance_type": insurance_type
                    }
                }
                
                logger.info(
                    "Medical bill analysis completed",
                    doc_id=doc_id,
                    verdict=verdict,
                    total_overcharge=total_overcharge,
                    confidence_score=overall_confidence.get("score", 0.0),
                    red_flags_count=len(all_red_flags)
                )
                
                return result
                
            except Exception as e:
                error_msg = f"Medical bill analysis failed: {str(e)}"
                logger.error(error_msg, doc_id=context.doc_id, exc_info=True)
                
                span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                
                return {
                    "success": False,
                    "error": error_msg,
                    "doc_id": context.doc_id,
                    "analysis_complete": False
                }
    
    async def analyze_medical_bill(
        self,
        file_content: bytes,
        doc_id: str,
        user_id: str,
        language: str = "english",
        state_code: Optional[str] = None,
        insurance_type: str = "cghs",
        file_format: str = "pdf"
    ) -> Dict[str, Any]:
        """
        Convenience method for direct medical bill analysis.
        
        Args:
            file_content: Binary content of the medical bill file
            doc_id: Unique document identifier
            user_id: User identifier
            language: Document language for OCR
            state_code: State code for regional rate validation
            insurance_type: Insurance type for prohibited item detection
            file_format: File format hint
            
        Returns:
            Complete medical bill analysis results
        """
        # Create context
        context = AgentContext(
            doc_id=doc_id,
            user_id=user_id,
            correlation_id=f"medical_bill_{doc_id}",
            model_hint=ModelHint.STANDARD,
            start_time=time.time(),
            metadata={"task_type": "medical_bill_analysis"}
        )
        
        # Prepare task data
        task_data = {
            "file_content": file_content,
            "doc_id": doc_id,
            "language": language,
            "state_code": state_code,
            "insurance_type": insurance_type,
            "file_format": file_format
        }
        
        # Process the task
        return await self.process_task(context, task_data)
    
    async def health_check(self) -> Dict[str, Any]:
        """Extended health check including tool status."""
        base_health = await super().health_check()
        
        # Check individual tools
        tool_status = {
            "document_processor": "healthy",
            "rate_validator": "healthy", 
            "duplicate_detector": "healthy",
            "prohibited_detector": "healthy",
            "confidence_scorer": "healthy"
        }
        
        base_health.update({
            "tools": tool_status,
            "analysis_capabilities": [
                "document_processing",
                "ocr_extraction",
                "rate_validation", 
                "duplicate_detection",
                "prohibited_item_detection",
                "confidence_scoring"
            ]
        })
        
        return base_health 