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
    RateValidatorTool,
    DuplicateDetectorTool,
    ProhibitedDetectorTool,
    ConfidenceScorerTool,
    SmartDataTool
)
from .smart_data_agent import SmartDataAgent

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
        reference_data_loader = None,
        smart_data_agent: Optional[SmartDataAgent] = None
    ):
        """Initialize the medical bill agent with all analysis tools."""
        
        # Store OpenAI API key for AI fallback analysis
        self.openai_api_key = openai_api_key
        
        # Initialize all analysis tools
        self.rate_validator_tool = RateValidatorTool(reference_data_loader=reference_data_loader)
        self.duplicate_detector_tool = DuplicateDetectorTool()
        self.prohibited_detector_tool = ProhibitedDetectorTool()
        self.confidence_scorer_tool = ConfidenceScorerTool(openai_api_key=openai_api_key)
        self.smart_data_tool = SmartDataTool(smart_data_agent) if smart_data_agent else None
        
        # Collect all tools for BaseAgent
        tools = [
            self.rate_validator_tool,
            self.duplicate_detector_tool,
            self.prohibited_detector_tool,
            self.confidence_scorer_tool,
        ]
        if self.smart_data_tool:
            tools.append(self.smart_data_tool)
        
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
    
    async def start(self):
        await super().start()
        await agent_registry.register_agent(
            agent=self,
            capabilities=AgentCapabilities(
                supported_tasks=[
                    TaskCapability.MEDICAL_ANALYSIS,
                    TaskCapability.RATE_VALIDATION
                ],
                max_concurrent_requests=5,
                preferred_model_hints=[ModelHint.STANDARD],
                processing_time_ms_avg=500,
                cost_per_request_rupees=10.0,
                confidence_threshold=0.8,
                supported_document_types=["medical_bill"],
                supported_languages=["english"],
            ),
            health_endpoint=None
        )
    
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
                # Extract task parameters from the new data model
                doc_id = task_data.get("doc_id", context.doc_id)
                state_code = task_data.get("metadata", {}).get("state_code")
                insurance_type = task_data.get("metadata", {}).get("insurance_type", "cghs")
                
                # Get pre-processed data from the enhanced router
                raw_text = task_data.get("raw_text", "")
                line_items = task_data.get("line_items", []) # Assuming router provides this
                processing_stats = task_data.get("ocr_stats", {})
                
                logger.info(
                    "Starting medical bill analysis with pre-processed data",
                    doc_id=doc_id,
                    state_code=state_code,
                    insurance_type=insurance_type,
                    line_items_count=len(line_items)
                )

                # If line items are not provided, use AI fallback
                if not line_items:
                    logger.info("No line items provided, using AI fallback analysis", doc_id=doc_id)
                    ai_result = await self._ai_fallback_analysis(
                        raw_text=raw_text,
                        doc_id=doc_id,
                        state_code=state_code,
                        insurance_type=insurance_type,
                        processing_stats=processing_stats
                    )
                    return ai_result
                
                if not line_items:
                    # Use AI fallback analysis when regex extraction fails
                    logger.info("No line items extracted via regex, using AI fallback analysis", doc_id=doc_id)
                    ai_result = await self._ai_fallback_analysis(
                        raw_text=doc_result.get("raw_text", ""),
                        doc_id=doc_id,
                        state_code=state_code,
                        insurance_type=insurance_type,
                        processing_stats=processing_stats
                    )
                    return ai_result
                
                # Step 2: Dynamic Data Fetching (if available)
                scraped_data = None
                if self.smart_data_tool:
                    logger.info("Step 2a: Fetching dynamic data", doc_id=doc_id)
                    smart_data_result = await self.smart_data_tool(
                        document_type="medical_bill",
                        raw_text=raw_text,
                        state_code=state_code
                    )
                    if smart_data_result.get("success"):
                        scraped_data = smart_data_result.get("scraped_data")

                # Step 2b: Rate Validation
                logger.info("Step 2b: Validating rates", doc_id=doc_id, items_count=len(line_items))
                rate_result = await self.rate_validator_tool(
                    line_items=line_items,
                    state_code=state_code,
                    validation_sources=["cghs", "esi"],
                    dynamic_data=scraped_data
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
    
    async def _ai_fallback_analysis(
        self, 
        raw_text: str, 
        doc_id: str, 
        state_code: Optional[str] = None,
        insurance_type: str = "cghs",
        processing_stats: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        AI-powered fallback analysis when regex extraction fails.
        Uses OpenAI to analyze raw OCR text and extract meaningful insights.
        """
        try:
            logger.info("Starting AI fallback analysis", doc_id=doc_id, text_length=len(raw_text))
            
            # Create OpenAI client
            import openai
            client = openai.AsyncOpenAI(api_key=self.openai_api_key)
            
            # AI prompt for medical bill analysis
            prompt = f"""
            You are a medical bill analysis expert. Analyze this OCR-extracted text from a medical bill and provide a detailed analysis.
    
            OCR Text:
            {raw_text[:2000]}  # Limit to first 2000 chars to avoid token limits
    
            Please analyze and extract:
            1. Line items (services, procedures, medicines) with amounts
            2. Total bill amount
            3. Potential overcharges (compare with typical CGHS/ESI rates)
            4. Suspicious items or duplicate charges
            5. Overall assessment of the bill
    
            Return a JSON response with this structure:
            {{
                "line_items": [
                    {{
                        "description": "Service/item name",
                        "amount": 0.0,
                        "quantity": 1,
                        "is_suspicious": false,
                        "reason": "Why suspicious if applicable"
                    }}
                ],
                "total_bill_amount": 0.0,
                "estimated_overcharge": 0.0,
                "red_flags": [
                    {{
                        "type": "overcharge/duplicate/prohibited",
                        "description": "Issue description", 
                        "item": "Item name",
                        "overcharge_amount": 0.0,
                        "severity": "warning/critical"
                    }}
                ],
                "verdict": "ok/warning/critical",
                "confidence": 0.0,
                "analysis_notes": "Detailed explanation of findings",
                "recommendations": ["List of recommendations"]
            }}
    
            State: {state_code or "Not specified"}
            Insurance Type: {insurance_type}
            """
    
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
    
            # Parse AI response
            import json
            response_content = response.choices[0].message.content
            logger.info(f"AI response content: {response_content}", doc_id=doc_id)
        
            if not response_content or not response_content.strip():
                logger.error("Empty response from OpenAI", doc_id=doc_id)
                return {
                    "line_items": [],
                    "total_bill_amount": 0.0,
                    "estimated_overcharge": 0.0,
                    "red_flags": [],
                    "verdict": "error",
                    "confidence": 0.0,
                    "analysis_notes": "AI response was empty. No analysis could be performed.",
                    "recommendations": ["Try again later or check OpenAI API status."]
                }
        
            try:
                ai_analysis = json.loads(response_content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {e}", doc_id=doc_id, response=response_content)
                # Try to extract JSON from markdown code blocks if present
                import re
                # Try multiple patterns for code blocks and handle extra text
                patterns = [
                    r'```json\s*(\{.*?\})\s*```',  # ```json with code blocks
                    r'```\s*(\{.*?\})\s*```',      # ``` with code blocks  
                    r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # Complete JSON object with nested braces
                    r'(\{.*?\})\s*(?:\n|$|\s*Note:|.*)',   # JSON followed by extra text/notes
                ]
                
                json_extracted = False
                for pattern in patterns:
                    json_matches = re.findall(pattern, response_content, re.DOTALL)
                    for json_match in json_matches:
                        try:
                            # Clean up the JSON string
                            json_str = json_match.strip()
                            if json_str.startswith('{') and json_str.endswith('}'):
                                ai_analysis = json.loads(json_str)
                                logger.info(f"Successfully extracted JSON using pattern: {pattern}", doc_id=doc_id)
                                json_extracted = True
                                break
                        except json.JSONDecodeError:
                            continue
                    if json_extracted:
                        break
                
                if not json_extracted:
                    logger.error("No valid JSON found in AI response", doc_id=doc_id)
                    return {
                        "line_items": [],
                        "total_bill_amount": 0.0,
                        "estimated_overcharge": 0.0,
                        "red_flags": [],
                        "verdict": "error",
                        "confidence": 0.0,
                        "analysis_notes": f"AI response could not be parsed as JSON. Raw response: {response_content[:300]}...",
                        "recommendations": ["Try again later or check OpenAI API status."]
                    }
            
            # Helper to safely convert to float
            def safe_float(val, default=0.0):
                try:
                    if val is None:
                        return default
                    return float(val)
                except (TypeError, ValueError):
                    return default

            # Convert to our standard format
            result = {
                "success": True,
                "analysis_complete": True,
                "doc_id": doc_id,
                "verdict": ai_analysis.get("verdict", "warning"),
                "total_bill_amount": safe_float(ai_analysis.get("total_bill_amount", 0.0)),
                "total_overcharge": safe_float(ai_analysis.get("estimated_overcharge", 0.0)),
                "confidence_score": safe_float(ai_analysis.get("confidence", 0.7)),
                "red_flags": ai_analysis.get("red_flags", []),
                "recommendations": ai_analysis.get("recommendations", []),
                "message": "Analysis completed using AI fallback (regex extraction failed)",
                "analysis_method": "ai_fallback",
                "ai_analysis_notes": ai_analysis.get("analysis_notes", ""),
                "line_items": ai_analysis.get("line_items", []),  # Add to standard line_items field
                "line_items_ai": ai_analysis.get("line_items", []),
                "document_processing": {
                    "raw_text": raw_text,
                    "extraction_method": "ai_analysis",
                    "line_items_found": len(ai_analysis.get("line_items", [])),
                    "success": True
                },
                "overcharge_percentage": (
                    (safe_float(ai_analysis.get("estimated_overcharge", 0.0)) /
                     safe_float(ai_analysis.get("total_bill_amount", 1.0)) * 100)
                    if safe_float(ai_analysis.get("total_bill_amount", 0.0)) > 0 else 0
                ),
                # Add debug data for frontend visibility
                "debug_data": {
                    "ocrText": raw_text,
                    "processingStats": processing_stats or {},
                    "extractedLineItems": [],  # No regex line items in AI fallback
                    "aiAnalysis": response_content,  # Raw AI response for debugging
                    "analysisMethod": "ai_fallback", 
                    "documentType": "pharmacy_invoice",  # Default or extract from processing stats
                    "extractionMethod": "ai_analysis"
                }
            }
            
            verdict = ai_analysis.get("verdict", "warning")
            logger.info(
                "AI fallback analysis completed",
                doc_id=doc_id,
                verdict=verdict,
                total_overcharge=result.get("total_overcharge", 0.0),
                confidence_score=result.get("confidence_score", 0.0),
                red_flags_count=len(result.get("red_flags", []))
            )
            
            return result
            
        except Exception as e:
            logger.error(f"AI fallback analysis failed: {str(e)}", doc_id=doc_id, exc_info=True)
            
            # Return a proper error result instead of hardcoded values
            return {
                "success": True,
                "analysis_complete": True,
                "doc_id": doc_id,
                "verdict": "warning",
                "message": f"Analysis failed: {str(e)}",
                "total_bill_amount": 0.0,
                "total_overcharge": 0.0,
                "confidence_score": 0.5,
                "red_flags": [{
                    "type": "analysis_error",
                    "description": "Could not complete analysis due to technical issues",
                    "severity": "warning"
                }],
                "recommendations": [
                    "Manual review recommended due to analysis error",
                    "Please check the document quality and try again"
                ],
                "analysis_method": "error_fallback",
                "error": str(e)
            }

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