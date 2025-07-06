"""
Enhanced Router Agent - Intelligent document-aware routing with OCR integration.

This enhanced router works with the new architecture:
1. Generic OCR Tool extracts raw data
2. Document Type Classifier determines document type and requirements
3. Enhanced Router Agent makes intelligent routing decisions
4. Specialized Domain Agents handle domain-specific analysis

Features:
- Document-type-aware routing
- OCR quality assessment for routing decisions
- Multi-stage processing workflows
- Fallback and error handling
- Performance optimization based on document characteristics
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
from opentelemetry import trace

from .generic_ocr_tool import GenericOCRTool
from .document_type_classifier import DocumentTypeClassifier, DocumentType, RequiredCapability
from ..base_agent import BaseAgent, AgentContext, AgentResult, ModelHint
from ..agent_registry import AgentRegistry, TaskCapability, AgentRegistration
from ..router_agent import RouterAgent, RoutingStrategy, RoutingDecision, RoutingRequest

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class ProcessingStage(str, Enum):
    """Document processing stages."""
    OCR_EXTRACTION = "ocr_extraction"
    DOCUMENT_CLASSIFICATION = "document_classification"
    ROUTING_DECISION = "routing_decision"
    DOMAIN_ANALYSIS = "domain_analysis"
    RESULT_COMPILATION = "result_compilation"


@dataclass
class DocumentProcessingRequest:
    """Request for complete document processing."""
    file_content: bytes
    doc_id: str
    user_id: str
    language: str = "english"
    file_format: Optional[str] = None
    processing_priority: str = "normal"
    routing_strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_BASED
    metadata: Dict[str, Any] = None


@dataclass
class DocumentProcessingResult:
    """Complete document processing result."""
    success: bool
    doc_id: str
    processing_stages: Dict[ProcessingStage, Dict[str, Any]]
    document_type: str
    final_result: Dict[str, Any]
    total_processing_time_ms: int
    error: Optional[str] = None


class EnhancedRouterAgent(BaseAgent):
    """
    Enhanced router agent with integrated OCR and document classification.
    
    Provides end-to-end document processing workflow:
    1. OCR extraction using GenericOCRTool
    2. Document classification using DocumentTypeClassifier
    3. Intelligent routing based on document type and requirements
    4. Specialized agent execution
    5. Result compilation and optimization
    """
    
    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        redis_url: str = "redis://localhost:6379/1",
        openai_api_key: Optional[str] = None
    ):
        """Initialize the enhanced router agent."""
        super().__init__(
            agent_id="enhanced_router_agent",
            name="Enhanced Router Agent",
            instructions="Intelligent document processing with OCR, classification, and routing",
            tools=[],
            redis_url=redis_url
        )
        
        # Initialize processing tools
        self.ocr_tool = GenericOCRTool()
        self.document_classifier = DocumentTypeClassifier(openai_api_key=openai_api_key)
        self.base_router = RouterAgent(registry=registry, redis_url=redis_url)
        
        # Capability mapping for routing
        self.capability_mapping = {
            RequiredCapability.MEDICAL_ANALYSIS: TaskCapability.MEDICAL_ANALYSIS,
            RequiredCapability.FINANCIAL_ANALYSIS: TaskCapability.FINANCIAL_ANALYSIS,
            RequiredCapability.LEGAL_ANALYSIS: TaskCapability.LEGAL_ANALYSIS,
            RequiredCapability.RATE_VALIDATION: TaskCapability.RATE_VALIDATION,
            RequiredCapability.DUPLICATE_DETECTION: TaskCapability.DUPLICATE_DETECTION,
            RequiredCapability.COMPLIANCE_CHECK: TaskCapability.COMPLIANCE_CHECK,
            RequiredCapability.DATA_EXTRACTION: TaskCapability.DATA_EXTRACTION,
            RequiredCapability.DOCUMENT_VERIFICATION: TaskCapability.DOCUMENT_VERIFICATION
        }
        
        logger.info("Initialized EnhancedRouterAgent with OCR and classification capabilities")
    
    async def process_task(self, context: AgentContext, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complete document analysis task.
        
        Args:
            context: Agent execution context
            task_data: Task data containing document and processing parameters
            
        Returns:
            Complete document processing results
        """
        with tracer.start_as_current_span("enhanced_router.process_task") as span:
            span.set_attribute("doc_id", context.doc_id)
            span.set_attribute("user_id", context.user_id)
            
            try:
                # Create processing request
                request = DocumentProcessingRequest(
                    file_content=task_data["file_content"],
                    doc_id=context.doc_id,
                    user_id=context.user_id,
                    language=task_data.get("language", "english"),
                    file_format=task_data.get("file_format"),
                    processing_priority=task_data.get("priority", "normal"),
                    routing_strategy=RoutingStrategy(task_data.get("routing_strategy", "capability_based")),
                    metadata=task_data.get("metadata", {})
                )
                
                # Execute complete processing workflow
                result = await self.process_document_complete(request)
                
                return {
                    "success": result.success,
                    "doc_id": result.doc_id,
                    "document_type": result.document_type,
                    "processing_stages": result.processing_stages,
                    "final_result": result.final_result,
                    "total_processing_time_ms": result.total_processing_time_ms,
                    "error": result.error
                }
                
            except Exception as e:
                error_msg = f"Enhanced router processing failed: {str(e)}"
                logger.error(error_msg, doc_id=context.doc_id, exc_info=True)
                
                return {
                    "success": False,
                    "doc_id": context.doc_id,
                    "error": error_msg,
                    "processing_stages": {},
                    "final_result": {},
                    "total_processing_time_ms": 0
                }
    
    async def process_document_complete(
        self, 
        request: DocumentProcessingRequest
    ) -> DocumentProcessingResult:
        """
        Execute complete document processing workflow.
        
        Args:
            request: Document processing request
            
        Returns:
            Complete processing result with all stages
        """
        start_time = time.time()
        processing_stages = {}
        
        try:
            logger.info(
                "Starting complete document processing",
                doc_id=request.doc_id,
                user_id=request.user_id,
                language=request.language,
                file_size=len(request.file_content)
            )
            
            # Stage 1: OCR Extraction  
            logger.info("Stage 1: OCR extraction", doc_id=request.doc_id)
            ocr_start = time.time()
            
            ocr_result = await self.ocr_tool(
                file_content=request.file_content,
                doc_id=request.doc_id,
                language=request.language,
                file_format=request.file_format
            )
            
            ocr_time = (time.time() - ocr_start) * 1000
            processing_stages[ProcessingStage.OCR_EXTRACTION] = {
                "result": ocr_result,
                "processing_time_ms": ocr_time,
                "success": ocr_result.get("success", False)
            }
            
            if not ocr_result.get("success"):
                raise Exception(f"OCR extraction failed: {ocr_result.get('error', 'Unknown error')}")
            
            # Stage 2: Document Classification
            logger.info("Stage 2: Document classification", doc_id=request.doc_id)
            classification_start = time.time()
            
            classification_result = await self.document_classifier(
                raw_text=ocr_result["raw_text"],
                doc_id=request.doc_id,
                pages=ocr_result.get("pages", []),
                tables=ocr_result.get("tables", []),
                metadata=ocr_result.get("metadata", {})
            )
            
            classification_time = (time.time() - classification_start) * 1000
            processing_stages[ProcessingStage.DOCUMENT_CLASSIFICATION] = {
                "result": classification_result,
                "processing_time_ms": classification_time,
                "success": classification_result.get("success", False)
            }
            
            if not classification_result.get("success"):
                raise Exception(f"Document classification failed: {classification_result.get('error', 'Unknown error')}")
            
            # Stage 3: Routing Decision
            logger.info("Stage 3: Routing decision", doc_id=request.doc_id)
            routing_start = time.time()
            
            routing_result = await self._make_routing_decision(
                classification_result=classification_result,
                processing_request=request,
                ocr_quality=ocr_result.get("processing_stats", {}).get("ocr_confidence", 0.0)
            )
            
            routing_time = (time.time() - routing_start) * 1000
            processing_stages[ProcessingStage.ROUTING_DECISION] = {
                "result": routing_result,
                "processing_time_ms": routing_time,
                "success": routing_result.get("success", False)
            }
            
            if not routing_result.get("success"):
                raise Exception(f"Routing decision failed: {routing_result.get('error', 'Unknown error')}")
            
            # Stage 4: Domain Analysis
            logger.info("Stage 4: Domain analysis", doc_id=request.doc_id)
            analysis_start = time.time()
            
            domain_result = await self._execute_domain_analysis(
                routing_decision=routing_result,
                ocr_data=ocr_result,
                classification_data=classification_result,
                processing_request=request
            )
            
            analysis_time = (time.time() - analysis_start) * 1000
            processing_stages[ProcessingStage.DOMAIN_ANALYSIS] = {
                "result": domain_result,
                "processing_time_ms": analysis_time,
                "success": domain_result.get("success", False)
            }
            
            if not domain_result.get("success"):
                raise Exception(f"Domain analysis failed: {domain_result.get('error', 'Unknown error')}")
            
            # Stage 5: Result Compilation
            logger.info("Stage 5: Result compilation", doc_id=request.doc_id)
            compilation_start = time.time()
            
            final_result = self._compile_final_result(
                ocr_result=ocr_result,
                classification_result=classification_result,
                routing_result=routing_result,
                domain_result=domain_result,
                processing_stages=processing_stages
            )
            
            compilation_time = (time.time() - compilation_start) * 1000
            processing_stages[ProcessingStage.RESULT_COMPILATION] = {
                "result": final_result,
                "processing_time_ms": compilation_time,
                "success": True
            }
            
            # Calculate total time
            total_time = (time.time() - start_time) * 1000
            
            logger.info(
                "Complete document processing finished",
                doc_id=request.doc_id,
                document_type=classification_result.get("document_type"),
                total_time_ms=total_time,
                success=True
            )
            
            return DocumentProcessingResult(
                success=True,
                doc_id=request.doc_id,
                processing_stages=processing_stages,
                document_type=classification_result.get("document_type", "unknown"),
                final_result=final_result,
                total_processing_time_ms=int(total_time)
            )
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            error_msg = f"Document processing failed: {str(e)}"
            
            logger.error(
                error_msg,
                doc_id=request.doc_id,
                total_time_ms=total_time,
                processing_stages=list(processing_stages.keys()),
                exc_info=True
            )
            
            return DocumentProcessingResult(
                success=False,
                doc_id=request.doc_id,
                processing_stages=processing_stages,
                document_type="unknown",
                final_result={},
                total_processing_time_ms=int(total_time),
                error=error_msg
            )
    
    async def _make_routing_decision(
        self,
        classification_result: Dict[str, Any],
        processing_request: DocumentProcessingRequest,
        ocr_quality: float
    ) -> Dict[str, Any]:
        """Make intelligent routing decision based on classification and quality."""
        try:
            # Convert required capabilities to task capabilities
            required_capabilities = []
            for cap_str in classification_result.get("required_capabilities", []):
                if cap_str in self.capability_mapping:
                    required_capabilities.append(self.capability_mapping[cap_str])
            
            # Adjust routing strategy based on OCR quality
            routing_strategy = processing_request.routing_strategy
            if ocr_quality < 0.7:
                # Use more reliable agents for low quality OCR
                routing_strategy = RoutingStrategy.RELIABILITY_OPTIMIZED
            elif processing_request.processing_priority == "high":
                routing_strategy = RoutingStrategy.PERFORMANCE_OPTIMIZED
            
            # Create routing request
            routing_request = RoutingRequest(
                doc_id=processing_request.doc_id,
                user_id=processing_request.user_id,
                task_type=f"{classification_result.get('document_type', 'unknown')}_analysis",
                required_capabilities=required_capabilities,
                model_hint=ModelHint.STANDARD,
                routing_strategy=routing_strategy,
                max_agents=1,
                timeout_seconds=60,
                priority=5 if processing_request.processing_priority == "high" else 3,
                metadata={
                    "document_type": classification_result.get("document_type"),
                    "ocr_quality": ocr_quality,
                    "classification_confidence": classification_result.get("confidence", 0.0)
                }
            )
            
            # Make routing decision
            decision = await self.base_router.route_request(routing_request)
            
            return {
                "success": True,
                "routing_decision": decision,
                "selected_agent": decision.selected_agents[0].agent_id if decision.selected_agents else None,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Routing decision failed: {str(e)}",
                "selected_agent": classification_result.get("suggested_agent", "generic_document_agent"),
                "confidence": 0.0,
                "reasoning": "Fallback to suggested agent due to routing error"
            }
    
    async def _execute_domain_analysis(
        self,
        routing_decision: Dict[str, Any],
        ocr_data: Dict[str, Any],
        classification_data: Dict[str, Any],
        processing_request: DocumentProcessingRequest
    ) -> Dict[str, Any]:
        """Execute domain-specific analysis using the routed agent."""
        try:
            selected_agent_id = routing_decision.get("selected_agent")
            
            if not selected_agent_id:
                raise Exception("No agent selected for domain analysis")
            
            # Get the selected agent from registry
            agent_registration = None
            for agent_reg in routing_decision.get("routing_decision", {}).get("selected_agents", []):
                if agent_reg.agent_id == selected_agent_id:
                    agent_registration = agent_reg
                    break
            
            if not agent_registration:
                raise Exception(f"Agent {selected_agent_id} not found in registry")
            
            # Prepare task data for domain agent
            domain_task_data = {
                "file_content": processing_request.file_content,
                "doc_id": processing_request.doc_id,
                "language": processing_request.language,
                "file_format": processing_request.file_format,
                "raw_text": ocr_data.get("raw_text", ""),
                "pages": ocr_data.get("pages", []),
                "tables": ocr_data.get("tables", []),
                "document_type": classification_data.get("document_type"),
                "classification_confidence": classification_data.get("confidence", 0.0),
                "ocr_stats": ocr_data.get("processing_stats", {}),
                "metadata": {
                    **processing_request.metadata,
                    **classification_data.get("metadata", {}),
                    "routing_confidence": routing_decision.get("confidence", 0.0)
                }
            }
            
            # Create agent context
            context = AgentContext(
                doc_id=processing_request.doc_id,
                user_id=processing_request.user_id,
                correlation_id=f"enhanced_router_{processing_request.doc_id}",
                model_hint=ModelHint.STANDARD,
                start_time=time.time()
            )
            
            # Execute domain analysis
            domain_result = await agent_registration.agent_instance.process_task(
                context=context,
                task_data=domain_task_data
            )
            
            return {
                "success": True,
                "agent_used": selected_agent_id,
                "domain_result": domain_result,
                "processing_metadata": {
                    "routing_confidence": routing_decision.get("confidence", 0.0),
                    "classification_confidence": classification_data.get("confidence", 0.0),
                    "ocr_confidence": ocr_data.get("processing_stats", {}).get("ocr_confidence", 0.0)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Domain analysis execution failed: {str(e)}",
                "agent_used": routing_decision.get("selected_agent", "unknown"),
                "domain_result": {},
                "processing_metadata": {}
            }
    
    def _compile_final_result(
        self,
        ocr_result: Dict[str, Any],
        classification_result: Dict[str, Any],
        routing_result: Dict[str, Any],
        domain_result: Dict[str, Any],
        processing_stages: Dict[ProcessingStage, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compile final comprehensive result."""
        
        # Calculate total processing time
        total_time = sum(
            stage_data.get("processing_time_ms", 0) 
            for stage_data in processing_stages.values()
        )
        
        # Extract key metrics
        ocr_stats = ocr_result.get("processing_stats", {})
        domain_analysis = domain_result.get("domain_result", {})
        
        # Compile comprehensive result
        return {
            "analysis_complete": True,
            "document_type": classification_result.get("document_type", "unknown"),
            "classification_confidence": classification_result.get("confidence", 0.0),
            "ocr_confidence": ocr_stats.get("ocr_confidence", 0.0),
            "routing_confidence": routing_result.get("confidence", 0.0),
            
            # OCR data
            "raw_text": ocr_result.get("raw_text", ""),
            "pages_processed": ocr_stats.get("pages_processed", 0),
            "tables_extracted": len(ocr_result.get("tables", [])),
            "text_length": len(ocr_result.get("raw_text", "")),
            
            # Classification data
            "required_capabilities": classification_result.get("required_capabilities", []),
            "suggested_agent": classification_result.get("suggested_agent", "unknown"),
            "classification_reasoning": classification_result.get("reasoning", ""),
            
            # Domain analysis results
            "domain_analysis": domain_analysis,
            "agent_used": domain_result.get("agent_used", "unknown"),
            
            # Processing metadata
            "total_processing_time_ms": total_time,
            "processing_stages_count": len(processing_stages),
            "processing_quality": {
                "ocr_quality": "high" if ocr_stats.get("ocr_confidence", 0) > 0.8 else "medium" if ocr_stats.get("ocr_confidence", 0) > 0.6 else "low",
                "classification_quality": "high" if classification_result.get("confidence", 0) > 0.8 else "medium" if classification_result.get("confidence", 0) > 0.6 else "low",
                "routing_quality": "high" if routing_result.get("confidence", 0) > 0.8 else "medium" if routing_result.get("confidence", 0) > 0.6 else "low"
            },
            
            # Final recommendations
            "recommendations": self._generate_recommendations(
                classification_result, domain_analysis, ocr_stats
            ),
            
            # Processing pipeline metadata
            "pipeline_metadata": {
                "architecture": "generic_ocr_classifier_router",
                "stages_completed": list(processing_stages.keys()),
                "total_time_ms": total_time,
                "agent_routing_successful": routing_result.get("success", False),
                "domain_analysis_successful": domain_result.get("success", False)
            }
        }
    
    def _generate_recommendations(
        self,
        classification_result: Dict[str, Any],
        domain_analysis: Dict[str, Any],
        ocr_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate processing recommendations based on results."""
        recommendations = []
        
        # OCR quality recommendations
        ocr_confidence = ocr_stats.get("ocr_confidence", 0.0)
        if ocr_confidence < 0.6:
            recommendations.append("Consider rescanning document with higher quality for better OCR results")
        elif ocr_confidence < 0.8:
            recommendations.append("OCR quality is moderate - verify extracted text for accuracy")
        
        # Classification recommendations
        classification_confidence = classification_result.get("confidence", 0.0)
        if classification_confidence < 0.7:
            recommendations.append("Document type classification has low confidence - manual review recommended")
        
        # Domain-specific recommendations
        if domain_analysis.get("recommendations"):
            recommendations.extend(domain_analysis.get("recommendations", []))
        
        # Processing efficiency recommendations
        if ocr_stats.get("processing_time_ms", 0) > 10000:  # > 10 seconds
            recommendations.append("Consider using optimized OCR settings for faster processing")
        
        return recommendations if recommendations else ["Document processed successfully with good quality"] 