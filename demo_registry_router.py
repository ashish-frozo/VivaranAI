"""
Demo: Agent Registry & Router System

Demonstrates the complete multi-agent system with:
- Agent registration and discovery
- Intelligent routing strategies
- Multi-agent workflow orchestration
- Real-time monitoring and metrics

This showcases the production-ready Registry & Router layer (PR #2).
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Any

import structlog

# Configure structured logging for demo
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

from agents import (
    BaseAgent,
    AgentContext,
    ModelHint,
    AgentRegistry,
    TaskCapability,
    AgentCapabilities,
    AgentStatus,
    RouterAgent,
    RoutingStrategy,
    WorkflowType,
    RoutingRequest,
    WorkflowDefinition,
    WorkflowStep,
    state_manager
)

logger = structlog.get_logger(__name__)


class DocumentProcessingAgent(BaseAgent):
    """Demo document processing agent for medical bills."""
    
    def __init__(self):
        super().__init__(
            agent_id="medical_doc_processor",
            name="Medical Document Processor",
            instructions="Specialized in processing medical bills and extracting structured data using OCR and NLP",
            tools=[]
        )
    
    async def process_task(self, context: AgentContext, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process medical document and extract line items."""
        
        # Simulate realistic processing time
        await asyncio.sleep(0.5)  # 500ms for OCR and extraction
        
        logger.info(
            "Processing medical document",
            agent_id=self.agent_id,
            doc_id=context.doc_id,
            task_type="document_processing"
        )
        
        # Mock extracted data from medical bill
        return {
            "extraction_success": True,
            "document_type": "medical_bill",
            "hospital_name": "City General Hospital",
            "patient_info": {
                "name": "John Doe",
                "patient_id": "P12345",
                "date_of_service": "2024-01-15"
            },
            "line_items": [
                {
                    "item_code": "CONS001",
                    "description": "Specialist Consultation",
                    "quantity": 1,
                    "unit_price": 800,
                    "total_amount": 800,
                    "category": "consultation"
                },
                {
                    "item_code": "LAB001", 
                    "description": "Complete Blood Count (CBC)",
                    "quantity": 1,
                    "unit_price": 450,
                    "total_amount": 450,
                    "category": "laboratory"
                }
            ],
            "total_amount": 1250,
            "processing_metadata": {
                "ocr_confidence": 0.94,
                "extraction_time_ms": 480
            }
        }


class RateValidationAgent(BaseAgent):
    """Demo rate validation agent for CGHS/ESI compliance."""
    
    def __init__(self):
        super().__init__(
            agent_id="cghs_rate_validator",
            name="CGHS Rate Validator",
            instructions="Validates medical charges against CGHS 2023 rates and identifies overcharges",
            tools=[]
        )
    
    async def process_task(self, context: AgentContext, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate charges against CGHS rates."""
        
        # Simulate rate lookup and validation
        await asyncio.sleep(0.3)  # 300ms for rate validation
        
        logger.info(
            "Validating rates against CGHS 2023",
            agent_id=self.agent_id,
            doc_id=context.doc_id,
            task_type="rate_validation"
        )
        
        # Mock CGHS rate validation results
        return {
            "validation_complete": True,
            "reference_source": "CGHS 2023 Tariff",
            "validation_results": [
                {
                    "item_code": "CONS001",
                    "description": "Specialist Consultation",
                    "billed_amount": 800,
                    "cghs_rate": 500,
                    "overcharge_amount": 300,
                    "overcharge_percentage": 60.0,
                    "violation_type": "rate_excess",
                    "severity": "high"
                }
            ],
            "summary": {
                "total_billed": 800,
                "total_cghs_allowed": 500,
                "total_overcharge": 300,
                "overall_overcharge_percentage": 60.0,
                "violation_count": 1,
                "compliance_status": "non_compliant"
            }
        }


async def register_demo_agents(registry: AgentRegistry) -> Dict[str, BaseAgent]:
    """Register all demo agents with their capabilities."""
    
    logger.info("🔧 Registering demo agents with capabilities...")
    
    # Create agents
    agents = {
        "document_processor": DocumentProcessingAgent(),
        "rate_validator": RateValidationAgent()
    }
    
    # Define agent capabilities
    capabilities = {
        "document_processor": AgentCapabilities(
            supported_tasks=[
                TaskCapability.DOCUMENT_PROCESSING,
                TaskCapability.OCR_EXTRACTION,
                TaskCapability.TEXT_ANALYSIS
            ],
            max_concurrent_requests=3,
            preferred_model_hints=[ModelHint.STANDARD, ModelHint.PREMIUM],
            processing_time_ms_avg=2000,
            cost_per_request_rupees=3.2,
            confidence_threshold=0.85,
            supported_document_types=["pdf", "image", "scan"],
            supported_languages=["en", "hi"]
        ),
        "rate_validator": AgentCapabilities(
            supported_tasks=[
                TaskCapability.RATE_VALIDATION,
                TaskCapability.DATA_VALIDATION,
                TaskCapability.TEXT_ANALYSIS
            ],
            max_concurrent_requests=5,
            preferred_model_hints=[ModelHint.STANDARD],
            processing_time_ms_avg=1500,
            cost_per_request_rupees=2.1,
            confidence_threshold=0.92,
            supported_document_types=["json", "xml"],
            supported_languages=["en"]
        )
    }
    
    # Start and register each agent
    for agent_id, agent in agents.items():
        await agent.start()
        
        success = await registry.register_agent(
            agent=agent,
            capabilities=capabilities[agent_id]
        )
        
        if success:
            logger.info(
                "✅ Agent registered successfully",
                agent_id=agent_id,
                name=agent.name,
                capabilities=len(capabilities[agent_id].supported_tasks)
            )
        else:
            logger.error("❌ Failed to register agent", agent_id=agent_id)
    
    return agents


async def demo_routing_strategies(router: RouterAgent):
    """Demonstrate different routing strategies."""
    
    logger.info("🎯 Demonstrating intelligent routing strategies...")
    
    # Strategy 1: Cost-optimized routing
    cost_request = RoutingRequest(
        doc_id="cost_demo_doc",
        user_id="demo_user",
        task_type="rate_validation",
        required_capabilities=[TaskCapability.RATE_VALIDATION],
        model_hint=ModelHint.STANDARD,
        routing_strategy=RoutingStrategy.COST_OPTIMIZED,
        max_agents=1,
        timeout_seconds=30,
        priority=5,
        metadata={"strategy_demo": "cost_optimized"}
    )
    
    cost_decision = await router.route_request(cost_request)
    
    logger.info(
        "💰 Cost-optimized routing decision",
        selected_agent=cost_decision.selected_agents[0].agent_id if cost_decision.selected_agents else "none",
        estimated_cost=cost_decision.estimated_cost_rupees,
        confidence=cost_decision.confidence,
        reasoning=cost_decision.reasoning
    )


async def demo_sequential_workflow(router: RouterAgent):
    """Demonstrate sequential multi-agent workflow."""
    
    logger.info("🔗 Demonstrating sequential multi-agent workflow...")
    
    # Define medical bill analysis workflow
    workflow = WorkflowDefinition(
        workflow_id="demo_medical_analysis",
        workflow_type=WorkflowType.SEQUENTIAL,
        steps=[
            WorkflowStep(
                step_id="extract_document_data",
                agent_id="medical_doc_processor",
                task_input="Process medical bill PDF: patient_bill_demo.pdf",
                dependencies=[],
                timeout_seconds=30,
                metadata={"document_type": "medical_bill"}
            ),
            WorkflowStep(
                step_id="validate_cghs_rates",
                agent_id="cghs_rate_validator",
                task_input="Validate all extracted charges against CGHS 2023 rates",
                dependencies=["extract_document_data"],
                timeout_seconds=30,
                metadata={"reference_rates": "cghs_2023"}
            )
        ],
        max_parallel_steps=1,
        total_timeout_seconds=120,
        failure_strategy="fail_fast",
        metadata={"workflow_type": "medical_bill_analysis"}
    )
    
    # Execute workflow
    context = AgentContext(
        doc_id="demo_medical_bill_001",
        user_id="demo_patient_123",
        correlation_id=str(uuid.uuid4()),
        model_hint=ModelHint.STANDARD,
        start_time=time.time(),
        metadata={"task_type": "sequential_workflow_demo"}
    )
    
    workflow_start = time.time()
    result = await router.execute_workflow(workflow, context)
    workflow_time = time.time() - workflow_start
    
    logger.info(
        "✅ Sequential workflow completed",
        workflow_id=result.workflow_id,
        success=result.success,
        steps_executed=len(result.step_results),
        total_cost=result.total_cost_rupees,
        execution_time_ms=result.total_execution_time_ms,
        actual_time_seconds=round(workflow_time, 2)
    )
    
    # Display detailed results
    for step_id, step_result in result.step_results.items():
        logger.info(
            f"📋 Step '{step_id}' results",
            agent_id=step_result.agent_id,
            success=step_result.success,
            cost=step_result.cost_rupees,
            time_ms=step_result.execution_time_ms,
            confidence=step_result.confidence
        )


async def main():
    """Main demo function showcasing the complete Registry & Router system."""
    
    logger.info("🚀 Starting MedBillGuard Agent Registry & Router Demo")
    logger.info("=" * 80)
    
    try:
        # Initialize system components
        logger.info("🔧 Initializing system components...")
        
        # Start state manager
        await state_manager.connect()
        
        # Create and start registry
        registry = AgentRegistry()
        await registry.start()
        
        # Create and start router
        router = RouterAgent(registry=registry)
        await router.start()
        
        logger.info("✅ System components initialized successfully")
        logger.info("")
        
        # Demo 1: Agent Registration
        agents = await register_demo_agents(registry)
        logger.info("")
        
        # Demo 2: Routing Strategies
        await demo_routing_strategies(router)
        logger.info("")
        
        # Demo 3: Sequential Workflow
        await demo_sequential_workflow(router)
        logger.info("")
        
        logger.info("🎉 Demo completed successfully!")
        logger.info("✨ Registry & Router system is production-ready")
        
    except Exception as e:
        logger.error("❌ Demo failed", error=str(e), exc_info=True)
        
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up resources...")
        
        # Stop agents
        for agent in agents.values():
            try:
                await agent.stop()
            except Exception as e:
                logger.warning("Failed to stop agent", agent_id=agent.agent_id, error=str(e))
        
        # Stop system components
        try:
            await router.stop()
            await registry.stop()
            await state_manager.disconnect()
        except Exception as e:
            logger.warning("Failed to stop system components", error=str(e))
        
        logger.info("✅ Cleanup completed")
        logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
