"""
Router Agent - Intelligent request routing and multi-agent orchestration.

Provides centralized routing logic, workflow orchestration, and coordination
between specialized agents in the MedBillGuard multi-agent system.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram, Gauge

from .base_agent import BaseAgent
from .interfaces import AgentContext, AgentResult, ModelHint
from .agent_registry import (
    AgentRegistry, 
    TaskCapability, 
    AgentRegistration,
    agent_registry,
    ROUTING_DECISIONS
)
from .redis_state import RedisStateManager, state_manager

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Prometheus metrics
ROUTER_REQUESTS = Counter(
    'router_requests_total',
    'Total router requests',
    ['request_type', 'status']
)

ROUTING_TIME = Histogram(
    'routing_time_seconds',
    'Time spent on routing decisions',
    ['routing_strategy']
)

WORKFLOW_EXECUTION_TIME = Histogram(
    'workflow_execution_seconds',
    'Total workflow execution time',
    ['workflow_type', 'agent_count']
)

ACTIVE_WORKFLOWS = Gauge(
    'active_workflows_total',
    'Number of currently active workflows',
    ['workflow_type']
)


class RoutingStrategy(str, Enum):
    """Routing strategy types."""
    CAPABILITY_BASED = "capability_based"    # Route based on required capabilities
    LOAD_BALANCED = "load_balanced"          # Balance load across agents
    COST_OPTIMIZED = "cost_optimized"        # Minimize cost
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Minimize latency
    RELIABILITY_OPTIMIZED = "reliability_optimized"  # Maximize success rate


class WorkflowType(str, Enum):
    """Workflow execution types."""
    SEQUENTIAL = "sequential"      # Execute agents one by one
    PARALLEL = "parallel"          # Execute agents concurrently
    CONDITIONAL = "conditional"    # Execute based on conditions
    PIPELINE = "pipeline"          # Pass results between agents


@dataclass
class RoutingRequest:
    """Request for agent routing."""
    doc_id: str
    user_id: str
    task_type: str
    required_capabilities: List[TaskCapability]
    model_hint: ModelHint
    routing_strategy: RoutingStrategy
    max_agents: int
    timeout_seconds: int
    priority: int  # 1-10, higher = more priority
    metadata: Dict[str, Any]


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    selected_agents: List[AgentRegistration]
    routing_strategy: RoutingStrategy
    estimated_cost_rupees: float
    estimated_time_ms: int
    confidence: float
    reasoning: str
    fallback_agents: List[AgentRegistration]


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    step_id: str
    agent_id: str
    task_input: str
    dependencies: List[str]  # Step IDs this step depends on
    timeout_seconds: int
    metadata: Dict[str, Any]


@dataclass
class WorkflowDefinition:
    """Multi-agent workflow definition."""
    workflow_id: str
    workflow_type: WorkflowType
    steps: List[WorkflowStep]
    max_parallel_steps: int
    total_timeout_seconds: int
    failure_strategy: str  # "fail_fast", "continue", "retry"
    metadata: Dict[str, Any]


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_id: str
    success: bool
    step_results: Dict[str, AgentResult]
    total_execution_time_ms: int
    total_cost_rupees: float
    error: Optional[str]
    metadata: Dict[str, Any]


class RouterAgent(BaseAgent):
    """
    Intelligent router agent for multi-agent coordination.
    
    Features:
    - Capability-based agent discovery and selection
    - Multiple routing strategies (cost, performance, reliability)
    - Multi-agent workflow orchestration (sequential, parallel, pipeline)
    - Load balancing and fallback handling
    - Real-time agent health monitoring
    - Cost and performance optimization
    """
    
    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        state_manager: Optional[RedisStateManager] = None,
        redis_url: str = "redis://localhost:6379/1"
    ):
        super().__init__(
            agent_id="router_agent",
            name="Router Agent",
            instructions="Intelligent routing and orchestration for multi-agent workflows",
            tools=[],
            redis_url=redis_url
        )
        
        self.registry = registry or agent_registry
        from .redis_state import state_manager as default_state_manager
        self.state_manager = state_manager or default_state_manager
        
        # Routing configuration
        self.DEFAULT_TIMEOUT = 30  # seconds
        self.MAX_ROUTING_ATTEMPTS = 3
        self.FALLBACK_THRESHOLD = 0.7  # Use fallback if confidence < 70%
        
        # Workflow execution tracking
        self._active_workflows: Dict[str, WorkflowDefinition] = {}
        
    async def start(self):
        """Start the router agent."""
        await super().start()
        
        # Ensure registry is started
        if not self.registry._running:
            await self.registry.start()
        
        logger.info("Router agent started successfully")
    
    async def stop(self):
        """Stop the router agent."""
        # Cancel any active workflows
        for workflow_id in list(self._active_workflows.keys()):
            logger.info("Cancelling active workflow", workflow_id=workflow_id)
            del self._active_workflows[workflow_id]
        
        await super().stop()
        logger.info("Router agent stopped")
    
    async def process_task(self, context: AgentContext, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process routing task - required by BaseAgent.
        
        This method handles routing requests and workflow executions.
        """
        task_type = task_data.get("task_type", "route_request")
        
        if task_type == "route_request":
            routing_request = RoutingRequest(**task_data["routing_request"])
            decision = await self.route_request(routing_request)
            return {
                "decision": decision,
                "selected_agents": [reg.agent_id for reg in decision.selected_agents],
                "estimated_cost": decision.estimated_cost_rupees,
                "confidence": decision.confidence
            }
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    async def route_request(self, routing_request: RoutingRequest) -> RoutingDecision:
        """
        Make intelligent routing decision for a request.
        
        Args:
            routing_request: Request with routing requirements
            
        Returns:
            RoutingDecision with selected agents and metadata
        """
        with tracer.start_as_current_span("router_agent.route_request") as span:
            span.set_attribute("doc_id", routing_request.doc_id)
            span.set_attribute("task_type", routing_request.task_type)
            span.set_attribute("routing_strategy", routing_request.routing_strategy.value)
            
            routing_start = time.time()
            
            try:
                # Discover candidate agents
                candidates = await self.registry.discover_agents(
                    required_capabilities=routing_request.required_capabilities,
                    model_hint=routing_request.model_hint,
                    max_agents=routing_request.max_agents * 2  # Get more for better selection
                )
                
                if not candidates:
                    raise RuntimeError(
                        f"No agents found for capabilities: {routing_request.required_capabilities}"
                    )
                
                # Apply routing strategy
                selected_agents, fallback_agents = await self._apply_routing_strategy(
                    candidates=candidates,
                    routing_request=routing_request
                )
                
                # Calculate estimates
                estimated_cost = sum(
                    agent.capabilities.cost_per_request_rupees for agent in selected_agents
                )
                estimated_time = max(
                    agent.capabilities.processing_time_ms_avg for agent in selected_agents
                ) if selected_agents else 0
                
                # Calculate confidence based on agent reliability
                confidence = self._calculate_routing_confidence(selected_agents)
                
                # Generate reasoning
                reasoning = self._generate_routing_reasoning(
                    selected_agents, routing_request.routing_strategy
                )
                
                decision = RoutingDecision(
                    selected_agents=selected_agents,
                    routing_strategy=routing_request.routing_strategy,
                    estimated_cost_rupees=estimated_cost,
                    estimated_time_ms=estimated_time,
                    confidence=confidence,
                    reasoning=reasoning,
                    fallback_agents=fallback_agents
                )
                
                # Update metrics
                routing_time = time.time() - routing_start
                ROUTING_TIME.labels(
                    routing_strategy=routing_request.routing_strategy.value
                ).observe(routing_time)
                
                ROUTING_DECISIONS.labels(
                    routing_strategy=routing_request.routing_strategy.value,
                    agent_selected=selected_agents[0].agent_id if selected_agents else "none"
                ).inc()
                
                ROUTER_REQUESTS.labels(
                    request_type=routing_request.task_type,
                    status="success"
                ).inc()
                
                logger.info(
                    "Routing decision completed",
                    doc_id=routing_request.doc_id,
                    selected_agents=[a.agent_id for a in selected_agents],
                    estimated_cost=estimated_cost,
                    estimated_time=estimated_time,
                    confidence=confidence,
                    routing_time_ms=routing_time * 1000
                )
                
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("agents_selected", len(selected_agents))
                span.set_attribute("estimated_cost", estimated_cost)
                span.set_attribute("confidence", confidence)
                
                return decision
                
            except Exception as e:
                ROUTER_REQUESTS.labels(
                    request_type=routing_request.task_type,
                    status="error"
                ).inc()
                
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                logger.error(
                    "Routing decision failed",
                    doc_id=routing_request.doc_id,
                    error=str(e),
                    exc_info=True
                )
                raise
    
    async def _apply_routing_strategy(
        self,
        candidates: List[AgentRegistration],
        routing_request: RoutingRequest
    ) -> Tuple[List[AgentRegistration], List[AgentRegistration]]:
        """Apply routing strategy to select best agents."""
        
        if routing_request.routing_strategy == RoutingStrategy.COST_OPTIMIZED:
            # Sort by cost (ascending)
            candidates.sort(key=lambda a: a.capabilities.cost_per_request_rupees)
        
        elif routing_request.routing_strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
            # Sort by response time (ascending)
            candidates.sort(key=lambda a: a.avg_response_time_ms)
        
        elif routing_request.routing_strategy == RoutingStrategy.RELIABILITY_OPTIMIZED:
            # Sort by success rate (descending)
            candidates.sort(
                key=lambda a: (
                    (a.total_requests - a.total_errors) / max(a.total_requests, 1)
                ),
                reverse=True
            )
        
        elif routing_request.routing_strategy == RoutingStrategy.LOAD_BALANCED:
            # Sort by current load (ascending)
            candidates.sort(
                key=lambda a: a.current_requests / max(a.capabilities.max_concurrent_requests, 1)
            )
        
        # Default: capability-based (already sorted by suitability score)
        
        # Select primary agents
        selected = candidates[:routing_request.max_agents]
        
        # Select fallback agents (next best options)
        fallback = candidates[routing_request.max_agents:routing_request.max_agents * 2]
        
        return selected, fallback
    
    def _calculate_routing_confidence(self, agents: List[AgentRegistration]) -> float:
        """Calculate confidence score for routing decision."""
        if not agents:
            return 0.0
        
        total_confidence = 0.0
        for agent in agents:
            # Base confidence from agent success rate
            if agent.total_requests > 0:
                success_rate = (agent.total_requests - agent.total_errors) / agent.total_requests
            else:
                success_rate = 1.0  # New agents get benefit of doubt
            
            # Adjust for load
            load_factor = 1.0 - (
                agent.current_requests / max(agent.capabilities.max_concurrent_requests, 1)
            )
            
            # Combine factors
            agent_confidence = success_rate * 0.7 + load_factor * 0.3
            total_confidence += agent_confidence
        
        return min(1.0, total_confidence / len(agents))
    
    def _generate_routing_reasoning(
        self,
        agents: List[AgentRegistration],
        strategy: RoutingStrategy
    ) -> str:
        """Generate human-readable reasoning for routing decision."""
        if not agents:
            return "No suitable agents found"
        
        agent_names = [agent.name for agent in agents]
        
        reasoning_map = {
            RoutingStrategy.COST_OPTIMIZED: f"Selected {agent_names} for lowest cost per request",
            RoutingStrategy.PERFORMANCE_OPTIMIZED: f"Selected {agent_names} for fastest response time",
            RoutingStrategy.RELIABILITY_OPTIMIZED: f"Selected {agent_names} for highest success rate",
            RoutingStrategy.LOAD_BALANCED: f"Selected {agent_names} with lowest current load",
            RoutingStrategy.CAPABILITY_BASED: f"Selected {agent_names} with best capability match"
        }
        
        return reasoning_map.get(strategy, f"Selected {agent_names} using default strategy")


# Global router instance
router_agent = RouterAgent()
