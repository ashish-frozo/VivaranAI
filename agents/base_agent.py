"""
Base Agent Foundation - Core primitives for all MedBillGuard agents.

This module provides the foundational BaseAgent class that all specialized agents inherit from.
Includes OpenAI SDK integration, OTEL tracing, Redis state management, and cost tracking.
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram, Gauge
import redis.asyncio as redis

# OpenAI Agents SDK imports (temporarily disabled for testing)
# from agents import Agent as OpenAIAgent, Runner
# from agents.tools import function_tool

# Mock imports for testing
class OpenAIAgent:
    def __init__(self, **kwargs):
        pass

class Runner:
    @staticmethod
    def run_sync(agent, task_input):
        class MockResult:
            final_output = "Mock response"
            usage = type('obj', (object,), {'prompt_tokens': 100, 'completion_tokens': 50})()
        return MockResult()

def function_tool(func):
    return func

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Prometheus metrics
AGENT_EXECUTION_TIME = Histogram(
    'agent_execution_seconds',
    'Time spent executing agent tasks',
    ['agent_type', 'task_type', 'model']
)

OPENAI_COST_TOTAL = Counter(
    'openai_rupees_total',
    'Total OpenAI API costs in rupees',
    ['agent_type', 'model', 'request_type']
)

AGENT_REQUESTS_TOTAL = Counter(
    'agent_requests_total',
    'Total agent requests',
    ['agent_type', 'status']
)

ACTIVE_AGENTS = Gauge(
    'active_agents_total',
    'Number of currently active agents',
    ['agent_type']
)

# OpenAI API cost mapping (rupees per 1K tokens)
OPENAI_COSTS = {
    "gpt-4o": {"input": 0.42, "output": 1.26},      # $0.005/$0.015 * 84 INR/USD
    "gpt-3.5-turbo": {"input": 0.08, "output": 0.17}, # $0.001/$0.002 * 84 INR/USD
    "gpt-4": {"input": 2.52, "output": 5.04}        # $0.03/$0.06 * 84 INR/USD
}


class ModelHint(str, Enum):
    """Model complexity hints from RouterAgent."""
    CHEAP = "cheap"        # GPT-3.5 or local Mistral
    STANDARD = "standard"  # GPT-4o for complex analysis
    PREMIUM = "premium"    # GPT-4 for highest accuracy


@dataclass
class AgentResult:
    """Standardized result format for agent responses."""
    success: bool
    data: Dict[str, Any]
    agent_id: str
    task_type: str
    execution_time_ms: int
    model_used: str
    cost_rupees: float
    confidence: float = 1.0
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentContext:
    """Context shared across agent execution."""
    doc_id: str
    user_id: str
    correlation_id: str
    model_hint: ModelHint
    start_time: float
    metadata: Dict[str, Any]


class CPUTimeoutError(Exception):
    """Raised when agent exceeds CPU time slice."""
    pass


class BaseAgent(ABC):
    """
    Base class for all MedBillGuard agents.
    
    Provides:
    - OpenAI SDK integration with cost tracking
    - OTEL tracing and span management
    - Redis state management
    - CPU time slice enforcement (150ms)
    - Standardized error handling
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        instructions: str,
        tools: Optional[List[Callable]] = None,
        redis_url: str = "redis://localhost:6379/1",
        default_model: str = "gpt-4o"
    ):
        self.agent_id = agent_id
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.redis_url = redis_url
        self.default_model = default_model
        
        # Redis client (will be initialized in start())
        self.redis_client: Optional[redis.Redis] = None
        
        # OpenAI Agent (will be created per request)
        self._openai_agent = None
        
        logger.info(
            "Initialized BaseAgent",
            agent_id=agent_id,
            name=name,
            tools_count=len(self.tools)
        )
    
    async def start(self):
        """Initialize Redis connection and prepare agent for requests."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            logger.info("Agent started successfully", agent_id=self.agent_id)
            
        except Exception as e:
            logger.error(
                "Failed to start agent",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def stop(self):
        """Cleanup Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Agent stopped", agent_id=self.agent_id)
    
    def select_model(self, model_hint: ModelHint) -> str:
        """
        Select appropriate model based on hint and agent's complexity detector.
        Agents can downgrade but not upgrade without explicit flag.
        """
        if model_hint == ModelHint.CHEAP:
            return "gpt-3.5-turbo"
        elif model_hint == ModelHint.STANDARD:
            return "gpt-4o"
        elif model_hint == ModelHint.PREMIUM:
            return "gpt-4"
        else:
            return self.default_model
    
    def _create_openai_agent(self, model: str) -> OpenAIAgent:
        """Create OpenAI Agent instance with tools and configuration."""
        # Convert function tools to OpenAI SDK format
        openai_tools = []
        for tool_func in self.tools:
            if hasattr(tool_func, '_tool_schema'):
                openai_tools.append(function_tool(tool_func))
            else:
                # Wrap function as tool if not already wrapped
                openai_tools.append(function_tool(tool_func))
        
        return OpenAIAgent(
            name=self.name,
            instructions=self.instructions,
            tools=openai_tools,
            model=model
        )
    
    async def execute(self, context: AgentContext, task_input: str) -> AgentResult:
        """
        Execute agent task with full observability and error handling.
        
        Args:
            context: Agent execution context
            task_input: Task description/query for the agent
            
        Returns:
            AgentResult with execution details and metrics
        """
        with tracer.start_as_current_span(
            f"{self.agent_id}.execute",
            attributes={
                "agent.id": self.agent_id,
                "agent.name": self.name,
                "doc.id": context.doc_id,
                "user.id": context.user_id,
                "correlation.id": context.correlation_id,
                "model.hint": context.model_hint.value
            }
        ) as span:
            
            # Increment active agents gauge
            ACTIVE_AGENTS.labels(agent_type=self.agent_id).inc()
            
            try:
                # Select model based on hint
                selected_model = self.select_model(context.model_hint)
                span.set_attribute("model.selected", selected_model)
                
                # Create OpenAI agent for this request
                openai_agent = self._create_openai_agent(selected_model)
                
                # Enforce CPU time slice (150ms)
                start_cpu_time = time.process_time()
                
                # Execute agent task
                logger.info(
                    "Starting agent execution",
                    agent_id=self.agent_id,
                    doc_id=context.doc_id,
                    model=selected_model,
                    task_input_length=len(task_input)
                )
                
                # Run agent with timeout protection
                result = await asyncio.wait_for(
                    self._execute_with_timeout(openai_agent, task_input, start_cpu_time),
                    timeout=30.0  # 30 second overall timeout
                )
                
                # Calculate execution time and cost
                execution_time_ms = int((time.time() - context.start_time) * 1000)
                cost_rupees = self._calculate_cost(result, selected_model)
                
                # Update metrics
                AGENT_EXECUTION_TIME.labels(
                    agent_type=self.agent_id,
                    task_type=context.metadata.get("task_type", "unknown"),
                    model=selected_model
                ).observe(execution_time_ms / 1000)
                
                OPENAI_COST_TOTAL.labels(
                    agent_type=self.agent_id,
                    model=selected_model,
                    request_type="completion"
                ).inc(cost_rupees)
                
                AGENT_REQUESTS_TOTAL.labels(
                    agent_type=self.agent_id,
                    status="success"
                ).inc()
                
                # Parse agent result
                agent_data = await self._parse_agent_result(result)
                
                # Create successful result
                agent_result = AgentResult(
                    success=True,
                    data=agent_data,
                    agent_id=self.agent_id,
                    task_type=context.metadata.get("task_type", "analysis"),
                    execution_time_ms=execution_time_ms,
                    model_used=selected_model,
                    cost_rupees=cost_rupees,
                    confidence=agent_data.get("confidence", 1.0),
                    metadata={
                        "correlation_id": context.correlation_id,
                        "span_id": span.get_span_context().span_id
                    }
                )
                
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("execution.time_ms", execution_time_ms)
                span.set_attribute("cost.rupees", cost_rupees)
                
                logger.info(
                    "Agent execution completed successfully",
                    agent_id=self.agent_id,
                    doc_id=context.doc_id,
                    execution_time_ms=execution_time_ms,
                    cost_rupees=cost_rupees,
                    confidence=agent_result.confidence
                )
                
                return agent_result
                
            except CPUTimeoutError as e:
                return self._handle_timeout_error(e, context, span)
            except asyncio.TimeoutError as e:
                return self._handle_timeout_error(e, context, span)
            except Exception as e:
                return self._handle_general_error(e, context, span)
            finally:
                # Decrement active agents gauge
                ACTIVE_AGENTS.labels(agent_type=self.agent_id).dec()
    
    async def _execute_with_timeout(
        self, 
        openai_agent: OpenAIAgent, 
        task_input: str, 
        start_cpu_time: float
    ):
        """Execute OpenAI agent with CPU time slice enforcement."""
        
        def check_cpu_timeout():
            """Check if we've exceeded 150ms CPU time."""
            cpu_elapsed = (time.process_time() - start_cpu_time) * 1000
            if cpu_elapsed > 150:  # 150ms CPU slice
                raise CPUTimeoutError(f"CPU time slice exceeded: {cpu_elapsed:.1f}ms")
        
        # Periodically check CPU time during execution
        async def monitored_execution():
            # Start the agent execution
            runner_task = asyncio.create_task(
                asyncio.to_thread(Runner.run_sync, openai_agent, task_input)
            )
            
            # Monitor CPU usage every 50ms
            while not runner_task.done():
                check_cpu_timeout()
                await asyncio.sleep(0.05)  # 50ms check interval
            
            return await runner_task
        
        return await monitored_execution()
    
    def _calculate_cost(self, result, model: str) -> float:
        """Calculate OpenAI API cost in rupees."""
        if not hasattr(result, 'usage') or not result.usage:
            return 0.0
        
        costs = OPENAI_COSTS.get(model, {"input": 0, "output": 0})
        
        input_cost = (result.usage.prompt_tokens / 1000) * costs["input"]
        output_cost = (result.usage.completion_tokens / 1000) * costs["output"]
        
        return round(input_cost + output_cost, 4)
    
    async def _parse_agent_result(self, result) -> Dict[str, Any]:
        """Parse OpenAI agent result into standardized format."""
        if hasattr(result, 'final_output'):
            return {"output": result.final_output}
        else:
            return {"output": str(result)}
    
    def _handle_timeout_error(self, error, context: AgentContext, span) -> AgentResult:
        """Handle timeout errors with appropriate logging and metrics."""
        error_msg = f"Agent execution timeout: {str(error)}"
        
        span.set_status(Status(StatusCode.ERROR, error_msg))
        
        AGENT_REQUESTS_TOTAL.labels(
            agent_type=self.agent_id,
            status="timeout"
        ).inc()
        
        logger.warning(
            "Agent execution timeout",
            agent_id=self.agent_id,
            doc_id=context.doc_id,
            error=error_msg
        )
        
        return AgentResult(
            success=False,
            data={},
            agent_id=self.agent_id,
            task_type=context.metadata.get("task_type", "analysis"),
            execution_time_ms=150,  # Timeout at 150ms
            model_used="unknown",
            cost_rupees=0.0,
            error=error_msg
        )
    
    def _handle_general_error(self, error, context: AgentContext, span) -> AgentResult:
        """Handle general errors with appropriate logging and metrics."""
        error_msg = f"Agent execution error: {str(error)}"
        
        span.set_status(Status(StatusCode.ERROR, error_msg))
        
        AGENT_REQUESTS_TOTAL.labels(
            agent_type=self.agent_id,
            status="error"
        ).inc()
        
        logger.error(
            "Agent execution error",
            agent_id=self.agent_id,
            doc_id=context.doc_id,
            error=error_msg,
            exc_info=True
        )
        
        return AgentResult(
            success=False,
            data={},
            agent_id=self.agent_id,
            task_type=context.metadata.get("task_type", "analysis"),
            execution_time_ms=int((time.time() - context.start_time) * 1000),
            model_used="unknown",
            cost_rupees=0.0,
            error=error_msg
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Return agent health status for monitoring."""
        try:
            # Check Redis connectivity
            redis_healthy = False
            if self.redis_client:
                await self.redis_client.ping()
                redis_healthy = True
            
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "status": "healthy" if redis_healthy else "degraded",
                "redis_connected": redis_healthy,
                "tools_count": len(self.tools),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @abstractmethod
    async def process_task(self, context: AgentContext, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process agent-specific task - must be implemented by subclasses.
        
        Args:
            context: Agent execution context
            task_data: Task-specific data
            
        Returns:
            Dict containing task results
        """
        pass 