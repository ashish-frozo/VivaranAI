"""
Agent Registry - Dynamic agent discovery and capability management.

Provides centralized registration, discovery, and health monitoring of agents
in the MedBillGuard multi-agent system.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Gauge, Counter

from .base_agent import BaseAgent
from .interfaces import AgentContext, ModelHint
from .interfaces import IAgent, AgentCapabilityDeclaration
from .redis_state import RedisStateManager

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Prometheus metrics
REGISTERED_AGENTS = Gauge(
    'registered_agents_total',
    'Total number of registered agents',
    ['status']
)

AGENT_REGISTRATIONS = Counter(
    'agent_registrations_total',
    'Total agent registration events',
    ['agent_type', 'status']
)

ROUTING_DECISIONS = Counter(
    'routing_decisions_total',
    'Total routing decisions made',
    ['routing_strategy', 'agent_selected']
)


class AgentStatus(str, Enum):
    """Agent status enumeration."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"


class TaskCapability(str, Enum):
    """Task capability types that agents can handle."""
    DOCUMENT_PROCESSING = "document_processing"
    OCR_EXTRACTION = "ocr_extraction"
    RATE_VALIDATION = "rate_validation"
    DUPLICATE_DETECTION = "duplicate_detection"
    PROHIBITED_DETECTION = "prohibited_detection"
    CONFIDENCE_SCORING = "confidence_scoring"
    EXPLANATION_BUILDING = "explanation_building"
    TEXT_ANALYSIS = "text_analysis"
    DATA_VALIDATION = "data_validation"
    # Domain-specific analysis capabilities
    MEDICAL_ANALYSIS = "medical_analysis"
    FINANCIAL_ANALYSIS = "financial_analysis"
    LEGAL_ANALYSIS = "legal_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    DOCUMENT_VERIFICATION = "document_verification"
    DATA_EXTRACTION = "data_extraction"
    # Loan-specific capabilities for pack-driven architecture
    LOAN_ANALYSIS = "loan_analysis"
    LOAN_RISK_ASSESSMENT = "loan_risk_assessment"
    LOAN_COMPLIANCE_CHECK = "loan_compliance_check"


@dataclass
class AgentCapabilities:
    """Agent capability metadata."""
    supported_tasks: List[TaskCapability]
    max_concurrent_requests: int
    preferred_model_hints: List[ModelHint]
    processing_time_ms_avg: int
    cost_per_request_rupees: float
    confidence_threshold: float
    supported_document_types: List[str]
    supported_languages: List[str]


@dataclass
class AgentRegistration:
    """Agent registration record."""
    agent_id: str
    name: str
    description: str
    status: AgentStatus
    capabilities: AgentCapabilities
    health_endpoint: Optional[str]
    last_heartbeat: datetime
    registration_time: datetime
    current_requests: int
    total_requests: int
    total_errors: int
    avg_response_time_ms: int
    metadata: Dict[str, Any]
    
    def to_redis_value(self) -> str:
        """Convert to JSON string for Redis storage."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        data['registration_time'] = self.registration_time.isoformat()
        return json.dumps(data)
    
    @classmethod
    def from_redis_value(cls, value: str) -> "AgentRegistration":
        """Create from Redis JSON string."""
        data = json.loads(value)
        # Convert ISO strings back to datetime
        data['last_heartbeat'] = datetime.fromisoformat(data['last_heartbeat'])
        data['registration_time'] = datetime.fromisoformat(data['registration_time'])
        # Convert capability enums
        data['capabilities']['supported_tasks'] = [
            TaskCapability(task) for task in data['capabilities']['supported_tasks']
        ]
        data['capabilities']['preferred_model_hints'] = [
            ModelHint(hint) for hint in data['capabilities']['preferred_model_hints']
        ]
        data['capabilities'] = AgentCapabilities(**data['capabilities'])
        data['status'] = AgentStatus(data['status'])
        return cls(**data)


class AgentRegistry:
    """
    Centralized registry for agent discovery and lifecycle management.
    
    Features:
    - Dynamic agent registration and deregistration
    - Health monitoring with heartbeat checks
    - Capability-based agent discovery
    - Load balancing based on current load
    - Agent performance tracking
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/1"):
        self.redis_url = redis_url
        self.state_manager = RedisStateManager(redis_url)
        
        # Constants - Railway-optimized for longer sleep periods
        self.HEARTBEAT_TIMEOUT = 600  # 10 minutes - increased for Railway cold starts
        self.HEARTBEAT_INTERVAL = 120  # 2 minutes - reduced frequency for Railway
        self.CLEANUP_INTERVAL = 600  # 10 minutes - longer cleanup interval
        self.REGISTRATION_TTL = 1800  # 30 minutes - longer TTL for Railway
        
        # In-memory cache for faster lookups
        self._agent_cache: Dict[str, AgentRegistration] = {}
        self._cache_last_update = 0
        self._cache_ttl = 10  # seconds
        
        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the agent registry service."""
        try:
            await self.state_manager.connect()
            
            # Start background tasks
            self._running = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            logger.info("Agent registry started successfully")
            
        except Exception as e:
            logger.error("Failed to start agent registry", error=str(e), exc_info=True)
            raise
    
    async def stop(self):
        """Stop the agent registry service."""
        self._running = False
        
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
            
        # Wait for tasks to complete
        if self._heartbeat_task:
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        if self._cleanup_task:
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        await self.state_manager.disconnect()
        logger.info("Agent registry stopped")
    
    async def register_agent_self(
        self,
        agent: IAgent,
        health_endpoint: Optional[str] = None
    ) -> bool:
        """
        Self-register an agent using its built-in capabilities.
        
        Args:
            agent: IAgent instance to register
            health_endpoint: Optional health check endpoint
            
        Returns:
            True if registration successful
        """
        try:
            # Get capabilities from agent
            capabilities_decl = await agent.get_capabilities()
            
            # Convert to legacy format for compatibility
            legacy_capabilities = AgentCapabilities(
                supported_tasks=[TaskCapability(task) for task in capabilities_decl.supported_tasks],
                max_concurrent_requests=capabilities_decl.max_concurrent_requests,
                preferred_model_hints=capabilities_decl.preferred_model_hints,
                processing_time_ms_avg=capabilities_decl.processing_time_ms_avg,
                cost_per_request_rupees=capabilities_decl.cost_per_request_rupees,
                confidence_threshold=capabilities_decl.confidence_threshold,
                supported_document_types=capabilities_decl.supported_document_types,
                supported_languages=capabilities_decl.supported_languages
            )
            
            # Use legacy registration method
            return await self.register_agent_legacy(
                agent=agent,
                capabilities=legacy_capabilities,
                health_endpoint=health_endpoint
            )
            
        except Exception as e:
            logger.error(
                "Failed to self-register agent",
                agent_id=getattr(agent, 'agent_id', 'unknown'),
                error=str(e),
                exc_info=True
            )
            return False
    
    async def register_agent(
        self,
        agent,
        capabilities: Optional[AgentCapabilities] = None,
        health_endpoint: Optional[str] = None
    ) -> bool:
        """
        Universal agent registration method that supports both legacy and new interfaces.
        
        Args:
            agent: Agent instance (BaseAgent or IAgent)
            capabilities: Optional legacy capabilities (for BaseAgent)
            health_endpoint: Optional health check endpoint
            
        Returns:
            True if registration successful
        """
        # Check if agent implements IAgent interface
        if hasattr(agent, 'get_capabilities') and callable(getattr(agent, 'get_capabilities')):
            # Use self-registration for IAgent
            return await self.register_agent_self(agent, health_endpoint)
        elif capabilities is not None:
            # Use legacy registration for BaseAgent
            return await self.register_agent_legacy(agent, capabilities, health_endpoint)
        else:
            logger.error(
                "Cannot register agent: no capabilities provided and agent doesn't implement IAgent",
                agent_id=getattr(agent, 'agent_id', 'unknown')
            )
            return False
    
    async def register_agent_legacy(
        self,
        agent: BaseAgent,
        capabilities: AgentCapabilities,
        health_endpoint: Optional[str] = None
    ) -> bool:
        """
        Register an agent with the registry.
        
        Args:
            agent: BaseAgent instance to register
            capabilities: Agent capability metadata
            health_endpoint: Optional health check endpoint
            
        Returns:
            True if registration successful
        """
        with tracer.start_as_current_span("agent_registry.register_agent") as span:
            span.set_attribute("agent.id", agent.agent_id)
            span.set_attribute("agent.name", agent.name)
            
            try:
                registration = AgentRegistration(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    description=agent.instructions[:200],  # Truncate long descriptions
                    status=AgentStatus.STARTING,
                    capabilities=capabilities,
                    health_endpoint=health_endpoint,
                    last_heartbeat=datetime.utcnow(),
                    registration_time=datetime.utcnow(),
                    current_requests=0,
                    total_requests=0,
                    total_errors=0,
                    avg_response_time_ms=0,
                    metadata={}
                )
                
                # Store in Redis
                key = f"agent_registry:{agent.agent_id}"
                if not self.state_manager.redis_client:
                    raise RuntimeError("Redis client not connected")
                    
                await self.state_manager.redis_client.setex(
                    key,
                    self.REGISTRATION_TTL,  # Use longer TTL for Railway
                    registration.to_redis_value()
                )
                
                # Update cache
                self._agent_cache[agent.agent_id] = registration
                
                # Update metrics
                AGENT_REGISTRATIONS.labels(
                    agent_type=agent.agent_id,
                    status="success"
                ).inc()
                
                REGISTERED_AGENTS.labels(status="online").inc()
                
                logger.info(
                    "Agent registered successfully",
                    agent_id=agent.agent_id,
                    name=agent.name,
                    capabilities=len(capabilities.supported_tasks)
                )
                
                # Update status to online after successful registration
                await self._update_agent_status(agent.agent_id, AgentStatus.ONLINE)
                
                span.set_status(Status(StatusCode.OK))
                return True
                
            except Exception as e:
                AGENT_REGISTRATIONS.labels(
                    agent_type=agent.agent_id,
                    status="error"
                ).inc()
                
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                logger.error(
                    "Failed to register agent",
                    agent_id=agent.agent_id,
                    error=str(e),
                    exc_info=True
                )
                return False
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent from the registry.
        
        Args:
            agent_id: Agent identifier to deregister
            
        Returns:
            True if deregistration successful
        """
        with tracer.start_as_current_span("agent_registry.deregister_agent") as span:
            span.set_attribute("agent.id", agent_id)
            
            try:
                # Update status to stopping
                await self._update_agent_status(agent_id, AgentStatus.STOPPING)
                
                # Remove from Redis
                key = f"agent_registry:{agent_id}"
                if self.state_manager.redis_client:
                    await self.state_manager.redis_client.delete(key)
                
                # Remove from cache
                if agent_id in self._agent_cache:
                    del self._agent_cache[agent_id]
                
                # Update metrics
                REGISTERED_AGENTS.labels(status="online").dec()
                
                logger.info("Agent deregistered successfully", agent_id=agent_id)
                
                span.set_status(Status(StatusCode.OK))
                return True
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(
                    "Failed to deregister agent",
                    agent_id=agent_id,
                    error=str(e),
                    exc_info=True
                )
                return False
    
    async def discover_agents_by_task(
        self,
        required_tasks: List[str],
        model_hint: ModelHint = ModelHint.STANDARD,
        max_agents: int = 5,
        document_type: Optional[str] = None,
        language: Optional[str] = None
    ) -> List[AgentRegistration]:
        """
        Enhanced agent discovery by task requirements.
        
        Args:
            required_tasks: List of task types needed
            model_hint: Preferred model complexity
            max_agents: Maximum agents to return
            document_type: Optional document type filter
            language: Optional language filter
            
        Returns:
            List of suitable agent registrations, sorted by suitability
        """
        with tracer.start_as_current_span("discover_agents_by_task") as span:
            span.set_attributes({
                "required_tasks": str(required_tasks),
                "model_hint": model_hint.value,
                "max_agents": max_agents,
                "document_type": document_type or "any",
                "language": language or "any"
            })
            
            try:
                await self._refresh_cache_if_needed()
                
                suitable_agents = []
                
                for registration in self._agent_cache.values():
                    if registration.status != AgentStatus.ONLINE:
                        continue
                    
                    # Check task compatibility
                    agent_tasks = set(registration.capabilities.supported_tasks)
                    required_task_set = set(required_tasks)
                    
                    if not required_task_set.issubset(agent_tasks):
                        continue
                    
                    # Check document type compatibility
                    if (document_type and 
                        document_type not in registration.capabilities.supported_document_types):
                        continue
                    
                    # Check language compatibility
                    if (language and 
                        language not in registration.capabilities.supported_languages):
                        continue
                    
                    # Check model hint compatibility
                    if model_hint not in registration.capabilities.preferred_model_hints:
                        continue
                    
                    # Calculate suitability score
                    score = self._calculate_enhanced_suitability_score(
                        registration, required_tasks, model_hint, document_type, language
                    )
                    
                    suitable_agents.append((registration, score))
                
                # Sort by suitability score (descending)
                suitable_agents.sort(key=lambda x: x[1], reverse=True)
                
                # Return top agents
                result = [agent for agent, _ in suitable_agents[:max_agents]]
                
                # Update metrics
                ROUTING_DECISIONS.labels(
                    routing_strategy="task_based",
                    agent_selected=result[0].agent_id if result else "none"
                ).inc()
                
                logger.info(
                    "Agent discovery completed",
                    required_tasks=required_tasks,
                    agents_found=len(result),
                    top_agent=result[0].agent_id if result else None
                )
                
                return result
                
            except Exception as e:
                logger.error(
                    "Agent discovery failed",
                    required_tasks=required_tasks,
                    error=str(e),
                    exc_info=True
                )
                span.set_status(Status(StatusCode.ERROR, str(e)))
                return []
    
    async def discover_agents(
        self,
        required_capabilities: List[TaskCapability],
        model_hint: ModelHint = ModelHint.STANDARD,
        max_agents: int = 5
    ) -> List[AgentRegistration]:
        """Discover agents that can handle the required capabilities."""
        try:
            # Refresh cache first
            await self._refresh_cache_if_needed()
            
            suitable_agents = []
            
            logger.info(
                "Starting agent discovery",
                required_capabilities=[cap.value for cap in required_capabilities],
                model_hint=model_hint.value,
                cached_agents=len(self._agent_cache)
            )
            
            for agent_id, registration in self._agent_cache.items():
                logger.info(
                    "Evaluating agent",
                    agent_id=agent_id,
                    status=registration.status.value,
                    capabilities=[cap.value for cap in registration.capabilities.supported_tasks],
                    model_hints=[hint.value for hint in registration.capabilities.preferred_model_hints]
                )
                
                if self._agent_matches_requirements(registration, required_capabilities, model_hint):
                    score = self._calculate_suitability_score(registration, required_capabilities, model_hint)
                    suitable_agents.append((registration, score))
                    logger.info(
                        "Agent matches requirements",
                        agent_id=agent_id,
                        score=score
                    )
                else:
                    logger.info(
                        "Agent does not match requirements",
                        agent_id=agent_id,
                        status=registration.status.value,
                        reason="status_check" if registration.status not in [AgentStatus.ONLINE, AgentStatus.DEGRADED] else "other"
                    )
            
            # Sort by suitability score (descending)
            suitable_agents.sort(key=lambda x: x[1], reverse=True)
            
            # Return top agents
            result = [agent for agent, _ in suitable_agents[:max_agents]]
            
            logger.info(
                "Agent discovery completed",
                required_capabilities=[cap.value for cap in required_capabilities],
                agents_found=len(result),
                agent_ids=[agent.agent_id for agent in result]
            )
            
            return result
            
        except Exception as e:
            logger.error("Agent discovery failed", error=str(e), exc_info=True)
            return []
    
    async def get_agent_status(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get current status of a specific agent."""
        await self._refresh_cache_if_needed()
        return self._agent_cache.get(agent_id)
    
    async def list_all_agents(self) -> List[AgentRegistration]:
        """List all registered agents."""
        await self._refresh_cache_if_needed()
        return list(self._agent_cache.values())
    
    async def list_online_agents(self) -> List[AgentRegistration]:
        """List only online agents."""
        await self._refresh_cache_if_needed()
        return [
            agent for agent in self._agent_cache.values() 
            if agent.status == AgentStatus.ONLINE
        ]
    
    async def update_agent_metrics(
        self,
        agent_id: str,
        execution_time_ms: int,
        success: bool
    ):
        """Update agent performance metrics after task execution."""
        try:
            registration = await self.get_agent_status(agent_id)
            if not registration:
                return
            
            # Update metrics
            registration.total_requests += 1
            if not success:
                registration.total_errors += 1
            
            # Update average response time (exponential moving average)
            alpha = 0.1  # Smoothing factor
            registration.avg_response_time_ms = int(
                alpha * execution_time_ms + 
                (1 - alpha) * registration.avg_response_time_ms
            )
            
            # Update heartbeat
            registration.last_heartbeat = datetime.utcnow()
            
            # Store updated registration
            key = f"agent_registry:{agent_id}"
            if self.state_manager.redis_client:
                await self.state_manager.redis_client.setex(
                    key,
                    self.HEARTBEAT_TIMEOUT * 2,
                    registration.to_redis_value()
                )
            
            # Update cache
            self._agent_cache[agent_id] = registration
            
        except Exception as e:
            logger.error(
                "Failed to update agent metrics",
                agent_id=agent_id,
                error=str(e),
                exc_info=True
            )
    
    def _agent_matches_requirements(
        self,
        registration: AgentRegistration,
        required_capabilities: List[TaskCapability],
        model_hint: ModelHint
    ) -> bool:
        """Check if agent matches requirements."""
        # Must be online or degraded (degraded agents can still process requests)
        if registration.status not in [AgentStatus.ONLINE, AgentStatus.DEGRADED]:
            return False
        
        # Must support all required capabilities
        agent_caps = set(registration.capabilities.supported_tasks)
        required_caps = set(required_capabilities)
        if not required_caps.issubset(agent_caps):
            return False
        
        # Check if agent supports the preferred model hint
        if model_hint not in registration.capabilities.preferred_model_hints:
            return False
        
        # Check if agent has capacity
        if registration.current_requests >= registration.capabilities.max_concurrent_requests:
            return False
        
        return True
    
    def _calculate_enhanced_suitability_score(
        self,
        registration: AgentRegistration,
        required_tasks: List[str],
        model_hint: ModelHint,
        document_type: Optional[str] = None,
        language: Optional[str] = None
    ) -> float:
        """
        Calculate enhanced suitability score for agent selection.
        
        Args:
            registration: Agent registration to score
            required_tasks: Required task types
            model_hint: Preferred model complexity
            document_type: Optional document type requirement
            language: Optional language requirement
            
        Returns:
            Suitability score (0.0 to 1.0)
        """
        score = 0.0
        
        # Task coverage score (40% weight)
        agent_tasks = set(registration.capabilities.supported_tasks)
        required_task_set = set(required_tasks)
        task_coverage = len(required_task_set.intersection(agent_tasks)) / len(required_task_set)
        score += task_coverage * 0.4
        
        # Model hint compatibility (20% weight)
        if model_hint in registration.capabilities.preferred_model_hints:
            score += 0.2
        
        # Performance metrics (20% weight)
        # Lower response time and higher success rate = better score
        response_time_score = max(0, 1 - (registration.avg_response_time_ms / 10000))  # Normalize to 10s max
        success_rate = 1.0 - (registration.total_errors / max(registration.total_requests, 1))
        performance_score = (response_time_score + success_rate) / 2
        score += performance_score * 0.2
        
        # Document type compatibility (10% weight)
        if document_type:
            if document_type in registration.capabilities.supported_document_types:
                score += 0.1
        else:
            score += 0.1  # No specific requirement
        
        # Language compatibility (10% weight)
        if language:
            if language in registration.capabilities.supported_languages:
                score += 0.1
        else:
            score += 0.1  # No specific requirement
        
        return min(score, 1.0)
    
    def _calculate_suitability_score(
        self,
        registration: AgentRegistration,
        required_capabilities: List[TaskCapability],
        model_hint: ModelHint
    ) -> float:
        """Calculate suitability score for agent selection."""
        score = 0.0
        
        # Base score for capability match
        score += 10.0
        
        # Bonus for lower current load
        load_ratio = registration.current_requests / max(
            registration.capabilities.max_concurrent_requests, 1
        )
        score += (1.0 - load_ratio) * 5.0
        
        # Bonus for faster response times
        if registration.avg_response_time_ms > 0:
            # Normalize to 0-3 bonus points (faster = higher score)
            time_score = max(0, 3.0 - (registration.avg_response_time_ms / 1000))
            score += time_score
        
        # Bonus for higher success rate
        if registration.total_requests > 0:
            success_rate = 1.0 - (registration.total_errors / registration.total_requests)
            score += success_rate * 2.0
        
        # Bonus for lower cost
        if registration.capabilities.cost_per_request_rupees > 0:
            # Normalize cost (lower cost = higher score)
            cost_score = max(0, 2.0 - (registration.capabilities.cost_per_request_rupees / 5.0))
            score += cost_score
        
        return score
    
    async def _refresh_cache_if_needed(self):
        """Refresh agent cache if TTL expired."""
        current_time = time.time()
        if current_time - self._cache_last_update > self._cache_ttl:
            await self._refresh_cache()
    
    async def _refresh_cache(self):
        """Refresh agent cache from Redis."""
        try:
            if not self.state_manager.redis_client:
                return
            
            # Get all agent registrations
            pattern = "agent_registry:*"
            keys = await self.state_manager.redis_client.keys(pattern)
            
            new_cache = {}
            for key in keys:
                try:
                    value = await self.state_manager.redis_client.get(key)
                    if value:
                        agent_id = key.decode().split(':', 1)[1]
                        registration = AgentRegistration.from_redis_value(value.decode())
                        new_cache[agent_id] = registration
                except Exception as e:
                    logger.warning(
                        "Failed to parse agent registration",
                        key=key,
                        error=str(e)
                    )
            
            self._agent_cache = new_cache
            self._cache_last_update = time.time()
            
            # Update metrics
            REGISTERED_AGENTS.labels(status="online").set(
                len([a for a in new_cache.values() if a.status == AgentStatus.ONLINE])
            )
            
        except Exception as e:
            logger.error("Failed to refresh agent cache", error=str(e), exc_info=True)
    
    async def _update_agent_status(self, agent_id: str, status: AgentStatus):
        """Update agent status in registry."""
        try:
            registration = await self.get_agent_status(agent_id)
            if registration:
                registration.status = status
                registration.last_heartbeat = datetime.utcnow()
                
                # Update in Redis
                key = f"agent_registry:{agent_id}"
                if self.state_manager.redis_client:
                    await self.state_manager.redis_client.setex(
                        key,
                        self.REGISTRATION_TTL,
                        registration.to_redis_value()
                    )
                
                # Update cache
                self._agent_cache[agent_id] = registration
                
        except Exception as e:
            logger.error(
                "Failed to update agent status",
                agent_id=agent_id,
                status=status.value,
                error=str(e)
            )
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and update status."""
        while self._running:
            try:
                await self._check_agent_heartbeats()
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat monitor error", error=str(e), exc_info=True)
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
    
    async def _check_agent_heartbeats(self):
        """Check agent heartbeats and mark stale agents as offline."""
        current_time = datetime.utcnow()
        timeout_threshold = current_time - timedelta(seconds=self.HEARTBEAT_TIMEOUT)
        
        await self._refresh_cache()
        
        for agent_id, registration in self._agent_cache.items():
            if (registration.status == AgentStatus.ONLINE and 
                registration.last_heartbeat < timeout_threshold):
                
                logger.warning(
                    "Agent heartbeat timeout - marking as degraded",
                    agent_id=agent_id,
                    last_heartbeat=registration.last_heartbeat.isoformat()
                )
                
                await self._update_agent_status(agent_id, AgentStatus.DEGRADED)
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of stale agent registrations."""
        while self._running:
            try:
                await self._cleanup_stale_agents()
                await asyncio.sleep(self.CLEANUP_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup task error", error=str(e), exc_info=True)
                await asyncio.sleep(self.CLEANUP_INTERVAL)
    
    async def _cleanup_stale_agents(self):
        """Remove agents that have been offline for too long."""
        current_time = datetime.utcnow()
        stale_threshold = current_time - timedelta(seconds=self.HEARTBEAT_TIMEOUT * 3)
        
        await self._refresh_cache()
        
        stale_agents = []
        for agent_id, registration in self._agent_cache.items():
            if (registration.status in [AgentStatus.DEGRADED, AgentStatus.OFFLINE] and
                registration.last_heartbeat < stale_threshold):
                stale_agents.append(agent_id)
        
        for agent_id in stale_agents:
            logger.info("Cleaning up stale agent", agent_id=agent_id)
            await self.deregister_agent(agent_id)
            
    async def refresh_agent_cache(self):
        """Public method to refresh the agent cache.
        
        This is a wrapper around the internal _refresh_cache method.
        """
        try:
            await self._refresh_cache()
            logger.info("Agent cache refreshed successfully")
            return True
        except Exception as e:
            logger.error("Failed to refresh agent cache", error=str(e))
            return False


# Global registry instance
agent_registry = AgentRegistry() 