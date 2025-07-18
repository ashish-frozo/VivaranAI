"""
MedBillGuard Agents Package.

Multi-agent system for medical bill analysis with production-ready infrastructure.
"""

from .base_agent import BaseAgent
from .interfaces import AgentContext, AgentResult, ModelHint
from .redis_state import RedisStateManager, DocumentState, AgentResultCache, state_manager
from .agent_registry import (
    AgentRegistry, 
    AgentStatus, 
    TaskCapability, 
    AgentCapabilities, 
    AgentRegistration,
    agent_registry
)
from .router_agent import (
    RouterAgent,
    RoutingStrategy,
    WorkflowType,
    RoutingRequest,
    RoutingDecision,
    WorkflowStep,
    WorkflowDefinition,
    WorkflowResult,
    router_agent
)
from .medical_bill_agent import MedicalBillAgent

__version__ = "1.0.0"
__all__ = [
    # Base components
    "BaseAgent",
    "AgentContext", 
    "AgentResult",
    "ModelHint",
    
    # State management
    "RedisStateManager",
    "DocumentState",
    "AgentResultCache",
    "state_manager",
    
    # Agent registry
    "AgentRegistry",
    "AgentStatus",
    "TaskCapability",
    "AgentCapabilities",
    "AgentRegistration",
    "agent_registry",
    
    # Router agent
    "RouterAgent",
    "RoutingStrategy",
    "WorkflowType",
    "RoutingRequest",
    "RoutingDecision",
    "WorkflowStep",
    "WorkflowDefinition",
    "WorkflowResult",
    "router_agent",
    
    # Medical bill analysis
    "MedicalBillAgent"
] 