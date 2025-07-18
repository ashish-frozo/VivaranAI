"""
Agent and Tool Interface Definitions - Phase 1 Architecture Standardization

This module defines the formal contracts that all agents and tools must implement
to ensure consistency, extensibility, and maintainability across the system.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum

from .base_agent import AgentContext, AgentResult, ModelHint


class AgentState(str, Enum):
    """Agent lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    WARMING_UP = "warming_up"
    ACTIVE = "active"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    ERROR = "error"


class ToolState(str, Enum):
    """Tool lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentCapabilityDeclaration:
    """Enhanced agent capability declaration."""
    supported_tasks: List[str]
    supported_document_types: List[str]
    supported_languages: List[str]
    max_concurrent_requests: int
    preferred_model_hints: List[ModelHint]
    processing_time_ms_avg: int
    cost_per_request_rupees: float
    confidence_threshold: float
    requires_tools: List[str]  # Tool dependencies
    provides_tools: List[str]  # Tools this agent exposes
    version: str = "1.0.0"
    metadata: Dict[str, Any] = None


@dataclass
class ToolCapabilityDeclaration:
    """Tool capability declaration."""
    tool_name: str
    tool_version: str
    description: str
    supported_operations: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str]  # Other tools this tool depends on
    cost_per_operation_rupees: float
    avg_execution_time_ms: int
    metadata: Dict[str, Any] = None


@dataclass
class LifecycleHooks:
    """Lifecycle hook configuration."""
    on_initialize: Optional[str] = None
    on_warm_up: Optional[str] = None
    on_activate: Optional[str] = None
    on_degrade: Optional[str] = None
    on_shutdown: Optional[str] = None
    on_error: Optional[str] = None


@runtime_checkable
class ILifecycleManager(Protocol):
    """Interface for lifecycle management."""
    
    @property
    def state(self) -> AgentState:
        """Get current state."""
        ...
    
    async def initialize(self) -> bool:
        """Initialize the component."""
        ...
    
    async def warm_up(self) -> bool:
        """Warm up the component (pre-load models, cache data, etc.)."""
        ...
    
    async def activate(self) -> bool:
        """Activate the component for processing."""
        ...
    
    async def degrade(self, reason: str) -> bool:
        """Gracefully degrade component functionality."""
        ...
    
    async def shutdown(self) -> bool:
        """Shutdown the component."""
        ...
    
    async def handle_error(self, error: Exception) -> bool:
        """Handle errors and attempt recovery."""
        ...


@runtime_checkable
class ITool(Protocol):
    """Formal interface that all tools must implement."""
    
    @property
    def tool_name(self) -> str:
        """Get tool name."""
        ...
    
    @property
    def tool_version(self) -> str:
        """Get tool version."""
        ...
    
    @property
    def state(self) -> ToolState:
        """Get current tool state."""
        ...
    
    async def get_capabilities(self) -> ToolCapabilityDeclaration:
        """Get tool capabilities and metadata."""
        ...
    
    async def initialize(self) -> bool:
        """Initialize the tool."""
        ...
    
    async def execute(
        self, 
        operation: str, 
        context: AgentContext, 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a tool operation."""
        ...
    
    async def validate_input(self, operation: str, **kwargs) -> bool:
        """Validate input parameters for an operation."""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Check tool health status."""
        ...
    
    async def shutdown(self) -> bool:
        """Shutdown the tool."""
        ...


@runtime_checkable
class IAgent(Protocol):
    """Formal interface that all agents must implement."""
    
    @property
    def agent_id(self) -> str:
        """Get agent identifier."""
        ...
    
    @property
    def name(self) -> str:
        """Get agent name."""
        ...
    
    @property
    def version(self) -> str:
        """Get agent version."""
        ...
    
    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        ...
    
    async def get_capabilities(self) -> AgentCapabilityDeclaration:
        """Get agent capabilities and metadata."""
        ...
    
    async def initialize(self) -> bool:
        """Initialize the agent."""
        ...
    
    async def warm_up(self) -> bool:
        """Warm up the agent (pre-load models, cache data, etc.)."""
        ...
    
    async def activate(self) -> bool:
        """Activate the agent for processing."""
        ...
    
    async def process_task(
        self, 
        context: AgentContext, 
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a task - core agent functionality."""
        ...
    
    async def execute(
        self, 
        context: AgentContext, 
        task_input: str
    ) -> AgentResult:
        """Execute agent task with full observability."""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health status."""
        ...
    
    async def degrade(self, reason: str) -> bool:
        """Gracefully degrade agent functionality."""
        ...
    
    async def shutdown(self) -> bool:
        """Shutdown the agent."""
        ...
    
    async def self_register(self) -> bool:
        """Register agent with the registry."""
        ...
    
    async def get_tools(self) -> List[ITool]:
        """Get tools provided by this agent."""
        ...


class BaseLifecycleManager:
    """Base implementation of lifecycle management."""
    
    def __init__(self, hooks: Optional[LifecycleHooks] = None):
        self._state = AgentState.UNINITIALIZED
        self._hooks = hooks or LifecycleHooks()
        self._error_count = 0
        self._last_error: Optional[Exception] = None
    
    @property
    def state(self) -> AgentState:
        return self._state
    
    async def initialize(self) -> bool:
        """Initialize the component."""
        try:
            self._state = AgentState.INITIALIZING
            if self._hooks.on_initialize:
                await self._execute_hook(self._hooks.on_initialize)
            self._state = AgentState.READY
            return True
        except Exception as e:
            self._state = AgentState.ERROR
            self._last_error = e
            return False
    
    async def warm_up(self) -> bool:
        """Warm up the component."""
        try:
            if self._state != AgentState.READY:
                return False
            
            self._state = AgentState.WARMING_UP
            if self._hooks.on_warm_up:
                await self._execute_hook(self._hooks.on_warm_up)
            self._state = AgentState.ACTIVE
            return True
        except Exception as e:
            self._state = AgentState.ERROR
            self._last_error = e
            return False
    
    async def activate(self) -> bool:
        """Activate the component."""
        try:
            if self._state not in [AgentState.READY, AgentState.DEGRADED]:
                return False
            
            if self._hooks.on_activate:
                await self._execute_hook(self._hooks.on_activate)
            self._state = AgentState.ACTIVE
            return True
        except Exception as e:
            self._state = AgentState.ERROR
            self._last_error = e
            return False
    
    async def degrade(self, reason: str) -> bool:
        """Gracefully degrade functionality."""
        try:
            self._state = AgentState.DEGRADED
            if self._hooks.on_degrade:
                await self._execute_hook(self._hooks.on_degrade, reason=reason)
            return True
        except Exception as e:
            self._state = AgentState.ERROR
            self._last_error = e
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the component."""
        try:
            self._state = AgentState.SHUTTING_DOWN
            if self._hooks.on_shutdown:
                await self._execute_hook(self._hooks.on_shutdown)
            self._state = AgentState.STOPPED
            return True
        except Exception as e:
            self._state = AgentState.ERROR
            self._last_error = e
            return False
    
    async def handle_error(self, error: Exception) -> bool:
        """Handle errors and attempt recovery."""
        try:
            self._error_count += 1
            self._last_error = error
            
            if self._hooks.on_error:
                await self._execute_hook(self._hooks.on_error, error=error)
            
            # Simple recovery strategy - degrade if too many errors
            if self._error_count > 3:
                await self.degrade(f"Too many errors: {self._error_count}")
            
            return True
        except Exception:
            self._state = AgentState.ERROR
            return False
    
    async def _execute_hook(self, hook_name: str, **kwargs):
        """Execute a lifecycle hook."""
        # This would be implemented by subclasses to execute actual hook logic
        pass


class AgentInterfaceValidator:
    """Utility class to validate agent interface compliance."""
    
    @staticmethod
    def validate_agent(agent: Any) -> List[str]:
        """Validate that an agent implements the IAgent interface."""
        violations = []
        
        # Check if agent implements IAgent protocol
        if not isinstance(agent, IAgent):
            violations.append("Agent does not implement IAgent protocol")
        
        # Check required properties
        required_properties = ['agent_id', 'name', 'version', 'state']
        for prop in required_properties:
            if not hasattr(agent, prop):
                violations.append(f"Missing required property: {prop}")
        
        # Check required methods
        required_methods = [
            'get_capabilities', 'initialize', 'warm_up', 'activate',
            'process_task', 'execute', 'health_check', 'degrade',
            'shutdown', 'self_register', 'get_tools'
        ]
        for method in required_methods:
            if not hasattr(agent, method) or not callable(getattr(agent, method)):
                violations.append(f"Missing required method: {method}")
        
        return violations
    
    @staticmethod
    def validate_tool(tool: Any) -> List[str]:
        """Validate that a tool implements the ITool interface."""
        violations = []
        
        # Check if tool implements ITool protocol
        if not isinstance(tool, ITool):
            violations.append("Tool does not implement ITool protocol")
        
        # Check required properties
        required_properties = ['tool_name', 'tool_version', 'state']
        for prop in required_properties:
            if not hasattr(tool, prop):
                violations.append(f"Missing required property: {prop}")
        
        # Check required methods
        required_methods = [
            'get_capabilities', 'initialize', 'execute', 'validate_input',
            'health_check', 'shutdown'
        ]
        for method in required_methods:
            if not hasattr(tool, method) or not callable(getattr(tool, method)):
                violations.append(f"Missing required method: {method}")
        
        return violations
