"""
Simple Document Router - Railway-optimized agent routing without registry complexity.

This router is designed for Railway deployments where:
1. Agents are created on-demand, not pre-registered
2. No heartbeat monitoring or "degradation" states
3. Simple document type detection → agent creation
4. Fully stateless and Railway cold-start friendly
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import structlog
from opentelemetry import trace

from agents.medical_bill_agent import MedicalBillAgent
from agents.base_agent import AgentContext, ModelHint

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class DocumentType(str, Enum):
    """Document types that can be automatically detected."""
    MEDICAL_BILL = "medical_bill"
    INSURANCE_CLAIM = "insurance_claim"
    LEGAL_CONTRACT = "legal_contract"
    FINANCIAL_STATEMENT = "financial_statement"
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    """Simple routing decision without complex registry lookups."""
    document_type: DocumentType
    agent_type: str
    confidence: float
    processing_time_ms: int


class SimpleDocumentRouter:
    """Simple document router optimized for Railway deployments."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", openai_api_key: str = None):
        self.redis_url = redis_url
        self.openai_api_key = openai_api_key
        self._agent_cache = {}  # In-memory cache, not persistent
        
    async def detect_document_type(self, file_content: str, filename: str = None) -> DocumentType:
        """Detect document type from content and filename."""
        
        # Simple heuristic-based detection (can be enhanced with ML later)
        content_lower = file_content.lower()
        filename_lower = (filename or "").lower()
        
        # Medical bill indicators
        medical_indicators = [
            "medical", "hospital", "clinic", "doctor", "patient", "diagnosis",
            "medicine", "prescription", "treatment", "bill", "invoice",
            "cghs", "esi", "insurance", "medicare", "mediclaim"
        ]
        
        if any(indicator in content_lower or indicator in filename_lower for indicator in medical_indicators):
            return DocumentType.MEDICAL_BILL
            
        # Future: Add other document type detection
        # if "contract" in content_lower or "agreement" in content_lower:
        #     return DocumentType.LEGAL_CONTRACT
        
        return DocumentType.UNKNOWN
    
    async def create_agent(self, agent_type: str):
        """Create agent on-demand without registry complexity."""
        
        # Check cache first
        if agent_type in self._agent_cache:
            return self._agent_cache[agent_type]
            
        if agent_type == "medical_bill_agent":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required for medical bill agent")
                
            agent = MedicalBillAgent(
                redis_url=self.redis_url,
                openai_api_key=self.openai_api_key
            )
            
            # Cache for reuse (in-memory, not persistent)
            self._agent_cache[agent_type] = agent
            return agent
            
        # Future: Add other agent types
        # elif agent_type == "legal_agent":
        #     return LegalAgent(...)
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    async def route_document(
        self, 
        file_content: str, 
        doc_id: str,
        user_id: str,
        filename: str = None,
        **kwargs
    ) -> RoutingDecision:
        """Route document to appropriate agent with simple logic."""
        
        start_time = time.time()
        
        # Detect document type
        doc_type = await self.detect_document_type(file_content, filename)
        
        # Map document type to agent type
        agent_mapping = {
            DocumentType.MEDICAL_BILL: "medical_bill_agent",
            DocumentType.INSURANCE_CLAIM: "medical_bill_agent",  # Same agent for now
            # Future mappings:
            # DocumentType.LEGAL_CONTRACT: "legal_agent",
            # DocumentType.FINANCIAL_STATEMENT: "financial_agent",
        }
        
        agent_type = agent_mapping.get(doc_type, "medical_bill_agent")  # Default fallback
        
        # Calculate confidence (simple heuristic)
        confidence = 0.9 if doc_type != DocumentType.UNKNOWN else 0.5
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return RoutingDecision(
            document_type=doc_type,
            agent_type=agent_type,
            confidence=confidence,
            processing_time_ms=processing_time_ms
        )
    
    async def execute_analysis(
        self,
        routing_decision: RoutingDecision,
        file_content: bytes,
        doc_id: str,
        user_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute analysis using the routed agent."""
        
        # Create agent on-demand
        agent = await self.create_agent(routing_decision.agent_type)
        
        # Execute based on agent type
        if routing_decision.agent_type == "medical_bill_agent":
            return await agent.analyze_medical_bill(
                file_content=file_content,
                doc_id=doc_id,
                user_id=user_id,
                **kwargs
            )
        
        # Future: Add other agent execution paths
        
        else:
            raise ValueError(f"Unknown agent execution path: {routing_decision.agent_type}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Simple health check without registry dependency."""
        return {
            "status": "healthy",
            "router_type": "simple",
            "cached_agents": list(self._agent_cache.keys()),
            "timestamp": time.time()
        } 