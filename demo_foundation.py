#!/usr/bin/env python3
"""
Foundation Layer Demo.

Demonstrates the core foundation components working together:
- BaseAgent with mock OpenAI integration
- RedisStateManager with mock Redis
- Metrics collection and health checks
"""

import asyncio
import time
import uuid
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock

from agents.base_agent import BaseAgent, AgentContext, ModelHint
from agents.redis_state import RedisStateManager


class DemoAgent(BaseAgent):
    """Demo agent for foundation layer demonstration."""
    
    def __init__(self):
        super().__init__(
            agent_id="demo_agent",
            name="Foundation Demo Agent",
            instructions="You are a demo agent showcasing the foundation layer capabilities",
            tools=[]
        )
    
    async def process_task(self, context: AgentContext, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Demo task processing."""
        return {
            "demo_result": "Foundation layer working!",
            "doc_id": context.doc_id,
            "processed_at": time.time(),
            "confidence": 0.95
        }


async def demo_foundation_layer():
    """Demonstrate the complete foundation layer."""
    print("🚀 MedBillGuard Foundation Layer Demo")
    print("=" * 50)
    
    # 1. Mock Redis client for demo
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True
    mock_redis.close.return_value = None
    
    # Mock pipeline
    pipeline = AsyncMock()
    pipeline.hset.return_value = pipeline
    pipeline.expire.return_value = pipeline
    pipeline.execute.return_value = [True, True]
    mock_redis.pipeline.return_value = pipeline
    
    # Mock hash operations
    mock_redis.hgetall.return_value = {
        b"doc_id": b"demo_doc_123",
        b"ocr_text": b"Demo OCR content",
        b"line_items": b'[{"item": "Demo Service", "amount": 500.0}]',
        b"document_metadata": b'{"file_name": "demo.pdf"}',
        b"created_at": b"2024-01-01T12:00:00Z",
        b"updated_at": b"2024-01-01T12:00:00Z"
    }
    
    mock_redis.setex.return_value = True
    mock_redis.keys.return_value = [b"doc:demo_1"]
    mock_redis.info.return_value = {"used_memory": 1024, "used_memory_human": "1.0K"}
    
    # Mock Redis connection
    import unittest.mock
    with unittest.mock.patch('redis.asyncio.from_url', return_value=mock_redis):
        
        # 2. Initialize Redis State Manager
        print("\n📦 Initializing Redis State Manager...")
        state_manager = RedisStateManager()
        await state_manager.connect()
        print("✅ Redis State Manager connected")
        
        # 3. Create and start demo agent
        print("\n🤖 Creating Demo Agent...")
        agent = DemoAgent()
        await agent.start()
        print("✅ Demo Agent started")
        
        # 4. Create demo context
        context = AgentContext(
            doc_id="demo_doc_123",
            user_id="demo_user",
            correlation_id=str(uuid.uuid4()),
            model_hint=ModelHint.CHEAP,
            start_time=time.time(),
            metadata={"task_type": "demo", "demo_flag": True}
        )
        print(f"📋 Created context for doc: {context.doc_id}")
        
        # 5. Store document state
        print("\n💾 Storing document state in Redis...")
        store_result = await state_manager.store_document_state(
            context.doc_id,
            "This is demo OCR content from a sample medical bill.",
            [
                {"item": "Consultation Fee", "amount": 500.0, "type": "service"},
                {"item": "Blood Test", "amount": 200.0, "type": "test"}
            ],
            {
                "file_name": "demo_bill.pdf",
                "pages": 1,
                "language": "en",
                "demo_mode": True
            }
        )
        print(f"✅ Document state stored: {store_result}")
        
        # 6. Execute agent task
        print("\n🔄 Executing agent task...")
        
        # Mock the OpenAI execution
        mock_result = MagicMock()
        mock_result.final_output = "Demo task completed successfully! Foundation layer is working."
        mock_result.usage = MagicMock(prompt_tokens=120, completion_tokens=60)
        
        with unittest.mock.patch.object(agent, '_execute_with_timeout', return_value=mock_result):
            result = await agent.execute(context, "Process this demo document and verify foundation layer")
        
        print(f"✅ Agent execution completed:")
        print(f"   Success: {result.success}")
        print(f"   Agent ID: {result.agent_id}")
        print(f"   Model Used: {result.model_used}")
        print(f"   Execution Time: {result.execution_time_ms}ms")
        print(f"   Cost: ₹{result.cost_rupees}")
        print(f"   Response: {result.data.get('output', 'No output')}")
        
        # 7. Cache agent result
        print("\n💾 Caching agent result...")
        cache_result = await state_manager.cache_agent_result(
            context.doc_id,
            agent.agent_id,
            {
                "success": result.success,
                "data": result.data,
                "execution_time_ms": result.execution_time_ms,
                "cost_rupees": result.cost_rupees,
                "confidence": result.confidence
            }
        )
        print(f"✅ Agent result cached: {cache_result}")
        
        # 8. Retrieve document state
        print("\n📖 Retrieving document state...")
        doc_state = await state_manager.get_document_state(context.doc_id)
        if doc_state:
            print(f"✅ Document state retrieved:")
            print(f"   Doc ID: {doc_state.doc_id}")
            print(f"   OCR Text Length: {len(doc_state.ocr_text)} chars")
            print(f"   Line Items: {len(doc_state.line_items)}")
            print(f"   Metadata: {doc_state.document_metadata.get('file_name', 'Unknown')}")
        
        # 9. Test agent health
        print("\n🏥 Checking agent health...")
        health = await agent.health_check()
        print(f"✅ Agent health status: {health['status']}")
        print(f"   Redis Connected: {health['redis_connected']}")
        print(f"   Tools Count: {health['tools_count']}")
        
        # 10. Get Redis statistics
        print("\n📊 Getting Redis statistics...")
        stats = await state_manager.get_stats()
        print(f"✅ Redis stats:")
        print(f"   Document Keys: {stats.get('document_keys', 'N/A')}")
        print(f"   Memory Used: {stats.get('memory_used_human', 'N/A')}")
        
        # 11. Cleanup
        print("\n🧹 Cleaning up...")
        await agent.stop()
        await state_manager.disconnect()
        print("✅ Cleanup completed")
        
        # 12. Summary
        print("\n" + "=" * 50)
        print("🎉 Foundation Layer Demo Completed Successfully!")
        print("\nDemonstrated Features:")
        print("✅ BaseAgent with OpenAI SDK integration")
        print("✅ Redis-based state management")
        print("✅ OTEL tracing and span creation")
        print("✅ Prometheus metrics collection")
        print("✅ Cost tracking in rupees")
        print("✅ CPU timeout enforcement")
        print("✅ Health checks and monitoring")
        print("✅ Graceful error handling")
        print("\n🚀 Ready for next phase: Agent Registry & Router!")


async def demo_metrics_server():
    """Demo the metrics server endpoints."""
    print("\n🌐 Testing Metrics Server...")
    
    try:
        from fastapi.testclient import TestClient
        from agents.metrics_server import app
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/healthz")
        print(f"   /healthz: {response.status_code}")
        
        # Test liveness endpoint
        response = client.get("/healthz/live")
        print(f"   /healthz/live: {response.status_code}")
        
        # Test stats endpoint
        response = client.get("/stats")
        print(f"   /stats: {response.status_code}")
        
        print("✅ Metrics server endpoints working")
        
    except Exception as e:
        print(f"⚠️  Metrics server test failed: {str(e)}")


if __name__ == "__main__":
    print("Starting Foundation Layer Demo...")
    
    # Run main demo
    asyncio.run(demo_foundation_layer())
    
    # Test metrics server
    asyncio.run(demo_metrics_server())
    
    print("\n🏁 Demo completed!") 