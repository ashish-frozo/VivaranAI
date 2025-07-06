# MedBillGuard Agents Foundation Layer

This directory contains the foundational components for the MedBillGuard multi-agent system. The foundation layer provides core primitives for building production-ready, observable, and scalable agents.

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BaseAgent     │    │ RedisStateManager│    │  MetricsServer  │
│                 │    │                 │    │                 │
│ • OpenAI SDK    │◄──►│ • Document State│◄──►│ • /metrics      │
│ • OTEL Tracing  │    │ • Agent Cache   │    │ • /healthz      │
│ • Cost Tracking │    │ • Coordination  │    │ • /stats        │
│ • CPU Limits    │    │ • TTL Management│    │ • Prometheus    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Redis Cluster       │
                    │                          │
                    │ • doc:{id} -> Hash       │
                    │ • file_hash:{sha} -> TTL │
                    │ • lock:{key} -> Atomic   │
                    └──────────────────────────┘
```

## 📁 **Components**

### 1. **BaseAgent** (`base_agent.py`)
Core agent class that all specialized agents inherit from.

**Features:**
- ✅ OpenAI Agents SDK integration
- ✅ OpenTelemetry tracing with span correlation
- ✅ Prometheus metrics (cost, execution time, requests)
- ✅ CPU time slice enforcement (150ms limit)
- ✅ Model selection with cost optimization
- ✅ Redis state management integration
- ✅ Graceful error handling and retry logic

**Usage:**
```python
from agents.base_agent import BaseAgent, AgentContext, ModelHint

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="my_agent",
            name="My Specialized Agent",
            instructions="You are a specialized agent that...",
            tools=[my_tool_function]
        )
    
    async def process_task(self, context, task_data):
        return {"result": "processed"}

# Usage
agent = MyAgent()
await agent.start()

context = AgentContext(
    doc_id="doc123",
    user_id="user456", 
    correlation_id="corr789",
    model_hint=ModelHint.STANDARD,
    start_time=time.time(),
    metadata={"task_type": "analysis"}
)

result = await agent.execute(context, "Process this document")
await agent.stop()
```

### 2. **RedisStateManager** (`redis_state.py`)
Hash-based state management for multi-agent coordination.

**Features:**
- ✅ Atomic document state operations
- ✅ Agent result caching with TTL
- ✅ File hash deduplication (24h cache)
- ✅ Distributed coordination locks
- ✅ Automatic cleanup and housekeeping

**Redis Key Patterns:**
```
doc:{doc_id}                    # Document state hash
doc:{doc_id}:results:{agent_id} # Agent result cache
file_hash:{sha256}              # File content cache
lock:{lock_key}                 # Coordination locks
```

**Usage:**
```python
from agents.redis_state import state_manager

# Store document state
await state_manager.store_document_state(
    doc_id="doc123",
    ocr_text="Extracted text...",
    line_items=[{"item": "Test", "amount": 100.0}],
    metadata={"file_name": "bill.pdf"}
)

# Cache agent result
await state_manager.cache_agent_result(
    doc_id="doc123",
    agent_id="medical_agent",
    result_data={"success": True, "cost_rupees": 2.50}
)

# Coordination locks
acquired = await state_manager.acquire_lock("process_doc123", "worker_1")
if acquired:
    # Do work
    await state_manager.release_lock("process_doc123", "worker_1")
```

### 3. **MetricsServer** (`metrics_server.py`)
FastAPI-based metrics and health check server.

**Endpoints:**
- `GET /metrics` - Prometheus metrics
- `GET /healthz?selftest=true` - Health check with Redis test
- `GET /healthz/ready` - Kubernetes readiness probe  
- `GET /healthz/live` - Kubernetes liveness probe
- `GET /stats` - Detailed statistics

**Metrics Collected:**
```
agent_execution_seconds         # Execution time histogram
openai_rupees_total            # Cost tracking counter
agent_requests_total           # Request success/failure counter
active_agents_total            # Current active agents gauge
agent_redis_connected          # Redis connectivity status
agent_server_uptime_seconds    # Server uptime
```

## 🚀 **Quick Start**

### 1. Install Dependencies
```bash
# Install OpenAI Agents SDK
pip install git+https://github.com/openai/openai-agents-python.git

# Install other dependencies
pip install redis fastapi uvicorn prometheus-client opentelemetry-api
```

### 2. Start Redis
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

### 3. Run Tests
```bash
# Unit tests
pytest tests/test_base_agent.py tests/test_redis_state.py -v

# Integration tests  
pytest tests/test_foundation_integration.py -v
```

### 4. Start Metrics Server
```bash
python -m agents.metrics_server
# Server available at http://localhost:8080
```

## 📊 **Observability**

### OpenTelemetry Tracing
Every agent execution creates spans with attributes:
```json
{
  "agent.id": "medical_agent",
  "agent.name": "Medical Bill Analyzer", 
  "doc.id": "doc123",
  "user.id": "user456",
  "correlation.id": "corr789",
  "model.hint": "standard",
  "model.selected": "gpt-4o",
  "execution.time_ms": 1500,
  "cost.rupees": 2.50
}
```

### Prometheus Metrics
Monitor agent performance:
```bash
# Check metrics
curl http://localhost:8080/metrics

# Check health
curl http://localhost:8080/healthz?selftest=true

# Get stats
curl http://localhost:8080/stats
```

## 🔒 **Production Considerations**

### Cost Control
- ✅ Model selection funnel (cheap → standard → premium)
- ✅ Real-time cost tracking in rupees
- ✅ CPU time slice enforcement (150ms limit)
- ✅ File hash caching to avoid duplicate processing

### Error Handling
- ✅ Retry once with exponential backoff
- ✅ Graceful degradation with partial results
- ✅ Never hard-fail unless all agents fail
- ✅ Prometheus alerts when degraded rate > 1%

### Scalability
- ✅ Independent parallel execution
- ✅ Redis-based state coordination
- ✅ Kubernetes-ready health checks
- ✅ Horizontal scaling via K8s HPA

## 🧪 **Testing**

The foundation layer includes comprehensive test coverage:

- **Unit Tests**: `test_base_agent.py`, `test_redis_state.py`
- **Integration Tests**: `test_foundation_integration.py`
- **Coverage Target**: ≥90%

### Running Tests
```bash
# All tests with coverage
pytest --cov=agents --cov-report=html

# Specific test categories
pytest -m unit        # Unit tests only
pytest -m integration # Integration tests only
pytest -m slow        # Slow tests (can be skipped)
```

## 🔧 **Configuration**

### Environment Variables
```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/1

# OpenAI Configuration  
OPENAI_API_KEY=your_api_key_here

# Tracing Configuration
OTEL_SERVICE_NAME=medbillguard-agent
OTEL_EXPORTER_JAEGER_ENDPOINT=http://jaeger:14268/api/traces

# Metrics Configuration
METRICS_PORT=8080
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_metrics
```

### Redis TTL Configuration
```python
# Default TTL values (seconds)
DOC_STATE_TTL = 24 * 60 * 60      # 24 hours
LINE_ITEMS_TTL = 6 * 60 * 60      # 6 hours  
AGENT_RESULTS_TTL = 6 * 60 * 60   # 6 hours
FILE_HASH_TTL = 24 * 60 * 60      # 24 hours
LOCK_TTL = 5 * 60                 # 5 minutes
```

## 🎯 **Next Steps**

This foundation layer enables:

1. **Agent Registry & Router** (Next PR)
2. **Medical Bill Agent** (Refactor existing logic)
3. **Kubernetes Deployment** (Containerization)
4. **Production Monitoring** (Grafana dashboards)

The foundation is **production-ready** and follows industry best practices for observability, cost control, and scalability. 