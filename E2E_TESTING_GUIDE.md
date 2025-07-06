# End-to-End Testing Guide for MedBillGuardAgent

This guide will help you set up and run comprehensive end-to-end tests for the MedBillGuardAgent multi-agent system.

## 📋 Prerequisites

### 1. System Requirements
- Python 3.8+
- Docker and Docker Compose
- Git
- At least 4GB RAM available
- 2GB free disk space

### 2. Required API Keys
- **OpenAI API Key**: Required for LLM operations
  - Get from: https://platform.openai.com/api-keys
  - Set as environment variable: `export OPENAI_API_KEY="sk-your-key-here"`

### 3. Development Dependencies
```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Additional dependencies for E2E testing
pip install aiohttp redis psycopg2-binary
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and navigate to project
cd /path/to/VivaranAI

# Set required environment variables
export OPENAI_API_KEY="sk-your-openai-api-key-here"
export REDIS_URL="redis://localhost:6379/1"
export DATABASE_URL="postgresql://medbillguard:medbillguard_dev_password@localhost:5432/medbillguard"

# Optional: Set debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG
```

### 2. Start Infrastructure
```bash
# Start all services using Docker Compose
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs if needed
docker-compose logs medbillguard-agent
```

### 3. Run Tests

#### Quick Smoke Tests (2-3 minutes)
```bash
python test_e2e_runner.py --quick
```

#### Full Test Suite (10-15 minutes)
```bash
python test_e2e_runner.py --full
```

#### Specific Scenario Test
```bash
python test_e2e_runner.py --scenario "High Overcharge with Duplicates"
```

#### Performance/Load Tests
```bash
python test_e2e_runner.py --load
```

## 🔧 Detailed Setup

### 1. Infrastructure Components

The E2E testing requires the following services to be running:

| Service | Port | Purpose |
|---------|------|---------|
| MedBillGuard Agent | 8001 | Main application server |
| Redis | 6379 | State management and caching |
| PostgreSQL | 5432 | Data persistence |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Metrics visualization |
| Jaeger | 16686 | Distributed tracing |

### 2. Service Health Checks

Before running tests, verify all services are healthy:

```bash
# Check agent server
curl http://localhost:8001/health

# Check Redis
docker exec medbillguard-redis redis-cli ping

# Check PostgreSQL
docker exec medbillguard-postgres pg_isready -U medbillguard

# Check all services
docker-compose ps
```

### 3. Configuration Options

You can customize the test configuration by modifying the `E2ETestConfig` class in `test_e2e_runner.py`:

```python
@dataclass
class E2ETestConfig:
    agent_server_url: str = "http://localhost:8001"
    redis_url: str = "redis://localhost:6379/1"
    timeout_seconds: int = 30
    load_test_concurrent_requests: int = 5
    # ... other options
```

## 📊 Test Scenarios

### Available Test Scenarios

1. **High Overcharge with Duplicates**
   - Apollo Hospitals bill with multiple duplicate charges
   - Expected: High confidence duplicate detection
   - Expected overcharge: ~₹5,000

2. **Normal Bill - Minimal Issues**
   - MAX Healthcare pediatric consultation
   - Expected: Clean bill with no issues

3. **Complex Surgery with Prohibited Items**
   - Fortis Hospital surgery bill with non-medical charges
   - Expected: Detection of prohibited items (food, TV, visitor passes)

4. **Hindi Language Bill**
   - Government hospital bill in Hindi
   - Expected: Multilingual OCR processing

5. **Emergency Department Bill**
   - AIIMS emergency bill with time-sensitive charges
   - Expected: Proper handling of emergency charges

### Test Categories

#### 🔥 Smoke Tests (`--quick`)
- Agent server health check
- Basic medical bill analysis
- **Duration**: 2-3 minutes

#### 🔧 Infrastructure Tests
- Redis connection and operations
- PostgreSQL connectivity
- Prometheus metrics endpoint
- Jaeger tracing availability

#### ⚕️ Workflow Tests
- Complete end-to-end medical bill analysis
- All 5 test scenarios
- Validation of analysis results

#### 🚀 Performance Tests (`--load`)
- Single request latency measurement
- Concurrent request handling
- Memory usage analysis
- Error rate under load

## 📈 Understanding Test Results

### Test Report Structure

The test runner generates a comprehensive JSON report:

```json
{
  "summary": {
    "total_tests": 10,
    "passed_tests": 9,
    "failed_tests": 1,
    "success_rate": 90.0,
    "total_duration_seconds": 45.2,
    "timestamp": "2024-01-15T10:30:00"
  },
  "results": [...],
  "failed_tests": [...]
}
```

### Success Criteria

#### ✅ Passing Tests Should Show:
- HTTP 200 responses from all endpoints
- Complete analysis results with all components
- Processing times under 10 seconds per request
- Successful duplicate detection for test scenarios
- Proper confidence scoring
- No critical errors in logs

#### ❌ Common Failure Points:
- Missing OpenAI API key
- Services not running (Redis, PostgreSQL)
- Network connectivity issues
- Insufficient memory/resources
- API rate limiting

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Services Not Starting
```bash
# Check Docker daemon is running
docker version

# Restart services
docker-compose down
docker-compose up -d

# Check resource usage
docker stats
```

#### 2. OpenAI API Issues
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API key directly
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.openai.com/v1/models
```

#### 3. Connection Timeouts
```bash
# Increase timeout in test config
# Modify timeout_seconds in E2ETestConfig

# Check network connectivity
curl -v http://localhost:8001/health
```

#### 4. Memory Issues
```bash
# Check available memory
free -h

# Reduce concurrent requests
# Modify load_test_concurrent_requests in config

# Monitor container memory usage
docker stats medbillguard-agent
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Set debug environment variables
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run tests with verbose output
python test_e2e_runner.py --quick 2>&1 | tee debug.log
```

### Log Locations

- **Application logs**: `docker-compose logs medbillguard-agent`
- **Test logs**: `test_e2e.log`
- **Redis logs**: `docker-compose logs redis`
- **PostgreSQL logs**: `docker-compose logs postgres`

## 📚 Test Data

### Sample Medical Bills

The test suite includes 5 comprehensive medical bill scenarios:

1. **High-value private hospital bill** with duplicate charges
2. **Simple pediatric consultation** with standard rates
3. **Complex surgical procedure** with prohibited items
4. **Hindi language bill** for multilingual testing
5. **Emergency department bill** with time-sensitive charges

### Adding Custom Test Data

To add your own test scenarios:

1. Edit `tests/test_data/sample_medical_bills.py`
2. Add new scenario to `TEST_SCENARIOS` list
3. Include expected findings for validation
4. Run tests with your new scenario

## 🔍 Monitoring During Tests

### Real-time Monitoring

While tests are running, you can monitor:

- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090
- **Jaeger Traces**: http://localhost:16686
- **Redis Insight**: http://localhost:8080

### Key Metrics to Watch

- Request latency and throughput
- Memory and CPU usage
- Error rates and status codes
- Agent health and availability
- Cache hit rates

## 🚀 Advanced Usage

### Custom Configuration

Create a custom config file for specific testing needs:

```python
# custom_test_config.py
from test_e2e_runner import E2ETestConfig

config = E2ETestConfig(
    agent_server_url="http://production-server:8001",
    timeout_seconds=60,
    load_test_concurrent_requests=10
)
```

### Continuous Integration

For CI/CD pipelines:

```bash
# Run in CI mode (no interactive output)
python test_e2e_runner.py --quick --output ci_report.json

# Check exit code
if [ $? -eq 0 ]; then
    echo "All tests passed"
else
    echo "Tests failed"
    exit 1
fi
```

### Performance Benchmarking

Compare performance across versions:

```bash
# Baseline run
python test_e2e_runner.py --load --output baseline_report.json

# After changes
python test_e2e_runner.py --load --output current_report.json

# Compare results
python compare_reports.py baseline_report.json current_report.json
```

## 📞 Support

If you encounter issues:

1. Check this troubleshooting guide
2. Review the logs for error messages
3. Verify all prerequisites are met
4. Test individual components separately
5. Create an issue with detailed error information

## 🎯 Next Steps

After successful E2E testing:

1. **Production Deployment**: Use the K8s manifests in `k8s/`
2. **Monitoring Setup**: Configure production monitoring
3. **Load Testing**: Scale up the load tests for production volumes
4. **Security Testing**: Run security scans and penetration tests
5. **Documentation**: Update deployment and operational documentation 