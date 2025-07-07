# VivaranAI MedBillGuardAgent API Documentation

## Base URL

**Production (Railway)**: `https://endearing-prosperity-production.up.railway.app`
**Local Development**: `http://localhost:8001`

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible for testing purposes.

## API Overview

The VivaranAI MedBillGuardAgent API provides endpoints for:
- Medical bill analysis and overcharge detection
- Document processing and OCR
- System health monitoring
- Agent status and metrics
- Enhanced multi-stage document analysis

## Core Endpoints

### 1. Health Check

#### `GET /health`

Check the overall health status of the API and its components.

**Response:**
```json
{
  "status": "healthy", // or "unhealthy" 
  "timestamp": 1751878939.4060893,
  "uptime_seconds": 526.4947047233582,
  "version": "1.0.0",
  "components": {
    "redis": "healthy",
    "registry": "healthy", 
    "medical_agent": "degraded" // or "healthy"
  }
}
```

**Status Codes:**
- `200` - System is healthy
- `503` - System is unhealthy

---

### 2. List Available Agents

#### `GET /agents`

Get a list of all available analysis agents and their capabilities.

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "medical_bill_agent",
      "name": "Medical Bill Analysis Agent",
      "status": "ONLINE",
      "capabilities": {
        "document_types": ["medical_bill", "pharmacy_invoice"],
        "languages": ["english", "hindi", "bengali", "tamil"],
        "processing_time_ms_avg": 25000,
        "confidence_threshold": 0.8
      }
    }
  ],
  "total": 1
}
```

---

### 3. System Metrics

#### `GET /metrics/summary`

Get comprehensive system performance metrics.

**Response:**
```json
{
  "timestamp": 1751878939.4060893,
  "active_agents": 1,
  "total_analyses": 45,
  "average_confidence": 0.85,
  "uptime_seconds": 526.4947047233582
}
```

---

### 4. Medical Bill Analysis

#### `POST /analyze`

Analyze a medical bill for overcharges and compliance issues.

**Request Body:**
```json
{
  "file_content": "base64_encoded_file_content",
  "doc_id": "unique_document_identifier",
  "user_id": "user_identifier", 
  "language": "english",
  "state_code": "DL",
  "insurance_type": "cghs",
  "file_format": "pdf"
}
```

**Parameters:**
- `file_content` (string, required): Base64 encoded file content
- `doc_id` (string, required): Unique identifier for the document
- `user_id` (string, required): User identifier
- `language` (string, optional): Document language (`english`, `hindi`, `bengali`, `tamil`). Default: `english`
- `state_code` (string, optional): State code for region-specific rates (`DL`, `MH`, `KA`, etc.)
- `insurance_type` (string, optional): Insurance type (`cghs`, `esi`, `private`). Default: `cghs`
- `file_format` (string, optional): File format (`pdf`, `jpg`, `png`, `txt`). Default: `pdf`

**Response:**
```json
{
  "success": true,
  "doc_id": "unique_document_identifier",
  "analysis_complete": true,
  "verdict": "overcharge_detected", // "ok", "warning", "critical"
  "total_bill_amount": 15000.0,
  "total_overcharge": 3000.0,
  "confidence_score": 0.92,
  "red_flags": [
    {
      "item": "Consultation Fee",
      "reason": "Exceeds CGHS rate by ₹1,500",
      "billed": 2000.0,
      "max_allowed": 500.0,
      "overcharge_amount": 1500.0,
      "confidence": 0.95,
      "type": "overcharge"
    }
  ],
  "recommendations": [
    "Contest consultation fee as it exceeds government rate by ₹1,500",
    "Request itemized breakdown for diagnostic charges"
  ],
  "processing_time_seconds": 12.5
}
```

**Status Codes:**
- `200` - Analysis completed successfully
- `400` - Invalid request parameters
- `500` - Internal server error during analysis
- `503` - No agents available

---

### 5. Enhanced Document Analysis

#### `POST /analyze-enhanced`

Advanced multi-stage document analysis with intelligent routing.

**Request Body:**
```json
{
  "file_content": "base64_encoded_file_content",
  "doc_id": "doc_001",
  "user_id": "user_123",
  "language": "english",
  "file_format": "pdf",
  "routing_strategy": "capability_based",
  "priority": "normal"
}
```

**Parameters:**
- `file_content` (string, required): Base64 encoded file content
- `doc_id` (string, required): Unique document identifier
- `user_id` (string, required): User identifier
- `language` (string, optional): Document language. Default: `english`
- `file_format` (string, optional): File format hint
- `routing_strategy` (string, optional): Routing strategy (`capability_based`, `performance_optimized`). Default: `capability_based`
- `priority` (string, optional): Processing priority (`low`, `normal`, `high`). Default: `normal`

**Response:**
```json
{
  "success": true,
  "doc_id": "doc_001",
  "document_type": "medical_bill",
  "processing_stages": {
    "ocr_extraction": {
      "completed": true,
      "processing_time_ms": 3500,
      "confidence": 0.94
    },
    "document_classification": {
      "completed": true,
      "processing_time_ms": 1200,
      "document_type": "medical_bill"
    },
    "domain_analysis": {
      "completed": true,
      "processing_time_ms": 8500,
      "agent_used": "medical_bill_agent"
    }
  },
  "final_result": {
    "verdict": "overcharge_detected",
    "total_bill_amount": 15000.0,
    "total_overcharge": 3000.0,
    "confidence_score": 0.92,
    "red_flags": [...],
    "recommendations": [...]
  },
  "total_processing_time_ms": 13200,
  "error": null
}
```

---

### 6. Liveness Probe

#### `GET /health/liveness`

Simple liveness check for container orchestration.

**Response:**
```json
{
  "status": "alive",
  "timestamp": 1751878939.4060893
}
```

---

### 7. Readiness Probe

#### `GET /health/readiness`

Detailed readiness check including dependencies.

**Response:**
```json
{
  "ready": true,
  "timestamp": 1751878939.4060893,
  "components": {
    "database": "ready",
    "redis": "ready",
    "agents": "ready"
  }
}
```

---

## Error Handling

### Standard Error Response Format

```json
{
  "detail": "Error message describing what went wrong",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": 1751878939.4060893,
  "request_id": "req_12345"
}
```

### Common Error Codes

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_REQUEST | Invalid request parameters |
| 400 | FILE_TOO_LARGE | File size exceeds limit (15MB) |
| 400 | UNSUPPORTED_FORMAT | File format not supported |
| 404 | AGENT_NOT_FOUND | Specified agent does not exist |
| 422 | VALIDATION_ERROR | Request validation failed |
| 500 | ANALYSIS_FAILED | Analysis process failed |
| 500 | INTERNAL_ERROR | Internal server error |
| 503 | NO_AGENTS_AVAILABLE | No agents available to process request |
| 503 | SERVICE_UNAVAILABLE | Service temporarily unavailable |

---

## Rate Limiting

Currently, there are no rate limits enforced on the API. However, for production use, the following limits are recommended:
- **Requests per minute**: 60 per IP
- **File uploads per hour**: 100 per IP
- **Concurrent requests**: 10 per IP

---

## File Upload Guidelines

### Supported Formats
- **PDF**: Up to 15MB, multi-page supported
- **JPEG/JPG**: Up to 15MB, high resolution recommended
- **PNG**: Up to 15MB, high resolution recommended
- **TXT**: Up to 1MB, UTF-8 encoded

### File Encoding
All files must be base64 encoded before sending to the API.

**Example (JavaScript):**
```javascript
// Convert file to base64
const file = document.getElementById('fileInput').files[0];
const reader = new FileReader();
reader.onload = function(e) {
  const base64Content = e.target.result.split(',')[1]; // Remove data URL prefix
  // Send base64Content to API
};
reader.readAsDataURL(file);
```

**Example (Python):**
```python
import base64

with open("medical_bill.pdf", "rb") as file:
    file_content = base64.b64encode(file.read()).decode('utf-8')
    # Send file_content to API
```

---

## Response Time Expectations

| Endpoint | Typical Response Time | Maximum Response Time |
|----------|----------------------|----------------------|
| `/health` | <1 second | 5 seconds |
| `/agents` | <1 second | 5 seconds |
| `/metrics/summary` | <2 seconds | 10 seconds |
| `/analyze` | 10-30 seconds | 60 seconds |
| `/analyze-enhanced` | 15-35 seconds | 90 seconds |

**Note**: Railway cold starts can add 30-60 seconds to the first request after inactivity.

---

## Testing Examples

### cURL Examples

#### Health Check
```bash
curl -X GET "https://endearing-prosperity-production.up.railway.app/health"
```

#### List Agents
```bash
curl -X GET "https://endearing-prosperity-production.up.railway.app/agents"
```

#### Analyze Medical Bill
```bash
curl -X POST "https://endearing-prosperity-production.up.railway.app/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_content": "JVBERi0xLjQKJcOkw7zDtsO...",
    "doc_id": "test_bill_001",
    "user_id": "test_user",
    "language": "english",
    "state_code": "DL",
    "insurance_type": "cghs",
    "file_format": "pdf"
  }'
```

### Python Examples

#### Using requests library
```python
import requests
import base64

# Read and encode file
with open("medical_bill.pdf", "rb") as f:
    file_content = base64.b64encode(f.read()).decode('utf-8')

# API request
response = requests.post(
    "https://endearing-prosperity-production.up.railway.app/analyze",
    json={
        "file_content": file_content,
        "doc_id": "test_bill_001", 
        "user_id": "test_user",
        "language": "english",
        "state_code": "DL",
        "insurance_type": "cghs",
        "file_format": "pdf"
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Verdict: {result['verdict']}")
    print(f"Total Overcharge: ₹{result['total_overcharge']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### JavaScript/Node.js Examples

```javascript
const fs = require('fs');
const axios = require('axios');

// Read and encode file
const fileBuffer = fs.readFileSync('medical_bill.pdf');
const fileContent = fileBuffer.toString('base64');

// API request
axios.post('https://endearing-prosperity-production.up.railway.app/analyze', {
  file_content: fileContent,
  doc_id: 'test_bill_001',
  user_id: 'test_user', 
  language: 'english',
  state_code: 'DL',
  insurance_type: 'cghs',
  file_format: 'pdf'
})
.then(response => {
  console.log('Verdict:', response.data.verdict);
  console.log('Total Overcharge: ₹', response.data.total_overcharge);
})
.catch(error => {
  console.error('Error:', error.response?.data || error.message);
});
```

---

## Interactive API Documentation

For interactive API documentation with the ability to test endpoints directly:

**Production**: [https://endearing-prosperity-production.up.railway.app/docs](https://endearing-prosperity-production.up.railway.app/docs)

The interactive documentation provides:
- Complete API schema
- Request/response examples
- Try-it-out functionality
- Parameter descriptions
- Error code explanations

---

## Webhook Support (Future)

Future versions will support webhooks for:
- Analysis completion notifications
- System health alerts
- Batch processing status updates

---

## SDK and Client Libraries (Planned)

Planned client libraries:
- **Python SDK**: `pip install vivaranai-client`
- **JavaScript SDK**: `npm install vivaranai-client`
- **Java SDK**: Maven/Gradle support
- **PHP SDK**: Composer package

---

## Support and Issues

For API support:
- **GitHub Issues**: [Create an issue](https://github.com/ashish-frozo/VivaranAI/issues)
- **Documentation**: [Full documentation](../README.md)
- **Production Dashboard**: Use the frontend dashboard for testing

---

## Changelog

### v1.0.0 (Current)
- ✅ Railway production deployment
- ✅ Complete medical bill analysis API
- ✅ Enhanced document processing
- ✅ Real-time health monitoring
- ✅ Comprehensive error handling
- ✅ Production frontend dashboard

### Upcoming (v1.1.0)
- 🔄 Webhook support
- 🔄 Batch processing endpoints
- 🔄 User authentication
- 🔄 Rate limiting
- 🔄 WebSocket real-time updates 