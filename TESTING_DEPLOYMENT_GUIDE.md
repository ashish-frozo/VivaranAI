# VivaranAI MedBillGuardAgent - Testing Guide

## 🎯 **Deployment URL**
**Base URL**: `https://endearing-prosperity-production.up.railway.app`

## 🔍 **Quick Health Check**

### 1. Basic Health Status
```bash
curl -s https://endearing-prosperity-production.up.railway.app/health | jq .
```

### 2. Readiness Check
```bash
curl -s https://endearing-prosperity-production.up.railway.app/health/readiness
```

### 3. Liveness Check
```bash
curl -s https://endearing-prosperity-production.up.railway.app/health/liveness
```

## 🧪 **API Endpoint Testing**

### 1. List Available Agents
```bash
curl -s https://endearing-prosperity-production.up.railway.app/agents | jq .
```

### 2. Get Agent Details
```bash
curl -s https://endearing-prosperity-production.up.railway.app/agents/medical_bill_agent | jq .
```

### 3. Check Metrics
```bash
curl -s https://endearing-prosperity-production.up.railway.app/metrics/summary | jq .
```

## 📄 **Medical Bill Analysis Testing**

### Test 1: Basic Medical Bill Analysis
```bash
# Create a sample medical bill (base64 encoded)
echo "Sample Medical Bill Content" | base64 > sample_bill.txt

# Test the analysis endpoint
curl -X POST https://endearing-prosperity-production.up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "file_content": "'$(cat sample_bill.txt)'",
    "doc_id": "test_doc_001",
    "user_id": "test_user_001",
    "language": "english",
    "state_code": "delhi",
    "insurance_type": "cghs",
    "file_format": "pdf"
  }' | jq .
```

### Test 2: Enhanced Analysis with Auto-Detection
```bash
curl -X POST https://endearing-prosperity-production.up.railway.app/analyze-enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "file_content": "'$(cat sample_bill.txt)'",
    "doc_id": "test_doc_002",
    "user_id": "test_user_002",
    "language": "english",
    "routing_strategy": "capability_based",
    "priority": "high"
  }' | jq .
```

## 🧪 **Advanced Testing**

### Load Testing with Real Medical Bill
```bash
# Use a real medical bill (you'll need to encode it)
# For testing, create a more realistic sample:

cat > realistic_medical_bill.txt << 'EOF'
APOLLO HOSPITALS
Medical Bill Receipt
Date: 2024-01-15
Patient: John Doe
ID: 12345

SERVICES:
1. Consultation Fee - Dr. Smith    ₹ 800
2. Blood Test - CBC Complete       ₹ 1200  
3. X-Ray Chest                     ₹ 600
4. Medicine - Paracetamol 500mg    ₹ 150
5. Room Charges (1 day)            ₹ 2000

TOTAL: ₹ 4750
EOF

# Encode and test
base64 realistic_medical_bill.txt > encoded_bill.txt

curl -X POST https://endearing-prosperity-production.up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "file_content": "'$(cat encoded_bill.txt)'",
    "doc_id": "apollo_test_001",
    "user_id": "test_user_003",
    "language": "english",
    "state_code": "delhi",
    "insurance_type": "cghs",
    "file_format": "text"
  }' | jq .
```

## 🌐 **Web Interface Testing**

### 1. Open in Browser
Visit: `https://endearing-prosperity-production.up.railway.app/health`

### 2. API Documentation
If available: `https://endearing-prosperity-production.up.railway.app/docs`

## 🔧 **Testing Tools**

### Option 1: Using curl (Command Line)
```bash
# Simple health check
curl https://endearing-prosperity-production.up.railway.app/health

# With pretty JSON formatting
curl -s https://endearing-prosperity-production.up.railway.app/health | jq .
```

### Option 2: Using Postman
1. Import the base URL: `https://endearing-prosperity-production.up.railway.app`
2. Create requests for each endpoint
3. Test with sample data

### Option 3: Python Testing Script
```python
import requests
import json
import base64

# Base URL
BASE_URL = "https://endearing-prosperity-production.up.railway.app"

# Test health
health = requests.get(f"{BASE_URL}/health")
print("Health:", health.json())

# Test agents
agents = requests.get(f"{BASE_URL}/agents")
print("Agents:", agents.json())

# Test medical bill analysis
sample_bill = "Sample Medical Bill Content"
encoded_bill = base64.b64encode(sample_bill.encode()).decode()

analysis_data = {
    "file_content": encoded_bill,
    "doc_id": "python_test_001",
    "user_id": "python_user",
    "language": "english",
    "state_code": "delhi",
    "insurance_type": "cghs",
    "file_format": "text"
}

analysis = requests.post(f"{BASE_URL}/analyze", json=analysis_data)
print("Analysis:", analysis.json())
```

## 📊 **Expected Responses**

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": 1751878053.8977962,
  "uptime_seconds": 127.03798627853394,
  "version": "1.0.0",
  "components": {
    "redis": "healthy",
    "registry": "healthy",
    "medical_agent": "healthy"
  }
}
```

### Analysis Response
```json
{
  "success": true,
  "doc_id": "test_doc_001",
  "analysis_complete": true,
  "verdict": "ACCEPTABLE",
  "total_bill_amount": 4750.0,
  "total_overcharge": 0.0,
  "confidence_score": 0.85,
  "red_flags": [],
  "recommendations": ["Bill appears to be within normal ranges"],
  "processing_time_seconds": 2.3
}
```

## 🚨 **Troubleshooting**

### If you get 503 errors:
```bash
# Check if service is ready
curl -s https://endearing-prosperity-production.up.railway.app/health/readiness
```

### If you get timeout errors:
```bash
# Check if service is alive
curl -s https://endearing-prosperity-production.up.railway.app/health/liveness
```

### If analysis fails:
1. Check that `file_content` is properly base64 encoded
2. Ensure `doc_id` and `user_id` are provided
3. Verify the JSON structure is correct

## 📝 **Test Scenarios**

### 1. **Normal Bill Test**
- Use a standard medical bill
- Expect: `verdict: "ACCEPTABLE"`
- Expect: `confidence_score > 0.8`

### 2. **Overcharged Bill Test**
- Use a bill with inflated prices
- Expect: `verdict: "OVERCHARGED"`
- Expect: `red_flags` array with issues

### 3. **Load Test**
- Send multiple concurrent requests
- Monitor response times
- Check for any failures

## 🎯 **Success Criteria**

✅ **Health endpoint returns 200 OK**
✅ **Agents endpoint lists medical_bill_agent**
✅ **Analysis endpoint accepts and processes requests**
✅ **Response time < 10 seconds**
✅ **Confidence score > 0.7**
✅ **No 5xx server errors**

## 📞 **Support**

If you encounter issues:
1. Check the health endpoints first
2. Verify your request format matches the examples
3. Ensure all required fields are provided
4. Check that file_content is properly base64 encoded

**🎉 Your VivaranAI MedBillGuardAgent is ready for testing!** 