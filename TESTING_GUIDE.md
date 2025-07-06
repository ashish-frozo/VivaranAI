# 🧪 MedBillGuardAgent Testing Guide

This guide covers all the ways to test the MedBillGuardAgent AI micro-service for detecting overcharges in Indian hospital bills.

## 🚀 Quick Start

### 1. **Web Dashboard (Recommended)**
The easiest way to test the product with a visual interface:

```bash
# Terminal 1: Start the API server
python -m uvicorn medbillguardagent:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start the dashboard
cd frontend
python serve.py

# Open browser: http://localhost:3000
```

**Features:**
- ✅ File upload (PDF, JPEG, PNG)
- ✅ Real-time API status monitoring
- ✅ Visual results with overcharge details
- ✅ Quick test with sample data
- ✅ Multi-language and state selection

### 2. **API Testing with cURL**

```bash
# Health check
curl http://localhost:8000/healthz

# Quick test (no file upload)
curl http://localhost:8000/debug/example

# Full analysis with file upload
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@fixtures/sample_bill.txt" \
  -F "doc_id=test-123" \
  -F "language=en" \
  -F "state_code=DL"

# API documentation
curl http://localhost:8000/api/docs
```

### 3. **Python Test Script**

```bash
python test_api.py
```

## 📊 Testing Scenarios

### **Scenario 1: Duplicate Detection**
Upload a bill with duplicate charges (like our sample bill with 4 CBC tests):
- **Expected**: Detects 3 duplicate CBC tests, flags ₹1,500 overcharge
- **Verdict**: WARNING or CRITICAL

### **Scenario 2: Overpriced Services**
Test with consultation charges above market rates:
- **Expected**: Flags overpriced consultations
- **Verdict**: WARNING or CRITICAL

### **Scenario 3: Clean Bill**
Upload a bill with reasonable charges:
- **Expected**: No red flags, minimal overcharge
- **Verdict**: OK

### **Scenario 4: Multi-language Testing**
Test with Hindi/Bengali/Tamil medical bills:
- **Expected**: Proper OCR extraction and analysis
- **Verdict**: Based on charges detected

## 🛠️ Advanced Testing

### **Performance Testing**
```bash
# Load testing with Locust
poetry run locust --host=http://localhost:8000 --users=10 --spawn-rate=2

# Unit tests with coverage
poetry run pytest --cov=medbillguardagent --cov-report=html

# Performance benchmarks
poetry run pytest tests/test_performance.py -v
```

### **Docker Testing**
```bash
# Build and run with Docker
docker-compose up --build

# Test containerized service
curl http://localhost:8000/healthz
```

### **Kubernetes Testing**
```bash
# Deploy to local cluster
kubectl apply -f k8s/

# Port forward and test
kubectl port-forward service/medbillguardagent 8000:80
curl http://localhost:8000/healthz
```

## 📝 Test Data

### **Sample Bills Available:**
- `fixtures/sample_bill.txt` - Text bill with duplicates
- Upload your own PDF/image bills via dashboard

### **Expected Results for Sample Bill:**
```json
{
  "verdict": "warning",
  "totalBillAmount": 23550,
  "totalOverchargeAmount": 3100,
  "redFlags": [
    {
      "item": "Doctor Consultation (General Medicine)",
      "reason": "Duplicate consultation detected",
      "overcharge_amount": 2500,
      "confidence": 0.95
    },
    {
      "item": "Complete Blood Count (CBC)", 
      "reason": "3 duplicate tests detected",
      "overcharge_amount": 1500,
      "confidence": 0.90
    }
  ]
}
```

## 🔍 What to Look For

### **✅ Successful Test Indicators:**
- API status shows "Healthy" in dashboard
- Quick test returns analysis results
- File uploads process without errors
- Duplicate charges are detected
- Overcharge amounts are calculated
- Confidence scores are reasonable (>0.7)
- Processing time < 10 seconds

### **❌ Issues to Watch For:**
- API status shows "Offline"
- File upload errors (size/format)
- OCR extraction failures
- Missing red flags on obvious duplicates
- Unrealistic overcharge calculations
- Low confidence scores (<0.5)

## 🐛 Troubleshooting

### **Common Issues:**

1. **API Server Won't Start**
   ```bash
   # Check dependencies
   poetry install
   
   # Check Redis connection
   redis-cli ping
   ```

2. **Dashboard Shows API Offline**
   ```bash
   # Verify server is running
   curl http://localhost:8000/healthz
   
   # Check CORS headers
   curl -H "Origin: http://localhost:3000" http://localhost:8000/healthz
   ```

3. **File Upload Fails**
   - Check file size (<15MB)
   - Verify file format (PDF, JPEG, PNG)
   - Ensure proper form data encoding

4. **OCR Extraction Issues**
   ```bash
   # Install Tesseract
   brew install tesseract  # macOS
   sudo apt-get install tesseract-ocr  # Ubuntu
   
   # Check language packs
   tesseract --list-langs
   ```

## 📈 Monitoring & Metrics

### **Built-in Metrics:**
- Processing latency (target: <5s)
- Confidence scores (target: >0.8)
- Error rates (target: <1%)
- Memory usage
- Request throughput

### **Access Metrics:**
```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Health status
curl http://localhost:8000/healthz
```

## 🎯 Success Criteria

A successful test should demonstrate:

1. **Functional Requirements:**
   - ✅ Document processing (PDF/image → text)
   - ✅ Duplicate detection (exact + similar)
   - ✅ Rate validation against reference data
   - ✅ Overcharge calculation
   - ✅ Confidence scoring

2. **Performance Requirements:**
   - ✅ <10s processing time
   - ✅ >90% accuracy on test cases
   - ✅ Handles 15MB files
   - ✅ Multi-language support

3. **User Experience:**
   - ✅ Clear verdict (ok/warning/critical)
   - ✅ Actionable recommendations
   - ✅ Detailed explanations
   - ✅ Error handling

## 🔗 Additional Resources

- **API Documentation**: http://localhost:8000/docs (when server is running)
- **Swagger UI**: http://localhost:8000/redoc
- **Source Code**: `/medbillguardagent/` directory
- **Test Suite**: `/tests/` directory
- **Configuration**: `/config/` directory

---

**Need Help?** Check the logs, run the test suite, or review the API documentation for more details. 