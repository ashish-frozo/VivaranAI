"""
MedBillGuardAgent - Simplified version for testing
AI-powered medical bill analysis service for detecting overcharges and irregularities
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
from http import HTTPStatus

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import json

# Simple logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances (simplified)
reference_data: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Simplified application lifespan manager."""
    logger.info("Starting MedBillGuardAgent service (simplified)")
    
    # Load minimal reference data for testing
    global reference_data
    reference_data = {
        "cghs_rates": {
            "Doctor Consultation (General Medicine)": 500,
            "Doctor Consultation (Cardiology)": 800,
            "Complete Blood Count (CBC)": 200,
            "Chest X-Ray": 300,
            "ECG": 150
        },
        "state_rates": {
            "DL": {
                "Doctor Consultation (General Medicine)": 600,
                "Complete Blood Count (CBC)": 250
            }
        }
    }
    
    logger.info("Simplified services initialized")
    yield
    logger.info("Shutting down MedBillGuardAgent service")

# Create FastAPI app
app = FastAPI(
    title="MedBillGuardAgent (Simplified)",
    description="AI-powered medical bill analysis service - Testing Version",
    version="1.0.0-test",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0-test"
    }

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "MedBillGuardAgent",
        "version": "1.0.0-test",
        "description": "AI-powered medical bill analysis service",
        "endpoints": {
            "health": "/healthz",
            "docs": "/docs",
            "analyze": "/analyze",
            "debug": "/debug/example"
        }
    }

@app.get("/debug/example")
async def debug_example():
    """Quick test endpoint with sample analysis."""
    logger.info("Processing debug example")
    
    # Simulate analysis of the sample bill
    sample_analysis = {
        "docId": f"debug-{int(time.time())}",
        "verdict": "warning",
        "totalBillAmount": 23550,
        "totalOverchargeAmount": 3100,
        "confidenceScore": 0.92,
        "redFlags": [
            {
                "item": "Doctor Consultation (General Medicine)",
                "reason": "Duplicate consultation detected",
                "billed": 2500,
                "max_allowed": 600,
                "overcharge_amount": 1900,
                "confidence": 0.95,
                "severity": "high"
            },
            {
                "item": "Complete Blood Count (CBC)",
                "reason": "3 duplicate tests detected",
                "billed": 800,
                "max_allowed": 250,
                "overcharge_amount": 1650,
                "confidence": 0.90,
                "severity": "high"
            }
        ],
        "counsellingMessage": "Your medical bill shows some concerning overcharges. Multiple duplicate tests and consultations have been detected, resulting in significant extra costs.",
        "nextSteps": [
            "Contact the hospital billing department to clarify the duplicate charges",
            "Request an itemized breakdown of all services provided",
            "Consider filing a complaint with the hospital administration",
            "Keep all documentation for potential insurance claims"
        ],
        "explanationMarkdown": "## Analysis Summary\n\nYour bill analysis shows **WARNING** level issues with total overcharges of ₹3,100 out of ₹23,550.\n\n### Issues Found:\n- Duplicate consultations: ₹1,900 overcharge\n- Duplicate CBC tests: ₹1,650 overcharge",
        "latencyMs": 1250,
        "processingStats": {
            "documentProcessingMs": 500,
            "rateValidationMs": 300,
            "duplicateDetectionMs": 250,
            "confidenceScoringMs": 100,
            "totalProcessingMs": 1250
        }
    }
    
    return sample_analysis

@app.get("/api/docs")
async def api_docs():
    """API documentation endpoint."""
    return {
        "service": "MedBillGuardAgent API",
        "version": "1.0.0-test",
        "description": "AI-powered medical bill analysis and overcharge detection",
        "endpoints": [
            {
                "path": "/healthz",
                "method": "GET",
                "description": "Health check endpoint"
            },
            {
                "path": "/debug/example",
                "method": "GET", 
                "description": "Quick test with sample data"
            },
            {
                "path": "/analyze",
                "method": "POST",
                "description": "Analyze uploaded medical bill",
                "parameters": {
                    "file": "Medical bill file (PDF, JPEG, PNG)",
                    "doc_id": "Optional document ID",
                    "language": "Document language (en, hi, bn, ta)",
                    "state_code": "State code for regional rates"
                }
            }
        ],
        "sample_usage": {
            "curl_health": "curl http://localhost:8000/healthz",
            "curl_debug": "curl http://localhost:8000/debug/example",
            "curl_analyze": "curl -X POST -F 'file=@bill.pdf' -F 'doc_id=test-123' http://localhost:8000/analyze"
        }
    }

@app.post("/analyze")
async def analyze_medical_bill(
    request: Request,
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    language: str = Form("en"),
    state_code: Optional[str] = Form(None)
):
    """Simplified analyze endpoint for testing."""
    start_time = time.time()
    
    if not doc_id:
        doc_id = f"upload-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    
    logger.info(f"Analyzing file: {file.filename}, doc_id: {doc_id}")
    
    # Validate file
    if file.size > 15 * 1024 * 1024:  # 15MB limit
        raise HTTPException(status_code=413, detail="File size too large (max 15MB)")
    
    allowed_types = ["application/pdf", "image/jpeg", "image/png", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    
    # Simulate processing
    await asyncio.sleep(0.5)  # Simulate processing time
    
    # Mock analysis based on filename or content
    if "sample" in file.filename.lower() or "bill" in file.filename.lower():
        # Return analysis similar to debug example but with file info
        analysis = {
            "docId": doc_id,
            "verdict": "warning",
            "totalBillAmount": 23550,
            "totalOverchargeAmount": 3100,
            "confidenceScore": 0.88,
            "redFlags": [
                {
                    "item": "Doctor Consultation (General Medicine)",
                    "reason": "Duplicate consultation detected in uploaded file",
                    "billed": 2500,
                    "max_allowed": 600,
                    "overcharge_amount": 1900,
                    "confidence": 0.95,
                    "severity": "high"
                },
                {
                    "item": "Complete Blood Count (CBC)",
                    "reason": "Multiple duplicate tests found",
                    "billed": 800,
                    "max_allowed": 250,
                    "overcharge_amount": 1200,
                    "confidence": 0.85,
                    "severity": "medium"
                }
            ],
            "counsellingMessage": f"Analysis of your uploaded file '{file.filename}' shows potential overcharges. Please review the detailed breakdown below.",
            "nextSteps": [
                "Review the detected duplicate charges with your healthcare provider",
                "Request clarification on the billing discrepancies",
                "Consider seeking a second opinion on the charges",
                "Document all communications for your records"
            ],
            "explanationMarkdown": f"## File Analysis: {file.filename}\n\nDetected **WARNING** level issues with overcharges totaling ₹3,100.\n\n### Key Findings:\n- File processed successfully\n- Multiple billing irregularities found\n- Recommended action required",
            "latencyMs": int((time.time() - start_time) * 1000),
            "processingStats": {
                "fileSize": file.size,
                "fileName": file.filename,
                "contentType": file.content_type,
                "language": language,
                "stateCode": state_code,
                "processingTimeMs": int((time.time() - start_time) * 1000)
            }
        }
    else:
        # Clean bill scenario
        analysis = {
            "docId": doc_id,
            "verdict": "ok",
            "totalBillAmount": 15000,
            "totalOverchargeAmount": 0,
            "confidenceScore": 0.95,
            "redFlags": [],
            "counsellingMessage": f"Good news! Your uploaded file '{file.filename}' appears to have reasonable charges with no significant overcharges detected.",
            "nextSteps": [
                "Your bill appears to be in order",
                "Keep this analysis for your records",
                "Continue to monitor future medical bills"
            ],
            "explanationMarkdown": f"## File Analysis: {file.filename}\n\n✅ **OK** - No significant issues detected.\n\nYour medical bill appears to have reasonable charges.",
            "latencyMs": int((time.time() - start_time) * 1000),
            "processingStats": {
                "fileSize": file.size,
                "fileName": file.filename,
                "contentType": file.content_type,
                "language": language,
                "stateCode": state_code,
                "processingTimeMs": int((time.time() - start_time) * 1000)
            }
        }
    
    return analysis

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 