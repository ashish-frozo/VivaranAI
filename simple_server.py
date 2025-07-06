#!/usr/bin/env python3
"""
Simplified MedBillGuardAgent Server for Testing
This version avoids Prometheus conflicts and focuses on the core functionality.
"""

import asyncio
import json
import logging
import time
import base64
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config.env_config import config, check_required_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the agent
try:
    from agents.medical_bill_agent import MedicalBillAgent
    logger.info("✅ Successfully imported MedicalBillAgent")
except ImportError as e:
    logger.error(f"❌ Failed to import MedicalBillAgent: {e}")
    sys.exit(1)

# Try to import enhanced router tools
try:
    from agents.tools.enhanced_router_agent import EnhancedRouterAgent
    from agents.tools.generic_ocr_tool import GenericOCRTool
    from agents.tools.document_type_classifier import DocumentTypeClassifier
    logger.info("✅ Successfully imported Enhanced Router tools")
    ENHANCED_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Enhanced Router tools not available: {e}")
    ENHANCED_AVAILABLE = False

# FastAPI app
app = FastAPI(
    title="MedBillGuardAgent API",
    description="AI-powered medical bill analysis and overcharge detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AnalysisRequest(BaseModel):
    file_content: str  # Base64 encoded
    filename: str
    language: str = "english"
    insurance_type: str = "cghs"
    state_code: str = None
    patient_id: str = None

class AnalysisResponse(BaseModel):
    success: bool
    analysis_id: str = None
    status: str = None
    processing_time_ms: int = None
    results: Dict[str, Any] = None
    verdict: str = None
    total_bill_amount: float = None
    total_overcharge: float = None
    confidence_score: float = None
    red_flags: list = None
    recommendations: list = None
    error: Optional[str] = None

class EnhancedAnalysisRequest(BaseModel):
    file_content: str  # Base64 encoded
    doc_id: str
    user_id: str
    language: str = "english"
    file_format: Optional[str] = None
    routing_strategy: str = "capability_based"
    priority: str = "normal"

class EnhancedAnalysisResponse(BaseModel):
    success: bool
    doc_id: str
    document_type: str
    processing_stages: Dict[str, Any]
    final_result: Dict[str, Any]
    total_processing_time_ms: int
    error: Optional[str] = None

# Global agent instance
agent = None
enhanced_router = None

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup"""
    global agent, enhanced_router
    try:
        logger.info("🚀 Initializing MedicalBillAgent...")
        
        # Validate configuration
        if not check_required_config():
            raise ValueError("Required configuration missing")
        
        # Get OpenAI API key from config
        api_key = config.openai_api_key
        
        # Initialize agent
        agent = MedicalBillAgent(openai_api_key=api_key)
        logger.info("✅ MedicalBillAgent initialized successfully")
        
        # Initialize enhanced router if available
        if ENHANCED_AVAILABLE:
            try:
                enhanced_router = EnhancedRouterAgent(
                    registry=None,  # Simple mode without registry
                    redis_url=None,  # Simple mode without Redis
                    openai_api_key=api_key
                )
                logger.info("✅ Enhanced Router initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Enhanced Router: {e}")
                enhanced_router = None
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        health = await agent.health_check()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "agent_status": health,
            "message": "MedBillGuardAgent is ready"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_medical_bill(request: AnalysisRequest):
    """Analyze a medical bill for overcharges and issues"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    start_time = time.time()
    analysis_id = f"analysis_{int(start_time)}_{hash(request.filename) % 10000}"
    
    try:
        logger.info(f"🔍 Starting analysis for {request.filename}")
        
        # Decode base64 file content
        try:
            file_bytes = base64.b64decode(request.file_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 content: {str(e)}")
        
        # Determine file format from filename
        file_format = None
        filename_lower = request.filename.lower()
        if filename_lower.endswith('.pdf'):
            file_format = 'pdf'
        elif filename_lower.endswith(('.jpg', '.jpeg')):
            file_format = 'jpg'
        elif filename_lower.endswith('.png'):
            file_format = 'png'
        elif filename_lower.endswith('.txt'):
            file_format = 'txt'
        
        logger.info(f"🔍 Processing file: {request.filename}, format: {file_format}, size: {len(file_bytes)} bytes")
        
        # Run the analysis
        result = await agent.analyze_medical_bill(
            file_content=file_bytes,
            doc_id=analysis_id,
            user_id="dashboard_user",
            language=request.language,
            state_code=request.state_code,
            insurance_type=request.insurance_type,
            file_format=file_format
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"✅ Analysis completed in {processing_time}ms")
        
        # Add debug information for OCR text and line items
        debug_info = {}
        if hasattr(result, 'results') and result.results:
            results = result.results
            if hasattr(results, 'processing_results') and results.processing_results:
                debug_info['debug_ocr_text'] = getattr(results.processing_results, 'raw_text', 'Not available')
                debug_info['debug_line_items'] = [
                    {
                        'description': item.description,
                        'total_amount': float(item.total_amount),
                        'quantity': item.quantity
                    }
                    for item in getattr(results.processing_results, 'line_items', [])
                ]
                # Also add raw OCR text to see what was actually read
                raw_text = getattr(results.processing_results, 'raw_text', 'Not available')
                logger.info(f"🔍 DEBUG - Raw OCR Text: {raw_text[:500]}...")
                logger.info(f"🔍 DEBUG - Extracted {len(debug_info.get('debug_line_items', []))} line items")
                for i, item in enumerate(debug_info.get('debug_line_items', [])[:10]):  # Show first 10
                    logger.info(f"🔍 DEBUG - Item {i+1}: {item['description']} = ₹{item['total_amount']}")
        
        # Format response with debug info
        response = AnalysisResponse(
            success=result.get("success", False),
            analysis_id=analysis_id,
            status="completed" if result.get("success") else "failed",
            processing_time_ms=processing_time,
            results={**result.get("results", {}), **debug_info},
            verdict=result.get("verdict", "unknown"),
            total_bill_amount=result.get("total_bill_amount", 0),
            total_overcharge=result.get("total_overcharge", 0),
            confidence_score=result.get("confidence_score", 0),
            red_flags=result.get("red_flags", []),
            recommendations=result.get("recommendations", []),
            error=result.get("error") if not result.get("success") else None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"❌ Analysis failed after {processing_time}ms: {e}")
        
        return AnalysisResponse(
            success=False,
            analysis_id=analysis_id,
            status="failed",
            processing_time_ms=processing_time,
            error=str(e)
        )

@app.post("/analyze-enhanced", response_model=EnhancedAnalysisResponse)
async def analyze_document_enhanced(request: EnhancedAnalysisRequest):
    """
    Enhanced document analysis using Generic OCR + Document Classification + Smart Routing.
    
    This endpoint demonstrates the new architecture:
    1. Generic OCR Tool extracts raw data from any document type
    2. Document Type Classifier determines document type and requirements  
    3. Smart routing to appropriate analysis based on document type
    """
    if not ENHANCED_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="Enhanced analysis not available. Missing required dependencies."
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"🧠 Starting enhanced analysis for doc: {request.doc_id}")
        
        # Decode base64 file content
        try:
            file_bytes = base64.b64decode(request.file_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 content: {str(e)}")
        
        logger.info(f"🔍 Processing enhanced analysis - doc_id: {request.doc_id}, size: {len(file_bytes)} bytes")
        
        # Use direct tool approach for demonstration
        # Step 1: Generic OCR
        ocr_tool = GenericOCRTool()
        ocr_result = await ocr_tool(
            file_content=file_bytes,
            doc_id=request.doc_id,
            language=request.language,
            file_format=request.file_format
        )
        
        ocr_stage = {
            "success": ocr_result.get("success", False),
            "raw_text": ocr_result.get("raw_text", ""),
            "text_preview": ocr_result.get("raw_text", "")[:500] + "..." if len(ocr_result.get("raw_text", "")) > 500 else ocr_result.get("raw_text", ""),
            "text_length": len(ocr_result.get("raw_text", "")),
            "pages": len(ocr_result.get("pages", [])),
            "tables": len(ocr_result.get("tables", [])),
            "processing_stats": ocr_result.get("processing_stats", {}),
            "strategy_used": ocr_result.get("processing_stats", {}).get("strategy_used", "unknown"),
            "confidence": ocr_result.get("processing_stats", {}).get("confidence", 0)
        }
        
        # Step 2: Document Classification
        api_key = config.openai_api_key
        logger.info(f"🔑 Server: Creating DocumentTypeClassifier with API key: {bool(api_key)}, length: {len(api_key) if api_key else 0}")
        classifier = DocumentTypeClassifier(openai_api_key=api_key)
        classification_result = await classifier(
            raw_text=ocr_result.get("raw_text", ""),
            doc_id=request.doc_id,
            pages=ocr_result.get("pages", []),
            tables=ocr_result.get("tables", [])
        )
        
        classification_stage = {
            "success": classification_result.get("success", False),
            "document_type": classification_result.get("document_type", "unknown"),
            "confidence": classification_result.get("confidence", 0.0),
            "suggested_agent": classification_result.get("suggested_agent", "unknown"),
            "required_capabilities": classification_result.get("required_capabilities", []),
            "reasoning": classification_result.get("reasoning", ""),
            "llm_output": classification_result.get("metadata", {}).get("raw_llm_response", ""),
            "llm_model": classification_result.get("metadata", {}).get("llm_model", ""),
            "llm_temperature": classification_result.get("metadata", {}).get("llm_temperature", 0)
        }
        
        # Step 3: Domain Analysis (route to medical agent for medical documents)
        domain_stage = {"success": False, "agent_id": "none"}
        final_result = {}
        
        if (classification_result.get("document_type") in ["medical_bill", "pharmacy_invoice", "diagnostic_report"] 
            and agent is not None):
            
            # Route to medical bill agent for medical documents
            try:
                medical_result = await agent.analyze_medical_bill(
                    file_content=file_bytes,
                    doc_id=request.doc_id,
                    user_id=request.user_id,
                    language=request.language,
                    state_code=None,
                    insurance_type="cghs",
                    file_format=request.file_format
                )
                
                domain_stage = {
                    "success": medical_result.get("success", False),
                    "agent_id": "medical_bill_agent"
                }
                
                final_result = {
                    "verdict": medical_result.get("verdict", "unknown"),
                    "total_bill_amount": medical_result.get("total_bill_amount", 0),
                    "total_overcharge": medical_result.get("total_overcharge", 0),
                    "confidence_score": medical_result.get("confidence_score", 0),
                    "red_flags": medical_result.get("red_flags", []),
                    "recommendations": medical_result.get("recommendations", [])
                }
                
            except Exception as e:
                logger.error(f"Medical analysis failed: {e}")
                domain_stage = {"success": False, "agent_id": "medical_bill_agent", "error": str(e)}
                final_result = {"verdict": "error", "error": f"Medical analysis failed: {str(e)}"}
        else:
            # For non-medical documents, provide basic analysis
            final_result = {
                "verdict": "info",
                "document_type": classification_result.get("document_type", "unknown"),
                "message": f"Document classified as {classification_result.get('document_type', 'unknown')}. Specialized analysis not yet implemented.",
                "confidence_score": classification_result.get("confidence", 0) * 100
            }
            domain_stage = {"success": True, "agent_id": "generic_classifier"}
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"✅ Enhanced analysis completed in {processing_time}ms")
        
        # Format enhanced response
        response = EnhancedAnalysisResponse(
            success=True,
            doc_id=request.doc_id,
            document_type=classification_result.get("document_type", "unknown"),
            processing_stages={
                "ocr_extraction": ocr_stage,
                "document_classification": classification_stage,
                "domain_analysis": domain_stage
            },
            final_result=final_result,
            total_processing_time_ms=processing_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"❌ Enhanced analysis failed after {processing_time}ms: {e}")
        
        return EnhancedAnalysisResponse(
            success=False,
            doc_id=request.doc_id,
            document_type="unknown",
            processing_stages={},
            final_result={"error": str(e)},
            total_processing_time_ms=processing_time,
            error=str(e)
        )

@app.get("/debug/last-analysis")
async def debug_last_analysis():
    """Debug endpoint to show detailed OCR and parsing results"""
    return {
        "message": "Check backend_debug.log for detailed OCR and line item extraction",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze", 
            "docs": "/docs"
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MedBillGuardAgent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    # Check configuration
    if not check_required_config():
        print("💡 To get started:")
        print("   1. Copy env.example to .env: cp env.example .env")
        print("   2. Edit .env and add your OpenAI API key")
        print("   3. Run the server again")
        sys.exit(1)
    
    print("🚀 Starting MedBillGuardAgent Simple Server")
    print("=" * 50)
    print(f"📱 Frontend: http://localhost:8000/dashboard.html")
    print(f"🔧 API: http://localhost:8001")
    print(f"📚 Docs: http://localhost:8001/docs")
    print("=" * 50)
    
    uvicorn.run(
        "simple_server:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        access_log=True,
        log_level=config.log_level.lower()
    ) 