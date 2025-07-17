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
from fastapi import FastAPI, HTTPException, Depends
from database.models import get_async_db
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys

# --- Load environment variables from .env if present ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    print('Loaded OPENAI_API_KEY:', os.environ.get('OPENAI_API_KEY'))
except ImportError:
    print('⚠️  python-dotenv not installed. Install with: pip install python-dotenv')

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

class ChatRequest(BaseModel):
    user_id: str
    doc_id: str = None  # Optional: if not provided, use most recent bill
    message: str
    conversation_history: Optional[list] = []

class ChatResponse(BaseModel):
    success: bool
    doc_id: str
    message: str
    timestamp: float


from typing import List
from pydantic import BaseModel

class BillSummary(BaseModel):
    doc_id: str
    filename: str
    created_at: str
    status: str
    analysis_type: str
    total_amount: float = 0
    suspected_overcharges: float = 0
    confidence_level: float = 0

class BillsListResponse(BaseModel):
    bills: List[BillSummary]

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

from database.bill_chat_context import save_bill_analysis, get_user_bills, get_bill_by_id

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bill_context(request: ChatRequest, db: AsyncSession = Depends(get_async_db)):
    """
    Chat with the assistant about a bill. If doc_id is omitted, use most recent bill.
    """
    # Determine which bill to use for context
    bill = None
    if request.doc_id:
        bill = await get_bill_by_id(db, request.doc_id)
    else:
        bills = await get_user_bills(db, user_id=request.user_id, limit=1)
        bill = bills[0] if bills else None
    if not bill:
        return ChatResponse(success=False, doc_id=request.doc_id or '', message="No bill found for context.", timestamp=time.time())

    # Extract bill information for context
    bill_info = getattr(bill, 'raw_analysis', None) or {}
    structured_results = getattr(bill, 'structured_results', None) or {}
    
    # Get bill amount from multiple possible locations
    bill_amount = 0
    for attr_name in ['total_amount', 'total_bill_amount']:
        amount = getattr(bill, attr_name, None)
        if amount:
            if isinstance(amount, str):
                try:
                    bill_amount = float(amount)
                    break
                except (ValueError, TypeError):
                    pass
            else:
                bill_amount = amount
                break
    
    # If still no amount, try to get it from structured results
    if bill_amount == 0 and structured_results:
        if isinstance(structured_results, str):
            try:
                structured_results = json.loads(structured_results)
            except json.JSONDecodeError:
                structured_results = {}
        bill_amount = structured_results.get('total_amount', 0) or structured_results.get('total_bill_amount', 0)
    
    # Create a detailed bill summary for context
    bill_summary = f"Bill: {bill.filename}, Amount: ₹{bill_amount}, Status: {bill.status}"
    
    # Add detailed bill information if available
    bill_details = ""
    
    # Process raw_analysis for detailed information
    if bill_info:
        # Handle string serialization if needed
        if isinstance(bill_info, str):
            try:
                bill_info = json.loads(bill_info)
            except json.JSONDecodeError:
                bill_info = {"error": "Could not parse bill info"}
        
        # Look for line items in multiple possible locations
        line_items = []
        
        # Check in raw_analysis directly
        if isinstance(bill_info, dict):
            line_items = bill_info.get('line_items', [])
            
            # Check in final_result if present
            if not line_items and 'final_result' in bill_info:
                final_result = bill_info.get('final_result', {})
                line_items = final_result.get('line_items', [])
            
            # Check in processing_stages if present
            if not line_items and 'processing_stages' in bill_info:
                processing_stages = bill_info.get('processing_stages', {})
                domain_analysis = processing_stages.get('domain_analysis', {})
                if domain_analysis:
                    line_items = domain_analysis.get('line_items', [])
            
            # Check in debug_line_items if present
            if not line_items and 'debug_line_items' in bill_info:
                line_items = bill_info.get('debug_line_items', [])
            
            # Check in results if present
            if not line_items and 'results' in bill_info:
                results = bill_info.get('results', {})
                line_items = results.get('line_items', []) or results.get('debug_line_items', [])
        
        # Format line items if found
        if line_items:
            bill_details += "\n\nLine Items/Medicines:\n"
            for i, item in enumerate(line_items, 1):
                if isinstance(item, dict):
                    name = item.get('description', '') or item.get('name', '') or item.get('item_name', '') or "Unknown item"
                    qty = item.get('quantity', 1)
                    price = item.get('price', 0) or item.get('total_amount', 0) or item.get('amount', 0)
                    bill_details += f"{i}. {name} - Qty: {qty}, Price: ₹{price}\n"
        
        # If no line items found, try to extract from OCR text
        if not line_items:
            ocr_text = ""
            # Try to find OCR text in various locations
            if isinstance(bill_info, dict):
                # Check direct locations
                ocr_text = bill_info.get('raw_text', '') or bill_info.get('text', '')
                
                # Check in processing_stages
                if not ocr_text and 'processing_stages' in bill_info:
                    stages = bill_info.get('processing_stages', {})
                    ocr_extraction = stages.get('ocr_extraction', {})
                    ocr_text = ocr_extraction.get('raw_text', '')
                
                # Check in results
                if not ocr_text and 'results' in bill_info:
                    results = bill_info.get('results', {})
                    ocr_text = results.get('debug_ocr_text', '') or results.get('raw_text', '')
            
            if ocr_text:
                # Add a sample of the OCR text to provide context
                bill_details += "\n\nExtracted Text Sample:\n"
                # Take first 300 chars as a sample
                text_sample = ocr_text[:300] + "..." if len(ocr_text) > 300 else ocr_text
                bill_details += text_sample
    
    # Send to OpenAI for processing
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create prompt with bill context including full AI analysis
        
        # Add AI analysis results if available
        ai_analysis = ""
        medicines_list = ""
        
        # First, try to extract medicines from structured_results
        if structured_results:
            if isinstance(structured_results, str):
                try:
                    structured_results = json.loads(structured_results)
                except json.JSONDecodeError:
                    structured_results = {}
            
            # Look for line_items in structured_results (from AI analysis)
            line_items = structured_results.get('line_items', [])
            if not line_items:
                # Also check for line_items in raw_analysis
                if isinstance(bill_info, dict):
                    # Try to find AI analysis response in raw_analysis
                    ai_response = None
                    
                    # Check for AI response in domain_analysis
                    if 'processing_stages' in bill_info:
                        stages = bill_info.get('processing_stages', {})
                        domain_analysis = stages.get('domain_analysis', {})
                        ai_response = domain_analysis.get('ai_response', None)
                    
                    # Check for AI response directly in raw_analysis
                    if not ai_response:
                        ai_response = bill_info.get('ai_response', None)
                    
                    # Check in final_result
                    if not ai_response and 'final_result' in bill_info:
                        final_result = bill_info.get('final_result', {})
                        ai_response = final_result.get('ai_response', None)
                    
                    # If we found an AI response, try to parse it
                    if ai_response:
                        if isinstance(ai_response, str):
                            try:
                                ai_response = json.loads(ai_response)
                            except json.JSONDecodeError:
                                ai_response = {}
                        
                        if isinstance(ai_response, dict):
                            line_items = ai_response.get('line_items', [])
            
            # If we found line_items, format them for the medicines list
            if line_items:
                medicines_list = "\n\nMedicines/Items in this bill:\n"
                for i, item in enumerate(line_items, 1):
                    if isinstance(item, dict):
                        name = item.get('description', '') or item.get('name', '') or "Unknown item"
                        qty = item.get('quantity', 1)
                        price = item.get('amount', 0) or item.get('price', 0) or item.get('total_amount', 0)
                        medicines_list += f"{i}. {name} - Qty: {qty}, Price: ₹{price}\n"
            
            # Format the AI analysis results
            ai_analysis += "\n\nAI Analysis Results:\n"
            
            # Add verdict if available
            verdict = structured_results.get('verdict', '')
            if verdict:
                ai_analysis += f"Verdict: {verdict}\n"
            
            # Add overcharge information if available
            overcharge = structured_results.get('total_overcharge', 0) or structured_results.get('estimated_overcharge', 0)
            if overcharge:
                ai_analysis += f"Suspected Overcharge: ₹{overcharge}\n"
            
            # Add confidence score if available
            confidence = structured_results.get('confidence_score', 0) or structured_results.get('confidence', 0)
            if confidence:
                ai_analysis += f"Confidence Score: {confidence:.2f}\n"
            
            # Add analysis notes if available
            notes = structured_results.get('analysis_notes', '')
            if notes:
                ai_analysis += f"\nAnalysis Notes: {notes}\n"
            
            # Add red flags if available
            red_flags = structured_results.get('red_flags', [])
            if red_flags:
                ai_analysis += "\nRed Flags:\n"
                for i, flag in enumerate(red_flags, 1):
                    ai_analysis += f"{i}. {flag}\n"
            
            # Add recommendations if available
            recommendations = structured_results.get('recommendations', [])
            if recommendations:
                ai_analysis += "\nRecommendations:\n"
                for i, rec in enumerate(recommendations, 1):
                    ai_analysis += f"{i}. {rec}\n"
            
            # Add detailed line item analysis if available
            line_item_analysis = structured_results.get('line_item_analysis', [])
            if line_item_analysis:
                ai_analysis += "\nLine Item Analysis:\n"
                for i, item in enumerate(line_item_analysis, 1):
                    if isinstance(item, dict):
                        name = item.get('description', '') or item.get('name', '') or "Item"
                        status = item.get('status', '') or item.get('issue', '')
                        reason = item.get('reason', '') or item.get('details', '')
                        ai_analysis += f"{i}. {name}: {status} - {reason}\n"
        
        # Create the full prompt with all available context
        prompt = f"""You are a helpful medical bill assistant. The user is asking about their medical bill.\n\n
            Bill Summary: {bill_summary}\n{medicines_list}\n{bill_details}\n{ai_analysis}\n\n
            User Question: {request.message}\n\n
            Please provide a helpful response about this medical bill, addressing the user's question directly.
            If you don't know something specific, be honest about it.
            Focus on answering the specific question asked by the user using the information provided.
            When asked about medicines or items in the bill, always refer to the Medicines/Items section if available."""
        
        # Log the prompt for debugging
        logger.info(f"Chat prompt for bill {bill.id}: {prompt[:200]}...")
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use a faster model for chat to reduce latency
            messages=[
                {"role": "system", "content": "You are a helpful medical bill assistant that provides accurate information about medical bills. Answer questions directly and concisely based on the bill information provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract response
        assistant_message = response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        # Fallback response if API call fails
        assistant_message = f"I'm sorry, I couldn't process your question about the bill due to a technical issue. Here's what I know about your bill: {bill_summary}"

    return ChatResponse(success=True, doc_id=str(bill.id), message=assistant_message, timestamp=time.time())

@app.get("/bills", response_model=BillsListResponse)
async def list_user_bills(user_id: str, db: AsyncSession = Depends(get_async_db)):
    """
    List all bill analyses for a user (summarized).
    """
    bills = await get_user_bills(db, user_id=user_id, limit=50)
    summaries = []
    for bill in bills:
        summaries.append(BillSummary(
            doc_id=str(bill.id),
            filename=bill.filename,
            created_at=bill.created_at if isinstance(bill.created_at, str) else bill.created_at.isoformat() if bill.created_at else "",
            status=bill.status,
            analysis_type=bill.analysis_type,
            total_amount=float(bill.total_amount or 0),
            suspected_overcharges=float(bill.suspected_overcharges or 0),
            confidence_level=float(bill.confidence_level or 0)
        ))
    return BillsListResponse(bills=summaries)

@app.post("/analyze-enhanced", response_model=EnhancedAnalysisResponse)
async def analyze_document_enhanced(request: EnhancedAnalysisRequest, db: AsyncSession = Depends(get_async_db)):
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
                # Build task_data for the agent with pre-processed OCR and classification results
                task_data = {
                    "doc_id": request.doc_id,
                    "user_id": request.user_id,
                    "raw_text": ocr_result.get("raw_text", ""),
                    "line_items": [],  # You may add line item extraction logic if available
                    "ocr_stats": ocr_result.get("processing_stats", {}),
                    "metadata": {
                        "state_code": None,
                        "insurance_type": "cghs"
                    },
                    "file_format": request.file_format,
                    "language": request.language
                }
                from agents.base_agent import AgentContext
                context = AgentContext(
                    doc_id=request.doc_id,
                    user_id=request.user_id,
                    correlation_id=f"medical_bill_{request.doc_id}",
                    model_hint=None,
                    start_time=time.time(),
                    metadata={"task_type": "medical_bill_analysis"}
                )
                medical_result = await agent.process_task(context, task_data=task_data)
                
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

        # Persist analysis result to DB
        try:
            # Compute file_hash, file_size, content_type for persistence (mocked for now)
            file_hash = "mocked_hash"  # TODO: Compute real hash if needed
            file_size = len(file_bytes)
            content_type = request.file_format or "application/octet-stream"
            analysis_type = classification_result.get("document_type", "medical_bill")
            await save_bill_analysis(
                session=db,
                user_id=request.user_id,
                doc_id=request.doc_id,
                filename=getattr(request, 'filename', f"doc_{request.doc_id}"),
                file_hash=file_hash,
                file_size=file_size,
                content_type=content_type,
                analysis_type=analysis_type,
                raw_analysis=response.dict(),
                structured_results=final_result,
                status="completed"
            )
            logger.info(f"✅ Bill analysis persisted to DB for doc_id={request.doc_id}")
        except Exception as db_exc:
            logger.error(f"❌ Failed to persist bill analysis to DB: {db_exc}")

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
    
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="VivaranAI MedBillGuardAgent Simple Server")

    # Allow CORS for local frontend and production dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "https://endearing-prosperity-production.up.railway.app",
            "*"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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