"""
MedBillGuard Agent - Advanced Medical Bill Analysis System

This module provides comprehensive analysis of medical bills including:
- OCR text extraction and document processing
- Rate validation against government schemes (CGHS, ESI, NPPA)
- Duplicate detection and overcharge identification
- Prohibited item detection and compliance checking
- Confidence scoring and risk assessment
"""

import asyncio
import logging
import os
import tempfile
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
import structlog

from shared.schemas.schemas import (
    MedBillGuardResponse, 
    LineItem, 
    RedFlag, 
    Verdict, 
    LineItemType, 
    DocumentType, 
    Language
)
from shared.processors.document_processor import DocumentProcessor, process_document
from shared.tools.confidence_scorer import ConfidenceScorer
from shared.utils.cache_manager import cache_manager
from shared.tools.duplicate_detector import DuplicateDetector
from medbillguardagent.rate_validator import RateValidator
from medbillguardagent.prohibited_detector import ProhibitedDetector
from medbillguardagent.reference_data_loader import ReferenceDataLoader
from medbillguardagent.explanation_builder import ExplanationBuilder

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global service instances
reference_loader: Optional[ReferenceDataLoader] = None
rate_validator: Optional[RateValidator] = None
confidence_scorer: Optional[ConfidenceScorer] = None
duplicate_detector: Optional[DuplicateDetector] = None
prohibited_detector: Optional[ProhibitedDetector] = None

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Configure OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add OTEL exporters
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=14268,
)
console_exporter = ConsoleSpanExporter()

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import prometheus_client

# Prometheus metrics definitions
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
ANALYSIS_VERDICTS = Counter('analysis_verdicts_total', 'Analysis verdicts', ['verdict'])
RED_FLAGS_DETECTED = Counter('red_flags_detected_total', 'Red flags detected', ['flag_type', 'severity'])
PROCESSING_ERRORS = Counter('processing_errors_total', 'Processing errors', ['error_type'])
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')
CACHE_OPERATIONS = Counter('cache_operations_total', 'Cache operations', ['operation', 'result'])

# Rate limiter configuration
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379/2"  # Use different Redis DB for rate limiting
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting MedBillGuardAgent service")
    
    global reference_loader, rate_validator, confidence_scorer, duplicate_detector, prohibited_detector
    
    try:
        # Initialize cache manager
        await cache_manager.initialize()
        logger.info("Cache manager initialized")
        
        # Initialize reference data loader
        reference_loader = ReferenceDataLoader()
        await reference_loader.initialize()
        logger.info("Reference data loaded")
        
        # Initialize validators and detectors
        rate_validator = RateValidator(reference_loader)
        confidence_scorer = ConfidenceScorer()
        duplicate_detector = DuplicateDetector()
        prohibited_detector = ProhibitedDetector()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MedBillGuardAgent service")


# Create FastAPI app
app = FastAPI(
    title="MedBillGuardAgent",
    description="AI-powered medical bill analysis service for detecting overcharges and irregularities",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add rate limiting middleware
app.add_middleware(SlowAPIMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    """Add unique request ID to all responses."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect Prometheus metrics for requests."""
    method = request.method
    endpoint = request.url.path
    
    with ACTIVE_REQUESTS.track_inprogress():
        with REQUEST_LATENCY.labels(method=method, endpoint=endpoint).time():
            response = await call_next(request)
            
            # Record request metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=response.status_code
            ).inc()
            
            return response


def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, 'request_id', str(uuid.uuid4()))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with RFC 7807 format."""
    request_id = get_request_id(request)
    
    # Map status codes to problem types
    problem_types = {
        400: "https://tools.ietf.org/html/rfc7231#section-6.5.1",
        401: "https://tools.ietf.org/html/rfc7235#section-3.1",
        403: "https://tools.ietf.org/html/rfc7231#section-6.5.3",
        404: "https://tools.ietf.org/html/rfc7231#section-6.5.4",
        409: "https://tools.ietf.org/html/rfc7231#section-6.5.8",
        413: "https://tools.ietf.org/html/rfc7231#section-6.5.11",
        422: "https://tools.ietf.org/html/rfc4918#section-11.2",
        429: "https://tools.ietf.org/html/rfc6585#section-4",
        500: "https://tools.ietf.org/html/rfc7231#section-6.6.1"
    }
    
    error_response = ErrorResponse(
        type=problem_types.get(exc.status_code, "about:blank"),
        title=HTTPStatus(exc.status_code).phrase,
        status=exc.status_code,
        detail=str(exc.detail),
        instance=str(request.url),
        request_id=request_id
    )
    
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=str(exc.detail),
        request_id=request_id,
        url=str(request.url)
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(by_alias=True),
        headers={"Content-Type": "application/problem+json"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with RFC 7807 format."""
    request_id = get_request_id(request)
    
    error_response = ErrorResponse(
        type="https://tools.ietf.org/html/rfc7231#section-6.6.1",
        title="Internal Server Error",
        status=500,
        detail="An unexpected error occurred while processing your request",
        instance=str(request.url),
        request_id=request_id,
        context={
            "error_type": type(exc).__name__,
            "error_message": str(exc)
        }
    )
    
    logger.error(
        "Unhandled exception occurred",
        error_type=type(exc).__name__,
        error_message=str(exc),
        request_id=request_id,
        url=str(request.url),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(by_alias=True),
        headers={"Content-Type": "application/problem+json"}
    )


@app.get("/healthz")
@limiter.limit("100/minute")  # High limit for health checks
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/debug/example")
@limiter.limit("30/minute")  # More lenient for debug endpoint
async def debug_example():
    """
    Debug endpoint that processes a fixture file for testing.
    Should complete in under 10 seconds.
    """
    import os
    from pathlib import Path
    
    start_time = time.time()
    doc_id = f"debug_{uuid.uuid4().hex[:8]}"
    
    logger.info("Processing debug example", doc_id=doc_id)
    
    try:
        # Look for fixture file
        fixture_paths = [
            "fixtures/example.pdf",
            "fixtures/cghs_sample_bill.pdf", 
            "fixtures/pharmacy_invoice.pdf"
        ]
        
        fixture_file = None
        for path in fixture_paths:
            if os.path.exists(path):
                fixture_file = path
                break
        
        if not fixture_file:
            # Create a mock response if no fixture exists
            return {
                "doc_id": doc_id,
                "verdict": "ok",
                "message": "No fixture file found - returning mock response",
                "total_bill_amount": 2500.0,
                "total_overcharge_amount": 0.0,
                "confidence_score": 0.95,
                "red_flags": [],
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "next_steps": ["This is a mock response for testing purposes."]
            }
        
        # Read fixture file
        with open(fixture_file, "rb") as f:
            file_content = f.read()
        
        # Process the document
        document_processor = DocumentProcessor()
        extracted_doc = await document_processor.process_document(
            file_content, doc_id, Language.ENGLISH
        )
        
        # Quick analysis with minimal processing
        line_items = extracted_doc.line_items
        
        if line_items:
            items = [item.description for item in line_items]
            item_costs = {item.description: float(item.total_amount) for item in line_items}
            
            # Rate validation
            rate_matches = await rate_validator.validate_item_rates(items, item_costs)
            
            # Generate red flags
            red_flags = rate_validator.generate_red_flags(rate_matches)
            
            total_bill_amount = sum(float(item.total_amount) for item in line_items)
            total_overcharge = _calculate_total_overcharge(red_flags)
            verdict = _determine_verdict(red_flags, total_overcharge, total_bill_amount)
            explanation_markdown, explanation_ssml = build_explanation(verdict, red_flags, total_overcharge, total_bill_amount)
        else:
            red_flags = []
            total_bill_amount = 0.0
            total_overcharge = 0.0
            verdict = "ok"
            explanation_markdown, explanation_ssml = build_explanation(verdict, red_flags, total_overcharge, total_bill_amount)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        counselling_message = _build_counselling_message(verdict, red_flags, total_overcharge, total_bill_amount)
        
        result = {
            "doc_id": doc_id,
            "verdict": verdict,
            "fixture_file": fixture_file,
            "total_bill_amount": total_bill_amount,
            "total_overcharge_amount": total_overcharge,
            "confidence_score": 0.85,  # Mock confidence
            "red_flags": red_flags,
            "analysis_summary": {
                "items_analyzed": len(line_items) if line_items else 0,
                "rate_matches_found": len(rate_matches) if line_items else 0,
                "processing_time_ms": processing_time
            },
            "explanation_markdown": explanation_markdown,
            "explanation_ssml": explanation_ssml,
            "next_steps": _generate_next_steps(verdict, red_flags, total_overcharge),
            "counselling_message": counselling_message
        }
        
        logger.info(
            "Debug example completed",
            doc_id=doc_id,
            verdict=verdict,
            processing_time_ms=processing_time,
            fixture_file=fixture_file
        )
        
        return result
        
    except Exception as e:
        logger.error("Error in debug example", doc_id=doc_id, error=str(e), exc_info=True)
        return {
            "doc_id": doc_id,
            "error": str(e),
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "MedBillGuardAgent",
        "version": "1.0.0",
        "description": "AI-powered medical bill analysis and overcharge detection",
        "endpoints": {
            "/": "Service information",
            "/healthz": "Health check",
            "/analyze": "Main analysis endpoint",
            "/debug/example": "Debug endpoint with fixture file",
            "/api/docs": "API documentation"
        }
    }


@app.get("/api/docs")
async def api_docs():
    """API documentation endpoint."""
    return {
        "service": "MedBillGuardAgent API",
        "version": "1.0.0",
        "description": "AI-powered medical bill analysis and overcharge detection service",
        "base_url": "http://localhost:8000",
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Service information and available endpoints",
                "response": "Service metadata"
            },
            {
                "path": "/healthz",
                "method": "GET", 
                "description": "Health check endpoint",
                "response": "Health status"
            },
            {
                "path": "/analyze",
                "method": "POST",
                "description": "Main medical bill analysis endpoint",
                "parameters": {
                    "file": "Medical bill file (PDF, JPEG, PNG)",
                    "doc_id": "Optional document ID for idempotency",
                    "language": "Optional document language (default: ENGLISH)",
                    "state_code": "Optional state code for regional validation"
                },
                "response": "Complete analysis with overcharge detection"
            },
            {
                "path": "/debug/example",
                "method": "GET",
                "description": "Debug endpoint using fixture files",
                "response": "Quick analysis results for testing"
            }
        ],
        "schemas": {
            "MedBillGuardResponse": {
                "doc_id": "string",
                "verdict": "string (ok/warning/alert)",
                "total_bill_amount": "number",
                "total_overcharge_amount": "number", 
                "confidence_score": "number (0-1)",
                "red_flags": "array of detected issues",
                "next_steps": "array of recommended actions",
                "analysis_summary": "detailed analysis breakdown"
            },
            "ErrorResponse": {
                "error": "string",
                "detail": "string",
                "request_id": "string"
            }
        },
        "usage_examples": {
            "curl_analyze": """curl -X POST "http://localhost:8000/analyze" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@medical_bill.pdf" \\
  -F "state_code=DL" """,
            "python_analyze": """
import requests

with open('medical_bill.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/analyze',
        files={'file': f},
        data={'state_code': 'DL'}
    )
    result = response.json()
"""
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/analyze", response_model=MedBillGuardResponse)
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def analyze_medical_bill(
    request: Request,
    file: UploadFile = File(..., description="Medical bill document (PDF, JPEG, PNG)"),
    doc_id: Optional[str] = Form(None, description="Optional document ID for idempotency"),
    language: Language = Form(Language.ENGLISH, description="Document language"),
    state_code: Optional[str] = Form(None, description="State code for regional rate validation")
) -> MedBillGuardResponse:
    """
    Analyze a medical bill document for overcharges, duplicates, and prohibited items.
    
    This endpoint processes uploaded medical bills and returns detailed analysis including:
    - Rate validation against CGHS, ESI, NPPA, and state tariffs
    - Duplicate item detection
    - Prohibited item detection
    - Confidence scoring and red flag generation
    - Actionable recommendations
    """
    start_time = time.time()
    request_id = get_request_id(request)
    
    # Generate doc_id if not provided
    if not doc_id:
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    
    logger.info(
        "Starting medical bill analysis",
        doc_id=doc_id,
        filename=file.filename,
        content_type=file.content_type,
        language=language.value,
        state_code=state_code,
        request_id=request_id
    )
    
    try:
        # Validate file
        await _validate_upload_file(file)
        
        # Check for idempotency using doc_id
        cached_result = await _check_idempotency(doc_id)
        if cached_result:
            logger.info("Returning cached result for doc_id", doc_id=doc_id)
            return cached_result
        
        # Read file content
        file_content = await file.read()
        
        # Process document
        logger.info("Starting document processing", doc_id=doc_id)
        document_processor = DocumentProcessor()
        extracted_doc = await document_processor.process_document(
            file_content, doc_id, language
        )
        logger.info("Document processing completed", doc_id=doc_id, 
                   pages_processed=extracted_doc.processing_stats.pages_processed)
        
        # Extract line items for analysis
        line_items = extracted_doc.line_items
        if not line_items:
            logger.warning("No line items found in document", doc_id=doc_id)
            return _create_empty_response(doc_id, extracted_doc, start_time)
        
        # Prepare items and costs for validation
        items = [item.description for item in line_items]
        item_costs = {item.description: float(item.total_amount) for item in line_items}
        
        logger.info("Starting rate validation", doc_id=doc_id, item_count=len(items))
        
        # Rate validation
        rate_matches = await rate_validator.validate_item_rates(
            items, item_costs, state_code=state_code
        )
        
        # Duplicate detection
        logger.info("Starting duplicate detection", doc_id=doc_id)
        duplicates = duplicate_detector.detect_duplicates(line_items)
        
        # Prohibited item detection
        logger.info("Starting prohibited item detection", doc_id=doc_id)
        prohibited_items = prohibited_detector.detect_prohibited_items(line_items)
        
        # Generate red flags
        logger.info("Generating red flags", doc_id=doc_id)
        red_flags = rate_validator.generate_red_flags(rate_matches)
        
        # Add duplicate red flags
        for duplicate_group in duplicates:
            if len(duplicate_group.items) > 1:
                total_duplicate_cost = sum(item.total_amount for item in duplicate_group.items[1:])
                red_flags.append({
                    "type": "duplicate",
                    "severity": "warning",
                    "item": duplicate_group.normalized_description,
                    "reason": f"Duplicate item found {len(duplicate_group.items)} times",
                    "overcharge_amount": float(total_duplicate_cost),
                    "confidence": duplicate_group.confidence
                })
        
        # Add prohibited item red flags
        for prohibited_item in prohibited_items:
            red_flags.append({
                "type": "prohibited",
                "severity": "critical", 
                "item": prohibited_item.item.description,
                "reason": f"Prohibited item: {prohibited_item.reason}",
                "overcharge_amount": float(prohibited_item.item.total_amount),
                "confidence": prohibited_item.confidence
            })
        
        # Calculate totals
        total_bill_amount = sum(float(item.total_amount) for item in line_items)
        total_overcharge = _calculate_total_overcharge(red_flags)
        
        # Determine verdict
        verdict = _determine_verdict(red_flags, total_overcharge, total_bill_amount)
        
        # Calculate confidence score
        logger.info("Calculating confidence score", doc_id=doc_id)
        confidence_score = await confidence_scorer.calculate_confidence(
            extracted_doc, rate_matches, duplicates, prohibited_items
        )
        
        # Record business metrics
        ANALYSIS_VERDICTS.labels(verdict=verdict).inc()
        
        for flag in red_flags:
            RED_FLAGS_DETECTED.labels(
                flag_type=flag.get("type", "unknown"),
                severity=flag.get("severity", "unknown")
            ).inc()
        
        # Build response
        explanation_markdown, explanation_ssml = build_explanation(verdict, red_flags, total_overcharge, total_bill_amount)
        counselling_message = _build_counselling_message(verdict, red_flags, total_overcharge, total_bill_amount)
        response = MedBillGuardResponse(
            doc_id=doc_id,
            verdict=verdict,
            total_bill_amount=total_bill_amount,
            total_overcharge_amount=total_overcharge,
            confidence_score=confidence_score,
            red_flags=red_flags,
            analysis_summary={
                "items_analyzed": len(line_items),
                "rate_matches_found": len(rate_matches),
                "duplicates_detected": len(duplicates),
                "prohibited_items_found": len(prohibited_items),
                "state_validation_used": state_code is not None
            },
            processing_stats=extracted_doc.processing_stats,
            metadata=extracted_doc.metadata,
            explanation_markdown=explanation_markdown,
            explanation_ssml=explanation_ssml,
            next_steps=_generate_next_steps(verdict, red_flags, total_overcharge),
            counselling_message=counselling_message,
            latency_ms=int((time.time() - start_time) * 1000)
        )
        
        # Cache result for idempotency
        await _cache_result(doc_id, response)
        
        logger.info(
            "Medical bill analysis completed",
            doc_id=doc_id,
            verdict=verdict,
            total_overcharge=total_overcharge,
            confidence_score=confidence_score,
            latency_ms=response.latency_ms
        )
        
        return response
        
    except Exception as e:
        # Record error metrics
        PROCESSING_ERRORS.labels(error_type=type(e).__name__).inc()
        logger.error(
            "Error during medical bill analysis",
            doc_id=doc_id,
            error=str(e),
            error_type=type(e).__name__,
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze medical bill: {str(e)}"
        )


async def _validate_upload_file(file: UploadFile):
    """Validate uploaded file."""
    # Check file size (15MB limit)
    MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB
    
    # Read file to check size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Reset file pointer
    await file.seek(0)
    
    # Check file type
    allowed_types = [
        "application/pdf",
        "image/jpeg", 
        "image/jpg",
        "image/png"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed types: {allowed_types}"
        )


async def _check_idempotency(doc_id: str) -> Optional[MedBillGuardResponse]:
    """Check if result already exists for this doc_id."""
    try:
        cached_data = await cache_manager.get(f"result:{doc_id}")
        if cached_data:
            return MedBillGuardResponse(**cached_data)
    except Exception as e:
        logger.warning("Failed to check idempotency cache", doc_id=doc_id, error=str(e))
    return None


async def _cache_result(doc_id: str, response: MedBillGuardResponse):
    """Cache result for idempotency."""
    try:
        await cache_manager.set(
            f"result:{doc_id}", 
            response.dict(), 
            ttl=86400  # 24 hours
        )
    except Exception as e:
        logger.warning("Failed to cache result", doc_id=doc_id, error=str(e))


def _create_empty_response(doc_id: str, extracted_doc, start_time: float) -> MedBillGuardResponse:
    """Create response when no line items are found."""
    return MedBillGuardResponse(
        doc_id=doc_id,
        verdict="ok",
        total_bill_amount=0.0,
        total_overcharge_amount=0.0,
        confidence_score=0.0,
        red_flags=[],
        analysis_summary={
            "items_analyzed": 0,
            "rate_matches_found": 0,
            "duplicates_detected": 0,
            "prohibited_items_found": 0,
            "state_validation_used": False
        },
        processing_stats=extracted_doc.processing_stats,
        metadata=extracted_doc.metadata,
        explanation_markdown="",
        explanation_ssml="",
        next_steps=["No billable items found in the document. Please verify the document is a valid medical bill."],
        counselling_message="",
        latency_ms=int((time.time() - start_time) * 1000)
    )


def _determine_verdict(red_flags, total_overcharge, total_bill_amount):
    """
    Determine the verdict for the bill analysis.
    - 'ok': No red flags, overcharge < 1% of bill
    - 'warning': Minor overcharge (1-10%) or minor red flags
    - 'critical': Major overcharge (>10%) or critical red flags
    """
    if not red_flags or total_overcharge < 0.01 * (total_bill_amount or 1):
        return "ok"
    
    critical_flags = [f for f in red_flags if f.get("severity") == "critical" or f.get("type") == "prohibited"]
    if critical_flags or total_overcharge > 0.10 * (total_bill_amount or 1):
        return "critical"
    
    return "warning"


def _generate_next_steps(verdict, red_flags, total_overcharge):
    """
    Generate next-step suggestions based on verdict and red flags.
    """
    if verdict == "ok":
        return [
            "No action needed. Your bill appears reasonable.",
            "If you have questions, you may request a detailed breakdown from the hospital."
        ]
    if verdict == "warning":
        steps = [
            "Review the flagged items in your bill.",
            "Contact the hospital billing department for clarification on the flagged charges.",
        ]
        if total_overcharge > 0:
            steps.append("Request a refund or adjustment for any overcharged items.")
        return steps
    # critical
    steps = [
        "Significant over-charges or prohibited items detected.",
        "Contact the hospital billing department immediately and request a detailed explanation.",
        "If unresolved, escalate to your insurance provider or the state health ombudsman.",
        "You may be eligible for a refund for overcharged or prohibited items."
    ]
    return steps


# Helper to calculate total overcharge

def _calculate_total_overcharge(red_flags):
    """Sum all overcharge_amounts from red flags."""
    return sum(flag.get("overcharge_amount", 0) for flag in red_flags)


def _build_counselling_message(verdict, red_flags, total_overcharge, total_bill_amount):
    """
    Generate a plain-language, empathetic summary for the user.
    """
    if verdict == "ok":
        return (
            "Your bill looks good! We did not find any over-charges or issues. "
            "If you have any doubts, you can always ask your hospital for a detailed explanation."
        )
    if verdict == "warning":
        return (
            "We found some items in your bill that may be overcharged or need clarification. "
            "It's a good idea to review these with your hospital's billing team. "
            "Most issues can be resolved by asking for an itemized explanation."
        )
    # critical
    return (
        "We found serious issues in your bill, such as significant over-charges or prohibited items. "
        "Please contact your hospital immediately and ask for a detailed explanation. "
        "If you do not get a satisfactory response, you can escalate to your insurance provider or the health ombudsman."
    )


# Instrument FastAPI after app creation
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "medbillguardagent:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 