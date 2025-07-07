"""
Pydantic schemas for MedBillGuardAgent API.

This module defines the request/response models for the MedBillGuard service,
following the contract-first approach specified in the engineering ruleset.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal

from pydantic import BaseModel, Field, HttpUrl, validator, model_validator
from pydantic.config import ConfigDict


class DocumentType(str, Enum):
    """Supported document types for analysis."""
    HOSPITAL_BILL = "hospital_bill"
    PHARMACY_INVOICE = "pharmacy_invoice"
    DIAGNOSTIC_REPORT = "diagnostic_report"
    INSURANCE_CLAIM = "insurance_claim"
    UNKNOWN = "unknown"


class Language(str, Enum):
    """Supported languages for OCR and analysis."""
    ENGLISH = "en"
    HINDI = "hi"
    BENGALI = "bn"
    TAMIL = "ta"


class Priority(str, Enum):
    """Processing priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Verdict(str, Enum):
    """Analysis verdict categories."""
    OK = "ok"                    # No issues found
    WARNING = "warning"          # Minor over-charges or issues
    CRITICAL = "critical"        # Major over-charges or fraud indicators


class LineItemType(str, Enum):
    """Types of line items found in medical bills."""
    CONSULTATION = "consultation"
    PROCEDURE = "procedure"
    DIAGNOSTIC = "diagnostic"
    MEDICATION = "medication"
    ROOM_CHARGE = "room_charge"
    EQUIPMENT = "equipment"
    CONSUMABLE = "consumable"
    SERVICE = "service"
    TAX = "tax"
    OTHER = "other"


class RedFlag(BaseModel):
    """A single red flag or over-charge detected in the bill."""
    
    item: str = Field(
        ..., 
        description="Name of the charged item/service",
        min_length=1,
        max_length=200,
        example="HRCT Scan"
    )
    
    item_type: LineItemType = Field(
        default=LineItemType.OTHER,
        description="Category of the line item"
    )
    
    quantity: Optional[int] = Field(
        default=1,
        description="Quantity charged",
        ge=0
    )
    
    unit_cost: Optional[Decimal] = Field(
        default=None,
        description="Unit cost in rupees",
        ge=0
    )
    
    billed: Decimal = Field(
        ...,
        description="Amount billed in rupees",
        ge=0,
        example=4500.00
    )
    
    max_allowed: Optional[Decimal] = Field(
        default=None,
        description="Maximum allowed rate (CGHS/ESI) in rupees",
        ge=0,
        example=2500.00
    )
    
    overcharge_amount: Optional[Decimal] = Field(
        default=None,
        description="Over-charge amount in rupees",
        ge=0
    )
    
    overcharge_pct: Optional[float] = Field(
        default=None,
        description="Over-charge percentage",
        ge=0,
        le=1000,  # Allow up to 1000% over-charge
        example=80.0
    )
    
    confidence: float = Field(
        ...,
        description="Confidence score for this red flag (0.0-1.0)",
        ge=0.0,
        le=1.0,
        example=0.94
    )
    
    source: str = Field(
        default="analysis",
        description="Source of the validation (cghs, esi, nppa, rule)",
        example="cghs"
    )
    
    reason: str = Field(
        ...,
        description="Human-readable reason for flagging",
        min_length=1,
        max_length=500,
        example="Rate exceeds CGHS 2023 tariff by 80%"
    )
    
    is_duplicate: bool = Field(
        default=False,
        description="Whether this item appears multiple times"
    )
    
    is_prohibited: bool = Field(
        default=False,
        description="Whether this item is in prohibited list"
    )

    @model_validator(mode='before')
    @classmethod
    def calculate_overcharge(cls, values):
        """Calculate over-charge amount and percentage if not provided."""
        if isinstance(values, dict):
            billed = values.get('billed')
            max_allowed = values.get('max_allowed')
            
            if billed and max_allowed and max_allowed > 0:
                if values.get('overcharge_amount') is None:
                    values['overcharge_amount'] = billed - max_allowed
                
                if values.get('overcharge_pct') is None:
                    values['overcharge_pct'] = float((billed - max_allowed) / max_allowed * 100)
        
        return values


class NextStep(BaseModel):
    """Suggested next action for the user."""
    
    action: str = Field(
        ...,
        description="Action type",
        example="download_refund_letter"
    )
    
    title: str = Field(
        ...,
        description="User-friendly action title",
        example="Download Refund Request Letter"
    )
    
    description: str = Field(
        ...,
        description="Detailed description of the action",
        example="Get a pre-filled refund request letter to submit to hospital administration"
    )
    
    url: Optional[HttpUrl] = Field(
        default=None,
        description="URL for the action (if applicable)"
    )
    
    priority: Priority = Field(
        default=Priority.NORMAL,
        description="Priority level of this action"
    )


class ProcessingStats(BaseModel):
    """Statistics about the document processing."""
    
    pages_processed: int = Field(
        ...,
        description="Number of pages processed",
        ge=1
    )
    
    ocr_confidence: float = Field(
        ...,
        description="Average OCR confidence score",
        ge=0.0,
        le=1.0
    )
    
    text_extracted_chars: int = Field(
        ...,
        description="Number of characters extracted",
        ge=0
    )
    
    tables_found: int = Field(
        default=0,
        description="Number of tables detected",
        ge=0
    )
    
    line_items_found: int = Field(
        default=0,
        description="Number of line items parsed",
        ge=0
    )


class LineItem(BaseModel):
    """A single line item from a medical bill."""
    
    description: str = Field(
        ...,
        description="Description of the item/service",
        min_length=1,
        max_length=300
    )
    
    quantity: int = Field(
        default=1,
        description="Quantity of the item",
        ge=1
    )
    
    unit_cost: Optional[Decimal] = Field(
        default=None,
        description="Unit cost in rupees",
        ge=0
    )
    
    total_amount: Decimal = Field(
        ...,
        description="Total amount for this line item in rupees",
        ge=0
    )
    
    item_type: LineItemType = Field(
        default=LineItemType.OTHER,
        description="Category of the line item"
    )
    
    date: Optional[datetime] = Field(
        default=None,
        description="Date when the service was provided"
    )
    
    department: Optional[str] = Field(
        default=None,
        description="Hospital department",
        max_length=100
    )
    
    doctor: Optional[str] = Field(
        default=None,
        description="Doctor who provided the service",
        max_length=100
    )


class MedBillGuardRequest(BaseModel):
    """Request schema for medical bill analysis."""
    
    doc_id: str = Field(
        ...,
        description="Unique document identifier",
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_-]+$',
        example="bill-001"
    )
    
    user_id: str = Field(
        ...,
        description="User identifier",
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_-]+$',
        example="user-123"
    )
    
    s3_url: HttpUrl = Field(
        ...,
        description="S3 URL of the document to analyze",
        example="s3://vivaranai-docs/bills/hospital-bill.pdf"
    )
    
    doc_type: DocumentType = Field(
        default=DocumentType.HOSPITAL_BILL,
        description="Type of document being analyzed"
    )
    
    language: Language = Field(
        default=Language.ENGLISH,
        description="Primary language of the document"
    )
    
    priority: Priority = Field(
        default=Priority.NORMAL,
        description="Processing priority"
    )
    
    patient_state: Optional[str] = Field(
        default=None,
        description="Patient's state for state-specific rate validation",
        min_length=2,
        max_length=2,
        pattern=r'^[A-Z]{2}$',
        example="DL"
    )
    
    hospital_type: Optional[str] = Field(
        default=None,
        description="Type of hospital (government, private, trust)",
        example="private"
    )
    
    insurance_type: Optional[str] = Field(
        default=None,
        description="Insurance type (cghs, esi, private, none)",
        example="cghs"
    )
    
    callback_url: Optional[HttpUrl] = Field(
        default=None,
        description="Webhook URL for async processing completion"
    )
    
    metadata: Optional[Dict[str, Union[str, int, float]]] = Field(
        default=None,
        description="Additional metadata for processing"
    )

    @validator('s3_url')
    def validate_s3_url(cls, v):
        """Ensure S3 URL is valid and accessible."""
        url_str = str(v)
        if not url_str.startswith('s3://'):
            raise ValueError('URL must be a valid S3 URL starting with s3://')
        return v


class MedBillGuardResponse(BaseModel):
    """Complete response for medical bill analysis - React-friendly JSON envelope."""
    doc_id: str = Field(..., description="Document identifier", alias="docId")
    verdict: str = Field(..., description="Overall verdict: ok, warning, critical")
    total_bill_amount: float = Field(..., description="Total bill amount in ₹", alias="totalBillAmount")
    total_overcharge_amount: float = Field(..., description="Total overcharge detected in ₹", alias="totalOverchargeAmount")
    confidence_score: float = Field(..., description="Analysis confidence (0-1)", alias="confidenceScore")
    red_flags: List[RedFlag] = Field(default_factory=list, description="List of detected issues", alias="redFlags")
    line_items: List[LineItem] = Field(default_factory=list, description="Extracted line items", alias="lineItems")
    processing_stats: Optional[Dict[str, Any]] = Field(None, description="Processing statistics", alias="processingStats")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    explanation_markdown: str = Field(default="", description="Markdown explanation", alias="explanationMarkdown")
    explanation_ssml: str = Field(default="", description="SSML explanation", alias="explanationSsml")
    counselling_message: str = Field(default="", description="Plain language counselling", alias="counsellingMessage")
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps", alias="nextSteps")
    latency_ms: int = Field(..., description="Processing latency in milliseconds", alias="latencyMs")
    
    # React-friendly configuration
    model_config = ConfigDict(
        populate_by_name=True,  # Allow both snake_case and camelCase
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: float
        },
        extra="forbid"
    )


class HealthCheckResponse(BaseModel):
    """Health check endpoint response."""
    
    status: str = Field(
        default="ok",
        description="Service health status"
    )
    
    version: str = Field(
        default="1.0.0",
        description="Service version"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    
    dependencies: Optional[Dict[str, str]] = Field(
        default=None,
        description="Status of external dependencies"
    )


class ErrorResponse(BaseModel):
    """RFC 7807 compliant error response."""
    type: str = Field(..., description="URI identifying the problem type")
    title: str = Field(..., description="Human-readable summary of the problem")
    status: int = Field(..., description="HTTP status code")
    detail: Optional[str] = Field(None, description="Human-readable explanation specific to this occurrence")
    instance: Optional[str] = Field(None, description="URI identifying the specific occurrence")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking", alias="requestId")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the error occurred")
    
    # Additional context fields
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed validation errors")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    ) 