"""
Document Type Classifier - LLM-powered document classification and routing.

This tool analyzes raw OCR text to:
- Classify document types (medical bills, legal docs, financial statements, etc.)
- Determine routing requirements and agent capabilities needed
- Extract high-level document metadata
- Provide confidence scoring for classification decisions

The classifier makes routing decisions that the RouterAgent can use to select
the appropriate specialized domain agents.
"""

import asyncio
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger(__name__)


class DocumentType(str, Enum):
    """Supported document types."""
    MEDICAL_BILL = "medical_bill"
    PHARMACY_INVOICE = "pharmacy_invoice" 
    DIAGNOSTIC_REPORT = "diagnostic_report"
    INSURANCE_CLAIM = "insurance_claim"
    LEGAL_CONTRACT = "legal_contract"
    FINANCIAL_STATEMENT = "financial_statement"
    BUSINESS_INVOICE = "business_invoice"
    GOVERNMENT_FORM = "government_form"
    ACADEMIC_TRANSCRIPT = "academic_transcript"
    UNKNOWN = "unknown"


class RequiredCapability(str, Enum):
    """Agent capabilities required for different document types."""
    MEDICAL_ANALYSIS = "medical_analysis"
    FINANCIAL_ANALYSIS = "financial_analysis"  
    LEGAL_ANALYSIS = "legal_analysis"
    RATE_VALIDATION = "rate_validation"
    DUPLICATE_DETECTION = "duplicate_detection"
    COMPLIANCE_CHECK = "compliance_check"
    DATA_EXTRACTION = "data_extraction"
    DOCUMENT_VERIFICATION = "document_verification"


class ClassificationResult(BaseModel):
    """Result of document classification."""
    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    required_capabilities: List[RequiredCapability]
    suggested_agent: str
    reasoning: str
    metadata: Dict[str, Any] = {}
    
    # Domain-specific metadata
    medical_context: Optional[Dict[str, Any]] = None
    financial_context: Optional[Dict[str, Any]] = None
    legal_context: Optional[Dict[str, Any]] = None


class DocumentTypeClassifier:
    """
    LLM-powered document type classifier for intelligent routing.
    
    Uses advanced prompt engineering and structured outputs to classify
    documents and determine routing requirements.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.1
    ):
        """Initialize the document type classifier."""
        logger.info(f"🔑 DocumentTypeClassifier init: API key provided: {bool(openai_api_key)}, length: {len(openai_api_key) if openai_api_key else 0}")
        self.client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
        self.model = model
        self.temperature = temperature
        logger.info(f"🔑 DocumentTypeClassifier init complete: client created: {self.client is not None}")
        
        # Document type to agent capability mapping
        self.capability_mapping = {
            DocumentType.MEDICAL_BILL: [
                RequiredCapability.MEDICAL_ANALYSIS,
                RequiredCapability.RATE_VALIDATION,
                RequiredCapability.DUPLICATE_DETECTION
            ],
            DocumentType.PHARMACY_INVOICE: [
                RequiredCapability.MEDICAL_ANALYSIS,
                RequiredCapability.RATE_VALIDATION,
                RequiredCapability.COMPLIANCE_CHECK
            ],
            DocumentType.DIAGNOSTIC_REPORT: [
                RequiredCapability.MEDICAL_ANALYSIS,
                RequiredCapability.DOCUMENT_VERIFICATION
            ],
            DocumentType.INSURANCE_CLAIM: [
                RequiredCapability.MEDICAL_ANALYSIS,
                RequiredCapability.FINANCIAL_ANALYSIS,
                RequiredCapability.COMPLIANCE_CHECK
            ],
            DocumentType.LEGAL_CONTRACT: [
                RequiredCapability.LEGAL_ANALYSIS,
                RequiredCapability.COMPLIANCE_CHECK
            ],
            DocumentType.FINANCIAL_STATEMENT: [
                RequiredCapability.FINANCIAL_ANALYSIS,
                RequiredCapability.DATA_EXTRACTION
            ],
            DocumentType.BUSINESS_INVOICE: [
                RequiredCapability.FINANCIAL_ANALYSIS,
                RequiredCapability.RATE_VALIDATION
            ]
        }
        
        # Agent routing suggestions
        self.agent_routing = {
            DocumentType.MEDICAL_BILL: "medical_bill_agent",
            DocumentType.PHARMACY_INVOICE: "medical_bill_agent", 
            DocumentType.DIAGNOSTIC_REPORT: "medical_bill_agent",
            DocumentType.INSURANCE_CLAIM: "insurance_claim_agent",
            DocumentType.LEGAL_CONTRACT: "legal_document_agent",
            DocumentType.FINANCIAL_STATEMENT: "financial_analysis_agent",
            DocumentType.BUSINESS_INVOICE: "business_invoice_agent",
            DocumentType.GOVERNMENT_FORM: "government_form_agent",
            DocumentType.ACADEMIC_TRANSCRIPT: "academic_document_agent",
            DocumentType.UNKNOWN: "generic_document_agent"
        }
        
        logger.info("Initialized DocumentTypeClassifier")
    
    def _perform_heuristic_detection(self, raw_text: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform fast, heuristic-based document detection."""
        content_lower = raw_text.lower()
        filename = metadata.get("filename", "").lower()

        # Medical bill indicators
        medical_indicators = [
            "medical", "hospital", "clinic", "doctor", "patient", "diagnosis",
            "medicine", "prescription", "treatment", "bill", "invoice",
            "cghs", "esi", "insurance", "medicare", "mediclaim"
        ]

        if any(indicator in content_lower or indicator in filename for indicator in medical_indicators):
            return {
                "success": True,
                "document_type": DocumentType.MEDICAL_BILL.value,
                "confidence": 0.9,
                "required_capabilities": [RequiredCapability.MEDICAL_ANALYSIS.value, RequiredCapability.RATE_VALIDATION.value],
                "suggested_agent": "medical_bill_agent",
                "reasoning": "Heuristically detected based on common medical keywords.",
                "metadata": {"detection_method": "heuristic"}
            }
        return None

    async def __call__(
        self,
        raw_text: str,
        doc_id: str,
        pages: Optional[List[str]] = None,
        tables: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify document type and determine routing requirements.
        Uses a fast heuristic check first, then falls back to LLM.
        """
        # 1. Fast heuristic check
        heuristic_result = self._perform_heuristic_detection(raw_text, metadata or {})
        if heuristic_result:
            logger.info("Document type detected heuristically", doc_id=doc_id)
            return heuristic_result

        # 2. Fallback to LLM-based classification
        logger.info("Heuristic detection failed, falling back to LLM classification", doc_id=doc_id)
        
        try:
            logger.info(
                "Starting document classification",
                doc_id=doc_id,
                text_length=len(raw_text),
                has_tables=bool(tables),
                pages_count=len(pages) if pages else 0
            )
            
            # Classify the document
            classification = await self.classify_document(
                raw_text=raw_text,
                pages=pages,
                tables=tables,
                metadata=metadata or {}
            )
            
            result = {
                "success": True,
                "doc_id": doc_id,
                "document_type": classification.document_type.value,
                "confidence": classification.confidence,
                "required_capabilities": [cap.value for cap in classification.required_capabilities],
                "suggested_agent": classification.suggested_agent,
                "reasoning": classification.reasoning,
                "metadata": classification.metadata,
                "medical_context": classification.medical_context,
                "financial_context": classification.financial_context,
                "legal_context": classification.legal_context
            }
            
            logger.info(
                "Document classification completed",
                doc_id=doc_id,
                document_type=classification.document_type.value,
                confidence=classification.confidence,
                suggested_agent=classification.suggested_agent
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Document classification failed: {str(e)}"
            logger.error(error_msg, doc_id=doc_id, exc_info=True)
            
            return {
                "success": False,
                "doc_id": doc_id,
                "error": error_msg,
                "document_type": DocumentType.UNKNOWN.value,
                "confidence": 0.0,
                "required_capabilities": [],
                "suggested_agent": "generic_document_agent",
                "reasoning": "Classification failed, using fallback",
                "metadata": {}
            }
    
    async def classify_document(
        self,
        raw_text: str,
        pages: Optional[List[str]] = None,
        tables: Optional[List[Dict]] = None,
        metadata: Dict[str, Any] = None
    ) -> ClassificationResult:
        """
        Classify document using LLM analysis.
        
        Args:
            raw_text: Complete OCR extracted text
            pages: Page-by-page text breakdown
            tables: Extracted table data
            metadata: Additional document metadata
            
        Returns:
            ClassificationResult with all analysis
        """
        logger.info(f"🤖 DocumentTypeClassifier.classify_document called with text_length={len(raw_text)}, has_client={self.client is not None}")
        
        if not self.client:
            # Fallback to rule-based classification
            logger.warning("🚨 No OpenAI client available, using fallback classification")
            return self._fallback_classification(raw_text, metadata or {})
        
        try:
            # Prepare context for LLM
            analysis_context = self._prepare_analysis_context(raw_text, pages, tables, metadata)
            logger.info(f"🤖 Prepared analysis context: {len(analysis_context.get('text_sample', ''))} chars sample")
            
            # Create classification prompt
            prompt = self._create_classification_prompt(analysis_context)
            logger.info(f"🤖 Created classification prompt: {len(prompt)} chars")
            
            # Call LLM for classification
            logger.info(f"🤖 Calling OpenAI API with model {self.model}")
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            # Store raw LLM response for debugging/display
            raw_llm_response = response.choices[0].message.content
            logger.info(f"🤖 LLM Response received: {len(raw_llm_response)} chars: {raw_llm_response[:200]}...")
            
            # Parse LLM response
            classification_result = self._parse_llm_response(raw_llm_response)
            
            # Add raw response to result for transparency
            classification_result.metadata['raw_llm_response'] = raw_llm_response
            classification_result.metadata['llm_model'] = self.model
            classification_result.metadata['llm_temperature'] = self.temperature
            
            logger.info(f"🤖 Classification completed: {classification_result.document_type} with {classification_result.confidence:.2f} confidence")
            return classification_result
            
        except Exception as e:
            logger.error(f"🚨 LLM classification failed: {e}, falling back to rule-based")
            return self._fallback_classification(raw_text, metadata or {})
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for document classification."""
        return """You are an expert document classifier that analyzes OCR-extracted text to determine document types and routing requirements.

Your task is to:
1. Classify the document type based on content, structure, and terminology
2. Assess confidence level of classification
3. Determine required processing capabilities
4. Suggest appropriate agent routing
5. Extract domain-specific context and metadata
6. Provide clear reasoning for your decisions

Supported document types:
- medical_bill: Hospital bills, medical invoices with treatments/procedures
- pharmacy_invoice: Pharmacy receipts with medicines and prescriptions  
- diagnostic_report: Lab reports, test results, medical imaging reports
- insurance_claim: Insurance claim forms and related documents
- legal_contract: Legal agreements, contracts, terms documents
- financial_statement: Financial reports, accounting documents, balance sheets
- business_invoice: General business invoices and commercial bills
- government_form: Government forms, tax documents, official paperwork
- academic_transcript: Educational records, transcripts, certificates
- unknown: Documents that don't fit the above categories

Required capabilities map to specific agent functions:
- medical_analysis: Medical terminology analysis, procedure validation
- financial_analysis: Financial calculations, cost analysis  
- legal_analysis: Legal terminology and clause analysis
- rate_validation: Price/rate comparison against standards
- duplicate_detection: Finding duplicate entries or charges
- compliance_check: Regulatory compliance verification
- data_extraction: Structured data extraction
- document_verification: Authenticity and completeness verification

Respond with valid JSON containing all required fields."""
    
    def _create_classification_prompt(self, context: Dict[str, Any]) -> str:
        """Create the classification prompt with document context."""
        prompt = f"""Analyze this document and provide classification in JSON format:

Document Text (first 2000 chars):
{context['text_sample']}

Document Statistics:
- Total text length: {context['total_length']} characters
- Number of pages: {context['page_count']}
- Has tables: {context['has_tables']}
- Tables count: {context['table_count']}

Key Terms Found: {', '.join(context['key_terms'])}

Provide classification as JSON with this exact structure:
{{
    "document_type": "one of the supported types",
    "confidence": 0.95,
    "required_capabilities": ["list", "of", "capabilities"],
    "suggested_agent": "agent_name",
    "reasoning": "Clear explanation of why this classification was chosen",
    "metadata": {{
        "domain": "medical/financial/legal/etc",
        "complexity": "low/medium/high",
        "contains_sensitive_data": true/false,
        "estimated_processing_time": "seconds",
        "key_indicators": ["list", "of", "key", "terms", "that", "led", "to", "classification"]
    }},
    "medical_context": {{
        "hospital_name": "extracted if medical",
        "bill_amount": "total if found",
        "patient_info": "if medical document",
        "medical_procedures": ["list if medical"]
    }},
    "financial_context": {{
        "currency": "currency if financial",
        "total_amount": "amount if financial",
        "invoice_number": "if applicable",
        "vendor_info": "if business document"
    }},
    "legal_context": {{
        "contract_type": "if legal document",
        "parties_involved": ["if legal"],
        "jurisdiction": "if mentioned"
    }}
}}

Focus on accuracy and provide high confidence only when certain. Use key terminology, document structure, and data patterns to make decisions."""
        
        return prompt
    
    def _prepare_analysis_context(
        self,
        raw_text: str,
        pages: Optional[List[str]],
        tables: Optional[List[Dict]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare document context for LLM analysis."""
        # Extract key terms for quick classification hints
        key_terms = self._extract_key_terms(raw_text)
        
        return {
            "text_sample": raw_text[:2000],  # First 2000 chars for analysis
            "total_length": len(raw_text),
            "page_count": len(pages) if pages else 1,
            "has_tables": bool(tables),
            "table_count": len(tables) if tables else 0,
            "key_terms": key_terms,
            "metadata": metadata or {}
        }
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key classification terms from text."""
        text_lower = text.lower()
        
        key_terms_patterns = {
            "medical": ["hospital", "patient", "doctor", "treatment", "medicine", "diagnosis", "procedure", "consultation"],
            "financial": ["invoice", "amount", "total", "payment", "balance", "account", "transaction", "billing"],
            "legal": ["agreement", "contract", "terms", "conditions", "party", "clause", "jurisdiction", "liable"],
            "pharmacy": ["pharmacy", "prescription", "tablet", "capsule", "medicine", "drug", "mrp", "batch"],
            "insurance": ["claim", "policy", "coverage", "premium", "deductible", "beneficiary", "insured"]
        }
        
        found_terms = []
        for category, terms in key_terms_patterns.items():
            for term in terms:
                if term in text_lower:
                    found_terms.append(f"{category}:{term}")
        
        return found_terms[:10]  # Limit to most relevant terms
    
    def _parse_llm_response(self, response_content: str) -> ClassificationResult:
        """Parse LLM JSON response into ClassificationResult."""
        import json
        
        try:
            data = json.loads(response_content)
            
            # Map string values to enums
            doc_type = DocumentType(data.get("document_type", "unknown"))
            capabilities = [RequiredCapability(cap) for cap in data.get("required_capabilities", [])]
            
            return ClassificationResult(
                document_type=doc_type,
                confidence=data.get("confidence", 0.0),
                required_capabilities=capabilities,
                suggested_agent=data.get("suggested_agent", self.agent_routing.get(doc_type, "generic_document_agent")),
                reasoning=data.get("reasoning", "LLM classification"),
                metadata=data.get("metadata", {}),
                medical_context=data.get("medical_context"),
                financial_context=data.get("financial_context"),
                legal_context=data.get("legal_context")
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise ValueError(f"Invalid LLM response format: {e}")
    
    def _fallback_classification(self, text: str, metadata: Dict[str, Any]) -> ClassificationResult:
        """Fallback rule-based classification when LLM is unavailable."""
        text_lower = text.lower()
        
        # Simple rule-based classification
        if any(term in text_lower for term in ["hospital", "patient", "doctor", "consultation", "treatment"]):
            doc_type = DocumentType.MEDICAL_BILL
        elif any(term in text_lower for term in ["pharmacy", "medicine", "tablet", "prescription"]):
            doc_type = DocumentType.PHARMACY_INVOICE
        elif any(term in text_lower for term in ["laboratory", "test result", "diagnosis", "specimen"]):
            doc_type = DocumentType.DIAGNOSTIC_REPORT
        elif any(term in text_lower for term in ["invoice", "bill", "amount due", "payment"]):
            doc_type = DocumentType.BUSINESS_INVOICE
        else:
            doc_type = DocumentType.UNKNOWN
        
        capabilities = self.capability_mapping.get(doc_type, [RequiredCapability.DATA_EXTRACTION])
        suggested_agent = self.agent_routing.get(doc_type, "generic_document_agent")
        
        return ClassificationResult(
            document_type=doc_type,
            confidence=0.7,  # Lower confidence for rule-based
            required_capabilities=capabilities,
            suggested_agent=suggested_agent,
            reasoning=f"Rule-based classification identified key terms for {doc_type.value}",
            metadata={"classification_method": "rule_based", "fallback": True}
        ) 