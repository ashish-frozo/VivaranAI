"""
Generic OCR + Router Architecture Demo

This demo shows how the new architecture would work:
1. Generic OCR Tool extracts raw text from any document
2. Document Type Classifier determines document type using LLM
3. Router Agent makes intelligent routing decisions  
4. Specialized agents handle domain-specific analysis

This is much more scalable than the current tightly-coupled approach.
"""

import asyncio
import json
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# 1. GENERIC OCR TOOL (Domain-agnostic)
# =============================================================================

class GenericOCRTool:
    """Generic OCR tool that works with any document type."""
    
    async def process_document(self, file_content: bytes, doc_id: str, language: str = "english") -> Dict[str, Any]:
        """Extract raw text and structure from document."""
        
        # Simulate OCR processing
        await asyncio.sleep(0.5)  # OCR processing time
        
        # Mock extracted text based on document type (in real implementation, this would be actual OCR)
        if b"APOLLO" in file_content or b"HOSPITAL" in file_content:
            raw_text = """
            APOLLO HOSPITALS
            Patient: John Doe
            Date: 2024-01-15
            
            CHARGES:
            Consultation Fee - Dr. Smith    Rs. 800
            Blood Test - CBC               Rs. 450  
            X-Ray Chest                    Rs. 300
            
            TOTAL AMOUNT: Rs. 1550
            """
        elif b"CONTRACT" in file_content or b"AGREEMENT" in file_content:
            raw_text = """
            SERVICE AGREEMENT
            
            This agreement is entered into between Party A and Party B
            effective January 1, 2024.
            
            TERMS AND CONDITIONS:
            1. Service delivery within 30 days
            2. Payment due within 15 days
            3. Liability limited to contract value
            
            Total Contract Value: $50,000
            """
        elif b"INVOICE" in file_content or b"BILLING" in file_content:
            raw_text = """
            BUSINESS INVOICE
            Invoice #: INV-2024-001
            Date: 2024-01-15
            
            ITEMS:
            Software License (Annual)       $1,200
            Support Services                  $300
            Training Sessions                 $500
            
            SUBTOTAL: $2,000
            TAX (18%): $360
            TOTAL: $2,360
            """
        else:
            raw_text = "Generic document content extracted via OCR..."
        
        return {
            "success": True,
            "doc_id": doc_id,
            "raw_text": raw_text.strip(),
            "pages": [raw_text.strip()],  # Single page for demo
            "tables": [],  # Would contain extracted tables
            "language_detected": language,
            "processing_stats": {
                "ocr_confidence": 0.92,
                "pages_processed": 1,
                "text_extracted_chars": len(raw_text),
                "processing_time_ms": 500
            },
            "metadata": {
                "file_format": "pdf",  # Would be detected
                "total_pages": 1
            }
        }


# =============================================================================
# 2. DOCUMENT TYPE CLASSIFIER (LLM-powered)
# =============================================================================

class DocumentType(str, Enum):
    MEDICAL_BILL = "medical_bill"
    LEGAL_CONTRACT = "legal_contract"
    BUSINESS_INVOICE = "business_invoice"
    INSURANCE_CLAIM = "insurance_claim"
    UNKNOWN = "unknown"


class DocumentTypeClassifier:
    """LLM-powered document type classifier."""
    
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm  # For demo, we'll use rule-based classification
    
    async def classify_document(self, raw_text: str, doc_id: str) -> Dict[str, Any]:
        """Classify document type and determine routing requirements."""
        
        # Simulate LLM processing time
        await asyncio.sleep(0.2)
        
        text_lower = raw_text.lower()
        
        # Rule-based classification (in real implementation, this would be LLM-powered)
        if any(term in text_lower for term in ["hospital", "patient", "doctor", "consultation", "medical"]):
            return {
                "success": True,
                "doc_id": doc_id,
                "document_type": DocumentType.MEDICAL_BILL.value,
                "confidence": 0.94,
                "required_capabilities": ["medical_analysis", "rate_validation", "duplicate_detection"],
                "suggested_agent": "medical_bill_agent",
                "reasoning": "Document contains medical terminology, patient info, and medical procedures/charges",
                "metadata": {
                    "domain": "healthcare",
                    "complexity": "medium",
                    "contains_sensitive_data": True,
                    "key_indicators": ["hospital", "patient", "consultation", "charges"]
                }
            }
        
        elif any(term in text_lower for term in ["contract", "agreement", "terms", "conditions", "party"]):
            return {
                "success": True,
                "doc_id": doc_id,
                "document_type": DocumentType.LEGAL_CONTRACT.value,
                "confidence": 0.91,
                "required_capabilities": ["legal_analysis", "compliance_check"],
                "suggested_agent": "legal_document_agent",
                "reasoning": "Document contains legal terminology, contract clauses, and agreement terms",
                "metadata": {
                    "domain": "legal",
                    "complexity": "high",
                    "contains_sensitive_data": True,
                    "key_indicators": ["agreement", "terms", "conditions", "contract"]
                }
            }
        
        elif any(term in text_lower for term in ["invoice", "billing", "payment", "total", "tax"]):
            return {
                "success": True,
                "doc_id": doc_id,
                "document_type": DocumentType.BUSINESS_INVOICE.value,
                "confidence": 0.88,
                "required_capabilities": ["financial_analysis", "data_extraction"],
                "suggested_agent": "business_invoice_agent",
                "reasoning": "Document contains invoicing terminology, itemized charges, and payment information",
                "metadata": {
                    "domain": "financial",
                    "complexity": "low",
                    "contains_sensitive_data": False,
                    "key_indicators": ["invoice", "billing", "total", "payment"]
                }
            }
        
        else:
            return {
                "success": True,
                "doc_id": doc_id,
                "document_type": DocumentType.UNKNOWN.value,
                "confidence": 0.60,
                "required_capabilities": ["data_extraction"],
                "suggested_agent": "generic_document_agent",
                "reasoning": "Document type could not be determined with high confidence",
                "metadata": {
                    "domain": "unknown",
                    "complexity": "unknown",
                    "contains_sensitive_data": False,
                    "key_indicators": []
                }
            }


# =============================================================================
# 3. ROUTER AGENT (Intelligent routing)
# =============================================================================

class RouterAgent:
    """Intelligent router that selects appropriate agents based on document classification."""
    
    def __init__(self):
        # Mock agent registry
        self.available_agents = {
            "medical_bill_agent": {
                "capabilities": ["medical_analysis", "rate_validation", "duplicate_detection"],
                "load": 0.3,
                "success_rate": 0.95,
                "avg_response_time_ms": 2000
            },
            "legal_document_agent": {
                "capabilities": ["legal_analysis", "compliance_check"],
                "load": 0.1,
                "success_rate": 0.92,
                "avg_response_time_ms": 3000
            },
            "business_invoice_agent": {
                "capabilities": ["financial_analysis", "data_extraction"],
                "load": 0.5,
                "success_rate": 0.97,
                "avg_response_time_ms": 1500
            },
            "generic_document_agent": {
                "capabilities": ["data_extraction"],
                "load": 0.2,
                "success_rate": 0.88,
                "avg_response_time_ms": 1000
            }
        }
    
    async def make_routing_decision(
        self, 
        classification_result: Dict[str, Any],
        ocr_quality: float
    ) -> Dict[str, Any]:
        """Make intelligent routing decision based on classification and quality."""
        
        # Simulate routing decision time
        await asyncio.sleep(0.1)
        
        suggested_agent = classification_result.get("suggested_agent", "generic_document_agent")
        required_capabilities = classification_result.get("required_capabilities", [])
        
        # Find best agent based on capabilities, load, and quality
        best_agent = suggested_agent
        confidence = 0.9
        
        # If OCR quality is low, prefer more reliable agents
        if ocr_quality < 0.8 and suggested_agent in self.available_agents:
            agent_info = self.available_agents[suggested_agent]
            if agent_info["success_rate"] < 0.9:
                # Look for more reliable fallback
                for agent_id, info in self.available_agents.items():
                    if (info["success_rate"] > 0.9 and 
                        any(cap in info["capabilities"] for cap in required_capabilities)):
                        best_agent = agent_id
                        confidence = 0.85
                        break
        
        agent_info = self.available_agents.get(best_agent, {})
        
        return {
            "success": True,
            "selected_agent": best_agent,
            "confidence": confidence,
            "reasoning": f"Selected {best_agent} based on required capabilities: {required_capabilities}",
            "agent_metadata": {
                "capabilities": agent_info.get("capabilities", []),
                "current_load": agent_info.get("load", 0.0),
                "success_rate": agent_info.get("success_rate", 0.0),
                "estimated_response_time_ms": agent_info.get("avg_response_time_ms", 1000)
            },
            "routing_strategy": "capability_based_with_quality_adjustment"
        }


# =============================================================================
# 4. SPECIALIZED DOMAIN AGENTS
# =============================================================================

class MedicalBillAgent:
    """Specialized agent for medical bill analysis."""
    
    async def process_task(self, ocr_data: Dict, classification_data: Dict) -> Dict[str, Any]:
        """Process medical bill with domain-specific analysis."""
        
        # Simulate medical analysis processing
        await asyncio.sleep(1.0)
        
        raw_text = ocr_data.get("raw_text", "")
        
        # Mock medical bill analysis
        return {
            "success": True,
            "analysis_complete": True,
            "document_type": "medical_bill",
            "verdict": "potential_overcharge",
            "total_bill_amount": 1550.0,
            "total_overcharge": 150.0,
            "confidence_score": 0.92,
            "red_flags": [
                {
                    "type": "rate_validation",
                    "description": "Consultation fee Rs. 800 exceeds CGHS rate of Rs. 650",
                    "overcharge_amount": 150.0,
                    "confidence": 0.95
                }
            ],
            "line_items": [
                {"item": "Consultation Fee", "amount": 800, "standard_rate": 650, "overcharge": 150},
                {"item": "Blood Test - CBC", "amount": 450, "standard_rate": 400, "overcharge": 50},
                {"item": "X-Ray Chest", "amount": 300, "standard_rate": 300, "overcharge": 0}
            ],
            "recommendations": [
                "Request explanation for consultation fee above CGHS rates",
                "Consider getting second opinion on charges"
            ],
            "agent_used": "medical_bill_agent",
            "processing_metadata": {
                "medical_patterns_detected": ["consultation", "blood test", "x-ray"],
                "rate_validation_performed": True,
                "duplicate_detection_performed": True
            }
        }


class LegalDocumentAgent:
    """Specialized agent for legal document analysis."""
    
    async def process_task(self, ocr_data: Dict, classification_data: Dict) -> Dict[str, Any]:
        """Process legal document with domain-specific analysis."""
        
        await asyncio.sleep(1.5)
        
        return {
            "success": True,
            "analysis_complete": True,
            "document_type": "legal_contract",
            "contract_type": "service_agreement",
            "risk_assessment": "medium",
            "key_terms": [
                {"term": "Service delivery", "timeline": "30 days", "risk": "low"},
                {"term": "Payment terms", "timeline": "15 days", "risk": "medium"},
                {"term": "Liability clause", "scope": "limited", "risk": "low"}
            ],
            "compliance_check": {
                "jurisdiction_identified": False,
                "standard_clauses_present": True,
                "potential_issues": ["No clear jurisdiction specified"]
            },
            "recommendations": [
                "Add jurisdiction clause for legal clarity",
                "Consider adding dispute resolution mechanism"
            ],
            "agent_used": "legal_document_agent"
        }


class BusinessInvoiceAgent:
    """Specialized agent for business invoice analysis."""
    
    async def process_task(self, ocr_data: Dict, classification_data: Dict) -> Dict[str, Any]:
        """Process business invoice with domain-specific analysis."""
        
        await asyncio.sleep(0.8)
        
        return {
            "success": True,
            "analysis_complete": True,
            "document_type": "business_invoice",
            "invoice_validation": {
                "invoice_number": "INV-2024-001",
                "total_amount": 2360.0,
                "tax_calculation_correct": True,
                "line_items_valid": True
            },
            "financial_analysis": {
                "subtotal": 2000.0,
                "tax_rate": 0.18,
                "tax_amount": 360.0,
                "total": 2360.0
            },
            "recommendations": [
                "Invoice appears accurate and complete",
                "Payment terms should be clarified"
            ],
            "agent_used": "business_invoice_agent"
        }


# =============================================================================
# 5. ENHANCED ROUTER (Orchestrates entire workflow)
# =============================================================================

class EnhancedRouter:
    """Enhanced router that orchestrates the complete workflow."""
    
    def __init__(self):
        self.ocr_tool = GenericOCRTool()
        self.document_classifier = DocumentTypeClassifier()
        self.router_agent = RouterAgent()
        
        # Domain agents
        self.domain_agents = {
            "medical_bill_agent": MedicalBillAgent(),
            "legal_document_agent": LegalDocumentAgent(),
            "business_invoice_agent": BusinessInvoiceAgent()
        }
    
    async def process_document_complete(
        self, 
        file_content: bytes, 
        doc_id: str, 
        language: str = "english"
    ) -> Dict[str, Any]:
        """Complete document processing workflow."""
        
        print(f"\n🔄 Starting complete document processing for {doc_id}")
        
        try:
            # Stage 1: OCR Extraction
            print("📄 Stage 1: OCR Extraction...")
            ocr_result = await self.ocr_tool.process_document(file_content, doc_id, language)
            print(f"✅ OCR completed - {ocr_result['processing_stats']['text_extracted_chars']} chars extracted")
            
            # Stage 2: Document Classification
            print("🏷️  Stage 2: Document Classification...")
            classification_result = await self.document_classifier.classify_document(
                ocr_result["raw_text"], doc_id
            )
            print(f"✅ Classified as: {classification_result['document_type']} (confidence: {classification_result['confidence']:.2f})")
            
            # Stage 3: Routing Decision
            print("🎯 Stage 3: Routing Decision...")
            routing_result = await self.router_agent.make_routing_decision(
                classification_result, 
                ocr_result["processing_stats"]["ocr_confidence"]
            )
            print(f"✅ Routed to: {routing_result['selected_agent']}")
            
            # Stage 4: Domain Analysis
            print("🔬 Stage 4: Domain Analysis...")
            selected_agent_id = routing_result["selected_agent"]
            
            if selected_agent_id in self.domain_agents:
                domain_agent = self.domain_agents[selected_agent_id]
                domain_result = await domain_agent.process_task(ocr_result, classification_result)
                print(f"✅ Domain analysis completed by {selected_agent_id}")
            else:
                domain_result = {
                    "success": True,
                    "message": f"Generic processing completed for {classification_result['document_type']}",
                    "agent_used": "generic_fallback"
                }
                print(f"⚠️  Used generic fallback processing")
            
            # Stage 5: Compile Results
            print("📋 Stage 5: Compiling Final Results...")
            
            final_result = {
                "success": True,
                "doc_id": doc_id,
                "processing_pipeline": {
                    "ocr_extraction": {
                        "success": ocr_result["success"],
                        "confidence": ocr_result["processing_stats"]["ocr_confidence"],
                        "processing_time_ms": ocr_result["processing_stats"]["processing_time_ms"]
                    },
                    "document_classification": {
                        "document_type": classification_result["document_type"],
                        "confidence": classification_result["confidence"],
                        "reasoning": classification_result["reasoning"]
                    },
                    "routing_decision": {
                        "selected_agent": routing_result["selected_agent"],
                        "confidence": routing_result["confidence"],
                        "reasoning": routing_result["reasoning"]
                    },
                    "domain_analysis": domain_result
                },
                "final_analysis": domain_result,
                "architecture": "generic_ocr_classifier_router",
                "benefits": [
                    "Separation of concerns - OCR, classification, and domain analysis are independent",
                    "Scalable - easy to add new document types and agents",
                    "Maintainable - each component can be updated independently",
                    "Reusable - OCR and classification can be used across all document types"
                ]
            }
            
            print("✅ Complete processing finished successfully!")
            return final_result
            
        except Exception as e:
            error_result = {
                "success": False,
                "doc_id": doc_id,
                "error": f"Processing failed: {str(e)}",
                "architecture": "generic_ocr_classifier_router"
            }
            print(f"❌ Processing failed: {str(e)}")
            return error_result


# =============================================================================
# 6. DEMO EXECUTION
# =============================================================================

async def demo_new_architecture():
    """Demonstrate the new generic OCR + router architecture."""
    
    print("=" * 80)
    print("🚀 GENERIC OCR + ROUTER ARCHITECTURE DEMO")
    print("=" * 80)
    
    enhanced_router = EnhancedRouter()
    
    # Test different document types
    test_documents = [
        {
            "content": b"APOLLO HOSPITALS Patient medical bill",
            "doc_id": "medical_001",
            "type": "Medical Bill"
        },
        {
            "content": b"SERVICE CONTRACT AGREEMENT between parties", 
            "doc_id": "legal_001",
            "type": "Legal Contract"
        },
        {
            "content": b"BUSINESS INVOICE billing statement",
            "doc_id": "invoice_001", 
            "type": "Business Invoice"
        }
    ]
    
    for doc in test_documents:
        print(f"\n📄 Testing {doc['type']} (ID: {doc['doc_id']})")
        print("-" * 50)
        
        result = await enhanced_router.process_document_complete(
            file_content=doc["content"],
            doc_id=doc["doc_id"],
            language="english"
        )
        
        if result["success"]:
            pipeline = result["processing_pipeline"]
            
            print(f"\n📊 RESULTS SUMMARY:")
            print(f"   Document Type: {pipeline['document_classification']['document_type']}")
            print(f"   Classification Confidence: {pipeline['document_classification']['confidence']:.2f}")
            print(f"   Agent Used: {pipeline['routing_decision']['selected_agent']}")
            print(f"   OCR Confidence: {pipeline['ocr_extraction']['confidence']:.2f}")
            
            if "verdict" in pipeline["domain_analysis"]:
                print(f"   Analysis Verdict: {pipeline['domain_analysis']['verdict']}")
            
        else:
            print(f"❌ Processing failed: {result['error']}")
        
        print()
    
    print("=" * 80)
    print("✅ DEMO COMPLETED")
    print("\n🎯 KEY BENEFITS OF NEW ARCHITECTURE:")
    print("   1. Generic OCR Tool - Works with ANY document type")
    print("   2. LLM-Powered Classification - Intelligent document type detection")
    print("   3. Smart Routing - Selects best agent based on document characteristics")
    print("   4. Specialized Agents - Focus on domain-specific analysis only")
    print("   5. Scalable Design - Easy to add new document types and agents")
    print("   6. Maintainable Code - Clear separation of concerns")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demo_new_architecture()) 