"""
Architecture Comparison: Current vs. Proposed Generic OCR + Router

This shows the key differences between the current tightly-coupled approach
and the proposed generic OCR + router architecture.
"""

# =============================================================================
# CURRENT ARCHITECTURE (Tightly Coupled)
# =============================================================================

class CurrentMedicalBillProcessor:
    """Current approach - OCR + medical analysis tightly coupled."""
    
    def __init__(self):
        # Medical-specific OCR patterns hardcoded
        self.medical_patterns = [
            r'consultation.*(\d+)',
            r'medicine.*(\d+)', 
            r'hospital.*charges'
        ]
        
        # Medical-specific validation rules
        self.cghs_rates = {"consultation": 650, "blood_test": 400}
    
    async def process_medical_bill(self, file_content: bytes) -> dict:
        """Process medical bill - OCR and analysis in one component."""
        
        # OCR extraction with medical-specific preprocessing
        raw_text = self._ocr_with_medical_preprocessing(file_content)
        
        # Medical-specific line item extraction
        line_items = self._extract_medical_line_items(raw_text)
        
        # Medical analysis
        analysis = self._analyze_medical_charges(line_items)
        
        return analysis
    
    def _ocr_with_medical_preprocessing(self, file_content: bytes) -> str:
        """OCR with medical bill specific preprocessing."""
        # This is coupled to medical bills only
        return "OCR text with medical preprocessing..."
    
    def _extract_medical_line_items(self, text: str) -> list:
        """Extract line items using hardcoded medical patterns."""
        # Only works for medical bills
        return [{"item": "consultation", "amount": 800}]
    
    def _analyze_medical_charges(self, line_items: list) -> dict:
        """Medical-specific analysis."""
        return {"verdict": "overcharge", "total_overcharge": 150}


# Problems with current approach:
# 1. OCR logic is tied to medical bills only
# 2. Adding new document types requires duplicating OCR code
# 3. Hard to test OCR vs business logic separately
# 4. Not scalable for legal docs, financial statements, etc.


# =============================================================================
# PROPOSED ARCHITECTURE (Generic OCR + Router)
# =============================================================================

class GenericOCRTool:
    """Generic OCR that works with ANY document type."""
    
    async def extract_text(self, file_content: bytes) -> dict:
        """Extract raw text without domain-specific logic."""
        return {
            "raw_text": "Generic OCR extracted text...",
            "tables": [],
            "confidence": 0.92,
            "pages": 1
        }


class DocumentTypeClassifier:
    """LLM-powered document type classifier."""
    
    async def classify(self, raw_text: str) -> dict:
        """Classify document and determine routing needs."""
        
        # Use LLM to classify document type
        if "hospital" in raw_text.lower():
            return {
                "document_type": "medical_bill",
                "confidence": 0.94,
                "required_capabilities": ["medical_analysis", "rate_validation"],
                "suggested_agent": "medical_bill_agent"
            }
        elif "contract" in raw_text.lower():
            return {
                "document_type": "legal_contract", 
                "confidence": 0.91,
                "required_capabilities": ["legal_analysis"],
                "suggested_agent": "legal_agent"
            }
        else:
            return {
                "document_type": "unknown",
                "confidence": 0.60,
                "suggested_agent": "generic_agent"
            }


class RouterAgent:
    """Intelligent router that selects appropriate agents."""
    
    def __init__(self):
        self.agents = {
            "medical_bill_agent": MedicalBillAgent(),
            "legal_agent": LegalAgent(),
            "financial_agent": FinancialAgent()
        }
    
    async def route_and_process(self, classification: dict, ocr_data: dict) -> dict:
        """Route to appropriate agent based on classification."""
        
        agent_name = classification["suggested_agent"]
        agent = self.agents.get(agent_name)
        
        if agent:
            return await agent.process(ocr_data, classification)
        else:
            return {"error": f"No agent found for {agent_name}"}


class MedicalBillAgent:
    """Specialized medical bill agent - only handles medical analysis."""
    
    async def process(self, ocr_data: dict, classification: dict) -> dict:
        """Process medical bill with domain-specific analysis."""
        raw_text = ocr_data["raw_text"]
        
        # Medical-specific processing (no OCR logic here)
        line_items = self._extract_medical_items(raw_text)
        overcharges = self._validate_against_cghs(line_items)
        
        return {
            "verdict": "overcharge" if overcharges else "ok",
            "analysis": "Medical analysis complete",
            "overcharges": overcharges
        }
    
    def _extract_medical_items(self, text: str) -> list:
        """Extract medical line items from pre-processed text."""
        return [{"item": "consultation", "amount": 800}]
    
    def _validate_against_cghs(self, items: list) -> list:
        """Validate against CGHS rates."""
        return [{"item": "consultation", "overcharge": 150}]


class LegalAgent:
    """Specialized legal document agent."""
    
    async def process(self, ocr_data: dict, classification: dict) -> dict:
        """Process legal document with legal-specific analysis."""
        return {
            "contract_type": "service_agreement",
            "risk_level": "medium",
            "compliance_issues": []
        }


class FinancialAgent:
    """Specialized financial document agent."""
    
    async def process(self, ocr_data: dict, classification: dict) -> dict:
        """Process financial document with financial analysis."""
        return {
            "document_type": "invoice",
            "total_amount": 2360.0,
            "tax_calculation": "correct"
        }


class EnhancedProcessingPipeline:
    """Enhanced pipeline using generic OCR + router architecture."""
    
    def __init__(self):
        self.ocr_tool = GenericOCRTool()
        self.classifier = DocumentTypeClassifier()
        self.router = RouterAgent()
    
    async def process_any_document(self, file_content: bytes) -> dict:
        """Process ANY document type using the new architecture."""
        
        # Step 1: Generic OCR (works for all document types)
        ocr_result = await self.ocr_tool.extract_text(file_content)
        
        # Step 2: LLM-powered document classification
        classification = await self.classifier.classify(ocr_result["raw_text"])
        
        # Step 3: Intelligent routing to specialized agent
        final_result = await self.router.route_and_process(classification, ocr_result)
        
        return {
            "ocr_data": ocr_result,
            "classification": classification,
            "analysis": final_result,
            "architecture": "Generic OCR + Router"
        }


# =============================================================================
# COMPARISON DEMO
# =============================================================================

async def compare_architectures():
    """Compare current vs. proposed architecture."""
    
    print("=" * 60)
    print("ARCHITECTURE COMPARISON")
    print("=" * 60)
    
    # Mock file content
    medical_bill = b"APOLLO HOSPITAL Patient: John Doe Consultation: Rs. 800"
    legal_contract = b"SERVICE CONTRACT between Party A and Party B"
    
    print("\n1. CURRENT ARCHITECTURE (Tightly Coupled)")
    print("-" * 40)
    
    current_processor = CurrentMedicalBillProcessor()
    
    # Can only process medical bills
    result = await current_processor.process_medical_bill(medical_bill)
    print(f"Medical Bill Result: {result}")
    
    # Cannot process legal contracts - would need separate processor
    print("❌ Cannot process legal contracts without new processor")
    
    print("\n2. PROPOSED ARCHITECTURE (Generic OCR + Router)")
    print("-" * 50)
    
    enhanced_pipeline = EnhancedProcessingPipeline()
    
    # Can process medical bills
    medical_result = await enhanced_pipeline.process_any_document(medical_bill)
    print(f"Medical Bill: {medical_result['classification']['document_type']} -> {medical_result['classification']['suggested_agent']}")
    
    # Can also process legal contracts
    legal_result = await enhanced_pipeline.process_any_document(legal_contract)
    print(f"Legal Contract: {legal_result['classification']['document_type']} -> {legal_result['classification']['suggested_agent']}")
    
    print("\n" + "=" * 60)
    print("KEY BENEFITS OF NEW ARCHITECTURE:")
    print("=" * 60)
    print("✅ Generic OCR works with ANY document type")
    print("✅ LLM classifier intelligently determines document type")
    print("✅ Router selects appropriate specialized agent")
    print("✅ Easy to add new document types and agents")
    print("✅ Clear separation of concerns")
    print("✅ Much more maintainable and testable")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(compare_architectures()) 