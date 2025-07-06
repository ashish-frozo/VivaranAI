# New Generic OCR + Router Architecture

## Overview

The user's suggestion to create a **generic OCR tool** that sends raw data to an **LLM router** for agent selection is an excellent architectural improvement. This document outlines the refactored architecture that separates concerns and provides much better scalability.

## Current Architecture Problems

### 1. Tight Coupling
- `DocumentProcessor` mixes generic OCR with medical-specific logic
- Hard-coded medical patterns, validation rules, and classifications
- Difficult to extend to other document types

### 2. Limited Scalability
- Adding new document types requires modifying existing code
- OCR logic is replicated across different domain agents
- No separation between document processing and domain analysis

### 3. Mixed Responsibilities
- Single component handles both OCR extraction AND medical analysis
- Makes testing, maintenance, and debugging difficult

## New Architecture: Generic OCR → LLM Router → Specialized Agents

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Upload   │ -> │  Generic OCR     │ -> │  Document Type  │ -> │  Router Agent   │
│                 │    │  Tool            │    │  Classifier     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
                                                                                │
                                                                                v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Legal Document  │    │  Insurance Claim │    │  Financial      │    │  Medical Bill   │
│ Agent           │    │  Agent           │    │  Statement Agent│    │  Agent          │
│                 │    │                  │    │                 │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
```

## Component Breakdown

### 1. Generic OCR Tool (`agents/tools/generic_ocr_tool.py`)

**Purpose**: Domain-agnostic document processing and text extraction

**Responsibilities**:
- PDF and image processing (JPEG, PNG)
- Multi-language OCR (English, Hindi, Bengali, Tamil)
- Advanced image preprocessing with multiple strategies
- Table detection and extraction
- Raw text and structured data output
- Quality assessment and error handling

**Key Features**:
```python
class GenericOCRTool:
    async def __call__(self, file_content: bytes, doc_id: str, language: str) -> Dict:
        # Returns raw OCR data without domain-specific processing
        return {
            "success": True,
            "raw_text": "extracted text...",
            "pages": ["page 1 text", "page 2 text"],
            "tables": [{"headers": [...], "rows": [...]}],
            "language_detected": "english",
            "processing_stats": {
                "ocr_confidence": 0.92,
                "pages_processed": 2,
                "processing_time_ms": 1500
            }
        }
```

### 2. Document Type Classifier (`agents/tools/document_type_classifier.py`)

**Purpose**: LLM-powered intelligent document classification

**Responsibilities**:
- Analyze raw OCR text to determine document type
- Classify into categories (medical_bill, legal_contract, financial_statement, etc.)
- Determine required processing capabilities
- Suggest appropriate agent routing
- Provide confidence scoring

**Key Features**:
```python
class DocumentTypeClassifier:
    async def __call__(self, raw_text: str, doc_id: str, pages: List[str]) -> Dict:
        # Uses LLM to classify document and determine routing
        return {
            "success": True,
            "document_type": "medical_bill",
            "confidence": 0.94,
            "required_capabilities": ["medical_analysis", "rate_validation"],
            "suggested_agent": "medical_bill_agent",
            "reasoning": "Document contains medical procedures, hospital billing..."
        }
```

### 3. Enhanced Router Agent (`agents/enhanced_router_agent.py`)

**Purpose**: Orchestrates the complete processing workflow

**Responsibilities**:
- Coordinate OCR extraction
- Trigger document classification
- Make intelligent routing decisions
- Execute domain-specific analysis
- Compile comprehensive results

**Processing Flow**:
```python
async def process_document_complete(self, request):
    # Stage 1: OCR Extraction
    ocr_result = await self.ocr_tool(file_content, doc_id, language)
    
    # Stage 2: Document Classification  
    classification = await self.document_classifier(
        ocr_result["raw_text"], doc_id, ocr_result["pages"]
    )
    
    # Stage 3: Routing Decision
    routing_decision = await self.make_routing_decision(
        classification, ocr_quality=ocr_result["processing_stats"]["ocr_confidence"]
    )
    
    # Stage 4: Domain Analysis
    domain_result = await self.execute_domain_analysis(
        routing_decision, ocr_result, classification
    )
    
    # Stage 5: Result Compilation
    return self.compile_final_result(ocr_result, classification, domain_result)
```

### 4. Specialized Domain Agents

**Medical Bill Agent**: Handles medical document analysis
- Rate validation against CGHS/ESI/NPPA
- Duplicate detection
- Prohibited item detection
- Medical procedure classification

**Legal Document Agent**: Handles legal document analysis
- Contract clause analysis
- Compliance verification
- Legal terminology processing

**Financial Statement Agent**: Handles financial document analysis
- Financial ratio calculation
- Audit trail verification
- Regulatory compliance checks

## Implementation Plan

### Phase 1: Create Generic OCR Tool
```bash
# Create the generic OCR tool
touch agents/tools/generic_ocr_tool.py

# Extract OCR logic from existing DocumentProcessor
# Remove medical-specific patterns and validations
# Focus on pure text extraction and image processing
```

### Phase 2: Build Document Type Classifier
```bash
# Create LLM-powered classifier
touch agents/tools/document_type_classifier.py

# Implement classification prompts
# Add document type enumeration
# Create capability mapping
```

### Phase 3: Enhanced Router Agent
```bash
# Create orchestration layer
touch agents/enhanced_router_agent.py

# Implement multi-stage processing workflow
# Add intelligent routing based on document type
# Create result compilation logic
```

### Phase 4: Refactor Medical Bill Agent
```bash
# Modify existing medical bill agent
# Remove OCR dependencies
# Focus on domain-specific analysis
# Accept pre-processed OCR data
```

## Benefits of New Architecture

### 1. **Separation of Concerns**
- OCR becomes a pure service without domain knowledge
- Document classification is handled by specialized LLM component
- Routing decisions are made based on document characteristics
- Domain analysis is handled by specialized agents

### 2. **Scalability**
- Easy to add new document types (insurance claims, legal contracts, etc.)
- New agents can be added without modifying existing OCR logic
- Classification system can learn new document patterns

### 3. **Reusability**
- Generic OCR tool can be used across all document types
- Document classifier can be enhanced with new categories
- Router logic can be optimized for different processing strategies

### 4. **Maintainability**
- Clear separation makes testing easier
- OCR bugs don't affect domain logic
- Domain-specific changes don't impact OCR quality
- Easier to optimize individual components

### 5. **Performance Optimization**
- Can cache OCR results for multiple analysis runs
- Parallel processing of classification and routing
- Intelligent routing based on document complexity and quality

## Usage Example

```python
# Initialize the enhanced router
enhanced_router = EnhancedRouterAgent(
    registry=agent_registry,
    openai_api_key="sk-..."
)

# Process any document type
result = await enhanced_router.process_document_complete(
    DocumentProcessingRequest(
        file_content=pdf_bytes,
        doc_id="doc_123",
        user_id="user_456",
        language="english",
        routing_strategy=RoutingStrategy.CAPABILITY_BASED
    )
)

# Result contains:
# - OCR extraction data
# - Document classification
# - Routing decision reasoning
# - Domain-specific analysis
# - Quality assessments
# - Processing recommendations
```

## Migration Strategy

### 1. **Backward Compatibility**
- Keep existing `DocumentProcessor` for current medical bill processing
- Gradually migrate to new architecture
- Add feature flags to control which system to use

### 2. **Gradual Rollout**
- Start with new document types (legal, financial)
- Validate OCR quality matches or exceeds current system
- Migrate medical bill processing once confident

### 3. **Testing Strategy**
- Unit tests for each component
- Integration tests for complete workflow
- Performance benchmarks against current system
- A/B testing with real documents

## Technical Considerations

### 1. **OCR Quality**
- Ensure generic OCR tool matches current medical bill OCR quality
- Implement quality assessment and fallback strategies
- Add monitoring for OCR confidence scores

### 2. **Classification Accuracy**
- Train and validate document type classification
- Implement confidence thresholds for routing decisions
- Add human-in-the-loop for low-confidence classifications

### 3. **Performance**
- Optimize for parallel processing where possible
- Implement caching for OCR results
- Monitor processing times across all stages

### 4. **Error Handling**
- Graceful degradation when components fail
- Fallback to rule-based classification if LLM fails
- Comprehensive error reporting and logging

## Conclusion

This new architecture addresses the user's excellent suggestion by:

1. **Creating a generic OCR tool** that focuses purely on text extraction
2. **Using an LLM router** (document classifier + enhanced router) to make intelligent routing decisions
3. **Allowing specialized agents** to focus on domain-specific analysis
4. **Providing much better scalability** for new document types and use cases

The architecture is cleaner, more maintainable, and provides a solid foundation for expanding beyond medical bill analysis to other document processing use cases. 