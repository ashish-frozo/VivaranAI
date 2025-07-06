"""
Entity to Source Mapper for AI Web Scraping

This module maps document entities (procedures, drugs, services) to relevant 
government data sources based on document type and content analysis.
"""

import structlog
from typing import Dict, List, Optional, Set
from enum import Enum

logger = structlog.get_logger(__name__)


class DocumentType(Enum):
    """Types of documents that can be processed"""
    MEDICAL_BILL = "medical_bill"
    PHARMACY_INVOICE = "pharmacy_invoice"
    INSURANCE_CLAIM = "insurance_claim"
    DIAGNOSTIC_REPORT = "diagnostic_report"
    UNKNOWN = "unknown"


class EntityToSourceMapper:
    """Maps document entities to relevant government data sources"""
    
    # Government source mapping
    SOURCE_MAPPING = {
        # Medical procedures and consultations
        "medical_procedures": {
            "cghs": "https://cghs.gov.in/ShowContentL2.aspx?id=1208",
            "esi": "https://esic.nic.in/medical-care-benefits",
            "aiims": "https://www.aiims.edu/aiims/departments-and-centres/"
        },
        
        # Drug and medication sources
        "medications": {
            "nppa": "https://nppa.gov.in/drug-pricing",
            "cdsco": "https://cdsco.gov.in/opencms/opencms/en/Drugs/",
            "jan_aushadhi": "https://janaushadhi.gov.in/product-list.aspx"
        },
        
        # Insurance and claim sources
        "insurance_claims": {
            "irdai": "https://irdai.gov.in/",
            "gic": "https://www.gicofindia.in/",
            "lic": "https://licindia.in/"
        },
        
        # State-specific sources
        "state_health": {
            "delhi": "https://health.delhi.gov.in/rates",
            "karnataka": "https://arogya.karnataka.gov.in/",
            "maharashtra": "https://arogya.maharashtra.gov.in/",
            "tamil_nadu": "https://health.tn.gov.in/",
            "west_bengal": "https://wbhealth.gov.in/"
        },
        
        # Diagnostic and lab test sources
        "diagnostics": {
            "nabl": "https://www.nabl-india.org/",
            "cdsco_diagnostics": "https://cdsco.gov.in/opencms/opencms/en/Drugs/",
            "icmr": "https://www.icmr.nic.in/"
        }
    }
    
    # Entity type patterns for classification
    ENTITY_PATTERNS = {
        "medical_procedures": [
            "consultation", "surgery", "operation", "procedure", "treatment",
            "examination", "checkup", "visit", "therapy", "counseling"
        ],
        "medications": [
            "tablet", "capsule", "syrup", "injection", "medicine", "drug",
            "antibiotic", "painkiller", "vitamin", "supplement", "drops"
        ],
        "diagnostics": [
            "test", "scan", "x-ray", "mri", "ct", "ultrasound", "echo",
            "blood", "urine", "pathology", "lab", "report", "analysis"
        ],
        "insurance_services": [
            "claim", "premium", "policy", "coverage", "deductible",
            "copay", "reimbursement", "cashless", "benefit"
        ]
    }
    
    def __init__(self):
        """Initialize the entity mapper"""
        self.logger = logger.bind(component="entity_mapper")
    
    async def map_entities_to_sources(
        self, 
        entities: Dict[str, List[str]], 
        document_type: str = "medical_bill",
        state_code: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Map document entities to relevant government sources
        
        Args:
            entities: Dictionary of entity types and their values
            document_type: Type of document being processed
            state_code: State code for state-specific sources
            
        Returns:
            Dictionary mapping source names to URLs
        """
        self.logger.info(f"Mapping entities for {document_type} document")
        
        relevant_sources = {}
        
        # Classify entities and map to source categories
        entity_categories = self._classify_entities(entities)
        
        # Map categories to specific sources
        for category, category_entities in entity_categories.items():
            if category in self.SOURCE_MAPPING:
                sources = self.SOURCE_MAPPING[category]
                
                # Add all sources for this category
                for source_name, source_url in sources.items():
                    # Filter state-specific sources
                    if category == "state_health" and state_code:
                        state_key = self._normalize_state_code(state_code)
                        if source_name == state_key and state_key in sources:
                            relevant_sources[f"state_{state_key}"] = sources[state_key]
                    else:
                        relevant_sources[f"{category}_{source_name}"] = source_url
        
        # Add document-specific sources
        doc_specific_sources = self._get_document_specific_sources(document_type)
        relevant_sources.update(doc_specific_sources)
        
        self.logger.info(f"Mapped to {len(relevant_sources)} sources: {list(relevant_sources.keys())}")
        return relevant_sources
    
    def _classify_entities(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Classify entities into government source categories"""
        classified = {
            "medical_procedures": [],
            "medications": [],
            "diagnostics": [],
            "insurance_services": []
        }
        
        # Process each entity
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_lower = entity.lower()
                
                # Check against patterns
                for category, patterns in self.ENTITY_PATTERNS.items():
                    if any(pattern in entity_lower for pattern in patterns):
                        classified[category].append(entity)
                        break
                else:
                    # Default classification based on entity type
                    if entity_type in ["procedures", "consultations", "treatments"]:
                        classified["medical_procedures"].append(entity)
                    elif entity_type in ["medications", "drugs", "medicines"]:
                        classified["medications"].append(entity)
                    elif entity_type in ["tests", "diagnostics", "lab_tests"]:
                        classified["diagnostics"].append(entity)
                    elif entity_type in ["insurance", "claims", "policies"]:
                        classified["insurance_services"].append(entity)
        
        # Remove empty categories
        classified = {k: v for k, v in classified.items() if v}
        
        return classified
    
    def _get_document_specific_sources(self, document_type: str) -> Dict[str, str]:
        """Get sources specific to document type"""
        doc_sources = {}
        
        if document_type == "medical_bill":
            doc_sources.update({
                "cghs_main": "https://cghs.gov.in/ShowContentL2.aspx?id=1208",
                "esi_main": "https://esic.nic.in/medical-care-benefits"
            })
        elif document_type == "pharmacy_invoice":
            doc_sources.update({
                "nppa_main": "https://nppa.gov.in/drug-pricing",
                "jan_aushadhi_main": "https://janaushadhi.gov.in/product-list.aspx"
            })
        elif document_type == "insurance_claim":
            doc_sources.update({
                "irdai_main": "https://irdai.gov.in/",
                "insurance_companies": "https://www.gicofindia.in/"
            })
        elif document_type == "diagnostic_report":
            doc_sources.update({
                "nabl_main": "https://www.nabl-india.org/",
                "icmr_main": "https://www.icmr.nic.in/"
            })
        
        return doc_sources
    
    def _normalize_state_code(self, state_code: str) -> str:
        """Normalize state code to match source keys"""
        if not state_code:
            return None
            
        state_mapping = {
            "DL": "delhi",
            "DELHI": "delhi",
            "KA": "karnataka", 
            "KARNATAKA": "karnataka",
            "MH": "maharashtra",
            "MAHARASHTRA": "maharashtra",
            "TN": "tamil_nadu",
            "TAMIL NADU": "tamil_nadu",
            "WB": "west_bengal",
            "WEST BENGAL": "west_bengal"
        }
        
        state_upper = state_code.upper()
        return state_mapping.get(state_upper, state_code.lower())
    
    async def get_priority_sources(
        self, 
        entities: Dict[str, List[str]], 
        document_type: str = "medical_bill"
    ) -> List[str]:
        """
        Get priority-ordered list of sources for scraping
        
        Args:
            entities: Document entities
            document_type: Type of document
            
        Returns:
            List of source URLs in priority order
        """
        all_sources = await self.map_entities_to_sources(entities, document_type)
        
        # Define priority order
        priority_order = [
            "cghs_main", "esi_main", "nppa_main",  # Government priorities
            "state_delhi", "state_karnataka", "state_maharashtra",  # State sources
            "medical_procedures_cghs", "medications_nppa",  # Category sources
            "diagnostics_nabl", "insurance_claims_irdai"  # Specialized sources
        ]
        
        # Order sources by priority
        priority_sources = []
        for priority_key in priority_order:
            if priority_key in all_sources:
                priority_sources.append(all_sources[priority_key])
        
        # Add remaining sources
        for source_url in all_sources.values():
            if source_url not in priority_sources:
                priority_sources.append(source_url)
        
        return priority_sources
    
    def validate_source_accessibility(self, source_urls: List[str]) -> Dict[str, bool]:
        """
        Validate if sources are accessible (placeholder for now)
        
        Args:
            source_urls: List of URLs to validate
            
        Returns:
            Dictionary mapping URLs to accessibility status
        """
        # Placeholder implementation
        # In production, this would check actual URL accessibility
        accessibility = {}
        
        for url in source_urls:
            # Simple heuristic based on URL patterns
            if any(domain in url for domain in ["cghs.gov.in", "nppa.gov.in", "esic.nic.in"]):
                accessibility[url] = True  # Assume government sites are accessible
            else:
                accessibility[url] = True  # For now, assume all are accessible
        
        return accessibility
    
    def get_source_metadata(self, source_url: str) -> Dict[str, str]:
        """Get metadata about a specific source"""
        metadata = {
            "type": "unknown",
            "authority": "unknown",
            "data_format": "html",
            "update_frequency": "unknown"
        }
        
        # Categorize by URL patterns
        if "cghs.gov.in" in source_url:
            metadata.update({
                "type": "medical_rates",
                "authority": "central_government",
                "data_format": "html_tables",
                "update_frequency": "yearly"
            })
        elif "nppa.gov.in" in source_url:
            metadata.update({
                "type": "drug_prices",
                "authority": "regulatory_body",
                "data_format": "html_tables",
                "update_frequency": "monthly"
            })
        elif "esic.nic.in" in source_url:
            metadata.update({
                "type": "medical_rates",
                "authority": "insurance_scheme", 
                "data_format": "html_tables",
                "update_frequency": "yearly"
            })
        elif "irdai.gov.in" in source_url:
            metadata.update({
                "type": "insurance_rates",
                "authority": "regulatory_body",
                "data_format": "html_pages",
                "update_frequency": "quarterly"
            })
        
        return metadata 