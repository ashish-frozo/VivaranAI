"""
Smart Data Agent for Dynamic Government Data Fetching

This agent analyzes documents, extracts entities, maps them to government sources,
and uses AI web scraping to fetch relevant real-time data.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .base_agent import BaseAgent
from .tools.ai_web_scraper import AIWebScrapingTool, ScrapingResult
from .tools.entity_mapper import EntityToSourceMapper, DocumentType

logger = structlog.get_logger(__name__)


class EntityExtractionResult(BaseModel):
    """Result of entity extraction from document"""
    success: bool
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    document_type: str
    total_entities: int = 0
    extraction_notes: str = ""
    error: Optional[str] = None


class DataFetchingResult(BaseModel):
    """Result of data fetching operation"""
    success: bool
    sources_scraped: int = 0
    sources_successful: int = 0
    total_data_points: int = 0
    scraped_data: Dict[str, Any] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    processing_time_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None


class SmartDataAgent(BaseAgent):
    """AI agent that dynamically fetches relevant government data"""
    
    agent_id = "smart_data_agent"
    capabilities = [
        "entity_extraction",
        "source_mapping", 
        "ai_web_scraping",
        "data_validation",
        "cache_management"
    ]
    
    def __init__(self, openai_api_key: str, cache_manager=None):
        """Initialize the Smart Data Agent"""
        # Initialize BaseAgent with required parameters
        super().__init__(
            agent_id="smart_data_agent",
            name="Smart Data Agent",
            instructions="Extract entities from documents and fetch relevant government data using AI web scraping",
            tools=[],  # No OpenAI SDK tools needed for this agent
            default_model="gpt-4"
        )
        
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.text_model = "gpt-4"
        self.max_tokens = 4000
        self.temperature = 0.1
        
        # Initialize components
        self.web_scraper = AIWebScrapingTool(openai_api_key)
        self.entity_mapper = EntityToSourceMapper()
        self.cache_manager = cache_manager
        
        # Configuration
        self.max_sources_per_request = 5
        self.scraping_timeout = 30  # seconds
        self.cache_ttl = 6 * 3600  # 6 hours for dynamic data
        
        self.logger = logger.bind(component="smart_data_agent")
    
    async def fetch_relevant_data(
        self, 
        document_type: str,
        raw_text: str,
        state_code: Optional[str] = None,
        max_sources: int = 3
    ) -> DataFetchingResult:
        """
        Main method: Extract entities and fetch relevant government data
        
        Args:
            document_type: Type of document (medical_bill, pharmacy_invoice, etc.)
            raw_text: Raw text from document OCR
            state_code: State code for state-specific sources
            max_sources: Maximum number of sources to scrape
            
        Returns:
            DataFetchingResult with scraped government data
        """
        start_time = datetime.now()
        self.logger.info(f"Starting data fetch for {document_type}")
        
        try:
            # Step 1: Extract entities from document
            entity_result = await self.extract_entities(raw_text, document_type)
            
            if not entity_result.success:
                return DataFetchingResult(
                    success=False,
                    error=f"Entity extraction failed: {entity_result.error}"
                )
            
            # Step 2: Map entities to government sources
            relevant_sources = await self.entity_mapper.map_entities_to_sources(
                entities=entity_result.entities,
                document_type=document_type,
                state_code=state_code
            )
            
            if not relevant_sources:
                return DataFetchingResult(
                    success=False,
                    error="No relevant government sources found"
                )
            
            # Step 3: Prioritize and limit sources
            priority_sources = await self.entity_mapper.get_priority_sources(
                entities=entity_result.entities,
                document_type=document_type
            )
            limited_sources = priority_sources[:max_sources]
            
            # Step 4: Check cache first
            cache_key = self._generate_cache_key(entity_result.entities, document_type)
            cached_data = await self._get_from_cache(cache_key)
            
            if cached_data:
                self.logger.info("Returning cached data")
                return DataFetchingResult(
                    success=True,
                    sources_scraped=len(limited_sources),
                    sources_successful=len(limited_sources),
                    scraped_data=cached_data,
                    processing_time_ms=0  # Cache hit
                )
            
            # Step 5: Scrape fresh data
            scraped_data = await self._scrape_multiple_sources(
                sources=limited_sources,
                entities=entity_result.entities,
                document_type=document_type
            )
            
            # Step 6: Validate and cache results
            validated_data = await self._validate_scraped_data(scraped_data)
            
            if validated_data:
                await self._save_to_cache(cache_key, validated_data)
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Step 7: Calculate success metrics
            successful_sources = sum(1 for result in scraped_data.values() if result.get("success", False))
            total_data_points = sum(
                len(result.get("data", {}).get("extracted_data", []))
                for result in scraped_data.values()
                if result.get("success", False)
            )
            
            return DataFetchingResult(
                success=len(validated_data) > 0,
                sources_scraped=len(limited_sources),
                sources_successful=successful_sources,
                total_data_points=total_data_points,
                scraped_data=validated_data,
                confidence_scores={
                    source: result.get("confidence", 0.0)
                    for source, result in scraped_data.items()
                },
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Data fetching failed: {e}")
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return DataFetchingResult(
                success=False,
                processing_time_ms=processing_time,
                error=str(e)
            )
    
    async def extract_entities(self, raw_text: str, document_type: str) -> EntityExtractionResult:
        """
        Extract relevant entities from document text using GPT-4
        
        Args:
            raw_text: Raw text from document
            document_type: Type of document
            
        Returns:
            EntityExtractionResult with extracted entities
        """
        try:
            self.logger.info("Extracting entities from document")
            
            prompt = f"""
            Extract medical/healthcare entities from this {document_type} document.
            
            DOCUMENT TEXT:
            {raw_text[:6000]}  # Limit for token constraints
            
            EXTRACTION INSTRUCTIONS:
            Extract and categorize entities relevant for government rate validation:
            
            1. PROCEDURES: Medical procedures, consultations, surgeries, treatments
            2. MEDICATIONS: Drugs, medicines, tablets, injections, syrups
            3. DIAGNOSTICS: Lab tests, scans, X-rays, blood tests, pathology
            4. SERVICES: Room charges, nursing, emergency services
            5. SPECIALTIES: Cardiology, orthopedic, pediatric, etc.
            
            RETURN FORMAT:
            {{
                "entities": {{
                    "procedures": ["consultation", "surgery", "..."],
                    "medications": ["paracetamol", "amoxicillin", "..."],
                    "diagnostics": ["blood test", "x-ray", "..."],
                    "services": ["room charges", "nursing", "..."],
                    "specialties": ["cardiology", "orthopedic", "..."]
                }},
                "confidence": 0.0-1.0,
                "document_type": "{document_type}",
                "extraction_notes": "Notes about the extraction process"
            }}
            
            GUIDELINES:
            - Extract only entities that appear in the document
            - Use generic names (e.g., "consultation" not "Dr. Smith consultation")
            - Include variations (e.g., "CBC", "Complete Blood Count")
            - Provide confidence based on text clarity
            - Return empty arrays for categories not found
            
            Return only valid JSON.
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.text_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Parse response
            response_content = response.choices[0].message.content
            extracted_result = json.loads(response_content)
            
            # Count total entities
            entities = extracted_result.get("entities", {})
            total_entities = sum(len(entity_list) for entity_list in entities.values())
            
            return EntityExtractionResult(
                success=True,
                entities=entities,
                confidence=extracted_result.get("confidence", 0.7),
                document_type=extracted_result.get("document_type", document_type),
                total_entities=total_entities,
                extraction_notes=extracted_result.get("extraction_notes", "")
            )
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return EntityExtractionResult(
                success=False,
                entities={},
                confidence=0.0,
                document_type=document_type,
                total_entities=0,
                extraction_notes="",
                error=str(e)
            )
    
    async def _scrape_multiple_sources(
        self, 
        sources: List[str], 
        entities: Dict[str, List[str]],
        document_type: str
    ) -> Dict[str, Dict[str, Any]]:
        """Scrape multiple government sources concurrently"""
        
        self.logger.info(f"Scraping {len(sources)} sources")
        
        # Create scraping tasks
        tasks = []
        for i, source_url in enumerate(sources):
            task = self._scrape_single_source(
                url=source_url,
                entities=entities,
                source_name=f"source_{i}",
                document_type=document_type
            )
            tasks.append(task)
        
        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.scraping_timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Scraping timeout exceeded")
            results = []
        
        # Process results
        scraped_data = {}
        for i, result in enumerate(results):
            source_name = f"source_{i}"
            
            if isinstance(result, Exception):
                scraped_data[source_name] = {
                    "success": False,
                    "error": str(result),
                    "source_url": sources[i] if i < len(sources) else "unknown"
                }
            elif isinstance(result, dict):
                scraped_data[source_name] = result
        
        return scraped_data
    
    async def _scrape_single_source(
        self, 
        url: str, 
        entities: Dict[str, List[str]], 
        source_name: str,
        document_type: str
    ) -> Dict[str, Any]:
        """Scrape a single government source"""
        
        try:
            # Determine schema type based on document type
            schema_mapping = {
                "medical_bill": "medical_rates",
                "pharmacy_invoice": "drug_prices", 
                "insurance_claim": "insurance_rates",
                "diagnostic_report": "medical_rates"
            }
            schema_type = schema_mapping.get(document_type, "medical_rates")
            
            # Scrape using AI web scraper
            result = await self.web_scraper.scrape_government_data(
                url=url,
                entities=entities,
                schema_type=schema_type
            )
            
            return {
                "success": result.success,
                "data": result.data,
                "confidence": result.confidence,
                "strategy_used": result.strategy_used,
                "source_url": result.source_url,
                "timestamp": result.timestamp.isoformat(),
                "error": result.error
            }
            
        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            return {
                "success": False,
                "source_url": url,
                "error": str(e)
            }
    
    async def _validate_scraped_data(self, scraped_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and merge scraped data from multiple sources"""
        
        validated_data = {}
        
        for source_name, source_result in scraped_data.items():
            if source_result.get("success") and source_result.get("confidence", 0) >= 0.5:
                # Extract data
                data = source_result.get("data", {})
                extracted_items = data.get("extracted_data", [])
                
                if extracted_items:
                    validated_data[source_name] = {
                        "source_url": source_result.get("source_url"),
                        "confidence": source_result.get("confidence"),
                        "data_points": len(extracted_items),
                        "extracted_data": extracted_items,
                        "timestamp": source_result.get("timestamp")
                    }
        
        return validated_data
    
    def _generate_cache_key(self, entities: Dict[str, List[str]], document_type: str) -> str:
        """Generate cache key for entities and document type"""
        # Create a stable hash of entities
        entities_str = json.dumps(entities, sort_keys=True)
        cache_key = f"smart_data:{document_type}:{hash(entities_str)}"
        return cache_key
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache if available"""
        if not self.cache_manager:
            return None
        
        try:
            return await self.cache_manager.get(cache_key)
        except Exception as e:
            self.logger.warning(f"Cache get failed: {e}")
            return None
    
    async def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save data to cache"""
        if not self.cache_manager:
            return
        
        try:
            await self.cache_manager.set(cache_key, data, ttl=self.cache_ttl)
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the Smart Data Agent"""
        try:
            # Test OpenAI connectivity
            test_response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            
            openai_status = bool(test_response.choices)
        except Exception:
            openai_status = False
        
        return {
            "agent_id": self.agent_id,
            "status": "healthy" if openai_status else "degraded",
            "capabilities": self.capabilities,
            "openai_connection": openai_status,
            "cache_available": self.cache_manager is not None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for the agent"""
        # Placeholder for statistics tracking
        return {
            "agent_id": self.agent_id,
            "total_requests": 0,  # Would track in production
            "successful_extractions": 0,
            "failed_extractions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_processing_time_ms": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    async def process_task(self, context, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task using the Smart Data Agent
        
        This is the abstract method implementation required by BaseAgent.
        It delegates to the main fetch_relevant_data method.
        """
        try:
            # Extract parameters from task_data
            document_type = task_data.get("document_type", "medical_bill")
            raw_text = task_data.get("raw_text", "")
            state_code = task_data.get("state_code")
            max_sources = task_data.get("max_sources", 3)
            
            # Call the main data fetching method
            result = await self.fetch_relevant_data(
                document_type=document_type,
                raw_text=raw_text,
                state_code=state_code,
                max_sources=max_sources
            )
            
            # Convert result to dictionary format expected by BaseAgent
            return {
                "success": result.success,
                "sources_scraped": result.sources_scraped,
                "sources_successful": result.sources_successful,
                "total_data_points": result.total_data_points,
                "scraped_data": result.scraped_data,
                "confidence_scores": result.confidence_scores,
                "processing_time_ms": result.processing_time_ms,
                "error": result.error
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sources_scraped": 0,
                "sources_successful": 0,
                "total_data_points": 0
            } 