"""
AI Web Scraping Tool for Government Data Extraction

This module implements AI-powered web scraping capabilities using GPT-4 Vision 
and text analysis to extract government tariff data from websites.
"""

import asyncio
import json
import re
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any

import structlog
import aiohttp
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ScrapingResult(BaseModel):
    """Result of web scraping operation"""
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    strategy_used: str
    source_url: str
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None


class AIWebScrapingTool:
    """AI-powered web scraping tool for government data"""
    
    def __init__(self, openai_api_key: str):
        """Initialize the AI web scraping tool"""
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.text_model = "gpt-4"
        self.max_tokens = 4000
        self.temperature = 0.1
        self.logger = logger.bind(component="ai_web_scraper")
    
    def _extract_json_from_response(self, response_content: str) -> dict:
        """Extract JSON from GPT-4 response that might contain code blocks and explanatory text"""
        try:
            # First try to parse as direct JSON
            return json.loads(response_content.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from code blocks
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to extract any JSON-like structure
        json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        matches = re.findall(json_pattern, response_content, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, return a default structure
        return {
            "extracted_data": [],
            "confidence": 0.0,
            "data_source": "extraction_failed",
            "extraction_notes": "No valid JSON found in response"
        }
    
    async def scrape_government_data(
        self, 
        url: str,
        entities: Dict[str, List[str]],
        schema_type: str = "medical_rates"
    ) -> ScrapingResult:
        """
        Main scraping method using HTML analysis
        
        Args:
            url: Government website URL
            entities: Entities to look for (procedures, drugs, etc.)
            schema_type: Type of data schema to use
            
        Returns:
            ScrapingResult with extracted data
        """
        self.logger.info(f"[SCRAPE] Starting to scrape URL: {url}", url=url, entities=entities, schema_type=schema_type)
        
        try:
            # For now, use HTML analysis only (simpler implementation)
            return await self._scrape_with_html(url, entities, schema_type)
            
        except Exception as e:
            self.logger.error(f"Scraping failed: {e}")
            return ScrapingResult(
                success=False,
                source_url=url,
                strategy_used="html_analysis",
                confidence=0.0,
                error=str(e)
            )
    
    async def _scrape_with_html(
        self, 
        url: str, 
        entities: Dict[str, List[str]], 
        schema_type: str
    ) -> ScrapingResult:
        """Scrape using HTML content analysis with GPT-4"""
        
        try:
            # Fetch HTML content
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    html_content = await response.text()

            # Log the first 500 characters of HTML content
            snippet = html_content[:500]
            self.logger.info(f"[SCRAPE] HTML content snippet for {url}", url=url, html_snippet=snippet)
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract relevant content
            relevant_content = self._extract_relevant_html_content(soup)
            
            # Prepare prompt
            entity_context = self._format_entities_for_prompt(entities)
            
            prompt = f"""
            Extract government rate/pricing data from this HTML content.
            
            ENTITIES TO FIND:
            {entity_context}
            
            HTML CONTENT:
            {relevant_content[:8000]}
            
            EXTRACTION SCHEMA:
            Return ONLY valid JSON in this exact format:
            {{
                "extracted_data": [
                    {{
                        "name": "procedure/drug name",
                        "rate": "price in INR",
                        "category": "type of service",
                        "source": "data source"
                    }}
                ],
                "confidence": 0.8,
                "data_source": "website section",
                "extraction_notes": "notes about extraction"
            }}
            
            CRITICAL INSTRUCTIONS:
            1. Return ONLY the JSON object, no explanatory text
            2. If no rate data found, return empty "extracted_data" array
            3. Set confidence between 0.0 and 1.0
            4. Do not include markdown code blocks
            5. Ensure valid JSON syntax
            
            JSON Response:
            """
            
            # Call GPT-4
            response = await self.openai_client.chat.completions.create(
                model=self.text_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Parse response with improved JSON extraction
            response_content = response.choices[0].message.content
            extracted_data = self._extract_json_from_response(response_content)

            # Log extraction result
            num_items = len(extracted_data.get("extracted_data", [])) if isinstance(extracted_data.get("extracted_data", []), list) else 0
            confidence = extracted_data.get("confidence", 0.0)
            self.logger.info(f"[SCRAPE] Extraction result for {url}", url=url, num_items=num_items, confidence=confidence)
            
            return ScrapingResult(
                success=True,
                data=extracted_data,
                confidence=confidence,
                strategy_used="html_analysis",
                source_url=url
            )
            
        except Exception as e:
            self.logger.error(f"HTML scraping failed: {e}")
            return ScrapingResult(
                success=False,
                source_url=url,
                strategy_used="html_analysis",
                confidence=0.0,
                error=str(e)
            )
    
    def _format_entities_for_prompt(self, entities: Dict[str, List[str]]) -> str:
        """Format entities for GPT prompt"""
        formatted = []
        for entity_type, entity_list in entities.items():
            formatted.append(f"{entity_type}: {', '.join(entity_list[:10])}")
        return '\n'.join(formatted)
    
    def _extract_relevant_html_content(self, soup: BeautifulSoup) -> str:
        """Extract relevant HTML content that might contain rates"""
        relevant_content = []
        
        # Find tables (most likely to contain rates)
        tables = soup.find_all('table')
        for table in tables[:5]:
            relevant_content.append(f"TABLE: {table.get_text(strip=True)}")
        
        # Find divs with rate/price related classes or text
        rate_divs = soup.find_all('div', class_=lambda x: x and any(
            keyword in x.lower() for keyword in ['rate', 'price', 'tariff', 'cost']
        ))
        for div in rate_divs[:5]:
            relevant_content.append(f"RATE_DIV: {div.get_text(strip=True)}")
        
        # Find list items
        lists = soup.find_all(['ul', 'ol'])
        for lst in lists[:3]:
            relevant_content.append(f"LIST: {lst.get_text(strip=True)}")
        
        return '\n\n'.join(relevant_content)
    
    async def test_scraping_capability(self, test_url: str = "https://example.com") -> Dict[str, Any]:
        """Test scraping capability with a simple URL"""
        try:
            test_entities = {
                "procedures": ["consultation", "x-ray", "blood test"],
                "rates": ["500", "1000", "200"]
            }
            
            result = await self.scrape_government_data(
                url=test_url,
                entities=test_entities,
                schema_type="medical_rates"
            )
            
            return {
                "test_url": test_url,
                "success": result.success,
                "confidence": result.confidence,
                "strategy_used": result.strategy_used,
                "data_extracted": len(result.data.get("extracted_data", [])),
                "error": result.error
            }
            
        except Exception as e:
            return {
                "test_url": test_url,
                "success": False,
                "error": str(e)
            } 