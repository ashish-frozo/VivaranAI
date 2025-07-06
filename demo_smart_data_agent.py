#!/usr/bin/env python3
"""
Demo script for Smart Data Agent with AI Web Scraping

This script demonstrates the new architecture:
1. Document classification
2. Entity extraction
3. Source mapping
4. AI web scraping
5. Real-time data integration
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config.env_config import config, check_required_config

from agents.smart_data_agent import SmartDataAgent
from agents.tools.ai_web_scraper import AIWebScrapingTool
from agents.tools.entity_mapper import EntityToSourceMapper


class SmartDataDemo:
    """Demo class for Smart Data Agent"""
    
    def __init__(self):
        """Initialize demo with required components"""
        self.openai_api_key = config.openai_api_key
        
        self.smart_agent = SmartDataAgent(openai_api_key=self.openai_api_key)
        self.entity_mapper = EntityToSourceMapper()
        self.web_scraper = AIWebScrapingTool(openai_api_key=self.openai_api_key)
    
    async def run_complete_demo(self):
        """Run complete demo of the Smart Data Agent"""
        print("🚀 Smart Data Agent - AI Web Scraping Demo")
        print("=" * 60)
        
        # Demo data - sample medical bill text
        sample_medical_bill = """
        APOLLO HOSPITAL MEDICAL BILL
        ============================
        
        Patient: John Doe
        Date: 2024-01-15
        
        SERVICES:
        - Consultation (Cardiologist)     ₹2,000
        - ECG Test                        ₹800
        - Blood Test (CBC)                ₹500
        - X-Ray Chest                     ₹1,200
        - Echocardiogram                  ₹3,500
        - Paracetamol 500mg (10 tablets)  ₹200
        - Atorvastatin 20mg (30 tablets)  ₹450
        - Room Charges (1 day)            ₹5,000
        
        TOTAL: ₹13,650
        """
        
        # Step 1: Test Entity Extraction
        print("\n📋 Step 1: Entity Extraction")
        print("-" * 30)
        
        entity_result = await self.smart_agent.extract_entities(
            raw_text=sample_medical_bill,
            document_type="medical_bill"
        )
        
        if entity_result.success:
            print(f"✅ Extracted {entity_result.total_entities} entities")
            print(f"📊 Confidence: {entity_result.confidence:.2f}")
            
            for category, entities in entity_result.entities.items():
                if entities:
                    print(f"   {category}: {entities[:3]}...")  # Show first 3
        else:
            print(f"❌ Entity extraction failed: {entity_result.error}")
            return
        
        # Step 2: Test Source Mapping
        print("\n🗺️  Step 2: Source Mapping")
        print("-" * 30)
        
        relevant_sources = await self.entity_mapper.map_entities_to_sources(
            entities=entity_result.entities,
            document_type="medical_bill",
            state_code="DL"  # Delhi
        )
        
        print(f"✅ Mapped to {len(relevant_sources)} government sources:")
        for source_name, source_url in list(relevant_sources.items())[:5]:  # Show first 5
            print(f"   📍 {source_name}: {source_url}")
        
        # Step 3: Test Priority Ordering
        print("\n🎯 Step 3: Priority Source Ordering")
        print("-" * 30)
        
        priority_sources = await self.entity_mapper.get_priority_sources(
            entities=entity_result.entities,
            document_type="medical_bill"
        )
        
        print(f"✅ Priority ordered sources ({len(priority_sources)}):")
        for i, source_url in enumerate(priority_sources[:3], 1):
            print(f"   {i}. {source_url}")
        
        # Step 4: Test AI Web Scraping (with demo URL)
        print("\n🕸️  Step 4: AI Web Scraping Test")
        print("-" * 30)
        
        # Use a simple test URL for demonstration
        test_url = "https://httpbin.org/html"  # Simple HTML page for testing
        
        scraping_result = await self.web_scraper.scrape_government_data(
            url=test_url,
            entities=entity_result.entities,
            schema_type="medical_rates"
        )
        
        print(f"✅ Scraping result:")
        print(f"   Success: {scraping_result.success}")
        print(f"   Strategy: {scraping_result.strategy_used}")
        print(f"   Confidence: {scraping_result.confidence:.2f}")
        if scraping_result.error:
            print(f"   Error: {scraping_result.error}")
        
        # Step 5: Test Complete Data Fetching Pipeline
        print("\n🔄 Step 5: Complete Data Fetching Pipeline")
        print("-" * 30)
        
        # Note: This would try to scrape real government sites
        # For demo, we'll simulate the process
        print("⚠️  Simulating complete pipeline (government sites may block scraping)")
        
        data_result = await self.smart_agent.fetch_relevant_data(
            document_type="medical_bill",
            raw_text=sample_medical_bill,
            state_code="DL",
            max_sources=2  # Limit to 2 sources for demo
        )
        
        print(f"✅ Data fetching result:")
        print(f"   Success: {data_result.success}")
        print(f"   Sources scraped: {data_result.sources_scraped}")
        print(f"   Sources successful: {data_result.sources_successful}")
        print(f"   Total data points: {data_result.total_data_points}")
        print(f"   Processing time: {data_result.processing_time_ms}ms")
        
        if data_result.error:
            print(f"   Error: {data_result.error}")
        
        # Step 6: Show Architecture Benefits
        print("\n🏗️  Step 6: Architecture Benefits")
        print("-" * 30)
        
        print("✨ New AI Web Scraping Architecture Benefits:")
        print("   🎯 Smart entity extraction from any document")
        print("   🗺️  Automatic mapping to relevant government sources")
        print("   🤖 AI-powered scraping with multiple fallback strategies")
        print("   ⚡ Real-time data fetching vs static JSON files")
        print("   🔄 Extensible to any domain (insurance, pharma, etc.)")
        print("   📊 Confidence scoring and validation")
        print("   💾 Intelligent caching for performance")
        
        # Health check
        print("\n🏥 Health Check")
        print("-" * 30)
        
        health = await self.smart_agent.health_check()
        print(f"✅ Agent Status: {health['status']}")
        print(f"🔗 OpenAI Connection: {health['openai_connection']}")
        print(f"📦 Cache Available: {health['cache_available']}")
        
    async def demo_comparison(self):
        """Demo comparing old vs new architecture"""
        print("\n🆚 Architecture Comparison")
        print("=" * 60)
        
        print("📊 OLD ARCHITECTURE (Static Data):")
        print("   1. Document → OCR → Text")
        print("   2. Fixed JSON files (cghs_rates_2023.json)")
        print("   3. Static rate lookups")
        print("   4. Manual updates required")
        print("   5. Limited to pre-loaded data")
        
        print("\n🚀 NEW ARCHITECTURE (AI Web Scraping):")
        print("   1. Document → OCR → Entity Extraction")
        print("   2. Smart Source Mapping")
        print("   3. AI Web Scraping (GPT-4 Vision + HTML)")
        print("   4. Real-time data fetching")
        print("   5. Extensible to any domain")
        
        print("\n💡 Key Improvements:")
        print("   ✅ Always fresh, up-to-date data")
        print("   ✅ No manual data maintenance") 
        print("   ✅ Handles new procedures/drugs automatically")
        print("   ✅ Multi-domain support (medical, insurance, etc.)")
        print("   ✅ Intelligent fallback strategies")
        print("   ✅ Confidence-based validation")
    
    async def demo_entity_extraction(self):
        """Demo entity extraction with different document types"""
        print("\n🔍 Entity Extraction Demo")
        print("=" * 60)
        
        # Different document types
        documents = {
            "medical_bill": """
            AIIMS Medical Bill
            Consultation - Cardiology: ₹500
            ECG Test: ₹300  
            Blood Test - CBC: ₹250
            Chest X-Ray: ₹400
            """,
            
            "pharmacy_invoice": """
            MedPlus Pharmacy Invoice
            Paracetamol 500mg (10 tablets): ₹25
            Amoxicillin 250mg (21 capsules): ₹180
            Vitamin D3 (30 tablets): ₹320
            """,
            
            "insurance_claim": """
            Health Insurance Claim
            Policy Number: POL123456
            Hospitalization charges: ₹45,000
            ICU charges: ₹15,000
            Surgery charges: ₹25,000
            """
        }
        
        for doc_type, content in documents.items():
            print(f"\n📄 {doc_type.upper()}:")
            print("-" * 30)
            
            result = await self.smart_agent.extract_entities(content, doc_type)
            
            if result.success:
                print(f"✅ Entities extracted (confidence: {result.confidence:.2f})")
                for category, entities in result.entities.items():
                    if entities:
                        print(f"   {category}: {entities}")
            else:
                print(f"❌ Failed: {result.error}")
    
    async def demo_source_mapping(self):
        """Demo source mapping for different entity types"""
        print("\n🗺️  Source Mapping Demo")
        print("=" * 60)
        
        # Test different entity combinations
        test_cases = [
            {
                "name": "Medical Procedures",
                "entities": {
                    "procedures": ["consultation", "surgery", "x-ray"],
                    "specialties": ["cardiology", "orthopedic"]
                },
                "doc_type": "medical_bill"
            },
            {
                "name": "Pharmacy Items", 
                "entities": {
                    "medications": ["paracetamol", "amoxicillin", "vitamin"],
                    "drugs": ["antibiotics", "painkillers"]
                },
                "doc_type": "pharmacy_invoice"
            },
            {
                "name": "Insurance Claims",
                "entities": {
                    "services": ["hospitalization", "icu", "surgery"],
                    "insurance": ["policy", "claim", "premium"]
                },
                "doc_type": "insurance_claim"
            }
        ]
        
        for test_case in test_cases:
            print(f"\n📋 {test_case['name']}:")
            print("-" * 30)
            
            sources = await self.entity_mapper.map_entities_to_sources(
                entities=test_case["entities"],
                document_type=test_case["doc_type"],
                state_code="DL"
            )
            
            print(f"✅ Mapped to {len(sources)} sources:")
            for source_name, url in list(sources.items())[:3]:
                print(f"   📍 {source_name}")
                print(f"      {url}")


async def main():
    """Main demo function"""
    try:
        demo = SmartDataDemo()
        
        # Run different demo scenarios
        await demo.run_complete_demo()
        await demo.demo_comparison()
        await demo.demo_entity_extraction()
        await demo.demo_source_mapping()
        
        print("\n🎉 Demo completed successfully!")
        print("💡 The Smart Data Agent is ready for integration!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("🔧 Make sure OPENAI_API_KEY is set and required packages are installed")


if __name__ == "__main__":
    # Check configuration
    if not check_required_config():
        print("💡 To get started:")
        print("   1. Copy env.example to .env: cp env.example .env")
        print("   2. Edit .env and add your OpenAI API key")
        print("   3. Run the demo again")
        sys.exit(1)
    
    print("🚀 Starting Smart Data Agent Demo...")
    asyncio.run(main()) 