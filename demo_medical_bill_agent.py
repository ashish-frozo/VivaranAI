"""
Medical Bill Agent Demo - Comprehensive demonstration of async medical bill analysis.

This demo showcases the MedicalBillAgent with realistic medical bill scenarios,
including document processing, rate validation, duplicate detection, prohibited
item detection, and confidence scoring.
"""

import asyncio
import base64
import json
import logging
import structlog
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from agents.medical_bill_agent import MedicalBillAgent
from agents.redis_state import RedisStateManager


# Configure structured logging for demo
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class MedicalBillAgentDemo:
    """Comprehensive demo of MedicalBillAgent capabilities."""
    
    def __init__(self):
        """Initialize the demo with agent and sample data."""
        self.agent = None
        
        # Sample medical bill scenarios
        self.sample_bills = {
            "clean_bill": self._generate_clean_bill_data(),
            "overcharged_bill": self._generate_overcharged_bill_data(),
            "complex_bill": self._generate_complex_bill_data()
        }
    
    async def setup(self):
        """Setup the demo environment."""
        logger.info("Setting up Medical Bill Agent Demo")
        
        try:
            # Initialize the medical bill agent
            self.agent = MedicalBillAgent(
                redis_url="redis://localhost:6379/1",
                openai_api_key=None  # Will use env variable or mock for demo
            )
            
            # Test Redis connection
            state_manager = RedisStateManager("redis://localhost:6379/1")
            await state_manager.ping()
            
            logger.info("Demo setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Demo setup failed: {str(e)}")
            return False
    
    async def demonstrate_analysis_workflow(self):
        """Demonstrate the complete medical bill analysis workflow."""
        logger.info("=== Medical Bill Analysis Workflow Demo ===")
        
        for bill_name, bill_data in self.sample_bills.items():
            logger.info(f"\\nAnalyzing {bill_name}...")
            
            try:
                # Analyze the medical bill
                result = await self.agent.analyze_medical_bill(
                    file_content=bill_data["file_content"],
                    doc_id=bill_data["doc_id"],
                    user_id="demo_user",
                    language=bill_data.get("language", "english"),
                    state_code=bill_data.get("state_code"),
                    insurance_type=bill_data.get("insurance_type", "cghs"),
                    file_format=bill_data.get("file_format", "pdf")
                )
                
                # Display results
                self._display_analysis_result(bill_name, result)
                
            except Exception as e:
                logger.error(f"Analysis failed for {bill_name}: {str(e)}")
    
    async def demonstrate_individual_tools(self):
        """Demonstrate individual tool capabilities."""
        logger.info("\\n=== Individual Tool Demonstrations ===")
        
        # Sample line items for tool testing
        sample_items = [
            {
                "description": "Specialist Consultation - Cardiology",
                "quantity": 1,
                "unit_price": 1200.0,
                "total_amount": 1200.0,
                "item_type": "consultation",
                "confidence": 0.95
            },
            {
                "description": "Complete Blood Count (CBC)",
                "quantity": 1,
                "unit_price": 800.0,
                "total_amount": 800.0,
                "item_type": "diagnostic",
                "confidence": 0.92
            },
            {
                "description": "Complete Blood Count (CBC)",  # Duplicate
                "quantity": 1,
                "unit_price": 800.0,
                "total_amount": 800.0,
                "item_type": "diagnostic",
                "confidence": 0.90
            }
        ]
        
        # Demonstrate Rate Validator Tool
        logger.info("\\n--- Rate Validator Tool ---")
        rate_result = await self.agent.rate_validator_tool(
            line_items=sample_items,
            state_code="DL",
            validation_sources=["cghs", "esi"]
        )
        logger.info(f"Rate validation result: {json.dumps(rate_result, indent=2)}")
        
        # Demonstrate Duplicate Detector Tool
        logger.info("\\n--- Duplicate Detector Tool ---")
        duplicate_result = await self.agent.duplicate_detector_tool(
            line_items=sample_items,
            similarity_threshold=0.8
        )
        logger.info(f"Duplicate detection result: {json.dumps(duplicate_result, indent=2)}")
        
        # Demonstrate Confidence Scorer Tool
        logger.info("\\n--- Confidence Scorer Tool ---")
        confidence_result = await self.agent.confidence_scorer_tool(
            analysis_results={
                "rate_validation": rate_result,
                "duplicate_detection": duplicate_result
            },
            processing_stats={
                "ocr_confidence": 94.5,
                "errors_encountered": [],
                "pages_processed": 1,
                "line_items_found": 3
            },
            red_flags=rate_result.get("red_flags", []) + duplicate_result.get("red_flags", [])
        )
        logger.info(f"Confidence scoring result: {json.dumps(confidence_result, indent=2)}")
    
    async def demonstrate_error_handling(self):
        """Demonstrate error handling capabilities."""
        logger.info("\\n=== Error Handling Demonstrations ===")
        
        # Test with invalid file content
        logger.info("\\n--- Testing Invalid File Content ---")
        try:
            result = await self.agent.analyze_medical_bill(
                file_content=b"invalid_content",
                doc_id="error_test_001",
                user_id="demo_user"
            )
            logger.info(f"Invalid content result: {json.dumps(result, indent=2)}")
        except Exception as e:
            logger.error(f"Expected error for invalid content: {str(e)}")
        
        # Test with missing required parameters
        logger.info("\\n--- Testing Missing Parameters ---")
        try:
            from agents.base_agent import AgentContext, ModelHint
            import time
            
            context = AgentContext(
                doc_id="error_test_002",
                user_id="demo_user",
                correlation_id="error_demo",
                model_hint=ModelHint.STANDARD,
                start_time=time.time()
            )
            
            result = await self.agent.process_task(context, {})  # Missing file_content
            logger.info(f"Missing params result: {json.dumps(result, indent=2)}")
            
        except Exception as e:
            logger.error(f"Expected error for missing params: {str(e)}")
    
    async def demonstrate_health_check(self):
        """Demonstrate health check functionality."""
        logger.info("\\n=== Health Check Demonstration ===")
        
        health = await self.agent.health_check()
        logger.info(f"Agent health status: {json.dumps(health, indent=2)}")
    
    def _generate_clean_bill_data(self) -> Dict[str, Any]:
        """Generate sample data for a clean medical bill."""
        # In a real scenario, this would be actual PDF content
        sample_content = """
        MEDICAL BILL - APOLLO HOSPITAL
        Patient: John Doe
        Date: 2024-01-15
        
        Consultation - General Medicine: Rs. 500
        Blood Test - CBC: Rs. 300
        X-Ray Chest: Rs. 400
        
        Total: Rs. 1200
        """.encode('utf-8')
        
        return {
            "doc_id": "clean_bill_001",
            "file_content": sample_content,
            "language": "english",
            "state_code": "DL",
            "insurance_type": "cghs",
            "file_format": "pdf",
            "description": "A clean medical bill with CGHS-compliant rates"
        }
    
    def _generate_overcharged_bill_data(self) -> Dict[str, Any]:
        """Generate sample data for an overcharged medical bill."""
        sample_content = """
        MEDICAL BILL - PREMIUM HOSPITAL
        Patient: Jane Smith
        Date: 2024-01-16
        
        Specialist Consultation - Cardiology: Rs. 2000
        Complete Blood Count (CBC): Rs. 800
        ECG: Rs. 600
        Room Charges (1 day): Rs. 5000
        
        Total: Rs. 8400
        """.encode('utf-8')
        
        return {
            "doc_id": "overcharged_bill_001",
            "file_content": sample_content,
            "language": "english",
            "state_code": "MH",
            "insurance_type": "cghs",
            "file_format": "pdf",
            "description": "A medical bill with significant overcharges"
        }
    
    def _generate_complex_bill_data(self) -> Dict[str, Any]:
        """Generate sample data for a complex medical bill with multiple issues."""
        sample_content = """
        MEDICAL BILL - MULTI SPECIALTY HOSPITAL
        Patient: Robert Johnson
        Date: 2024-01-17
        
        Emergency Consultation: Rs. 1500
        Complete Blood Count (CBC): Rs. 700
        Complete Blood Count (CBC): Rs. 700  # Duplicate
        X-Ray Chest: Rs. 800
        MRI Brain: Rs. 12000
        Room Charges (2 days): Rs. 8000
        Physiotherapy Session: Rs. 1000
        Medicines - Prohibited Drug XYZ: Rs. 2000
        
        Total: Rs. 26700
        """.encode('utf-8')
        
        return {
            "doc_id": "complex_bill_001",
            "file_content": sample_content,
            "language": "english",
            "state_code": "KA",
            "insurance_type": "esi",
            "file_format": "pdf",
            "description": "A complex bill with overcharges, duplicates, and prohibited items"
        }
    
    def _display_analysis_result(self, bill_name: str, result: Dict[str, Any]):
        """Display analysis result in a formatted way."""
        logger.info(f"\\n📋 Analysis Results for {bill_name.upper()}:")
        
        if not result.get("success"):
            logger.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return
        
        # Basic info
        logger.info(f"📄 Document ID: {result.get('doc_id')}")
        logger.info(f"✅ Status: {result.get('verdict', 'unknown').upper()}")
        logger.info(f"🎯 Confidence: {result.get('confidence_score', 0):.2f}")
        
        # Financial summary
        total_amount = result.get('total_bill_amount', 0)
        overcharge = result.get('total_overcharge', 0)
        overcharge_pct = result.get('overcharge_percentage', 0)
        
        logger.info(f"💰 Total Bill Amount: ₹{total_amount:,.2f}")
        logger.info(f"⚠️ Total Overcharge: ₹{overcharge:,.2f} ({overcharge_pct:.1f}%)")
        
        # Red flags
        red_flags = result.get('red_flags', [])
        logger.info(f"🚩 Red Flags Found: {len(red_flags)}")
        
        for i, flag in enumerate(red_flags[:3], 1):  # Show first 3 red flags
            logger.info(f"  {i}. {flag.get('type', 'unknown').title()}: {flag.get('item', 'N/A')}")
            logger.info(f"     Reason: {flag.get('reason', 'N/A')}")
            logger.info(f"     Amount: ₹{flag.get('overcharge_amount', 0):,.2f}")
        
        if len(red_flags) > 3:
            logger.info(f"  ... and {len(red_flags) - 3} more red flags")
        
        # Recommendations
        recommendations = result.get('recommendations', [])
        if recommendations:
            logger.info(f"💡 Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                logger.info(f"  {i}. {rec}")
        
        # Analysis summary
        summary = result.get('analysis_summary', {})
        logger.info(f"📊 Analysis Summary:")
        logger.info(f"  • Items Analyzed: {summary.get('items_analyzed', 0)}")
        logger.info(f"  • Rate Matches: {summary.get('rate_matches_found', 0)}")
        logger.info(f"  • Duplicates: {summary.get('duplicates_detected', 0)}")
        logger.info(f"  • Prohibited Items: {summary.get('prohibited_items_found', 0)}")
    
    async def cleanup(self):
        """Cleanup demo resources."""
        logger.info("Cleaning up demo resources...")
        
        if self.agent and hasattr(self.agent, 'redis_client'):
            try:
                await self.agent.redis_client.aclose()
            except:
                pass


async def main():
    """Main demo execution."""
    demo = MedicalBillAgentDemo()
    
    try:
        print("🏥 Medical Bill Agent Demo Starting...")
        
        # Setup
        if not await demo.setup():
            print("❌ Demo setup failed. Please ensure Redis is running.")
            return
        
        print("✅ Demo setup completed")
        
        # Run demonstrations
        await demo.demonstrate_health_check()
        await demo.demonstrate_analysis_workflow()
        await demo.demonstrate_individual_tools()
        await demo.demonstrate_error_handling()
        
        print("\\n🎉 Medical Bill Agent Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo execution failed: {str(e)}", exc_info=True)
        print("❌ Demo execution failed. Check logs for details.")
        
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 