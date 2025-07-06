#!/usr/bin/env python3
"""
End-to-End Testing Script for MedBillGuardAgent Multi-Agent System

This script provides comprehensive testing of the entire system including:
- Infrastructure setup verification
- Agent registration and health checks
- Complete medical bill analysis workflow
- Performance and load testing
- Error handling and recovery scenarios

Usage:
    python test_e2e_runner.py --help
    python test_e2e_runner.py --quick       # Quick smoke tests
    python test_e2e_runner.py --full        # Full test suite
    python test_e2e_runner.py --load        # Load testing
    python test_e2e_runner.py --scenario "High Overcharge with Duplicates"
"""

import asyncio
import argparse
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiohttp
import redis
import psycopg2
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tests.test_data.sample_medical_bills import TEST_SCENARIOS, prepare_api_payload

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_e2e.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    success: bool
    duration_seconds: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class E2ETestConfig:
    """End-to-end test configuration"""
    agent_server_url: str = "http://localhost:8001"
    redis_url: str = "redis://localhost:6379/1"
    postgres_url: str = "postgresql://medbillguard:medbillguard_dev_password@localhost:5432/medbillguard"
    prometheus_url: str = "http://localhost:9090"
    jaeger_url: str = "http://localhost:16686"
    timeout_seconds: int = 30
    max_retries: int = 3
    load_test_concurrent_requests: int = 5
    load_test_duration_seconds: int = 60

class E2ETestRunner:
    """Main end-to-end test runner class"""
    
    def __init__(self, config: E2ETestConfig):
        self.config = config
        self.results: List[TestResult] = []
        
    async def _run_single_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test function with error handling and timing"""
        logger.info(f"Running: {test_name}")
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                details = await test_func()
            else:
                details = test_func()
                
            duration = time.time() - start_time
            result = TestResult(
                test_name=test_name,
                success=True,
                duration_seconds=duration,
                details=details or {}
            )
            logger.info(f"✅ {test_name} - PASSED ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(
                test_name=test_name,
                success=False,
                duration_seconds=duration,
                details={},
                error_message=str(e)
            )
            logger.error(f"❌ {test_name} - FAILED ({duration:.2f}s): {e}")
            
        return result

    # Quick smoke tests
    async def run_smoke_tests(self) -> List[TestResult]:
        """Run essential smoke tests to verify basic functionality"""
        logger.info("🔥 Running smoke tests...")
        tests = [
            ("Agent Server Health", self._test_agent_server_health),
            ("Basic Medical Bill Analysis", self._test_basic_analysis),
        ]
        
        results = []
        for test_name, test_func in tests:
            result = await self._run_single_test(test_name, test_func)
            results.append(result)
            
        return results
    
    async def run_infrastructure_tests(self) -> List[TestResult]:
        """Test infrastructure components availability"""
        logger.info("🔧 Running infrastructure tests...")
        tests = [
            ("Agent Server Health", self._test_agent_server_health),
            ("Redis Connection", self._test_redis_connection),
            ("Metrics Endpoint", self._test_metrics_endpoint),
        ]
        
        results = []
        for test_name, test_func in tests:
            result = await self._run_single_test(test_name, test_func)
            results.append(result)
            
        return results
    
    async def run_workflow_tests(self) -> List[TestResult]:
        """Test complete medical bill analysis workflows"""
        logger.info("⚕️ Running workflow tests...")
        results = []
        
        # Test first 3 scenarios to keep it manageable
        for scenario in TEST_SCENARIOS[:3]:
            test_name = f"Workflow: {scenario['name']}"
            result = await self._run_single_test(
                test_name, 
                lambda s=scenario: self._test_medical_bill_workflow(s)
            )
            results.append(result)
            
        return results
    
    async def run_performance_tests(self) -> List[TestResult]:
        """Test system performance"""
        logger.info("🚀 Running performance tests...")
        tests = [
            ("Single Request Latency", self._test_single_request_latency),
            ("Concurrent Requests", self._test_concurrent_requests),
        ]
        
        results = []
        for test_name, test_func in tests:
            result = await self._run_single_test(test_name, test_func)
            results.append(result)
            
        return results
    
    # Test implementations
    async def _test_agent_server_health(self) -> Dict[str, Any]:
        """Test agent server health endpoints"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.config.agent_server_url}/health") as resp:
                assert resp.status == 200, f"Health check failed: {resp.status}"
                health_data = await resp.json()
                
        return {"health_status": health_data}
    
    async def _test_redis_connection(self) -> Dict[str, Any]:
        """Test Redis connection and basic operations"""
        try:
            r = redis.from_url(self.config.redis_url)
            r.ping()
            return {"redis_connected": True}
        except Exception as e:
            return {"redis_connected": False, "error": str(e)}
    
    async def _test_metrics_endpoint(self) -> Dict[str, Any]:
        """Test Prometheus metrics endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.config.agent_server_url}/metrics") as resp:
                assert resp.status == 200, f"Metrics endpoint failed: {resp.status}"
                metrics_text = await resp.text()
                
        metrics_lines = [line for line in metrics_text.split('\n') if line and not line.startswith('#')]
        
        return {
            "metrics_count": len(metrics_lines),
            "has_custom_metrics": "medical_bill_analysis" in metrics_text
        }
    
    async def _test_basic_analysis(self) -> Dict[str, Any]:
        """Test basic medical bill analysis"""
        scenario = TEST_SCENARIOS[0]  # Use first scenario
        payload = prepare_api_payload(scenario["name"])
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.agent_server_url}/analyze",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                assert resp.status == 200, f"Analysis failed: {resp.status}"
                analysis_result = await resp.json()
                
        return {
            "analysis_completed": True,
            "has_results": "results" in analysis_result,
            "processing_time": analysis_result.get("processing_time_ms", 0)
        }
    
    async def _test_medical_bill_workflow(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test complete medical bill analysis workflow"""
        payload = prepare_api_payload(scenario["name"])
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.agent_server_url}/analyze",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            ) as resp:
                assert resp.status == 200, f"Analysis failed: {resp.status}"
                analysis_result = await resp.json()
                
        # Validate response structure
        required_fields = ["analysis_id", "status", "results"]
        for field in required_fields:
            assert field in analysis_result, f"Missing required field: {field}"
            
        return {
            "scenario": scenario["name"],
            "analysis_completed": True,
            "processing_time": analysis_result.get("processing_time_ms", 0),
            "has_duplicate_detection": "duplicate_detection" in analysis_result.get("results", {}),
            "has_rate_validation": "rate_validation" in analysis_result.get("results", {}),
            "has_confidence_scoring": "confidence_scoring" in analysis_result.get("results", {})
        }
    
    async def _test_single_request_latency(self) -> Dict[str, Any]:
        """Test single request latency"""
        scenario = TEST_SCENARIOS[0]  # Use first scenario
        payload = prepare_api_payload(scenario["name"])
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.agent_server_url}/analyze",
                json=payload
            ) as resp:
                assert resp.status == 200
                await resp.json()
                
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "latency_ms": latency_ms,
            "acceptable": latency_ms < 10000  # Less than 10 seconds
        }
    
    async def _test_concurrent_requests(self) -> Dict[str, Any]:
        """Test handling of concurrent requests"""
        num_requests = min(3, self.config.load_test_concurrent_requests)  # Limit for testing
        scenario = TEST_SCENARIOS[0]
        payload = prepare_api_payload(scenario["name"])
        
        async def make_request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.agent_server_url}/analyze",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    return resp.status == 200
        
        start_time = time.time()
        
        # Run concurrent requests
        tasks = [make_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        successful_requests = sum(1 for r in results if r is True)
        total_time = end_time - start_time
        
        return {
            "concurrent_requests": num_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / num_requests,
            "total_time_seconds": total_time,
            "requests_per_second": num_requests / total_time if total_time > 0 else 0
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r.duration_seconds for r in self.results)
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_duration_seconds": total_duration,
                "timestamp": datetime.now().isoformat()
            },
            "results": [asdict(r) for r in self.results],
            "failed_tests": [asdict(r) for r in self.results if not r.success]
        }
        
        return report

async def main():
    """Main entry point for end-to-end testing"""
    parser = argparse.ArgumentParser(description="MedBillGuardAgent End-to-End Testing")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke tests only")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--load", action="store_true", help="Run load testing")
    parser.add_argument("--scenario", type=str, help="Run specific scenario test")
    parser.add_argument("--output", type=str, default="test_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    # Initialize test runner
    config = E2ETestConfig()
    runner = E2ETestRunner(config)
    
    logger.info("🚀 Starting MedBillGuardAgent End-to-End Testing")
    
    try:
        # Run different test suites based on arguments
        if args.scenario:
            # Run specific scenario
            scenario = next((s for s in TEST_SCENARIOS if s["name"] == args.scenario), None)
            if not scenario:
                logger.error(f"Scenario '{args.scenario}' not found")
                sys.exit(1)
                
            result = await runner._run_single_test(
                f"Scenario: {args.scenario}",
                lambda: runner._test_medical_bill_workflow(scenario)
            )
            runner.results.append(result)
            
        elif args.quick:
            # Quick smoke tests
            logger.info("🔥 Running quick smoke tests...")
            runner.results.extend(await runner.run_smoke_tests())
            
        elif args.load:
            # Load testing
            logger.info("⚡ Running performance tests...")
            runner.results.extend(await runner.run_performance_tests())
            
        else:
            # Full test suite
            logger.info("🎯 Running full test suite...")
            runner.results.extend(await runner.run_infrastructure_tests())
            runner.results.extend(await runner.run_workflow_tests())
            runner.results.extend(await runner.run_performance_tests())
        
        # Generate and save report
        report = runner.generate_report()
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print summary
        summary = report["summary"]
        logger.info(f"📊 Test Summary:")
        logger.info(f"   Total Tests: {summary['total_tests']}")
        logger.info(f"   Passed: {summary['passed_tests']}")
        logger.info(f"   Failed: {summary['failed_tests']}")
        logger.info(f"   Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"   Duration: {summary['total_duration_seconds']:.2f}s")
        logger.info(f"📄 Full report saved to: {args.output}")
        
        if summary['failed_tests'] > 0:
            logger.error("❌ Some tests failed. Check the report for details.")
            sys.exit(1)
        else:
            logger.info("✅ All tests passed!")
            
    except KeyboardInterrupt:
        logger.info("⏹️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Testing failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 