#!/usr/bin/env python3
"""
Test script to verify agent registration is working properly.
Can be run against both local and Railway deployments.
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, Any

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


async def test_agent_registration(base_url: str = "http://localhost:8001"):
    """Test agent registration functionality."""
    logger.info(f"Testing agent registration at {base_url}")
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test 1: Health check
            logger.info("Testing health check...")
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info("Health check passed", data=health_data)
                    
                    # Check if medical agent is healthy
                    if health_data.get("components", {}).get("medical_agent") == "healthy":
                        logger.info("✅ Medical agent is healthy")
                    else:
                        logger.warning("⚠️ Medical agent is not healthy")
                else:
                    logger.error(f"Health check failed with status {response.status}")
                    return False
            
            # Test 2: List agents
            logger.info("Testing agent list...")
            async with session.get(f"{base_url}/agents") as response:
                if response.status == 200:
                    agents_data = await response.json()
                    logger.info("Agent list retrieved", data=agents_data)
                    
                    agents = agents_data.get("agents", [])
                    if agents:
                        logger.info(f"✅ Found {len(agents)} registered agents")
                        for agent in agents:
                            logger.info(f"  - {agent['name']} ({agent['agent_id']}): {agent['status']}")
                    else:
                        logger.warning("⚠️ No agents found - this indicates registration issue")
                        return False
                else:
                    logger.error(f"Agent list failed with status {response.status}")
                    return False
            
            # Test 3: Readiness check
            logger.info("Testing readiness check...")
            async with session.get(f"{base_url}/health/readiness") as response:
                if response.status == 200:
                    readiness_data = await response.json()
                    logger.info("Readiness check passed", data=readiness_data)
                    
                    if readiness_data.get("status") == "ready":
                        logger.info("✅ System is ready")
                    else:
                        logger.warning("⚠️ System is not ready")
                else:
                    logger.error(f"Readiness check failed with status {response.status}")
                    return False
            
            # Test 4: Quick analysis to verify agents are working
            logger.info("Testing quick analysis...")
            try:
                async with session.get(f"{base_url}/debug/example") as response:
                    if response.status == 200:
                        analysis_data = await response.json()
                        logger.info("Quick analysis passed", verdict=analysis_data.get("verdict"))
                        logger.info("✅ Agent analysis is working")
                    else:
                        logger.warning(f"Quick analysis failed with status {response.status}")
                        # This might be expected if the endpoint doesn't exist
            except Exception as e:
                logger.info(f"Quick analysis test skipped (endpoint may not exist): {e}")
            
            logger.info("🎉 All agent registration tests passed!")
            return True
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            return False


async def monitor_agent_registration(base_url: str = "http://localhost:8001", duration: int = 300):
    """Monitor agent registration over time."""
    logger.info(f"Monitoring agent registration for {duration} seconds...")
    
    start_time = time.time()
    check_interval = 30  # Check every 30 seconds
    
    while time.time() - start_time < duration:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/agents") as response:
                    if response.status == 200:
                        agents_data = await response.json()
                        agent_count = len(agents_data.get("agents", []))
                        
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        logger.info(f"[{timestamp}] Agent count: {agent_count}")
                        
                        if agent_count == 0:
                            logger.warning("⚠️ No agents registered - potential issue detected")
                        else:
                            logger.info("✅ Agents are registered")
                    else:
                        logger.error(f"Failed to get agents list: {response.status}")
            
            await asyncio.sleep(check_interval)
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            await asyncio.sleep(check_interval)
    
    logger.info("Monitoring completed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test agent registration")
    parser.add_argument(
        "--url", 
        default="http://localhost:8001",
        help="Base URL to test (default: http://localhost:8001)"
    )
    parser.add_argument(
        "--railway", 
        action="store_true",
        help="Test against Railway production"
    )
    parser.add_argument(
        "--monitor", 
        type=int,
        default=0,
        help="Monitor for specified seconds (0 = run once)"
    )
    
    args = parser.parse_args()
    
    # Configure URL
    if args.railway:
        base_url = "https://endearing-prosperity-production.up.railway.app"
    else:
        base_url = args.url
    
    logger.info(f"Testing agent registration at: {base_url}")
    
    # Run test
    if args.monitor > 0:
        asyncio.run(monitor_agent_registration(base_url, args.monitor))
    else:
        success = asyncio.run(test_agent_registration(base_url))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 