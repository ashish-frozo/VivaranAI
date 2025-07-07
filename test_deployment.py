#!/usr/bin/env python3
"""
Test script for deployed VivaranAI MedBillGuardAgent on Railway
"""

import requests
import json
import base64
import time

# Configuration
BASE_URL = "https://endearing-prosperity-production.up.railway.app"
TIMEOUT = 30  # seconds

def test_health():
    """Test health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health: {health_data['status']}")
            print(f"⏱️  Uptime: {health_data['uptime_seconds']:.1f}s")
            print("📊 Components:")
            for component, status in health_data['components'].items():
                emoji = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
                print(f"  {emoji} {component}: {status}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
    
    print("-" * 50)

def test_agents():
    """Test agents endpoint"""
    print("🤖 Testing Agents Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/agents", timeout=TIMEOUT)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            agents_data = response.json()
            print(f"✅ Found {agents_data['total']} agents")
            
            for agent in agents_data['agents']:
                print(f"  🏥 {agent['name']} ({agent['agent_id']})")
                print(f"     Status: {agent['status']}")
                print(f"     Capabilities: {len(agent['capabilities'])}")
        else:
            print(f"❌ Agents check failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Agents check error: {e}")
    
    print("-" * 50)

def test_medical_bill_analysis():
    """Test medical bill analysis"""
    print("🏥 Testing Medical Bill Analysis...")
    
    # Create a sample medical bill
    sample_bill = """APOLLO HOSPITALS
Medical Bill Receipt
Date: 2024-01-15
Patient: John Doe
ID: 12345

SERVICES:
1. Consultation Fee - Dr. Smith    ₹ 800
2. Blood Test - CBC Complete       ₹ 1200  
3. X-Ray Chest                     ₹ 600
4. Medicine - Paracetamol 500mg    ₹ 150
5. Room Charges (1 day)            ₹ 2000

TOTAL: ₹ 4750"""
    
    # Encode to base64
    encoded_bill = base64.b64encode(sample_bill.encode()).decode()
    
    # Prepare request data
    analysis_data = {
        "file_content": encoded_bill,
        "doc_id": f"test_doc_{int(time.time())}",
        "user_id": "test_user",
        "language": "english",
        "state_code": "delhi",
        "insurance_type": "cghs",
        "file_format": "text"
    }
    
    try:
        print("📤 Sending analysis request...")
        response = requests.post(
            f"{BASE_URL}/analyze", 
            json=analysis_data, 
            timeout=TIMEOUT
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis completed successfully!")
            print(f"📄 Document ID: {result.get('doc_id')}")
            print(f"📊 Verdict: {result.get('verdict')}")
            print(f"💰 Total Amount: ₹{result.get('total_bill_amount', 'N/A')}")
            print(f"⚠️  Overcharge: ₹{result.get('total_overcharge', 'N/A')}")
            print(f"🎯 Confidence: {result.get('confidence_score', 'N/A')}")
            print(f"⏱️  Processing Time: {result.get('processing_time_seconds', 'N/A')}s")
            
            if result.get('red_flags'):
                print("🚨 Red Flags:")
                for flag in result['red_flags']:
                    print(f"  - {flag}")
            
            if result.get('recommendations'):
                print("💡 Recommendations:")
                for rec in result['recommendations']:
                    print(f"  - {rec}")
                    
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            if response.text:
                print(f"Response: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Analysis error: {e}")
    
    print("-" * 50)

def test_readiness():
    """Test readiness endpoint"""
    print("🚀 Testing Readiness...")
    try:
        response = requests.get(f"{BASE_URL}/health/readiness", timeout=TIMEOUT)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Service is ready!")
        else:
            print(f"⚠️  Service not ready: {response.status_code}")
            if response.text:
                print(f"Response: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Readiness check error: {e}")
    
    print("-" * 50)

def main():
    """Run all tests"""
    print("🎯 VivaranAI MedBillGuardAgent Deployment Test")
    print(f"🌐 Base URL: {BASE_URL}")
    print("=" * 50)
    
    # Run tests
    test_health()
    test_agents()
    test_readiness()
    test_medical_bill_analysis()
    
    print("🎉 Testing completed!")
    print("\n📋 Summary:")
    print("✅ If all tests passed, your deployment is working correctly!")
    print("⚠️  If some tests failed, check the error messages above.")
    print("📚 For more testing options, see: TESTING_DEPLOYMENT_GUIDE.md")

if __name__ == "__main__":
    main() 