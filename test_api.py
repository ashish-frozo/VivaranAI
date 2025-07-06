#!/usr/bin/env python3
"""
Simple test script for MedBillGuardAgent API endpoints.
Run this after starting the FastAPI server to test basic functionality.
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", **kwargs):
    """Test an API endpoint and return the response."""
    print(f"\n{'='*50}")
    print(f"Testing {method} {endpoint}")
    print('='*50)
    
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=30)
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", timeout=30, **kwargs)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"Error: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def main():
    """Run all API tests."""
    print("MedBillGuardAgent API Test Suite")
    print("Make sure the server is running on http://localhost:8000")
    
    # Test 1: Root endpoint
    test_endpoint("/")
    
    # Test 2: Health check
    test_endpoint("/healthz")
    
    # Test 3: API documentation
    test_endpoint("/api/docs")
    
    # Test 4: Debug example
    print("\nTesting debug example (may take a few seconds)...")
    start_time = time.time()
    result = test_endpoint("/debug/example")
    if result:
        processing_time = time.time() - start_time
        print(f"Total test time: {processing_time:.2f}s")
        if 'processing_time_ms' in result:
            print(f"Server processing time: {result['processing_time_ms']}ms")
    
    # Test 5: Analyze endpoint (if fixture file exists)
    fixture_paths = [
        "fixtures/example.pdf",
        "fixtures/cghs_sample_bill.pdf", 
        "fixtures/pharmacy_invoice.pdf"
    ]
    
    fixture_file = None
    for path in fixture_paths:
        if Path(path).exists():
            fixture_file = path
            break
    
    if fixture_file:
        print(f"\nTesting analyze endpoint with {fixture_file}...")
        with open(fixture_file, 'rb') as f:
            files = {'file': f}
            data = {'state_code': 'DL'}
            test_endpoint("/analyze", method="POST", files=files, data=data)
    else:
        print("\nSkipping analyze endpoint test - no fixture file found")
        print("Create a fixture file in fixtures/ directory to test file upload")
    
    print(f"\n{'='*50}")
    print("API Testing Complete!")
    print('='*50)

if __name__ == "__main__":
    main() 