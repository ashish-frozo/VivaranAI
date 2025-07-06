"""
Locust performance test file for MedBillGuardAgent
Run with: locust --host=http://localhost:8000
"""

from locust import HttpUser, task, between
import json
import base64


class MedBillGuardAgentUser(HttpUser):
    """Load test user for MedBillGuardAgent service."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts - setup any test data."""
        # Health check to ensure service is running
        self.client.get("/healthz")
    
    @task(10)
    def health_check(self):
        """Basic health check endpoint - most frequent task."""
        self.client.get("/healthz")
    
    @task(5)
    def debug_example(self):
        """Debug example endpoint for testing."""
        self.client.get("/debug/example")
    
    @task(3)
    def get_metrics(self):
        """Get Prometheus metrics."""
        self.client.get("/metrics")
    
    @task(1)
    def analyze_document(self):
        """Simulate document analysis request - heaviest task."""
        # Create a mock document analysis request
        test_payload = {
            "doc_id": "test-doc-123",
            "user_id": "test-user-456", 
            "s3_url": "s3://test-bucket/sample-bill.pdf",
            "doc_type": "hospital_bill",
            "language": "en",
            "priority": "normal"
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Locust-LoadTest/1.0"
        }
        
        # POST to analyze endpoint (will be implemented later)
        with self.client.post(
            "/api/v1/analyze", 
            json=test_payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 404:
                # Expected during development - endpoint not implemented yet
                response.success()
            elif response.status_code != 200:
                response.failure(f"Unexpected status: {response.status_code}")


class HighVolumeUser(HttpUser):
    """High-volume user simulating bulk processing."""
    
    wait_time = between(0.1, 0.5)  # Faster requests for load testing
    weight = 1  # Lower weight - fewer of these users
    
    @task
    def rapid_health_checks(self):
        """Rapid health checks to test rate limiting."""
        self.client.get("/healthz")


# Test configuration
if __name__ == "__main__":
    import locust.main
    locust.main.main() 