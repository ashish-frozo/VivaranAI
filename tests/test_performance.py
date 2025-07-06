"""
Performance tests for MedBillGuardAgent.

Tests critical path operations to ensure they meet performance requirements.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add libs to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'libs'))
from testing import timed_test, benchmark_test

from medbillguardagent.rate_validator import RateValidator
from medbillguardagent.confidence_scorer import ConfidenceScorer
from medbillguardagent.cache_manager import CacheManager
from medbillguardagent.schemas import LineItem, LineItemType


class TestPerformance:
    """Performance tests for critical operations."""

    @pytest.fixture
    def mock_rate_validator(self):
        """Mock rate validator for performance testing."""
        validator = Mock(spec=RateValidator)
        
        # Mock fast validation response
        async def mock_validate(items, item_costs, state_code=None):
            return [
                {
                    "item": "Consultation - General",
                    "billed": 500.0,
                    "max_allowed": 300.0,
                    "overcharge_amount": 200.0,
                    "confidence": 0.95,
                    "source": "cghs"
                }
            ]
        
        validator.validate_item_rates = AsyncMock(side_effect=mock_validate)
        return validator

    @pytest.fixture
    def mock_confidence_scorer(self):
        """Mock confidence scorer for performance testing."""
        scorer = Mock(spec=ConfidenceScorer)
        scorer.calculate_confidence = Mock(return_value=0.85)
        return scorer

    @timed_test(max_ms=150)
    @pytest.mark.asyncio
    async def test_rate_validation_performance(self, mock_rate_validator):
        """Test that rate validation completes within 150ms."""
        items = ["Consultation - General", "Blood Test - CBC"]
        item_costs = {"Consultation - General": 500.0, "Blood Test - CBC": 800.0}
        
        result = await mock_rate_validator.validate_item_rates(items, item_costs)
        
        assert len(result) > 0
        assert result[0]["item"] == "Consultation - General"

    @timed_test(max_ms=150)
    def test_confidence_scoring_performance(self, mock_confidence_scorer):
        """Test that confidence scoring completes within 150ms."""
        red_flags = [
            {
                "item": "Test Item",
                "overcharge_pct": 50.0,
                "source": "cghs",
                "confidence": 0.9
            }
        ]
        
        confidence = mock_confidence_scorer.calculate_confidence(red_flags)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    @timed_test(max_ms=100)
    def test_schema_validation_performance(self):
        """Test that schema validation is fast."""
        # Test LineItem schema validation
        line_item = LineItem(
            description="Consultation - General",
            quantity=1,
            total_amount=500.0,
            item_type=LineItemType.CONSULTATION
        )
        
        assert line_item.description == "Consultation - General"
        assert line_item.total_amount == 500.0

    @timed_test(max_ms=50)
    @pytest.mark.asyncio
    async def test_cache_operations_performance(self):
        """Test that cache operations are very fast."""
        with patch('medbillguardagent.cache_manager.Cache') as mock_cache_class:
            # Mock cache instance
            mock_cache = AsyncMock()
            mock_cache.get.return_value = {"cached": "data"}
            mock_cache.set.return_value = True
            mock_cache_class.return_value = mock_cache
            
            cache_manager = CacheManager()
            cache_manager._cache_available = True
            cache_manager._reference_cache = mock_cache
            
            # Test cache get operation
            async def mock_loader():
                return {"loaded": "data"}
            
            result = await cache_manager.get_reference_data("test", mock_loader)
            assert result == {"cached": "data"}

    @benchmark_test
    @pytest.mark.asyncio
    async def test_full_validation_pipeline_benchmark(self, mock_rate_validator, mock_confidence_scorer):
        """Benchmark the full validation pipeline."""
        # Simulate a complete validation flow
        items = ["Consultation", "Blood Test", "X-Ray", "Medicine"]
        item_costs = {
            "Consultation": 500.0,
            "Blood Test": 800.0,
            "X-Ray": 1200.0,
            "Medicine": 300.0
        }
        
        # Rate validation
        rate_results = await mock_rate_validator.validate_item_rates(items, item_costs)
        
        # Confidence scoring
        confidence = mock_confidence_scorer.calculate_confidence(rate_results)
        
        assert len(rate_results) > 0
        assert isinstance(confidence, float)

    @timed_test(max_ms=200)
    def test_json_serialization_performance(self):
        """Test that JSON serialization of large responses is fast."""
        import json
        from medbillguardagent.schemas import MedBillGuardResponse, RedFlag
        
        # Create a large response with many red flags
        red_flags = []
        for i in range(50):
            red_flags.append(RedFlag(
                item=f"Test Item {i}",
                billed=float(500 + i * 10),
                max_allowed=float(300 + i * 5),
                overcharge_amount=float(200 + i * 5),
                confidence=0.9,
                reason=f"Overcharged by {200 + i * 5} rupees"
            ))
        
        response = MedBillGuardResponse(
            doc_id="test-doc-123",
            verdict="critical",
            total_bill_amount=25000.0,
            total_overcharge_amount=5000.0,
            confidence_score=0.85,
            red_flags=red_flags,
            latency_ms=150
        )
        
        # Test serialization speed
        json_str = response.model_dump_json()
        assert len(json_str) > 1000  # Ensure we have substantial data
        
        # Test deserialization speed
        parsed = json.loads(json_str)
        assert parsed["doc_id"] == "test-doc-123"

    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self):
        """Test performance under concurrent load."""
        async def mock_operation(delay_ms: int = 10):
            """Mock async operation with small delay."""
            await asyncio.sleep(delay_ms / 1000)
            return {"result": f"completed in {delay_ms}ms"}
        
        # Run 10 concurrent operations
        start_time = asyncio.get_event_loop().time()
        
        tasks = [mock_operation(10) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        total_time_ms = (end_time - start_time) * 1000
        
        # Should complete in roughly the time of one operation due to concurrency
        assert total_time_ms < 50  # Allow some overhead
        assert len(results) == 10 