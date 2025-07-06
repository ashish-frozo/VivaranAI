"""
Tests for the confidence scoring engine.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock

from medbillguardagent.confidence_scorer import (
    ConfidenceScorer,
    ConfidenceScore,
    ConfidenceWeights,
    ConfidenceSource
)
from medbillguardagent.schemas import RedFlag, Verdict, LineItemType


class TestConfidenceScorer:
    """Test suite for ConfidenceScorer class."""

    @pytest.fixture
    def sample_red_flag(self):
        """Create a sample red flag for testing."""
        return RedFlag(
            item="Blood Test - CBC",
            item_type=LineItemType.DIAGNOSTIC,
            billed=Decimal('500.00'),
            max_allowed=Decimal('300.00'),
            overcharge_amount=Decimal('200.00'),
            overcharge_pct=66.67,
            confidence=0.8,
            source="cghs",
            reason="Rate exceeds CGHS tariff by 66.67%"
        )

    @pytest.fixture
    def confidence_scorer(self):
        """Create a ConfidenceScorer instance for testing."""
        return ConfidenceScorer(enable_llm_fallback=False)

    def test_initialization(self):
        """Test ConfidenceScorer initialization."""
        scorer = ConfidenceScorer(enable_llm_fallback=False)
        assert scorer.enable_llm_fallback is False
        assert scorer.openai_client is None
        assert isinstance(scorer.weights, ConfidenceWeights)
        assert scorer.llm_threshold == 0.7

    def test_get_source_weight(self, confidence_scorer):
        """Test source weight calculation."""
        assert confidence_scorer._get_source_weight("cghs") == 0.95
        assert confidence_scorer._get_source_weight("esi") == 0.90
        assert confidence_scorer._get_source_weight("unknown") == 0.7

    def test_rule_based_confidence_basic(self, confidence_scorer, sample_red_flag):
        """Test basic rule-based confidence calculation."""
        score = confidence_scorer.calculate_rule_based_confidence(sample_red_flag)
        
        assert isinstance(score, ConfidenceScore)
        assert score.source == ConfidenceSource.RULE_BASED
        assert 0.0 <= score.score <= 1.0
        assert "source_reliability" in score.factors

    @pytest.mark.asyncio
    async def test_hybrid_confidence(self, confidence_scorer, sample_red_flag):
        """Test hybrid confidence calculation."""
        score = await confidence_scorer.calculate_hybrid_confidence(sample_red_flag)
        
        assert isinstance(score, ConfidenceScore)
        assert score.source == ConfidenceSource.HYBRID
        assert 0.0 <= score.score <= 1.0

    def test_determine_verdict_critical(self, confidence_scorer):
        """Test verdict determination for critical cases."""
        high_confidence = ConfidenceScore(0.9, ConfidenceSource.HYBRID, "Test", {})
        verdict = confidence_scorer.determine_verdict(high_confidence, Decimal('6000'), 3)
        assert verdict == Verdict.CRITICAL

    def test_determine_verdict_warning(self, confidence_scorer):
        """Test verdict determination for warning cases."""
        medium_confidence = ConfidenceScore(0.7, ConfidenceSource.HYBRID, "Test", {})
        verdict = confidence_scorer.determine_verdict(medium_confidence, Decimal('2000'), 2)
        assert verdict == Verdict.WARNING

    def test_determine_verdict_ok(self, confidence_scorer):
        """Test verdict determination for OK cases."""
        medium_confidence = ConfidenceScore(0.6, ConfidenceSource.HYBRID, "Test", {})
        verdict = confidence_scorer.determine_verdict(medium_confidence, Decimal('300'), 1)
        assert verdict == Verdict.OK
