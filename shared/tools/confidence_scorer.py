"""
Confidence Scoring Engine for MedBillGuardAgent

This module implements a hybrid confidence scoring system that combines:
1. Rule-based weights from different validation sources
2. LLM fallback for complex cases requiring contextual understanding
3. Overall confidence aggregation for the final analysis

The confidence scorer helps determine the reliability of detected red flags
and the overall analysis verdict.
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

import structlog
import openai
from openai import AsyncOpenAI

from shared.schemas.schemas import RedFlag, Verdict, LineItemType
from medbillguardagent.rate_validator import RateMatch, ValidationSource
from shared.tools.duplicate_detector import DuplicateGroup
from medbillguardagent.prohibited_detector import ProhibitedMatch
from shared.utils.cache_manager import cache_manager, cached_llm_response

logger = structlog.get_logger(__name__)


class ConfidenceSource(str, Enum):
    """Sources of confidence scoring."""
    RULE_BASED = "rule_based"
    LLM_ANALYSIS = "llm_analysis"
    HYBRID = "hybrid"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class ConfidenceScore:
    """Individual confidence score with metadata."""
    score: float  # 0.0 to 1.0
    source: ConfidenceSource
    reasoning: str
    factors: Dict[str, float]  # Contributing factors
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ConfidenceWeights:
    """Configurable weights for different validation sources."""
    # Source reliability weights
    cghs_weight: float = 0.95  # CGHS is highly reliable
    esi_weight: float = 0.90   # ESI is reliable
    nppa_weight: float = 0.85  # NPPA is reliable for drugs
    state_tariff_weight: float = 0.80  # State tariffs vary in quality
    
    # Detection type weights
    exact_match_weight: float = 1.0   # Exact matches are most reliable
    fuzzy_match_weight: float = 0.8   # Fuzzy matches are good
    keyword_match_weight: float = 0.6 # Keyword matches are uncertain
    
    # Special detection weights
    duplicate_weight: float = 0.95    # Duplicate detection is reliable
    prohibited_weight: float = 0.90   # Prohibited items are reliable
    
    # Overcharge severity weights
    high_overcharge_boost: float = 0.1   # >100% overcharge increases confidence
    medium_overcharge_boost: float = 0.05 # 50-100% overcharge slight boost
    low_overcharge_penalty: float = -0.1  # <10% overcharge decreases confidence
    
    # OCR and extraction quality weights
    ocr_confidence_weight: float = 0.3    # OCR quality affects confidence
    item_extraction_weight: float = 0.2   # Line item extraction quality


class ConfidenceScorer:
    """Hybrid confidence scoring engine."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4-turbo",
        weights: Optional[ConfidenceWeights] = None,
        llm_threshold: float = 0.7,
        enable_llm_fallback: bool = True
    ):
        """Initialize the confidence scorer.
        
        Args:
            openai_api_key: OpenAI API key
            model: OpenAI model to use
            weights: Custom confidence weights
            llm_threshold: Threshold below which to use LLM fallback
            enable_llm_fallback: Whether to enable LLM fallback
        """
        self.logger = logger.bind(component="confidence_scorer")
        self.weights = weights or ConfidenceWeights()
        self.llm_threshold = llm_threshold
        self.enable_llm_fallback = enable_llm_fallback
        
        # Initialize OpenAI client if API key provided
        self.openai_client = None
        if openai_api_key and enable_llm_fallback:
            try:
                self.openai_client = AsyncOpenAI(api_key=openai_api_key)
                self.model = model
                self.logger.info(f"Initialized OpenAI client with model {model}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.enable_llm_fallback = False
        else:
            self.enable_llm_fallback = False
            self.logger.info("LLM fallback disabled")

    def calculate_rule_based_confidence(
        self,
        red_flag: RedFlag,
        context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceScore:
        """Calculate confidence using rule-based approach.
        
        Args:
            red_flag: Red flag to score
            context: Additional context (OCR confidence, etc.)
            
        Returns:
            ConfidenceScore with rule-based confidence
        """
        factors = {}
        base_confidence = red_flag.confidence
        
        # Source reliability factor
        source_weight = self._get_source_weight(red_flag.source)
        factors["source_reliability"] = source_weight
        
        # Match type factor (if available in context)
        match_type_factor = 1.0
        if context and "match_method" in context:
            match_method = context["match_method"]
            if match_method == "exact":
                match_type_factor = self.weights.exact_match_weight
            elif match_method == "fuzzy":
                match_type_factor = self.weights.fuzzy_match_weight
            elif match_method == "keyword":
                match_type_factor = self.weights.keyword_match_weight
        factors["match_type"] = match_type_factor
        
        # Overcharge severity factor
        overcharge_factor = self._calculate_overcharge_factor(red_flag)
        factors["overcharge_severity"] = overcharge_factor
        
        # Special detection factors
        special_factor = 1.0
        if red_flag.is_duplicate:
            special_factor = max(special_factor, self.weights.duplicate_weight)
            factors["duplicate_detection"] = self.weights.duplicate_weight
        if red_flag.is_prohibited:
            special_factor = max(special_factor, self.weights.prohibited_weight)
            factors["prohibited_detection"] = self.weights.prohibited_weight
        
        # OCR quality factor
        ocr_factor = 1.0
        if context and "ocr_confidence" in context:
            ocr_confidence = context["ocr_confidence"]
            ocr_factor = 0.5 + (ocr_confidence * 0.5)  # Scale 0.5-1.0
            factors["ocr_quality"] = ocr_factor
        
        # Calculate weighted confidence
        weighted_confidence = (
            base_confidence * 
            source_weight * 
            match_type_factor * 
            special_factor *
            ocr_factor
        ) + overcharge_factor
        
        # Clamp to valid range
        final_confidence = max(0.0, min(1.0, weighted_confidence))
        
        reasoning = self._generate_rule_reasoning(factors, final_confidence)
        
        return ConfidenceScore(
            score=final_confidence,
            source=ConfidenceSource.RULE_BASED,
            reasoning=reasoning,
            factors=factors
        )

    def _get_source_weight(self, source: str) -> float:
        """Get reliability weight for validation source."""
        source_weights = {
            "cghs": self.weights.cghs_weight,
            "esi": self.weights.esi_weight,
            "nppa": self.weights.nppa_weight,
            "state_tariff": self.weights.state_tariff_weight,
            "duplicate": self.weights.duplicate_weight,
            "prohibited": self.weights.prohibited_weight,
            "analysis": 0.7,  # Default for general analysis
        }
        return source_weights.get(source.lower(), 0.7)

    def _calculate_overcharge_factor(self, red_flag: RedFlag) -> float:
        """Calculate confidence adjustment based on overcharge severity."""
        if not red_flag.overcharge_pct:
            return 0.0
            
        overcharge_pct = red_flag.overcharge_pct
        
        if overcharge_pct > 100:
            return self.weights.high_overcharge_boost
        elif overcharge_pct > 50:
            return self.weights.medium_overcharge_boost
        elif overcharge_pct < 10:
            return self.weights.low_overcharge_penalty
        else:
            return 0.0

    def _generate_rule_reasoning(
        self, 
        factors: Dict[str, float], 
        final_confidence: float
    ) -> str:
        """Generate human-readable reasoning for rule-based confidence."""
        reasoning_parts = []
        
        # Source reliability
        if "source_reliability" in factors:
            reliability = factors["source_reliability"]
            if reliability >= 0.9:
                reasoning_parts.append("highly reliable source")
            elif reliability >= 0.8:
                reasoning_parts.append("reliable source")
            else:
                reasoning_parts.append("moderate reliability source")
        
        # Match quality
        if "match_type" in factors:
            match_quality = factors["match_type"]
            if match_quality >= 0.9:
                reasoning_parts.append("exact match")
            elif match_quality >= 0.7:
                reasoning_parts.append("good fuzzy match")
            else:
                reasoning_parts.append("keyword-based match")
        
        # Special detections
        if "duplicate_detection" in factors:
            reasoning_parts.append("duplicate pattern detected")
        if "prohibited_detection" in factors:
            reasoning_parts.append("prohibited item identified")
        
        # Overcharge severity
        if "overcharge_severity" in factors:
            severity = factors["overcharge_severity"]
            if severity > 0.05:
                reasoning_parts.append("significant overcharge")
            elif severity < -0.05:
                reasoning_parts.append("minor overcharge")
        
        # OCR quality
        if "ocr_quality" in factors:
            ocr_quality = factors["ocr_quality"]
            if ocr_quality < 0.7:
                reasoning_parts.append("low OCR confidence")
        
        base_text = f"Rule-based confidence: {final_confidence:.1%}"
        if reasoning_parts:
            return f"{base_text} based on {', '.join(reasoning_parts)}"
        else:
            return base_text

    async def calculate_llm_confidence(
        self,
        red_flag: RedFlag,
        context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceScore:
        """Calculate confidence using LLM analysis.
        
        Args:
            red_flag: Red flag to analyze
            context: Additional context for analysis
            
        Returns:
            ConfidenceScore with LLM-based confidence
        """
        if not self.openai_client:
            # Fallback to rule-based if LLM unavailable
            self.logger.warning("LLM client not available, falling back to rule-based")
            rule_score = self.calculate_rule_based_confidence(red_flag, context)
            rule_score.source = ConfidenceSource.LLM_ANALYSIS
            rule_score.reasoning = f"LLM unavailable, used rule-based: {rule_score.reasoning}"
            return rule_score
        
        try:
            # Prepare context for LLM
            llm_context = self._prepare_llm_context(red_flag, context)
            
            # Call OpenAI API with caching
            cache_context = {
                "red_flag_id": f"{red_flag.item}_{red_flag.billed}_{red_flag.source}",
                "model": self.model,
                "context_keys": list((context or {}).keys())
            }
            response, llm_metadata = await self._call_openai_api(llm_context, cache_context)
            
            # Parse response
            confidence_score = self._parse_llm_response(response)
            
            # Add LLM metadata to the confidence score factors
            confidence_score.factors.update(llm_metadata)
            
            return confidence_score
            
        except Exception as e:
            self.logger.error(f"LLM confidence calculation failed: {e}")
            # Fallback to rule-based
            rule_score = self.calculate_rule_based_confidence(red_flag, context)
            rule_score.reasoning = f"LLM failed ({str(e)}), used rule-based: {rule_score.reasoning}"
            return rule_score

    def _prepare_llm_context(
        self,
        red_flag: RedFlag,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Prepare context string for LLM analysis."""
        context_parts = [
            "Analyze the confidence level for this medical bill red flag:",
            f"Item: {red_flag.item}",
            f"Item Type: {red_flag.item_type.value}",
            f"Billed Amount: ₹{red_flag.billed}",
        ]
        
        if red_flag.max_allowed:
            context_parts.append(f"Reference Rate: ₹{red_flag.max_allowed}")
            context_parts.append(f"Overcharge: ₹{red_flag.overcharge_amount} ({red_flag.overcharge_pct:.1f}%)")
        
        context_parts.extend([
            f"Source: {red_flag.source}",
            f"Current Confidence: {red_flag.confidence:.1%}",
            f"Reason: {red_flag.reason}",
        ])
        
        if red_flag.is_duplicate:
            context_parts.append("⚠️ Duplicate item detected")
        if red_flag.is_prohibited:
            context_parts.append("🚫 Prohibited item detected")
        
        if context:
            if "match_method" in context:
                context_parts.append(f"Match Method: {context['match_method']}")
            if "ocr_confidence" in context:
                context_parts.append(f"OCR Confidence: {context['ocr_confidence']:.1%}")
            if "document_type" in context:
                context_parts.append(f"Document Type: {context['document_type']}")
        
        context_parts.extend([
            "",
            "Please analyze this red flag and provide:",
            "1. A confidence score between 0.0 and 1.0",
            "2. Brief reasoning for the confidence level",
            "3. Key factors that influenced your assessment",
            "",
            "Consider factors like:",
            "- Source reliability (CGHS > ESI > NPPA > State)",
            "- Match quality (exact > fuzzy > keyword)",
            "- Overcharge magnitude and reasonableness",
            "- Item type and typical billing patterns",
            "- Duplicate/prohibited item flags",
            "",
            "Respond in JSON format:",
            '{"confidence": 0.85, "reasoning": "...", "factors": ["factor1", "factor2"]}'
        ])
        
        return "\n".join(context_parts)

    @cached_llm_response()
    async def _call_openai_api(self, prompt: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Call OpenAI API for confidence analysis with caching."""
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert medical billing analyst specializing in "
                            "Indian healthcare systems (CGHS, ESI, NPPA). Analyze medical "
                            "bill red flags and provide confidence assessments based on "
                            "your knowledge of typical billing practices, rate structures, "
                            "and fraud patterns."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            raw_response = response.choices[0].message.content
            llm_metadata = {
                "raw_llm_response": raw_response,
                "llm_model": self.model,
                "llm_temperature": 0.1,
                "llm_max_tokens": 500
            }
            
            return raw_response, llm_metadata
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise

    def _parse_llm_response(self, response: str) -> ConfidenceScore:
        """Parse LLM response into ConfidenceScore."""
        try:
            data = json.loads(response)
            
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range
            
            reasoning = data.get("reasoning", "LLM analysis")
            factors_list = data.get("factors", [])
            
            # Convert factors list to dict with equal weights
            factors = {f"llm_factor_{i}": 1.0 for i, _ in enumerate(factors_list)}
            if factors_list:
                factors["llm_factors"] = factors_list
            
            return ConfidenceScore(
                score=confidence,
                source=ConfidenceSource.LLM_ANALYSIS,
                reasoning=f"LLM analysis: {reasoning}",
                factors=factors
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            # Return default confidence
            return ConfidenceScore(
                score=0.5,
                source=ConfidenceSource.LLM_ANALYSIS,
                reasoning=f"LLM parsing failed: {str(e)}",
                factors={"parsing_error": 1.0}
            )

    async def calculate_hybrid_confidence(
        self,
        red_flag: RedFlag,
        context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceScore:
        """Calculate confidence using hybrid approach.
        
        Args:
            red_flag: Red flag to analyze
            context: Additional context
            
        Returns:
            ConfidenceScore with hybrid confidence
        """
        # Always calculate rule-based confidence
        rule_score = self.calculate_rule_based_confidence(red_flag, context)
        
        # Use LLM fallback if rule confidence is below threshold
        if (rule_score.score < self.llm_threshold and 
            self.enable_llm_fallback and 
            self.openai_client):
            
            self.logger.info(
                f"Rule confidence {rule_score.score:.2f} below threshold "
                f"{self.llm_threshold}, using LLM fallback"
            )
            
            llm_score = await self.calculate_llm_confidence(red_flag, context)
            
            # Combine scores (weighted average)
            rule_weight = 0.4
            llm_weight = 0.6
            
            combined_confidence = (
                rule_score.score * rule_weight + 
                llm_score.score * llm_weight
            )
            
            # Combine factors
            combined_factors = {**rule_score.factors, **llm_score.factors}
            combined_factors["rule_weight"] = rule_weight
            combined_factors["llm_weight"] = llm_weight
            
            reasoning = (
                f"Hybrid analysis: Rule={rule_score.score:.2f}, "
                f"LLM={llm_score.score:.2f}, Combined={combined_confidence:.2f}"
            )
            
            return ConfidenceScore(
                score=combined_confidence,
                source=ConfidenceSource.HYBRID,
                reasoning=reasoning,
                factors=combined_factors
            )
        else:
            # Use rule-based confidence
            rule_score.source = ConfidenceSource.HYBRID
            rule_score.reasoning = f"Rule-based (above threshold): {rule_score.reasoning}"
            return rule_score

    async def score_red_flags(
        self,
        red_flags: List[RedFlag],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[RedFlag, ConfidenceScore]]:
        """Score confidence for multiple red flags.
        
        Args:
            red_flags: List of red flags to score
            context: Shared context for all red flags
            
        Returns:
            List of (RedFlag, ConfidenceScore) tuples
        """
        results = []
        
        # Process red flags in parallel for better performance
        tasks = []
        for red_flag in red_flags:
            task = self.calculate_hybrid_confidence(red_flag, context)
            tasks.append(task)
        
        confidence_scores = await asyncio.gather(*tasks, return_exceptions=True)
        
        for red_flag, score_result in zip(red_flags, confidence_scores):
            if isinstance(score_result, Exception):
                self.logger.error(f"Failed to score red flag {red_flag.item}: {score_result}")
                # Create fallback score
                fallback_score = ConfidenceScore(
                    score=red_flag.confidence,
                    source=ConfidenceSource.RULE_BASED,
                    reasoning=f"Scoring failed: {str(score_result)}",
                    factors={"error": 1.0}
                )
                results.append((red_flag, fallback_score))
            else:
                results.append((red_flag, score_result))
        
        return results

    def calculate_overall_confidence(
        self,
        red_flag_scores: List[Tuple[RedFlag, ConfidenceScore]],
        context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceScore:
        """Calculate overall confidence for the entire analysis.
        
        Args:
            red_flag_scores: List of scored red flags
            context: Analysis context
            
        Returns:
            Overall ConfidenceScore
        """
        if not red_flag_scores:
            return ConfidenceScore(
                score=0.0,
                source=ConfidenceSource.RULE_BASED,
                reasoning="No red flags to analyze",
                factors={"no_red_flags": 1.0}
            )
        
        # Calculate weighted average based on overcharge amounts
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for red_flag, confidence_score in red_flag_scores:
            # Weight by overcharge amount (higher overcharges get more weight)
            weight = float(red_flag.overcharge_amount or 1.0)
            total_weight += weight
            weighted_confidence += confidence_score.score * weight
        
        if total_weight > 0:
            overall_confidence = weighted_confidence / total_weight
        else:
            # Fallback to simple average
            overall_confidence = sum(
                score.score for _, score in red_flag_scores
            ) / len(red_flag_scores)
        
        # Apply context factors
        if context:
            # OCR quality affects overall confidence
            if "ocr_confidence" in context:
                ocr_factor = context["ocr_confidence"]
                overall_confidence *= (0.5 + ocr_factor * 0.5)
            
            # Document processing quality
            if "processing_errors" in context:
                error_count = len(context["processing_errors"])
                if error_count > 0:
                    error_penalty = min(0.2, error_count * 0.05)
                    overall_confidence *= (1.0 - error_penalty)
        
        # Determine source distribution
        source_counts = {}
        for _, score in red_flag_scores:
            source = score.source
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Generate reasoning
        avg_individual = sum(score.score for _, score in red_flag_scores) / len(red_flag_scores)
        reasoning = (
            f"Overall confidence: {overall_confidence:.1%} "
            f"(weighted by overcharge amounts, avg individual: {avg_individual:.1%})"
        )
        
        return ConfidenceScore(
            score=max(0.0, min(1.0, overall_confidence)),
            source=ConfidenceSource.HYBRID,
            reasoning=reasoning,
            factors={
                "red_flag_count": len(red_flag_scores),
                "source_distribution": source_counts,
                "weighted_average": overall_confidence,
                "simple_average": avg_individual
            }
        )

    def determine_verdict(
        self,
        overall_confidence: ConfidenceScore,
        total_overcharge: Decimal,
        red_flag_count: int
    ) -> Verdict:
        """Determine analysis verdict based on confidence and findings.
        
        Args:
            overall_confidence: Overall confidence score
            total_overcharge: Total overcharge amount
            red_flag_count: Number of red flags
            
        Returns:
            Analysis verdict
        """
        confidence = overall_confidence.score
        overcharge_amount = float(total_overcharge)
        
        # Critical verdict conditions
        if (confidence >= 0.8 and overcharge_amount > 5000) or red_flag_count >= 5:
            return Verdict.CRITICAL
        
        # Warning verdict conditions  
        if (confidence >= 0.6 and overcharge_amount > 1000) or red_flag_count >= 2:
            return Verdict.WARNING
        
        # OK verdict (low confidence or minor issues)
        if confidence >= 0.5 and overcharge_amount <= 500 and red_flag_count <= 1:
            return Verdict.OK
        
        # Default to WARNING for uncertain cases
        return Verdict.WARNING

    def get_scoring_statistics(
        self,
        red_flag_scores: List[Tuple[RedFlag, ConfidenceScore]]
    ) -> Dict[str, Any]:
        """Get statistics about confidence scoring.
        
        Args:
            red_flag_scores: List of scored red flags
            
        Returns:
            Dictionary with scoring statistics
        """
        if not red_flag_scores:
            return {
                "total_red_flags": 0,
                "avg_confidence": 0.0,
                "source_distribution": {},
                "confidence_range": {"min": 0.0, "max": 0.0}
            }
        
        confidences = [score.score for _, score in red_flag_scores]
        sources = [score.source for _, score in red_flag_scores]
        
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_red_flags": len(red_flag_scores),
            "avg_confidence": sum(confidences) / len(confidences),
            "confidence_range": {
                "min": min(confidences),
                "max": max(confidences)
            },
            "source_distribution": source_counts,
            "high_confidence_count": sum(1 for c in confidences if c >= 0.8),
            "low_confidence_count": sum(1 for c in confidences if c < 0.6),
        }


# Convenience function for external use
async def score_analysis_confidence(
    red_flags: List[RedFlag],
    context: Optional[Dict[str, Any]] = None,
    openai_api_key: Optional[str] = None,
    **kwargs
) -> Tuple[List[Tuple[RedFlag, ConfidenceScore]], ConfidenceScore]:
    """Convenience function to score analysis confidence.
    
    Args:
        red_flags: List of red flags to score
        context: Analysis context
        openai_api_key: OpenAI API key for LLM fallback
        **kwargs: Additional arguments for ConfidenceScorer
        
    Returns:
        Tuple of (scored_red_flags, overall_confidence)
    """
    scorer = ConfidenceScorer(openai_api_key=openai_api_key, **kwargs)
    
    # Score individual red flags
    red_flag_scores = await scorer.score_red_flags(red_flags, context)
    
    # Calculate overall confidence
    overall_confidence = scorer.calculate_overall_confidence(red_flag_scores, context)
    
    return red_flag_scores, overall_confidence 