"""
Confidence Scorer Tool - Async wrapper for ConfidenceScorer.

This tool wraps the existing ConfidenceScorer component to make it compatible
with the OpenAI Agent SDK framework for medical bill analysis.
"""

import asyncio
import structlog
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from shared.tools.confidence_scorer import ConfidenceScorer

logger = structlog.get_logger(__name__)


class ConfidenceScorerInput(BaseModel):
    """Input schema for confidence scorer tool."""
    analysis_results: Dict[str, Any] = Field(..., description="Complete analysis results")
    processing_stats: Dict[str, Any] = Field(..., description="Document processing statistics")
    red_flags: List[Dict[str, Any]] = Field(default=[], description="List of red flags found")


class ConfidenceScorerTool:
    """
    Async tool wrapper for ConfidenceScorer.
    
    Provides confidence scoring capabilities including:
    - Overall analysis confidence calculation
    - Individual red flag confidence scoring
    - OCR quality impact assessment
    - Processing error impact evaluation
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the confidence scorer tool."""
        self.scorer = ConfidenceScorer(openai_api_key=openai_api_key)
        logger.info("Initialized ConfidenceScorerTool")
    
    async def __call__(
        self,
        analysis_results: Dict[str, Any],
        processing_stats: Dict[str, Any],
        red_flags: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate confidence scores for medical bill analysis results.
        
        Args:
            analysis_results: Complete analysis results from all components
            processing_stats: Document processing statistics and quality metrics
            red_flags: List of red flags found during analysis
            
        Returns:
            Dict containing confidence scores and analysis recommendations
        """
        try:
            logger.info(
                "Starting confidence scoring",
                red_flags_count=len(red_flags or []),
                has_processing_stats=bool(processing_stats)
            )
            
            red_flags = red_flags or []
            
            # Convert red flags to the format expected by scorer
            processed_red_flags = []
            for flag in red_flags:
                # Create a simple object-like structure
                red_flag = type('RedFlag', (), {
                    'type': flag.get('type', 'unknown'),
                    'severity': flag.get('severity', 'medium'),
                    'item': flag.get('item', ''),
                    'reason': flag.get('reason', ''),
                    'overcharge_amount': float(flag.get('overcharge_amount', 0)),
                    'confidence': float(flag.get('confidence', 0.5)),
                    'metadata': flag.get('metadata', {})
                })()
                processed_red_flags.append(red_flag)
            
            # Prepare context for confidence calculation
            context = {
                "ocr_confidence": processing_stats.get("ocr_confidence", 0.0) / 100.0,  # Convert percentage to decimal
                "processing_errors": processing_stats.get("errors_encountered", []),
                "pages_processed": processing_stats.get("pages_processed", 1),
                "text_extracted_chars": processing_stats.get("text_extracted_chars", 0),
                "tables_found": processing_stats.get("tables_found", 0),
                "line_items_found": processing_stats.get("line_items_found", 0),
                "processing_time_ms": processing_stats.get("processing_time_ms", 0)
            }
            
            # Calculate confidence scores
            if processed_red_flags:
                # Score individual red flags and get overall confidence
                red_flag_scores = await self.scorer.score_red_flags(processed_red_flags, context)
                overall_confidence = self.scorer.calculate_overall_confidence(red_flag_scores, context)
                
                # Convert scored red flags to serializable format
                scored_flags = []
                for red_flag, score in red_flag_scores:
                    scored_flags.append({
                        "red_flag": {
                            "type": red_flag.type,
                            "severity": red_flag.severity,
                            "item": red_flag.item,
                            "reason": red_flag.reason,
                            "overcharge_amount": float(red_flag.overcharge_amount),
                            "metadata": red_flag.metadata
                        },
                        "confidence_score": {
                            "score": float(score.score),
                            "source": score.source.value,
                            "reasoning": score.reasoning,
                            "factors": score.factors
                        }
                    })
            else:
                # No red flags - high confidence in "clean" bill
                overall_confidence = type('ConfidenceScore', (), {
                    'score': 0.95,
                    'source': 'rule_based',
                    'reasoning': 'No issues detected - high confidence in clean bill',
                    'factors': {'clean_bill': True, 'no_red_flags': True}
                })()
                scored_flags = []
            
            # Determine verdict based on confidence and red flags
            verdict = self._determine_verdict(red_flags, overall_confidence.score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                verdict, red_flags, overall_confidence.score, context
            )
            
            result = {
                "success": True,
                "overall_confidence": {
                    "score": float(overall_confidence.score),
                    "source": overall_confidence.source,
                    "reasoning": overall_confidence.reasoning,
                    "factors": overall_confidence.factors
                },
                "scored_red_flags": scored_flags,
                "verdict": verdict,
                "recommendations": recommendations,
                "confidence_summary": {
                    "red_flags_analyzed": len(red_flags),
                    "average_red_flag_confidence": (
                        sum(flag.get("confidence", 0) for flag in red_flags) / len(red_flags)
                        if red_flags else 1.0
                    ),
                    "ocr_quality_impact": context["ocr_confidence"],
                    "processing_quality_impact": len(context["processing_errors"]) == 0
                }
            }
            
            logger.info(
                "Confidence scoring completed",
                overall_confidence=overall_confidence.score,
                verdict=verdict,
                red_flags_scored=len(scored_flags)
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Confidence scoring failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error": error_msg,
                "overall_confidence": {
                    "score": 0.0,
                    "source": "error",
                    "reasoning": "Confidence scoring failed",
                    "factors": {}
                },
                "scored_red_flags": [],
                "verdict": "unknown",
                "recommendations": ["Manual review required due to confidence scoring error"]
            }
    
    def _determine_verdict(self, red_flags: List[Dict], confidence_score: float) -> str:
        """Determine overall verdict based on red flags and confidence."""
        if not red_flags and confidence_score > 0.8:
            return "ok"
        
        critical_flags = [f for f in red_flags if f.get("severity") == "critical"]
        if critical_flags or confidence_score < 0.5:
            return "critical"
        
        return "warning"
    
    def _generate_recommendations(
        self, 
        verdict: str, 
        red_flags: List[Dict], 
        confidence_score: float,
        context: Dict
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if verdict == "ok":
            recommendations.append("No issues detected. The bill appears to be compliant.")
            if confidence_score < 0.9:
                recommendations.append("Consider manual spot check due to moderate confidence.")
        
        elif verdict == "warning":
            recommendations.append("Minor issues detected. Review flagged items with hospital.")
            recommendations.append("Request detailed explanation for flagged charges.")
            
        elif verdict == "critical":
            recommendations.append("Serious issues detected. Immediate action required.")
            recommendations.append("Contact hospital billing department for clarification.")
            
            critical_flags = [f for f in red_flags if f.get("severity") == "critical"]
            if critical_flags:
                recommendations.append("Consider escalating to insurance ombudsman.")
        
        # Add context-specific recommendations
        if context.get("ocr_confidence", 1.0) < 0.7:
            recommendations.append("Poor OCR quality detected. Consider manual verification.")
        
        if context.get("processing_errors"):
            recommendations.append("Processing errors encountered. Manual review recommended.")
        
        return recommendations


# Tool schema for OpenAI Agent SDK
ConfidenceScorerTool._tool_schema = {
    "type": "function",
    "function": {
        "name": "calculate_confidence",
        "description": "Calculate confidence scores for medical bill analysis results and generate recommendations",
        "parameters": {
            "type": "object",
            "properties": {
                "analysis_results": {
                    "type": "object",
                    "description": "Complete analysis results from document processing, rate validation, and detection tools",
                    "properties": {
                        "document_processing": {"type": "object"},
                        "rate_validation": {"type": "object"},
                        "duplicate_detection": {"type": "object"},
                        "prohibited_detection": {"type": "object"}
                    }
                },
                "processing_stats": {
                    "type": "object",
                    "description": "Document processing statistics and quality metrics",
                    "properties": {
                        "ocr_confidence": {
                            "type": "number",
                            "description": "OCR confidence percentage (0-100)"
                        },
                        "errors_encountered": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of processing errors"
                        },
                        "pages_processed": {"type": "number"},
                        "line_items_found": {"type": "number"}
                    }
                },
                "red_flags": {
                    "type": "array",
                    "description": "List of red flags from all analysis components",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "severity": {"type": "string"},
                            "item": {"type": "string"},
                            "reason": {"type": "string"},
                            "overcharge_amount": {"type": "number"},
                            "confidence": {"type": "number"}
                        }
                    }
                }
            },
            "required": ["analysis_results", "processing_stats"]
        }
    }
} 