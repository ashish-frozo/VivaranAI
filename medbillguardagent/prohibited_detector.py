"""
Prohibited Item Detection Module for MedBillGuardAgent

This module implements logic to detect prohibited medical services, procedures,
and fees that are not covered under CGHS, ESI, or other insurance schemes.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import structlog
from difflib import SequenceMatcher
from pydantic import BaseModel, Field

from .schemas import RedFlag, LineItemType

logger = structlog.get_logger(__name__)


class ProhibitedItem(BaseModel):
    """Represents a prohibited medical item."""
    
    name: str = Field(..., description="Name of the prohibited item")
    category: str = Field(..., description="Category (surgery, cosmetic, dental, etc.)")
    reason: str = Field(..., description="Reason why it's prohibited")
    source: str = Field(..., description="Source of the prohibition (CGHS, ESI, etc.)")


class ProhibitedMatch(BaseModel):
    """Represents a match between a bill item and prohibited item."""
    
    bill_item: str = Field(..., description="Item name from the bill")
    prohibited_item: ProhibitedItem = Field(..., description="Matched prohibited item")
    similarity_score: float = Field(..., description="Similarity score (0.0-1.0)")
    confidence: float = Field(..., description="Confidence in the match")
    match_type: str = Field(..., description="Type of match (exact, fuzzy, keyword)")


class ProhibitedDetector:
    """Detects prohibited medical services and procedures in bills."""
    
    def __init__(self, data_file: Optional[str] = None):
        """Initialize the prohibited detector.
        
        Args:
            data_file: Path to prohibited items JSON file
        """
        self.logger = logger.bind(component="prohibited_detector")
        self.data_file = data_file or "data/prohibited.json"
        self.prohibited_items: List[ProhibitedItem] = []
        self.prohibited_keywords: Set[str] = set()
        self.category_keywords: Dict[str, Set[str]] = {}
        
        # Similarity threshold for fuzzy matching
        self.similarity_threshold = 0.7
        
        # Load prohibited items
        self._load_prohibited_items()
        self._build_keyword_sets()

    def _load_prohibited_items(self) -> None:
        """Load prohibited items from JSON file."""
        try:
            data_path = Path(self.data_file)
            if not data_path.exists():
                self.logger.warning(f"Prohibited items file not found: {self.data_file}")
                return
                
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            items_data = data.get('prohibited_items', [])
            self.prohibited_items = [ProhibitedItem(**item) for item in items_data]
            
            self.logger.info(f"Loaded {len(self.prohibited_items)} prohibited items")
            
        except Exception as e:
            self.logger.error(f"Failed to load prohibited items: {e}")
            self.prohibited_items = []

    def _build_keyword_sets(self) -> None:
        """Build keyword sets for efficient matching."""
        self.prohibited_keywords = set()
        self.category_keywords = {}
        
        for item in self.prohibited_items:
            # Add item name keywords
            name_words = self._extract_keywords(item.name)
            self.prohibited_keywords.update(name_words)
            
            # Group by category
            if item.category not in self.category_keywords:
                self.category_keywords[item.category] = set()
            self.category_keywords[item.category].update(name_words)
            
        self.logger.debug(f"Built {len(self.prohibited_keywords)} keywords across {len(self.category_keywords)} categories")

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of keywords
        """
        if not text:
            return set()
            
        # Convert to lowercase and split
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'shall'
        }
        
        return {word for word in words if len(word) > 2 and word not in stop_words}

    def _normalize_item_name(self, item_name: str) -> str:
        """Normalize item name for comparison.
        
        Args:
            item_name: Raw item name from bill
            
        Returns:
            Normalized item name
        """
        if not item_name:
            return ""
            
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^\w\s]', ' ', item_name.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized

    def _calculate_similarity(self, item1: str, item2: str) -> float:
        """Calculate similarity between two item names.
        
        Args:
            item1: First item name
            item2: Second item name
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if not item1 or not item2:
            return 0.0
            
        # Normalize both items
        norm1 = self._normalize_item_name(item1)
        norm2 = self._normalize_item_name(item2)
        
        if norm1 == norm2:
            return 1.0
            
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Boost similarity based on keyword overlap
        words1 = self._extract_keywords(norm1)
        words2 = self._extract_keywords(norm2)
        
        if words1 and words2:
            word_overlap = len(words1 & words2) / len(words1 | words2)
            similarity = max(similarity, word_overlap)
            
        return similarity

    def _find_exact_matches(self, item_name: str) -> List[ProhibitedMatch]:
        """Find exact matches for an item name.
        
        Args:
            item_name: Item name from bill
            
        Returns:
            List of exact matches
        """
        matches = []
        normalized_item = self._normalize_item_name(item_name)
        
        for prohibited in self.prohibited_items:
            normalized_prohibited = self._normalize_item_name(prohibited.name)
            
            if normalized_item == normalized_prohibited:
                match = ProhibitedMatch(
                    bill_item=item_name,
                    prohibited_item=prohibited,
                    similarity_score=1.0,
                    confidence=1.0,
                    match_type="exact"
                )
                matches.append(match)
                
        return matches

    def _find_fuzzy_matches(self, item_name: str) -> List[ProhibitedMatch]:
        """Find fuzzy matches for an item name.
        
        Args:
            item_name: Item name from bill
            
        Returns:
            List of fuzzy matches
        """
        matches = []
        
        for prohibited in self.prohibited_items:
            similarity = self._calculate_similarity(item_name, prohibited.name)
            
            if similarity >= self.similarity_threshold:
                # Calculate confidence based on similarity
                confidence = min(0.95, max(0.6, similarity))
                
                match = ProhibitedMatch(
                    bill_item=item_name,
                    prohibited_item=prohibited,
                    similarity_score=similarity,
                    confidence=confidence,
                    match_type="fuzzy"
                )
                matches.append(match)
                
        return matches

    def _find_keyword_matches(self, item_name: str) -> List[ProhibitedMatch]:
        """Find keyword-based matches for an item name.
        
        Args:
            item_name: Item name from bill
            
        Returns:
            List of keyword matches
        """
        matches = []
        item_keywords = self._extract_keywords(item_name)
        
        if not item_keywords:
            return matches
            
        for prohibited in self.prohibited_items:
            prohibited_keywords = self._extract_keywords(prohibited.name)
            
            if not prohibited_keywords:
                continue
                
            # Check for keyword overlap
            overlap = item_keywords & prohibited_keywords
            if overlap:
                overlap_ratio = len(overlap) / len(prohibited_keywords)
                
                # Require at least 50% keyword overlap
                if overlap_ratio >= 0.5:
                    similarity = overlap_ratio
                    confidence = min(0.8, max(0.4, overlap_ratio))
                    
                    match = ProhibitedMatch(
                        bill_item=item_name,
                        prohibited_item=prohibited,
                        similarity_score=similarity,
                        confidence=confidence,
                        match_type="keyword"
                    )
                    matches.append(match)
                    
        return matches

    def _classify_item_type(self, item_name: str, category: str) -> LineItemType:
        """Classify the type of medical item based on category.
        
        Args:
            item_name: Item name
            category: Prohibited item category
            
        Returns:
            LineItemType classification
        """
        category_lower = category.lower()
        
        if category_lower in ['surgery', 'procedure']:
            return LineItemType.PROCEDURE
        elif category_lower in ['cosmetic', 'aesthetic']:
            return LineItemType.PROCEDURE
        elif category_lower in ['dental']:
            return LineItemType.PROCEDURE
        elif category_lower in ['treatment', 'therapy']:
            return LineItemType.SERVICE
        elif category_lower in ['consultation']:
            return LineItemType.CONSULTATION
        elif category_lower in ['diagnostic', 'test']:
            return LineItemType.DIAGNOSTIC
        else:
            return LineItemType.OTHER

    def detect_prohibited_items(
        self, 
        items: List[str], 
        item_costs: Optional[Dict[str, float]] = None
    ) -> Tuple[List[ProhibitedMatch], List[RedFlag]]:
        """Detect prohibited items in a list of bill items.
        
        Args:
            items: List of item names from the bill
            item_costs: Optional mapping of item names to costs
            
        Returns:
            Tuple of (prohibited_matches, red_flags)
        """
        if not items:
            return [], []
            
        self.logger.info(f"Starting prohibited item detection for {len(items)} items")
        
        all_matches = []
        
        for item in items:
            # Try exact matches first
            exact_matches = self._find_exact_matches(item)
            if exact_matches:
                all_matches.extend(exact_matches)
                continue
                
            # Try fuzzy matches
            fuzzy_matches = self._find_fuzzy_matches(item)
            if fuzzy_matches:
                # Take the best fuzzy match
                best_match = max(fuzzy_matches, key=lambda m: m.similarity_score)
                all_matches.append(best_match)
                continue
                
            # Try keyword matches
            keyword_matches = self._find_keyword_matches(item)
            if keyword_matches:
                # Take the best keyword match
                best_match = max(keyword_matches, key=lambda m: m.confidence)
                all_matches.append(best_match)
                
        # Generate red flags
        red_flags = self._generate_red_flags(all_matches, item_costs)
        
        self.logger.info(
            f"Prohibited detection complete: {len(all_matches)} matches, {len(red_flags)} red flags"
        )
        
        return all_matches, red_flags

    def _generate_red_flags(
        self, 
        matches: List[ProhibitedMatch],
        item_costs: Optional[Dict[str, float]] = None
    ) -> List[RedFlag]:
        """Generate red flags for prohibited items.
        
        Args:
            matches: List of prohibited matches
            item_costs: Optional mapping of item names to costs
            
        Returns:
            List of RedFlag objects
        """
        red_flags = []
        
        for match in matches:
            # Get item cost if available
            item_cost = 0.0
            if item_costs:
                item_cost = item_costs.get(match.bill_item, 0.0)
                
            # Create red flag
            red_flag = RedFlag(
                item=match.bill_item,
                item_type=self._classify_item_type(match.bill_item, match.prohibited_item.category),
                billed=item_cost,
                max_allowed=0.0,  # Prohibited items have 0 allowable cost
                overcharge_amount=item_cost,  # Full amount is overcharge
                overcharge_pct=100.0 if item_cost > 0 else 0.0,  # 100% overcharge
                confidence=match.confidence,
                source="prohibited_detection",
                reason=f"Item '{match.bill_item}' is prohibited: {match.prohibited_item.reason}. "
                       f"Source: {match.prohibited_item.source}. "
                       f"Match type: {match.match_type} ({match.confidence:.1%} confidence).",
                is_duplicate=False,
                is_prohibited=True
            )
            red_flags.append(red_flag)
            
        self.logger.info(f"Generated {len(red_flags)} prohibited red flags")
        return red_flags

    def get_prohibited_categories(self) -> Dict[str, int]:
        """Get count of prohibited items by category.
        
        Returns:
            Dictionary mapping category to count
        """
        categories = {}
        for item in self.prohibited_items:
            category = item.category
            categories[category] = categories.get(category, 0) + 1
        return categories

    def is_item_prohibited(self, item_name: str, threshold: float = 0.7) -> bool:
        """Check if a single item is prohibited.
        
        Args:
            item_name: Item name to check
            threshold: Minimum confidence threshold
            
        Returns:
            True if item is prohibited
        """
        matches, _ = self.detect_prohibited_items([item_name])
        return any(match.confidence >= threshold for match in matches)

    def get_prohibited_statistics(self, matches: List[ProhibitedMatch]) -> Dict[str, any]:
        """Get statistics about detected prohibited items.
        
        Args:
            matches: List of prohibited matches
            
        Returns:
            Dictionary with statistics
        """
        if not matches:
            return {
                "total_matches": 0,
                "by_category": {},
                "by_match_type": {},
                "avg_confidence": 0.0
            }
            
        stats = {
            "total_matches": len(matches),
            "by_category": {},
            "by_match_type": {},
            "avg_confidence": sum(m.confidence for m in matches) / len(matches)
        }
        
        # Count by category
        for match in matches:
            category = match.prohibited_item.category
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
        # Count by match type
        for match in matches:
            match_type = match.match_type
            stats["by_match_type"][match_type] = stats["by_match_type"].get(match_type, 0) + 1
            
        return stats 