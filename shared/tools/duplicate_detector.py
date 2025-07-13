"""
Duplicate Detection Module for MedBillGuardAgent

This module implements logic to detect duplicate medical tests, procedures,
and services in hospital bills and pharmacy invoices.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import structlog
from difflib import SequenceMatcher
from pydantic import BaseModel, Field

from shared.schemas.schemas import RedFlag, LineItemType

logger = structlog.get_logger(__name__)


class DuplicateItem(BaseModel):
    """Represents a potential duplicate item."""
    
    original_item: str = Field(..., description="Original item name")
    duplicate_item: str = Field(..., description="Duplicate item name")
    similarity_score: float = Field(..., description="Similarity score (0.0-1.0)")
    item_type: LineItemType = Field(default=LineItemType.OTHER)
    confidence: float = Field(..., description="Confidence in duplicate detection")
    reason: str = Field(..., description="Reason for flagging as duplicate")


class DuplicateGroup(BaseModel):
    """Group of similar/duplicate items."""
    
    canonical_name: str = Field(..., description="Standardized name for the group")
    items: List[str] = Field(..., description="List of item names in this group")
    item_type: LineItemType = Field(default=LineItemType.OTHER)
    total_occurrences: int = Field(..., description="Total number of occurrences")
    confidence: float = Field(..., description="Confidence in grouping")


class DuplicateDetector:
    """Detects duplicate medical tests and procedures in bills."""
    
    def __init__(self):
        """Initialize the duplicate detector."""
        self.logger = logger.bind(component="duplicate_detector")
        
        # Common medical test patterns for better matching
        self.test_patterns = {
            # Blood tests
            r'\b(cbc|complete blood count|hemogram)\b': 'CBC',
            r'\b(fbs|fasting blood sugar|glucose fasting)\b': 'FBS',
            r'\b(ppbs|post prandial|glucose pp)\b': 'PPBS',
            r'\b(hba1c|glycated hemoglobin)\b': 'HbA1c',
            r'\b(lipid profile|cholesterol)\b': 'Lipid Profile',
            r'\b(liver function|lft)\b': 'LFT',
            r'\b(kidney function|rft|creatinine)\b': 'RFT',
            r'\b(thyroid|tsh|t3|t4)\b': 'Thyroid Function',
            
            # Imaging
            r'\b(x-ray|xray|radiography)\b': 'X-Ray',
            r'\b(ultrasound|usg|sonography)\b': 'Ultrasound',
            r'\b(ct scan|computed tomography)\b': 'CT Scan',
            r'\b(mri|magnetic resonance)\b': 'MRI',
            r'\b(ecg|ekg|electrocardiogram)\b': 'ECG',
            r'\b(echo|echocardiogram)\b': 'Echo',
            
            # Consultations
            r'\b(consultation|consult|visit)\b': 'Consultation',
            r'\b(follow up|followup)\b': 'Follow-up',
            
            # Procedures
            r'\b(biopsy)\b': 'Biopsy',
            r'\b(endoscopy)\b': 'Endoscopy',
            r'\b(colonoscopy)\b': 'Colonoscopy',
        }
        
        # Words to ignore when comparing (common variations)
        self.ignore_words = {
            'test', 'examination', 'study', 'scan', 'report', 'analysis',
            'with', 'without', 'contrast', 'iv', 'oral', 'both', 'sides',
            'left', 'right', 'bilateral', 'unilateral', 'single', 'double',
            'charges', 'fee', 'cost', 'amount', 'total'
        }
        
        # Minimum similarity threshold for duplicate detection
        self.similarity_threshold = 0.75
        self.exact_match_threshold = 0.95

    def normalize_item_name(self, item_name: str) -> str:
        """Normalize item name for comparison.
        
        Args:
            item_name: Raw item name from bill
            
        Returns:
            Normalized item name
        """
        if not item_name:
            return ""
            
        # Convert to lowercase
        normalized = item_name.lower().strip()
        
        # Remove special characters and extra spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Apply test patterns for standardization
        for pattern, standard_name in self.test_patterns.items():
            if re.search(pattern, normalized, re.IGNORECASE):
                return standard_name.lower()
                
        return normalized

    def calculate_similarity(self, item1: str, item2: str) -> float:
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
        norm1 = self.normalize_item_name(item1)
        norm2 = self.normalize_item_name(item2)
        
        if norm1 == norm2:
            return 1.0
            
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Boost similarity if key medical terms match
        words1 = set(norm1.split()) - self.ignore_words
        words2 = set(norm2.split()) - self.ignore_words
        
        if words1 and words2:
            word_overlap = len(words1 & words2) / len(words1 | words2)
            similarity = max(similarity, word_overlap)
            
        return similarity

    def detect_exact_duplicates(self, items: List[str]) -> List[DuplicateGroup]:
        """Detect exact duplicate items.
        
        Args:
            items: List of item names to check
            
        Returns:
            List of duplicate groups
        """
        duplicates = []
        item_counts = {}
        normalized_map = {}
        
        # Count occurrences of normalized items
        for item in items:
            normalized = self.normalize_item_name(item)
            if normalized:
                if normalized not in item_counts:
                    item_counts[normalized] = 0
                    normalized_map[normalized] = []
                item_counts[normalized] += 1
                normalized_map[normalized].append(item)
        
        # Find items that appear more than once
        for normalized, count in item_counts.items():
            if count > 1:
                group = DuplicateGroup(
                    canonical_name=normalized,
                    items=normalized_map[normalized],
                    total_occurrences=count,
                    confidence=1.0,  # High confidence for exact matches
                    item_type=self._classify_item_type(normalized)
                )
                duplicates.append(group)
                
        self.logger.info(f"Found {len(duplicates)} exact duplicate groups")
        return duplicates

    def detect_similar_duplicates(self, items: List[str]) -> List[DuplicateGroup]:
        """Detect similar items that might be duplicates.
        
        Args:
            items: List of item names to check
            
        Returns:
            List of potential duplicate groups
        """
        duplicates = []
        processed = set()
        
        for i, item1 in enumerate(items):
            if i in processed:
                continue
                
            similar_items = [item1]
            similar_indices = [i]
            
            for j, item2 in enumerate(items[i+1:], i+1):
                if j in processed:
                    continue
                    
                similarity = self.calculate_similarity(item1, item2)
                
                if similarity >= self.similarity_threshold:
                    similar_items.append(item2)
                    similar_indices.append(j)
                    
            if len(similar_items) > 1:
                # Mark all as processed
                processed.update(similar_indices)
                
                # Create duplicate group
                canonical_name = self.normalize_item_name(item1)
                confidence = min(0.9, max(0.6, 
                    sum(self.calculate_similarity(item1, item) for item in similar_items[1:]) / len(similar_items[1:])
                ))
                
                group = DuplicateGroup(
                    canonical_name=canonical_name,
                    items=similar_items,
                    total_occurrences=len(similar_items),
                    confidence=confidence,
                    item_type=self._classify_item_type(canonical_name)
                )
                duplicates.append(group)
                
        self.logger.info(f"Found {len(duplicates)} similar duplicate groups")
        return duplicates

    def _classify_item_type(self, item_name: str) -> LineItemType:
        """Classify the type of medical item.
        
        Args:
            item_name: Normalized item name
            
        Returns:
            LineItemType classification
        """
        item_lower = item_name.lower()
        
        # Consultation patterns
        if any(word in item_lower for word in ['consultation', 'consult', 'visit', 'follow']):
            return LineItemType.CONSULTATION
            
        # Diagnostic patterns
        if any(word in item_lower for word in ['test', 'blood', 'urine', 'scan', 'x-ray', 'mri', 'ct', 'ultrasound', 'ecg', 'echo']):
            return LineItemType.DIAGNOSTIC
            
        # Procedure patterns
        if any(word in item_lower for word in ['surgery', 'operation', 'procedure', 'biopsy', 'endoscopy']):
            return LineItemType.PROCEDURE
            
        # Medication patterns
        if any(word in item_lower for word in ['tablet', 'capsule', 'syrup', 'injection', 'medicine', 'drug']):
            return LineItemType.MEDICATION
            
        # Room/accommodation patterns
        if any(word in item_lower for word in ['room', 'bed', 'ward', 'icu', 'accommodation']):
            return LineItemType.ROOM_CHARGE
            
        return LineItemType.OTHER

    def generate_duplicate_red_flags(
        self, 
        duplicate_groups: List[DuplicateGroup],
        item_costs: Optional[Dict[str, float]] = None
    ) -> List[RedFlag]:
        """Generate red flags for detected duplicates.
        
        Args:
            duplicate_groups: List of duplicate groups
            item_costs: Optional mapping of item names to costs
            
        Returns:
            List of RedFlag objects for duplicates
        """
        red_flags = []
        
        for group in duplicate_groups:
            if group.total_occurrences <= 1:
                continue
                
            # Calculate potential overcharge
            total_cost = 0.0
            if item_costs:
                for item in group.items:
                    total_cost += item_costs.get(item, 0.0)
            
            # Estimate cost of first occurrence (should be legitimate)
            legitimate_cost = total_cost / group.total_occurrences if total_cost > 0 else 0.0
            overcharge = total_cost - legitimate_cost
            
            # Create red flag
            red_flag = RedFlag(
                item=f"{group.canonical_name} (appears {group.total_occurrences} times)",
                item_type=group.item_type,
                billed=total_cost,
                max_allowed=legitimate_cost,
                overcharge_amount=overcharge,
                overcharge_pct=(overcharge / legitimate_cost * 100) if legitimate_cost > 0 else 0.0,
                confidence=group.confidence,
                source="duplicate_detection",
                reason=f"Item '{group.canonical_name}' appears {group.total_occurrences} times. "
                       f"Duplicate charges detected with {group.confidence:.1%} confidence.",
                is_duplicate=True,
                is_prohibited=False
            )
            red_flags.append(red_flag)
            
        self.logger.info(f"Generated {len(red_flags)} duplicate red flags")
        return red_flags

    def detect_duplicates(
        self, 
        items: List[str], 
        item_costs: Optional[Dict[str, float]] = None
    ) -> Tuple[List[DuplicateGroup], List[RedFlag]]:
        """Main method to detect duplicates and generate red flags.
        
        Args:
            items: List of item names from the bill
            item_costs: Optional mapping of item names to costs
            
        Returns:
            Tuple of (duplicate_groups, red_flags)
        """
        if not items:
            return [], []
            
        self.logger.info(f"Starting duplicate detection for {len(items)} items")
        
        # Detect exact duplicates first
        exact_duplicates = self.detect_exact_duplicates(items)
        
        # Detect similar duplicates
        similar_duplicates = self.detect_similar_duplicates(items)
        
        # Combine all duplicates
        all_duplicates = exact_duplicates + similar_duplicates
        
        # Generate red flags
        red_flags = self.generate_duplicate_red_flags(all_duplicates, item_costs)
        
        self.logger.info(
            f"Duplicate detection complete: {len(all_duplicates)} groups, {len(red_flags)} red flags"
        )
        
        return all_duplicates, red_flags

    def get_duplicate_statistics(self, duplicate_groups: List[DuplicateGroup]) -> Dict[str, int]:
        """Get statistics about detected duplicates.
        
        Args:
            duplicate_groups: List of duplicate groups
            
        Returns:
            Dictionary with duplicate statistics
        """
        if not duplicate_groups:
            return {
                "total_groups": 0,
                "total_duplicates": 0,
                "by_type": {}
            }
            
        stats = {
            "total_groups": len(duplicate_groups),
            "total_duplicates": sum(group.total_occurrences - 1 for group in duplicate_groups),
            "by_type": {}
        }
        
        # Count by item type
        for group in duplicate_groups:
            item_type = group.item_type.value
            if item_type not in stats["by_type"]:
                stats["by_type"][item_type] = 0
            stats["by_type"][item_type] += group.total_occurrences - 1
            
        return stats 