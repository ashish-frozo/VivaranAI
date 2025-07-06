"""
Rate Validation and Comparison Engine for MedBillGuardAgent

This module implements the core logic to compare medical bill items against
reference rates from CGHS, ESI, NPPA, and state tariffs to detect overcharges.
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from enum import Enum

import structlog
from difflib import SequenceMatcher

from .schemas import RedFlag, LineItemType
from .reference_data_loader import ReferenceDataLoader
from .cache_manager import cache_manager, cached_validation, cached_fuzzy_match

logger = structlog.get_logger(__name__)


class ValidationSource(Enum):
    """Source of rate validation."""
    CGHS = "cghs"
    ESI = "esi"
    NPPA = "nppa"
    STATE_TARIFF = "state_tariff"
    UNKNOWN = "unknown"


@dataclass
class RateMatch:
    """Represents a match between bill item and reference rate."""
    bill_item: str
    reference_item: str
    billed_amount: float
    reference_rate: float
    overcharge_amount: float
    overcharge_percentage: float
    source: ValidationSource
    confidence: float
    item_type: LineItemType
    match_method: str  # exact, fuzzy, keyword
    state_code: Optional[str] = None


@dataclass
class StateValidationConfig:
    """Configuration for state-specific validation."""
    state_code: str
    state_name: Optional[str] = None
    enable_state_priority: bool = True
    fallback_to_central: bool = True
    state_confidence_boost: float = 0.1  # Boost confidence for state matches
    min_state_confidence: float = 0.5


class RateValidator:
    """Validates medical bill rates against reference data."""
    
    # State code mapping for normalization
    STATE_CODE_MAPPING = {
        "DELHI": "DL",
        "KARNATAKA": "KA", 
        "MAHARASHTRA": "MH",
        "TAMIL NADU": "TN",
        "WEST BENGAL": "WB",
        "UTTAR PRADESH": "UP",
        "RAJASTHAN": "RJ",
        "GUJARAT": "GJ",
        "ANDHRA PRADESH": "AP",
        "TELANGANA": "TS",
        "KERALA": "KL",
        "ODISHA": "OR",
        "PUNJAB": "PB",
        "HARYANA": "HR",
        "BIHAR": "BR",
        "JHARKHAND": "JH",
        "ASSAM": "AS",
        "HIMACHAL PRADESH": "HP",
        "UTTARAKHAND": "UK",
        "GOA": "GA",
        "MANIPUR": "MN",
        "MEGHALAYA": "ML",
        "MIZORAM": "MZ",
        "NAGALAND": "NL",
        "SIKKIM": "SK",
        "TRIPURA": "TR",
        "ARUNACHAL PRADESH": "AR"
    }
    
    def __init__(self, reference_loader: Optional[ReferenceDataLoader] = None):
        """Initialize the rate validator.
        
        Args:
            reference_loader: Reference data loader instance
        """
        self.logger = logger.bind(component="rate_validator")
        self.reference_loader = reference_loader or ReferenceDataLoader()
        
        # Matching thresholds
        self.fuzzy_threshold = 75  # Minimum fuzzy match score
        self.confidence_threshold = 0.6  # Minimum confidence for matches
        
        # Load reference data
        self._load_reference_data()

    def _load_reference_data(self) -> None:
        """Load all reference data sources."""
        try:
            self.cghs_data = self.reference_loader.get_cghs_rates()
            self.esi_data = self.reference_loader.get_esi_rates()
            self.nppa_data = self.reference_loader.get_nppa_data()
            self.state_data = self.reference_loader.get_state_tariffs()
            
            self.logger.info(
                f"Loaded reference data: CGHS={len(self.cghs_data)}, "
                f"ESI={len(self.esi_data)}, NPPA={len(self.nppa_data)}, "
                f"States={len(self.state_data)}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load reference data: {e}")
            self.cghs_data = {}
            self.esi_data = {}
            self.nppa_data = {}
            self.state_data = {}

    def _normalize_item_name(self, item_name: str) -> str:
        """Normalize item name for comparison.
        
        Args:
            item_name: Raw item name
            
        Returns:
            Normalized item name
        """
        if not item_name:
            return ""
            
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^\w\s]', ' ', item_name.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Handle common medical abbreviations
        abbreviations = {
            'cbc': 'complete blood count',
            'ecg': 'electrocardiogram',
            'ekg': 'electrocardiogram',
            'mri': 'magnetic resonance imaging',
            'ct': 'computed tomography',
            'usg': 'ultrasonography',
            'xray': 'x ray',
            'x ray': 'x ray',
            'opd': 'outpatient department',
            'ipd': 'inpatient department',
        }
        
        for abbr, full in abbreviations.items():
            normalized = re.sub(rf'\b{abbr}\b', full, normalized)
            
        return normalized

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        if not text:
            return []
            
        # Extract words and filter out common stop words
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'shall', 'test', 'examination'
        }
        
        return [word for word in words if len(word) > 2 and word not in stop_words]

    def _classify_item_type(self, item_name: str) -> LineItemType:
        """Classify the type of medical item.
        
        Args:
            item_name: Item name
            
        Returns:
            LineItemType classification
        """
        item_lower = item_name.lower()
        
        # Consultation patterns
        consultation_patterns = [
            r'consultation', r'visit', r'opd', r'doctor', r'physician',
            r'specialist', r'follow.*up', r'check.*up'
        ]
        
        # Diagnostic patterns
        diagnostic_patterns = [
            r'test', r'blood', r'urine', r'x.*ray', r'xray', r'scan', r'mri',
            r'ct', r'ultrasound', r'usg', r'ecg', r'ekg', r'echo', r'biopsy',
            r'pathology', r'lab', r'culture', r'screening'
        ]
        
        # Procedure patterns
        procedure_patterns = [
            r'surgery', r'operation', r'procedure', r'biopsy', r'endoscopy',
            r'catheter', r'injection', r'infusion', r'dialysis', r'therapy'
        ]
        
        # Medication patterns
        medication_patterns = [
            r'tablet', r'capsule', r'syrup', r'injection', r'medicine',
            r'drug', r'medication', r'antibiotic', r'painkiller'
        ]
        
        # Room/service patterns
        service_patterns = [
            r'room', r'bed', r'ward', r'icu', r'nursing', r'attendant',
            r'ambulance', r'admission', r'discharge'
        ]
        
        # Check patterns in order of specificity
        for pattern in consultation_patterns:
            if re.search(pattern, item_lower):
                return LineItemType.CONSULTATION
                
        for pattern in diagnostic_patterns:
            if re.search(pattern, item_lower):
                return LineItemType.DIAGNOSTIC
                
        for pattern in procedure_patterns:
            if re.search(pattern, item_lower):
                return LineItemType.PROCEDURE
                
        for pattern in medication_patterns:
            if re.search(pattern, item_lower):
                return LineItemType.MEDICATION
                
        for pattern in service_patterns:
            if re.search(pattern, item_lower):
                return LineItemType.SERVICE
                
        return LineItemType.OTHER

    def _find_exact_matches(
        self, 
        item_name: str, 
        reference_data: Dict,
        source: ValidationSource
    ) -> List[Tuple[str, float, float]]:
        """Find exact matches in reference data.
        
        Args:
            item_name: Item name to match
            reference_data: Reference data dictionary
            source: Data source
            
        Returns:
            List of (reference_item, reference_rate, confidence) tuples
        """
        matches = []
        normalized_item = self._normalize_item_name(item_name)
        
        for ref_item, rate_info in reference_data.items():
            normalized_ref = self._normalize_item_name(ref_item)
            
            if normalized_item == normalized_ref:
                rate = self._extract_rate(rate_info)
                if rate > 0:
                    matches.append((ref_item, rate, 1.0))
                    
        return matches

    def _find_fuzzy_matches(
        self, 
        item_name: str, 
        reference_data: Dict,
        source: ValidationSource
    ) -> List[Tuple[str, float, float]]:
        """Find fuzzy matches in reference data.
        
        Args:
            item_name: Item name to match
            reference_data: Reference data dictionary
            source: Data source
            
        Returns:
            List of (reference_item, reference_rate, confidence) tuples
        """
        matches = []
        normalized_item = self._normalize_item_name(item_name)
        
        # Use similarity matching to find similar items
        for ref_item, rate_info in reference_data.items():
            normalized_ref = self._normalize_item_name(ref_item)
            
            # Calculate similarity using SequenceMatcher
            similarity = SequenceMatcher(None, normalized_item, normalized_ref).ratio()
            score = similarity * 100  # Convert to percentage
            
            if score >= self.fuzzy_threshold:
                rate = self._extract_rate(rate_info)
                if rate > 0:
                    confidence = min(0.95, similarity)
                    matches.append((ref_item, rate, confidence))
                        
        return matches

    def _find_keyword_matches(
        self, 
        item_name: str, 
        reference_data: Dict,
        source: ValidationSource
    ) -> List[Tuple[str, float, float]]:
        """Find keyword-based matches in reference data.
        
        Args:
            item_name: Item name to match
            reference_data: Reference data dictionary
            source: Data source
            
        Returns:
            List of (reference_item, reference_rate, confidence) tuples
        """
        matches = []
        item_keywords = set(self._extract_keywords(item_name))
        
        if not item_keywords:
            return matches
            
        for ref_item, rate_info in reference_data.items():
            ref_keywords = set(self._extract_keywords(ref_item))
            
            if not ref_keywords:
                continue
                
            # Calculate keyword overlap
            overlap = item_keywords & ref_keywords
            if overlap:
                overlap_ratio = len(overlap) / len(ref_keywords)
                
                # Require at least 60% keyword overlap
                if overlap_ratio >= 0.6:
                    rate = self._extract_rate(rate_info)
                    if rate > 0:
                        confidence = min(0.8, max(0.4, overlap_ratio))
                        matches.append((ref_item, rate, confidence))
                        
        return matches

    def _extract_rate(self, rate_info: Union[float, Dict, str]) -> float:
        """Extract rate from various data formats.
        
        Args:
            rate_info: Rate information (float, dict, or string)
            
        Returns:
            Extracted rate as float
        """
        if isinstance(rate_info, (int, float)):
            return float(rate_info)
        elif isinstance(rate_info, dict):
            # Try common rate field names
            for field in ['rate', 'amount', 'cost', 'price', 'mrp', 'tariff']:
                if field in rate_info:
                    return float(rate_info[field])
            # If no rate field found, try to get any numeric value
            for value in rate_info.values():
                if isinstance(value, (int, float)):
                    return float(value)
        elif isinstance(rate_info, str):
            # Try to extract numeric value from string
            numbers = re.findall(r'\d+\.?\d*', rate_info)
            if numbers:
                return float(numbers[0])
                
        return 0.0

    def _calculate_overcharge(
        self, 
        billed_amount: float, 
        reference_rate: float
    ) -> Tuple[float, float]:
        """Calculate overcharge amount and percentage.
        
        Args:
            billed_amount: Amount billed
            reference_rate: Reference rate
            
        Returns:
            Tuple of (overcharge_amount, overcharge_percentage)
        """
        if reference_rate <= 0:
            return 0.0, 0.0
            
        overcharge_amount = max(0.0, billed_amount - reference_rate)
        overcharge_percentage = (overcharge_amount / reference_rate) * 100.0
        
        # Round to 2 decimal places
        overcharge_amount = float(Decimal(str(overcharge_amount)).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        ))
        overcharge_percentage = float(Decimal(str(overcharge_percentage)).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        ))
        
        return overcharge_amount, overcharge_percentage

    def _normalize_state_code(self, state_input: str) -> Optional[str]:
        """Normalize state input to standard state code.
        
        Args:
            state_input: State name or code
            
        Returns:
            Normalized state code or None if invalid
        """
        if not state_input:
            return None
            
        state_input = state_input.strip().upper()
        
        # Direct state code match
        if len(state_input) == 2 and state_input in self.state_data:
            return state_input
            
        # State name to code mapping
        if state_input in self.STATE_CODE_MAPPING:
            return self.STATE_CODE_MAPPING[state_input]
            
        # Check if it's already a valid state code
        for code in self.state_data.keys():
            if state_input == code:
                return code
                
        self.logger.warning(f"Unknown state: {state_input}")
        return None

    def _get_state_validation_config(self, state_code: Optional[str]) -> Optional[StateValidationConfig]:
        """Get state validation configuration.
        
        Args:
            state_code: State code
            
        Returns:
            StateValidationConfig or None
        """
        if not state_code:
            return None
            
        normalized_code = self._normalize_state_code(state_code)
        if not normalized_code:
            return None
            
        # Get state name from reverse mapping
        state_name = None
        for name, code in self.STATE_CODE_MAPPING.items():
            if code == normalized_code:
                state_name = name.title()
                break
                
        return StateValidationConfig(
            state_code=normalized_code,
            state_name=state_name,
            enable_state_priority=True,
            fallback_to_central=True,
            state_confidence_boost=0.1,
            min_state_confidence=0.5
        )

    @cached_validation()
    async def validate_item_rates(
        self, 
        items: List[str], 
        item_costs: Dict[str, float],
        state_code: Optional[str] = None
    ) -> List[RateMatch]:
        """Validate item rates against reference data with enhanced state-specific logic.
        
        Args:
            items: List of item names
            item_costs: Dictionary mapping item names to costs
            state_code: Optional state code for state-specific rates
            
        Returns:
            List of RateMatch objects
        """
        if not items:
            return []
            
        # Get state validation configuration
        state_config = self._get_state_validation_config(state_code)
        
        self.logger.info(
            f"Starting rate validation for {len(items)} items" +
            (f" with state-specific rates for {state_config.state_code}" if state_config else "")
        )
        
        all_matches = []
        
        for item in items:
            billed_amount = item_costs.get(item, 0.0)
            if billed_amount <= 0:
                continue
                
            # Find best match using enhanced state-specific logic
            best_match = self._find_best_match_with_state_priority(
                item, billed_amount, state_config
            )
            
            if best_match and best_match.confidence >= self.confidence_threshold:
                all_matches.append(best_match)
                
        self.logger.info(f"Rate validation complete: {len(all_matches)} matches found")
        return all_matches

    def _find_best_match_with_state_priority(
        self, 
        item_name: str, 
        billed_amount: float,
        state_config: Optional[StateValidationConfig]
    ) -> Optional[RateMatch]:
        """Find best match with state-specific prioritization.
        
        Args:
            item_name: Item name
            billed_amount: Billed amount
            state_config: State validation configuration
            
        Returns:
            Best RateMatch or None
        """
        all_matches = []
        
        # 1. Try state tariffs first (if available and enabled)
        if (state_config and state_config.enable_state_priority and 
            state_config.state_code in self.state_data):
            
            state_tariffs = self.state_data[state_config.state_code]
            state_matches = self._find_matches_in_source(
                item_name, billed_amount, state_tariffs, 
                ValidationSource.STATE_TARIFF, state_config.state_code
            )
            
            # Apply state confidence boost
            for match in state_matches:
                match.confidence = min(1.0, match.confidence + state_config.state_confidence_boost)
                
            all_matches.extend(state_matches)
            
            # If we have a high-confidence state match, prefer it
            high_confidence_state = [m for m in state_matches if m.confidence >= 0.8]
            if high_confidence_state:
                return max(high_confidence_state, key=lambda m: m.confidence)
        
        # 2. Try CGHS rates (if no high-confidence state match or fallback enabled)
        if not all_matches or (state_config and state_config.fallback_to_central):
            cghs_matches = self._find_matches_in_source(
                item_name, billed_amount, self.cghs_data, ValidationSource.CGHS
            )
            all_matches.extend(cghs_matches)
        
        # 3. Try ESI rates
        if not all_matches or max(m.confidence for m in all_matches) < 0.8:
            esi_matches = self._find_matches_in_source(
                item_name, billed_amount, self.esi_data, ValidationSource.ESI
            )
            all_matches.extend(esi_matches)
        
        # 4. Try NPPA rates (for medications)
        item_type = self._classify_item_type(item_name)
        if (item_type == LineItemType.MEDICATION and 
            (not all_matches or max(m.confidence for m in all_matches) < 0.8)):
            nppa_matches = self._find_matches_in_source(
                item_name, billed_amount, self.nppa_data, ValidationSource.NPPA
            )
            all_matches.extend(nppa_matches)
        
        # Select best match with state preference
        if not all_matches:
            return None
            
        # Prioritize state matches, then by confidence
        state_matches = [m for m in all_matches if m.source == ValidationSource.STATE_TARIFF]
        if state_matches:
            return max(state_matches, key=lambda m: m.confidence)
            
        return max(all_matches, key=lambda m: m.confidence)

    def _find_matches_in_source(
        self, 
        item_name: str, 
        billed_amount: float,
        reference_data: Dict, 
        source: ValidationSource,
        state_code: Optional[str] = None
    ) -> List[RateMatch]:
        """Find matches in a specific data source.
        
        Args:
            item_name: Item name
            billed_amount: Billed amount
            reference_data: Reference data dictionary
            source: Data source
            state_code: Optional state code
            
        Returns:
            List of RateMatch objects
        """
        matches = []
        
        # Try exact matches first
        exact_matches = self._find_exact_matches(item_name, reference_data, source)
        for ref_item, rate, confidence in exact_matches:
            overcharge_amt, overcharge_pct = self._calculate_overcharge(billed_amount, rate)
            
            match = RateMatch(
                bill_item=item_name,
                reference_item=ref_item,
                billed_amount=billed_amount,
                reference_rate=rate,
                overcharge_amount=overcharge_amt,
                overcharge_percentage=overcharge_pct,
                source=source,
                confidence=confidence,
                item_type=self._classify_item_type(item_name),
                match_method="exact",
                state_code=state_code
            )
            matches.append(match)
            
        # If no exact matches, try fuzzy matches
        if not matches:
            fuzzy_matches = self._find_fuzzy_matches(item_name, reference_data, source)
            for ref_item, rate, confidence in fuzzy_matches:
                overcharge_amt, overcharge_pct = self._calculate_overcharge(billed_amount, rate)
                
                match = RateMatch(
                    bill_item=item_name,
                    reference_item=ref_item,
                    billed_amount=billed_amount,
                    reference_rate=rate,
                    overcharge_amount=overcharge_amt,
                    overcharge_percentage=overcharge_pct,
                    source=source,
                    confidence=confidence,
                    item_type=self._classify_item_type(item_name),
                    match_method="fuzzy",
                    state_code=state_code
                )
                matches.append(match)
                
        # If still no matches, try keyword matches
        if not matches:
            keyword_matches = self._find_keyword_matches(item_name, reference_data, source)
            for ref_item, rate, confidence in keyword_matches:
                overcharge_amt, overcharge_pct = self._calculate_overcharge(billed_amount, rate)
                
                match = RateMatch(
                    bill_item=item_name,
                    reference_item=ref_item,
                    billed_amount=billed_amount,
                    reference_rate=rate,
                    overcharge_amount=overcharge_amt,
                    overcharge_percentage=overcharge_pct,
                    source=source,
                    confidence=confidence,
                    item_type=self._classify_item_type(item_name),
                    match_method="keyword",
                    state_code=state_code
                )
                matches.append(match)
                
        return matches

    def generate_red_flags(self, rate_matches: List[RateMatch]) -> List[RedFlag]:
        """Generate red flags from rate matches with enhanced state-specific information.
        
        Args:
            rate_matches: List of rate matches
            
        Returns:
            List of RedFlag objects
        """
        red_flags = []
        
        for match in rate_matches:
            # Only flag items with significant overcharges
            if match.overcharge_amount > 0 and match.overcharge_percentage >= 10.0:
                
                # Enhanced reason with state-specific information
                reason = (
                    f"Item '{match.bill_item}' is overcharged by ₹{match.overcharge_amount:.2f} "
                    f"({match.overcharge_percentage:.1f}%) compared to "
                )
                
                if match.source == ValidationSource.STATE_TARIFF and match.state_code:
                    reason += f"{match.state_code} state tariff"
                else:
                    reason += f"{match.source.value.upper()}"
                    
                reason += (
                    f" rate of ₹{match.reference_rate:.2f}. "
                    f"Reference: '{match.reference_item}' "
                    f"(Match: {match.match_method}, {match.confidence:.1%} confidence)"
                )
                
                if match.source == ValidationSource.STATE_TARIFF:
                    reason += f". State-specific validation applied for {match.state_code}."
                
                red_flag = RedFlag(
                    item=match.bill_item,
                    item_type=match.item_type,
                    billed=match.billed_amount,
                    max_allowed=match.reference_rate,
                    overcharge_amount=match.overcharge_amount,
                    overcharge_pct=match.overcharge_percentage,
                    confidence=match.confidence,
                    source=f"rate_validation_{match.source.value}",
                    reason=reason,
                    is_duplicate=False,
                    is_prohibited=False
                )
                red_flags.append(red_flag)
                
        self.logger.info(f"Generated {len(red_flags)} rate validation red flags")
        return red_flags

    def get_validation_statistics(self, rate_matches: List[RateMatch]) -> Dict[str, any]:
        """Get statistics about rate validation results with state-specific metrics.
        
        Args:
            rate_matches: List of rate matches
            
        Returns:
            Dictionary with statistics including state-specific data
        """
        if not rate_matches:
            return {
                "total_matches": 0,
                "by_source": {},
                "by_match_method": {},
                "by_item_type": {},
                "by_state": {},
                "total_overcharge": 0.0,
                "avg_overcharge_pct": 0.0,
                "avg_confidence": 0.0,
                "state_validation_used": False,
                "state_priority_effective": False
            }
            
        # Basic statistics
        stats = {
            "total_matches": len(rate_matches),
            "by_source": {},
            "by_match_method": {},
            "by_item_type": {},
            "by_state": {},
            "total_overcharge": sum(m.overcharge_amount for m in rate_matches),
            "avg_overcharge_pct": sum(m.overcharge_percentage for m in rate_matches) / len(rate_matches),
            "avg_confidence": sum(m.confidence for m in rate_matches) / len(rate_matches)
        }
        
        # Count by source
        for match in rate_matches:
            source_key = match.source.value
            stats["by_source"][source_key] = stats["by_source"].get(source_key, 0) + 1
            
            method_key = match.match_method
            stats["by_match_method"][method_key] = stats["by_match_method"].get(method_key, 0) + 1
            
            type_key = match.item_type.value
            stats["by_item_type"][type_key] = stats["by_item_type"].get(type_key, 0) + 1
            
            # State-specific statistics
            if match.state_code:
                stats["by_state"][match.state_code] = stats["by_state"].get(match.state_code, 0) + 1
        
        # State validation metrics
        state_matches = [m for m in rate_matches if m.source == ValidationSource.STATE_TARIFF]
        stats["state_validation_used"] = len(state_matches) > 0
        stats["state_priority_effective"] = len(state_matches) > 0 and len(state_matches) >= len(rate_matches) * 0.5
        
        if state_matches:
            stats["state_matches_count"] = len(state_matches)
            stats["state_avg_confidence"] = sum(m.confidence for m in state_matches) / len(state_matches)
            stats["state_total_overcharge"] = sum(m.overcharge_amount for m in state_matches)
        
        return stats

    def get_available_states(self) -> List[Dict[str, str]]:
        """Get list of available states for validation.
        
        Returns:
            List of dictionaries with state code and name
        """
        available_states = []
        
        for state_code in self.state_data.keys():
            state_name = None
            for name, code in self.STATE_CODE_MAPPING.items():
                if code == state_code:
                    state_name = name.title()
                    break
                    
            available_states.append({
                "code": state_code,
                "name": state_name or state_code,
                "rates_count": len(self.state_data[state_code])
            })
            
        return sorted(available_states, key=lambda x: x["name"] or x["code"])

    def validate_state_coverage(self, items: List[str], state_code: str) -> Dict[str, any]:
        """Validate coverage of items in state tariffs.
        
        Args:
            items: List of item names
            state_code: State code
            
        Returns:
            Coverage analysis
        """
        normalized_state = self._normalize_state_code(state_code)
        if not normalized_state or normalized_state not in self.state_data:
            return {
                "valid_state": False,
                "error": f"Invalid or unsupported state: {state_code}"
            }
            
        state_tariffs = self.state_data[normalized_state]
        covered_items = []
        uncovered_items = []
        
        for item in items:
            # Try to find matches in state data
            exact_matches = self._find_exact_matches(item, state_tariffs, ValidationSource.STATE_TARIFF)
            fuzzy_matches = self._find_fuzzy_matches(item, state_tariffs, ValidationSource.STATE_TARIFF)
            
            if exact_matches or fuzzy_matches:
                covered_items.append({
                    "item": item,
                    "match_type": "exact" if exact_matches else "fuzzy",
                    "confidence": exact_matches[0][2] if exact_matches else fuzzy_matches[0][2]
                })
            else:
                uncovered_items.append(item)
                
        return {
            "valid_state": True,
            "state_code": normalized_state,
            "total_items": len(items),
            "covered_items": len(covered_items),
            "uncovered_items": len(uncovered_items),
            "coverage_percentage": (len(covered_items) / len(items)) * 100 if items else 0,
            "covered_details": covered_items,
            "uncovered_details": uncovered_items,
            "state_tariff_count": len(state_tariffs)
        } 