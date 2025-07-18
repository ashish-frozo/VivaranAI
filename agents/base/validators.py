"""
Base validation classes for multi-vertical pack-driven architecture.

This module provides the foundational validation classes that can be extended
for different verticals (medical, loan, rent, etc.) using external rule packs.
"""

from __future__ import annotations

import asyncio
import structlog
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml

logger = structlog.get_logger(__name__)


@dataclass
class ValidationDelta:
    """Represents a validation finding or delta."""
    item_description: str
    item_amount: float
    reference_amount: Optional[float] = None
    delta_amount: Optional[float] = None
    delta_percentage: Optional[float] = None
    violation_type: str = "unknown"  # overcharge, undercharge, prohibited, duplicate
    severity: str = "medium"  # low, medium, high, critical
    rule_source: str = "unknown"  # cghs, esi, pack_rule, etc.
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Calculate delta if not provided
        if self.reference_amount is not None and self.delta_amount is None:
            self.delta_amount = self.item_amount - self.reference_amount
            
        # Calculate percentage if not provided
        if self.reference_amount is not None and self.reference_amount > 0 and self.delta_percentage is None:
            self.delta_percentage = (self.delta_amount / self.reference_amount) * 100


class BaseRateValidator(ABC):
    """
    Base class for rate validation across different verticals.
    
    This class provides the foundation for implementing rate validation
    for different domains (medical, loan, rent, etc.) using external
    rule packs instead of hard-coded logic.
    """
    
    def __init__(self, pack_id: str, pack_loader=None):
        """
        Initialize the base rate validator.
        
        Args:
            pack_id: Identifier for the rule pack (e.g., 'medical', 'loan')
            pack_loader: Optional custom pack loader
        """
        self.pack_id = pack_id
        self.pack_loader = pack_loader
        self.pack_config: Dict[str, Any] = {}
        self.rate_sources: Dict[str, Any] = {}
        self.regex_rules: List[Dict[str, Any]] = []
        self.entity_mappings: Dict[str, str] = {}
        self._initialized = False
        
        logger.info(f"Initialized BaseRateValidator for pack: {pack_id}")
    
    async def load_pack(self, pack_id: str) -> bool:
        """
        Load rule pack configuration and data.
        
        Args:
            pack_id: Identifier for the rule pack to load
            
        Returns:
            bool: True if pack loaded successfully, False otherwise
        """
        try:
            # Load pack configuration
            pack_config = await self._load_pack_config(pack_id)
            if not pack_config:
                logger.error(f"Failed to load pack config for: {pack_id}")
                return False
            
            self.pack_config = pack_config
            
            # Load rate sources
            await self._load_rate_sources(pack_config.get('rate_sources', []))
            
            # Load regex rules
            self.regex_rules = pack_config.get('regex_rules', [])
            
            # Load entity mappings
            await self._load_entity_mappings(pack_id)
            
            self._initialized = True
            logger.info(f"Successfully loaded pack: {pack_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load pack {pack_id}: {str(e)}")
            return False
    
    async def _load_pack_config(self, pack_id: str) -> Optional[Dict[str, Any]]:
        """Load pack configuration from YAML file."""
        try:
            pack_path = Path(f"packs/{pack_id}/rules.yaml")
            if not pack_path.exists():
                logger.warning(f"Pack config not found: {pack_path}")
                return None
                
            with open(pack_path, 'r') as f:
                config = yaml.safe_load(f)
                
            return config
            
        except Exception as e:
            logger.error(f"Failed to load pack config: {str(e)}")
            return None
    
    async def _load_rate_sources(self, rate_source_files: List[str]) -> None:
        """Load rate source data files."""
        for source_file in rate_source_files:
            try:
                source_path = Path(f"packs/{self.pack_id}/rate_sources/{source_file}")
                if source_path.exists():
                    with open(source_path, 'r') as f:
                        if source_file.endswith('.json'):
                            import json
                            data = json.load(f)
                        elif source_file.endswith('.yaml') or source_file.endswith('.yml'):
                            data = yaml.safe_load(f)
                        else:
                            logger.warning(f"Unsupported rate source format: {source_file}")
                            continue
                            
                    self.rate_sources[source_file] = data
                    logger.info(f"Loaded rate source: {source_file}")
                else:
                    logger.warning(f"Rate source not found: {source_path}")
                    
            except Exception as e:
                logger.error(f"Failed to load rate source {source_file}: {str(e)}")
    
    async def _load_entity_mappings(self, pack_id: str) -> None:
        """Load entity mappings from CSV file."""
        try:
            mapping_path = Path(f"packs/{pack_id}/entity_map.csv")
            if mapping_path.exists():
                import csv
                with open(mapping_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Assuming CSV has 'input' and 'mapped' columns
                        if 'input' in row and 'mapped' in row:
                            self.entity_mappings[row['input'].lower()] = row['mapped']
                            
                logger.info(f"Loaded {len(self.entity_mappings)} entity mappings")
            else:
                logger.info(f"No entity mappings found for pack: {pack_id}")
                
        except Exception as e:
            logger.error(f"Failed to load entity mappings: {str(e)}")
    
    @abstractmethod
    async def validate(self, items: List[Dict[str, Any]], **kwargs) -> List[ValidationDelta]:
        """
        Validate items against pack rules and rate sources.
        
        Args:
            items: List of items to validate
            **kwargs: Additional validation parameters
            
        Returns:
            List of validation deltas/findings
        """
        pass
    
    async def detect_duplicates(self, items: List[Dict[str, Any]]) -> List[ValidationDelta]:
        """
        Generic duplicate detection logic.
        
        Args:
            items: List of items to check for duplicates
            
        Returns:
            List of duplicate validation deltas
        """
        duplicates = []
        seen_items = {}
        
        for i, item in enumerate(items):
            # Create a key for duplicate detection
            key = self._create_duplicate_key(item)
            
            if key in seen_items:
                # Found duplicate
                original_idx = seen_items[key]
                duplicate_delta = ValidationDelta(
                    item_description=item.get('description', 'Unknown item'),
                    item_amount=float(item.get('amount', item.get('total_amount', 0))),
                    violation_type="duplicate",
                    severity="high",
                    rule_source=f"pack_{self.pack_id}",
                    confidence=0.9,
                    metadata={
                        "original_index": original_idx,
                        "duplicate_index": i,
                        "duplicate_key": key
                    }
                )
                duplicates.append(duplicate_delta)
            else:
                seen_items[key] = i
                
        return duplicates
    
    def _create_duplicate_key(self, item: Dict[str, Any]) -> str:
        """Create a key for duplicate detection."""
        # Default implementation - can be overridden by subclasses
        description = item.get('description', '').lower().strip()
        amount = item.get('amount', item.get('total_amount', 0))
        return f"{description}_{amount}"
    
    async def apply_regex_rules(self, items: List[Dict[str, Any]]) -> List[ValidationDelta]:
        """
        Apply regex-based rules from pack configuration.
        
        Args:
            items: List of items to validate
            
        Returns:
            List of validation deltas from regex rules
        """
        violations = []
        
        for rule in self.regex_rules:
            try:
                import re
                pattern = rule.get('match', '')
                cap_amount = rule.get('cap', 0)
                rule_type = rule.get('type', 'overcharge')
                
                if not pattern:
                    continue
                    
                regex = re.compile(pattern, re.IGNORECASE)
                
                for item in items:
                    description = item.get('description', '')
                    amount = float(item.get('amount', item.get('total_amount', 0)))
                    
                    if regex.search(description) and amount > cap_amount:
                        violation = ValidationDelta(
                            item_description=description,
                            item_amount=amount,
                            reference_amount=cap_amount,
                            violation_type=rule_type,
                            severity=rule.get('severity', 'medium'),
                            rule_source=f"pack_{self.pack_id}_regex",
                            confidence=0.8,
                            metadata={
                                "rule_pattern": pattern,
                                "rule_cap": cap_amount
                            }
                        )
                        violations.append(violation)
                        
            except Exception as e:
                logger.error(f"Failed to apply regex rule {rule}: {str(e)}")
                
        return violations
    
    def map_entity(self, input_text: str) -> str:
        """
        Map input text to standardized entity using entity mappings.
        
        Args:
            input_text: Input text to map
            
        Returns:
            Mapped entity or original text if no mapping found
        """
        normalized_input = input_text.lower().strip()
        return self.entity_mappings.get(normalized_input, input_text)
    
    async def calculate_overcharge_percentage(self, 
                                           total_billed: float, 
                                           total_allowed: float) -> float:
        """
        Calculate overcharge percentage.
        
        Args:
            total_billed: Total billed amount
            total_allowed: Total allowed amount
            
        Returns:
            Overcharge percentage
        """
        if total_allowed <= 0:
            return 0.0
            
        return ((total_billed - total_allowed) / total_allowed) * 100
    
    def is_initialized(self) -> bool:
        """Check if the validator is properly initialized."""
        return self._initialized
