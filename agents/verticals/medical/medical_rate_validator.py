"""
Medical-specific rate validator using pack-driven architecture.

This module provides medical bill validation capabilities using the
external medical rule pack instead of hard-coded logic.
"""

from __future__ import annotations

import asyncio
import structlog
from typing import Dict, Any, List, Optional
from pathlib import Path

from agents.base.validators import BaseRateValidator, ValidationDelta
from packs import get_pack_loader

logger = structlog.get_logger(__name__)


class MedicalRateValidator(BaseRateValidator):
    """
    Medical-specific rate validator using pack-driven architecture.
    
    Validates medical bills against CGHS, ESI, NPPA rates and medical
    pack rules without hard-coded logic.
    """
    
    def __init__(self, pack_loader=None):
        """Initialize medical rate validator."""
        super().__init__(pack_id="medical", pack_loader=pack_loader)
        self.cghs_rates: Dict[str, float] = {}
        self.esi_rates: Dict[str, float] = {}
        self.nppa_rates: Dict[str, float] = {}
        
    async def initialize(self) -> bool:
        """Initialize the medical validator with pack data."""
        success = await self.load_pack("medical")
        if success:
            await self._load_medical_rates()
        return success
    
    async def _load_medical_rates(self) -> None:
        """Load medical-specific rate data from pack."""
        rate_sources_data = self.pack_config.get('rate_sources_data', {})
        
        # Load CGHS rates
        if 'cghs_rates_2023.json' in rate_sources_data:
            cghs_data = rate_sources_data['cghs_rates_2023.json']
            if cghs_data:
                self.cghs_rates = self._extract_rates(cghs_data)
                logger.info(f"Loaded {len(self.cghs_rates)} CGHS rates")
        
        # Load ESI rates
        if 'esi_rates.json' in rate_sources_data:
            esi_data = rate_sources_data['esi_rates.json']
            if esi_data:
                self.esi_rates = self._extract_rates(esi_data)
                logger.info(f"Loaded {len(self.esi_rates)} ESI rates")
        
        # Load NPPA rates
        if 'nppa_rates.json' in rate_sources_data:
            nppa_data = rate_sources_data['nppa_rates.json']
            if nppa_data:
                self.nppa_rates = self._extract_rates(nppa_data)
                logger.info(f"Loaded {len(self.nppa_rates)} NPPA rates")
    
    def _extract_rates(self, rate_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract rates from rate data structure."""
        rates = {}
        
        # Handle different rate data formats
        if isinstance(rate_data, dict):
            if 'rates' in rate_data:
                # Format: {"rates": {"item": rate}}
                rates.update(rate_data['rates'])
            elif 'items' in rate_data:
                # Format: {"items": [{"name": "item", "rate": rate}]}
                for item in rate_data['items']:
                    if 'name' in item and 'rate' in item:
                        rates[item['name'].lower()] = float(item['rate'])
            else:
                # Direct format: {"item": rate}
                for key, value in rate_data.items():
                    if isinstance(value, (int, float)):
                        rates[key.lower()] = float(value)
        
        return rates
    
    async def validate(self, items: List[Dict[str, Any]], **kwargs) -> List[ValidationDelta]:
        """
        Validate medical bill items against pack rules and rate sources.
        
        Args:
            items: List of medical bill items to validate
            **kwargs: Additional validation parameters (state_code, validation_sources)
            
        Returns:
            List of validation deltas/findings
        """
        if not self.is_initialized():
            logger.error("Medical validator not initialized")
            return []
        
        all_deltas = []
        
        try:
            # 1. Detect duplicates
            duplicate_deltas = await self.detect_duplicates(items)
            all_deltas.extend(duplicate_deltas)
            
            # 2. Apply regex rules from pack
            regex_deltas = await self.apply_regex_rules(items)
            all_deltas.extend(regex_deltas)
            
            # 3. Validate against rate sources
            rate_deltas = await self._validate_against_rates(items, **kwargs)
            all_deltas.extend(rate_deltas)
            
            # 4. Check for prohibited items
            prohibited_deltas = await self._check_prohibited_items(items)
            all_deltas.extend(prohibited_deltas)
            
            logger.info(f"Medical validation completed: {len(all_deltas)} findings")
            return all_deltas
            
        except Exception as e:
            logger.error(f"Medical validation failed: {str(e)}")
            return []
    
    async def _validate_against_rates(self, items: List[Dict[str, Any]], **kwargs) -> List[ValidationDelta]:
        """Validate items against medical rate sources."""
        deltas = []
        validation_sources = kwargs.get('validation_sources', ['cghs', 'esi'])
        
        for item in items:
            description = item.get('description', '').lower().strip()
            amount = float(item.get('amount', item.get('total_amount', 0)))
            
            if amount <= 0:
                continue
            
            # Map entity using pack mappings
            mapped_entity = self.map_entity(description)
            
            # Check against each validation source
            reference_rate = None
            rate_source = None
            
            for source in validation_sources:
                if source == 'cghs' and mapped_entity in self.cghs_rates:
                    reference_rate = self.cghs_rates[mapped_entity]
                    rate_source = 'cghs'
                    break
                elif source == 'esi' and mapped_entity in self.esi_rates:
                    reference_rate = self.esi_rates[mapped_entity]
                    rate_source = 'esi'
                    break
                elif source == 'nppa' and mapped_entity in self.nppa_rates:
                    reference_rate = self.nppa_rates[mapped_entity]
                    rate_source = 'nppa'
                    break
            
            # Create validation delta if reference found
            if reference_rate is not None:
                delta_amount = amount - reference_rate
                overcharge_threshold = self.pack_config.get('validation_settings', {}).get('overcharge_threshold_percentage', 20)
                
                if delta_amount > 0 and (delta_amount / reference_rate) * 100 > overcharge_threshold:
                    delta = ValidationDelta(
                        item_description=description,
                        item_amount=amount,
                        reference_amount=reference_rate,
                        violation_type="overcharge",
                        severity="high" if (delta_amount / reference_rate) * 100 > 50 else "medium",
                        rule_source=rate_source,
                        confidence=0.9,
                        metadata={
                            "mapped_entity": mapped_entity,
                            "validation_source": rate_source,
                            "overcharge_threshold": overcharge_threshold
                        }
                    )
                    deltas.append(delta)
        
        return deltas
    
    async def _check_prohibited_items(self, items: List[Dict[str, Any]]) -> List[ValidationDelta]:
        """Check for prohibited items from pack configuration."""
        deltas = []
        prohibited_items = self.pack_config.get('prohibited_items', [])
        
        for item in items:
            description = item.get('description', '').lower().strip()
            amount = float(item.get('amount', item.get('total_amount', 0)))
            
            for prohibited in prohibited_items:
                if prohibited.lower() in description:
                    delta = ValidationDelta(
                        item_description=description,
                        item_amount=amount,
                        violation_type="prohibited",
                        severity="high",
                        rule_source="pack_medical",
                        confidence=0.95,
                        metadata={
                            "prohibited_pattern": prohibited
                        }
                    )
                    deltas.append(delta)
                    break
        
        return deltas
    
    def _create_duplicate_key(self, item: Dict[str, Any]) -> str:
        """Create a key for duplicate detection specific to medical items."""
        description = item.get('description', '').lower().strip()
        amount = item.get('amount', item.get('total_amount', 0))
        
        # Map entity for better duplicate detection
        mapped_entity = self.map_entity(description)
        
        return f"{mapped_entity}_{amount}"
    
    async def get_rate_for_item(self, item_description: str, validation_sources: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get reference rate for a specific item.
        
        Args:
            item_description: Description of the medical item
            validation_sources: Sources to check for rates
            
        Returns:
            Rate information or None if not found
        """
        if not self.is_initialized():
            return None
        
        if validation_sources is None:
            validation_sources = ['cghs', 'esi', 'nppa']
        
        mapped_entity = self.map_entity(item_description.lower().strip())
        
        for source in validation_sources:
            rate = None
            if source == 'cghs' and mapped_entity in self.cghs_rates:
                rate = self.cghs_rates[mapped_entity]
            elif source == 'esi' and mapped_entity in self.esi_rates:
                rate = self.esi_rates[mapped_entity]
            elif source == 'nppa' and mapped_entity in self.nppa_rates:
                rate = self.nppa_rates[mapped_entity]
            
            if rate is not None:
                return {
                    'item_description': item_description,
                    'mapped_entity': mapped_entity,
                    'reference_rate': rate,
                    'source': source
                }
        
        return None
