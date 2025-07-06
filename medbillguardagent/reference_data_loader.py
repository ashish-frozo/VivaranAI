"""Reference Data Loader for MedBillGuardAgent.

This module handles loading, parsing, and caching of reference rate data from:
- CGHS (Central Government Health Scheme) tariffs
- ESI (Employee State Insurance) rates  
- NPPA (National Pharmaceutical Pricing Authority) MRP data
- State-specific tariffs
- Prohibited items list

Features:
- 24-hour caching with TTL using aiocache
- Async data loading and parsing
- Fuzzy matching for procedure/drug names
- State-specific rate lookups
- Comprehensive error handling and logging
"""

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import structlog
from pydantic import BaseModel, Field

from .cache_manager import cache_manager, cached_reference_data

logger = structlog.get_logger(__name__)


class ReferenceRate(BaseModel):
    """Reference rate for a medical procedure or item."""
    
    code: str = Field(..., description="Procedure/item code")
    name: str = Field(..., description="Procedure/item name")
    category: str = Field(..., description="Category")
    rate: float = Field(..., description="Maximum allowed rate in INR")
    source: str = Field(..., description="Data source (CGHS, ESI, NPPA)")
    state_code: Optional[str] = Field(default=None, description="State code")
    effective_date: datetime = Field(..., description="Effective date")


class DrugRate(BaseModel):
    """Drug rate from NPPA MRP data."""
    
    drug_name: str = Field(..., description="Generic drug name")
    brand_name: Optional[str] = Field(default=None, description="Brand name")
    strength: str = Field(..., description="Drug strength (e.g., 500mg)")
    dosage_form: str = Field(..., description="Tablet, Capsule, Syrup, etc.")
    pack_size: str = Field(..., description="Pack size (e.g., 10 tablets)")
    mrp: float = Field(..., description="Maximum Retail Price in INR")
    manufacturer: Optional[str] = Field(default=None, description="Manufacturer name")
    updated_date: datetime = Field(..., description="Last updated date")
    category: Optional[str] = Field(default=None, description="Drug category")
    schedule: Optional[str] = Field(default=None, description="Drug schedule (H/OTC)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProhibitedItem(BaseModel):
    """Prohibited or non-payable item."""
    
    name: str = Field(..., description="Item name")
    category: str = Field(..., description="Category")
    reason: str = Field(..., description="Reason for prohibition")
    source: str = Field(..., description="Source regulation")


class ReferenceDataLoader:
    """Loads and manages reference rate data with caching."""
    
    def __init__(self, data_dir: Path = Path("data")):
        """Initialize the reference data loader.
        
        Args:
            data_dir: Directory containing reference data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Data storage
        self._cghs_rates: Dict[str, ReferenceRate] = {}
        self._esi_rates: Dict[str, ReferenceRate] = {}
        self._nppa_drugs: Dict[str, DrugRate] = {}
        self._state_rates: Dict[str, Dict[str, ReferenceRate]] = {}
        self._prohibited_items: Dict[str, ProhibitedItem] = {}
        
        # Last loaded timestamps
        self._last_loaded: Dict[str, datetime] = {}
        
    async def initialize(self) -> None:
        """Initialize and load all reference data."""
        logger.info("Initializing reference data loader")
        
        try:
            # Initialize cache manager
            await cache_manager.initialize()
            
            # Load all data sources in parallel (will use cache if available)
            await asyncio.gather(
                self.load_cghs_rates(),
                self.load_esi_rates(),
                self.load_nppa_drugs(),
                self.load_state_rates(),
                self.load_prohibited_items(),
                return_exceptions=True
            )
            
            # Warm up cache with common data
            await cache_manager.warm_cache(self)
            
            logger.info("Reference data loader initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize reference data loader", error=str(e))
            raise

    @cached_reference_data("cghs")
    async def load_cghs_rates(self) -> Dict[str, ReferenceRate]:
        """Load CGHS rates from JSON file."""
        logger.info("Loading CGHS rates")
        
        json_file = self.data_dir / "cghs_rates_2023.json"
        
        try:
            if json_file.exists():
                async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
                    
                rates = {}
                for item in data.get('rates', []):
                    rate = ReferenceRate(**item)
                    rates[rate.code] = rate
                    
                self._cghs_rates = rates
                self._last_loaded['cghs'] = datetime.now()
                
                logger.info(f"Loaded {len(rates)} CGHS rates")
                return rates
                
            else:
                logger.warning("CGHS rates file not found")
                return {}
                
        except Exception as e:
            logger.error("Failed to load CGHS rates", error=str(e))
            return {}

    @cached_reference_data("esi")
    async def load_esi_rates(self) -> Dict[str, ReferenceRate]:
        """Load ESI rates from JSON file."""
        logger.info("Loading ESI rates")
        
        json_file = self.data_dir / "esi_rates.json"
        
        try:
            if json_file.exists():
                async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
                    
                rates = {}
                for item in data.get('rates', []):
                    rate = ReferenceRate(**item)
                    rates[rate.code] = rate
                    
                self._esi_rates = rates
                self._last_loaded['esi'] = datetime.now()
                
                logger.info(f"Loaded {len(rates)} ESI rates")
                return rates
            else:
                logger.warning("ESI rates file not found")
                return {}
                
        except Exception as e:
            logger.error("Failed to load ESI rates", error=str(e))
            return {}

    @cached_reference_data("nppa")
    async def load_nppa_drugs(self) -> Dict[str, DrugRate]:
        """Load NPPA drug MRP data from CSV or JSON format."""
        logger.info("Loading NPPA drug data")
        
        csv_file = self.data_dir / "nppa_mrp.csv"
        json_file = self.data_dir / "nppa_mrp.json"
        
        try:
            drugs = {}
            
            # Load JSON first (legacy support)
            if json_file.exists():
                drugs.update(await self._load_nppa_from_json(json_file))
                
            # Load CSV last (preferred format - overwrites JSON)
            if csv_file.exists():
                drugs.update(await self._load_nppa_from_csv(csv_file))
                
            if not drugs:
                logger.warning("No NPPA drug data files found")
                return {}
                
            self._nppa_drugs = drugs
            self._last_loaded['nppa'] = datetime.now()
            
            logger.info(f"Loaded {len(drugs)} NPPA drug rates")
            return drugs
                
        except Exception as e:
            logger.error("Failed to load NPPA drug data", error=str(e))
            return {}

    async def _load_nppa_from_csv(self, csv_file: Path) -> Dict[str, DrugRate]:
        """Load NPPA drugs from CSV file."""
        logger.info(f"Loading NPPA data from CSV: {csv_file}")
        
        drugs = {}
        
        async with aiofiles.open(csv_file, 'r', encoding='utf-8') as f:
            content = await f.read()
            
        # Parse CSV content
        reader = csv.DictReader(content.splitlines())
        
        for row in reader:
            try:
                # Convert string values to appropriate types
                drug_data = {
                    'drug_name': row['drug_name'],
                    'brand_name': row.get('brand_name') or None,
                    'strength': row['strength'],
                    'dosage_form': row['dosage_form'],
                    'pack_size': row['pack_size'],
                    'mrp': float(row['mrp']),
                    'manufacturer': row.get('manufacturer') or None,
                    'updated_date': row['updated_date'],
                    'category': row.get('category') or None,
                    'schedule': row.get('schedule') or None
                }
                
                drug = DrugRate(**drug_data)
                
                # Use drug_name + strength as key for better matching
                key = f"{drug.drug_name.lower()}_{drug.strength}".replace(" ", "_")
                drugs[key] = drug
                
                # Also index by drug name alone for fallback
                drugs[drug.drug_name.lower()] = drug
                
            except Exception as e:
                logger.warning(f"Failed to parse drug row: {row}", error=str(e))
                continue
                
        logger.info(f"Loaded {len(drugs)} drugs from CSV")
        return drugs

    async def _load_nppa_from_json(self, json_file: Path) -> Dict[str, DrugRate]:
        """Load NPPA drugs from JSON file (legacy support)."""
        logger.info(f"Loading NPPA data from JSON: {json_file}")
        
        drugs = {}
        
        async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
            data = json.loads(await f.read())
            
        for item in data.get('drugs', []):
            try:
                drug = DrugRate(**item)
                
                # Use drug_name + strength as key for better matching
                key = f"{drug.drug_name.lower()}_{drug.strength}".replace(" ", "_")
                drugs[key] = drug
                
                # Also index by drug name alone for fallback
                drugs[drug.drug_name.lower()] = drug
                
            except Exception as e:
                logger.warning(f"Failed to parse drug item: {item}", error=str(e))
                continue
                
        logger.info(f"Loaded {len(drugs)} drugs from JSON")
        return drugs

    @cached_reference_data("state")
    async def load_state_rates(self) -> Dict[str, Dict[str, ReferenceRate]]:
        """Load state-specific tariffs from JSON files."""
        logger.info("Loading state tariff data")
        
        state_tariffs_dir = self.data_dir / "state_tariffs"
        
        try:
            if not state_tariffs_dir.exists():
                logger.warning("State tariffs directory not found")
                return {}
                
            state_rates = {}
            
            # Load all state tariff files
            for state_file in state_tariffs_dir.glob("*.json"):
                state_code = state_file.stem.upper()
                
                async with aiofiles.open(state_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
                    
                rates = {}
                for item in data.get('rates', []):
                    rate = ReferenceRate(**item)
                    rates[rate.code] = rate
                    
                state_rates[state_code] = rates
                logger.info(f"Loaded {len(rates)} rates for state {state_code}")
                
            self._state_rates = state_rates
            self._last_loaded['state_rates'] = datetime.now()
            
            logger.info(f"Loaded tariffs for {len(state_rates)} states")
            return state_rates
            
        except Exception as e:
            logger.error("Failed to load state tariff data", error=str(e))
            return {}

    @cached_reference_data("prohibited")
    async def load_prohibited_items(self) -> Dict[str, ProhibitedItem]:
        """Load prohibited items list.
        
        Returns:
            Dictionary of prohibited items keyed by name
        """
        logger.info("Loading prohibited items")
        
        json_file = self.data_dir / "prohibited.json"
        
        try:
            if json_file.exists():
                async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
                    
                items = {}
                for item_data in data.get('prohibited_items', []):
                    item = ProhibitedItem(**item_data)
                    items[item.name.lower()] = item
                    
                self._prohibited_items = items
                self._last_loaded['prohibited'] = datetime.now()
                
                logger.info(f"Loaded {len(items)} prohibited items")
                return items
            else:
                logger.warning("Prohibited items file not found")
                return {}
                
        except Exception as e:
            logger.error("Failed to load prohibited items", error=str(e))
            return {}

    async def find_procedure_rate(
        self,
        procedure_name: str,
        state_code: Optional[str] = None
    ) -> Optional[ReferenceRate]:
        """Find procedure rate by name."""
        if not procedure_name:
            return None
            
        procedure_name = procedure_name.strip().lower()
        
        # Search state-specific rates first if state_code provided
        if state_code and state_code in self._state_rates:
            for rate in self._state_rates[state_code].values():
                if procedure_name in rate.name.lower():
                    return rate
        
        # Search in CGHS first, then ESI
        for rates_dict in [self._cghs_rates, self._esi_rates]:
            for rate in rates_dict.values():
                if procedure_name in rate.name.lower():
                    return rate
                    
        return None

    async def find_drug_rate(
        self, 
        drug_name: str, 
        strength: Optional[str] = None,
        brand_name: Optional[str] = None
    ) -> Optional[DrugRate]:
        """Find drug rate by name, strength, and/or brand name.
        
        Args:
            drug_name: Generic drug name
            strength: Drug strength (e.g., "500mg")
            brand_name: Brand name (e.g., "Crocin")
            
        Returns:
            Best matching DrugRate or None
        """
        if not drug_name:
            return None
            
        drug_name = drug_name.strip().lower()
        
        # Try exact match with strength first
        if strength:
            strength_key = f"{drug_name}_{strength.lower()}".replace(" ", "_")
            if strength_key in self._nppa_drugs:
                return self._nppa_drugs[strength_key]
        
        # Try exact drug name match
        if drug_name in self._nppa_drugs:
            return self._nppa_drugs[drug_name]
            
        # Try brand name match
        if brand_name:
            brand_name = brand_name.strip().lower()
            for drug in self._nppa_drugs.values():
                if drug.brand_name and brand_name in drug.brand_name.lower():
                    return drug
                    
        # Partial match on drug name
        for name, drug in self._nppa_drugs.items():
            if drug_name in name or name in drug_name:
                return drug
                
        # Fuzzy match on drug properties
        for drug in self._nppa_drugs.values():
            # Check if any part of the search matches drug properties
            search_terms = drug_name.split()
            drug_terms = (drug.drug_name + " " + (drug.brand_name or "")).lower().split()
            
            if any(term in drug_terms for term in search_terms):
                return drug
                
        return None

    async def is_prohibited_item(self, item_name: str) -> Optional[ProhibitedItem]:
        """Check if an item is prohibited or non-payable.
        
        Args:
            item_name: Name of the item to check
            
        Returns:
            ProhibitedItem if found, None otherwise
        """
        if not item_name:
            return None
            
        item_name = item_name.strip().lower()
        
        # Exact match
        if item_name in self._prohibited_items:
            return self._prohibited_items[item_name]
            
        # Partial match
        for name, item in self._prohibited_items.items():
            if item_name in name or name in item_name:
                return item
                
        return None

    def get_cghs_rates(self) -> Dict[str, ReferenceRate]:
        """Get loaded CGHS rates (synchronous)."""
        return self._cghs_rates.copy()
        
    def get_esi_rates(self) -> Dict[str, ReferenceRate]:
        """Get loaded ESI rates (synchronous)."""
        return self._esi_rates.copy()
        
    def get_nppa_data(self) -> Dict[str, DrugRate]:
        """Get loaded NPPA drug data (synchronous)."""
        return self._nppa_drugs.copy()
        
    def get_state_tariffs(self) -> Dict[str, Dict[str, ReferenceRate]]:
        """Get loaded state tariffs (synchronous)."""
        return {state: rates.copy() for state, rates in self._state_rates.items()}
        
    def get_prohibited_items(self) -> Dict[str, ProhibitedItem]:
        """Get loaded prohibited items (synchronous)."""
        return self._prohibited_items.copy()

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and data freshness info.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'data_sources': {
                'cghs_rates': len(self._cghs_rates),
                'esi_rates': len(self._esi_rates),
                'nppa_drugs': len(self._nppa_drugs),
                'state_rates': {state: len(rates) for state, rates in self._state_rates.items()},
                'prohibited_items': len(self._prohibited_items)
            },
            'last_loaded': {
                source: timestamp.isoformat() if timestamp else None
                for source, timestamp in self._last_loaded.items()
            }
        }
        
        return stats

    async def refresh_data(self, force: bool = False) -> None:
        """Refresh all reference data.
        
        Args:
            force: Force refresh even if cache is still valid
        """
        logger.info("Refreshing reference data", force=force)
        
        # Reload all data
        await self.initialize()


# Global instance
reference_loader = ReferenceDataLoader() 