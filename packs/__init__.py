"""
Pack loader utility for multi-vertical pack-driven architecture.

This module provides utilities for loading and managing rule packs
for different verticals (medical, loan, rent, etc.).
"""

from __future__ import annotations

import asyncio
import structlog
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import json

logger = structlog.get_logger(__name__)


class PackLoader:
    """
    Utility class for loading and managing rule packs.
    
    Handles loading of pack configurations, rate sources, and entity mappings
    for different verticals in a consistent manner.
    """
    
    def __init__(self, packs_directory: str = "packs"):
        """
        Initialize the pack loader.
        
        Args:
            packs_directory: Directory containing pack definitions
        """
        self.packs_directory = Path(packs_directory)
        self.loaded_packs: Dict[str, Dict[str, Any]] = {}
        self._pack_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized PackLoader with directory: {packs_directory}")
    
    async def load_pack(self, pack_id: str, force_reload: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load a rule pack by ID.
        
        Args:
            pack_id: Identifier for the pack to load
            force_reload: Whether to force reload even if cached
            
        Returns:
            Pack configuration dictionary or None if failed
        """
        if not force_reload and pack_id in self._pack_cache:
            logger.debug(f"Using cached pack: {pack_id}")
            return self._pack_cache[pack_id]
        
        try:
            pack_path = self.packs_directory / pack_id
            if not pack_path.exists():
                logger.error(f"Pack directory not found: {pack_path}")
                return None
            
            # Load main pack configuration
            config_path = pack_path / "rules.yaml"
            if not config_path.exists():
                logger.error(f"Pack config not found: {config_path}")
                return None
            
            with open(config_path, 'r') as f:
                pack_config = yaml.safe_load(f)
            
            # Load rate sources
            rate_sources = {}
            rate_sources_dir = pack_path / "rate_sources"
            if rate_sources_dir.exists():
                for source_file in pack_config.get('rate_sources', []):
                    source_path = rate_sources_dir / source_file
                    if source_path.exists():
                        rate_sources[source_file] = await self._load_rate_source(source_path)
            
            # Load entity mappings
            entity_mappings = {}
            entity_map_path = pack_path / "entity_map.csv"
            if entity_map_path.exists():
                entity_mappings = await self._load_entity_mappings(entity_map_path)
            
            # Combine all pack data
            full_pack = {
                **pack_config,
                'rate_sources_data': rate_sources,
                'entity_mappings': entity_mappings,
                'pack_path': str(pack_path)
            }
            
            # Cache the pack
            self._pack_cache[pack_id] = full_pack
            self.loaded_packs[pack_id] = full_pack
            
            logger.info(f"Successfully loaded pack: {pack_id}")
            return full_pack
            
        except Exception as e:
            logger.error(f"Failed to load pack {pack_id}: {str(e)}")
            return None
    
    async def _load_rate_source(self, source_path: Path) -> Optional[Dict[str, Any]]:
        """Load a rate source file."""
        try:
            with open(source_path, 'r') as f:
                if source_path.suffix == '.json':
                    return json.load(f)
                elif source_path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                else:
                    logger.warning(f"Unsupported rate source format: {source_path}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to load rate source {source_path}: {str(e)}")
            return None
    
    async def _load_entity_mappings(self, mapping_path: Path) -> Dict[str, str]:
        """Load entity mappings from CSV file."""
        try:
            import csv
            mappings = {}
            
            with open(mapping_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'input' in row and 'mapped' in row:
                        mappings[row['input'].lower()] = row['mapped']
            
            logger.info(f"Loaded {len(mappings)} entity mappings from {mapping_path}")
            return mappings
            
        except Exception as e:
            logger.error(f"Failed to load entity mappings from {mapping_path}: {str(e)}")
            return {}
    
    def get_available_packs(self) -> List[str]:
        """
        Get list of available pack IDs.
        
        Returns:
            List of available pack identifiers
        """
        if not self.packs_directory.exists():
            return []
        
        packs = []
        for item in self.packs_directory.iterdir():
            if item.is_dir() and (item / "rules.yaml").exists():
                packs.append(item.name)
        
        return packs
    
    def is_pack_loaded(self, pack_id: str) -> bool:
        """Check if a pack is loaded."""
        return pack_id in self.loaded_packs
    
    def get_pack_info(self, pack_id: str) -> Optional[Dict[str, Any]]:
        """Get basic information about a pack."""
        if pack_id not in self.loaded_packs:
            return None
        
        pack = self.loaded_packs[pack_id]
        return {
            'id': pack.get('id'),
            'name': pack.get('name'),
            'version': pack.get('version'),
            'description': pack.get('description'),
            'rate_sources_count': len(pack.get('rate_sources_data', {})),
            'regex_rules_count': len(pack.get('regex_rules', [])),
            'entity_mappings_count': len(pack.get('entity_mappings', {}))
        }
    
    async def validate_pack(self, pack_id: str) -> Dict[str, Any]:
        """
        Validate a pack configuration.
        
        Args:
            pack_id: Pack to validate
            
        Returns:
            Validation results
        """
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            pack = await self.load_pack(pack_id)
            if not pack:
                validation_result['errors'].append(f"Failed to load pack: {pack_id}")
                return validation_result
            
            # Check required fields
            required_fields = ['id', 'name', 'version']
            for field in required_fields:
                if field not in pack:
                    validation_result['errors'].append(f"Missing required field: {field}")
            
            # Validate regex rules
            regex_rules = pack.get('regex_rules', [])
            for i, rule in enumerate(regex_rules):
                if 'match' not in rule:
                    validation_result['errors'].append(f"Regex rule {i} missing 'match' field")
                if 'cap' not in rule:
                    validation_result['warnings'].append(f"Regex rule {i} missing 'cap' field")
            
            # Check rate sources
            rate_sources = pack.get('rate_sources', [])
            rate_sources_data = pack.get('rate_sources_data', {})
            for source in rate_sources:
                if source not in rate_sources_data:
                    validation_result['warnings'].append(f"Rate source not loaded: {source}")
            
            validation_result['valid'] = len(validation_result['errors']) == 0
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result


# Global pack loader instance
_pack_loader: Optional[PackLoader] = None


def get_pack_loader() -> PackLoader:
    """Get the global pack loader instance."""
    global _pack_loader
    if _pack_loader is None:
        _pack_loader = PackLoader()
    return _pack_loader


async def load_pack(pack_id: str, force_reload: bool = False) -> Optional[Dict[str, Any]]:
    """Convenience function to load a pack."""
    loader = get_pack_loader()
    return await loader.load_pack(pack_id, force_reload)


def get_available_packs() -> List[str]:
    """Convenience function to get available packs."""
    loader = get_pack_loader()
    return loader.get_available_packs()
