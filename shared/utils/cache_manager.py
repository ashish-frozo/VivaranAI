"""
Cache Manager for MedBillGuardAgent

This module provides a comprehensive caching layer using aiocache with Redis backend
to cache expensive operations like reference data loading, rate validation results,
fuzzy matching computations, and LLM responses.

Features:
- Redis-backed caching with 24-hour TTL
- Configurable cache keys and TTL values
- Cache warming and invalidation
- Cache statistics and monitoring
- Graceful fallback when cache is unavailable
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps

import structlog
from aiocache import Cache, cached
from aiocache.serializers import JsonSerializer, PickleSerializer
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class CacheConfig(BaseModel):
    """Cache configuration settings."""
    
    # Redis connection
    redis_url: str = "redis://localhost:6379/0"
    redis_encoding: str = "utf-8"
    
    # TTL settings (in seconds)
    reference_data_ttl: int = 24 * 60 * 60  # 24 hours
    validation_results_ttl: int = 24 * 60 * 60  # 24 hours
    fuzzy_match_ttl: int = 24 * 60 * 60  # 24 hours
    llm_response_ttl: int = 24 * 60 * 60  # 24 hours
    
    # Cache key prefixes
    reference_data_prefix: str = "medbill:ref_data"
    validation_prefix: str = "medbill:validation"
    fuzzy_match_prefix: str = "medbill:fuzzy"
    llm_response_prefix: str = "medbill:llm"
    
    # Cache behavior
    cache_enabled: bool = True
    cache_warmup_enabled: bool = True
    cache_stats_enabled: bool = True


class CacheStats(BaseModel):
    """Cache statistics."""
    
    hits: int = 0
    misses: int = 0
    errors: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    last_updated: datetime = datetime.now()
    
    def update_hit(self) -> None:
        """Update hit statistics."""
        self.hits += 1
        self.total_requests += 1
        self.hit_rate = self.hits / self.total_requests if self.total_requests > 0 else 0.0
        self.last_updated = datetime.now()
    
    def update_miss(self) -> None:
        """Update miss statistics."""
        self.misses += 1
        self.total_requests += 1
        self.hit_rate = self.hits / self.total_requests if self.total_requests > 0 else 0.0
        self.last_updated = datetime.now()
    
    def update_error(self) -> None:
        """Update error statistics."""
        self.errors += 1
        self.total_requests += 1
        self.last_updated = datetime.now()


class CacheManager:
    """Manages caching operations for MedBillGuardAgent."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.logger = logger.bind(component="cache_manager")
        
        # Initialize cache instances
        self._reference_cache: Optional[Cache] = None
        self._validation_cache: Optional[Cache] = None
        self._fuzzy_cache: Optional[Cache] = None
        self._llm_cache: Optional[Cache] = None
        
        # Statistics
        self.stats = {
            "reference_data": CacheStats(),
            "validation": CacheStats(),
            "fuzzy_match": CacheStats(),
            "llm_response": CacheStats()
        }
        
        # Cache availability
        self._cache_available = False
        
    async def initialize(self) -> None:
        """Initialize cache connections."""
        if not self.config.cache_enabled:
            self.logger.info("Cache disabled in configuration")
            return
            
        try:
            # Initialize Redis caches with different serializers
            self._reference_cache = Cache(
                Cache.REDIS,
                endpoint=self.config.redis_url,
                serializer=JsonSerializer(),
                namespace=self.config.reference_data_prefix,
                ttl=self.config.reference_data_ttl
            )
            
            self._validation_cache = Cache(
                Cache.REDIS,
                endpoint=self.config.redis_url,
                serializer=PickleSerializer(),
                namespace=self.config.validation_prefix,
                ttl=self.config.validation_results_ttl
            )
            
            self._fuzzy_cache = Cache(
                Cache.REDIS,
                endpoint=self.config.redis_url,
                serializer=JsonSerializer(),
                namespace=self.config.fuzzy_match_prefix,
                ttl=self.config.fuzzy_match_ttl
            )
            
            self._llm_cache = Cache(
                Cache.REDIS,
                endpoint=self.config.redis_url,
                serializer=JsonSerializer(),
                namespace=self.config.llm_response_prefix,
                ttl=self.config.llm_response_ttl
            )
            
            # Test cache connectivity
            await self._test_cache_connectivity()
            
            self.logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {e}")
            self._cache_available = False
    
    async def _test_cache_connectivity(self) -> None:
        """Test cache connectivity."""
        try:
            test_key = "test_connectivity"
            test_value = {"timestamp": datetime.now().isoformat()}
            
            if self._reference_cache:
                await self._reference_cache.set(test_key, test_value, ttl=60)
                result = await self._reference_cache.get(test_key)
                
                if result:
                    self._cache_available = True
                    await self._reference_cache.delete(test_key)
                    self.logger.info("Cache connectivity test passed")
                else:
                    raise Exception("Cache connectivity test failed")
                    
        except Exception as e:
            self.logger.error(f"Cache connectivity test failed: {e}")
            self._cache_available = False
            raise
    
    def _generate_cache_key(self, prefix: str, *args: Any) -> str:
        """Generate a cache key from arguments.
        
        Args:
            prefix: Key prefix
            *args: Arguments to include in key
            
        Returns:
            Generated cache key
        """
        # Create a stable hash from arguments
        key_data = json.dumps(args, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def get_reference_data(self, data_type: str, loader_func: Callable) -> Any:
        """Get reference data with caching.
        
        Args:
            data_type: Type of reference data (cghs, esi, nppa, state, prohibited)
            loader_func: Function to load data if not cached
            
        Returns:
            Reference data
        """
        if not self._cache_available or not self._reference_cache:
            return await loader_func()
            
        cache_key = f"ref_data:{data_type}"
        
        try:
            # Try to get from cache
            cached_data = await self._reference_cache.get(cache_key)
            
            if cached_data is not None:
                self.stats["reference_data"].update_hit()
                self.logger.debug(f"Cache hit for reference data: {data_type}")
                return cached_data
                
            # Cache miss - load data
            self.stats["reference_data"].update_miss()
            self.logger.debug(f"Cache miss for reference data: {data_type}")
            
            data = await loader_func()
            
            # Cache the result
            await self._reference_cache.set(cache_key, data)
            
            return data
            
        except Exception as e:
            self.stats["reference_data"].update_error()
            self.logger.error(f"Cache error for reference data {data_type}: {e}")
            # Fallback to direct loading
            return await loader_func()
    
    async def get_validation_result(
        self, 
        items: List[str], 
        item_costs: Dict[str, float],
        state_code: Optional[str],
        validator_func: Callable
    ) -> Any:
        """Get validation result with caching.
        
        Args:
            items: List of items to validate
            item_costs: Item costs dictionary
            state_code: State code for validation
            validator_func: Function to perform validation if not cached
            
        Returns:
            Validation result
        """
        if not self._cache_available or not self._validation_cache:
            return await validator_func()
            
        cache_key = self._generate_cache_key(
            "validation", 
            sorted(items), 
            sorted(item_costs.items()), 
            state_code
        )
        
        try:
            # Try to get from cache
            cached_result = await self._validation_cache.get(cache_key)
            
            if cached_result is not None:
                self.stats["validation"].update_hit()
                self.logger.debug("Cache hit for validation result")
                return cached_result
                
            # Cache miss - perform validation
            self.stats["validation"].update_miss()
            self.logger.debug("Cache miss for validation result")
            
            result = await validator_func()
            
            # Cache the result
            await self._validation_cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            self.stats["validation"].update_error()
            self.logger.error(f"Cache error for validation: {e}")
            # Fallback to direct validation
            return await validator_func()
    
    async def get_fuzzy_matches(
        self, 
        item_name: str, 
        reference_items: List[str],
        matcher_func: Callable
    ) -> List[Dict[str, Any]]:
        """Get fuzzy matching results with caching.
        
        Args:
            item_name: Item name to match
            reference_items: List of reference items
            matcher_func: Function to perform fuzzy matching if not cached
            
        Returns:
            Fuzzy matching results
        """
        if not self._cache_available or not self._fuzzy_cache:
            return await matcher_func()
            
        cache_key = self._generate_cache_key(
            "fuzzy", 
            item_name, 
            sorted(reference_items)
        )
        
        try:
            # Try to get from cache
            cached_matches = await self._fuzzy_cache.get(cache_key)
            
            if cached_matches is not None:
                self.stats["fuzzy_match"].update_hit()
                self.logger.debug(f"Cache hit for fuzzy matches: {item_name}")
                return cached_matches
                
            # Cache miss - perform fuzzy matching
            self.stats["fuzzy_match"].update_miss()
            self.logger.debug(f"Cache miss for fuzzy matches: {item_name}")
            
            matches = await matcher_func()
            
            # Cache the result
            await self._fuzzy_cache.set(cache_key, matches)
            
            return matches
            
        except Exception as e:
            self.stats["fuzzy_match"].update_error()
            self.logger.error(f"Cache error for fuzzy matching: {e}")
            # Fallback to direct matching
            return await matcher_func()
    
    async def get_llm_response(
        self, 
        prompt: str, 
        context: Dict[str, Any],
        llm_func: Callable
    ) -> Any:
        """Get LLM response with caching.
        
        Args:
            prompt: LLM prompt
            context: Context data
            llm_func: Function to call LLM if not cached
            
        Returns:
            LLM response
        """
        if not self._cache_available or not self._llm_cache:
            return await llm_func()
            
        cache_key = self._generate_cache_key("llm", prompt, context)
        
        try:
            # Try to get from cache
            cached_response = await self._llm_cache.get(cache_key)
            
            if cached_response is not None:
                self.stats["llm_response"].update_hit()
                self.logger.debug("Cache hit for LLM response")
                return cached_response
                
            # Cache miss - call LLM
            self.stats["llm_response"].update_miss()
            self.logger.debug("Cache miss for LLM response")
            
            response = await llm_func()
            
            # Cache the result
            await self._llm_cache.set(cache_key, response)
            
            return response
            
        except Exception as e:
            self.stats["llm_response"].update_error()
            self.logger.error(f"Cache error for LLM response: {e}")
            # Fallback to direct LLM call
            return await llm_func()
    
    async def invalidate_reference_data(self, data_type: Optional[str] = None) -> None:
        """Invalidate reference data cache.
        
        Args:
            data_type: Specific data type to invalidate, or None for all
        """
        if not self._cache_available or not self._reference_cache:
            return
            
        try:
            if data_type:
                cache_key = f"ref_data:{data_type}"
                await self._reference_cache.delete(cache_key)
                self.logger.info(f"Invalidated reference data cache: {data_type}")
            else:
                # Clear all reference data cache
                await self._reference_cache.clear()
                self.logger.info("Invalidated all reference data cache")
                
        except Exception as e:
            self.logger.error(f"Failed to invalidate reference data cache: {e}")
    
    async def invalidate_all_caches(self) -> None:
        """Invalidate all caches."""
        if not self._cache_available:
            return
            
        try:
            caches = [
                ("reference", self._reference_cache),
                ("validation", self._validation_cache),
                ("fuzzy", self._fuzzy_cache),
                ("llm", self._llm_cache)
            ]
            
            for name, cache in caches:
                if cache:
                    await cache.clear()
                    self.logger.info(f"Cleared {name} cache")
                    
        except Exception as e:
            self.logger.error(f"Failed to clear caches: {e}")
    
    async def warm_cache(self, reference_loader) -> None:
        """Warm up the cache with commonly used data.
        
        Args:
            reference_loader: Reference data loader instance
        """
        if not self.config.cache_warmup_enabled or not self._cache_available:
            return
            
        try:
            self.logger.info("Starting cache warmup")
            
            # Warm up reference data
            await asyncio.gather(
                self.get_reference_data("cghs", reference_loader.load_cghs_rates),
                self.get_reference_data("esi", reference_loader.load_esi_rates),
                self.get_reference_data("nppa", reference_loader.load_nppa_drugs),
                self.get_reference_data("state", reference_loader.load_state_rates),
                self.get_reference_data("prohibited", reference_loader.load_prohibited_items),
                return_exceptions=True
            )
            
            self.logger.info("Cache warmup completed")
            
        except Exception as e:
            self.logger.error(f"Cache warmup failed: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        return {
            "cache_available": self._cache_available,
            "cache_enabled": self.config.cache_enabled,
            "stats": {
                name: {
                    "hits": stat.hits,
                    "misses": stat.misses,
                    "errors": stat.errors,
                    "total_requests": stat.total_requests,
                    "hit_rate": stat.hit_rate,
                    "last_updated": stat.last_updated.isoformat()
                }
                for name, stat in self.stats.items()
            },
            "config": {
                "reference_data_ttl": self.config.reference_data_ttl,
                "validation_results_ttl": self.config.validation_results_ttl,
                "fuzzy_match_ttl": self.config.fuzzy_match_ttl,
                "llm_response_ttl": self.config.llm_response_ttl
            }
        }
    
    async def cleanup(self) -> None:
        """Clean up cache connections."""
        try:
            caches = [
                self._reference_cache,
                self._validation_cache,
                self._fuzzy_cache,
                self._llm_cache
            ]
            
            for cache in caches:
                if cache:
                    await cache.close()
                    
            self.logger.info("Cache manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")

    # Basic cache operations for general use
    async def get(self, key: str) -> Any:
        """Get value from cache."""
        if not self._cache_available or not self._reference_cache:
            return None
        try:
            return await self._reference_cache.get(key)
        except Exception as e:
            self.logger.error(f"Cache get failed for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self._cache_available or not self._reference_cache:
            return False
        try:
            await self._reference_cache.set(key, value, ttl=ttl or self.config.reference_data_ttl)
            return True
        except Exception as e:
            self.logger.error(f"Cache set failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._cache_available or not self._reference_cache:
            return False
        try:
            await self._reference_cache.delete(key)
            return True
        except Exception as e:
            self.logger.error(f"Cache delete failed for key {key}: {e}")
            return False

    async def set_if_not_exists(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value only if key doesn't exist (for idempotency)."""
        if not self._cache_available or not self._reference_cache:
            return False
        try:
            # Check if key exists
            existing = await self._reference_cache.get(key)
            if existing is not None:
                return False
            
            # Set if not exists
            await self._reference_cache.set(key, value, ttl=ttl or 300)  # Default 5 min TTL
            return True
        except Exception as e:
            self.logger.error(f"Cache set_if_not_exists failed for key {key}: {e}")
            return False


# Global cache manager instance
cache_manager = CacheManager()


def cached_reference_data(data_type: str):
    """Decorator for caching reference data loading.
    
    Args:
        data_type: Type of reference data
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await cache_manager.get_reference_data(
                data_type, 
                lambda: func(*args, **kwargs)
            )
        return wrapper
    return decorator


def cached_validation():
    """Decorator for caching validation results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, items, item_costs, state_code=None, *args, **kwargs):
            return await cache_manager.get_validation_result(
                items, 
                item_costs, 
                state_code,
                lambda: func(self, items, item_costs, state_code, *args, **kwargs)
            )
        return wrapper
    return decorator


def cached_fuzzy_match():
    """Decorator for caching fuzzy matching results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, item_name, reference_items, *args, **kwargs):
            return await cache_manager.get_fuzzy_matches(
                item_name,
                reference_items,
                lambda: func(self, item_name, reference_items, *args, **kwargs)
            )
        return wrapper
    return decorator


def cached_llm_response():
    """Decorator for caching LLM responses."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, prompt, context, *args, **kwargs):
            return await cache_manager.get_llm_response(
                prompt,
                context,
                lambda: func(self, prompt, context, *args, **kwargs)
            )
        return wrapper
    return decorator 