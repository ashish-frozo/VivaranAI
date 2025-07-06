"""
Test cases for the cache manager module.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from medbillguardagent.cache_manager import (
    CacheManager,
    CacheConfig,
    CacheStats,
    cache_manager,
    cached_reference_data,
    cached_validation,
    cached_fuzzy_match,
    cached_llm_response
)


@pytest.fixture
def cache_config():
    """Create test cache configuration."""
    return CacheConfig(
        redis_url="redis://localhost:6379/1",  # Use test database
        reference_data_ttl=3600,  # 1 hour for testing
        validation_results_ttl=3600,
        fuzzy_match_ttl=3600,
        llm_response_ttl=3600,
        cache_enabled=True,
        cache_warmup_enabled=False  # Disable warmup for tests
    )


@pytest.fixture
def mock_cache_manager(cache_config):
    """Create a mock cache manager for testing."""
    manager = CacheManager(cache_config)
    
    # Mock the cache instances
    manager._reference_cache = AsyncMock()
    manager._validation_cache = AsyncMock()
    manager._fuzzy_cache = AsyncMock()
    manager._llm_cache = AsyncMock()
    manager._cache_available = True
    
    return manager


class TestCacheConfig:
    """Test cache configuration."""
    
    def test_default_config(self):
        """Test default cache configuration."""
        config = CacheConfig()
        
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.reference_data_ttl == 24 * 60 * 60  # 24 hours
        assert config.cache_enabled is True
        assert config.reference_data_prefix == "medbill:ref_data"
    
    def test_custom_config(self):
        """Test custom cache configuration."""
        config = CacheConfig(
            redis_url="redis://custom:6379/2",
            reference_data_ttl=7200,
            cache_enabled=False
        )
        
        assert config.redis_url == "redis://custom:6379/2"
        assert config.reference_data_ttl == 7200
        assert config.cache_enabled is False


class TestCacheStats:
    """Test cache statistics."""
    
    def test_initial_stats(self):
        """Test initial cache statistics."""
        stats = CacheStats()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.errors == 0
        assert stats.total_requests == 0
        assert stats.hit_rate == 0.0
    
    def test_update_hit(self):
        """Test updating hit statistics."""
        stats = CacheStats()
        
        stats.update_hit()
        
        assert stats.hits == 1
        assert stats.total_requests == 1
        assert stats.hit_rate == 1.0
    
    def test_update_miss(self):
        """Test updating miss statistics."""
        stats = CacheStats()
        
        stats.update_miss()
        
        assert stats.misses == 1
        assert stats.total_requests == 1
        assert stats.hit_rate == 0.0
    
    def test_update_error(self):
        """Test updating error statistics."""
        stats = CacheStats()
        
        stats.update_error()
        
        assert stats.errors == 1
        assert stats.total_requests == 1
        assert stats.hit_rate == 0.0
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats()
        
        stats.update_hit()
        stats.update_hit()
        stats.update_miss()
        
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.total_requests == 3
        assert stats.hit_rate == 2/3


class TestCacheManager:
    """Test cache manager functionality."""
    
    def test_initialization(self, cache_config):
        """Test cache manager initialization."""
        manager = CacheManager(cache_config)
        
        assert manager.config == cache_config
        assert manager._cache_available is False
        assert len(manager.stats) == 4
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_cache_manager):
        """Test successful cache initialization."""
        # Mock successful Redis connection
        with patch.object(mock_cache_manager, '_test_cache_connectivity', return_value=None):
            result = await mock_cache_manager.initialize()
            assert result is None  # Method returns None on success
            assert mock_cache_manager._cache_available is True
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, mock_cache_manager):
        """Test cache initialization failure."""
        # Mock failed Redis connection
        with patch.object(mock_cache_manager, '_test_cache_connectivity', side_effect=Exception("Connection failed")):
            result = await mock_cache_manager.initialize()
            assert result is None  # Method doesn't raise, just logs error
            assert mock_cache_manager._cache_available is False
    
    @pytest.mark.asyncio
    async def test_get_reference_data_cache_hit(self, mock_cache_manager):
        """Test reference data cache hit."""
        # Mock cache hit
        cached_data = {"cghs_rates": "test_data"}
        mock_cache_manager._reference_cache.get.return_value = cached_data
        
        loader_func = AsyncMock()
        result = await mock_cache_manager.get_reference_data("cghs", loader_func)
        
        assert result == cached_data
        assert mock_cache_manager.stats["reference_data"].hits == 1
        loader_func.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_reference_data_cache_miss(self, mock_cache_manager):
        """Test reference data cache miss."""
        # Mock cache miss
        mock_cache_manager._reference_cache.get.return_value = None
        mock_cache_manager._reference_cache.set = AsyncMock()
        
        loader_data = {"cghs_rates": "loaded_data"}
        loader_func = AsyncMock(return_value=loader_data)
        
        result = await mock_cache_manager.get_reference_data("cghs", loader_func)
        
        assert result == loader_data
        assert mock_cache_manager.stats["reference_data"].misses == 1
        loader_func.assert_called_once()
        mock_cache_manager._reference_cache.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_validation_result_cache_hit(self, mock_cache_manager):
        """Test validation result cache hit."""
        # Mock cache hit
        cached_result = [{"match": "test_match"}]
        mock_cache_manager._validation_cache.get.return_value = cached_result
        
        validator_func = AsyncMock()
        result = await mock_cache_manager.get_validation_result(
            ["item1"], {"item1": 100.0}, "DL", validator_func
        )
        
        assert result == cached_result
        assert mock_cache_manager.stats["validation"].hits == 1
        validator_func.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_fuzzy_matches_cache_miss(self, mock_cache_manager):
        """Test fuzzy matches cache miss."""
        # Mock cache miss
        mock_cache_manager._fuzzy_cache.get.return_value = None
        mock_cache_manager._fuzzy_cache.set = AsyncMock()
        
        matcher_result = [{"match": "fuzzy_match", "score": 0.8}]
        matcher_func = AsyncMock(return_value=matcher_result)
        
        result = await mock_cache_manager.get_fuzzy_matches(
            "test_item", ["ref1", "ref2"], matcher_func
        )
        
        assert result == matcher_result
        assert mock_cache_manager.stats["fuzzy_match"].misses == 1
        matcher_func.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_llm_response_cache_hit(self, mock_cache_manager):
        """Test LLM response cache hit."""
        # Mock cache hit
        cached_response = {"confidence": 0.85, "reasoning": "cached"}
        mock_cache_manager._llm_cache.get.return_value = cached_response
        
        llm_func = AsyncMock()
        result = await mock_cache_manager.get_llm_response(
            "test prompt", {"context": "test"}, llm_func
        )
        
        assert result == cached_response
        assert mock_cache_manager.stats["llm_response"].hits == 1
        llm_func.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cache_error_fallback(self, mock_cache_manager):
        """Test fallback when cache operations fail."""
        # Mock cache error
        mock_cache_manager._reference_cache.get.side_effect = Exception("Cache error")
        
        loader_data = {"fallback": "data"}
        loader_func = AsyncMock(return_value=loader_data)
        
        result = await mock_cache_manager.get_reference_data("cghs", loader_func)
        
        assert result == loader_data
        assert mock_cache_manager.stats["reference_data"].errors == 1
        loader_func.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invalidate_reference_data(self, mock_cache_manager):
        """Test reference data cache invalidation."""
        mock_cache_manager._reference_cache.delete = AsyncMock()
        
        await mock_cache_manager.invalidate_reference_data("cghs")
        
        mock_cache_manager._reference_cache.delete.assert_called_once_with("ref_data:cghs")
    
    @pytest.mark.asyncio
    async def test_invalidate_all_caches(self, mock_cache_manager):
        """Test invalidating all caches."""
        # Mock clear methods
        for cache in [mock_cache_manager._reference_cache, 
                     mock_cache_manager._validation_cache,
                     mock_cache_manager._fuzzy_cache, 
                     mock_cache_manager._llm_cache]:
            cache.clear = AsyncMock()
        
        await mock_cache_manager.invalidate_all_caches()
        
        # Verify all caches were cleared
        for cache in [mock_cache_manager._reference_cache, 
                     mock_cache_manager._validation_cache,
                     mock_cache_manager._fuzzy_cache, 
                     mock_cache_manager._llm_cache]:
            cache.clear.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_warmup(self, mock_cache_manager):
        """Test cache warmup functionality."""
        mock_cache_manager.config.cache_warmup_enabled = True
        mock_cache_manager.get_reference_data = AsyncMock()
        
        # Mock reference loader
        mock_loader = Mock()
        mock_loader.load_cghs_rates = AsyncMock()
        mock_loader.load_esi_rates = AsyncMock()
        mock_loader.load_nppa_drugs = AsyncMock()
        mock_loader.load_state_rates = AsyncMock()
        mock_loader.load_prohibited_items = AsyncMock()
        
        await mock_cache_manager.warm_cache(mock_loader)
        
        # Verify all data types were warmed up
        assert mock_cache_manager.get_reference_data.call_count == 5
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self, mock_cache_manager):
        """Test getting cache statistics."""
        # Update some stats
        mock_cache_manager.stats["reference_data"].update_hit()
        mock_cache_manager.stats["validation"].update_miss()
        
        stats = await mock_cache_manager.get_cache_stats()
        
        assert stats["cache_available"] is True
        assert stats["cache_enabled"] is True
        assert "stats" in stats
        assert "config" in stats
        assert stats["stats"]["reference_data"]["hits"] == 1
        assert stats["stats"]["validation"]["misses"] == 1
    
    @pytest.mark.asyncio
    async def test_cleanup(self, mock_cache_manager):
        """Test cache cleanup."""
        # Mock close methods
        for cache in [mock_cache_manager._reference_cache, 
                     mock_cache_manager._validation_cache,
                     mock_cache_manager._fuzzy_cache, 
                     mock_cache_manager._llm_cache]:
            cache.close = AsyncMock()
        
        await mock_cache_manager.cleanup()
        
        # Verify all caches were closed
        for cache in [mock_cache_manager._reference_cache, 
                     mock_cache_manager._validation_cache,
                     mock_cache_manager._fuzzy_cache, 
                     mock_cache_manager._llm_cache]:
            cache.close.assert_called_once()


class TestCacheDecorators:
    """Test cache decorators."""
    
    @pytest.mark.asyncio
    async def test_cached_reference_data_decorator(self):
        """Test cached reference data decorator."""
        # Mock the cache manager
        with patch('medbillguardagent.cache_manager.cache_manager') as mock_manager:
            mock_manager.get_reference_data = AsyncMock(return_value={"test": "data"})
            
            @cached_reference_data("test_type")
            async def test_loader():
                return {"original": "data"}
            
            result = await test_loader()
            
            assert result == {"test": "data"}
            mock_manager.get_reference_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cached_validation_decorator(self):
        """Test cached validation decorator."""
        # Mock the cache manager
        with patch('medbillguardagent.cache_manager.cache_manager') as mock_manager:
            mock_manager.get_validation_result = AsyncMock(return_value=[{"match": "cached"}])
            
            class TestValidator:
                @cached_validation()
                async def validate_items(self, items, item_costs, state_code=None):
                    return [{"match": "original"}]
            
            validator = TestValidator()
            result = await validator.validate_items(["item1"], {"item1": 100.0}, "DL")
            
            assert result == [{"match": "cached"}]
            mock_manager.get_validation_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cached_fuzzy_match_decorator(self):
        """Test cached fuzzy match decorator."""
        # Mock the cache manager
        with patch('medbillguardagent.cache_manager.cache_manager') as mock_manager:
            mock_manager.get_fuzzy_matches = AsyncMock(return_value=[{"match": "cached"}])
            
            class TestMatcher:
                @cached_fuzzy_match()
                async def find_matches(self, item_name, reference_items):
                    return [{"match": "original"}]
            
            matcher = TestMatcher()
            result = await matcher.find_matches("test", ["ref1", "ref2"])
            
            assert result == [{"match": "cached"}]
            mock_manager.get_fuzzy_matches.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cached_llm_response_decorator(self):
        """Test cached LLM response decorator."""
        # Mock the cache manager
        with patch('medbillguardagent.cache_manager.cache_manager') as mock_manager:
            mock_manager.get_llm_response = AsyncMock(return_value={"response": "cached"})
            
            class TestLLM:
                @cached_llm_response()
                async def call_llm(self, prompt, context):
                    return {"response": "original"}
            
            llm = TestLLM()
            result = await llm.call_llm("test prompt", {"context": "test"})
            
            assert result == {"response": "cached"}
            mock_manager.get_llm_response.assert_called_once()


class TestCacheIntegration:
    """Test cache integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self, cache_config):
        """Test behavior when cache is disabled."""
        cache_config.cache_enabled = False
        manager = CacheManager(cache_config)
        
        loader_data = {"disabled": "cache"}
        loader_func = AsyncMock(return_value=loader_data)
        
        result = await manager.get_reference_data("cghs", loader_func)
        
        assert result == loader_data
        loader_func.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_unavailable(self, mock_cache_manager):
        """Test behavior when cache is unavailable."""
        mock_cache_manager._cache_available = False
        
        loader_data = {"unavailable": "cache"}
        loader_func = AsyncMock(return_value=loader_data)
        
        result = await mock_cache_manager.get_reference_data("cghs", loader_func)
        
        assert result == loader_data
        loader_func.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, mock_cache_manager):
        """Test cache key generation."""
        key1 = mock_cache_manager._generate_cache_key("prefix", "arg1", "arg2")
        key2 = mock_cache_manager._generate_cache_key("prefix", "arg1", "arg2")
        key3 = mock_cache_manager._generate_cache_key("prefix", "arg2", "arg1")
        
        # Same arguments should generate same key
        assert key1 == key2
        # Different order should generate different key
        assert key1 != key3
        # Key should contain prefix
        assert key1.startswith("prefix:")
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, mock_cache_manager):
        """Test concurrent cache operations."""
        # Mock cache operations
        mock_cache_manager._reference_cache.get.return_value = None
        mock_cache_manager._reference_cache.set = AsyncMock()
        
        # Create multiple concurrent operations
        async def load_data(data_type):
            loader_func = AsyncMock(return_value=f"{data_type}_data")
            return await mock_cache_manager.get_reference_data(data_type, loader_func)
        
        # Run operations concurrently
        results = await asyncio.gather(
            load_data("cghs"),
            load_data("esi"),
            load_data("nppa")
        )
        
        assert results == ["cghs_data", "esi_data", "nppa_data"]
        assert mock_cache_manager.stats["reference_data"].misses == 3 