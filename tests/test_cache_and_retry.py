"""Tests for in-memory response caching, retry backoff, and fallback failover."""

import pytest
from app.cache import ResponseCache
from app.providers.mock_provider import MockProvider
from app.reliability import ReliabilityManager


@pytest.mark.asyncio
async def test_cache_hit_and_eviction():
    cache = ResponseCache(max_entries=2, ttl_seconds=60)
    messages = [{"role": "user", "content": "What is Python?"}]
    key1 = cache.generate_key("model-a", messages, 0.7)
    key2 = cache.generate_key("model-b", messages, 0.7)
    key3 = cache.generate_key("model-c", messages, 0.7)

    # Store entries
    await cache.set(key1, {"data": "response 1"})
    await cache.set(key2, {"data": "response 2"})

    assert (await cache.get(key1)) == {"data": "response 1"}

    # Add 3rd entry -> should evict LRU (key2 since key1 was accessed)
    await cache.set(key3, {"data": "response 3"})
    assert (await cache.get(key2)) is None  # evicted
    assert (await cache.get(key1)) is not None  # preserved
    assert (await cache.get(key3)) is not None


@pytest.mark.asyncio
async def test_retry_on_transient_failure():
    mock_provider = MockProvider()
    mock_provider.set_forced_failures(2)  # Fail first 2 attempts, 3rd succeeds

    manager = ReliabilityManager(max_retries=3, base_backoff=0.01, max_backoff=0.05)
    messages = [{"role": "user", "content": "Ping"}]

    resp, retries, fallback_used = await manager.execute_with_resilience(
        provider=mock_provider,
        primary_model="test-primary",
        messages=messages,
    )

    assert resp is not None
    assert retries == 2
    assert fallback_used is False


@pytest.mark.asyncio
async def test_automatic_fallback_when_primary_exhausted():
    mock_provider = MockProvider()
    mock_provider.set_forced_failures(3)  # Fails 3 primary attempts (initial + 2 retries), fallback succeeds

    manager = ReliabilityManager(max_retries=2, base_backoff=0.01, max_backoff=0.05)
    messages = [{"role": "user", "content": "Critical task"}]

    resp, retries, fallback_used = await manager.execute_with_resilience(
        provider=mock_provider,
        primary_model="failing-primary-model",
        messages=messages,
        fallback_model="backup-model",
    )

    assert resp is not None
    assert fallback_used is True
