"""In-memory response caching with Time-To-Live (TTL) and LRU eviction."""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from app.config import get_settings


class CacheEntry:
    def __init__(self, data: Dict[str, Any], expires_at: float):
        self.data = data
        self.expires_at = expires_at


class ResponseCache:
    """Async-safe in-memory LRU response cache with TTL."""

    def __init__(self, max_entries: Optional[int] = None, ttl_seconds: Optional[int] = None):
        settings = get_settings()
        self.max_entries = max_entries or settings.CACHE_MAX_ENTRIES
        self.ttl_seconds = ttl_seconds or settings.CACHE_TTL_SECONDS
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def generate_key(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Create a deterministic SHA-256 fingerprint for the request payload."""
        payload = {
            "model": model.strip().lower(),
            "messages": [
                {"role": m.get("role"), "content": m.get("content", "").strip()}
                for m in messages
            ],
            "temperature": round(temperature, 2),
            "max_tokens": max_tokens,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if key exists and has not expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None

        now = time.time()
        if now > entry.expires_at:
            # Expired -> prune
            del self._cache[key]
            return None

        # Move to end for LRU refresh
        self._cache.move_to_end(key)
        return entry.data

    async def set(self, key: str, data: Dict[str, Any], custom_ttl: Optional[int] = None) -> None:
        """Store response with expiration timestamp, evicting oldest if capacity reached."""
        ttl = custom_ttl or self.ttl_seconds
        expires_at = time.time() + ttl

        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.max_entries:
            # Evict least recently used (first item in OrderedDict)
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(data=data, expires_at=expires_at)

    async def clear(self) -> None:
        """Clear all entries in cache."""
        self._cache.clear()

    async def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)


# Global cache instance
response_cache = ResponseCache()
