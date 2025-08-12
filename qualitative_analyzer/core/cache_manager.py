"""Simple optional caching for API responses."""

import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleCacheManager:
    """Optional simple caching for API responses."""
    
    def __init__(
        self,
        cache_dir: str = "cache",
        max_age_days: int = 7,
        enabled: bool = True
    ):
        """Initialize cache manager."""
        self.enabled = enabled
        if not self.enabled:
            return
            
        self.cache_dir = Path(cache_dir) / "api_responses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_age = timedelta(days=max_age_days)
        
        # Simple statistics
        self.hits = 0
        self.misses = 0
    
    def _generate_cache_key(self, prompt: str) -> str:
        """Generate simple cache key from prompt."""
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()
    
    def get_api_response(self, prompt: str) -> Optional[str]:
        """Get cached API response."""
        if not self.enabled:
            return None
            
        cache_key = self._generate_cache_key(prompt)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            self.misses += 1
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cached_data["timestamp"])
            if datetime.now() - cache_time > self.max_age:
                cache_file.unlink()  # Remove expired cache
                self.misses += 1
                return None
            
            self.hits += 1
            logger.debug(f"Cache hit for API response")
            return cached_data["response"]
            
        except Exception as e:
            logger.warning(f"Failed to read cached API response: {e}")
            self.misses += 1
            return None
    
    def cache_api_response(self, prompt: str, response: str) -> None:
        """Cache API response."""
        if not self.enabled:
            return
            
        cache_key = self._generate_cache_key(prompt)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cache_entry = {
                "timestamp": datetime.now().isoformat(),
                "response": response
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=2)
            
            logger.debug(f"Cached API response")
            
        except Exception as e:
            logger.warning(f"Failed to cache API response: {e}")
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def clear_cache(self) -> int:
        """Clear all cache entries."""
        if not self.enabled:
            return 0
            
        removed = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")
        
        logger.info(f"Cleared {removed} cache entries")
        return removed