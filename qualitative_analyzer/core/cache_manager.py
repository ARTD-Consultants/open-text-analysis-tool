"""Intelligent caching system for API responses and embeddings."""

import os
import json
import hashlib
import pickle
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for API responses and computed results."""
    
    def __init__(
        self,
        cache_dir: str = "cache",
        max_age_days: int = 7,
        max_cache_size_mb: int = 100
    ):
        """Initialize cache manager."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_age = timedelta(days=max_age_days)
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        
        # Create subdirectories for different cache types
        self.api_cache_dir = self.cache_dir / "api_responses"
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        self.analysis_cache_dir = self.cache_dir / "analysis"
        
        for cache_subdir in [self.api_cache_dir, self.embedding_cache_dir, self.analysis_cache_dir]:
            cache_subdir.mkdir(exist_ok=True)
        
        # Load cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        return {
            "created": datetime.now().isoformat(),
            "last_cleanup": datetime.now().isoformat(),
            "total_hits": 0,
            "total_misses": 0,
            "cache_entries": {}
        }
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        elif isinstance(data, list):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
        
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, cache_type: str, cache_key: str) -> Path:
        """Get cache file path for a given type and key."""
        if cache_type == "api":
            return self.api_cache_dir / f"{cache_key}.json"
        elif cache_type == "embedding":
            return self.embedding_cache_dir / f"{cache_key}.pkl"
        elif cache_type == "analysis":
            return self.analysis_cache_dir / f"{cache_key}.pkl"
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def get_api_response(self, prompt: str, model_config: Dict[str, Any]) -> Optional[str]:
        """Get cached API response."""
        cache_data = {
            "prompt": prompt,
            "model_config": model_config
        }
        cache_key = self._generate_cache_key(cache_data)
        cache_file = self._get_cache_file_path("api", cache_key)
        
        if not cache_file.exists():
            self.metadata["total_misses"] += 1
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cached_data["timestamp"])
            if datetime.now() - cache_time > self.max_age:
                cache_file.unlink()  # Remove expired cache
                return None
            
            self.metadata["total_hits"] += 1
            logger.debug(f"Cache hit for API response: {cache_key[:8]}...")
            return cached_data["response"]
            
        except Exception as e:
            logger.warning(f"Failed to read cached API response: {e}")
            return None
    
    def cache_api_response(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        response: str
    ) -> None:
        """Cache API response."""
        cache_data = {
            "prompt": prompt,
            "model_config": model_config
        }
        cache_key = self._generate_cache_key(cache_data)
        cache_file = self._get_cache_file_path("api", cache_key)
        
        try:
            cache_entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt_hash": self._generate_cache_key(prompt),
                "response": response,
                "model_config": model_config
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=2)
            
            # Update metadata
            self.metadata["cache_entries"][cache_key] = {
                "type": "api",
                "created": cache_entry["timestamp"],
                "size_bytes": cache_file.stat().st_size
            }
            
            logger.debug(f"Cached API response: {cache_key[:8]}...")
            
        except Exception as e:
            logger.warning(f"Failed to cache API response: {e}")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding."""
        cache_key = self._generate_cache_key(text)
        cache_file = self._get_cache_file_path("embedding", cache_key)
        
        if not cache_file.exists():
            self.metadata["total_misses"] += 1
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cached_data["timestamp"])
            if datetime.now() - cache_time > self.max_age:
                cache_file.unlink()
                return None
            
            self.metadata["total_hits"] += 1
            logger.debug(f"Cache hit for embedding: {cache_key[:8]}...")
            return cached_data["embedding"]
            
        except Exception as e:
            logger.warning(f"Failed to read cached embedding: {e}")
            return None
    
    def cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding."""
        cache_key = self._generate_cache_key(text)
        cache_file = self._get_cache_file_path("embedding", cache_key)
        
        try:
            cache_entry = {
                "timestamp": datetime.now().isoformat(),
                "text_hash": cache_key,
                "embedding": embedding
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_entry, f)
            
            # Update metadata
            self.metadata["cache_entries"][cache_key] = {
                "type": "embedding",
                "created": cache_entry["timestamp"],
                "size_bytes": cache_file.stat().st_size
            }
            
            logger.debug(f"Cached embedding: {cache_key[:8]}...")
            
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def get_analysis_result(self, analysis_input: Dict[str, Any]) -> Optional[Any]:
        """Get cached analysis result."""
        cache_key = self._generate_cache_key(analysis_input)
        cache_file = self._get_cache_file_path("analysis", cache_key)
        
        if not cache_file.exists():
            self.metadata["total_misses"] += 1
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cached_data["timestamp"])
            if datetime.now() - cache_time > self.max_age:
                cache_file.unlink()
                return None
            
            self.metadata["total_hits"] += 1
            logger.debug(f"Cache hit for analysis: {cache_key[:8]}...")
            return cached_data["result"]
            
        except Exception as e:
            logger.warning(f"Failed to read cached analysis: {e}")
            return None
    
    def cache_analysis_result(self, analysis_input: Dict[str, Any], result: Any) -> None:
        """Cache analysis result."""
        cache_key = self._generate_cache_key(analysis_input)
        cache_file = self._get_cache_file_path("analysis", cache_key)
        
        try:
            cache_entry = {
                "timestamp": datetime.now().isoformat(),
                "input_hash": cache_key,
                "result": result
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_entry, f)
            
            # Update metadata
            self.metadata["cache_entries"][cache_key] = {
                "type": "analysis",
                "created": cache_entry["timestamp"],
                "size_bytes": cache_file.stat().st_size
            }
            
            logger.debug(f"Cached analysis result: {cache_key[:8]}...")
            
        except Exception as e:
            logger.warning(f"Failed to cache analysis result: {e}")
    
    def cleanup_expired_cache(self) -> Dict[str, int]:
        """Clean up expired cache entries."""
        removed_counts = {"api": 0, "embedding": 0, "analysis": 0}
        total_size_freed = 0
        
        current_time = datetime.now()
        
        for cache_type in ["api", "embedding", "analysis"]:
            cache_subdir = getattr(self, f"{cache_type}_cache_dir")
            
            for cache_file in cache_subdir.glob("*"):
                try:
                    if cache_type == "api":
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                    else:
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                    
                    cache_time = datetime.fromisoformat(cached_data["timestamp"])
                    
                    if current_time - cache_time > self.max_age:
                        file_size = cache_file.stat().st_size
                        cache_file.unlink()
                        removed_counts[cache_type] += 1
                        total_size_freed += file_size
                        
                        # Remove from metadata
                        cache_key = cache_file.stem
                        if cache_key in self.metadata["cache_entries"]:
                            del self.metadata["cache_entries"][cache_key]
                
                except Exception as e:
                    logger.warning(f"Error processing cache file {cache_file}: {e}")
        
        self.metadata["last_cleanup"] = current_time.isoformat()
        self._save_metadata()
        
        logger.info(f"Cache cleanup completed. Removed {sum(removed_counts.values())} files, "
                   f"freed {total_size_freed / 1024 / 1024:.1f} MB")
        
        return {
            **removed_counts,
            "total_removed": sum(removed_counts.values()),
            "size_freed_mb": total_size_freed / 1024 / 1024
        }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        total_files = 0
        total_size = 0
        type_counts = {"api": 0, "embedding": 0, "analysis": 0}
        type_sizes = {"api": 0, "embedding": 0, "analysis": 0}
        
        for cache_type in ["api", "embedding", "analysis"]:
            cache_subdir = getattr(self, f"{cache_type}_cache_dir")
            
            for cache_file in cache_subdir.glob("*"):
                if cache_file.is_file():
                    file_size = cache_file.stat().st_size
                    total_files += 1
                    total_size += file_size
                    type_counts[cache_type] += 1
                    type_sizes[cache_type] += file_size
        
        hit_rate = (
            self.metadata["total_hits"] / 
            (self.metadata["total_hits"] + self.metadata["total_misses"])
            if (self.metadata["total_hits"] + self.metadata["total_misses"]) > 0
            else 0
        )
        
        return {
            "total_files": total_files,
            "total_size_mb": total_size / 1024 / 1024,
            "cache_hit_rate": hit_rate,
            "total_hits": self.metadata["total_hits"],
            "total_misses": self.metadata["total_misses"],
            "type_breakdown": {
                cache_type: {
                    "count": type_counts[cache_type],
                    "size_mb": type_sizes[cache_type] / 1024 / 1024
                }
                for cache_type in type_counts
            },
            "last_cleanup": self.metadata["last_cleanup"]
        }
    
    def clear_cache(self, cache_type: Optional[str] = None) -> Dict[str, int]:
        """Clear cache entries."""
        removed_counts = {"api": 0, "embedding": 0, "analysis": 0}
        
        cache_types_to_clear = [cache_type] if cache_type else ["api", "embedding", "analysis"]
        
        for ctype in cache_types_to_clear:
            if ctype not in ["api", "embedding", "analysis"]:
                continue
                
            cache_subdir = getattr(self, f"{ctype}_cache_dir")
            
            for cache_file in cache_subdir.glob("*"):
                try:
                    cache_file.unlink()
                    removed_counts[ctype] += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        # Clear metadata for removed entries
        if not cache_type:  # Clear all metadata if clearing all caches
            self.metadata["cache_entries"] = {}
            self.metadata["total_hits"] = 0
            self.metadata["total_misses"] = 0
        
        self._save_metadata()
        
        total_removed = sum(removed_counts.values())
        logger.info(f"Cleared {total_removed} cache entries")
        
        return removed_counts