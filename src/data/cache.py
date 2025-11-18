"""Data caching system with TTL support."""

import os
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

from ..utils.logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class DataCache:
    """
    File-based caching system with TTL (Time To Live) support.

    Supports different TTLs for intraday and historical data.
    """

    def __init__(
        self,
        cache_dir: str = None,
        ttl_intraday: int = None,
        ttl_historical: int = None,
    ):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory for cache files (default: .cache)
            ttl_intraday: TTL for intraday data in seconds (default: 3600 = 1 hour)
            ttl_historical: TTL for historical data in seconds (default: 86400 = 24 hours)
        """
        self.cache_dir = Path(cache_dir or os.getenv("CACHE_DIR", ".cache"))
        self.ttl_intraday = ttl_intraday or int(os.getenv("CACHE_TTL_INTRADAY", 3600))
        self.ttl_historical = ttl_historical or int(os.getenv("CACHE_TTL_HISTORICAL", 86400))

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"Cache initialized: dir={self.cache_dir}, "
            f"ttl_intraday={self.ttl_intraday}s, ttl_historical={self.ttl_historical}s"
        )

    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate a unique cache key from arguments.

        Args:
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key

        Returns:
            MD5 hash string as cache key
        """
        # Create a string representation of all arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_string = "|".join(key_parts)

        # Generate MD5 hash
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.pkl"

    def get(
        self,
        key: str = None,
        ttl: int = None,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Retrieve data from cache if valid.

        Args:
            key: Cache key (if None, generated from args/kwargs)
            ttl: Custom TTL override in seconds
            *args: Arguments for key generation
            **kwargs: Keyword arguments for key generation

        Returns:
            Cached data if valid, None otherwise
        """
        if key is None:
            key = self._generate_key(*args, **kwargs)

        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {key}")
            return None

        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)

            # Check TTL
            cached_time = cached.get("timestamp", 0)
            cached_ttl = ttl or cached.get("ttl", self.ttl_historical)

            if time.time() - cached_time > cached_ttl:
                logger.debug(f"Cache expired: {key}")
                self.delete(key)
                return None

            logger.debug(f"Cache hit: {key}")
            return cached.get("data")

        except (pickle.PickleError, KeyError, EOFError) as e:
            logger.warning(f"Cache read error for {key}: {e}")
            self.delete(key)
            return None

    def set(
        self,
        data: Any,
        key: str = None,
        ttl: int = None,
        data_type: str = "historical",
        *args,
        **kwargs
    ) -> str:
        """
        Store data in cache.

        Args:
            data: Data to cache
            key: Cache key (if None, generated from args/kwargs)
            ttl: Custom TTL in seconds
            data_type: "intraday" or "historical" for default TTL
            *args: Arguments for key generation
            **kwargs: Keyword arguments for key generation

        Returns:
            Cache key used
        """
        if key is None:
            key = self._generate_key(*args, **kwargs)

        # Determine TTL
        if ttl is None:
            ttl = self.ttl_intraday if data_type == "intraday" else self.ttl_historical

        cache_path = self._get_cache_path(key)

        try:
            cached = {
                "data": data,
                "timestamp": time.time(),
                "ttl": ttl,
                "data_type": data_type,
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cached, f)

            logger.debug(f"Cache set: {key} (ttl={ttl}s)")
            return key

        except (pickle.PickleError, IOError) as e:
            logger.error(f"Cache write error for {key}: {e}")
            return key

    def delete(self, key: str) -> bool:
        """
        Delete a cache entry.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted, False if not found
        """
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Cache deleted: {key}")
            return True

        return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1

        logger.info(f"Cache cleared: {count} entries deleted")
        return count

    def clear_expired(self) -> int:
        """
        Clear only expired cache entries.

        Returns:
            Number of expired entries deleted
        """
        count = 0
        current_time = time.time()

        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)

                cached_time = cached.get("timestamp", 0)
                cached_ttl = cached.get("ttl", self.ttl_historical)

                if current_time - cached_time > cached_ttl:
                    cache_file.unlink()
                    count += 1

            except (pickle.PickleError, IOError, EOFError):
                # Delete corrupted cache files
                cache_file.unlink()
                count += 1

        logger.info(f"Expired cache cleared: {count} entries deleted")
        return count

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_size = 0
        total_files = 0
        expired_files = 0
        current_time = time.time()

        for cache_file in self.cache_dir.glob("*.pkl"):
            total_files += 1
            total_size += cache_file.stat().st_size

            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)

                cached_time = cached.get("timestamp", 0)
                cached_ttl = cached.get("ttl", self.ttl_historical)

                if current_time - cached_time > cached_ttl:
                    expired_files += 1

            except (pickle.PickleError, IOError, EOFError):
                expired_files += 1

        return {
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "expired_files": expired_files,
            "valid_files": total_files - expired_files,
            "cache_dir": str(self.cache_dir),
        }

    def get_or_set(
        self,
        func: callable,
        key: str = None,
        ttl: int = None,
        data_type: str = "historical",
        *args,
        **kwargs
    ) -> Any:
        """
        Get from cache or execute function and cache result.

        Args:
            func: Function to execute if cache miss
            key: Cache key
            ttl: Custom TTL in seconds
            data_type: "intraday" or "historical"
            *args: Arguments for function and key generation
            **kwargs: Keyword arguments for function and key generation

        Returns:
            Cached or fresh data
        """
        # Generate key from function name and arguments if not provided
        if key is None:
            func_name = getattr(func, "__name__", str(func))
            key = self._generate_key(func_name, *args, **kwargs)

        # Try to get from cache
        data = self.get(key, ttl)
        if data is not None:
            return data

        # Execute function and cache result
        data = func(*args, **kwargs)
        if data is not None:
            self.set(data, key, ttl, data_type)

        return data
