#!/usr/bin/env python3
"""
Caching utilities for expensive operations.

This module provides caching mechanisms for expensive operations like API calls,
transcription, and processing, helping to save time and resources when
working with large files or repeatedly processing the same content.
"""
import os
import json
import hashlib
import time
import logging
from typing import Dict, Any, Optional, Union, List, Callable
import pickle
from pathlib import Path

from transcribe_pkg.utils.logging_utils import get_logger

class CacheManager:
    """
    Manage caching of results for expensive operations.
    
    This class provides flexible caching with different storage backends,
    supporting both memory and disk-based caching with configurable
    expiration and invalidation policies.
    """
    
    def __init__(
        self,
        cache_dir: str = None,
        max_memory_items: int = 1000,
        disk_cache_enabled: bool = True,
        memory_cache_enabled: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for disk cache (default: ~/.cache/transcribe)
            max_memory_items: Maximum number of items in memory cache
            disk_cache_enabled: Whether disk caching is enabled
            memory_cache_enabled: Whether memory caching is enabled
            logger: Logger instance
        """
        self.logger = logger or get_logger(__name__)
        
        # Set up cache directory
        if cache_dir is None:
            home_dir = os.path.expanduser("~")
            cache_dir = os.path.join(home_dir, ".cache", "transcribe")
        
        self.cache_dir = cache_dir
        self.max_memory_items = max_memory_items
        self.disk_cache_enabled = disk_cache_enabled
        self.memory_cache_enabled = memory_cache_enabled
        
        # Create memory cache
        self.memory_cache = {}
        self.memory_timestamp = {}
        
        # Create cache directory if it doesn't exist
        if self.disk_cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def get(
        self, 
        key: str, 
        cache_type: str = "both", 
        max_age: Optional[float] = None
    ) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            cache_type: Type of cache to check ('memory', 'disk', or 'both')
            max_age: Maximum age of cached item in seconds (None = no limit)
            
        Returns:
            Cached value or None if not found or expired
        """
        # Generate a consistent hash key
        hashed_key = self._hash_key(key)
        
        # Check memory cache first if enabled
        if (cache_type in ["memory", "both"]) and self.memory_cache_enabled:
            if hashed_key in self.memory_cache:
                timestamp = self.memory_timestamp.get(hashed_key, 0)
                
                # Check if item is expired
                if max_age is None or (time.time() - timestamp) <= max_age:
                    self.logger.debug(f"Cache hit (memory): {key[:50]}...")
                    return self.memory_cache[hashed_key]
                else:
                    # Remove expired item
                    self._remove_from_memory(hashed_key)
        
        # Check disk cache if enabled
        if (cache_type in ["disk", "both"]) and self.disk_cache_enabled:
            cache_path = self._get_cache_path(hashed_key)
            
            if os.path.exists(cache_path):
                # Check if item is expired
                if max_age is not None:
                    modified_time = os.path.getmtime(cache_path)
                    if (time.time() - modified_time) > max_age:
                        # Remove expired file
                        self._remove_from_disk(hashed_key)
                        return None
                
                # Load item from disk
                try:
                    with open(cache_path, "rb") as f:
                        cached_value = pickle.load(f)
                    
                    # Add to memory cache for faster access next time
                    if self.memory_cache_enabled:
                        self._add_to_memory(hashed_key, cached_value)
                    
                    self.logger.debug(f"Cache hit (disk): {key[:50]}...")
                    return cached_value
                    
                except Exception as e:
                    self.logger.warning(f"Error loading cache from disk: {str(e)}")
        
        # Not found in any cache
        self.logger.debug(f"Cache miss: {key[:50]}...")
        return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        cache_type: str = "both"
    ) -> None:
        """
        Store an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cache to store in ('memory', 'disk', or 'both')
        """
        # Generate a consistent hash key
        hashed_key = self._hash_key(key)
        
        # Store in memory cache
        if (cache_type in ["memory", "both"]) and self.memory_cache_enabled:
            self._add_to_memory(hashed_key, value)
        
        # Store in disk cache
        if (cache_type in ["disk", "both"]) and self.disk_cache_enabled:
            try:
                cache_path = self._get_cache_path(hashed_key)
                
                # Save to disk atomically to prevent corruption
                tmp_path = f"{cache_path}.tmp"
                with open(tmp_path, "wb") as f:
                    pickle.dump(value, f)
                os.replace(tmp_path, cache_path)
                
                self.logger.debug(f"Cache set (disk): {key[:50]}...")
            except Exception as e:
                self.logger.warning(f"Error saving cache to disk: {str(e)}")
    
    def invalidate(self, key: str, cache_type: str = "both") -> None:
        """
        Remove an item from the cache.
        
        Args:
            key: Cache key to invalidate
            cache_type: Type of cache to invalidate ('memory', 'disk', or 'both')
        """
        hashed_key = self._hash_key(key)
        
        # Remove from memory cache
        if (cache_type in ["memory", "both"]) and self.memory_cache_enabled:
            self._remove_from_memory(hashed_key)
            
        # Remove from disk cache
        if (cache_type in ["disk", "both"]) and self.disk_cache_enabled:
            self._remove_from_disk(hashed_key)
    
    def clear(self, cache_type: str = "both") -> None:
        """
        Clear all items from the cache.
        
        Args:
            cache_type: Type of cache to clear ('memory', 'disk', or 'both')
        """
        # Clear memory cache
        if (cache_type in ["memory", "both"]) and self.memory_cache_enabled:
            self.memory_cache.clear()
            self.memory_timestamp.clear()
            self.logger.debug("Memory cache cleared")
        
        # Clear disk cache
        if (cache_type in ["disk", "both"]) and self.disk_cache_enabled:
            try:
                for file_name in os.listdir(self.cache_dir):
                    if file_name.endswith(".cache"):
                        try:
                            os.remove(os.path.join(self.cache_dir, file_name))
                        except:
                            pass
                self.logger.debug("Disk cache cleared")
            except Exception as e:
                self.logger.warning(f"Error clearing disk cache: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        # Count disk cache items and size
        disk_items = 0
        disk_size = 0
        
        if self.disk_cache_enabled and os.path.exists(self.cache_dir):
            try:
                for file_name in os.listdir(self.cache_dir):
                    if file_name.endswith(".cache"):
                        file_path = os.path.join(self.cache_dir, file_name)
                        disk_items += 1
                        disk_size += os.path.getsize(file_path)
            except Exception:
                pass
        
        return {
            "memory_items": len(self.memory_cache),
            "memory_enabled": self.memory_cache_enabled,
            "disk_items": disk_items,
            "disk_size": disk_size,
            "disk_enabled": self.disk_cache_enabled,
            "cache_dir": self.cache_dir
        }
    
    def _hash_key(self, key: str) -> str:
        """
        Generate a hash for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Hashed key
        """
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, hashed_key: str) -> str:
        """
        Get the file path for a disk cache item.
        
        Args:
            hashed_key: Hashed cache key
            
        Returns:
            File path for the cache item
        """
        return os.path.join(self.cache_dir, f"{hashed_key}.cache")
    
    def _add_to_memory(self, hashed_key: str, value: Any) -> None:
        """
        Add an item to the memory cache, evicting old items if needed.
        
        Args:
            hashed_key: Hashed cache key
            value: Value to cache
        """
        # Check if we need to evict items
        if len(self.memory_cache) >= self.max_memory_items:
            # Find the oldest item
            oldest_key = min(
                self.memory_timestamp.keys(),
                key=lambda k: self.memory_timestamp.get(k, 0)
            )
            self._remove_from_memory(oldest_key)
        
        # Add new item
        self.memory_cache[hashed_key] = value
        self.memory_timestamp[hashed_key] = time.time()
        self.logger.debug(f"Cache set (memory): {hashed_key[:8]}...")
    
    def _remove_from_memory(self, hashed_key: str) -> None:
        """
        Remove an item from the memory cache.
        
        Args:
            hashed_key: Hashed cache key
        """
        if hashed_key in self.memory_cache:
            del self.memory_cache[hashed_key]
        
        if hashed_key in self.memory_timestamp:
            del self.memory_timestamp[hashed_key]
    
    def _remove_from_disk(self, hashed_key: str) -> None:
        """
        Remove an item from the disk cache.
        
        Args:
            hashed_key: Hashed cache key
        """
        cache_path = self._get_cache_path(hashed_key)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception as e:
                self.logger.warning(f"Error removing cache file: {str(e)}")

def cached(
    key_func: Optional[Callable] = None,
    cache_type: str = "both",
    max_age: Optional[float] = None,
    cache_manager: Optional[CacheManager] = None
) -> Callable:
    """
    Decorator for caching function results.
    
    Args:
        key_func: Function to generate cache key from arguments
        cache_type: Type of cache to use ('memory', 'disk', or 'both')
        max_age: Maximum age of cached result in seconds
        cache_manager: CacheManager instance (creates a new one if None)
        
    Returns:
        Decorated function
    """
    # Create default cache manager if not provided
    if cache_manager is None:
        cache_manager = CacheManager()
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_func is not None:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key is function name and args/kwargs
                arg_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__module__}.{func.__name__}:{arg_str}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key, cache_type, max_age)
            if cached_result is not None:
                return cached_result
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Cache the result
            cache_manager.set(cache_key, result, cache_type)
            
            return result
        
        return wrapper
    
    return decorator

#fin