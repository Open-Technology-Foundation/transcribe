#!/usr/bin/env python3
"""
Caching utilities for the transcribe package.

This module provides caching functionality to reduce redundant API calls and processing
operations. It implements a disk-based caching system that stores serialized data to
improve performance and reduce costs.

Key features:
- Function result caching via the @cached decorator
- Persistent storage of cached data on disk
- Automatic cache key generation based on function arguments
- Configurable cache size and location
- Thread-safe cache access
- Environment variable control for cache behavior

The caching system can be controlled through environment variables:
- TRANSCRIBE_CACHE_DISABLED: Set to 1 to disable caching
- TRANSCRIBE_CACHE_DIR: Custom directory for cache storage
- TRANSCRIBE_CACHE_SIZE_MB: Maximum cache size in megabytes
"""

import os
import json
import hashlib
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps

# Global cache instance
_cache = None
_cache_lock = threading.RLock()

class Cache:
    """
    A simple disk-based cache for API responses and processed data.
    
    This cache stores serialized data on disk to avoid redundant API calls
    and processing operations.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 100):
        """
        Initialize the cache.
        
        Args:
            cache_dir (str, optional): Directory to store cache files. 
                                       Defaults to a temp directory.
            max_size_mb (int, optional): Maximum cache size in MB. Defaults to 100.
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            # Create a persistent cache directory in the system temp directory
            tmp_dir = Path(tempfile.gettempdir())
            self.cache_dir = tmp_dir / 'transcribe_cache'
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._metadata_path = self.cache_dir / 'metadata.json'
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load cache metadata from disk.
        
        Returns:
            Dict: Cache metadata
        """
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logging.warning(f"Error loading cache metadata: {str(e)}")
        
        # Default metadata
        return {
            'entries': {},
            'total_size_bytes': 0
        }
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self._metadata_path, 'w') as f:
                json.dump(self._metadata, f)
        except OSError as e:
            logging.warning(f"Error saving cache metadata: {str(e)}")
    
    def _get_cache_key(self, key_data: Any) -> str:
        """
        Generate a unique cache key from input data.
        
        Args:
            key_data: Data to hash for the key
            
        Returns:
            str: Cache key as a hex string
        """
        # Convert to JSON and hash
        if isinstance(key_data, str):
            data_str = key_data
        else:
            # Convert to a serializable form
            serializable_data = self._make_serializable(key_data)
            data_str = json.dumps(serializable_data, sort_keys=True)
        
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()
        
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert non-serializable objects to serializable types.
        
        This method recursively processes Python objects and converts them to types that
        can be serialized to JSON. It handles complex objects that cannot be directly
        serialized, ensuring they can be properly cached.
        
        Args:
            obj: Object to make serializable - can be any Python object
            
        Returns:
            Object in a serializable form (dict, list, str, int, float, bool, None)
            
        Notes:
            - Dictionaries are processed recursively for each key-value pair
            - Lists and tuples are processed recursively for each element
            - Primitive types (str, int, float, bool, None) are returned as-is
            - All other objects are converted to their string representation
            - This approach handles circular references by converting to strings
        """
        if isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(i) for i in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # For non-serializable types, convert to string representation
            return str(obj)
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get the file path for a cache entry.
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            Path: Path to cache file
        """
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, key_data: Any, default: Any = None) -> Any:
        """
        Get data from cache.
        
        Args:
            key_data: Data to hash for the key
            default: Default value if not in cache
            
        Returns:
            Any: Cached data or default
        """
        with _cache_lock:
            cache_key = self._get_cache_key(key_data)
            cache_path = self._get_cache_path(cache_key)
            
            # Check if in cache
            if cache_key in self._metadata['entries'] and cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        cached_data = json.load(f)
                    
                    # Update access timestamp
                    self._metadata['entries'][cache_key]['last_accessed'] = os.path.getmtime(cache_path)
                    self._save_metadata()
                    
                    return cached_data
                except (json.JSONDecodeError, OSError) as e:
                    logging.warning(f"Error reading cache entry: {str(e)}")
            
            return default
    
    def set(self, key_data: Any, value: Any) -> None:
        """
        Store data in cache.
        
        Args:
            key_data: Data to hash for the key
            value: Data to cache
        """
        with _cache_lock:
            cache_key = self._get_cache_key(key_data)
            cache_path = self._get_cache_path(cache_key)
            
            try:
                # Serialize data
                serialized = json.dumps(value)
                entry_size = len(serialized.encode('utf-8'))
                
                # Check if we need to make room
                if self._metadata['total_size_bytes'] + entry_size > self.max_size_bytes:
                    self._evict_entries(entry_size)
                
                # Write to disk
                with open(cache_path, 'w') as f:
                    f.write(serialized)
                
                # Update metadata
                self._metadata['entries'][cache_key] = {
                    'size_bytes': entry_size,
                    'last_accessed': os.path.getmtime(cache_path),
                    'created': os.path.getctime(cache_path)
                }
                self._metadata['total_size_bytes'] += entry_size
                self._save_metadata()
                
            except (TypeError, OSError) as e:
                logging.warning(f"Error caching data: {str(e)}")
    
    def delete(self, key_data: Any) -> bool:
        """
        Delete an entry from the cache.
        
        Args:
            key_data: Data to hash for the key
            
        Returns:
            bool: True if deleted, False otherwise
        """
        with _cache_lock:
            cache_key = self._get_cache_key(key_data)
            cache_path = self._get_cache_path(cache_key)
            
            if cache_key in self._metadata['entries']:
                try:
                    if cache_path.exists():
                        os.remove(cache_path)
                    
                    # Update metadata
                    entry_size = self._metadata['entries'][cache_key]['size_bytes']
                    self._metadata['total_size_bytes'] -= entry_size
                    del self._metadata['entries'][cache_key]
                    self._save_metadata()
                    return True
                except OSError as e:
                    logging.warning(f"Error deleting cache entry: {str(e)}")
            
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with _cache_lock:
            try:
                # Delete all cache files
                for entry_key in self._metadata['entries']:
                    cache_path = self._get_cache_path(entry_key)
                    if cache_path.exists():
                        os.remove(cache_path)
                
                # Reset metadata
                self._metadata = {
                    'entries': {},
                    'total_size_bytes': 0
                }
                self._save_metadata()
                
                logging.info("Cache cleared")
            except OSError as e:
                logging.warning(f"Error clearing cache: {str(e)}")
    
    def _evict_entries(self, needed_bytes: int) -> None:
        """
        Evict entries to make room for new data.
        
        Args:
            needed_bytes (int): Number of bytes needed
        """
        # Sort entries by last accessed time (oldest first)
        entries = sorted(
            self._metadata['entries'].items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        bytes_freed = 0
        for key, entry in entries:
            # Skip if we've freed enough space
            if bytes_freed >= needed_bytes:
                break
            
            # Delete the entry
            cache_path = self._get_cache_path(key)
            try:
                if cache_path.exists():
                    os.remove(cache_path)
                
                # Update metadata
                bytes_freed += entry['size_bytes']
                self._metadata['total_size_bytes'] -= entry['size_bytes']
                del self._metadata['entries'][key]
                
            except OSError as e:
                logging.warning(f"Error evicting cache entry: {str(e)}")
        
        # Save updated metadata
        self._save_metadata()
        
        if bytes_freed < needed_bytes:
            logging.warning(
                f"Could only free {bytes_freed} bytes, but needed {needed_bytes}. "
                f"Cache may exceed size limit."
            )

def get_cache() -> Cache:
    """
    Get the global cache instance.
    
    Returns:
        Cache: Global cache instance
    """
    global _cache
    with _cache_lock:
        if _cache is None:
            # Use environment variable for cache directory if available
            cache_dir = os.getenv('TRANSCRIBE_CACHE_DIR')
            max_size_mb = int(os.getenv('TRANSCRIBE_CACHE_SIZE_MB', '100'))
            _cache = Cache(cache_dir, max_size_mb)
        return _cache

def cached(key_func: Optional[Callable] = None):
    """
    Decorator to cache function results in a persistent disk-based cache.
    
    This decorator provides a way to cache function results, particularly useful
    for expensive operations like API calls, transcription processing, or other
    computationally intensive tasks. It automatically handles serialization and
    deserialization of the cached data.
    
    Args:
        key_func (callable, optional): Custom function to generate cache keys from arguments.
                                      If not provided, a default key generator is used that
                                      hashes function args and kwargs.
    
    Returns:
        callable: Decorated function with caching capability
        
    Notes:
        - Cache can be disabled by setting TRANSCRIBE_NO_CACHE=1 environment variable
        - Uses the global cache instance with configuration from environment variables
        - Thread-safe for concurrent access
        - Automatically handles serialization of complex objects
        
    Example:
        @cached
        def fetch_data_from_api(query, options=None):
            # This call will be cached based on query and options
            return expensive_api_call(query, options)
            
        # With custom key function
        @cached(key_func=lambda url, **_: url)
        def download_file(url, target_dir, options=None):
            # Cache key is just the URL, ignoring other arguments
            return download_and_process(url, target_dir, options)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if caching is disabled
            if os.environ.get('TRANSCRIBE_NO_CACHE') == '1':
                return func(*args, **kwargs)
                
            # Generate cache key
            try:
                if key_func:
                    # Create a copy of args without 'self' for instance methods
                    func_args = args
                    if args and hasattr(args[0], '__class__'):
                        func_args = args[1:]
                        
                    # Apply key function with proper args
                    if func_args:
                        key = key_func(*func_args, **kwargs)
                    else:
                        key = key_func(**kwargs)
                else:
                    # Use function name, args, and kwargs as key
                    key = {
                        'func': func.__name__,
                        'args': str(args),
                        'kwargs': str(kwargs)
                    }
            except Exception as e:
                logging.warning(f"Error generating cache key: {str(e)}")
                # Fall back to function call without caching
                return func(*args, **kwargs)
            
            # Check cache
            try:
                cache = get_cache()
                cached_result = cache.get(key)
                
                if cached_result is not None:
                    logging.debug(f"Cache hit for {func.__name__}")
                    return cached_result
            except Exception as e:
                logging.warning(f"Error checking cache: {str(e)}")
                # Fall back to function call without caching
                return func(*args, **kwargs)
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                try:
                    cache.set(key, result)
                except Exception as e:
                    logging.warning(f"Error caching result: {str(e)}")
            
            return result
        
        return wrapper
    
    # Handle case where decorator is used without arguments
    if callable(key_func):
        func = key_func
        key_func = None
        return decorator(func)
    
    return decorator

#fin