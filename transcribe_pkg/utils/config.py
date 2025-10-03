#!/usr/bin/env python3
"""
Configuration management for the transcribe package.

This module provides a centralized configuration system for the transcribe package,
handling configuration from multiple sources with appropriate precedence:
1. Command line arguments (highest priority)
2. Environment variables
3. Configuration files
4. Default values (lowest priority)

Key features:
- Multi-level configuration with dot notation access
- Environment variable integration
- Configuration file loading and saving
- Command-line argument parsing
- Default values for all settings
- Deep-copy protection to prevent accidental modification

The configuration system handles serialization of values for storage and provides
validation of user-provided values.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Any

# Default configuration values
DEFAULT_CONFIG = {
  # API settings
  "openai": {
    "api_key": "",  # Will be loaded from environment
    "models": {
      "transcription": "whisper-1",
      "completion": "gpt-4o",
      "summary": "gpt-4o-mini"
    }
  },
  
  # Transcription settings
  "transcription": {
    "chunk_length_ms": 600000,  # 10 minutes
    "temperature": 0.05,
    "parallel": True,
    "max_workers": 4
  },
  
  # Processing settings
  "processing": {
    "max_chunk_size": 3000,
    "temperature": 0.1,
    "max_tokens": 4096,
    "post_processing": True
  },
  
  # Output settings
  "output": {
    "save_raw": True,
    "create_paragraphs": True,
    "min_sentences_per_paragraph": 2,
    "max_sentences_per_paragraph": 8
  },
  
  # System settings
  "system": {
    "temp_dir": "",  # Will use system temp by default
    "log_level": "INFO"
  }
}

class Config:
  """Configuration manager for transcribe package."""
  
  def __init__(self, config_file: str | None = None):
    """
    Initialize configuration with optional config file.
    
    Args:
      config_file (str, optional): Path to configuration file. Defaults to None.
    """
    self._config = DEFAULT_CONFIG.copy()
    self._config_file = config_file
    
    # Load configuration from file if provided
    if config_file:
      self.load_from_file(config_file)
    
    # Load configuration from environment variables
    self.load_from_env()
  
  def load_from_file(self, config_file: str) -> None:
    """
    Load configuration from file.
    
    Args:
      config_file (str): Path to configuration file
      
    Raises:
      FileNotFoundError: If config file doesn't exist
      json.JSONDecodeError: If config file is not valid JSON
    """
    path = Path(config_file)
    if not path.exists():
      logging.warning(f"Configuration file not found: {config_file}")
      return
    
    try:
      with open(path, 'r') as f:
        file_config = json.load(f)
        
      # Update configuration with values from file
      self._update_nested_dict(self._config, file_config)
      logging.info(f"Loaded configuration from {config_file}")
      
    except json.JSONDecodeError as e:
      logging.error(f"Invalid JSON in configuration file: {str(e)}")
      raise
    except Exception as e:
      logging.error(f"Error loading configuration from file: {str(e)}")
      raise
  
  def load_from_env(self) -> None:
    """Load configuration from environment variables."""
    # Load API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
      self._config['openai']['api_key'] = api_key
    
    # Load model overrides from environment
    completion_model = os.getenv('OPENAI_COMPLETION_MODEL')
    if completion_model:
      self._config['openai']['models']['completion'] = completion_model
      
    summary_model = os.getenv('OPENAI_SUMMARY_MODEL')
    if summary_model:
      self._config['openai']['models']['summary'] = summary_model
    
    # Load transcription settings from environment
    chunk_length = os.getenv('TRANSCRIBE_CHUNK_LENGTH_MS')
    if chunk_length and chunk_length.isdigit():
      self._config['transcription']['chunk_length_ms'] = int(chunk_length)
      
    # Load parallel processing settings
    parallel = os.getenv('TRANSCRIBE_PARALLEL')
    if parallel is not None:
      self._config['transcription']['parallel'] = parallel.lower() in ('true', 'yes', '1')
      
    max_workers = os.getenv('TRANSCRIBE_MAX_WORKERS')
    if max_workers and max_workers.isdigit():
      self._config['transcription']['max_workers'] = int(max_workers)
    
    # Load processing settings
    max_chunk_size = os.getenv('TRANSCRIBE_MAX_CHUNK_SIZE')
    if max_chunk_size and max_chunk_size.isdigit():
      self._config['processing']['max_chunk_size'] = int(max_chunk_size)
      
    # Load system settings
    temp_dir = os.getenv('TRANSCRIBE_TEMP_DIR')
    if temp_dir:
      self._config['system']['temp_dir'] = temp_dir
      
    log_level = os.getenv('TRANSCRIBE_LOG_LEVEL')
    if log_level:
      self._config['system']['log_level'] = log_level
  
  def save_to_file(self, config_file: str | None = None) -> None:
    """
    Save configuration to file.
    
    Args:
      config_file (str, optional): Path to save configuration file. 
                                   Defaults to original config file.
    """
    path = Path(config_file or self._config_file)
    if not path:
      logging.warning("No configuration file specified for saving")
      return
    
    try:
      # Create directory if it doesn't exist
      path.parent.mkdir(parents=True, exist_ok=True)
      
      # Make a serializable copy of the config (convert any non-serializable objects to strings)
      config_copy = self._clean_for_json(self._config)
      
      with open(path, 'w') as f:
        json.dump(config_copy, f, indent=2)
        
      logging.info(f"Configuration saved to {path}")
      
    except Exception as e:
      logging.error(f"Error saving configuration to file: {str(e)}")
      raise
  
  def _clean_for_json(self, obj: Any) -> Any:
    """
    Convert non-serializable objects to strings for JSON serialization.
    
    This helper method recursively processes complex data structures (dictionaries, lists)
    and ensures all objects are JSON serializable. Non-serializable objects are converted
    to their string representation.
    
    Args:
      obj (Any): The object to make JSON serializable
      
    Returns:
      Any: A JSON serializable version of the input object
      
    Note:
      Handles the following types of non-serializable objects:
      - Custom objects with __dict__ (converted to string representation)
      - MagicMock and other test objects
      - Circular references (strings are safe from this issue)
      
    Example:
      config_dict = {'valid': 123, 'invalid': MagicMock()}
      # Returns {'valid': 123, 'invalid': 'MagicMock object at 0x...'}
    """
    if isinstance(obj, dict):
      return {key: self._clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
      return [self._clean_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):  # Handle custom objects
      return str(obj)
    else:
      try:
        # Test if object is JSON serializable
        json.dumps(obj)
        return obj
      except (TypeError, OverflowError):
        # If not serializable, convert to string
        return str(obj)
  
  def get(self, key: str, default: Any = None) -> Any:
    """
    Get configuration value by key.
    
    Args:
      key (str): Configuration key in dot notation (e.g., 'openai.models.completion')
      default (Any, optional): Default value if key not found. Defaults to None.
      
    Returns:
      Any: Configuration value
    """
    keys = key.split('.')
    value = self._config
    
    try:
      for k in keys:
        value = value[k]
      return value
    except (KeyError, TypeError):
      return default
  
  def set(self, key: str, value: Any) -> None:
    """
    Set configuration value by key.
    
    Args:
      key (str): Configuration key in dot notation (e.g., 'openai.models.completion')
      value (Any): Value to set
    """
    keys = key.split('.')
    config = self._config
    
    # Navigate to the nested dictionary
    for k in keys[:-1]:
      if k not in config:
        config[k] = {}
      config = config[k]
    
    # Set the value
    config[keys[-1]] = value
  
  def update(self, updates: dict[str, Any]) -> None:
    """
    Update configuration with dictionary of values.
    
    Args:
      updates (Dict[str, Any]): Dictionary of configuration updates
    """
    self._update_nested_dict(self._config, updates)
  
  def _update_nested_dict(self, d: dict[str, Any], u: dict[str, Any]) -> None:
    """
    Update nested dictionary recursively.
    
    This helper method recursively updates a nested dictionary with values from another
    dictionary. It preserves the structure of nested dictionaries rather than overwriting
    them completely. This allows partial updates of configuration sections.
    
    Args:
      d (Dict[str, Any]): Target dictionary to update (modified in place)
      u (Dict[str, Any]): Source dictionary with updates
      
    Note:
      This method modifies the target dictionary in place and does not return a value.
      It preserves nested dictionaries in the target when the same key in the source
      is also a dictionary.
      
    Example:
      target = {'a': {'b': 1, 'c': 2}}
      updates = {'a': {'b': 10, 'd': 20}}
      # After update: target = {'a': {'b': 10, 'c': 2, 'd': 20}}
    """
    for k, v in u.items():
      if isinstance(v, dict) and k in d and isinstance(d[k], dict):
        self._update_nested_dict(d[k], v)
      else:
        d[k] = v
  
  @property
  def as_dict(self) -> dict[str, Any]:
    """Get full configuration as dictionary."""
    # Make a deep copy to prevent modification of the internal config
    import copy
    return copy.deepcopy(self._config)
  
  def __str__(self) -> str:
    """String representation of configuration."""
    return json.dumps(self._config, indent=2)
  
  @classmethod
  def from_args(cls, args: argparse.Namespace) -> 'Config':
    """
    Create configuration from command line arguments.
    
    Args:
      args (argparse.Namespace): Command line arguments
      
    Returns:
      Config: Configuration object
    """
    # Initialize with config file if provided
    config_file = getattr(args, 'config', None)
    config = cls(config_file)
    
    # Update with command line arguments
    if hasattr(args, 'model') and args.model:
      config.set('openai.models.completion', args.model)
    
    if hasattr(args, 'chunk_length') and args.chunk_length:
      config.set('transcription.chunk_length_ms', args.chunk_length)
    
    if hasattr(args, 'max_chunk_size') and args.max_chunk_size:
      config.set('processing.max_chunk_size', args.max_chunk_size)
    
    if hasattr(args, 'temperature') and args.temperature is not None:
      config.set('processing.temperature', args.temperature)
    
    if hasattr(args, 'no_post_processing') and args.no_post_processing:
      config.set('processing.post_processing', False)
    
    if hasattr(args, 'parallel') and args.parallel is not None:
      config.set('transcription.parallel', args.parallel)
    
    if hasattr(args, 'workers') and args.workers is not None:
      config.set('transcription.max_workers', args.workers)
    
    # Set log level based on verbose/debug flags
    if hasattr(args, 'debug') and args.debug:
      config.set('system.log_level', 'DEBUG')
    elif hasattr(args, 'verbose') and args.verbose:
      config.set('system.log_level', 'INFO')
    
    return config

# Create global configuration instance
config = Config()

def get_config() -> Config:
  """Get global configuration instance."""
  return config

def init_config(config_file: str | None = None) -> Config:
  """
  Initialize global configuration.
  
  Args:
    config_file (str, optional): Path to configuration file. Defaults to None.
    
  Returns:
    Config: Global configuration instance
  """
  global config
  config = Config(config_file)
  return config

#fin