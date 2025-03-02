#!/usr/bin/env python3
"""
Tests for the configuration module.
"""
import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from transcribe_pkg.utils.config import Config

class TestConfig(unittest.TestCase):
    """Tests for Config class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, 'test_config.json')
        
        # Test configuration
        self.test_config = {
            'openai': {
                'models': {
                    'completion': 'test-model',
                    'summary': 'test-summary-model'
                }
            },
            'transcription': {
                'chunk_length_ms': 1000,
                'parallel': True
            }
        }
        
        # Create a test config file
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_load_from_file(self):
        """Test loading configuration from file."""
        config = Config(self.config_path)
        
        # Check loaded values
        self.assertEqual(config.get('openai.models.completion'), 'test-model')
        self.assertEqual(config.get('transcription.chunk_length_ms'), 1000)
        self.assertTrue(config.get('transcription.parallel'))
    
    def test_get_default_value(self):
        """Test getting default value for missing key."""
        config = Config()
        
        # Non-existent key should return default
        default_value = 'default'
        self.assertEqual(config.get('non.existent.key', default_value), default_value)
    
    def test_set_value(self):
        """Test setting configuration value."""
        config = Config()
        
        # Set a value
        test_value = 'new-value'
        config.set('test.key', test_value)
        
        # Check the value was set
        self.assertEqual(config.get('test.key'), test_value)
    
    def test_update_nested_dict(self):
        """Test updating nested dictionary."""
        config = Config()
        
        # Initial structure
        config.set('level1.level2.key1', 'value1')
        
        # Update with deeper nesting
        updates = {
            'level1': {
                'level2': {
                    'key1': 'updated',
                    'key2': 'new-value'
                }
            }
        }
        config.update(updates)
        
        # Check values
        self.assertEqual(config.get('level1.level2.key1'), 'updated')
        self.assertEqual(config.get('level1.level2.key2'), 'new-value')
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-api-key',
        'OPENAI_COMPLETION_MODEL': 'env-model',
        'TRANSCRIBE_PARALLEL': 'false'
    })
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        config = Config()
        
        # Check values loaded from environment
        self.assertEqual(config.get('openai.api_key'), 'test-api-key')
        self.assertEqual(config.get('openai.models.completion'), 'env-model')
        self.assertFalse(config.get('transcription.parallel'))
    
    def test_save_to_file(self):
        """Test saving configuration to file."""
        config = Config()
        
        # Set some values (use a simple string value, not a mock)
        config.set('test.key', 'test-value')
        
        # Save to a new file
        save_path = os.path.join(self.temp_dir.name, 'saved_config.json')
        
        try:
            config.save_to_file(save_path)
            
            # Load the saved file and check values
            with open(save_path, 'r') as f:
                saved_config = json.load(f)
            
            self.assertEqual(saved_config['test']['key'], 'test-value')
        except TypeError as e:
            if "not JSON serializable" in str(e):
                self.fail(f"Config contains non-serializable objects: {e}")
            else:
                raise
    
    def test_from_args(self):
        """Test creating configuration from command line arguments."""
        # Mock args
        args = MagicMock()
        args.model = 'args-model'
        args.chunk_length = 2000
        args.verbose = True
        args.debug = False
        args.config = None
        
        # Create config from args
        config = Config.from_args(args)
        
        # Check values
        self.assertEqual(config.get('openai.models.completion'), 'args-model')
        self.assertEqual(config.get('transcription.chunk_length_ms'), 2000)
        self.assertEqual(config.get('system.log_level'), 'INFO')  # From verbose=True
    
    def test_as_dict(self):
        """Test getting configuration as dictionary."""
        config = Config(self.config_path)
        
        # Get as dict
        config_dict = config.as_dict
        
        # Check values
        self.assertEqual(config_dict['openai']['models']['completion'], 'test-model')
        self.assertEqual(config_dict['transcription']['chunk_length_ms'], 1000)
        
        # Store the original value
        original_value = config.get('openai.models.completion')
        
        # Modify the copy
        config_dict['openai']['models']['completion'] = 'modified'
        
        # Ensure original is unchanged (use the stored value for comparison)
        self.assertEqual(config.get('openai.models.completion'), original_value)

if __name__ == '__main__':
    unittest.main()

#fin