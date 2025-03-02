#!/usr/bin/env python3
"""
Tests for the API utilities module.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from transcribe_pkg.utils.api_utils import (
    get_openai_client,
    call_llm,
    transcribe_audio,
    APIError,
    EmptyResponseError,
    AudioTranscriptionError
)

class TestAPIUtils(unittest.TestCase):
    """Tests for API utility functions."""

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
    @patch('transcribe_pkg.utils.api_utils.OpenAI')
    def test_get_openai_client(self, mock_openai):
        """Test getting OpenAI client with API key."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Call function
        client = get_openai_client()
        
        # Check results
        mock_openai.assert_called_once_with(api_key='test-api-key')
        self.assertEqual(client, mock_client)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': ''})  # Empty string instead of completely missing
    @patch('transcribe_pkg.utils.api_utils.sys.exit')
    @patch('transcribe_pkg.utils.api_utils.logging')
    def test_get_openai_client_no_key(self, mock_logging, mock_exit):
        """Test getting OpenAI client without API key."""
        # Call function, should exit
        get_openai_client()
        
        # Check logging and exit called
        mock_logging.error.assert_called()
        mock_exit.assert_called_once_with(1)
    
    @patch('transcribe_pkg.utils.api_utils.openai_client')
    def test_call_llm_success(self, mock_client):
        """Test successful LLM API call."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Call function
        result = call_llm("System prompt", "User input", model="test-model")
        
        # Check results
        self.assertEqual(result, "Test response")
        mock_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs['model'], "test-model")
        self.assertEqual(len(kwargs['messages']), 2)
    
    @patch('transcribe_pkg.utils.api_utils.openai_client')
    def test_call_llm_empty_response(self, mock_client):
        """Test LLM API call with empty response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response
        
        # Call function
        result = call_llm("System prompt", "User input")
        
        # Check results
        self.assertEqual(result, "")
    
    @patch('transcribe_pkg.utils.api_utils.openai_client')
    @patch('transcribe_pkg.utils.api_utils.logging')
    def test_call_llm_no_choices(self, mock_logging, mock_client):
        """Test LLM API call with no choices."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response
        
        try:
            # Call the implementation function directly to avoid retry
            from transcribe_pkg.utils.api_utils import _call_llm_impl
            _call_llm_impl("System prompt", "User input")
            self.fail("EmptyResponseError not raised")
        except EmptyResponseError:
            # Success - error was raised as expected
            pass
        
        # Check logging
        mock_logging.warning.assert_called()
        mock_logging.error.assert_called()
    
    @patch('transcribe_pkg.utils.api_utils.openai_client')
    @patch('transcribe_pkg.utils.api_utils.logging')
    def test_call_llm_api_error(self, mock_logging, mock_client):
        """Test LLM API call with API error."""
        # Setup mock to raise error
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        try:
            # Call the implementation function directly to avoid retry
            from transcribe_pkg.utils.api_utils import _call_llm_impl
            _call_llm_impl("System prompt", "User input")
            self.fail("APIError not raised")
        except APIError:
            # Success - error was raised as expected
            pass
        
        # Check logging
        mock_logging.error.assert_called()
    
    @patch('os.path.exists')
    @patch('transcribe_pkg.utils.api_utils.openai_client')
    def test_transcribe_audio_success(self, mock_client, mock_exists):
        """Test successful audio transcription."""
        # Setup mocks
        mock_exists.return_value = True
        mock_transcription = MagicMock()
        mock_transcription.text = "Test transcription"
        mock_client.audio.transcriptions.create.return_value = mock_transcription
        
        # Mock file open
        mock_file = MagicMock()
        mock_file.__enter__.return_value = "audio_data"
        
        with patch('builtins.open', return_value=mock_file):
            # Call function
            result = transcribe_audio("test.mp3", prompt="Test prompt")
        
        # Check results
        self.assertEqual(result, "Test transcription")
        mock_client.audio.transcriptions.create.assert_called_once()
    
    @patch('os.path.exists')
    def test_transcribe_audio_file_not_found(self, mock_exists):
        """Test audio transcription with file not found."""
        # Setup mock
        mock_exists.return_value = False
        
        try:
            # Call the implementation function directly to avoid retry
            from transcribe_pkg.utils.api_utils import _transcribe_audio_impl
            _transcribe_audio_impl("nonexistent.mp3")
            self.fail("FileNotFoundError not raised")
        except FileNotFoundError:
            # Success - error was raised as expected
            pass
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('transcribe_pkg.utils.api_utils.logging')
    def test_transcribe_audio_file_too_large(self, mock_logging, mock_getsize, mock_exists):
        """Test audio transcription with file size warning."""
        # Setup mocks
        mock_exists.return_value = True
        mock_getsize.return_value = 30 * 1024 * 1024  # 30MB
        
        # Need to mock the actual transcription to avoid real API call
        mock_transcription = MagicMock()
        mock_transcription.text = "Test transcription"
        
        with patch('transcribe_pkg.utils.api_utils.openai_client') as mock_client:
            mock_client.audio.transcriptions.create.return_value = mock_transcription
            
            # Mock file open
            with patch('builtins.open', MagicMock()):
                # Call function
                transcribe_audio("large.mp3")
        
        # Check logging
        mock_logging.warning.assert_called()

if __name__ == '__main__':
    unittest.main()

#fin