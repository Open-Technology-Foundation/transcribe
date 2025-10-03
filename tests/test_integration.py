#!/usr/bin/env python3
"""
Integration tests for the transcribe package.
"""
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from transcribe_pkg.core.transcriber import transcribe_audio_file
from transcribe_pkg.core.processor import process_transcript

class TestIntegration(unittest.TestCase):
    """Integration tests for the transcribe package."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, 'output.txt')
        
        # Sample text for mocking transcription
        self.sample_transcript = (
            "This is a test transcript. It has multiple sentences. "
            "Some of them are short. Others are a bit longer and contain more words. "
            "We need enough sentences to test paragraph creation."
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch('transcribe_pkg.core.transcriber.Transcriber')
    @patch('transcribe_pkg.core.processor.TranscriptProcessor')
    @patch('os.path.exists')
    def test_transcribe_audio_file_end_to_end(self, mock_exists, mock_processor_class, mock_transcriber_class):
        """Test end-to-end audio file transcription process."""
        # Setup mocks
        mock_exists.return_value = True

        # Mock transcriber instance and methods
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = self.sample_transcript
        mock_transcriber_class.return_value = mock_transcriber

        # Mock processor instance and methods
        mock_processor = MagicMock()
        mock_processor.process.return_value = "Processed transcript"
        mock_processor_class.return_value = mock_processor

        # Call function
        result = transcribe_audio_file(
            audio_path="test.mp3",
            output_file=self.output_path,
            context="test",
            model="test-model",
            parallel_processing=False,
            max_workers=1
        )

        # Check results
        self.assertEqual(result, "Processed transcript")

        # Verify transcriber was created and called
        mock_transcriber_class.assert_called_once()
        mock_transcriber.transcribe.assert_called_once()

        # Verify processor was created and called
        mock_processor_class.assert_called_once()
        mock_processor.process.assert_called_once()

        # Check output file
        self.assertTrue(os.path.exists(self.output_path))
        with open(self.output_path, 'r') as f:
            content = f.read()
            self.assertEqual(content, "Processed transcript")
    
    @patch('transcribe_pkg.core.processor.call_llm')
    @patch('transcribe_pkg.core.analyzer.call_llm') 
    @patch('transcribe_pkg.utils.prompts.call_llm')
    def test_process_transcript_integration(self, mock_prompts_call_llm, mock_analyzer_call_llm, mock_processor_call_llm):
        """Test transcript processing integration."""
        # Setup mocks to return appropriate responses
        mock_processor_call_llm.side_effect = ["First chunk processed.", "Second chunk processed."]
        mock_analyzer_call_llm.return_value = "general"  # content type
        mock_prompts_call_llm.return_value = "test,context"  # context extraction
        
        # Input text
        input_text = (
            "This is the first chunk of text. It has several sentences. "
            "This should be processed first.\n\n"
            "This is the second chunk of text. It also has several sentences. "
            "This should be processed second."
        )
        
        # Call function
        result = process_transcript(
            input_text,
            model="test-model",
            max_chunk_size=100,  # Small to force multiple chunks
            context="test,context",
            language="en"
        )
        
        # Check results
        self.assertIn("First chunk processed.", result)
        self.assertIn("Second chunk processed.", result)
        
        # Verify mock called at least once for processing
        self.assertGreater(mock_processor_call_llm.call_count, 0)
        
        # Check that call_llm was called with proper arguments
        call_args_list = mock_processor_call_llm.call_args_list
        self.assertGreater(len(call_args_list), 0)
        
        # Check first call has expected structure
        args, kwargs = call_args_list[0]
        # Function called with keyword arguments
        self.assertIn('user_prompt', kwargs)
        self.assertIn('system_prompt', kwargs)
        # Check that model parameter is passed in kwargs
        self.assertEqual(kwargs.get('model', ''), "test-model")

if __name__ == '__main__':
    unittest.main()

#fin