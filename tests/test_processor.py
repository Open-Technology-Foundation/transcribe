#!/usr/bin/env python3
"""
Tests for transcript processing functionality.
"""
import unittest
from unittest.mock import patch, MagicMock

from transcribe_pkg.core.processor import TranscriptProcessor
from transcribe_pkg.utils.text_utils import create_sentences, create_paragraphs

class TestTranscriptProcessor(unittest.TestCase):
    """Tests for the TranscriptProcessor class."""
    
    def test_add_oxford_comma(self):
        """Test adding Oxford comma to context string."""
        processor = TranscriptProcessor()
        
        # Test various cases
        self.assertEqual(processor._add_oxford_comma(""), "")
        self.assertEqual(processor._add_oxford_comma("science"), "science")
        self.assertEqual(processor._add_oxford_comma("science, math"), "science and math")
        self.assertEqual(processor._add_oxford_comma("science, math, philosophy"), 
                         "science, math, and philosophy")
        self.assertEqual(processor._add_oxford_comma("science, math, philosophy, history"), 
                         "science, math, philosophy, and history")
    
    def test_get_chunk_with_complete_sentences(self):
        """Test extracting a chunk with complete sentences."""
        processor = TranscriptProcessor()
        
        # Test with simple text
        text = "This is a test. This is another test. This is a third test."
        chunk, remaining = processor._get_chunk_with_complete_sentences(text, 25)
        self.assertEqual(chunk, "This is a test.")
        self.assertEqual(remaining, " This is another test. This is a third test.")
        
        # Test with empty text
        chunk, remaining = processor._get_chunk_with_complete_sentences("", 1000)
        self.assertEqual(chunk, "")
        self.assertEqual(remaining, "")
        
        # Test with large max_chunk_size
        chunk, remaining = processor._get_chunk_with_complete_sentences(text, 1000)
        self.assertEqual(chunk, text)
        self.assertEqual(remaining, "")
    
    @patch('transcribe_pkg.core.processor.call_llm')
    def test_process_chunk(self, mock_call_llm):
        """Test processing a chunk of text."""
        # Setup mock
        mock_call_llm.return_value = "Processed text"

        # Create processor
        processor = TranscriptProcessor()

        # Test processing
        result = processor._process_chunk(
            "This is a test chunk.",
            "science",
            "Previous context summary",
            "en"
        )

        # Verify result
        self.assertEqual(result, "Processed text")
        mock_call_llm.assert_called_once()

        # Check that call_llm was called with correct arguments
        call_args = mock_call_llm.call_args
        self.assertEqual(call_args[1]["user_prompt"], "This is a test chunk.")
        self.assertEqual(call_args[1]["model"], "gpt-4o")
        self.assertEqual(call_args[1]["temperature"], 0.05)

        # System prompt should contain context and language info
        system_prompt = call_args[1]["system_prompt"]
        self.assertIn("with extensive knowledge in science", system_prompt)
        self.assertIn("Context Summary", system_prompt)
        self.assertIn("Previous context summary", system_prompt)
    
    @patch('transcribe_pkg.core.processor.call_llm')
    def test_process_chunk_with_error(self, mock_call_llm):
        """Test handling API errors in chunk processing."""
        # Setup mock to raise exception
        from transcribe_pkg.utils.api_utils import APIError
        mock_call_llm.side_effect = APIError("API error")

        # Create processor
        processor = TranscriptProcessor()

        # Test processing with error
        result = processor._process_chunk("This is a test chunk.", "", None, "en")

        # On error, should return original chunk
        self.assertEqual(result, "This is a test chunk.")

class TestTextUtils(unittest.TestCase):
    """Tests for text utilities."""
    
    def test_create_sentences(self):
        """Test creating sentences from text."""
        text = "This is a test. This is another test. This is a third test."
        sentences = create_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertIn("This is a test.", sentences)
        self.assertIn("This is another test.", sentences)
        self.assertIn("This is a third test.", sentences)
        
        # Test with max length
        sentences = create_sentences(text, max_sentence_length=10)
        self.assertTrue(len(sentences) > 3)  # Should split sentences
    
    def test_create_paragraphs(self):
        """Test creating paragraphs from text."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        
        # Test with min_sentences=2, max_sentences=3
        paragraphs = create_paragraphs(text, min_sentences=2, max_sentences=3)
        para_list = paragraphs.split('\n\n')
        self.assertEqual(len(para_list), 2)  # Should create 2 paragraphs
        
        # Test with min_sentences=1, max_sentences=2
        paragraphs = create_paragraphs(text, min_sentences=1, max_sentences=2)
        para_list = paragraphs.split('\n\n')
        self.assertEqual(len(para_list), 3)  # Should create 3 paragraphs

if __name__ == '__main__':
    unittest.main()

#fin"""