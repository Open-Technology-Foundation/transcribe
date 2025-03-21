#!/usr/bin/env python3
"""
Tests for text utilities.
"""
import unittest
from transcribe_pkg.utils.text_utils import (
    create_sentences,
    create_paragraphs,
    clean_transcript_text,
    extract_key_topics,
    split_text_for_processing
)

class TestTextUtils(unittest.TestCase):
    """Test text utility functions."""
    
    def test_create_sentences(self):
        """Test creating sentences from text."""
        # Test basic sentence creation
        text = "This is a test. This is another test. This is a third test."
        sentences = create_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "This is a test.")
        self.assertEqual(sentences[1], "This is another test.")
        self.assertEqual(sentences[2], "This is a third test.")
        
        # Test handling of newlines
        text = "This is a test.\nThis is another test.\r\nThis is a third test."
        sentences = create_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "This is a test.")
        
        # Test with max sentence length
        text = "This is a very long sentence that will be split into multiple parts."
        sentences = create_sentences(text, max_sentence_length=15)
        self.assertTrue(len(sentences) > 1)
    
    def test_create_paragraphs(self):
        """Test creating paragraphs from text."""
        # Test basic paragraph creation
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        
        # Test with min_sentences=2, max_sentences=3
        paragraphs = create_paragraphs(text, min_sentences=2, max_sentences=3)
        para_list = paragraphs.split('\n\n')
        self.assertEqual(len(para_list), 2)
        
        # Test with min_sentences=1, max_sentences=2
        paragraphs = create_paragraphs(text, min_sentences=1, max_sentences=2)
        para_list = paragraphs.split('\n\n')
        self.assertEqual(len(para_list), 3)
        
        # Test with empty text
        paragraphs = create_paragraphs("")
        self.assertEqual(paragraphs, "")
    
    def test_clean_transcript_text(self):
        """Test basic text cleaning."""
        # Test hesitation removal
        text = "I, um, really think, uh, this is important."
        cleaned = clean_transcript_text(text)
        self.assertNotIn("um", cleaned)
        self.assertNotIn("uh", cleaned)
        
        # Test repeated word removal
        text = "I I I think this this is important."
        cleaned = clean_transcript_text(text)
        self.assertEqual(cleaned, "I think this is important.")
        
        # Test spacing fixes
        text = "Hello.World"
        cleaned = clean_transcript_text(text)
        self.assertEqual(cleaned, "Hello. World")
    
    def test_extract_key_topics(self):
        """Test extracting key topics from text."""
        text = "This is about artificial intelligence. AI is transforming our world. " \
               "Artificial intelligence can help with many tasks. AI is a technology."
        topics = extract_key_topics(text)
        self.assertTrue("artificial" in topics or "intelligence" in topics)
        
        # Test with empty text
        topics = extract_key_topics("")
        self.assertEqual(topics, [])
    
    def test_split_text_for_processing(self):
        """Test splitting text into chunks."""
        # Create a long text
        text = "This is a test. " * 100
        
        # Test basic chunking
        chunks = split_text_for_processing(text, max_chunk_size=200)
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertTrue(len(chunk.encode('utf-8')) <= 200)
        
        # Test with empty text
        chunks = split_text_for_processing("")
        self.assertEqual(chunks, [])

if __name__ == '__main__':
    unittest.main()

#fin