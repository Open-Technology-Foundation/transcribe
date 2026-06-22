#!/usr/bin/env python3
"""
Tests for transcript processing functionality.
"""
import unittest
from unittest.mock import patch

from transcribe_pkg.core.processor import TranscriptProcessor
from transcribe_pkg.utils.text_utils import create_sentences, create_paragraphs
from transcribe_pkg.constants import DEFAULT_LLM_MODEL, DEFAULT_SUMMARY_MODEL


class TestTranscriptProcessor(unittest.TestCase):
  """Tests for the TranscriptProcessor class."""

  def test_default_model_is_claude(self):
    """Test that default model is a Claude model."""
    processor = TranscriptProcessor()
    self.assertEqual(processor.model, DEFAULT_LLM_MODEL)
    self.assertTrue(processor.model.startswith("claude-"))

  def test_init_with_custom_model(self):
    """Test initialization with custom model."""
    processor = TranscriptProcessor(model="claude-haiku-4-5")
    self.assertEqual(processor.model, "claude-haiku-4-5")

  def test_init_with_provider(self):
    """Test initialization with provider override."""
    processor = TranscriptProcessor(provider="anthropic")
    self.assertEqual(processor.provider, "anthropic")

  def test_add_oxford_comma(self):
    """Test adding Oxford comma to context string."""
    processor = TranscriptProcessor()

    # Test various cases
    self.assertEqual(processor._add_oxford_comma(""), "")
    self.assertEqual(processor._add_oxford_comma("science"), "science")
    self.assertEqual(processor._add_oxford_comma("science, math"), "science and math")
    self.assertEqual(
      processor._add_oxford_comma("science, math, philosophy"), "science, math, and philosophy"
    )
    self.assertEqual(
      processor._add_oxford_comma("science, math, philosophy, history"),
      "science, math, philosophy, and history",
    )

  def test_get_chunk_with_complete_sentences(self):
    """Test extracting a chunk with complete sentences."""
    processor = TranscriptProcessor()

    # Test with simple text
    text = "This is a test. This is another test. This is a third test."
    chunk, remaining = processor._get_chunk_with_complete_sentences(text, 25)
    self.assertEqual(chunk, "This is a test.")
    # Remainder is the space-joined unconsumed sentences (no stray leading space)
    self.assertEqual(remaining, "This is another test. This is a third test.")

    # Test with empty text
    chunk, remaining = processor._get_chunk_with_complete_sentences("", 1000)
    self.assertEqual(chunk, "")
    self.assertEqual(remaining, "")

    # Test with large max_chunk_size
    chunk, remaining = processor._get_chunk_with_complete_sentences(text, 1000)
    self.assertEqual(chunk, text)
    self.assertEqual(remaining, "")

  def test_get_chunk_handles_long_text(self):
    """Test chunk extraction with text longer than max size."""
    processor = TranscriptProcessor()

    # Text longer than max_chunk_size
    text = "First sentence here. Second sentence here. Third sentence here."
    chunk, remaining = processor._get_chunk_with_complete_sentences(text, 50)

    # Should get complete sentences within size limit
    self.assertGreater(len(chunk), 0)
    self.assertTrue(chunk.endswith(".") or chunk.endswith("?") or chunk.endswith("!"))

  def test_get_chunk_handles_newlines_without_corruption(self):
    """A chunk spanning a newline must not corrupt the remainder.

    create_sentences() normalizes newlines to spaces, so locating the chunk via
    text.find() returned -1 when a chunk spanned a newline; the remainder then
    became a garbage tail and the sequential loop spun on it.
    """
    processor = TranscriptProcessor()
    text = "First sentence here.\nSecond sentence here. Third one."
    # max_chunk_size 45 -> the chunk holds the first two sentences (which the
    # original separates with a newline); the remainder is the third sentence.
    chunk, remaining = processor._get_chunk_with_complete_sentences(text, 45)
    self.assertEqual(chunk, "First sentence here. Second sentence here.")
    self.assertEqual(remaining, "Third one.")

  @patch("transcribe_pkg.core.processor.call_llm")
  def test_parallel_processing_does_not_duplicate_sentences(self, mock_call_llm):
    """Parallel post-processing must not duplicate overlapping sentences.

    Chunks were split with a 500-byte overlap (to attempt context preservation),
    but the reassembler joined them verbatim, so every overlapping sentence
    appeared twice in the output.
    """
    # Echo each chunk unchanged so the output is exactly the reassembled chunks.
    mock_call_llm.side_effect = lambda **kwargs: kwargs["user_prompt"]
    processor = TranscriptProcessor(
      max_chunk_size=200, content_aware=False, cache_enabled=False
    )
    sentences = [f"Sentence number {i} is unique and distinct." for i in range(15)]
    text = " ".join(sentences)
    result = processor.process(
      text, use_parallel=True, content_analysis=False, language="en"
    )
    for i in range(15):
      marker = f"Sentence number {i} is unique"
      count = result.count(marker)
      self.assertEqual(count, 1, f"sentence {i} appears {count}x (overlap duplication)")

  @patch("transcribe_pkg.core.processor.call_llm")
  def test_process_chunk(self, mock_call_llm):
    """Test processing a chunk of text."""
    # Setup mock
    mock_call_llm.return_value = "Processed text"

    # Create processor
    processor = TranscriptProcessor()

    # Test processing
    result = processor._process_chunk(
      "This is a test chunk.", "science", "Previous context summary", "en"
    )

    # Verify result
    self.assertEqual(result, "Processed text")
    mock_call_llm.assert_called_once()

    # Check that call_llm was called with correct arguments
    call_args = mock_call_llm.call_args
    self.assertEqual(call_args[1]["user_prompt"], "This is a test chunk.")
    self.assertEqual(call_args[1]["model"], DEFAULT_LLM_MODEL)
    self.assertEqual(call_args[1]["temperature"], 0.05)

    # System prompt should contain context and language info
    system_prompt = call_args[1]["system_prompt"]
    self.assertIn("with extensive knowledge in science", system_prompt)
    self.assertIn("Context Summary", system_prompt)
    self.assertIn("Previous context summary", system_prompt)

  @patch("transcribe_pkg.core.processor.call_llm")
  def test_process_chunk_passes_provider(self, mock_call_llm):
    """Test that process_chunk passes provider to call_llm."""
    mock_call_llm.return_value = "Processed"

    processor = TranscriptProcessor(provider="anthropic")
    processor._process_chunk("Test chunk.", "", None, "en")

    call_args = mock_call_llm.call_args
    self.assertEqual(call_args[1].get("provider"), "anthropic")

  @patch("transcribe_pkg.core.processor.call_llm")
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

  @patch("transcribe_pkg.core.processor.call_llm")
  def test_process_chunk_with_non_english(self, mock_call_llm):
    """Test processing chunk with non-English language."""
    mock_call_llm.return_value = "Translated and processed"

    processor = TranscriptProcessor()
    result = processor._process_chunk("Texto en español.", "general", None, "es")

    # Verify language task is in system prompt
    call_args = mock_call_llm.call_args
    system_prompt = call_args[1]["system_prompt"]
    self.assertIn("ES", system_prompt)


class TestProcessTranscript(unittest.TestCase):
  """Tests for the process_transcript function."""

  def test_default_model_constant_is_claude(self):
    """Test that DEFAULT_LLM_MODEL is a Claude model."""
    self.assertTrue(DEFAULT_LLM_MODEL.startswith("claude-"))
    self.assertIn("sonnet", DEFAULT_LLM_MODEL.lower())

  def test_default_summary_model_is_claude(self):
    """Test that DEFAULT_SUMMARY_MODEL is a Claude model."""
    self.assertTrue(DEFAULT_SUMMARY_MODEL.startswith("claude-"))
    self.assertIn("haiku", DEFAULT_SUMMARY_MODEL.lower())


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

  def test_create_sentences_preserves_abbreviations(self):
    """Test that common abbreviations don't cause false splits."""
    text = "Dr. Smith went to the U.S. for a conference. He met Prof. Jones."
    sentences = create_sentences(text)
    # Should handle abbreviations reasonably
    self.assertGreater(len(sentences), 0)

  def test_create_sentences_handles_questions(self):
    """Test sentence creation handles question marks."""
    text = "What is this? This is a test. Why is it here?"
    sentences = create_sentences(text)
    self.assertEqual(len(sentences), 3)

  def test_create_sentences_handles_exclamations(self):
    """Test sentence creation handles exclamation marks."""
    text = "Wow! This is amazing! I love it."
    sentences = create_sentences(text)
    self.assertEqual(len(sentences), 3)

  def test_create_paragraphs(self):
    """Test creating paragraphs from text."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."

    # Test with min_sentences=2, max_sentences=3
    paragraphs = create_paragraphs(text, min_sentences=2, max_sentences=3)
    para_list = paragraphs.split("\n\n")
    self.assertEqual(len(para_list), 2)  # Should create 2 paragraphs

    # Test with min_sentences=1, max_sentences=2
    paragraphs = create_paragraphs(text, min_sentences=1, max_sentences=2)
    para_list = paragraphs.split("\n\n")
    self.assertEqual(len(para_list), 3)  # Should create 3 paragraphs

  def test_create_paragraphs_empty_text(self):
    """Test paragraph creation with empty text."""
    paragraphs = create_paragraphs("")
    self.assertEqual(paragraphs, "")

  def test_create_paragraphs_single_sentence(self):
    """Test paragraph creation with single sentence."""
    text = "Just one sentence here."
    paragraphs = create_paragraphs(text)
    self.assertEqual(paragraphs.strip(), text)


class TestTranscriptProcessorIntegration(unittest.TestCase):
  """Integration tests for TranscriptProcessor."""

  @patch("transcribe_pkg.core.processor.call_llm")
  @patch("transcribe_pkg.core.analyzer.call_llm")
  @patch("transcribe_pkg.utils.prompts.call_llm")
  def test_full_processing_pipeline(
    self, mock_prompts_llm, mock_analyzer_llm, mock_processor_llm
  ):
    """Test the full processing pipeline."""
    # Setup mocks
    mock_processor_llm.side_effect = ["First processed.", "Second processed."]
    mock_analyzer_llm.return_value = "general"
    mock_prompts_llm.return_value = "technology"

    input_text = """
    This is the first part of the transcript. It contains some content.

    This is the second part. It also has content that needs processing.
    """

    processor = TranscriptProcessor(max_chunk_size=100)
    result = processor.process(input_text, context="technology")

    # Should have processed text
    self.assertIn("First processed", result)
    self.assertIn("Second processed", result)

  @patch("transcribe_pkg.core.processor.call_llm")
  def test_processor_calls_llm(self, mock_call_llm):
    """Test that processor calls LLM for processing."""
    mock_call_llm.return_value = "Processed content."

    processor = TranscriptProcessor(max_chunk_size=1000)

    # Process text
    processor._process_chunk("Simple text.", "general", None, "en")

    # Verify LLM was called
    mock_call_llm.assert_called_once()
    call_args = mock_call_llm.call_args
    self.assertEqual(call_args[1]["user_prompt"], "Simple text.")
    self.assertEqual(call_args[1]["model"], DEFAULT_LLM_MODEL)


class TestCacheKeyStability(unittest.TestCase):
  """Cache keys must be deterministic across processes and include every output-affecting param."""

  def test_stable_hash_is_deterministic_across_processes(self):
    """_stable_hash must return the same value in separate interpreters (no builtin hash())."""
    import subprocess
    import sys
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    code = (
      "from transcribe_pkg.core.processor import _stable_hash; "
      "print('KEY=' + _stable_hash('sample text', 'claude-haiku-4-5'))"
    )
    keys = []
    for _ in range(2):
      r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(root))
      self.assertEqual(r.returncode, 0, msg=r.stderr)
      line = next((ln for ln in r.stdout.splitlines() if ln.startswith("KEY=")), None)
      self.assertIsNotNone(line, msg=f"no KEY= line in stdout: {r.stdout!r}")
      keys.append(line[4:])
    self.assertEqual(keys[0], keys[1])  # stable across separate processes
    self.assertEqual(len(keys[0]), 64)  # sha256 hexdigest, not a builtin-hash int

  def test_chunk_cache_key_includes_all_output_affecting_params(self):
    """Changing provider, max_tokens, or prompt identity must change the chunk cache key."""
    from transcribe_pkg.core.processor import _chunk_cache_key
    base = _chunk_cache_key("chunk", "ctx", "en", "claude-sonnet-4-5", 0.1, 4096, "anthropic", "standard")
    # Identical inputs -> identical key
    self.assertEqual(
      base,
      _chunk_cache_key("chunk", "ctx", "en", "claude-sonnet-4-5", 0.1, 4096, "anthropic", "standard"),
    )
    # Each output-affecting param must change the key
    self.assertNotEqual(
      base,
      _chunk_cache_key("chunk", "ctx", "en", "claude-sonnet-4-5", 0.1, 4096, "openai", "standard"),
    )
    self.assertNotEqual(
      base,
      _chunk_cache_key("chunk", "ctx", "en", "claude-sonnet-4-5", 0.1, 2048, "anthropic", "standard"),
    )
    self.assertNotEqual(
      base,
      _chunk_cache_key("chunk", "ctx", "en", "claude-sonnet-4-5", 0.1, 4096, "anthropic", "specialized-xyz"),
    )

  def test_language_cache_key_depends_on_model(self):
    """Language-detection key must depend on the detection model."""
    from transcribe_pkg.core.processor import _language_cache_key
    self.assertNotEqual(
      _language_cache_key("hello world", "claude-haiku-4-5"),
      _language_cache_key("hello world", "gpt-4o"),
    )


if __name__ == "__main__":
  unittest.main()

#fin
