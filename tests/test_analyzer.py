#!/usr/bin/env python3
"""
Tests for content analysis functionality.
"""
import unittest
from unittest.mock import patch, MagicMock

from transcribe_pkg.core.analyzer import ContentAnalyzer, SpecializedProcessor
from transcribe_pkg.constants import DEFAULT_LLM_MODEL, DEFAULT_SUMMARY_MODEL


class TestContentAnalyzer(unittest.TestCase):
  """Tests for the ContentAnalyzer class."""

  def setUp(self):
    """Set up test fixtures."""
    self.analyzer = ContentAnalyzer()

  def test_init_defaults_to_claude_model(self):
    """Test that ContentAnalyzer defaults to Claude summary model."""
    analyzer = ContentAnalyzer()
    self.assertEqual(analyzer.model, DEFAULT_SUMMARY_MODEL)
    self.assertTrue(analyzer.model.startswith("claude-"))

  def test_init_with_custom_model(self):
    """Test ContentAnalyzer with custom model."""
    analyzer = ContentAnalyzer(model="claude-haiku-4-5")
    self.assertEqual(analyzer.model, "claude-haiku-4-5")

  def test_init_with_provider_override(self):
    """Test ContentAnalyzer with provider override."""
    analyzer = ContentAnalyzer(provider="anthropic")
    self.assertEqual(analyzer.provider, "anthropic")

  @patch("transcribe_pkg.core.analyzer.call_llm")
  def test_detect_content_type_dialogue(self, mock_call_llm):
    """Test content type detection for dialogue."""
    mock_call_llm.return_value = "dialogue"

    sample_text = """
    John: Hello, how are you today?
    Mary: I'm doing well, thanks for asking.
    John: Did you see the news this morning?
    Mary: Yes, it was quite surprising.
    """

    result = self.analyzer._detect_content_type(sample_text)
    self.assertEqual(result, "dialogue")

  @patch("transcribe_pkg.core.analyzer.call_llm")
  def test_detect_content_type_technical(self, mock_call_llm):
    """Test content type detection for technical content."""
    mock_call_llm.return_value = "technical"

    sample_text = """
    The algorithm complexity is O(n log n) in the average case.
    The implementation uses a hash table for constant-time lookups.
    Figure 1 shows the architecture of the system.
    """

    result = self.analyzer._detect_content_type(sample_text)
    self.assertEqual(result, "technical")

  @patch("transcribe_pkg.core.analyzer.call_llm")
  def test_detect_content_type_general(self, mock_call_llm):
    """Test content type detection for general content."""
    mock_call_llm.return_value = "general"

    sample_text = """
    This is a general text about various topics.
    It discusses different subjects in a casual manner.
    Nothing particularly technical or dialogue-based.
    """

    result = self.analyzer._detect_content_type(sample_text)
    self.assertEqual(result, "general")

  @patch("transcribe_pkg.core.analyzer.call_llm")
  def test_detect_content_type_fallback_on_error(self, mock_call_llm):
    """Test content type detection falls back to pattern matching on error."""
    mock_call_llm.side_effect = Exception("API Error")

    # Text with dialogue markers
    sample_text = 'He said "hello" to the audience. She replied "thank you".'

    result = self.analyzer._detect_content_type(sample_text)
    # Should fall back to basic pattern detection
    self.assertIn(result, ["dialogue", "technical", "speech", "lecture", "general"])

  @patch("transcribe_pkg.core.analyzer.call_llm")
  def test_detect_content_type_invalid_response(self, mock_call_llm):
    """Test content type detection handles invalid API response."""
    mock_call_llm.return_value = "invalid_type_xyz"

    sample_text = "Some text content here."

    result = self.analyzer._detect_content_type(sample_text)
    # Should fall back to pattern detection
    self.assertIn(result, ["dialogue", "technical", "speech", "lecture", "general"])

  def test_measure_technical_level(self):
    """Test technical level measurement."""
    # High technical content
    technical_text = """
    The algorithm implementation uses a recursive function call.
    The system architecture includes multiple interface components.
    Each module has specific parameter configurations.
    """
    high_score = self.analyzer._measure_technical_level(technical_text)
    self.assertGreater(high_score, 0)

    # Low technical content
    simple_text = "The cat sat on the mat. It was a sunny day."
    low_score = self.analyzer._measure_technical_level(simple_text)
    self.assertLess(low_score, high_score)

  def test_measure_dialogue_ratio(self):
    """Test dialogue ratio measurement."""
    # High dialogue content
    dialogue_text = """
    John: Hello there.
    Mary: Hi, how are you?
    John: "I'm doing great," he said.
    Mary: "That's wonderful," she replied.
    """
    high_ratio = self.analyzer._measure_dialogue_ratio(dialogue_text)
    self.assertGreater(high_ratio, 0)

    # Low dialogue content
    narrative_text = """
    The sun was setting over the horizon.
    Birds flew across the orange sky.
    It was a peaceful evening.
    """
    low_ratio = self.analyzer._measure_dialogue_ratio(narrative_text)
    self.assertLess(low_ratio, high_ratio)

  def test_analyze_structure(self):
    """Test structure analysis."""
    text = """First paragraph here.

    Second paragraph with more content.
    It has multiple lines.

    Third paragraph."""

    structure = self.analyzer._analyze_structure(text)

    self.assertIn("avg_paragraph_length", structure)
    self.assertIn("avg_line_length", structure)
    self.assertIn("paragraph_count", structure)
    self.assertIn("line_count", structure)
    self.assertGreater(structure["paragraph_count"], 1)

  @patch("transcribe_pkg.core.analyzer.call_llm")
  @patch.object(ContentAnalyzer, "_detect_language", return_value="en")
  @patch.object(ContentAnalyzer, "_extract_domains", return_value=["science"])
  def test_analyze_content_full(self, mock_domains, mock_lang, mock_call_llm):
    """Test full content analysis."""
    mock_call_llm.return_value = "general"

    text = "This is a sample text for analysis. It has multiple sentences."

    analysis = self.analyzer.analyze_content(text)

    self.assertIn("length", analysis)
    self.assertIn("content_type", analysis)
    self.assertIn("language", analysis)
    self.assertIn("technical_level", analysis)
    self.assertIn("dialogue_ratio", analysis)
    self.assertIn("domains", analysis)
    self.assertIn("structure", analysis)

  def test_get_specialized_prompt_general(self):
    """Test getting specialized prompt for general content."""
    analysis = {
      "content_type": "general",
      "language": "en",
      "domains": [],
    }

    prompt = self.analyzer.get_specialized_prompt(analysis)

    self.assertIsInstance(prompt, str)
    self.assertGreater(len(prompt), 0)

  def test_get_specialized_prompt_with_context(self):
    """Test specialized prompt includes context."""
    analysis = {
      "content_type": "technical",
      "language": "en",
      "domains": ["computer science", "algorithms"],
    }

    prompt = self.analyzer.get_specialized_prompt(analysis, context="programming")

    self.assertIn("programming", prompt)


class TestSpecializedProcessor(unittest.TestCase):
  """Tests for the SpecializedProcessor class."""

  def test_init_with_provider(self):
    """Test SpecializedProcessor initialization with provider."""
    processor = SpecializedProcessor(provider="anthropic")
    self.assertEqual(processor.provider, "anthropic")

  @patch("transcribe_pkg.core.analyzer.call_llm")
  def test_process_content(self, mock_call_llm):
    """Test content processing."""
    mock_call_llm.return_value = "Processed content here."

    processor = SpecializedProcessor()

    result = processor.process_content(
      "Raw text to process.",
      context="science",
      language="en",
      model=DEFAULT_LLM_MODEL,
    )

    self.assertEqual(result, "Processed content here.")

  @patch("transcribe_pkg.core.analyzer.call_llm")
  def test_process_content_error_returns_original(self, mock_call_llm):
    """Test that processing errors return original text."""
    mock_call_llm.side_effect = Exception("API Error")

    processor = SpecializedProcessor()
    original_text = "Original text content."

    result = processor.process_content(original_text)

    self.assertEqual(result, original_text)

  @patch("transcribe_pkg.core.analyzer.call_llm")
  def test_summarize_content(self, mock_call_llm):
    """Test content summarization."""
    mock_call_llm.return_value = "This is a summary."

    processor = SpecializedProcessor()

    result = processor.summarize_content(
      "Long text that needs summarization. " * 10,
      model=DEFAULT_SUMMARY_MODEL,
    )

    self.assertEqual(result, "This is a summary.")

  @patch("transcribe_pkg.core.analyzer.call_llm")
  def test_summarize_content_with_max_length(self, mock_call_llm):
    """Test summarization respects max_length parameter."""
    mock_call_llm.return_value = "Short summary."

    processor = SpecializedProcessor()

    processor.summarize_content("Long text here.", max_length=100)

    # Verify system prompt includes max length instruction
    call_args = mock_call_llm.call_args
    system_prompt = call_args[1]["system_prompt"]
    self.assertIn("100 characters", system_prompt)

  @patch("transcribe_pkg.core.analyzer.call_llm")
  def test_summarize_content_error_returns_truncated(self, mock_call_llm):
    """Test summarization error returns truncated original."""
    mock_call_llm.side_effect = Exception("API Error")

    processor = SpecializedProcessor()
    long_text = "A" * 200

    result = processor.summarize_content(long_text)

    self.assertTrue(result.endswith("..."))
    self.assertLess(len(result), len(long_text))

  @patch("transcribe_pkg.core.analyzer.call_llm")
  def test_process_content_uses_provider(self, mock_call_llm):
    """Test that process_content passes provider to call_llm."""
    mock_call_llm.return_value = "Processed"

    processor = SpecializedProcessor(provider="anthropic")
    processor.process_content("Test text", model="claude-sonnet-4-5")

    # Verify provider was passed
    call_args = mock_call_llm.call_args
    self.assertEqual(call_args[1].get("provider"), "anthropic")


if __name__ == "__main__":
  unittest.main()

#fin
