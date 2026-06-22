#!/usr/bin/env python3
"""
Tests for prompt management functionality.
"""
import unittest
from unittest.mock import patch

from transcribe_pkg.utils.prompts import PromptManager
from transcribe_pkg.constants import DEFAULT_SUMMARY_MODEL


class TestPromptManager(unittest.TestCase):
  """Tests for the PromptManager class."""

  def setUp(self):
    """Set up test fixtures."""
    self.manager = PromptManager()

  def test_init_with_provider(self):
    """Test PromptManager initialization with provider."""
    manager = PromptManager(provider="anthropic")
    self.assertEqual(manager.provider, "anthropic")

  def test_get_template_exists(self):
    """Test getting an existing template."""
    template = self.manager.get_template("transcript_processing")
    self.assertIsInstance(template, str)
    self.assertIn("Translation/Transcription", template)

  def test_get_template_not_found(self):
    """Test getting a non-existent template raises ValueError."""
    with self.assertRaises(ValueError) as context:
      self.manager.get_template("nonexistent_template")
    self.assertIn("not found", str(context.exception))

  def test_set_template(self):
    """Test setting a custom template."""
    custom_template = "Custom template content"
    self.manager.set_template("custom", custom_template)

    result = self.manager.get_template("custom")
    self.assertEqual(result, custom_template)

  def test_builtin_templates_exist(self):
    """Test that all built-in templates exist."""
    expected_templates = [
      "transcript_processing",
      "dialogue_cleaning",
      "technical_cleaning",
      "context_summary",
      "context_extraction",
      "language_detection",
    ]

    for template_name in expected_templates:
      template = self.manager.get_template(template_name)
      self.assertIsInstance(template, str)
      self.assertGreater(len(template), 0)

  def test_get_system_prompt_basic(self):
    """Test getting a basic system prompt."""
    prompt = self.manager.get_system_prompt("transcript_processing")

    self.assertIsInstance(prompt, str)
    self.assertIn("Translation/Transcription", prompt)

  def test_get_system_prompt_with_context(self):
    """Test system prompt includes context."""
    prompt = self.manager.get_system_prompt(
      "transcript_processing", context="neuroscience,psychology"
    )

    self.assertIn("extensive knowledge", prompt)
    self.assertIn("neuroscience", prompt)
    self.assertIn("psychology", prompt)

  def test_get_system_prompt_with_language(self):
    """Test system prompt includes language handling for non-English."""
    prompt = self.manager.get_system_prompt("transcript_processing", language="es")

    self.assertIn("translate", prompt.lower())
    self.assertIn("ES", prompt)

  def test_get_system_prompt_english_no_translation(self):
    """Test system prompt doesn't include translation for English."""
    prompt = self.manager.get_system_prompt("transcript_processing", language="en")

    # Should not have language task for English
    self.assertNotIn("translate/interpret", prompt)

  def test_get_system_prompt_with_context_summary(self):
    """Test system prompt includes context summary."""
    prompt = self.manager.get_system_prompt(
      "transcript_processing", context_summary="This is about AI research."
    )

    self.assertIn("Context Summary", prompt)
    self.assertIn("This is about AI research.", prompt)

  def test_add_oxford_comma_empty(self):
    """Test Oxford comma with empty string."""
    result = self.manager._add_oxford_comma("")
    self.assertEqual(result, "")

  def test_add_oxford_comma_single(self):
    """Test Oxford comma with single item."""
    result = self.manager._add_oxford_comma("science")
    self.assertEqual(result, "science")

  def test_add_oxford_comma_two_items(self):
    """Test Oxford comma with two items."""
    result = self.manager._add_oxford_comma("science, math")
    self.assertEqual(result, "science and math")

  def test_add_oxford_comma_three_items(self):
    """Test Oxford comma with three items."""
    result = self.manager._add_oxford_comma("science, math, philosophy")
    self.assertEqual(result, "science, math, and philosophy")

  def test_add_oxford_comma_four_items(self):
    """Test Oxford comma with four items."""
    result = self.manager._add_oxford_comma("science, math, philosophy, history")
    self.assertEqual(result, "science, math, philosophy, and history")

  @patch("transcribe_pkg.utils.prompts.call_llm")
  def test_extract_context(self, mock_call_llm):
    """Test context extraction."""
    mock_call_llm.return_value = "neuroscience,psychology,biology"

    result = self.manager.extract_context("Text about the brain and behavior.")

    self.assertEqual(result, "neuroscience,psychology,biology")
    mock_call_llm.assert_called_once()

    # Verify call_llm was called with correct model
    call_args = mock_call_llm.call_args
    self.assertEqual(call_args[1]["model"], DEFAULT_SUMMARY_MODEL)

  @patch("transcribe_pkg.utils.prompts.call_llm")
  def test_extract_context_with_custom_model(self, mock_call_llm):
    """Test context extraction with custom model."""
    mock_call_llm.return_value = "science"

    self.manager.extract_context("Some text.", model="claude-haiku-4-5")

    call_args = mock_call_llm.call_args
    self.assertEqual(call_args[1]["model"], "claude-haiku-4-5")

  @patch("transcribe_pkg.utils.prompts.call_llm")
  def test_extract_context_error_returns_empty(self, mock_call_llm):
    """Test context extraction returns empty string on error."""
    mock_call_llm.side_effect = Exception("API Error")

    result = self.manager.extract_context("Some text.")

    self.assertEqual(result, "")

  @patch("transcribe_pkg.utils.prompts.call_llm")
  def test_extract_context_uses_provider(self, mock_call_llm):
    """Test that extract_context passes provider to call_llm."""
    mock_call_llm.return_value = "science"

    manager = PromptManager(provider="anthropic")
    manager.extract_context("Test text.")

    call_args = mock_call_llm.call_args
    self.assertEqual(call_args[1].get("provider"), "anthropic")

  @patch("transcribe_pkg.utils.prompts.call_llm")
  def test_detect_language(self, mock_call_llm):
    """Test language detection."""
    mock_call_llm.return_value = "en"

    result = self.manager.detect_language("Hello, how are you?")

    self.assertEqual(result, "en")

  @patch("transcribe_pkg.utils.prompts.call_llm")
  def test_detect_language_non_english(self, mock_call_llm):
    """Test language detection for non-English text."""
    mock_call_llm.return_value = "es"

    result = self.manager.detect_language("Hola, como estas?")

    self.assertEqual(result, "es")

  @patch("transcribe_pkg.utils.prompts.call_llm")
  def test_detect_language_error_returns_english(self, mock_call_llm):
    """Test language detection returns English on error."""
    mock_call_llm.side_effect = Exception("API Error")

    result = self.manager.detect_language("Some text.")

    self.assertEqual(result, "en")

  @patch("transcribe_pkg.utils.prompts.call_llm")
  def test_detect_language_uses_provider(self, mock_call_llm):
    """Test that detect_language passes provider to call_llm."""
    mock_call_llm.return_value = "en"

    manager = PromptManager(provider="anthropic")
    manager.detect_language("Test text.")

    call_args = mock_call_llm.call_args
    self.assertEqual(call_args[1].get("provider"), "anthropic")

  @patch("transcribe_pkg.utils.prompts.call_llm")
  def test_generate_summary(self, mock_call_llm):
    """Test summary generation."""
    mock_call_llm.return_value = "This is a summary of the text."

    result = self.manager.generate_summary("Long text that needs summarization.")

    self.assertEqual(result, "This is a summary of the text.")
    mock_call_llm.assert_called_once()

  @patch("transcribe_pkg.utils.prompts.call_llm")
  def test_generate_summary_error_returns_empty(self, mock_call_llm):
    """Test summary generation returns empty string on error."""
    mock_call_llm.side_effect = Exception("API Error")

    result = self.manager.generate_summary("Some text.")

    self.assertEqual(result, "")

  @patch("transcribe_pkg.utils.prompts.call_llm")
  def test_generate_summary_uses_provider(self, mock_call_llm):
    """Test that generate_summary passes provider to call_llm."""
    mock_call_llm.return_value = "Summary"

    manager = PromptManager(provider="anthropic")
    manager.generate_summary("Test text.")

    call_args = mock_call_llm.call_args
    self.assertEqual(call_args[1].get("provider"), "anthropic")


class TestPromptTemplates(unittest.TestCase):
  """Tests for prompt template content."""

  def setUp(self):
    """Set up test fixtures."""
    self.manager = PromptManager()

  def test_transcript_processing_template_structure(self):
    """Test transcript processing template has required sections."""
    template = self.manager.get_template("transcript_processing")

    self.assertIn("Grammar", template)
    self.assertIn("Relevance", template)
    self.assertIn("Content Integrity", template)
    self.assertIn("{context}", template)  # Placeholder
    self.assertIn("{language_task}", template)  # Placeholder

  def test_dialogue_cleaning_template_structure(self):
    """Test dialogue cleaning template has required sections."""
    template = self.manager.get_template("dialogue_cleaning")

    self.assertIn("Conversation Flow", template)
    self.assertIn("Speaker", template)
    self.assertIn("Readability", template)

  def test_technical_cleaning_template_structure(self):
    """Test technical cleaning template has required sections."""
    template = self.manager.get_template("technical_cleaning")

    self.assertIn("Technical Accuracy", template)
    self.assertIn("Structural Clarity", template)

  def test_context_extraction_template(self):
    """Test context extraction template is concise."""
    template = self.manager.get_template("context_extraction")

    self.assertIn("academic", template.lower())
    self.assertIn("comma-separated", template.lower())

  def test_language_detection_template(self):
    """Test language detection template requests ISO code."""
    template = self.manager.get_template("language_detection")

    self.assertIn("ISO 639-1", template)
    self.assertIn("two-character", template.lower())


if __name__ == "__main__":
  unittest.main()

#fin
