#!/usr/bin/env python3
"""
Tests for constants and configuration values.
"""
import unittest

from transcribe_pkg.constants import (
  # Model defaults
  DEFAULT_LLM_MODEL,
  DEFAULT_SUMMARY_MODEL,
  DEFAULT_WHISPER_MODEL,
  DEFAULT_GPT_MODEL,
  # Temperature settings
  DEFAULT_TRANSCRIPTION_TEMPERATURE,
  DEFAULT_PROCESSING_TEMPERATURE,
  MIN_TEMPERATURE,
  MAX_TEMPERATURE,
  # Token limits
  DEFAULT_MAX_TOKENS,
  MIN_MAX_TOKENS,
  MAX_MAX_TOKENS,
  # Text processing
  DEFAULT_MAX_CHUNK_SIZE,
  MIN_CHUNK_SIZE,
  # Audio processing
  DEFAULT_CHUNK_LENGTH_MS,
  MAX_AUDIO_FILE_SIZE,
  # Supported formats
  SUPPORTED_AUDIO_FORMATS,
  SUPPORTED_SUBTITLE_FORMATS,
  # Exit codes
  EXIT_SUCCESS,
  EXIT_GENERAL_ERROR,
)


class TestModelConstants(unittest.TestCase):
  """Tests for model-related constants."""

  def test_default_llm_model_is_claude(self):
    """Test that DEFAULT_LLM_MODEL is a Claude model."""
    self.assertTrue(
      DEFAULT_LLM_MODEL.startswith("claude-"),
      f"DEFAULT_LLM_MODEL should be a Claude model, got: {DEFAULT_LLM_MODEL}",
    )

  def test_default_summary_model_is_claude(self):
    """Test that DEFAULT_SUMMARY_MODEL is a Claude model."""
    self.assertTrue(
      DEFAULT_SUMMARY_MODEL.startswith("claude-"),
      f"DEFAULT_SUMMARY_MODEL should be a Claude model, got: {DEFAULT_SUMMARY_MODEL}",
    )

  def test_default_llm_model_is_sonnet(self):
    """Test that DEFAULT_LLM_MODEL is Sonnet variant."""
    self.assertIn("sonnet", DEFAULT_LLM_MODEL.lower())

  def test_default_summary_model_is_haiku(self):
    """Test that DEFAULT_SUMMARY_MODEL is Haiku variant (fast/cheap)."""
    self.assertIn("haiku", DEFAULT_SUMMARY_MODEL.lower())

  def test_legacy_gpt_model_alias(self):
    """Test that DEFAULT_GPT_MODEL is aliased to DEFAULT_LLM_MODEL."""
    self.assertEqual(DEFAULT_GPT_MODEL, DEFAULT_LLM_MODEL)

  def test_whisper_model_unchanged(self):
    """Test that Whisper model is still OpenAI's whisper-1."""
    self.assertEqual(DEFAULT_WHISPER_MODEL, "whisper-1")


class TestTemperatureConstants(unittest.TestCase):
  """Tests for temperature-related constants."""

  def test_temperature_range(self):
    """Test temperature range is valid."""
    self.assertEqual(MIN_TEMPERATURE, 0.0)
    self.assertEqual(MAX_TEMPERATURE, 1.0)

  def test_default_temperatures_in_range(self):
    """Test default temperatures are within valid range."""
    self.assertGreaterEqual(DEFAULT_TRANSCRIPTION_TEMPERATURE, MIN_TEMPERATURE)
    self.assertLessEqual(DEFAULT_TRANSCRIPTION_TEMPERATURE, MAX_TEMPERATURE)
    self.assertGreaterEqual(DEFAULT_PROCESSING_TEMPERATURE, MIN_TEMPERATURE)
    self.assertLessEqual(DEFAULT_PROCESSING_TEMPERATURE, MAX_TEMPERATURE)

  def test_default_temperatures_are_low(self):
    """Test default temperatures are low for deterministic output."""
    self.assertLess(DEFAULT_TRANSCRIPTION_TEMPERATURE, 0.2)
    self.assertLess(DEFAULT_PROCESSING_TEMPERATURE, 0.2)


class TestTokenConstants(unittest.TestCase):
  """Tests for token-related constants."""

  def test_token_limits_valid(self):
    """Test token limits are valid."""
    self.assertEqual(MIN_MAX_TOKENS, 1)
    self.assertGreater(MAX_MAX_TOKENS, 0)
    self.assertGreater(DEFAULT_MAX_TOKENS, 0)

  def test_default_tokens_in_range(self):
    """Test default max tokens is within valid range."""
    self.assertGreaterEqual(DEFAULT_MAX_TOKENS, MIN_MAX_TOKENS)
    self.assertLessEqual(DEFAULT_MAX_TOKENS, MAX_MAX_TOKENS)


class TestTextProcessingConstants(unittest.TestCase):
  """Tests for text processing constants."""

  def test_chunk_size_valid(self):
    """Test chunk size constants are valid."""
    self.assertGreater(DEFAULT_MAX_CHUNK_SIZE, 0)
    self.assertGreater(MIN_CHUNK_SIZE, 0)
    self.assertGreater(DEFAULT_MAX_CHUNK_SIZE, MIN_CHUNK_SIZE)


class TestAudioProcessingConstants(unittest.TestCase):
  """Tests for audio processing constants."""

  def test_chunk_length_valid(self):
    """Test audio chunk length is valid."""
    self.assertGreater(DEFAULT_CHUNK_LENGTH_MS, 0)
    # 10 minutes = 600000 ms
    self.assertEqual(DEFAULT_CHUNK_LENGTH_MS, 600000)

  def test_max_file_size_valid(self):
    """Test max audio file size is 25MB (OpenAI limit)."""
    self.assertEqual(MAX_AUDIO_FILE_SIZE, 25 * 1024 * 1024)


class TestSupportedFormats(unittest.TestCase):
  """Tests for supported format constants."""

  def test_audio_formats_include_common_types(self):
    """Test common audio formats are supported."""
    common_formats = [".mp3", ".wav", ".m4a", ".mp4"]
    for fmt in common_formats:
      self.assertIn(fmt, SUPPORTED_AUDIO_FORMATS)

  def test_subtitle_formats_include_srt_vtt(self):
    """Test SRT and VTT subtitle formats are supported."""
    self.assertIn("srt", SUPPORTED_SUBTITLE_FORMATS)
    self.assertIn("vtt", SUPPORTED_SUBTITLE_FORMATS)


class TestExitCodes(unittest.TestCase):
  """Tests for exit code constants."""

  def test_success_is_zero(self):
    """Test EXIT_SUCCESS is 0."""
    self.assertEqual(EXIT_SUCCESS, 0)

  def test_error_is_nonzero(self):
    """Test EXIT_GENERAL_ERROR is non-zero."""
    self.assertNotEqual(EXIT_GENERAL_ERROR, 0)


if __name__ == "__main__":
  unittest.main()

#fin
