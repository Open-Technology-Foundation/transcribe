#!/usr/bin/env python3
"""
Integration tests using real audio files and API calls.

These tests require:
- OPENAI_API_KEY for Whisper transcription (audio tests)
- ANTHROPIC_API_KEY for Claude post-processing

Run all: python -m pytest tests/test_audio_integration.py -v
Run Claude-only: python -m pytest tests/test_audio_integration.py -v -k "not Whisper"
Skip entirely: python -m pytest tests/ --ignore=tests/test_audio_integration.py
"""
import os
import unittest
import tempfile
from pathlib import Path

# Check API keys
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Test audio paths
TEST_AUDIO_DIR = Path(__file__).parent.parent / "test_audio"
SMALL_AUDIO = TEST_AUDIO_DIR / "small_test_audio.mp3"
MEDIUM_AUDIO = TEST_AUDIO_DIR / "medium_test_audio.mp3"
MEDIUM_REFERENCE = TEST_AUDIO_DIR / "medium_test_audio.txt"


def audio_files_exist():
  """Check if test audio files exist."""
  return SMALL_AUDIO.exists() and MEDIUM_AUDIO.exists()


# =============================================================================
# Tests that require OPENAI_API_KEY (Whisper transcription)
# =============================================================================


@unittest.skipUnless(OPENAI_KEY, "OPENAI_API_KEY not set")
@unittest.skipUnless(audio_files_exist(), "Test audio files not found")
class TestWhisperTranscription(unittest.TestCase):
  """Tests for Whisper audio transcription (requires OpenAI API)."""

  def test_small_audio_transcription(self):
    """Test transcription of small audio file."""
    from transcribe_pkg.utils.api_utils import transcribe_audio

    result = transcribe_audio(str(SMALL_AUDIO))

    self.assertIsInstance(result, str)
    self.assertGreater(len(result), 0)

  def test_small_audio_with_prompt(self):
    """Test transcription with context prompt."""
    from transcribe_pkg.utils.api_utils import transcribe_audio

    result = transcribe_audio(
      str(SMALL_AUDIO),
      prompt="This is a Zen Buddhist text about seeking enlightenment.",
    )

    self.assertIsInstance(result, str)
    self.assertGreater(len(result), 0)


@unittest.skipUnless(OPENAI_KEY, "OPENAI_API_KEY not set")
@unittest.skipUnless(ANTHROPIC_KEY, "ANTHROPIC_API_KEY not set")
@unittest.skipUnless(audio_files_exist(), "Test audio files not found")
class TestWhisperPipeline(unittest.TestCase):
  """Tests for full transcription pipeline (requires both API keys)."""

  def setUp(self):
    """Set up test fixtures."""
    self.temp_dir = tempfile.TemporaryDirectory()
    self.output_path = Path(self.temp_dir.name) / "output.txt"

  def tearDown(self):
    """Clean up after tests."""
    self.temp_dir.cleanup()

  def test_transcribe_audio_file_function(self):
    """Test the transcribe_audio_file high-level function."""
    from transcribe_pkg.core.transcriber import transcribe_audio_file

    result = transcribe_audio_file(
      audio_path=str(SMALL_AUDIO),
      output_file=str(self.output_path),
      parallel_processing=False,
      max_workers=1,
    )

    self.assertIsInstance(result, str)
    self.assertGreater(len(result), 0)
    self.assertTrue(self.output_path.exists())

  def test_transcriber_class(self):
    """Test the Transcriber class directly."""
    from transcribe_pkg.core.transcriber import Transcriber

    transcriber = Transcriber()
    result = transcriber.transcribe(str(SMALL_AUDIO))

    self.assertIsInstance(result, str)
    self.assertGreater(len(result), 0)


# =============================================================================
# Tests that require ANTHROPIC_API_KEY only (Claude processing)
# =============================================================================


@unittest.skipUnless(ANTHROPIC_KEY, "ANTHROPIC_API_KEY not set")
@unittest.skipUnless(audio_files_exist(), "Test audio files not found")
class TestClaudePostProcessing(unittest.TestCase):
  """Tests for post-processing with Claude (no OpenAI needed)."""

  def test_process_transcript_with_claude(self):
    """Test transcript processing with Claude model."""
    from transcribe_pkg.core.processor import TranscriptProcessor
    from transcribe_pkg.constants import DEFAULT_LLM_MODEL

    # Use a sample from the reference text
    sample_text = """
    In search of the ox. In the pasture of the world, I endlessly push aside
    the tall grasses in search of the ox. Following unnamed rivers, lost upon
    the interpenetrating paths of distant mountains, my strength failing and
    my vitality exhausted, I cannot find the ox.
    """

    processor = TranscriptProcessor(
      model=DEFAULT_LLM_MODEL,
      max_chunk_size=2000,
    )

    result = processor.process(
      sample_text.strip(),
      context="Zen Buddhism, philosophy",
      language="en",
    )

    self.assertIsInstance(result, str)
    self.assertGreater(len(result), 0)
    # Should preserve key concepts
    result_lower = result.lower()
    self.assertTrue(
      "ox" in result_lower or "search" in result_lower,
      f"Result should preserve key concepts: {result[:200]}",
    )

  def test_content_analyzer_with_real_text(self):
    """Test content analysis with real text."""
    from transcribe_pkg.core.analyzer import ContentAnalyzer

    # Load reference text
    with open(MEDIUM_REFERENCE) as f:
      reference_text = f.read()

    analyzer = ContentAnalyzer()
    analysis = analyzer.analyze_content(reference_text)

    self.assertIn("content_type", analysis)
    self.assertIn("language", analysis)
    self.assertIn("domains", analysis)
    self.assertEqual(analysis["language"], "en")

  def test_prompt_manager_context_extraction(self):
    """Test context extraction with real text."""
    from transcribe_pkg.utils.prompts import PromptManager

    sample_text = """
    The ox represents the Buddha-nature, the true self that we seek through
    meditation and spiritual practice. The ten stages of ox-herding represent
    the journey from seeking enlightenment to returning to the world with wisdom.
    """

    manager = PromptManager()
    context = manager.extract_context(sample_text)

    self.assertIsInstance(context, str)
    self.assertGreater(len(context), 0)

  def test_language_detection(self):
    """Test language detection with English text."""
    from transcribe_pkg.utils.prompts import PromptManager

    english_text = "The quick brown fox jumps over the lazy dog."
    manager = PromptManager()
    language = manager.detect_language(english_text)

    self.assertEqual(language, "en")

  def test_summary_generation(self):
    """Test summary generation with real text."""
    from transcribe_pkg.utils.prompts import PromptManager

    with open(MEDIUM_REFERENCE) as f:
      reference_text = f.read()

    manager = PromptManager()
    summary = manager.generate_summary(reference_text[:2000])

    self.assertIsInstance(summary, str)
    self.assertGreater(len(summary), 0)
    self.assertLess(len(summary), len(reference_text))


@unittest.skipUnless(ANTHROPIC_KEY, "ANTHROPIC_API_KEY not set")
class TestClaudeProviderIntegration(unittest.TestCase):
  """Tests for Claude provider with real API calls."""

  def test_anthropic_provider_direct(self):
    """Test Anthropic provider directly."""
    from transcribe_pkg.utils.providers.registry import (
      get_client_for_model,
      clear_client_cache,
    )

    clear_client_cache()
    client = get_client_for_model("claude-haiku-4-5")

    result = client.chat_completion(
      system_prompt="You are a helpful assistant.",
      user_prompt="Say 'hello' and nothing else.",
      model="claude-haiku-4-5",
      max_tokens=50,
    )

    self.assertIsInstance(result, str)
    self.assertIn("hello", result.lower())

  def test_call_llm_with_claude(self):
    """Test call_llm function with Claude model."""
    from transcribe_pkg.utils.api_utils import call_llm
    from transcribe_pkg.constants import DEFAULT_SUMMARY_MODEL

    result = call_llm(
      user_prompt="What is 2 + 2? Reply with just the number.",
      system_prompt="You are a math assistant. Give brief answers.",
      model=DEFAULT_SUMMARY_MODEL,
      max_tokens=50,
    )

    self.assertIsInstance(result, str)
    self.assertIn("4", result)

  def test_call_llm_with_sonnet(self):
    """Test call_llm with Sonnet model."""
    from transcribe_pkg.utils.api_utils import call_llm
    from transcribe_pkg.constants import DEFAULT_LLM_MODEL

    result = call_llm(
      user_prompt="Complete this sentence: The sky is",
      system_prompt="Complete the sentence with one word.",
      model=DEFAULT_LLM_MODEL,
      max_tokens=50,
    )

    self.assertIsInstance(result, str)
    self.assertGreater(len(result), 0)

  def test_provider_caching(self):
    """Test that provider clients are properly cached."""
    from transcribe_pkg.utils.providers.registry import (
      get_client_for_model,
      clear_client_cache,
      _client_cache,
    )

    clear_client_cache()
    self.assertEqual(len(_client_cache), 0)

    # First call creates client
    client1 = get_client_for_model("claude-sonnet-4-5")
    self.assertEqual(len(_client_cache), 1)

    # Second call reuses cached client
    client2 = get_client_for_model("claude-haiku-4-5")
    self.assertEqual(len(_client_cache), 1)  # Same provider
    self.assertIs(client1, client2)


# =============================================================================
# Tests that don't require any API keys (validation only)
# =============================================================================


class TestAudioFileValidation(unittest.TestCase):
  """Tests for audio file validation (no API keys needed)."""

  def test_audio_directory_exists(self):
    """Verify test audio directory exists."""
    self.assertTrue(
      TEST_AUDIO_DIR.exists(), f"Test audio directory not found: {TEST_AUDIO_DIR}"
    )

  def test_small_audio_exists(self):
    """Verify small audio file exists."""
    self.assertTrue(SMALL_AUDIO.exists(), f"Small audio file not found: {SMALL_AUDIO}")

  def test_medium_audio_exists(self):
    """Verify medium audio file exists."""
    self.assertTrue(
      MEDIUM_AUDIO.exists(), f"Medium audio file not found: {MEDIUM_AUDIO}"
    )

  def test_reference_text_exists(self):
    """Verify reference text exists."""
    self.assertTrue(
      MEDIUM_REFERENCE.exists(), f"Reference text not found: {MEDIUM_REFERENCE}"
    )

  def test_audio_file_sizes(self):
    """Verify audio files are reasonable sizes."""
    if not audio_files_exist():
      self.skipTest("Audio files not found")

    small_size = SMALL_AUDIO.stat().st_size
    medium_size = MEDIUM_AUDIO.stat().st_size

    # Small should be under 5MB
    self.assertLess(small_size, 5 * 1024 * 1024, "Small audio too large")
    # Medium should be under 25MB (Whisper limit)
    self.assertLess(medium_size, 25 * 1024 * 1024, "Medium audio exceeds Whisper limit")
    # Medium should be larger than small
    self.assertGreater(medium_size, small_size)

  def test_reference_text_content(self):
    """Verify reference text has expected content."""
    if not MEDIUM_REFERENCE.exists():
      self.skipTest("Reference file not found")

    with open(MEDIUM_REFERENCE) as f:
      content = f.read()

    # Should contain key terms from Ten Bulls
    self.assertIn("ox", content.lower())
    self.assertIn("footprints", content.lower())
    self.assertGreater(len(content), 500)


class TestConstantsValidation(unittest.TestCase):
  """Validate that constants are set to Claude models."""

  def test_default_llm_model_is_claude(self):
    """Verify DEFAULT_LLM_MODEL is Claude."""
    from transcribe_pkg.constants import DEFAULT_LLM_MODEL

    self.assertTrue(DEFAULT_LLM_MODEL.startswith("claude-"))

  def test_default_summary_model_is_claude(self):
    """Verify DEFAULT_SUMMARY_MODEL is Claude."""
    from transcribe_pkg.constants import DEFAULT_SUMMARY_MODEL

    self.assertTrue(DEFAULT_SUMMARY_MODEL.startswith("claude-"))

  def test_provider_routing_for_defaults(self):
    """Verify default models route to Anthropic."""
    from transcribe_pkg.constants import DEFAULT_LLM_MODEL, DEFAULT_SUMMARY_MODEL
    from transcribe_pkg.utils.providers.registry import get_provider_for_model

    self.assertEqual(get_provider_for_model(DEFAULT_LLM_MODEL), "anthropic")
    self.assertEqual(get_provider_for_model(DEFAULT_SUMMARY_MODEL), "anthropic")


if __name__ == "__main__":
  unittest.main()

#fin
