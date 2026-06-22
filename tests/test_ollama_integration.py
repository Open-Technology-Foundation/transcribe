#!/usr/bin/env python3
"""
Integration tests using local Ollama models.

These tests require:
- Ollama server running locally (http://localhost:11434)
- At least one model pulled (e.g., qwen3:8b, gemma3:4b, llama3.2)

Run with: python -m pytest tests/test_ollama_integration.py -v
Skip with: python -m pytest tests/ --ignore=tests/test_ollama_integration.py
"""
import unittest
from pathlib import Path

# Check if Ollama is available
OLLAMA_AVAILABLE = False
OLLAMA_MODELS = []

try:
  import ollama
  client = ollama.Client()
  response = client.list()
  # Handle both old dict API and new typed object API
  if hasattr(response, "models"):
    OLLAMA_MODELS = [m.model for m in response.models]
  elif isinstance(response, dict):
    OLLAMA_MODELS = [m["name"] for m in response.get("models", [])]
  OLLAMA_AVAILABLE = len(OLLAMA_MODELS) > 0
except Exception:
  pass

# Preferred test models (in order of preference - smaller/faster first)
PREFERRED_MODELS = ["gemma3:4b", "qwen3:8b", "llama3.2", "mistral"]

def get_test_model():
  """Get the best available model for testing."""
  for model in PREFERRED_MODELS:
    base_name = model.split(":")[0]
    if any(base_name in m for m in OLLAMA_MODELS):
      # Find the exact model name
      for m in OLLAMA_MODELS:
        if base_name in m:
          return m
  # Fall back to first available model
  return OLLAMA_MODELS[0] if OLLAMA_MODELS else None

TEST_MODEL = get_test_model()

# Test audio paths
TEST_AUDIO_DIR = Path(__file__).parent.parent / "test_audio"
MEDIUM_REFERENCE = TEST_AUDIO_DIR / "medium_test_audio.txt"


@unittest.skipUnless(OLLAMA_AVAILABLE, "Ollama not available or no models installed")
class TestOllamaProvider(unittest.TestCase):
  """Tests for Ollama provider with local models."""

  def test_ollama_client_direct(self):
    """Test Ollama client directly."""
    from transcribe_pkg.utils.providers.ollama_client import OllamaClient

    client = OllamaClient()
    result = client.chat_completion(
      system_prompt="You are a helpful assistant. Be very brief.",
      user_prompt="Say 'hello' and nothing else.",
      model=TEST_MODEL,
      max_tokens=50,
    )

    self.assertIsInstance(result, str)
    self.assertGreater(len(result), 0)

  def test_ollama_client_with_prefix(self):
    """Test Ollama client strips ollama/ prefix."""
    from transcribe_pkg.utils.providers.ollama_client import OllamaClient

    client = OllamaClient()
    result = client.chat_completion(
      system_prompt="Be brief.",
      user_prompt="What is 1+1? Just the number.",
      model=f"ollama/{TEST_MODEL}",
      max_tokens=50,
    )

    self.assertIsInstance(result, str)
    self.assertIn("2", result)

  def test_ollama_no_system_prompt(self):
    """Test Ollama client without system prompt."""
    from transcribe_pkg.utils.providers.ollama_client import OllamaClient

    client = OllamaClient()
    result = client.chat_completion(
      system_prompt="",
      user_prompt="Say 'test' and nothing else.",
      model=TEST_MODEL,
      max_tokens=50,
    )

    self.assertIsInstance(result, str)
    self.assertGreater(len(result), 0)


@unittest.skipUnless(OLLAMA_AVAILABLE, "Ollama not available or no models installed")
class TestOllamaProviderRouting(unittest.TestCase):
  """Tests for provider routing to Ollama."""

  def setUp(self):
    """Clear provider cache before each test."""
    from transcribe_pkg.utils.providers.registry import clear_client_cache
    clear_client_cache()

  def test_ollama_prefix_routing(self):
    """Test that ollama/ prefix routes to Ollama provider."""
    from transcribe_pkg.utils.providers.registry import get_provider_for_model

    self.assertEqual(get_provider_for_model(f"ollama/{TEST_MODEL}"), "ollama")

  def test_llama_prefix_routing(self):
    """Test that llama prefix routes to Ollama provider."""
    from transcribe_pkg.utils.providers.registry import get_provider_for_model

    self.assertEqual(get_provider_for_model("llama3.2"), "ollama")
    self.assertEqual(get_provider_for_model("llama2"), "ollama")

  def test_mistral_prefix_routing(self):
    """Test that mistral prefix routes to Ollama provider."""
    from transcribe_pkg.utils.providers.registry import get_provider_for_model

    self.assertEqual(get_provider_for_model("mistral"), "ollama")
    self.assertEqual(get_provider_for_model("mistral-7b"), "ollama")

  def test_qwen_prefix_routing(self):
    """Test that qwen prefix routes to Ollama provider."""
    from transcribe_pkg.utils.providers.registry import get_provider_for_model

    self.assertEqual(get_provider_for_model("qwen2.5"), "ollama")
    self.assertEqual(get_provider_for_model("qwen3:8b"), "ollama")

  def test_get_client_for_ollama_model(self):
    """Test getting client for Ollama model."""
    from transcribe_pkg.utils.providers.registry import get_client_for_model
    from transcribe_pkg.utils.providers.ollama_client import OllamaClient

    client = get_client_for_model(f"ollama/{TEST_MODEL}")
    self.assertIsInstance(client, OllamaClient)


@unittest.skipUnless(OLLAMA_AVAILABLE, "Ollama not available or no models installed")
class TestOllamaCallLLM(unittest.TestCase):
  """Tests for call_llm with Ollama models."""

  def setUp(self):
    """Clear provider cache before each test."""
    from transcribe_pkg.utils.providers.registry import clear_client_cache
    clear_client_cache()

  def test_call_llm_with_ollama(self):
    """Test call_llm function routes to Ollama."""
    from transcribe_pkg.utils.api_utils import call_llm

    result = call_llm(
      user_prompt="What is 2 + 2? Reply with just the number.",
      system_prompt="You are a math assistant. Give only the answer.",
      model=f"ollama/{TEST_MODEL}",
      max_tokens=50,
    )

    self.assertIsInstance(result, str)
    self.assertIn("4", result)

  def test_call_llm_with_llama_prefix(self):
    """Test call_llm with llama prefix routes to Ollama."""
    from transcribe_pkg.utils.api_utils import call_llm

    # Skip if no llama model available
    if not any("llama" in m.lower() for m in OLLAMA_MODELS):
      self.skipTest("No llama model available")

    llama_model = next((m for m in OLLAMA_MODELS if "llama" in m.lower()), None)

    result = call_llm(
      user_prompt="Say 'yes'",
      system_prompt="Respond with only 'yes' or 'no'.",
      model=llama_model,
      max_tokens=50,
    )

    self.assertIsInstance(result, str)

  def test_call_llm_with_provider_override(self):
    """Test call_llm with explicit provider override to Ollama."""
    from transcribe_pkg.utils.api_utils import call_llm

    result = call_llm(
      user_prompt="Complete: The sky is",
      system_prompt="Complete with one word.",
      model=TEST_MODEL,
      provider="ollama",
      max_tokens=50,
    )

    self.assertIsInstance(result, str)
    self.assertGreater(len(result), 0)


@unittest.skipUnless(OLLAMA_AVAILABLE, "Ollama not available or no models installed")
@unittest.skipUnless(MEDIUM_REFERENCE.exists(), "Reference text not found")
class TestOllamaPostProcessing(unittest.TestCase):
  """Tests for transcript post-processing with Ollama models."""

  def setUp(self):
    """Clear provider cache before each test."""
    from transcribe_pkg.utils.providers.registry import clear_client_cache
    clear_client_cache()

  def test_process_transcript_with_ollama(self):
    """Test transcript processing with local Ollama model."""
    from transcribe_pkg.core.processor import TranscriptProcessor

    sample_text = """
    In search of the ox. In the pasture of the world, I endlessly push aside
    the tall grasses in search of the ox. Following unnamed rivers, lost upon
    the interpenetrating paths of distant mountains.
    """

    processor = TranscriptProcessor(
      model=f"ollama/{TEST_MODEL}",
      max_chunk_size=2000,
      provider="ollama",
    )

    result = processor.process(
      sample_text.strip(),
      context="Zen Buddhism",
      language="en",
    )

    self.assertIsInstance(result, str)
    self.assertGreater(len(result), 0)

  def test_prompt_manager_with_ollama(self):
    """Test PromptManager context extraction with Ollama."""
    from transcribe_pkg.utils.prompts import PromptManager

    sample_text = """
    The Buddha taught the Four Noble Truths: suffering exists, suffering has a cause,
    suffering can end, and there is a path to end suffering. This is the foundation
    of Buddhist philosophy and practice.
    """

    manager = PromptManager(provider="ollama")
    context = manager.extract_context(sample_text, model=f"ollama/{TEST_MODEL}")

    self.assertIsInstance(context, str)
    # Local models may give different results, just verify we get something
    self.assertGreater(len(context), 0)

  def test_language_detection_with_ollama(self):
    """Test language detection with Ollama."""
    from transcribe_pkg.utils.prompts import PromptManager

    english_text = "The quick brown fox jumps over the lazy dog."

    manager = PromptManager(provider="ollama")
    language = manager.detect_language(english_text, model=f"ollama/{TEST_MODEL}")

    self.assertIsInstance(language, str)
    # Local models should still detect English
    self.assertEqual(language.lower().strip(), "en")


@unittest.skipUnless(OLLAMA_AVAILABLE, "Ollama not available or no models installed")
class TestOllamaContentAnalysis(unittest.TestCase):
  """Tests for content analysis with Ollama models."""

  def setUp(self):
    """Clear provider cache before each test."""
    from transcribe_pkg.utils.providers.registry import clear_client_cache
    clear_client_cache()

  def test_content_analyzer_with_ollama(self):
    """Test ContentAnalyzer with Ollama model."""
    from transcribe_pkg.core.analyzer import ContentAnalyzer

    technical_text = """
    The algorithm has O(n log n) time complexity. The implementation uses
    a binary search tree for efficient lookups. The hash table provides
    constant time access to the cached values.
    """

    analyzer = ContentAnalyzer(
      model=f"ollama/{TEST_MODEL}",
      provider="ollama",
    )

    analysis = analyzer.analyze_content(technical_text)

    self.assertIn("content_type", analysis)
    self.assertIn("language", analysis)
    self.assertIn("technical_level", analysis)

  def test_specialized_processor_with_ollama(self):
    """Test SpecializedProcessor with Ollama model."""
    from transcribe_pkg.core.analyzer import SpecializedProcessor

    sample_text = "This is a simple test sentence for processing."

    processor = SpecializedProcessor(provider="ollama")
    result = processor.process_content(
      sample_text,
      model=f"ollama/{TEST_MODEL}",
    )

    self.assertIsInstance(result, str)
    self.assertGreater(len(result), 0)


class TestOllamaAvailability(unittest.TestCase):
  """Tests for Ollama availability (always run)."""

  def test_ollama_module_importable(self):
    """Test that ollama module can be imported."""
    try:
      import ollama
      self.assertTrue(True)
    except ImportError:
      self.skipTest("Ollama module not installed")

  def test_ollama_server_reachable(self):
    """Test that Ollama server is reachable."""
    if not OLLAMA_AVAILABLE:
      self.skipTest("Ollama server not available")

    import ollama
    client = ollama.Client()
    models = client.list()
    self.assertIn("models", models)

  def test_ollama_has_models(self):
    """Test that Ollama has models available."""
    if not OLLAMA_AVAILABLE:
      self.skipTest("Ollama not available")

    self.assertGreater(len(OLLAMA_MODELS), 0, "No Ollama models installed")

  def test_preferred_model_available(self):
    """Test that at least one preferred model is available."""
    if not OLLAMA_AVAILABLE:
      self.skipTest("Ollama not available")

    self.assertIsNotNone(TEST_MODEL, "No preferred test model available")


if __name__ == "__main__":
  # Print available models for debugging
  if OLLAMA_AVAILABLE:
    print(f"Ollama available with models: {OLLAMA_MODELS}")
    print(f"Using test model: {TEST_MODEL}")
  else:
    print("Ollama not available")

  unittest.main()

#fin
