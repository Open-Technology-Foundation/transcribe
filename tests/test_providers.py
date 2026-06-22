#!/usr/bin/env python3
"""
Tests for the multi-provider LLM system.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from transcribe_pkg.utils.providers.registry import (
  get_provider_for_model,
  get_client_for_model,
  clear_client_cache,
  ProviderError,
  PROVIDER_OPENAI,
  PROVIDER_ANTHROPIC,
  PROVIDER_GEMINI,
  PROVIDER_OLLAMA,
)
from transcribe_pkg.utils.providers.base import LLMClientProtocol
from transcribe_pkg.constants import DEFAULT_LLM_MODEL, DEFAULT_SUMMARY_MODEL


class TestProviderRouting(unittest.TestCase):
  """Tests for provider routing based on model prefix."""

  def test_claude_models_route_to_anthropic(self):
    """Test Claude model prefixes route to Anthropic provider."""
    claude_models = [
      "claude-sonnet-4-5",
      "claude-haiku-4-5",
      "claude-3-5-sonnet-20241022",
      "claude-3-opus-20240229",
      "claude-instant-1.2",
    ]
    for model in claude_models:
      provider = get_provider_for_model(model)
      self.assertEqual(provider, PROVIDER_ANTHROPIC, f"Model {model} should route to anthropic")

  def test_gpt_models_route_to_openai(self):
    """Test GPT model prefixes route to OpenAI provider."""
    gpt_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    for model in gpt_models:
      provider = get_provider_for_model(model)
      self.assertEqual(provider, PROVIDER_OPENAI, f"Model {model} should route to openai")

  def test_o1_o3_models_route_to_openai(self):
    """Test o1/o3 reasoning models route to OpenAI provider."""
    reasoning_models = ["o1-preview", "o1-mini", "o3-mini"]
    for model in reasoning_models:
      provider = get_provider_for_model(model)
      self.assertEqual(provider, PROVIDER_OPENAI, f"Model {model} should route to openai")

  def test_gemini_models_route_to_gemini(self):
    """Test Gemini model prefixes route to Gemini provider."""
    gemini_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
    for model in gemini_models:
      provider = get_provider_for_model(model)
      self.assertEqual(provider, PROVIDER_GEMINI, f"Model {model} should route to gemini")

  def test_ollama_models_route_to_ollama(self):
    """Test Ollama model prefixes route to Ollama provider."""
    ollama_models = [
      "ollama/llama3.2",
      "llama3.2",
      "llama2",
      "mistral",
      "mistral-7b",
      "qwen2.5",
      "phi3",
    ]
    for model in ollama_models:
      provider = get_provider_for_model(model)
      self.assertEqual(provider, PROVIDER_OLLAMA, f"Model {model} should route to ollama")

  def test_gpt_oss_routes_to_ollama(self):
    """Test that gpt-oss (Ollama-hosted) routes to ollama, not openai."""
    for model in ["gpt-oss", "gpt-oss:20b", "gpt-oss-120b"]:
      provider = get_provider_for_model(model)
      self.assertEqual(provider, PROVIDER_OLLAMA, f"Model {model} should route to ollama")

  def test_additional_local_models_route_to_ollama(self):
    """Test that common local model name prefixes route to ollama."""
    local_models = [
      "gemma",
      "gemma2",
      "gemma2:9b",
      "mixtral",
      "mixtral-8x7b",
      "codellama",
      "codellama:13b",
      "llava",
      "deepseek",
      "deepseek-coder",
    ]
    for model in local_models:
      provider = get_provider_for_model(model)
      self.assertEqual(provider, PROVIDER_OLLAMA, f"Model {model} should route to ollama")

  def test_case_insensitive_routing(self):
    """Test that model prefix routing is case-insensitive."""
    self.assertEqual(get_provider_for_model("CLAUDE-sonnet-4-5"), PROVIDER_ANTHROPIC)
    self.assertEqual(get_provider_for_model("GPT-4o"), PROVIDER_OPENAI)
    self.assertEqual(get_provider_for_model("Gemini-1.5-pro"), PROVIDER_GEMINI)

  def test_unknown_model_raises_error(self):
    """Test that unknown model prefix raises ProviderError."""
    with self.assertRaises(ProviderError) as context:
      get_provider_for_model("unknown-model-xyz")
    self.assertIn("Cannot determine provider", str(context.exception))

  def test_provider_override(self):
    """Test explicit provider override bypasses prefix routing."""
    # Claude model with OpenAI override
    provider = get_provider_for_model("claude-sonnet-4-5", provider_override="openai")
    self.assertEqual(provider, PROVIDER_OPENAI)

    # GPT model with Anthropic override
    provider = get_provider_for_model("gpt-4o", provider_override="anthropic")
    self.assertEqual(provider, PROVIDER_ANTHROPIC)

  def test_invalid_provider_override_raises_error(self):
    """Test that invalid provider override raises ProviderError."""
    with self.assertRaises(ProviderError) as context:
      get_provider_for_model("claude-sonnet-4-5", provider_override="invalid-provider")
    self.assertIn("Unknown provider", str(context.exception))

  def test_default_models_use_claude(self):
    """Test that default models are Claude models."""
    self.assertTrue(DEFAULT_LLM_MODEL.startswith("claude-"))
    self.assertTrue(DEFAULT_SUMMARY_MODEL.startswith("claude-"))
    self.assertEqual(get_provider_for_model(DEFAULT_LLM_MODEL), PROVIDER_ANTHROPIC)
    self.assertEqual(get_provider_for_model(DEFAULT_SUMMARY_MODEL), PROVIDER_ANTHROPIC)


class TestClientCreation(unittest.TestCase):
  """Tests for provider client creation and caching."""

  def setUp(self):
    """Clear client cache before each test."""
    clear_client_cache()

  def tearDown(self):
    """Clear client cache after each test."""
    clear_client_cache()

  @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"})
  @patch("transcribe_pkg.utils.providers.anthropic_client.Anthropic")
  def test_anthropic_client_creation(self, mock_anthropic):
    """Test Anthropic client is created with API key."""
    mock_anthropic.return_value = MagicMock()

    client = get_client_for_model("claude-sonnet-4-5")

    self.assertIsNotNone(client)
    mock_anthropic.assert_called_once_with(api_key="test-anthropic-key")

  @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
  @patch("transcribe_pkg.utils.providers.anthropic_client.Anthropic")
  def test_client_caching(self, mock_anthropic):
    """Test that clients are cached and reused."""
    mock_anthropic.return_value = MagicMock()

    client1 = get_client_for_model("claude-sonnet-4-5")
    client2 = get_client_for_model("claude-haiku-4-5")

    # Should be the same cached client
    self.assertIs(client1, client2)
    # Anthropic should only be instantiated once
    mock_anthropic.assert_called_once()

  @patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=True)
  def test_missing_api_key_raises_error(self):
    """Test that missing API key raises ProviderError."""
    # Remove the key entirely
    if "ANTHROPIC_API_KEY" in os.environ:
      del os.environ["ANTHROPIC_API_KEY"]

    with self.assertRaises(ProviderError) as context:
      get_client_for_model("claude-sonnet-4-5")
    self.assertIn("ANTHROPIC_API_KEY", str(context.exception))

  @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True)
  def test_missing_openai_key_raises_error(self):
    """Test that missing OpenAI API key raises ProviderError."""
    if "OPENAI_API_KEY" in os.environ:
      del os.environ["OPENAI_API_KEY"]

    with self.assertRaises(ProviderError) as context:
      get_client_for_model("gpt-4o")
    self.assertIn("OPENAI_API_KEY", str(context.exception))

  def test_clear_client_cache(self):
    """Test that clear_client_cache clears the cache."""
    # Import the internal cache to verify
    from transcribe_pkg.utils.providers import registry

    # Add something to cache manually
    registry._client_cache["test"] = MagicMock()
    self.assertEqual(len(registry._client_cache), 1)

    clear_client_cache()
    self.assertEqual(len(registry._client_cache), 0)


class TestAnthropicClient(unittest.TestCase):
  """Tests for the Anthropic client implementation."""

  @patch("transcribe_pkg.utils.providers.anthropic_client.Anthropic")
  def test_chat_completion(self, mock_anthropic_class):
    """Test Anthropic chat completion."""
    # Setup mock
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.text = "Test response from Claude"
    mock_response.content = [mock_block]
    mock_client.messages.create.return_value = mock_response

    # Create client and call
    from transcribe_pkg.utils.providers.anthropic_client import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    result = client.chat_completion(
      system_prompt="You are helpful.",
      user_prompt="Hello",
      model="claude-sonnet-4-5",
      temperature=0.1,
      max_tokens=1000,
    )

    # Verify
    self.assertEqual(result, "Test response from Claude")
    mock_client.messages.create.assert_called_once()

    # Check call arguments
    call_kwargs = mock_client.messages.create.call_args[1]
    self.assertEqual(call_kwargs["model"], "claude-sonnet-4-5")
    self.assertEqual(call_kwargs["max_tokens"], 1000)
    self.assertEqual(call_kwargs["system"], "You are helpful.")
    self.assertEqual(call_kwargs["messages"], [{"role": "user", "content": "Hello"}])

  @patch("transcribe_pkg.utils.providers.anthropic_client.Anthropic")
  def test_chat_completion_no_system_prompt(self, mock_anthropic_class):
    """Test Anthropic chat completion without system prompt."""
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.text = "Response"
    mock_response.content = [mock_block]
    mock_client.messages.create.return_value = mock_response

    from transcribe_pkg.utils.providers.anthropic_client import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    client.chat_completion(
      system_prompt="",
      user_prompt="Hello",
      model="claude-sonnet-4-5",
    )

    # System should not be in kwargs when empty
    call_kwargs = mock_client.messages.create.call_args[1]
    self.assertNotIn("system", call_kwargs)

  @patch("transcribe_pkg.utils.providers.anthropic_client.Anthropic")
  def test_chat_completion_temperature_capping(self, mock_anthropic_class):
    """Test that temperature is capped at 1.0 for Anthropic."""
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.text = "Response"
    mock_response.content = [mock_block]
    mock_client.messages.create.return_value = mock_response

    from transcribe_pkg.utils.providers.anthropic_client import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    client.chat_completion(
      system_prompt="Test",
      user_prompt="Hello",
      model="claude-sonnet-4-5",
      temperature=1.5,  # Above 1.0
    )

    call_kwargs = mock_client.messages.create.call_args[1]
    self.assertEqual(call_kwargs["temperature"], 1.0)  # Should be capped

  @patch("transcribe_pkg.utils.providers.anthropic_client.Anthropic")
  def test_chat_completion_zero_temperature_forwarded(self, mock_anthropic_class):
    """Test that temperature=0.0 is forwarded (deterministic), not dropped."""
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.text = "Response"
    mock_response.content = [mock_block]
    mock_client.messages.create.return_value = mock_response

    from transcribe_pkg.utils.providers.anthropic_client import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    client.chat_completion(
      system_prompt="Test",
      user_prompt="Hello",
      model="claude-sonnet-4-5",
      temperature=0.0,  # Deterministic — must be sent, not omitted
    )

    call_kwargs = mock_client.messages.create.call_args[1]
    self.assertIn("temperature", call_kwargs)
    self.assertEqual(call_kwargs["temperature"], 0.0)

  @patch("transcribe_pkg.utils.providers.anthropic_client.Anthropic")
  def test_chat_completion_negative_temperature_clamped(self, mock_anthropic_class):
    """Test that a negative temperature is clamped to the valid floor 0.0."""
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.text = "Response"
    mock_response.content = [mock_block]
    mock_client.messages.create.return_value = mock_response

    from transcribe_pkg.utils.providers.anthropic_client import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    client.chat_completion(
      system_prompt="Test",
      user_prompt="Hello",
      model="claude-sonnet-4-5",
      temperature=-0.5,
    )

    call_kwargs = mock_client.messages.create.call_args[1]
    self.assertEqual(call_kwargs["temperature"], 0.0)


class _FakeGeminiResponse:
  """Minimal stand-in for a google.generativeai response object.

  Mirrors the real SDK contract: the ``.text`` accessor raises ValueError
  when there is no usable Part (safety block / finish_reason != STOP).
  """

  def __init__(self, candidates, text=None, text_error=None):
    self.candidates = candidates
    self._text = text
    self._text_error = text_error

  @property
  def text(self):
    if self._text_error is not None:
      raise self._text_error
    return self._text


class TestGeminiClient(unittest.TestCase):
  """Tests for the Gemini client implementation."""

  def setUp(self):
    """Stub the (optional, possibly-uninstalled) google.generativeai SDK."""
    import sys

    self._sdk_patcher = patch.dict(
      sys.modules,
      {
        "google": MagicMock(),
        "google.generativeai": MagicMock(),
      },
    )
    self._sdk_patcher.start()

  def tearDown(self):
    self._sdk_patcher.stop()

  @patch("transcribe_pkg.utils.providers.gemini_client.genai")
  def test_chat_completion_returns_text(self, mock_genai):
    """Test Gemini chat completion returns response text."""
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model

    mock_model.generate_content.return_value = _FakeGeminiResponse(
      candidates=[object()], text="Hello from Gemini"
    )

    from transcribe_pkg.utils.providers.gemini_client import GeminiClient

    client = GeminiClient(api_key="test-key")
    result = client.chat_completion(
      system_prompt="Be helpful.",
      user_prompt="Hello",
      model="gemini-1.5-pro",
    )

    self.assertEqual(result, "Hello from Gemini")

  @patch("transcribe_pkg.utils.providers.gemini_client.genai")
  def test_chat_completion_no_candidates_returns_empty(self, mock_genai):
    """Test that an empty-candidates response returns '' instead of raising."""
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model

    mock_model.generate_content.return_value = _FakeGeminiResponse(candidates=[])

    from transcribe_pkg.utils.providers.gemini_client import GeminiClient

    client = GeminiClient(api_key="test-key")
    result = client.chat_completion(
      system_prompt="",
      user_prompt="Hello",
      model="gemini-1.5-pro",
    )

    self.assertEqual(result, "")

  @patch("transcribe_pkg.utils.providers.gemini_client.genai")
  def test_chat_completion_blocked_text_raises_returns_empty(self, mock_genai):
    """Test that a safety-blocked response (.text raises ValueError) returns ''."""
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model

    # Real SDK raises ValueError on .text when there is no valid Part.
    mock_model.generate_content.return_value = _FakeGeminiResponse(
      candidates=[object()], text_error=ValueError("no valid Part")
    )

    from transcribe_pkg.utils.providers.gemini_client import GeminiClient

    client = GeminiClient(api_key="test-key")
    result = client.chat_completion(
      system_prompt="",
      user_prompt="Hello",
      model="gemini-1.5-pro",
    )

    self.assertEqual(result, "")


class TestImportErrorChaining(unittest.TestCase):
  """Tests that broken-SDK ImportErrors preserve their cause and message."""

  def setUp(self):
    clear_client_cache()

  def tearDown(self):
    clear_client_cache()

  def _force_import_error(self, target_module: str):
    """Return a patched __import__ that raises ImportError for target_module."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
      if target_module in name:
        raise ImportError("broken sdk internals")
      return real_import(name, *args, **kwargs)

    return fake_import

  @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
  def test_anthropic_import_error_preserves_cause(self):
    """Anthropic factory: broken SDK should chain cause and include detail."""
    import builtins

    with patch.object(
      builtins, "__import__", self._force_import_error("anthropic_client")
    ):
      with self.assertRaises(ProviderError) as ctx:
        get_client_for_model("claude-sonnet-4-5")

    self.assertIsInstance(ctx.exception.__cause__, ImportError)
    self.assertIn("broken sdk internals", str(ctx.exception))

  @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
  def test_openai_import_error_preserves_cause(self):
    """OpenAI factory: broken SDK should chain cause and include detail."""
    import builtins

    with patch.object(
      builtins, "__import__", self._force_import_error("openai_client")
    ):
      with self.assertRaises(ProviderError) as ctx:
        get_client_for_model("gpt-4o")

    self.assertIsInstance(ctx.exception.__cause__, ImportError)
    self.assertIn("broken sdk internals", str(ctx.exception))

  @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
  def test_gemini_import_error_preserves_cause(self):
    """Gemini factory: broken SDK should chain cause and include detail."""
    import builtins

    with patch.object(
      builtins, "__import__", self._force_import_error("gemini_client")
    ):
      with self.assertRaises(ProviderError) as ctx:
        get_client_for_model("gemini-1.5-pro")

    self.assertIsInstance(ctx.exception.__cause__, ImportError)
    self.assertIn("broken sdk internals", str(ctx.exception))

  def test_ollama_import_error_preserves_cause(self):
    """Ollama factory: broken SDK should chain cause and include detail."""
    import builtins

    with patch.object(
      builtins, "__import__", self._force_import_error("ollama_client")
    ):
      with self.assertRaises(ProviderError) as ctx:
        get_client_for_model("ollama/llama3.2")

    self.assertIsInstance(ctx.exception.__cause__, ImportError)
    self.assertIn("broken sdk internals", str(ctx.exception))


class TestLLMClientProtocol(unittest.TestCase):
  """Tests for the LLMClientProtocol interface."""

  def test_protocol_is_runtime_checkable(self):
    """Test that LLMClientProtocol can be used with isinstance."""
    # Create a mock that implements the protocol
    class MockClient:
      def chat_completion(
        self, system_prompt, user_prompt, model, temperature=0.0, max_tokens=4096, **kwargs
      ):
        return "response"

    client = MockClient()
    self.assertIsInstance(client, LLMClientProtocol)

  def test_non_conforming_class_fails_check(self):
    """Test that classes not implementing protocol fail isinstance check."""

    class NotAClient:
      pass

    obj = NotAClient()
    self.assertNotIsInstance(obj, LLMClientProtocol)


if __name__ == "__main__":
  unittest.main()

#fin
