"""Provider registry with model prefix routing."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .base import LLMClientProtocol


class ProviderError(Exception):
  """Error related to LLM provider operations."""

  pass


# Provider name constants
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GEMINI = "gemini"
PROVIDER_OLLAMA = "ollama"

# Model prefix to provider mapping
PREFIX_MAPPING: dict[str, str] = {
  "gpt-": PROVIDER_OPENAI,
  "o1-": PROVIDER_OPENAI,
  "o3-": PROVIDER_OPENAI,
  "claude-": PROVIDER_ANTHROPIC,
  "gemini-": PROVIDER_GEMINI,
  "ollama/": PROVIDER_OLLAMA,
  "llama": PROVIDER_OLLAMA,
  "mistral": PROVIDER_OLLAMA,
  "mixtral": PROVIDER_OLLAMA,
  "qwen": PROVIDER_OLLAMA,
  "phi": PROVIDER_OLLAMA,
  "gemma": PROVIDER_OLLAMA,
  "codellama": PROVIDER_OLLAMA,
  "llava": PROVIDER_OLLAMA,
  "deepseek": PROVIDER_OLLAMA,
}

# Cached client instances
_client_cache: dict[str, "LLMClientProtocol"] = {}


def get_provider_for_model(model: str, provider_override: str | None = None) -> str:
  """Determine provider based on model name prefix.

  Args:
    model: Model identifier (e.g., "gpt-4o", "claude-3-sonnet")
    provider_override: Explicit provider name (overrides prefix detection)

  Returns:
    Provider name (openai, anthropic, gemini, ollama)

  Raises:
    ProviderError: If provider cannot be determined
  """
  if provider_override:
    valid_providers = {PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_GEMINI, PROVIDER_OLLAMA}
    if provider_override not in valid_providers:
      raise ProviderError(
        f"Unknown provider: {provider_override}. "
        f"Valid providers: {', '.join(sorted(valid_providers))}"
      )
    return provider_override

  model_lower = model.lower()

  # gpt-oss is an open-weight model served locally via Ollama; it must be
  # matched before the generic "gpt-" -> OpenAI prefix below.
  if model_lower.startswith("gpt-oss"):
    return PROVIDER_OLLAMA

  for prefix, provider in PREFIX_MAPPING.items():
    if model_lower.startswith(prefix):
      return provider

  raise ProviderError(
    f"Cannot determine provider for model: {model}. "
    "Use --provider to specify explicitly, or use a recognized model prefix "
    "(gpt-*, o1-*, o3-*, claude-*, gemini-*, ollama/*, llama*, mistral*, "
    "mixtral*, qwen*, phi*, gemma*, codellama*, llava*, deepseek*, gpt-oss*)."
  )


def get_client_for_model(
  model: str,
  provider_override: str | None = None,
) -> "LLMClientProtocol":
  """Get or create a provider client for the specified model.

  Args:
    model: Model identifier
    provider_override: Explicit provider name

  Returns:
    Provider client implementing LLMClientProtocol

  Raises:
    ProviderError: If provider SDK not installed or missing API key
  """
  provider = get_provider_for_model(model, provider_override)

  if provider in _client_cache:
    return _client_cache[provider]

  client = _create_client(provider)
  _client_cache[provider] = client
  return client


def _create_client(provider: str) -> "LLMClientProtocol":
  """Create a new provider client instance.

  Args:
    provider: Provider name

  Returns:
    Provider client instance

  Raises:
    ProviderError: If SDK not installed or API key missing
  """
  if provider == PROVIDER_OPENAI:
    return _create_openai_client()
  elif provider == PROVIDER_ANTHROPIC:
    return _create_anthropic_client()
  elif provider == PROVIDER_GEMINI:
    return _create_gemini_client()
  elif provider == PROVIDER_OLLAMA:
    return _create_ollama_client()
  else:
    raise ProviderError(f"Unknown provider: {provider}")


def _create_openai_client() -> "LLMClientProtocol":
  """Create OpenAI client."""
  api_key = os.environ.get("OPENAI_API_KEY")
  if not api_key:
    raise ProviderError(
      "OPENAI_API_KEY environment variable not set. "
      "Set it with: export OPENAI_API_KEY=sk-..."
    )

  try:
    from .openai_client import OpenAILLMClient

    return OpenAILLMClient(api_key=api_key)
  except ImportError as e:
    raise ProviderError(
      f"OpenAI SDK not available ({e}). "
      "Install or repair with: pip install openai"
    ) from e


def _create_anthropic_client() -> "LLMClientProtocol":
  """Create Anthropic client."""
  api_key = os.environ.get("ANTHROPIC_API_KEY")
  if not api_key:
    raise ProviderError(
      "ANTHROPIC_API_KEY environment variable not set. "
      "Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
    )

  try:
    from .anthropic_client import AnthropicClient

    return AnthropicClient(api_key=api_key)
  except ImportError as e:
    raise ProviderError(
      f"Anthropic SDK not available ({e}). "
      "Install or repair with: pip install anthropic"
    ) from e


def _create_gemini_client() -> "LLMClientProtocol":
  """Create Gemini client."""
  api_key = os.environ.get("GOOGLE_API_KEY")
  if not api_key:
    raise ProviderError(
      "GOOGLE_API_KEY environment variable not set. "
      "Set it with: export GOOGLE_API_KEY=..."
    )

  try:
    from .gemini_client import GeminiClient

    return GeminiClient(api_key=api_key)
  except ImportError as e:
    raise ProviderError(
      f"Google Generative AI SDK not available ({e}). "
      "Install or repair with: pip install google-generativeai"
    ) from e


def _create_ollama_client() -> "LLMClientProtocol":
  """Create Ollama client."""
  base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

  try:
    from .ollama_client import OllamaClient

    return OllamaClient(base_url=base_url)
  except ImportError as e:
    raise ProviderError(
      f"Ollama SDK not available ({e}). "
      "Install or repair with: pip install ollama"
    ) from e


def clear_client_cache() -> None:
  """Clear the client cache. Useful for testing."""
  _client_cache.clear()


#fin
