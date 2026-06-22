"""Anthropic Claude provider client."""

from anthropic import Anthropic

from .base import LLMClientProtocol


class AnthropicClient(LLMClientProtocol):
  """Client for Anthropic Claude models."""

  def __init__(self, api_key: str) -> None:
    """Initialize Anthropic client.

    Args:
      api_key: Anthropic API key
    """
    self._client = Anthropic(api_key=api_key)

  def chat_completion(
    self,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs,
  ) -> str:
    """Send chat completion request to Anthropic.

    Args:
      system_prompt: System instructions
      user_prompt: User message
      model: Model identifier (e.g., "claude-3-5-sonnet-20241022")
      temperature: Sampling temperature
      max_tokens: Maximum tokens in response
      **kwargs: Additional Anthropic-specific options

    Returns:
      Generated text response
    """
    # Anthropic uses 'system' as a top-level param, not in messages
    create_kwargs = {
      "model": model,
      "max_tokens": max_tokens,
      "messages": [{"role": "user", "content": user_prompt}],
      **kwargs,
    }

    if system_prompt:
      create_kwargs["system"] = system_prompt

    # Anthropic requires temperature in [0.0, 1.0]; 0.0 is valid (deterministic)
    # and must be forwarded explicitly, otherwise the server defaults to 1.0.
    create_kwargs["temperature"] = max(0.0, min(temperature, 1.0))

    response = self._client.messages.create(**create_kwargs)

    # Extract text from content blocks
    text_parts = []
    for block in response.content:
      if hasattr(block, "text"):
        text_parts.append(block.text)

    return "".join(text_parts)


#fin
