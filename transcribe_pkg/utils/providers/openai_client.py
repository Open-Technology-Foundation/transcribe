"""OpenAI provider client."""

from openai import OpenAI

from .base import LLMClientProtocol


class OpenAILLMClient(LLMClientProtocol):
  """Client for OpenAI GPT models (LLM-only, no audio transcription)."""

  def __init__(self, api_key: str, base_url: str | None = None) -> None:
    """Initialize OpenAI client.

    Args:
      api_key: OpenAI API key
      base_url: Optional custom base URL (for OpenAI-compatible APIs)
    """
    self._client = OpenAI(api_key=api_key, base_url=base_url)

  def chat_completion(
    self,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs,
  ) -> str:
    """Send chat completion request to OpenAI.

    Args:
      system_prompt: System instructions
      user_prompt: User message
      model: Model identifier (e.g., "gpt-4o", "gpt-4o-mini")
      temperature: Sampling temperature
      max_tokens: Maximum tokens in response
      **kwargs: Additional OpenAI-specific options

    Returns:
      Generated text response
    """
    messages = []

    if system_prompt:
      messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    response = self._client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
      **kwargs,
    )

    return response.choices[0].message.content or ""


#fin
