"""Ollama local model provider client."""

import ollama

from .base import LLMClientProtocol


class OllamaClient(LLMClientProtocol):
  """Client for local Ollama models."""

  def __init__(self, base_url: str = "http://localhost:11434") -> None:
    """Initialize Ollama client.

    Args:
      base_url: Ollama server URL (default: http://localhost:11434)
    """
    self._client = ollama.Client(host=base_url)

  def chat_completion(
    self,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs,
  ) -> str:
    """Send chat completion request to Ollama.

    Args:
      system_prompt: System instructions
      user_prompt: User message
      model: Model identifier (e.g., "ollama/llama3.2", "llama3.2")
      temperature: Sampling temperature
      max_tokens: Maximum tokens in response (num_predict in Ollama)
      **kwargs: Additional Ollama-specific options

    Returns:
      Generated text response
    """
    # Strip "ollama/" prefix if present
    if model.startswith("ollama/"):
      model = model[7:]

    messages = []

    if system_prompt:
      messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    # Ollama uses 'options' dict for generation parameters
    options = {
      "temperature": temperature,
      "num_predict": max_tokens,
    }

    response = self._client.chat(
      model=model,
      messages=messages,
      options=options,
    )

    return response["message"]["content"]


#fin
