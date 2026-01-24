"""Base protocol for LLM provider clients."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClientProtocol(Protocol):
  """Protocol defining the interface for LLM provider clients.

  All provider clients must implement chat_completion() to enable
  transparent routing through the provider registry.
  """

  def chat_completion(
    self,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs,
  ) -> str:
    """Send a chat completion request to the LLM provider.

    Args:
      system_prompt: System instructions for the model
      user_prompt: User message/prompt
      model: Model identifier (provider-specific)
      temperature: Sampling temperature (0.0-2.0)
      max_tokens: Maximum tokens in response
      **kwargs: Provider-specific options

    Returns:
      Generated text response from the model
    """
    ...


#fin
