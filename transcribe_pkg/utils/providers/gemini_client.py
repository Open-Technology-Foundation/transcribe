"""Google Gemini provider client."""

import google.generativeai as genai

from .base import LLMClientProtocol


class GeminiClient(LLMClientProtocol):
  """Client for Google Gemini models."""

  def __init__(self, api_key: str) -> None:
    """Initialize Gemini client.

    Args:
      api_key: Google API key
    """
    genai.configure(api_key=api_key)
    self._api_key = api_key

  def chat_completion(
    self,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs,
  ) -> str:
    """Send chat completion request to Gemini.

    Args:
      system_prompt: System instructions
      user_prompt: User message
      model: Model identifier (e.g., "gemini-1.5-pro", "gemini-1.5-flash")
      temperature: Sampling temperature
      max_tokens: Maximum tokens in response
      **kwargs: Additional Gemini-specific options

    Returns:
      Generated text response
    """
    # Create model with system instruction if provided
    model_kwargs = {}
    if system_prompt:
      model_kwargs["system_instruction"] = system_prompt

    gen_model = genai.GenerativeModel(model, **model_kwargs)

    # Configure generation settings
    generation_config = genai.GenerationConfig(
      temperature=temperature,
      max_output_tokens=max_tokens,
    )

    response = gen_model.generate_content(
      user_prompt,
      generation_config=generation_config,
    )

    return response.text or ""


#fin
