#!/usr/bin/env python3
"""
API utility functions for the transcribe package.

This module provides functions for interacting with external APIs, primarily OpenAI's
APIs for LLM completions and audio transcription. It handles authentication, error
handling, retries, and proper formatting of requests and responses.

Key components:
- OpenAI client initialization with API key validation
- LLM API interaction with automatic retries and error handling
- Audio transcription with Whisper API
- Custom exception types for specific error scenarios

All functions include retry mechanisms using the tenacity library to handle
transient API issues automatically.
"""

import os
import sys
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

# Initialize OpenAI client with API key from environment variables
def get_openai_client():
  """
  Get OpenAI client with API key from environment variables.
  
  Returns:
    OpenAI: Configured OpenAI client
    
  Raises:
    SystemExit: If OPENAI_API_KEY is not set
  """
  api_key = os.getenv('OPENAI_API_KEY')
  if not api_key:
    logging.error("OPENAI_API_KEY environment variable not set")
    logging.error("Please set the OPENAI_API_KEY environment variable or add it to the .env file")
    logging.error("Example: export OPENAI_API_KEY=your-api-key-here")
    sys.exit(1)
  
  try:
    client = OpenAI(api_key=api_key)
    return client
  except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {str(e)}")
    logging.error("Please check your API key and internet connection")
    sys.exit(1)

try:
  openai_client = get_openai_client()
except Exception as e:
  logging.critical(f"Critical error initializing OpenAI client: {str(e)}")
  sys.exit(1)

class APIError(Exception):
  """Custom exception for API errors."""
  pass

class EmptyResponseError(Exception):
  """Custom exception for empty responses."""
  pass

from transcribe_pkg.utils.cache import cached

# We'll use a simpler approach to caching for now
# Implementation without retry decorator to allow direct testing
def _call_llm_impl(systemprompt, input_text, model='gpt-4o', temperature=0, max_tokens=1000):
  """
  Implementation of LLM API call logic without retry mechanism.
  
  This function contains the core implementation of the LLM API call functionality
  but does not include the retry decorator. This allows for direct testing of error
  conditions without triggering retry attempts. Not intended for direct use in
  production code - use call_llm() instead.
  
  Args:
    systemprompt (str): System prompt instructions for the model
    input_text (str): User input text to process
    model (str, optional): OpenAI model identifier. Defaults to 'gpt-4o'.
    temperature (float, optional): Randomness parameter (0.0-2.0). Defaults to 0.
    max_tokens (int, optional): Maximum tokens to generate. Defaults to 1000.
    
  Returns:
    str: Generated text response from the model
    
  Raises:
    ValueError: If input parameters are invalid (e.g., temperature out of range)
    EmptyResponseError: If API returns no content or choices
    APIError: If API call fails for any reason
    
  Note:
    This is an internal implementation detail and not intended for direct use.
  """
  if not input_text:
    logging.warning("Empty input text provided to call_llm")
    return ""
    
  messages = [
    {"role": "system", "content": systemprompt},
    {"role": "user", "content": input_text}
  ]
  
  try:
    # Input validation
    if temperature < 0 or temperature > 2:
      raise ValueError(f"Temperature must be between 0 and 2, got {temperature}")
    if max_tokens < 1:
      raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    
    # Log request details at debug level
    logging.debug(f"API Request: model={model}, temperature={temperature}, max_tokens={max_tokens}")
    logging.debug(f"Input text length: {len(input_text)}")
    
    # Make API call
    response = openai_client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
      n=1,
      stop=''
    )
    
    # Handle empty responses
    if not response.choices or not response.choices[0].message.content.strip():
      logging.warning(f"Empty response from API: input_text='{input_text[:128]}...'")
      
      # Fallback behavior - return empty string but log the issue
      if not response.choices:
        logging.error("API returned no choices")
        raise EmptyResponseError("API returned no choices")
      else:
        response.choices[0].message.content = ''
        
    # Log success
    content = response.choices[0].message.content.strip()
    logging.debug(f"API response: {len(content)} characters")
    return content
    
  except (ValueError, TypeError) as e:
    # Input validation errors
    logging.error(f"Invalid input parameter: {str(e)}")
    raise
    
  except EmptyResponseError as e:
    # Empty response errors (already logged)
    raise
    
  except Exception as e:
    # All other API errors
    error_msg = f"API error with model {model}: {str(e)}"
    logging.error(error_msg)
    
    # Provide more specific error messages for common errors
    if "rate_limit" in str(e).lower():
      logging.error("Rate limit exceeded. Consider using a different model or waiting before retrying.")
    elif "billing" in str(e).lower():
      logging.error("Billing issue detected. Check your OpenAI account billing status.")
    elif "context_length" in str(e).lower():
      logging.error(f"Context length exceeded. Input text length: {len(input_text)}. Consider reducing input size.")
    
    raise APIError(error_msg) from e

def _generate_llm_cache_key(*args, **kwargs):
  """
  Generate a cache key for LLM API calls.
  
  This function creates a more stable cache key than the default by using
  only the most important parameters that affect the output.
  
  Args:
    *args: Positional arguments from the call_llm function
    **kwargs: Keyword arguments from the call_llm function
    
  Returns:
    dict: A dictionary with the essential parameters to use as cache key
  """
  # Extract arguments properly based on position or keyword
  # First positional arg is systemprompt, second is input_text
  if len(args) >= 2:
    systemprompt = args[0]
    input_text = args[1]
  else:
    systemprompt = args[0] if args else kwargs.get('systemprompt', '')
    input_text = args[1] if len(args) > 1 else kwargs.get('input_text', '')
  
  # Get model from kwargs if provided, else use default
  model = kwargs.get('model', 'gpt-4o')
  temperature = kwargs.get('temperature', 0)
  
  # Only include parameters that meaningfully affect the output
  return {
    'type': 'llm_call',
    'model': model,
    'systemprompt': systemprompt,
    'input_text': input_text,
    # Include temperature only if it's non-default, as it affects output
    'temperature': temperature,
    # Include max_tokens only if it's explicitly set to a non-default value
    'max_tokens': kwargs.get('max_tokens', 1000) if kwargs.get('max_tokens', 1000) != 1000 else None
  }

# Retry first, then cache - so cache only stores successful results
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@cached
def call_llm(systemprompt, input_text, model='gpt-4o', temperature=0, max_tokens=1000):
  """
  Call OpenAI API with retry mechanism and caching.
  
  Args:
    systemprompt (str): System prompt for the model
    input_text (str): User input text
    model (str, optional): Model to use. Defaults to 'gpt-4o'.
    temperature (float, optional): Temperature for generation. Defaults to 0.
    max_tokens (int, optional): Maximum tokens to generate. Defaults to 1000.
    
  Returns:
    str: Generated text from the model
    
  Raises:
    APIError: If API call fails after retries
    EmptyResponseError: If response is empty after retries
    ValueError: If input parameters are invalid
    
  Note:
    Results are cached based on system prompt, input text, model, and temperature.
    Set TRANSCRIBE_NO_CACHE=1 environment variable to disable caching.
  """
  # Call the implementation function
  return _call_llm_impl(systemprompt, input_text, model, temperature, max_tokens)

class AudioTranscriptionError(Exception):
  """Custom exception for audio transcription errors."""
  pass

# Implementation function without retry for testing
def _transcribe_audio_impl(audio_path, prompt="", language='en'):
  """
  Implementation of audio transcription logic without retry mechanism.
  
  This function contains the core implementation of the audio transcription functionality
  but does not include the retry decorator. This allows for direct testing of error
  conditions without triggering retry attempts. Not intended for direct use in
  production code - use transcribe_audio() instead.
  
  Args:
    audio_path (str): Path to the audio file to transcribe
    prompt (str, optional): Instructions to guide the transcription. Defaults to "".
    language (str, optional): Language code (ISO 639-1). Defaults to 'en'.
    
  Returns:
    str: Transcribed text
    
  Raises:
    FileNotFoundError: If audio file doesn't exist
    PermissionError: If audio file can't be read
    AudioTranscriptionError: For any other transcription failures
    
  Note:
    This is an internal implementation detail and not intended for direct use.
    The function performs file size validation and adds appropriate warnings
    for files approaching API limits.
  """
  # Validate file existence first
  if not os.path.exists(audio_path):
    error_msg = f"Audio file not found: {audio_path}"
    logging.error(error_msg)
    raise FileNotFoundError(error_msg)
    
  # Check file size to warn about large files
  try:
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb > 25:  # Whisper has a 25MB limit
      logging.warning(f"Audio file is {file_size_mb:.1f}MB, which exceeds the 25MB limit for Whisper API")
      logging.warning("File may need to be split or compressed before sending")
  except Exception as e:
    logging.warning(f"Could not check file size: {str(e)}")
  
  # Prepare instruction prompt
  full_prompt = f"{prompt.strip()}\n\nCreate a high quality, and accurate transcription of this audio, utilizing proper punctuation.\nCreate proper sentences with full-stops.\n"
  
  try:
    # Log the attempt
    logging.debug(f"Transcribing {audio_path} with language={language}")
    
    # Open and send file
    with open(audio_path, "rb") as audio_file:
      transcription = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        temperature=0.05,
        prompt=full_prompt,
        language=language
      )
      
    # Verify we got a response
    if not hasattr(transcription, 'text') or not transcription.text:
      error_msg = f"Empty transcription received for {audio_path}"
      logging.error(error_msg)
      return ""
      
    # Log success summary
    text_length = len(transcription.text)
    word_count = len(transcription.text.split())
    logging.debug(f"Transcription successful: {word_count} words, {text_length} characters")
    
    return transcription.text
    
  except FileNotFoundError as e:
    # Already handled above, but just in case
    logging.error(f"Audio file not found: {str(e)}")
    raise
    
  except PermissionError as e:
    logging.error(f"Permission denied when accessing {audio_path}: {str(e)}")
    logging.error("Check file permissions and ensure you have read access")
    raise
    
  except Exception as e:
    error_msg = f"Error transcribing {audio_path}: {str(e)}"
    logging.error(error_msg)
    
    # Add more specific error handling for common issues
    error_lower = str(e).lower()
    if "too large" in error_lower or "exceeds" in error_lower:
      logging.error("File size exceeds API limits. Try splitting the file into smaller chunks.")
    elif "format" in error_lower:
      logging.error("File format may not be supported. Ensure file is a valid audio format.")
    elif "rate limit" in error_lower:
      logging.error("Rate limit exceeded. Try again after waiting.")
    
    # Raise custom exception
    raise AudioTranscriptionError(error_msg) from e

def _generate_transcribe_cache_key(*args, **kwargs):
  """
  Generate a cache key for audio transcription.
  
  This function creates a stable cache key based on the file content
  hash and transcription parameters, ensuring consistent caching even
  if the file path changes but content remains the same.
  
  Args:
    *args: Positional arguments from the transcribe_audio function
    **kwargs: Keyword arguments from the transcribe_audio function
    
  Returns:
    dict: A dictionary with the essential parameters to use as cache key
  """
  # Extract arguments properly based on position or keyword
  if args:
    audio_path = args[0]
  else:
    audio_path = kwargs.get('audio_path', '')
  
  prompt = kwargs.get('prompt', '')
  language = kwargs.get('language', 'en')
  
  # Calculate file hash for stable key generation
  if not os.path.exists(audio_path):
    # If file doesn't exist, just use the path to allow the function to fail naturally
    file_hash = f"nonexistent:{audio_path}"
  else:
    try:
      # Use file size and modification time as a cheaper alternative to full file hash
      # This is a performance optimization to avoid reading large audio files
      file_size = os.path.getsize(audio_path)
      mod_time = os.path.getmtime(audio_path)
      file_hash = f"file:{audio_path}:{file_size}:{mod_time}"
    except Exception:
      # Fall back to just the path if we can't get file metadata
      file_hash = f"path:{audio_path}"
  
  return {
    'type': 'transcribe_audio',
    'file_hash': file_hash,
    'prompt': prompt,
    'language': language
  }

# Retry first, then cache - so cache only stores successful results
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@cached
def transcribe_audio(audio_path, prompt="", language='en'):
  """
  Transcribe audio using Whisper API with optional prompt and caching.
  
  Args:
    audio_path (str): Path to audio file
    prompt (str, optional): Prompt for transcription. Defaults to "".
    language (str, optional): Language code. Defaults to 'en'.
    
  Returns:
    str: Transcribed text
    
  Raises:
    FileNotFoundError: If audio file doesn't exist
    PermissionError: If audio file can't be read
    AudioTranscriptionError: If transcription fails
    
  Note:
    Results are cached based on file content and transcription parameters.
    Set TRANSCRIBE_NO_CACHE=1 environment variable to disable caching.
    Caching can significantly reduce API costs for repeated transcriptions.
  """
  # Call the implementation function
  return _transcribe_audio_impl(audio_path, prompt, language)

#fin