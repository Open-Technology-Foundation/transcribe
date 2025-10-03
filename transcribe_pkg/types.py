#!/usr/bin/env python3
"""Type definitions and protocols for the transcribe package.

This module defines type aliases, TypedDict classes, and Protocol definitions
used throughout the transcribe package for better type safety and IDE support.
"""
from typing import Any, Protocol, TypedDict, runtime_checkable
import logging

# TypedDict Definitions for Structured Data

class WordTimestamp(TypedDict):
  """Word-level timestamp information from transcription."""
  word: str
  start: float
  end: float

class TranscriptionSegment(TypedDict):
  """Segment data from transcription with timestamps."""
  id: int
  start: float
  end: float
  text: str
  words: list[WordTimestamp]

class TranscriptionResult(TypedDict):
  """Complete transcription result with segments and timing information."""
  text: str
  segments: list[TranscriptionSegment]

# Protocol Definitions for Duck Typing

@runtime_checkable
class AudioProcessorProtocol(Protocol):
  """Protocol for audio processing implementations.

  Any class implementing this protocol must provide methods for splitting
  audio files into chunks and cleaning up temporary files.
  """

  def split_audio(self, audio_path: str, chunk_length_ms: int) -> list[str]:
    """Split audio file into chunks.

    Args:
      audio_path: Path to the audio file
      chunk_length_ms: Length of each chunk in milliseconds

    Returns:
      List of paths to audio chunk files

    Raises:
      FileNotFoundError: If audio file doesn't exist
      ValueError: If chunk_length_ms is invalid
    """
    ...

  def cleanup(self) -> None:
    """Clean up temporary files created during audio processing."""
    ...

@runtime_checkable
class APIClientProtocol(Protocol):
  """Protocol for API client implementations.

  Defines the interface for interacting with OpenAI API or compatible services.
  """

  def transcribe_audio(
    self,
    audio_file: str,
    model: str = "whisper-1",
    prompt: str = "",
    language: str | None = None,
    response_format: str = "text",
    temperature: float = 0.05
  ) -> str | dict[str, Any]:
    """Transcribe an audio file using the API.

    Args:
      audio_file: Path to the audio file
      model: Model to use for transcription
      prompt: Optional prompt to guide transcription
      language: ISO 639-1 language code
      response_format: Desired response format ('text' or 'json')
      temperature: Sampling temperature (0.0-1.0)

    Returns:
      Transcription text or structured result based on response_format

    Raises:
      APIError: If API request fails
      FileNotFoundError: If audio file doesn't exist
    """
    ...

  def chat_completion(
    self,
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 1000
  ) -> str:
    """Generate a chat completion using the API.

    Args:
      system_prompt: System message defining assistant behavior
      user_prompt: User message/query
      model: Model to use for completion
      temperature: Sampling temperature (0.0-1.0)
      max_tokens: Maximum tokens in response

    Returns:
      Generated completion text

    Raises:
      APIError: If API request fails
    """
    ...

@runtime_checkable
class LoggerProtocol(Protocol):
  """Protocol for logger implementations.

  Ensures compatibility with logging.Logger interface.
  """

  def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
    """Log debug message."""
    ...

  def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
    """Log info message."""
    ...

  def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
    """Log warning message."""
    ...

  def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
    """Log error message."""
    ...

@runtime_checkable
class CacheProtocol(Protocol):
  """Protocol for cache implementations."""

  def get(self, key: str) -> Any | None:
    """Retrieve value from cache.

    Args:
      key: Cache key

    Returns:
      Cached value or None if not found
    """
    ...

  def set(self, key: str, value: Any) -> None:
    """Store value in cache.

    Args:
      key: Cache key
      value: Value to cache
    """
    ...

  def clear(self) -> None:
    """Clear all cached values."""
    ...

# Type Aliases for Common Patterns

# Path type - can be string or pathlib.Path
PathLike = str

# Chunk data for parallel processing
ChunkData = tuple[int, str]  # (chunk_index, chunk_text)

# Processing result
ProcessingResult = tuple[int, str]  # (chunk_index, processed_text)

__all__ = [
  # TypedDict classes
  "WordTimestamp",
  "TranscriptionSegment",
  "TranscriptionResult",
  # Protocol classes
  "AudioProcessorProtocol",
  "APIClientProtocol",
  "LoggerProtocol",
  "CacheProtocol",
  # Type aliases
  "PathLike",
  "ChunkData",
  "ProcessingResult",
]

#fin
