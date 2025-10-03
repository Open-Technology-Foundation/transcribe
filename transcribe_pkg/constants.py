#!/usr/bin/env python3
"""Constants and configuration values for the transcribe package.

This module centralizes all magic numbers and configuration constants used
throughout the application, making them easy to find, update, and document.
"""

# OpenAI API Configuration

# Maximum context prompt length for OpenAI's Whisper API
# This is limited by the API's context window for the prompt parameter
MAX_CONTEXT_PROMPT_LENGTH = 896

# Default model names
DEFAULT_WHISPER_MODEL = "whisper-1"
DEFAULT_GPT_MODEL = "gpt-4o"
DEFAULT_SUMMARY_MODEL = "gpt-4o-mini"

# Temperature settings for API calls
# Lower values make output more deterministic
DEFAULT_TRANSCRIPTION_TEMPERATURE = 0.05
DEFAULT_PROCESSING_TEMPERATURE = 0.05
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0

# Token limits for API calls
DEFAULT_MAX_TOKENS = 4096
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 128000  # Current OpenAI limit for some models

# Text Processing Configuration

# Sample size for language detection (characters)
# 1000 characters is usually sufficient to identify language
LANGUAGE_DETECTION_SAMPLE_SIZE = 1000

# Default maximum chunk size for text processing (characters)
# Optimized for API token limits while maintaining context
DEFAULT_MAX_CHUNK_SIZE = 3000
MIN_CHUNK_SIZE = 100

# Overlap size between chunks to maintain context (characters)
DEFAULT_CHUNK_OVERLAP = 200

# Maximum sentence length in bytes for text splitting
# Helps prevent overly long sentences that break formatting
DEFAULT_MAX_SENTENCE_LENGTH = 3000

# Paragraph creation settings
DEFAULT_MIN_SENTENCES_PER_PARAGRAPH = 2
DEFAULT_MAX_SENTENCES_PER_PARAGRAPH = 8

# Audio Processing Configuration

# Default chunk length for audio splitting (milliseconds)
# 10 minutes per chunk - balances processing time and API limits
DEFAULT_CHUNK_LENGTH_MS = 600000

# Minimum and maximum chunk lengths (milliseconds)
MIN_CHUNK_LENGTH_MS = 1000  # 1 second
MAX_CHUNK_LENGTH_MS = 3600000  # 1 hour

# Maximum audio file size for Whisper API (bytes)
# OpenAI Whisper API limit is 25MB
MAX_AUDIO_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

# Parallel Processing Configuration

# Default number of parallel workers for transcription
DEFAULT_MAX_WORKERS = 1
MIN_WORKERS = 1

# Default number of parallel workers for post-processing
# None means use CPU count
DEFAULT_MAX_PARALLEL_WORKERS = None

# Content Analysis Configuration

# Minimum word count for content type detection
MIN_WORDS_FOR_ANALYSIS = 50

# Threshold ratios for content type detection
DIALOGUE_THRESHOLD_RATIO = 1 / 20  # dialogue_count / total_words
TECHNICAL_THRESHOLD_RATIO = 1 / 100  # technical_count / total_words
SPEECH_THRESHOLD_COUNT = 3  # min occurrences of speech indicators
LECTURE_THRESHOLD_COUNT = 3  # min occurrences of lecture indicators

# Cache Configuration

# Cache TTL (time to live) in seconds
DEFAULT_CACHE_TTL = 3600  # 1 hour

# Maximum number of cached items
MAX_CACHE_ITEMS = 1000

# Logging Configuration

# Default log levels
DEFAULT_LOG_LEVEL = "INFO"
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Progress bar update interval (seconds)
PROGRESS_BAR_UPDATE_INTERVAL = 0.1

# File Extensions

SUPPORTED_AUDIO_FORMATS = [
  ".mp3",
  ".mp4",
  ".mpeg",
  ".mpga",
  ".m4a",
  ".wav",
  ".webm",
]

SUPPORTED_SUBTITLE_FORMATS = ["srt", "vtt"]

# Exit Codes

EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_MISSING_ARGUMENT = 2
EXIT_INVALID_ARGUMENT = 22  # EINVAL
EXIT_INTERRUPT = 130  # 128 + SIGINT (2)

# Validation Constraints

# Maximum length for model names
MAX_MODEL_NAME_LENGTH = 100

# Valid language code pattern (ISO 639-1)
LANGUAGE_CODE_PATTERN = r"^[a-z]{2}(-[A-Z]{2})?$"

# API Rate Limiting

# Retry configuration for API calls
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff multiplier
INITIAL_RETRY_DELAY = 1  # seconds

__all__ = [
  # OpenAI API
  "MAX_CONTEXT_PROMPT_LENGTH",
  "DEFAULT_WHISPER_MODEL",
  "DEFAULT_GPT_MODEL",
  "DEFAULT_SUMMARY_MODEL",
  "DEFAULT_TRANSCRIPTION_TEMPERATURE",
  "DEFAULT_PROCESSING_TEMPERATURE",
  "MIN_TEMPERATURE",
  "MAX_TEMPERATURE",
  "DEFAULT_MAX_TOKENS",
  "MIN_MAX_TOKENS",
  "MAX_MAX_TOKENS",
  # Text Processing
  "LANGUAGE_DETECTION_SAMPLE_SIZE",
  "DEFAULT_MAX_CHUNK_SIZE",
  "MIN_CHUNK_SIZE",
  "DEFAULT_CHUNK_OVERLAP",
  "DEFAULT_MAX_SENTENCE_LENGTH",
  "DEFAULT_MIN_SENTENCES_PER_PARAGRAPH",
  "DEFAULT_MAX_SENTENCES_PER_PARAGRAPH",
  # Audio Processing
  "DEFAULT_CHUNK_LENGTH_MS",
  "MIN_CHUNK_LENGTH_MS",
  "MAX_CHUNK_LENGTH_MS",
  "MAX_AUDIO_FILE_SIZE",
  # Parallel Processing
  "DEFAULT_MAX_WORKERS",
  "MIN_WORKERS",
  "DEFAULT_MAX_PARALLEL_WORKERS",
  # Content Analysis
  "MIN_WORDS_FOR_ANALYSIS",
  "DIALOGUE_THRESHOLD_RATIO",
  "TECHNICAL_THRESHOLD_RATIO",
  "SPEECH_THRESHOLD_COUNT",
  "LECTURE_THRESHOLD_COUNT",
  # Cache
  "DEFAULT_CACHE_TTL",
  "MAX_CACHE_ITEMS",
  # Logging
  "DEFAULT_LOG_LEVEL",
  "VALID_LOG_LEVELS",
  "PROGRESS_BAR_UPDATE_INTERVAL",
  # File Extensions
  "SUPPORTED_AUDIO_FORMATS",
  "SUPPORTED_SUBTITLE_FORMATS",
  # Exit Codes
  "EXIT_SUCCESS",
  "EXIT_GENERAL_ERROR",
  "EXIT_MISSING_ARGUMENT",
  "EXIT_INVALID_ARGUMENT",
  "EXIT_INTERRUPT",
  # Validation
  "MAX_MODEL_NAME_LENGTH",
  "LANGUAGE_CODE_PATTERN",
  # Rate Limiting
  "MAX_RETRY_ATTEMPTS",
  "RETRY_BACKOFF_FACTOR",
  "INITIAL_RETRY_DELAY",
]

#fin
