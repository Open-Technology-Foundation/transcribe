#!/usr/bin/env python3
"""
Utility modules for transcription package.

This package contains utility functions and classes for API communication,
audio processing, caching, configuration, text processing, and more.
"""

__all__ = [
    # API utilities
    "OpenAIClient",
    "get_openai_client",
    "call_llm",
    "transcribe_audio",
    "APIError",
    "APIRateLimitError",
    "APIAuthenticationError",
    "APIConnectionError",
    "EmptyResponseError",
    "AudioTranscriptionError",
    # Audio utilities
    "AudioProcessor",
    # Cache utilities
    "CacheManager",
    "cached",
    # Configuration
    "Config",
    "init_config",
    "get_config",
    # Prompts
    "PromptManager",
    # Text utilities
    "create_sentences",
    "create_paragraphs",
    "split_text_for_processing",
    "clean_transcript_text",
    "extract_key_topics",
    # Subtitle utilities
    "generate_srt",
    "generate_vtt",
    "format_timestamp_srt",
    "format_timestamp_vtt",
]

#fin
