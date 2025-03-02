"""
Transcribe Package - Audio transcription and post-processing toolkit.

This package provides a comprehensive set of tools for transcribing audio files
and processing the resulting text. It uses OpenAI's Whisper model for audio
transcription and GPT models for post-processing tasks.

Major components:
- Core transcription and processing functionality
- Command-line interface tools
- Utility functions for text processing, API interaction, and more
- Configuration management system

The package is designed to be modular, flexible, and easy to extend. It includes
error handling, retry logic, and logging to ensure reliable operation.

Usage:
    from transcribe_pkg.core.transcriber import transcribe_audio_file
    from transcribe_pkg.core.processor import process_transcript
    
    # Transcribe an audio file
    transcript = transcribe_audio_file("audio_file.mp3", output_file="transcript.txt")
    
    # Process a transcript
    processed_text = process_transcript(transcript, model="gpt-4o")
"""

# Package version
__version__ = "1.0.0"