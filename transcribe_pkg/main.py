#!/usr/bin/env python3
"""
Main entry points for the transcription package command-line tools.

This module provides the entry points for the console scripts defined
in setup.py, including the main transcription tool and related utilities.
"""
import sys
import signal
import logging

# Setup signal handler for clean exit
def signal_handler(sig, frame):
    print('\033[0m^C\n')
    sys.exit(130)

signal.signal(signal.SIGINT, signal_handler)

# Import CLIs from the cli module
from transcribe_pkg.cli.commands import (
    transcribe_command,
    clean_transcript_command,
    create_sentences_command,
    language_codes_command
)

# Entry point functions that will be referenced in setup.py
def transcribe_main():
    """Entry point for the transcribe command."""
    return transcribe_command(sys.argv[1:])

def clean_transcript_main():
    """Entry point for the clean-transcript command."""
    try:
        return clean_transcript_command(sys.argv[1:])
    except KeyboardInterrupt:
        print('\033[0m^C\n')
        return 130

def create_sentences_main():
    """Entry point for the create-sentences command."""
    return create_sentences_command(sys.argv[1:])

def language_codes_main():
    """Entry point for the language-codes command."""
    return language_codes_command(sys.argv[1:])

if __name__ == "__main__":
    # Default to transcribe if run directly
    sys.exit(transcribe_main())

#fin