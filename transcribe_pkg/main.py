#!/usr/bin/env python3
"""
Main entry points for transcribe CLI tools.
"""

from transcribe_pkg.cli.commands import (
  setup_transcript_command,
  setup_clean_transcript_command,
  setup_create_sentences_command,
  setup_language_codes_command
)

def transcribe_main():
  """Entry point for transcribe command."""
  transcribe = setup_transcript_command()
  transcribe()

def clean_transcript_main():
  """Entry point for clean-transcript command."""
  clean_transcript = setup_clean_transcript_command()
  clean_transcript()

def create_sentences_main():
  """Entry point for create-sentences command."""
  create_sentences = setup_create_sentences_command()
  create_sentences()

def language_codes_main():
  """Entry point for language-codes command."""
  language_codes = setup_language_codes_command()
  language_codes()

if __name__ == "__main__":
  # This can be used for testing or direct invocation
  transcribe_main()

#fin