#!/usr/bin/env python3
"""
Example usage of the transcribe package.

This module provides example code for using the transcribe package programmatically.
"""

import os
import logging
from transcribe_pkg.utils.logging_utils import setup_logging
from transcribe_pkg.utils.config import Config
from transcribe_pkg.core.transcriber import transcribe_audio_file
from transcribe_pkg.core.processor import process_transcript
from transcribe_pkg.utils.text_utils import create_sentences, create_paragraphs

def example_basic_transcription():
    """Example of basic audio transcription."""
    # Set up logging
    setup_logging(verbose=True)
    
    # Transcribe audio file
    result = transcribe_audio_file(
        audio_path="path/to/audio.mp3",
        output_file="output.txt",
        context="interviews,politics",
        model="gpt-4o"
    )
    
    print(f"Transcription complete with {len(result.split())} words")

def example_with_configuration():
    """Example using configuration."""
    # Create configuration
    config = Config()
    
    # Update configuration
    config.set('transcription.parallel', True)
    config.set('transcription.max_workers', 4)
    config.set('processing.max_chunk_size', 5000)
    
    # Use in transcription
    result = transcribe_audio_file(
        audio_path="path/to/audio.mp3",
        output_file="output.txt",
        context="science,physics",
        model=config.get('openai.models.completion'),
        parallel_processing=config.get('transcription.parallel'),
        max_workers=config.get('transcription.max_workers'),
        chunk_size=config.get('processing.max_chunk_size')
    )
    
    print(f"Transcription complete with {len(result.split())} words")

def example_clean_transcript():
    """Example of cleaning an existing transcript."""
    # Set up logging
    setup_logging(verbose=True)
    
    # Read existing transcript
    with open("path/to/transcript.txt", 'r') as f:
        transcript = f.read()
    
    # Process the transcript
    processed = process_transcript(
        input_text=transcript,
        model="gpt-4o",
        max_chunk_size=3000,
        temperature=0.1,
        context="medical,biology",
        language="en"
    )
    
    # Save processed transcript
    with open("path/to/cleaned_transcript.txt", 'w') as f:
        f.write(processed)
    
    print(f"Transcript cleaned with {len(processed.split())} words")

def example_non_english():
    """Example of transcribing non-English audio."""
    # Set up logging
    setup_logging(verbose=True)
    
    # Transcribe audio file
    result = transcribe_audio_file(
        audio_path="path/to/spanish_audio.mp3",
        output_file="spanish_output.txt",
        language="es",
        model="gpt-4o"
    )
    
    print(f"Spanish transcription complete with {len(result.split())} words")

def example_text_processing():
    """Example of text processing utilities."""
    # Set up logging
    setup_logging(verbose=True)
    
    # Read text
    with open("path/to/text.txt", 'r') as f:
        text = f.read()
    
    # Create sentences
    sentences = create_sentences(text, max_sentence_length=2000)
    print(f"Created {len(sentences)} sentences")
    
    # Create paragraphs
    paragraphs = create_paragraphs(
        text, 
        min_sentences=3, 
        max_sentences=5,
        max_sentence_length=2000
    )
    
    # Save paragraphs
    with open("path/to/paragraphs.txt", 'w') as f:
        f.write(paragraphs)
    
    print(f"Created paragraphs with {len(paragraphs.split('\\n\\n'))} paragraphs")

if __name__ == "__main__":
    print("This module contains example code and is not meant to be run directly.")
    print("Please see the docstrings for examples.")

#fin