#!/usr/bin/env python3
"""
Command line interfaces for the transcribe package.
"""

import os
import sys
import argparse
import signal
import logging

from transcribe_pkg.utils.logging_utils import setup_logging
from transcribe_pkg.core.transcriber import transcribe_audio_file
from transcribe_pkg.core.processor import process_transcript
from transcribe_pkg.utils.text_utils import create_sentences
from transcribe_pkg.utils.language_utils import get_language_name, display_all_languages

def signal_handler(sig, frame):
  """
  Handle Ctrl+C interruption
  """
  print('\033[0m^C\n')
  sys.exit(130)

def setup_transcript_command():
  """
  Set up the transcribe command.
  
  Returns:
    function: Main function for transcript command
  """
  def main():
    # Set up the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
      description='Transcribe audio files using OpenAI\'s Whisper API',
      epilog="Example: transcribe input.mp3 -c 'neuroscience,biology' -m gpt-4o -o output.txt"
    )
    parser.add_argument('audio_path',
      help='Path to the input audio file')
    
    parser.add_argument('-o', '--output',
      help='Output file name (def: input filename with .txt extension)')
    
    parser.add_argument('-O', '--output-to-stdout', action='store_true',
      help='Output the transcription to stdout; overrides -o (def: disabled)')
    
    parser.add_argument('-P', '--no-post-processing', action='store_true',
      help='Disable post-processing cleanups (def: enabled)')
    
    parser.add_argument('-l', '--chunk-length', type=int, default=600000,
      help='Length of audio chunks in milliseconds (def: 600000)')
    
    parser.add_argument('-L', '--input-language', default=None,
      help='Define the language used in the input audio (def: None))')
    
    parser.add_argument('-c', '--context', default='',
      help='Provide context for post-processing; eg, medical,legal,technical (def: '')')
    
    parser.add_argument('-m', '--model', default='gpt-4o',
      help='OpenAI LLModel to use for post-processing, eg, gpt-4o, gpt-4o-mini (def: gpt-4o)')
    
    parser.add_argument('-s', '--max-chunk-size', type=int,
      help='Maximum chunk size for post-processing (default: 3000)', default=3000)
    
    parser.add_argument('-t', '--temperature', type=float,
      help='Temperature for text generation in post-processing, 0.0 - 1.0 (default: 0.1)', default=0.1)
    
    parser.add_argument('-p', '--prompt', default='',
      help='Provide a prompt to guide the initial transcription')
      
    parser.add_argument('--parallel', action='store_true', default=True,
      help='Enable parallel processing for transcription (def: True)')
      
    parser.add_argument('--no-parallel', action='store_false', dest='parallel',
      help='Disable parallel processing for transcription')
      
    parser.add_argument('-w', '--workers', type=int, default=None,
      help='Number of worker threads for parallel processing (def: auto)')
    
    # Cache control options
    cache_group = parser.add_argument_group('Cache Options')
    cache_group.add_argument('--cache-dir',
      help='Directory to store cache files (def: system temp directory)')
    cache_group.add_argument('--cache-size', type=int, default=100,
      help='Maximum cache size in MB (def: 100)')
    cache_group.add_argument('--no-cache', action='store_true',
      help='Disable caching')
    cache_group.add_argument('--clear-cache', action='store_true',
      help='Clear cache before running')
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument('--config',
      help='Path to configuration file')
    
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
      help='Enable verbose output')
      
    parser.add_argument('-d', '--debug', default=False, action='store_true',
      help='Enable debug output')
    
    args = parser.parse_args()
    
    # Set up logging based on verbose/quiet options
    setup_logging(args.verbose, args.debug)
    
    # Set up cache if requested
    if not args.no_cache:
      from transcribe_pkg.utils.cache import get_cache
      
      # Set environment variables for cache
      if args.cache_dir:
        os.environ['TRANSCRIBE_CACHE_DIR'] = args.cache_dir
      os.environ['TRANSCRIBE_CACHE_SIZE_MB'] = str(args.cache_size)
      
      # Clear cache if requested
      if args.clear_cache:
        cache = get_cache()
        cache.clear()
        logging.info("Cache cleared")
    
    # Check if input file exists
    if not os.path.exists(args.audio_path):
      logging.error(f"Input file '{args.audio_path}' does not exist.")
      sys.exit(1)
    
    # Set default output filename if not specified
    if args.output_to_stdout:
      args.output = sys.stdout
    elif not args.output:
      args.output = os.path.splitext(args.audio_path)[0] + ".txt"
    
    logging.info(f"Starting transcription process for: {args.audio_path}")
    
    # Call the transcriber function
    transcribe_audio_file(
      args.audio_path, 
      args.chunk_length, 
      args.output, 
      args.context, 
      args.prompt, 
      args.model, 
      args.no_post_processing, 
      args.input_language, 
      args.temperature, 
      args.max_chunk_size,
      args.parallel,
      args.workers
    )
  
  return main

def setup_clean_transcript_command():
  """
  Set up the clean-transcript command.
  
  Returns:
    function: Main function for clean transcript command
  """
  def main():
    # Set up the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
      description="Fix and clean up transcripts using OpenAI API.",
      epilog='Example: clean-transcript raw_transcript.txt -c "neuroscience, free will" -m gpt-4o -o clean_transcript.txt'
    )
    parser.add_argument("input_file",
      help="Path to the raw text/transcript file")
      
    parser.add_argument('-L', '--input-language', default=None,
      help='Define the language of the text. If this is specified, then the text is translated into English (def: None))')
      
    parser.add_argument("-c", "--context", default=None,
      help="Domain-specific context for the transcript (default: none)")
      
    parser.add_argument("-m", "--model", default='gpt-4o',
      help=f"OpenAI model to use (default: gpt-4o)")
      
    parser.add_argument("-M", "--max-tokens", type=int, default=4096,
      help=f"Maximum tokens (default: 4096)")
      
    parser.add_argument("-s", "--max-chunk-size", type=int, default=3000,
      help=f"Maximum chunk size for processing (default: 3000)")
      
    parser.add_argument("-t", "--temperature", type=float, default=0.05,
      help=f"Temperature for text generation, 0.0 - 1.0 (default: 0.05)")
      
    parser.add_argument("-o", "--output",
      help="Output file path (default: stdout)")
    
    # Cache control options
    cache_group = parser.add_argument_group('Cache Options')
    cache_group.add_argument('--cache-dir',
      help='Directory to store cache files (def: system temp directory)')
    cache_group.add_argument('--cache-size', type=int, default=100,
      help='Maximum cache size in MB (def: 100)')
    cache_group.add_argument('--no-cache', action='store_true',
      help='Disable caching')
    cache_group.add_argument('--clear-cache', action='store_true',
      help='Clear cache before running')
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument('--config',
      help='Path to configuration file')
      
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
      help='Enable verbose output')
      
    parser.add_argument('-d', '--debug', default=False, action='store_true',
      help='Enable debug output')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose, args.debug)
    
    # Set up cache if requested
    if not args.no_cache:
      from transcribe_pkg.utils.cache import get_cache
      
      # Set environment variables for cache
      if args.cache_dir:
        os.environ['TRANSCRIBE_CACHE_DIR'] = args.cache_dir
      os.environ['TRANSCRIBE_CACHE_SIZE_MB'] = str(args.cache_size)
      
      # Clear cache if requested
      if args.clear_cache:
        cache = get_cache()
        cache.clear()
        logging.info("Cache cleared")
    
    try:
      with open(args.input_file, 'r') as file:
        input_text = file.read()
    except IOError as e:
      logging.error(f"Error reading input file: {str(e)}")
      sys.exit(1)
    
    generated_text = process_transcript(
      input_text,
      model=args.model,
      max_tokens=args.max_tokens,
      temperature=args.temperature,
      context=args.context,
      language=args.input_language,
      max_chunk_size=args.max_chunk_size,
    )
    
    if args.output:
      try:
        with open(args.output, 'w') as file:
          file.write(generated_text)
      except IOError as e:
        logging.error(f"Error writing to output file: {str(e)}")
        sys.exit(1)
    else:
      print(generated_text)
  
  return main

def setup_create_sentences_command():
  """
  Set up the create-sentences command.
  
  Returns:
    function: Main function for create sentences command
  """
  def main():
    # Set up the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
      description="Create logical sentences from unstructured text.",
      epilog='Example: create-sentences textile.txt -s 2000 -S "|"'
    )
    parser.add_argument("input_file",
      help="Path to the text file")
      
    parser.add_argument('-s', '--max-sentence-size', type=int,
      help='Max size of sentence in bytes (default: 3000)', default=3000)
      
    parser.add_argument('-S', '--suffix', default='',
      help='Suffix to append to each printed line (def: "")')
      
    parser.add_argument('-o', '--output',
      help='Output file path (default: stdout)')
      
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
      help='Enable verbose output')
      
    parser.add_argument('-d', '--debug', default=False, action='store_true',
      help='Enable debug output')
      
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose, args.debug)
    
    try:
      with open(args.input_file, 'r') as file:
        input_text = file.read()
    except IOError as e:
      logging.error(f"Error reading input file: {str(e)}")
      sys.exit(1)
    
    sentences = create_sentences(input_text.rstrip(), max_sentence_length=args.max_sentence_size)
    
    if args.output:
      try:
        with open(args.output, 'w') as file:
          for sentence in sentences:
            file.write(f"{sentence}{args.suffix}\n")
      except IOError as e:
        logging.error(f"Error writing to output file: {str(e)}")
        sys.exit(1)
    else:
      for sentence in sentences:
        print(f"{sentence}{args.suffix}")
  
  return main

def setup_language_codes_command():
  """
  Set up the language-codes command.
  
  Returns:
    function: Main function for language codes command
  """
  def main():
    parser = argparse.ArgumentParser(
      description="Display language codes and names."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
      "-l",
      "--list",
      action="store_true",
      help="List all two-letter language codes and their names."
    )
    group.add_argument(
      "-c",
      "--code",
      metavar="CODE",
      type=str,
      help="Get the language name for the specified two-letter code."
    )
    
    args = parser.parse_args()
    
    if args.list:
      display_all_languages()
    elif args.code:
      name = get_language_name(args.code)
      print(name)
    else:
      display_all_languages()
  
  return main

#fin