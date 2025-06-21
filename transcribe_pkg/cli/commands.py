#!/usr/bin/env python3
"""
Command-line interface for the transcription package.

This module provides the command-line interface for the transcription package,
implementing the commands available through the console scripts.
"""
import os
import sys
import argparse
import logging
from typing import List, Optional, Any, Dict

from transcribe_pkg.utils.logging_utils import setup_logging
from transcribe_pkg.utils.config import Config
from transcribe_pkg.core.transcriber import Transcriber

def transcribe_command(args: Optional[List[str]] = None) -> int:
    """
    Implement the 'transcribe' command.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    epilog = """
Examples:
  # Basic transcription
  transcribe audio_file.mp3
  
  # Advanced transcription with processing options
  transcribe audio_file.mp3 -o transcript.txt --content-aware --parallel --cache
  
  # Transcription with context hints
  transcribe audio_file.mp3 -c "philosophy,science" -m gpt-4o
  
  # Subtitles generation
  transcribe audio_file.mp3 --srt
  
  # Use custom models
  transcribe audio_file.mp3 -W gpt-4o-mini-transcribe -m gpt-4o
    """
    
    parser = argparse.ArgumentParser(
        description='Transcribe audio files using OpenAI\'s Whisper API',
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input argument
    parser.add_argument('audio_path',
        help='Path to the input audio file')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('-o', '--output',
        help='Output file name (def: input filename with .txt extension)')
    output_group.add_argument('-O', '--output-to-stdout', action='store_true',
        help='Output the transcription to stdout; overrides -o (def: disabled)')
    
    # Processing options
    processing_group = parser.add_argument_group('Processing Options')
    processing_group.add_argument('-P', '--no-post-processing', action='store_true',
        help='Disable post-processing cleanups (def: enabled)')
    processing_group.add_argument('-l', '--chunk-length', type=int, default=600000,
        help='Length of audio chunks in milliseconds (def: 600000)')
    processing_group.add_argument('-L', '--input-language', default=None,
        help='Define the language used in the input audio (def: None)')
    processing_group.add_argument('-c', '--context', default='',
        help='Provide context for post-processing; eg, medical,legal,technical (def: \'\')')
    processing_group.add_argument('-W', '--transcribe-model', default='whisper-1',
        help='OpenAI Model to use for transcription, eg, gpt-4o-mini-transcribe (def:whisper-1)')
    processing_group.add_argument('-m', '--model', default='gpt-4o',
        help='OpenAI LLModel to use for post-processing, eg, gpt-4o, gpt-4o-mini (def: gpt-4o)')
    processing_group.add_argument('-s', '--max-chunk-size', type=int,
        help='Maximum chunk size for post-processing (default: 3000)', default=3000)
    processing_group.add_argument('-t', '--temperature', type=float,
        help='Temperature for text generation in post-processing, 0.0 - 1.0 (default: 0.05)', default=0.05)
    processing_group.add_argument('-p', '--prompt', default='',
        help='Provide a prompt to guide the initial transcription')
    processing_group.add_argument('-w', '--max-workers', type=int, default=1,
        help='Maximum number of parallel workers for transcription (default: 1)')
    
    # Post-processing advanced options
    post_processing_group = parser.add_argument_group('Post-Processing Advanced Options')
    post_processing_group.add_argument('--summary-model', default='gpt-4o-mini',
        help='OpenAI LLModel to use for context summarization (default: gpt-4o-mini)')
    post_processing_group.add_argument('--auto-context', action='store_true',
        help='Automatically determine the domain context from transcript content (def: disabled)')
    post_processing_group.add_argument('--raw', action='store_true',
        help='Save the raw transcript before post-processing (def: disabled)')
    post_processing_group.add_argument('--auto-language', action='store_true',
        help='Auto-detect language from transcript content (def: disabled)')
    post_processing_group.add_argument('--parallel', action='store_true',
        help='Enable parallel processing for large transcripts (def: disabled)')
    post_processing_group.add_argument('--max-parallel-workers', type=int, default=None,
        help='Maximum number of parallel workers (default: CPU count)')
    post_processing_group.add_argument('--cache', action='store_true',
        help='Enable caching of processing results for better performance (def: disabled)')
    post_processing_group.add_argument('--content-aware', action='store_true',
        help='Enable content-aware specialized processing (def: disabled)')
    post_processing_group.add_argument('--clear-cache', action='store_true',
        help='Clear any cached processing results before starting (def: disabled)')
    
    # Timestamp and subtitle options
    timestamp_group = parser.add_argument_group('Timestamp and Subtitle Options')
    timestamp_group.add_argument('-T', '--timestamps', action='store_true',
        help='Include timestamp information in the output (def: disabled)')
    timestamp_group.add_argument('--srt', action='store_true',
        help='Generate SRT subtitle file (enables timestamps automatically)')
    timestamp_group.add_argument('--vtt', action='store_true',
        help='Generate VTT subtitle file (enables timestamps automatically)')
    
    # Logging options
    logging_group = parser.add_argument_group('Logging Options')
    logging_group.add_argument('-v', '--verbose', default=False, action='store_true',
        help='Enable verbose output')
    logging_group.add_argument('-d', '--debug', default=False, action='store_true',
        help='Enable debug output')
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Set up logging
    logger = setup_logging(parsed_args.verbose, parsed_args.debug)
    
    # Check if input file exists
    if not os.path.exists(parsed_args.audio_path):
        logger.error(f"Input file '{parsed_args.audio_path}' does not exist.")
        return 1
    
    # Set default output filename if not specified
    if parsed_args.output_to_stdout:
        output = sys.stdout
    elif not parsed_args.output:
        output = os.path.splitext(parsed_args.audio_path)[0] + ".txt"
    else:
        output = parsed_args.output
    
    # Determine subtitle format
    subtitle_format = None
    if parsed_args.srt and parsed_args.vtt:
        logger.warning("Both SRT and VTT formats specified; defaulting to SRT")
        subtitle_format = "srt"
    elif parsed_args.srt:
        subtitle_format = "srt"
    elif parsed_args.vtt:
        subtitle_format = "vtt"
    
    # Enable timestamps if subtitle format is requested
    with_timestamps = parsed_args.timestamps or subtitle_format is not None
    if subtitle_format and not parsed_args.timestamps:
        logger.info("Enabling timestamps for subtitle generation")
    
    # Log the configuration
    logger.info(f"Starting transcription process for: {parsed_args.audio_path}")
    if subtitle_format:
        logger.info(f"Generating {subtitle_format.upper()} subtitle file")
    
    try:
        # Create transcriber
        transcriber = Transcriber(
            model=parsed_args.transcribe_model,
            language=parsed_args.input_language,
            temperature=0.05,  # Fixed temperature for transcription
            chunk_length_ms=parsed_args.chunk_length,
            logger=logger
        )
        
        # Determine max workers for transcription
        # If --parallel flag is set, use at least 2 workers even if max_workers is not set
        transcription_workers = parsed_args.max_workers
        if parsed_args.parallel and transcription_workers <= 1:
            # If parallel is requested but no specific worker count, use CPU count
            import multiprocessing
            transcription_workers = multiprocessing.cpu_count()
            logger.info(f"Parallel flag enabled, using {transcription_workers} workers for transcription")
        
        # Perform transcription
        result = transcriber.transcribe(
            audio_path=parsed_args.audio_path,
            prompt=parsed_args.prompt,
            with_timestamps=with_timestamps,
            max_workers=transcription_workers
        )
        
        # Handle raw transcript saving if requested
        if parsed_args.raw and output != sys.stdout and not with_timestamps:
            # Extract the text to save
            if isinstance(result, str):
                raw_text = result
            elif isinstance(result, dict) and "text" in result:
                raw_text = result.get("text", "")
            elif hasattr(result, "text"):
                raw_text = result.text
            else:
                raw_text = str(result)
                
            # Create a raw output path
            raw_output = f"{output}.raw"
            logger.info(f"Saving raw transcript to {raw_output}")
            try:
                with open(raw_output, "w") as f:
                    f.write(raw_text)
            except Exception as e:
                logger.error(f"Error saving raw transcript: {str(e)}")
        
        # Handle post-processing if enabled and not timestamped output
        if not parsed_args.no_post_processing and not with_timestamps:
            # Import required components
            from transcribe_pkg.utils.prompts import PromptManager
            from transcribe_pkg.core.processor import TranscriptProcessor
            
            # Get the transcript text
            if isinstance(result, str):
                text_to_process = result
            elif isinstance(result, dict) and "text" in result:
                text_to_process = result.get("text", "")
            elif hasattr(result, "text"):
                text_to_process = result.text
            else:
                text_to_process = str(result)
            
            # Create prompt manager for context extraction
            prompt_manager = PromptManager()
            
            # Handle context
            context = parsed_args.context
            if parsed_args.auto_context or (not context and not parsed_args.context):
                # Only use first chunk for context detection (1000 chars is sufficient)
                sample_text = text_to_process[:1000]
                
                # Create context string
                context = prompt_manager.extract_context(sample_text)
                logger.info(f"Auto-detected context: {context}")
            
            # Handle cache clearing if requested
            cache_manager = None
            if parsed_args.cache:
                from transcribe_pkg.utils.cache import CacheManager
                cache_manager = CacheManager(logger=logger)
                
                if parsed_args.clear_cache:
                    logger.info("Clearing processor cache")
                    cache_manager.clear()
            
            # Create processor instance
            processor = TranscriptProcessor(
                model=parsed_args.model,
                summary_model=parsed_args.summary_model,
                temperature=parsed_args.temperature,
                max_chunk_size=parsed_args.max_chunk_size,
                max_workers=parsed_args.max_parallel_workers,
                cache_enabled=parsed_args.cache,
                content_aware=parsed_args.content_aware,
                logger=logger,
                prompt_manager=prompt_manager
            )
            
            # Process the transcript
            logger.info(f"Post-processing transcript with model: {parsed_args.model}")
            logger.debug(f"Parallel processing: {parsed_args.parallel}")
            logger.debug(f"Content-aware processing: {parsed_args.content_aware}")
            logger.debug(f"Cache enabled: {parsed_args.cache}")
            
            processed_text = processor.process(
                text=text_to_process,
                context=context,
                language=None if parsed_args.auto_language else parsed_args.input_language,
                use_parallel=parsed_args.parallel,
                content_analysis=parsed_args.content_aware
            )
            
            # Update the result with processed text
            if isinstance(result, str):
                result = processed_text
            elif isinstance(result, dict):
                result["text"] = processed_text
            elif hasattr(result, "text"):
                result.text = processed_text
        
        # Handle different output formats
        if output == sys.stdout:
            if with_timestamps and not subtitle_format:
                # For timestamped output to stdout, provide a simplified format
                if isinstance(result, dict) and "segments" in result:
                    segments = result.get("segments", [])
                else:
                    # Handle object or unexpected format
                    segments = []
                    logger.warning("No segments found in transcription result for stdout output")
                
                for segment in segments:
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    text = segment.get("text", "").strip()
                    print(f"[{start_time:.2f} -> {end_time:.2f}] {text}")
            else:
                # Regular text output
                if isinstance(result, str):
                    print(result)
                elif isinstance(result, dict) and "text" in result:
                    print(result.get("text", ""))
                elif hasattr(result, "text"):
                    print(result.text)
                else:
                    print(str(result))
        else:
            # Write to file
            if with_timestamps and subtitle_format:
                # Generate subtitle file
                from transcribe_pkg.utils import subtitle_utils
                
                # Ensure result is in the expected format for subtitle generation
                if not isinstance(result, dict) or "segments" not in result:
                    # Convert result to dictionary if needed
                    if hasattr(result, "text") and hasattr(result, "segments"):
                        # Convert from object to dict
                        result_dict = {
                            "text": result.text,
                            "segments": []
                        }
                        for segment in result.segments:
                            segment_dict = {
                                "id": getattr(segment, "id", 0),
                                "start": getattr(segment, "start", 0),
                                "end": getattr(segment, "end", 0),
                                "text": getattr(segment, "text", ""),
                                "words": []
                            }
                            if hasattr(segment, "words"):
                                for word in segment.words:
                                    segment_dict["words"].append({
                                        "word": getattr(word, "word", ""),
                                        "start": getattr(word, "start", 0),
                                        "end": getattr(word, "end", 0)
                                    })
                            result_dict["segments"].append(segment_dict)
                        result = result_dict
                    else:
                        logger.error("Transcription result is not in expected format for subtitle generation")
                        return 1
                
                subtitle_path = subtitle_utils.save_subtitles(
                    result, 
                    f"{output}.{subtitle_format}", 
                    format_type=subtitle_format
                )
                if subtitle_path:
                    logger.info(f"Subtitle file created: {subtitle_path}")
                else:
                    logger.error("Failed to create subtitle file")
            else:
                # Write text transcript
                logger.info(f"Writing transcript to {output}")
                text_content = ""
                if isinstance(result, str):
                    text_content = result
                elif isinstance(result, dict) and "text" in result:
                    text_content = result.get("text", "")
                elif hasattr(result, "text"):
                    text_content = result.text
                else:
                    text_content = str(result)
                
                with open(output, "w") as f:
                    f.write(text_content)
        
        logger.info("Transcription complete")
        return 0
        
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        if parsed_args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

def clean_transcript_command(args: Optional[List[str]] = None) -> int:
    """
    Implement the 'clean-transcript' command.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Set up the ArgumentParser
    parser = argparse.ArgumentParser(
        description="Fix and clean up transcripts using OpenAI API.",
        epilog="Example: clean-transcript raw_transcript.txt -c \"neuroscience, free will\" -m gpt-4o -o clean_transcript.txt"
    )
    parser.add_argument("input_file",
        help="Path to the raw text/transcript file")
    parser.add_argument('-L', '--input-language', default=None,
        help='Define the language of the text. If specified, text is translated into English (def: None)')
    parser.add_argument("-c", "--context", default=None,
        help="Domain-specific context for the transcript (default: none)")
    parser.add_argument("-m", "--model", default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("-M", "--max-tokens", type=int, default=4096,
        help="Maximum tokens (default: 4096)")
    parser.add_argument("-s", "--max-chunk-size", type=int, default=3000,
        help="Maximum chunk size for processing (default: 3000)")
    parser.add_argument("-t", "--temperature", type=float, default=0.05,
        help="Temperature for text generation, 0.0 - 1.0 (default: 0.05)")
    parser.add_argument("-o", "--output",
        help="Output file path (default: stdout)")
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
        help='Enable verbose output')
    parser.add_argument('-d', '--debug', default=False, action='store_true',
        help='Enable debug output')

    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Set up logging
    logger = setup_logging(parsed_args.verbose, parsed_args.debug)
    
    # Import here to avoid circular imports
    from transcribe_pkg.core.processor import TranscriptProcessor
    
    try:
        # Read input file
        logger.info(f"Reading input file: {parsed_args.input_file}")
        try:
            with open(parsed_args.input_file, 'r') as file:
                input_text = file.read()
        except IOError as e:
            logger.error(f"Error reading input file: {str(e)}")
            return 1
            
        # Create processor
        processor = TranscriptProcessor(
            model=parsed_args.model,
            summary_model="gpt-4o-mini",  # Use a smaller model for summaries
            temperature=parsed_args.temperature,
            max_tokens=parsed_args.max_tokens,
            max_chunk_size=parsed_args.max_chunk_size,
            logger=logger
        )
        
        # Process the transcript
        logger.info(f"Processing transcript with model: {parsed_args.model}")
        generated_text = processor.process(
            text=input_text,
            context=parsed_args.context or "",
            language=parsed_args.input_language
        )
        
        # Output the result
        if parsed_args.output:
            logger.info(f"Writing output to file: {parsed_args.output}")
            try:
                with open(parsed_args.output, 'w') as file:
                    file.write(generated_text)
            except IOError as e:
                logger.error(f"Error writing to output file: {str(e)}")
                return 1
        else:
            # Write to stdout
            print(generated_text)
            
        logger.info("Transcript processing complete")
        return 0
        
    except Exception as e:
        logger.error(f"Error during transcript processing: {str(e)}")
        if parsed_args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

def create_sentences_command(args: Optional[List[str]] = None) -> int:
    """
    Implement the 'create-sentences' command.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Create logical sentences and paragraphs from unstructured text."
    )
    parser.add_argument("input_file", help="Path to the text file")
    parser.add_argument("-o", "--output", 
                     help="Output file (default: stdout)")
    parser.add_argument("-p", "--paragraphs", action="store_true",
                     help="Create paragraphs instead of sentences (default: sentences)")
    parser.add_argument("-m", "--min-sentences", type=int, default=2,
                     help="Minimum sentences per paragraph (default: 2)")
    parser.add_argument("-M", "--max-sentences", type=int, default=8,
                     help="Maximum sentences per paragraph (default: 8)")
    parser.add_argument("-s", "--max-sentence-length", type=int, default=3000,
                     help="Maximum sentence length in bytes (default: 3000)")
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                     help='Enable verbose output')
    parser.add_argument('-d', '--debug', default=False, action='store_true',
                     help='Enable debug output')
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Set up logging
    logger = setup_logging(parsed_args.verbose, parsed_args.debug)
    
    # Import text utilities
    from transcribe_pkg.utils.text_utils import create_sentences, create_paragraphs
    
    try:
        # Read input file
        logger.info(f"Reading input file: {parsed_args.input_file}")
        try:
            with open(parsed_args.input_file, 'r') as file:
                input_text = file.read()
        except IOError as e:
            logger.error(f"Error reading input file: {str(e)}")
            return 1
        
        # Process the text
        if parsed_args.paragraphs:
            logger.info(f"Creating paragraphs with min_sentences={parsed_args.min_sentences}, "
                        f"max_sentences={parsed_args.max_sentences}")
            output_text = create_paragraphs(
                input_text,
                min_sentences=parsed_args.min_sentences,
                max_sentences=parsed_args.max_sentences,
                max_sentence_length=parsed_args.max_sentence_length
            )
        else:
            logger.info("Creating sentences")
            sentences = create_sentences(
                input_text,
                max_sentence_length=parsed_args.max_sentence_length
            )
            output_text = '\n'.join(sentences)
        
        # Output the result
        if parsed_args.output:
            logger.info(f"Writing output to file: {parsed_args.output}")
            try:
                with open(parsed_args.output, 'w') as file:
                    file.write(output_text)
            except IOError as e:
                logger.error(f"Error writing to output file: {str(e)}")
                return 1
        else:
            # Write to stdout
            print(output_text)
        
        logger.info("Text processing complete")
        return 0
        
    except Exception as e:
        logger.error(f"Error during text processing: {str(e)}")
        if parsed_args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

def language_codes_command(args: Optional[List[str]] = None) -> int:
    """
    Implement the 'language-codes' command.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="List and lookup language codes."
    )
    parser.add_argument("query", nargs="?", help="Language name or code to look up")
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                      help='Enable verbose output')
    parser.add_argument('-d', '--debug', default=False, action='store_true',
                      help='Enable debug output')
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Set up logging
    logger = setup_logging(parsed_args.verbose, parsed_args.debug)
    
    # Common language codes used with Whisper API
    # This is a simplified list for basic functionality
    LANGUAGE_CODES = {
        "af": "Afrikaans",
        "ar": "Arabic",
        "hy": "Armenian",
        "az": "Azerbaijani",
        "be": "Belarusian",
        "bs": "Bosnian",
        "bg": "Bulgarian",
        "ca": "Catalan",
        "zh": "Chinese",
        "hr": "Croatian",
        "cs": "Czech",
        "da": "Danish",
        "nl": "Dutch",
        "en": "English",
        "et": "Estonian",
        "fi": "Finnish",
        "fr": "French",
        "gl": "Galician",
        "de": "German",
        "el": "Greek",
        "he": "Hebrew",
        "hi": "Hindi",
        "hu": "Hungarian",
        "is": "Icelandic",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "kn": "Kannada",
        "kk": "Kazakh",
        "ko": "Korean",
        "lv": "Latvian",
        "lt": "Lithuanian",
        "mk": "Macedonian",
        "ms": "Malay",
        "mr": "Marathi",
        "mi": "Maori",
        "ne": "Nepali",
        "no": "Norwegian",
        "fa": "Persian",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sr": "Serbian",
        "sk": "Slovak",
        "sl": "Slovenian",
        "es": "Spanish",
        "sw": "Swahili",
        "sv": "Swedish",
        "tl": "Tagalog",
        "ta": "Tamil",
        "th": "Thai",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "vi": "Vietnamese",
        "cy": "Welsh",
    }
    
    # Function to search by code or name
    def find_language(query):
        query = query.lower()
        results = []
        
        # Direct match by code
        if query in LANGUAGE_CODES:
            results.append((query, LANGUAGE_CODES[query]))
            return results
            
        # Search by name
        for code, name in LANGUAGE_CODES.items():
            if query in name.lower():
                results.append((code, name))
                
        return results
    
    try:
        if parsed_args.query:
            # Search for a specific language
            results = find_language(parsed_args.query)
            if results:
                print(f"Found {len(results)} language matches:")
                for code, name in results:
                    print(f"  {code}: {name}")
            else:
                print(f"No languages found matching '{parsed_args.query}'")
        else:
            # List all languages
            print("Available language codes:")
            for code in sorted(LANGUAGE_CODES.keys()):
                print(f"  {code}: {LANGUAGE_CODES[code]}")
                
        return 0
        
    except Exception as e:
        logger.error(f"Error during language code lookup: {str(e)}")
        if parsed_args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

#fin