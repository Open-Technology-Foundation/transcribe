#!/usr/bin/env python
import os
import sys
import openai
from pydub import AudioSegment
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import tempfile
import argparse
from dotenv import load_dotenv
from clean_transcript import process_transcript
from tenacity import retry, stop_after_attempt, wait_exponential
from language_codes import get_language_name

# Load environment variables
load_dotenv()

import signal
def signal_handler(sig, frame):
  print('\033[0m^C\n')
  sys.exit(130)

import logging
import colorlog
def setup_logging(verbose, debug=False):
  """Set up logging configuration with color."""
  logger = logging.getLogger()
  handler = colorlog.StreamHandler()
  if verbose or debug:
    if debug:
      logger.setLevel(logging.DEBUG)
    else:
      logger.setLevel(logging.INFO)
    datefmt="%H:%M:%S"
    logformat=f"%(log_color)s%(asctime)s:%(module)s:%(levelname)s: %(message)s"
  else:
    logger.setLevel(logging.ERROR)
    datefmt=None
    logformat=f"%(log_color)s%(module)s:%(levelname)s: %(message)s"
  formatter = colorlog.ColoredFormatter(
    logformat,
    datefmt=datefmt,
    reset=True,
    log_colors={ 'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white' },
    secondary_log_colors={},
    style='%'
  )
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  return logger

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_LLM(systemprompt, input_text, model='gpt-4o', temperature=0, max_tokens=4000):
  messages = [
    {"role": "system", "content": systemprompt},
    {"role": "user", "content": input_text}
  ]
  try:
    response = openai.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
    )
    if not response.choices or not response.choices[0].message.content.strip():
      logging.warning(f"Empty response from API: input_text='{input_text[:128]}...'")
      response.choices[0].message.content = ''
    return response.choices[0].message.content.strip()
  except Exception as e:
    logging.error(f"API error: {str(e)}")
    sys.exit(1)

# CONTEXT --------------------------------------------------------------------------
def create_field_context_string(input_text, model='gpt-4o-mini'):
  systemprompt="""You are a knowledgeable academic classifier specializing in categorizing content across academic disciplines, scientific fields, and cultural domains. Analyze the provided text and identify the most relevant fields it belongs to.

Provide exactly 3-5 primary fields, ordered by relevance, as a simple comma-separated list.

Format: field1,field2,field3[,field4][,field5]

Example outputs:
- economics,sociology,political science
- physics,astronomy,mathematics,engineering
- literature,cultural studies,history,sociology
  """

  """
You are an expert in determining what fields of culture or science texts are talking about.

You return 3-5 fields separated by commas, in order of relevance.

Eg, \"anthropology,politics,science,pop culture\"

Make no other commentary or preamble.
  """
  return call_LLM(systemprompt, input_text, model, 0, 100)

# LANGUAGE -------------------------------------------------------------------------
def determine_language(input_text, model='gpt-4o-mini'):
  systemprompt="""
You are an expert in determining what language a text is written in.

You return one two-character language code that corresponds to the language of the text; eg: en, id, ko, zh, ja.

Make no other commentary or preamble. Just output the language code.
"""
  lang=call_LLM(systemprompt, input_text, model, 0, 50)
  if 'Unknown' in get_language_name(lang):
    lang = 'en'
  return lang

def split_audio(audio_path, chunk_length_ms=600000):
  """Split audio into chunks."""
  logging.info(f"Loading audio file: {audio_path}")
  audio = AudioSegment.from_mp3(audio_path)
  total_length_ms = len(audio)
  chunks = []
  temp_dir = tempfile.mkdtemp()
  logging.info(f"Splitting audio into {chunk_length_ms/1000}-second chunks")
  for i in tqdm(range(0, total_length_ms, chunk_length_ms), desc="Splitting audio", disable=not logging.getLogger().isEnabledFor(logging.INFO)):
    chunk = audio[i:i+chunk_length_ms]
    chunk_path = os.path.join(temp_dir, f"chunk_{i//chunk_length_ms}.mp3")
    with open(chunk_path, 'wb') as f:
      chunk.export(f, format="mp3")
    chunks.append(chunk_path)
  logging.info(f"Audio split into {len(chunks)} chunks")
  return chunks

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def transcribe_audio(audio_path, prompt="", language='en', with_timestamps=False, model="whisper-1"):
  """Transcribe audio using Whisper API with optional prompt.
  
  Args:
      audio_path: Path to the audio file
      prompt: Optional context prompt to guide transcription
      language: Language code (e.g., 'en', 'fr')
      with_timestamps: If True, return word-level timestamp information
      model: The OpenAI model to use for transcription (e.g., 'whisper-1', 'gpt-4o-mini-transcribe')
      
  Returns:
      If with_timestamps=False: The transcribed text as a string
      If with_timestamps=True: A dictionary containing the text and timestamp information
  """
  prompt += f"{prompt.strip()}\n\nCreate a high quality, and accurate transcription of this audio, utilizing proper punctuation.\nCreate proper sentences with full-stops.\n"
  
  response_format = "verbose_json" if with_timestamps else "text"
  
  try:
    with open(audio_path, "rb") as audio_file:
      transcription = openai.audio.transcriptions.create(
        model=model,
        file=audio_file,
        temperature=0.05,
        prompt=prompt,
        language=language,
        response_format=response_format
      )
    
    if with_timestamps:
      # Convert the OpenAI response to a dictionary we can work with
      # Handle both object response (older API) and dict response (newer API)
      if hasattr(transcription, 'text'):
        # Object response style
        result = {
          "text": transcription.text,
          "segments": []
        }
        
        # Extract segments from the verbose_json response
        for segment in transcription.segments:
          segment_dict = {
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": []
          }
          
          # Extract word-level timestamps if available
          if hasattr(segment, 'words'):
            for word in segment.words:
              segment_dict["words"].append({
                "word": word.word,
                "start": word.start,
                "end": word.end
              })
              
          result["segments"].append(segment_dict)
      else:
        # Dict response style
        result = {
          "text": transcription.get("text", ""),
          "segments": []
        }
        
        # Extract segments from the verbose_json response
        for segment in transcription.get("segments", []):
          segment_dict = {
            "id": segment.get("id", 0),
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "text": segment.get("text", ""),
            "words": []
          }
          
          # Extract word-level timestamps if available
          for word in segment.get("words", []):
            segment_dict["words"].append({
              "word": word.get("word", ""),
              "start": word.get("start", 0),
              "end": word.get("end", 0)
            })
              
          result["segments"].append(segment_dict)
      
      return result
    else:
      # Handle both object response and string response
      if isinstance(transcription, str):
        return transcription
      elif hasattr(transcription, 'text'):
        return transcription.text
      else:
        return transcription.get("text", "")
  except Exception as e:
    logging.error(f"Error transcribing {audio_path}: {str(e)}")
    import traceback
    logging.debug(traceback.format_exc())
    return "" if not with_timestamps else {"text": "", "segments": []}

from create_sentences import create_sentences
def create_paragraphs(text, *, min_sentences=3, max_sentences=8, max_sentence_length=3000):
  """Create logical paragraphs from the text."""
  # Download necessary NLTK data
  SENTENCE_ENDINGS = {'.', '!', '?'}
  nltk.download('punkt', quiet=True)

  sentences = create_sentences(text, max_sentence_length=max_sentence_length)

  paragraphs = []
  current_paragraph = []
  for sentence in sentences:
    current_paragraph.append(sentence)
    if len(current_paragraph) >= min_sentences and (
        len(current_paragraph) >= max_sentences or
        sentence[-1] in SENTENCE_ENDINGS):
      paragraphs.append(' '.join(current_paragraph))
      current_paragraph = []
  if current_paragraph:
    paragraphs.append(' '.join(current_paragraph))
  return '\n\n'.join(paragraphs)

def transcribe_chunks(chunk_paths, Prompt='', language='en', with_timestamps=False, model="whisper-1"):
  """Transcribe audio chunks sequentially, updating the prompt each time.
  
  Args:
      chunk_paths: List of paths to audio chunks
      Prompt: Initial prompt to guide transcription
      language: Language code (e.g., 'en', 'fr')
      with_timestamps: If True, include timestamp information in output
      model: The OpenAI model to use for transcription (e.g., 'whisper-1', 'gpt-4o-mini-transcribe')
      
  Returns:
      If with_timestamps=False: List of transcribed text chunks
      If with_timestamps=True: Dictionary with 'text' and 'segments' containing 
                              all segments with timestamps adjusted for chunk position
  """
  prompt = Prompt[-896:]  # 224 tokens * 4 = 896
  transcripts = []
  
  # For timestamp mode, we need to track the total duration
  total_duration = 0
  all_segments = []
  
  for chunk_index, chunk_path in enumerate(tqdm(chunk_paths, desc="Transcribing chunks", disable=not logging.getLogger().isEnabledFor(logging.INFO))):
    try:
      transcript_result = transcribe_audio(chunk_path, prompt, language, with_timestamps, model)
      
      if with_timestamps:
        # Handle empty results
        if not transcript_result or not transcript_result.get("text"):
          logging.warning(f"Empty transcript for chunk {chunk_index}. Skipping.")
          continue
          
        # Extract text for prompt continuation
        transcript_text = transcript_result.get("text", "")
        
        # Adjust timestamps for this chunk's position in the overall audio
        segments = transcript_result.get("segments", [])
        for segment in segments:
          segment["start"] += total_duration
          segment["end"] += total_duration
          all_segments.append(segment)
        
        # Keep track of text chunks for regular transcription output
        transcripts.append(transcript_text)
        
        # Update the prompt with just the text
        prompt = f"{Prompt}\n{transcript_text}"
        
        # Update the total duration for the next chunk
        chunk_duration = segments[-1]["end"] if segments else 0
        total_duration += chunk_duration
      else:
        # Regular text-only mode
        # Handle empty results
        if not transcript_result:
          logging.warning(f"Empty transcript for chunk {chunk_index}. Skipping.")
          transcript_result = ""
          
        transcripts.append(transcript_result)
        prompt = f"{Prompt}\n{transcript_result}"
      
      prompt = prompt[-896:]
    except Exception as e:
      logging.error(f"Error in transcription for chunk {chunk_index}: {str(e)}")
      import traceback
      logging.debug(traceback.format_exc())
      
      # Add an empty transcript and continue with the next chunk
      if with_timestamps:
        transcripts.append("")
      else:
        transcripts.append("")
  
  if with_timestamps:
    # Handle case where all chunks failed
    if not transcripts:
      logging.warning("All chunks failed to transcribe. Returning empty result.")
      return {"text": "", "segments": []}
      
    return {
      "text": " ".join(transcripts),
      "segments": all_segments
    }
  else:
    return transcripts

# Import subtitle utility functions
from subtitle_utils import save_subtitles

#-----------------------------------------------------------------------------------------------------
def main(audio_path, chunk_length_ms, output_file, context, prompt, model, no_post_processing, 
         language, temperature, chunk_size, with_timestamps=False, subtitle_format=None, transcribe_model="whisper-1"):
  """
  Main function for transcribing audio files.
  
  Args:
      audio_path: Path to the audio file
      chunk_length_ms: Length of audio chunks in milliseconds
      output_file: Path to save the transcript
      context: Context hint for post-processing
      prompt: Initial prompt for transcription
      model: LLM model to use for post-processing
      no_post_processing: Skip post-processing if True
      language: Language code for transcription
      temperature: Temperature for text generation
      chunk_size: Maximum chunk size for post-processing
      with_timestamps: Include timestamp information if True
      subtitle_format: Output subtitle format (srt, vtt, or None)
      transcribe_model: OpenAI Model to use for transcription (default: 'whisper-1')
  """
  chunk_paths = []
  try:
    if not language and prompt != '':
      logging.info("Determining language from prompt...")
      language = determine_language(prompt, 'gpt-4o')
    else:
      language = 'en'
    logging.info(f"{language=}")
    
    # Enable timestamps if subtitle format is requested
    if subtitle_format and not with_timestamps:
      with_timestamps = True
      logging.info("Enabling timestamps for subtitle generation")

    transcript_result = None
    # Handle stdout case specially to ensure we always transcribe with timestamps if needed
    if output_file == sys.stdout and with_timestamps:
      # Always transcribe with timestamps for stdout if requested
      chunk_paths = split_audio(audio_path, chunk_length_ms)
      logging.info("Starting transcription process")
      transcript_result = transcribe_chunks(chunk_paths, prompt, language, with_timestamps=True, model=transcribe_model)
      transcript = transcript_result["text"]
    else:
      # Normal file-based workflow
      raw_path = f'{output_file}.raw' if output_file != sys.stdout else None
      
      if raw_path and not os.path.exists(raw_path):
        # Split audio into chunks
        chunk_paths = split_audio(audio_path, chunk_length_ms)
        
        # Transcribe each chunk
        logging.info("Starting transcription process")
        transcript_result = transcribe_chunks(chunk_paths, prompt, language, with_timestamps, model=transcribe_model)
        
        if with_timestamps:
          # Store the raw transcript text
          transcript = transcript_result["text"]
          
          # For timestamped output, also save raw JSON for potential reuse
          if output_file != sys.stdout:
            import json
            logging.info(f"Writing raw transcript with timestamps to {output_file}.json")
            with open(f'{output_file}.json', 'w') as f:
              json.dump(transcript_result, f, indent=2)
        else:
          # Combine all transcript chunks for text-only output
          logging.info("Combining transcripts")
          transcript = " ".join(transcript_result)
        
        # Save raw text transcript
        if output_file != sys.stdout:
          # Create paragraphs for text output
          logging.info("Creating paragraphs")
          formatted_transcript = create_paragraphs(transcript, min_sentences=2, max_sentence_length=chunk_size)
          logging.info(f"Writing raw transcript to {raw_path}")
          with open(raw_path, 'w') as f:
            f.write(formatted_transcript)
      elif raw_path:
        # If raw transcript exists and we need timestamps but don't have them,
        # we have to re-transcribe to get timestamps
        if with_timestamps:
          logging.warning("Raw transcript exists but timestamps required. Re-transcribing audio.")
          chunk_paths = split_audio(audio_path, chunk_length_ms)
          transcript_result = transcribe_chunks(chunk_paths, prompt, language, with_timestamps=True, model=transcribe_model)
          transcript = transcript_result["text"]
        else:
          # Otherwise, just read the existing raw transcript
          logging.info(f"Reading raw transcript {raw_path}")
          transcripts = []
          current_chunk = ""
          with open(raw_path, 'r') as file:
            for line in file:
              if len(current_chunk) + len(line) <= chunk_size:
                current_chunk += line
              else:
                if current_chunk:
                  transcripts.append(current_chunk.strip())
                current_chunk = line
            if current_chunk:
              transcripts.append(current_chunk.strip())
            transcript = " ".join(transcripts)
            logging.info(f'{transcript[:80]}=')

    # Post-process the full transcript (text only)
    if not with_timestamps and not no_post_processing:
      if not context:
        logging.info("Creating field context")
        # Use the first chunk of transcription for context detection
        first_chunk = transcript_result[0] if isinstance(transcript_result, list) else transcript[:1000]
        context = create_field_context_string(first_chunk)
        logging.info(f"{context=}")

      logging.info("Post-processing transcript")
      transcript = process_transcript(transcript,
            model=model, max_tokens=4096, temperature=temperature,
            context=f"{prompt} {context}",
            language=language,
            max_chunk_size=chunk_size
            )
    elif not with_timestamps:
      # Create paragraphs for text-only output without post-processing
      logging.info("Creating paragraphs")
      transcript = create_paragraphs(transcript, min_sentences=2, max_sentence_length=chunk_size)

    # Define transcript variable if it doesn't exist yet
    if with_timestamps and not 'transcript' in locals():
      transcript = transcript_result["text"]
      
    # Generate subtitle file if requested
    if with_timestamps and subtitle_format:
      logging.info(f"Generating {subtitle_format} subtitles")
      subtitle_path = save_subtitles(transcript_result, 
                                    output_file + f".{subtitle_format}", 
                                    format_type=subtitle_format)
      if subtitle_path:
        logging.info(f"Subtitle file created: {subtitle_path}")
      else:
        logging.error("Failed to create subtitle file")

    # Write full transcript to file or stdout
    if output_file == sys.stdout:
      logging.info("Writing full transcript to stdout")
      if with_timestamps and not subtitle_format:
        # For timestamped output to stdout, provide a simplified format
        segments = transcript_result.get("segments", [])
        for segment in segments:
          start_time = segment.get("start", 0)
          end_time = segment.get("end", 0)
          text = segment.get("text", "").strip()
          print(f"[{start_time:.2f} -> {end_time:.2f}] {text}")
      else:
        # Regular text output
        print(transcript)
    elif not with_timestamps or not subtitle_format:
      # Write full text transcript to file
      logging.info(f"Writing full transcript to {output_file}")
      with open(output_file, "w") as f:
        f.write(transcript)

    logging.info(f"Transcription complete.")
    if not with_timestamps or not subtitle_format:
      logging.info(f"Total words transcribed: {len(transcript.split())}")

  except Exception as e:
    logging.error(f"{str(e)}")
    import traceback
    logging.debug(traceback.format_exc())
  finally:
    # Clean up temporary files
    for chunk_path in chunk_paths:
      if os.path.exists(chunk_path):
        os.remove(chunk_path)
    logging.info("Temporary files cleaned up")

#----------------------------------------------------------------------------------------
if __name__ == "__main__":
  # Set up the signal handler
  signal.signal(signal.SIGINT, signal_handler)

  parser = argparse.ArgumentParser(description='Transcribe audio files using OpenAI\'s Whisper API')
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
      help='OpenAI Model to use for transcription, eg, gpt-4o-mini-transcribe, whisper-1 (def:whisper-1)')
  processing_group.add_argument('-m', '--model', default='gpt-4o',
      help='OpenAI LLModel to use for post-processing, eg, gpt-4o, gpt-4o-mini (def: gpt-4o)')
  processing_group.add_argument('-s', '--max-chunk-size', type=int,
      help='Maximum chunk size for post-processing (default: 3000)', default=3000)
  processing_group.add_argument('-t', '--temperature', type=float,
      help='Temperature for text generation in post-processing, 0.0 - 1.0 (default: 0.1)', default=0.1)
  processing_group.add_argument('-p', '--prompt', default='',
      help='Provide a prompt to guide the initial transcription')

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

  args = parser.parse_args()

  # Set up logging based on verbose/quiet options
  setup_logging(args.verbose, args.debug)

  # Check if input file exists
  if not os.path.exists(args.audio_path):
    logging.error(f"Input file '{args.audio_path}' does not exist.")
    sys.exit(1)

  # Set default output filename if not specified
  if args.output_to_stdout:
    args.output = sys.stdout
  elif not args.output:
    args.output = os.path.splitext(args.audio_path)[0] + ".txt"

  # Determine subtitle format
  subtitle_format = None
  if args.srt and args.vtt:
    logging.warning("Both SRT and VTT formats specified; defaulting to SRT")
    subtitle_format = "srt"
  elif args.srt:
    subtitle_format = "srt"
  elif args.vtt:
    subtitle_format = "vtt"

  # Log the configuration
  logging.info(f"Starting transcription process for: {args.audio_path}")
  if subtitle_format:
    logging.info(f"Generating {subtitle_format.upper()} subtitle file")
  
  # Call the main function with all arguments
  main(
    audio_path=args.audio_path, 
    chunk_length_ms=args.chunk_length, 
    output_file=args.output, 
    context=args.context, 
    prompt=args.prompt, 
    model=args.model, 
    no_post_processing=args.no_post_processing, 
    language=args.input_language, 
    temperature=args.temperature, 
    chunk_size=args.max_chunk_size,
    with_timestamps=args.timestamps,
    subtitle_format=subtitle_format,
    transcribe_model=args.transcribe_model
  )

# fin
