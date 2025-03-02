#!/usr/bin/env python3
"""
Core transcription functionality for the transcribe package.
"""

import os
import sys
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from tqdm import tqdm

from transcribe_pkg.utils.api_utils import call_llm, transcribe_audio
from transcribe_pkg.utils.text_utils import create_paragraphs
from transcribe_pkg.utils.language_utils import determine_language, get_language_name
from transcribe_pkg.core.processor import process_transcript

def create_field_context_string(input_text, model='gpt-4o-mini'):
  """
  Create a context string for the transcript based on content.
  
  Args:
    input_text (str): Input text to analyze
    model (str, optional): Model to use. Defaults to 'gpt-4o-mini'.
    
  Returns:
    str: Comma-separated field context
  """
  systemprompt="""You are a knowledgeable academic classifier specializing in categorizing content across academic disciplines, scientific fields, and cultural domains. Analyze the provided text and identify the most relevant fields it belongs to.

Provide exactly 3-5 primary fields, ordered by relevance, as a simple comma-separated list.

Format: field1,field2,field3[,field4][,field5]

Example outputs:
- economics,sociology,political science
- physics,astronomy,mathematics,engineering
- literature,cultural studies,history,sociology
  """
  return call_llm(systemprompt, input_text, model, 0, 100)

def split_audio(audio_path, chunk_length_ms=600000):
  """
  Split audio into chunks of specified length.
  
  Args:
    audio_path (str): Path to audio file
    chunk_length_ms (int, optional): Length of chunks in milliseconds. Defaults to 600000 (10 minutes).
    
  Returns:
    list: List of paths to chunk files
  """
  logging.info(f"Loading audio file: {audio_path}")
  audio = AudioSegment.from_mp3(audio_path)
  total_length_ms = len(audio)
  chunks = []
  temp_dir = tempfile.mkdtemp()
  
  logging.info(f"Splitting audio into {chunk_length_ms/1000}-second chunks")
  for i in tqdm(range(0, total_length_ms, chunk_length_ms), 
                desc="Splitting audio", 
                disable=not logging.getLogger().isEnabledFor(logging.INFO)):
    chunk = audio[i:i+chunk_length_ms]
    chunk_path = os.path.join(temp_dir, f"chunk_{i//chunk_length_ms}.mp3")
    with open(chunk_path, 'wb') as f:
      chunk.export(f, format="mp3")
    chunks.append(chunk_path)
    
  logging.info(f"Audio split into {len(chunks)} chunks")
  return chunks

def transcribe_chunk(args):
  """
  Transcribe a single audio chunk.
  
  Args:
    args (tuple): (chunk_path, prompt, language)
    
  Returns:
    str: Transcribed text
  """
  chunk_path, prompt, language = args
  try:
    return transcribe_audio(chunk_path, prompt, language)
  except Exception as e:
    logging.error(f"Error transcribing {chunk_path}: {str(e)}")
    return ""

def transcribe_chunks_parallel(chunk_paths, prompt='', language='en', max_workers=None):
  """
  Transcribe audio chunks in parallel using thread pool.
  
  Args:
    chunk_paths (list): Paths to audio chunks
    prompt (str, optional): Transcription prompt. Defaults to ''.
    language (str, optional): Language code. Defaults to 'en'.
    max_workers (int, optional): Maximum number of worker threads. Defaults to None (auto).
    
  Returns:
    list: List of transcribed texts
  """
  # Set default max_workers to min(8, number of chunks)
  if max_workers is None:
    max_workers = min(8, len(chunk_paths))
    
  logging.info(f"Transcribing chunks in parallel with {max_workers} workers")
  transcripts = []
  
  # Create arguments for each chunk
  chunk_args = [(chunk_path, prompt, language) for chunk_path in chunk_paths]
  
  # Use thread pool for parallel processing
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Map the transcribe_chunk function to each chunk and track progress
    for transcript in tqdm(
        executor.map(transcribe_chunk, chunk_args),
        total=len(chunk_paths),
        desc="Transcribing chunks",
        disable=not logging.getLogger().isEnabledFor(logging.INFO)
    ):
      transcripts.append(transcript)
      
  return transcripts

def transcribe_chunks_sequential(chunk_paths, prompt='', language='en'):
  """
  Transcribe audio chunks sequentially, updating prompt with context.
  
  Args:
    chunk_paths (list): Paths to audio chunks
    prompt (str, optional): Initial prompt. Defaults to ''.
    language (str, optional): Language code. Defaults to 'en'.
    
  Returns:
    list: List of transcribed texts
  """
  current_prompt = prompt[-896:] # 224 tokens * 4 = 896
  transcripts = []
  
  for chunk_path in tqdm(chunk_paths, 
                         desc="Transcribing chunks", 
                         disable=not logging.getLogger().isEnabledFor(logging.INFO)):
    try:
      transcript = transcribe_audio(chunk_path, current_prompt, language)
      transcripts.append(transcript)
      # Update prompt with context from this transcript
      current_prompt = f"{prompt}\n{transcript}"
      current_prompt = current_prompt[-896:]
    except Exception as e:
      logging.error(f"Error in transcription: {str(e)}")
      
  return transcripts

def transcribe_audio_file(audio_path, chunk_length_ms=600000, output_file=None, 
                         context=None, prompt='', model='gpt-4o', 
                         no_post_processing=False, language=None, 
                         temperature=0.1, chunk_size=3000, 
                         parallel_processing=True, max_workers=None):
  """
  Transcribe an audio file, optionally post-processing the transcript.
  
  Args:
    audio_path (str): Path to audio file
    chunk_length_ms (int, optional): Length of chunks in milliseconds. Defaults to 600000 (10 minutes).
    output_file (str, optional): Path to output file. Defaults to None (based on input file).
    context (str, optional): Domain context. Defaults to None (auto-detect).
    prompt (str, optional): Transcription prompt. Defaults to ''.
    model (str, optional): Model for post-processing. Defaults to 'gpt-4o'.
    no_post_processing (bool, optional): Skip post-processing. Defaults to False.
    language (str, optional): Language code. Defaults to None (auto-detect).
    temperature (float, optional): Temperature for generation. Defaults to 0.1.
    chunk_size (int, optional): Maximum chunk size. Defaults to 3000.
    parallel_processing (bool, optional): Use parallel processing. Defaults to True.
    max_workers (int, optional): Maximum worker threads. Defaults to None (auto).
    
  Returns:
    str: Transcribed and processed text
  """
  chunk_paths = []
  try:
    # Determine language if not specified
    if not language and prompt != '':
      logging.info("Determining language from prompt...")
      language = determine_language(prompt, 'gpt-4o')
    else:
      language = language or 'en'
    logging.info(f"Language: {language} ({get_language_name(language)})")
    
    if not os.path.exists(f'{output_file}.raw'):
      # Split audio into chunks
      chunk_paths = split_audio(audio_path, chunk_length_ms)
      
      # Transcribe chunks
      logging.info("Starting transcription process")
      if parallel_processing:
        transcripts = transcribe_chunks_parallel(chunk_paths, prompt, language, max_workers)
      else:
        transcripts = transcribe_chunks_sequential(chunk_paths, prompt, language)
      
      # Combine all transcript chunks
      logging.info("Combining transcripts")
      transcript = " ".join(transcripts)
      
      if output_file != sys.stdout:
        # Create paragraphs
        logging.info("Creating paragraphs")
        transcript = create_paragraphs(transcript, min_sentences=2, max_sentence_length=chunk_size)
        
        logging.info(f"Writing raw transcript to {output_file}.raw")
        with open(f'{output_file}.raw', 'w') as f:
          f.write(transcript)
    else:
      logging.info(f"Reading raw transcript {output_file}.raw")
      transcripts = []
      current_chunk = ""
      
      with open(f'{output_file}.raw', 'r') as file:
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
        logging.info(f"Loaded raw transcript (preview): {transcript[:80]}...")
    
    # Post-process the transcript if requested
    if not no_post_processing:
      if not context:
        logging.info("Creating field context")
        context = create_field_context_string(transcripts[0])
        logging.info(f"Context: {context}")
      
      logging.info("Post-processing transcript")
      transcript = process_transcript(
        transcript,
        model=model, 
        max_tokens=4096, 
        temperature=temperature,
        context=context,
        language=language,
        max_chunk_size=chunk_size
      )
    else:
      # Create paragraphs
      logging.info("Creating paragraphs")
      transcript = create_paragraphs(transcript, min_sentences=2, max_sentence_length=chunk_size)
    
    # Write full transcript to file or stdout
    if output_file == sys.stdout:
      logging.info("Writing full transcript to stdout")
      print(transcript)
    else:
      logging.info(f"Writing full transcript to {output_file}")
      with open(output_file, "w") as f:
        f.write(transcript)
    
    logging.info(f"Transcription complete")
    logging.info(f"Total words transcribed: {len(transcript.split())}")
    
    return transcript
    
  except FileNotFoundError as e:
    logging.error(f"File not found error: {str(e)}")
    logging.error(f"Check that the audio file exists and is accessible.")
    return ""
  except PermissionError as e:
    logging.error(f"Permission error: {str(e)}")
    logging.error(f"Check that you have read/write permissions for the files and directories.")
    return ""
  except (ValueError, TypeError) as e:
    logging.error(f"Value or type error: {str(e)}")
    logging.error(f"Check that the parameters are of the correct type and format.")
    return ""
  except KeyboardInterrupt:
    logging.info("Transcription process interrupted by user.")
    logging.info("Cleaning up and saving any progress...")
    # If we have partial results, save them
    if 'transcript' in locals() and transcript and output_file != sys.stdout:
      try:
        with open(f"{output_file}.partial", "w") as f:
          f.write(transcript)
        logging.info(f"Partial results saved to {output_file}.partial")
      except Exception as save_error:
        logging.error(f"Error saving partial results: {str(save_error)}")
    return ""
  except Exception as e:
    logging.error(f"Unexpected error in transcription process: {str(e)}")
    logging.debug(f"Error details: {type(e).__name__}: {str(e)}")
    logging.error(f"If this error persists, please check your API key and network connection.")
    return ""
  finally:
    # Clean up temporary files
    cleanup_success = True
    for chunk_path in chunk_paths:
      if os.path.exists(chunk_path):
        try:
          os.remove(chunk_path)
        except Exception as e:
          cleanup_success = False
          logging.warning(f"Failed to remove temporary file {chunk_path}: {str(e)}")
    
    if cleanup_success:
      logging.info("All temporary files cleaned up successfully")
    else:
      logging.warning("Some temporary files could not be cleaned up")

#fin