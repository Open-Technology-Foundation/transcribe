#!/usr/bin/env python3
"""
Core transcription module for processing audio files.

This module provides the core functionality for transcribing audio files,
including splitting audio into chunks, performing transcription using OpenAI's
Whisper API, and handling timestamped output.
"""
import os
import logging
import tempfile
from typing import Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from transcribe_pkg.utils.logging_utils import get_logger
from transcribe_pkg.utils.audio_utils import AudioProcessor
from transcribe_pkg.utils.api_utils import OpenAIClient, APIError

class Transcriber:
    """
    Core transcription engine for audio files.
    
    This class handles the transcription workflow for audio files,
    including splitting, transcribing, and combining results.
    """
    
    def __init__(
        self,
        api_client: Optional[OpenAIClient] = None,
        audio_processor: Optional[AudioProcessor] = None,
        model: str = "whisper-1",
        language: Optional[str] = None,
        temperature: float = 0.05,
        chunk_length_ms: int = 600000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize transcriber with configuration.
        
        Args:
            api_client: OpenAIClient instance for API calls
            audio_processor: AudioProcessor for handling audio files
            model: Transcription model to use
            language: Language code (e.g., 'en', 'fr')
            temperature: Temperature for generation
            chunk_length_ms: Length of audio chunks in milliseconds
            logger: Logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.api_client = api_client or OpenAIClient(logger=self.logger)
        self.audio_processor = audio_processor or AudioProcessor(logger=self.logger)
        self.model = model
        self.language = language
        self.temperature = temperature
        self.chunk_length_ms = chunk_length_ms
    
    def transcribe(
        self,
        audio_path: str,
        prompt: str = "",
        with_timestamps: bool = False,
        max_workers: int = 1
    ) -> Union[str, Dict[str, Any]]:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            prompt: Optional context prompt to guide transcription
            with_timestamps: Include timestamp information in output
            max_workers: Maximum number of parallel workers for chunks
            
        Returns:
            If with_timestamps=False: Transcribed text as string
            If with_timestamps=True: Dictionary with text and timestamp information
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            APIError: For API-related issues
            Exception: For other errors during transcription
        """
        self.logger.info(f"Starting transcription of {audio_path}")
        
        # Check if file exists
        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Step 1: Split audio into chunks
            self.logger.info(f"Splitting audio into {self.chunk_length_ms/1000}-second chunks")
            try:
                chunk_paths = self.audio_processor.split_audio(audio_path, self.chunk_length_ms)
                self.logger.info(f"Audio split into {len(chunk_paths)} chunks")
            except ValueError as e:
                self.logger.error(f"Error processing audio file: {str(e)}")
                # Reraise with more descriptive message
                raise ValueError(f"Failed to process audio file. Please check if the file is valid and not empty. Error: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error splitting audio: {str(e)}")
                raise
            
            # Step 2: Choose transcription method based on worker count
            if max_workers > 1:
                self.logger.info(f"Using parallel transcription with {max_workers} workers")
                result = self._transcribe_chunks_parallel(
                    chunk_paths, prompt, with_timestamps, max_workers
                )
            else:
                self.logger.info("Using sequential transcription")
                result = self._transcribe_chunks_sequential(
                    chunk_paths, prompt, with_timestamps
                )
            
            # Step 3: Return appropriate result format
            if with_timestamps:
                return result
            else:
                # For text-only output, just return the combined text
                return " ".join(result)
            
        except Exception as e:
            self.logger.error(f"Error in transcription: {str(e)}")
            raise
        finally:
            # Clean up temporary files
            self.audio_processor.cleanup()
    
    def _transcribe_chunks_sequential(
        self,
        chunk_paths: List[str],
        prompt: str = "",
        with_timestamps: bool = False
    ) -> Union[List[str], Dict[str, Any]]:
        """
        Transcribe audio chunks sequentially.
        
        Args:
            chunk_paths: List of paths to audio chunks
            prompt: Initial prompt to guide transcription
            with_timestamps: Include timestamp information in output
            
        Returns:
            If with_timestamps=False: List of transcribed text chunks
            If with_timestamps=True: Dictionary with text and segments
        """
        context_prompt = prompt[-896:] if prompt else ""  # Limit context prompt size
        transcripts = []
        
        # For timestamp mode, track the total duration and segments
        total_duration = 0
        all_segments = []
        
        # Process each chunk with a progress bar
        for chunk_index, chunk_path in enumerate(tqdm(
            chunk_paths, 
            desc="Transcribing chunks", 
            disable=not self.logger.isEnabledFor(logging.INFO)
        )):
            try:
                # Prepare response format based on timestamp requirement
                response_format = "verbose_json" if with_timestamps else "text"
                
                # Call API to transcribe the chunk
                transcription = self.api_client.transcribe_audio(
                    chunk_path,
                    model=self.model,
                    prompt=context_prompt,
                    language=self.language or "en",
                    response_format=response_format,
                    temperature=self.temperature
                )
                
                if with_timestamps:
                    # Handle empty results
                    if not transcription:
                        self.logger.warning(f"Empty transcript for chunk {chunk_index}. Skipping.")
                        continue
                    
                    # Extract text based on response type (object or dictionary)
                    if hasattr(transcription, 'text'):
                        transcript_text = transcription.text
                    elif isinstance(transcription, dict) and 'text' in transcription:
                        transcript_text = transcription['text']
                    else:
                        self.logger.warning(f"Unexpected response format for chunk {chunk_index}. Skipping.")
                        continue
                    
                    # Skip if text is empty
                    if not transcript_text:
                        self.logger.warning(f"Empty transcript text for chunk {chunk_index}. Skipping.")
                        continue
                    
                    # Get segments based on response type
                    if hasattr(transcription, 'segments'):
                        segments = transcription.segments
                    elif isinstance(transcription, dict) and 'segments' in transcription:
                        segments = transcription['segments']
                    else:
                        segments = []
                        self.logger.warning(f"No segments found for chunk {chunk_index}")
                    
                    # Adjust segment timestamps for this chunk's position and convert to dict
                    for segment in segments:
                        # Determine if we're dealing with an object or dictionary
                        if hasattr(segment, 'id'):
                            # Object response style
                            segment_dict = {
                                "id": segment.id if hasattr(segment, 'id') else 0,
                                "start": segment.start + total_duration if hasattr(segment, 'start') else 0 + total_duration,
                                "end": segment.end + total_duration if hasattr(segment, 'end') else 0 + total_duration,
                                "text": segment.text if hasattr(segment, 'text') else "",
                                "words": []
                            }
                            
                            # Extract word-level timestamps if available
                            if hasattr(segment, "words"):
                                for word in segment.words:
                                    segment_dict["words"].append({
                                        "word": word.word if hasattr(word, 'word') else "",
                                        "start": word.start + total_duration if hasattr(word, 'start') else 0 + total_duration,
                                        "end": word.end + total_duration if hasattr(word, 'end') else 0 + total_duration
                                    })
                        else:
                            # Dictionary response style
                            segment_dict = {
                                "id": segment.get("id", 0),
                                "start": segment.get("start", 0) + total_duration,
                                "end": segment.get("end", 0) + total_duration,
                                "text": segment.get("text", ""),
                                "words": []
                            }
                            
                            # Extract word-level timestamps if available
                            for word in segment.get("words", []):
                                segment_dict["words"].append({
                                    "word": word.get("word", ""),
                                    "start": word.get("start", 0) + total_duration,
                                    "end": word.get("end", 0) + total_duration
                                })
                                
                        all_segments.append(segment_dict)
                    
                    # Add text to results
                    transcripts.append(transcript_text)
                    
                    # Update context prompt with the transcribed text
                    context_prompt = f"{prompt}\n{transcript_text}"[-896:]
                    
                    # Update total duration for the next chunk
                    if segments:
                        # Get end timestamp from last segment
                        if hasattr(segments[-1], 'end'):
                            chunk_duration = segments[-1].end
                        else:
                            chunk_duration = segments[-1].get("end", 0)
                        total_duration += chunk_duration
                else:
                    # Text-only mode
                    # Handle empty results
                    if not transcription:
                        self.logger.warning(f"Empty transcript for chunk {chunk_index}. Skipping.")
                        transcription = ""
                    
                    transcripts.append(transcription)
                    
                    # Update context prompt with the transcribed text
                    context_prompt = f"{prompt}\n{transcription}"[-896:]
                
            except Exception as e:
                self.logger.error(f"Error in transcription for chunk {chunk_index}: {str(e)}")
                transcripts.append("")
        
        # Return appropriate result format
        if with_timestamps:
            return {
                "text": " ".join(transcripts),
                "segments": all_segments
            }
        else:
            return transcripts
    
    def _transcribe_chunks_parallel(
        self,
        chunk_paths: List[str],
        prompt: str = "",
        with_timestamps: bool = False,
        max_workers: int = 4
    ) -> Union[List[str], Dict[str, Any]]:
        """
        Transcribe audio chunks in parallel using multiple workers.
        
        Args:
            chunk_paths: List of paths to audio chunks
            prompt: Initial prompt to guide transcription
            with_timestamps: Include timestamp information in output
            max_workers: Maximum number of parallel workers
            
        Returns:
            If with_timestamps=False: List of transcribed text chunks
            If with_timestamps=True: Dictionary with text and segments
        """
        # Limit number of workers to the number of chunks
        workers = min(max_workers, len(chunk_paths))
        
        # Prepare results containers
        results = [None] * len(chunk_paths)
        
        # For timestamp mode
        all_segments = []
        
        # Submit transcription tasks to thread pool
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all transcription tasks
            futures = []
            for i, chunk_path in enumerate(chunk_paths):
                # Create a prompt specific to this chunk
                # For now, keep it simple - more advanced context prompting can be added later
                chunk_prompt = prompt
                
                # Configure response format based on timestamp requirement
                response_format = "verbose_json" if with_timestamps else "text"
                
                # Submit transcription task
                future = executor.submit(
                    self.api_client.transcribe_audio,
                    chunk_path,
                    model=self.model,
                    prompt=chunk_prompt,
                    language=self.language or "en",
                    response_format=response_format,
                    temperature=self.temperature
                )
                futures.append((i, future))
            
            # Process results as they complete with a progress bar
            for i, future in tqdm(
                futures,
                desc=f"Transcribing chunks (parallel: {workers} workers)",
                disable=not self.logger.isEnabledFor(logging.INFO)
            ):
                try:
                    result = future.result()
                    results[i] = result
                except Exception as e:
                    self.logger.error(f"Error in parallel transcription for chunk {i}: {str(e)}")
                    results[i] = "" if not with_timestamps else {"text": "", "segments": []}
        
        # Process results based on format required
        if with_timestamps:
            # We need to post-process to adjust the segment timestamps
            # This requires sequential processing to ensure proper ordering
            transcript_texts = []
            total_duration = 0
            
            for i, result in enumerate(results):
                if not result:
                    self.logger.warning(f"Empty transcript for chunk {i}. Skipping.")
                    continue
                
                # Extract text based on response type (object or dictionary)
                if hasattr(result, 'text'):
                    transcript_text = result.text
                elif isinstance(result, dict) and 'text' in result:
                    transcript_text = result['text']
                else:
                    self.logger.warning(f"Unexpected response format for chunk {i}. Skipping.")
                    continue
                
                # Skip if text is empty
                if not transcript_text:
                    self.logger.warning(f"Empty transcript text for chunk {i}. Skipping.")
                    continue
                
                transcript_texts.append(transcript_text)
                
                # Get segments based on response type
                if hasattr(result, 'segments'):
                    segments = result.segments
                elif isinstance(result, dict) and 'segments' in result:
                    segments = result['segments']
                else:
                    segments = []
                    self.logger.warning(f"No segments found for chunk {i}")
                
                # Adjust segment timestamps for this chunk's position and convert to dict
                for segment in segments:
                    # Determine if we're dealing with an object or dictionary
                    if hasattr(segment, 'id'):
                        # Object response style
                        segment_dict = {
                            "id": segment.id if hasattr(segment, 'id') else 0,
                            "start": segment.start + total_duration if hasattr(segment, 'start') else 0 + total_duration,
                            "end": segment.end + total_duration if hasattr(segment, 'end') else 0 + total_duration,
                            "text": segment.text if hasattr(segment, 'text') else "",
                            "words": []
                        }
                        
                        # Extract word-level timestamps if available
                        if hasattr(segment, "words"):
                            for word in segment.words:
                                segment_dict["words"].append({
                                    "word": word.word if hasattr(word, 'word') else "",
                                    "start": word.start + total_duration if hasattr(word, 'start') else 0 + total_duration,
                                    "end": word.end + total_duration if hasattr(word, 'end') else 0 + total_duration
                                })
                    else:
                        # Dictionary response style
                        segment_dict = {
                            "id": segment.get("id", 0),
                            "start": segment.get("start", 0) + total_duration,
                            "end": segment.get("end", 0) + total_duration,
                            "text": segment.get("text", ""),
                            "words": []
                        }
                        
                        # Extract word-level timestamps if available
                        for word in segment.get("words", []):
                            segment_dict["words"].append({
                                "word": word.get("word", ""),
                                "start": word.get("start", 0) + total_duration,
                                "end": word.get("end", 0) + total_duration
                            })
                            
                    all_segments.append(segment_dict)
                
                # Update total duration for the next chunk
                if segments:
                    # Get end timestamp from last segment
                    if hasattr(segments[-1], 'end'):
                        chunk_duration = segments[-1].end
                    else:
                        chunk_duration = segments[-1].get("end", 0)
                    total_duration += chunk_duration
            
            return {
                "text": " ".join(transcript_texts),
                "segments": all_segments
            }
        else:
            # For text-only output, just return the list of texts
            return [r if r else "" for r in results]

def transcribe_audio_file(
    audio_path: str,
    output_file: Optional[str] = None,
    context: str = "",
    model: str = "gpt-4o",
    language: Optional[str] = None,
    parallel_processing: bool = False,
    max_workers: int = 1,
    chunk_size: int = 3000,
    with_timestamps: bool = False
) -> str:
    """
    High-level function to transcribe an audio file with optional post-processing.
    
    Args:
        audio_path: Path to the audio file to transcribe
        output_file: Optional output file path (if None, returns text only)
        context: Context information for better transcription quality
        model: Model to use for post-processing
        language: Language code for transcription
        parallel_processing: Enable parallel processing
        max_workers: Number of workers for parallel processing
        chunk_size: Maximum chunk size for processing
        with_timestamps: Include timestamp information
        
    Returns:
        Transcribed text
        
    Raises:
        APIError: For API-related errors
        FileNotFoundError: If audio file doesn't exist
    """
    # Initialize transcriber
    transcriber = Transcriber(
        language=language,
        max_workers=max_workers if parallel_processing else 1
    )
    
    # Transcribe the audio
    transcript_result = transcriber.transcribe(
        audio_path=audio_path,
        prompt=context,
        with_timestamps=with_timestamps,
        max_workers=max_workers if parallel_processing else 1
    )
    
    # Extract text from result
    if with_timestamps and isinstance(transcript_result, dict):
        transcribed_text = transcript_result["text"]
    elif isinstance(transcript_result, list):
        transcribed_text = " ".join(transcript_result)
    else:
        transcribed_text = str(transcript_result)
    
    # Post-process if we have context or advanced processing is requested
    if context and model != "none":
        try:
            from transcribe_pkg.core.processor import TranscriptProcessor
            processor = TranscriptProcessor(
                model=model,
                max_chunk_size=chunk_size
            )
            processed_text = processor.process(
                text=transcribed_text,
                context=context,
                use_parallel=parallel_processing
            )
            final_text = processed_text
        except ImportError:
            # If processor not available, use raw transcription
            final_text = transcribed_text
    else:
        final_text = transcribed_text
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)
    
    return final_text

def process_transcript(input_text: str, **kwargs) -> str:
    """
    Alias function for transcript processing (used by tests).
    
    Args:
        input_text: Text to process
        **kwargs: Additional arguments passed to processor
        
    Returns:
        Processed text
    """
    from transcribe_pkg.core.processor import process_transcript as proc_process_transcript
    return proc_process_transcript(input_text, **kwargs)

#fin