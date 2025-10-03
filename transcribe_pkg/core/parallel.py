#!/usr/bin/env python3
"""
Parallel processing for transcription and text processing.

This module provides tools for processing large transcripts in parallel,
dividing work across multiple processors and threads for improved performance.
"""
import os
import logging
from typing import Any
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import json

from transcribe_pkg.utils.logging_utils import get_logger
from transcribe_pkg.utils.progress import ProgressDisplay
from transcribe_pkg.utils.text_utils import create_sentences, split_text_for_processing

class ParallelProcessor:
    """
    Process large transcripts in parallel for improved performance.
    
    This class provides parallel processing capabilities for handling
    large transcripts, distributing work across multiple threads or
    processes to improve processing speed.
    """
    
    def __init__(
        self,
        max_workers: int | None = None,
        use_processes: bool = False,
        chunk_size: int = 4000,
        overlap: int = 500,
        logger: logging.Logger | None = None
    ):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_processes: Use processes instead of threads
            chunk_size: Maximum size of each chunk in bytes
            overlap: Overlap between chunks in bytes
            logger: Logger instance
        """
        self.logger = logger or get_logger(__name__)
        
        # Set default max_workers based on available CPUs
        if max_workers is None:
            self.max_workers = os.cpu_count() or 4
        else:
            self.max_workers = max_workers
            
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        self.logger.debug(f"Initialized ParallelProcessor with {self.max_workers} workers")
    
    def process_text(
        self,
        text: str,
        process_func: Callable[[str, dict[str, Any]], str],
        combine_func: Callable[[list[str]], str] | None = None,
        show_progress: bool = True,
        **kwargs: Any
    ) -> str:
        """
        Process a large text in parallel chunks.
        
        Args:
            text: Text to process
            process_func: Function to apply to each chunk
            combine_func: Function to combine processed chunks
            show_progress: Display progress indicator
            **kwargs: Additional arguments to pass to process_func
            
        Returns:
            Processed text
        """
        # Split text into chunks for parallel processing
        chunks = split_text_for_processing(
            text, 
            max_chunk_size=self.chunk_size,
            overlap=self.overlap
        )
        
        total_chunks = len(chunks)
        self.logger.info(f"Split text into {total_chunks} chunks for parallel processing")
        
        # Create progress display if requested
        progress = None
        if show_progress:
            progress = ProgressDisplay(
                total=total_chunks,
                description="Processing chunks",
                unit="chunks",
            )
        
        # Process chunks in parallel
        processed_chunks = []
        
        # Choose executor based on configuration
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        # Process chunks in parallel
        chunk_results = {}
        completed = 0
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for i, chunk in enumerate(chunks):
                future = executor.submit(process_func, chunk, kwargs)
                future_to_idx[future] = i
            
            # Process results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    chunk_results[idx] = result
                except Exception as e:
                    self.logger.error(f"Error processing chunk {idx}: {str(e)}")
                    # Use original chunk on error
                    chunk_results[idx] = chunks[idx]
                
                # Update progress
                completed += 1
                if progress:
                    progress.update(completed)
        
        # Ensure progress is complete
        if progress:
            progress.complete()
        
        # Combine results in the original order
        ordered_results = [chunk_results[i] for i in range(total_chunks)]
        
        # Apply combine function if provided, otherwise join with spaces
        if combine_func:
            return combine_func(ordered_results)
        else:
            return "\n\n".join(ordered_results)
    
    def process_audio_chunks(
        self,
        chunk_paths: list[str],
        process_func: Callable[[str, dict[str, Any]], Any],
        combine_func: Callable[[list[Any]], Any] | None = None,
        show_progress: bool = True,
        **kwargs: Any
    ) -> Any:
        """
        Process audio chunks in parallel.
        
        Args:
            chunk_paths: List of audio chunk file paths
            process_func: Function to apply to each chunk
            combine_func: Function to combine processed chunks
            show_progress: Display progress indicator
            **kwargs: Additional arguments to pass to process_func
            
        Returns:
            Combined results from all chunks
        """
        total_chunks = len(chunk_paths)
        self.logger.info(f"Processing {total_chunks} audio chunks in parallel")
        
        # Create progress display if requested
        progress = None
        if show_progress:
            progress = ProgressDisplay(
                total=total_chunks,
                description="Processing audio chunks",
                unit="chunks",
            )
        
        # Choose executor based on configuration
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        # Process chunks in parallel
        chunk_results = {}
        completed = 0
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for i, chunk_path in enumerate(chunk_paths):
                # Create a dictionary with all kwargs plus the index
                chunk_kwargs = kwargs.copy()
                chunk_kwargs['chunk_index'] = i
                
                future = executor.submit(process_func, chunk_path, chunk_kwargs)
                future_to_idx[future] = i
            
            # Process results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    chunk_results[idx] = result
                except Exception as e:
                    self.logger.error(f"Error processing audio chunk {idx}: {str(e)}")
                    # Store None on error
                    chunk_results[idx] = None
                
                # Update progress
                completed += 1
                if progress:
                    progress.update(completed)
        
        # Ensure progress is complete
        if progress:
            progress.complete()
        
        # Combine results in the original order
        ordered_results = [
            chunk_results[i] for i in range(total_chunks) 
            if chunk_results[i] is not None
        ]
        
        # Apply combine function if provided
        if combine_func and ordered_results:
            return combine_func(ordered_results)
        else:
            return ordered_results

class ChunkProcessor:
    """
    Process text chunks with specialized handling for different content types.
    
    This class provides advanced chunk processing with different strategies
    for different types of content (e.g., dialogue, technical text, etc.)
    """
    
    def __init__(
        self,
        logger: logging.Logger | None = None
    ):
        """
        Initialize the chunk processor.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or get_logger(__name__)
    
    def combine_chunks(self, chunks: list[str]) -> str:
        """
        Combine processed chunks into a cohesive text.
        
        Args:
            chunks: List of processed text chunks
            
        Returns:
            Combined text
        """
        # For simple combination, just join with double newlines
        combined_text = "\n\n".join(chunks)
        
        # Clean up any duplicate paragraph breaks
        combined_text = combined_text.replace("\n\n\n\n", "\n\n")
        combined_text = combined_text.replace("\n\n\n", "\n\n")
        
        return combined_text
    
    def detect_content_type(self, text: str) -> str:
        """
        Detect the type of content in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Content type ('dialogue', 'technical', 'narrative', etc.)
        """
        # Check for dialogue patterns
        dialogue_markers = [":", '"', "'", "said", "asked", "replied"]
        dialogue_count = sum(text.count(marker) for marker in dialogue_markers)
        
        # Check for technical content
        technical_markers = ["Figure", "Table", "algorithm", "equation", "function"]
        technical_count = sum(text.count(marker) for marker in technical_markers)
        
        # Determine content type based on markers
        if dialogue_count > len(text.split()) / 20:
            return "dialogue"
        elif technical_count > len(text.split()) / 100:
            return "technical"
        else:
            return "narrative"
    
    def adjust_chunk_boundaries(
        self,
        chunks: list[str]
    ) -> list[str]:
        """
        Adjust chunk boundaries to improve coherence.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Adjusted text chunks
        """
        adjusted_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Find common sentences between this chunk and the previous one
                prev_chunk = chunks[i-1]
                sentences_current = create_sentences(chunk)
                sentences_prev = create_sentences(prev_chunk)
                
                # Check for overlapping sentences
                overlap_found = False
                for j in range(min(5, len(sentences_prev))):
                    for k in range(min(5, len(sentences_current))):
                        if sentences_prev[-j-1] == sentences_current[k]:
                            # Found an overlapping sentence
                            # Remove everything before this in current chunk
                            adjusted_chunk = " ".join(sentences_current[k:])
                            adjusted_chunks.append(adjusted_chunk)
                            overlap_found = True
                            break
                    if overlap_found:
                        break
                
                if not overlap_found:
                    adjusted_chunks.append(chunk)
            else:
                # First chunk doesn't need adjustment
                adjusted_chunks.append(chunk)
        
        return adjusted_chunks

#fin