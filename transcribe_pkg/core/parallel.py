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

from transcribe_pkg.utils.api_utils import APIError
from transcribe_pkg.utils.logging_utils import get_logger
from transcribe_pkg.utils.progress import ProgressDisplay
from transcribe_pkg.utils.text_utils import split_text_for_processing

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
            
        # Process-pool mode is unsupported for this I/O-bound API workload: the
        # worker closures capture a client-bearing object that cannot be pickled.
        # Accept the public parameter (external callers may pass it) but fall
        # back to threads instead of crashing later with a PicklingError.
        if use_processes:
            self.logger.warning(
                "use_processes=True is unsupported for this I/O-bound API "
                "workload; falling back to ThreadPoolExecutor"
            )
        self.use_processes = False
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
        
        # Choose executor based on configuration
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        # Process chunks in parallel
        chunk_results = {}
        completed = 0
        failed = 0

        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for i, chunk in enumerate(chunks):
                # Inject the chunk index so workers can log/cache per chunk
                # instead of every chunk reporting index 0.
                chunk_kwargs = kwargs.copy()
                chunk_kwargs["chunk_index"] = i
                future = executor.submit(process_func, chunk, chunk_kwargs)
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
                    failed += 1

                # Update progress
                completed += 1
                if progress:
                    progress.update(completed)

        # Ensure progress is complete
        if progress:
            progress.complete()

        # Surface an aggregate signal for worker failures: a total wipeout
        # must not masquerade as success (returning the raw input), and a
        # partial failure should be visible in the logs.
        if total_chunks and failed == total_chunks:
            raise APIError(f"All {total_chunks} chunks failed post-processing")
        elif failed:
            self.logger.warning(f"{failed}/{total_chunks} chunks fell back to raw text")

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
        
        # Combine results in the original order, preserving positions. Failed
        # chunks are kept as None placeholders so later chunks do not shift up
        # (which would silently misalign timestamps/ordering downstream).
        ordered_results = [chunk_results[i] for i in range(total_chunks)]
        missing = [i for i in range(total_chunks) if chunk_results[i] is None]
        if missing:
            self.logger.warning(
                f"{len(missing)}/{total_chunks} audio chunks missing at "
                f"indices {missing}; preserved as None placeholders"
            )

        # Apply combine function if provided
        if combine_func and ordered_results:
            return combine_func(ordered_results)
        else:
            return ordered_results
