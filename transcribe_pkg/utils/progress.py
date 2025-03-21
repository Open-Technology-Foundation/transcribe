#!/usr/bin/env python3
"""
Progress tracking utilities for long-running operations.

This module provides tools for displaying and tracking progress of 
long-running operations like transcription, processing, and API calls.
"""
import sys
import time
import logging
from typing import Optional, Any, Dict, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

class ProgressDisplay:
    """
    Display and track progress for long-running operations.
    
    This class provides a simple API for updating and displaying progress,
    with support for different display formats and styles.
    """
    
    def __init__(
        self, 
        total: int = 100,
        description: str = "Progress", 
        logger: Optional[logging.Logger] = None, 
        unit: str = "items",
        show_time: bool = True,
        show_percent: bool = True,
        show_bar: bool = True,
        width: int = 30,
        use_colors: bool = True,
        update_interval: float = 0.2
    ):
        """
        Initialize the progress display.
        
        Args:
            total: Total number of units to process
            description: Description of the operation being tracked
            logger: Logger instance for log output
            unit: Unit label for items being processed
            show_time: Show elapsed and estimated time
            show_percent: Show percentage complete
            show_bar: Show progress bar
            width: Width of the progress bar in characters
            use_colors: Use ANSI colors in terminal output
            update_interval: Minimum seconds between updates
        """
        self.total = total
        self.description = description
        self.logger = logger
        self.unit = unit
        self.show_time = show_time
        self.show_percent = show_percent
        self.show_bar = show_bar
        self.width = width
        self.use_colors = use_colors
        self.update_interval = update_interval
        
        # Internal state
        self.completed = 0
        self.start_time = time.time()
        self.last_update_time = 0
        self.is_complete = False
    
    def update(self, completed: int, force: bool = False) -> None:
        """
        Update the progress status.
        
        Args:
            completed: Number of items completed
            force: Force display update even if interval hasn't elapsed
        """
        # Update completion count
        self.completed = min(completed, self.total)
        
        # Check if it's time to update the display
        current_time = time.time()
        if not force and current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Calculate progress statistics
        percent = self.completed / self.total if self.total > 0 else 0
        elapsed_time = current_time - self.start_time
        
        # Estimate remaining time
        if percent > 0:
            total_estimated_time = elapsed_time / percent
            remaining_time = total_estimated_time - elapsed_time
        else:
            remaining_time = 0
        
        # Generate progress string
        progress_str = f"{self.description}: "
        
        # Add progress bar if enabled
        if self.show_bar:
            bar_width = self.width
            filled_width = int(bar_width * percent)
            
            if self.use_colors:
                # Green progress bar with color gradient
                bar = (
                    "\033[32m" + 
                    "█" * filled_width + 
                    "\033[37m" + 
                    "░" * (bar_width - filled_width) + 
                    "\033[0m"
                )
            else:
                bar = "[" + "█" * filled_width + " " * (bar_width - filled_width) + "]"
            
            progress_str += bar + " "
        
        # Add percentage if enabled
        if self.show_percent:
            if self.use_colors and percent >= 1.0:
                progress_str += "\033[32m"
            
            progress_str += f"{percent:.1%} "
            
            if self.use_colors and percent >= 1.0:
                progress_str += "\033[0m"
        
        # Add progress counts
        progress_str += f"({self.completed}/{self.total} {self.unit})"
        
        # Add time information if enabled
        if self.show_time:
            elapsed_str = self._format_time(elapsed_time)
            
            if percent < 1.0:
                remaining_str = self._format_time(remaining_time)
                progress_str += f" | {elapsed_str} elapsed, ~{remaining_str} remaining"
            else:
                progress_str += f" | completed in {elapsed_str}"
        
        # Display progress
        if self.logger:
            self.logger.info(progress_str)
        else:
            sys.stderr.write("\r" + progress_str.ljust(80))
            sys.stderr.flush()
            
            # Add a newline if complete
            if self.completed == self.total and not self.is_complete:
                sys.stderr.write("\n")
                self.is_complete = True
    
    def complete(self) -> None:
        """Mark the operation as complete and show final progress."""
        self.update(self.total, force=True)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

class ParallelProcessor:
    """
    Process items in parallel with progress tracking.
    
    This class handles parallel processing of items with a thread pool,
    while providing progress updates via a ProgressDisplay.
    """
    
    def __init__(
        self, 
        max_workers: int = 4,
        description: str = "Processing",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of workers in the thread pool
            description: Description of the processing operation
            logger: Logger instance for log output
        """
        self.max_workers = max_workers
        self.description = description
        self.logger = logger
    
    def process(
        self, 
        items: List[Any], 
        process_func: callable, 
        *args: Any,
        **kwargs: Any
    ) -> List[Any]:
        """
        Process items in parallel with progress tracking.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            *args: Additional positional arguments for process_func
            **kwargs: Additional keyword arguments for process_func
            
        Returns:
            List of processed results
        """
        total_items = len(items)
        results = [None] * total_items
        completed = 0
        
        # Create progress display
        progress = ProgressDisplay(
            total=total_items,
            description=self.description,
            logger=self.logger
        )
        
        # Define the worker function
        def worker(index, item):
            try:
                return index, process_func(item, *args, **kwargs)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error processing item {index}: {str(e)}")
                return index, None
        
        # Process items in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(worker, idx, item) 
                for idx, item in enumerate(items)
            ]
            
            # Collect results as they complete
            for future in futures:
                try:
                    index, result = future.result()
                    results[index] = result
                    completed += 1
                    progress.update(completed)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in worker: {str(e)}")
                    completed += 1
                    progress.update(completed)
        
        progress.complete()
        return results

#fin