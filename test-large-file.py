#!/usr/bin/env python3
"""
Test script to validate the parallel processing implementation with a large audio file.

This script compares the performance of different parallel processing configurations
using the large test audio file.
"""

import os
import sys
import time
import logging
import subprocess
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_test(test_name: str, audio_file: str, args: List[str]) -> Dict:
    """
    Run a test with the given args and measure performance.
    
    Args:
        test_name: Name of the test
        audio_file: Path to the audio file
        args: List of arguments to pass to the transcribe command
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Running test: {test_name}")
    
    # Prepare the command
    cmd = ["python3", "-m", "transcribe_pkg", audio_file] + args
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Measure execution time
    start_time = time.time()
    
    # Run the command
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Prepare results
    test_result = {
        "test_name": test_name,
        "elapsed_time": elapsed_time,
        "exit_code": result.returncode,
        "command": " ".join(cmd)
    }
    
    # Extract worker counts from stderr logs
    stderr_lines = result.stderr.splitlines()
    for line in stderr_lines:
        if "workers for transcription" in line:
            test_result["transcription_workers"] = line
        elif "Using parallel processing with" in line:
            test_result["post_processing_workers"] = line
    
    # Extract timing information from logs
    for line in stderr_lines:
        if "Transcription completed in" in line:
            test_result["transcription_time"] = line
        elif "Post-processing completed in" in line:
            test_result["post_processing_time"] = line
        
    # Log result
    if result.returncode == 0:
        logger.info(f"{test_name} completed in {elapsed_time:.2f} seconds")
    else:
        logger.error(f"{test_name} failed with exit code {result.returncode}")
        logger.error(f"Stderr: {result.stderr}")
    
    return test_result

def run_tests():
    """Run tests with the large audio file."""
    # Use the large test audio
    large_audio = "test_audio/large_test_audio.mp3"
    
    # Verify the file exists
    if not os.path.exists(large_audio):
        logger.error(f"Large audio file not found: {large_audio}")
        return
        
    # Define the test configurations
    tests = [
        {
            "name": "Sequential with no post-processing",
            "args": ["-o", "/tmp/large_seq_no_post.txt", "-P", "-v"]
        },
        {
            "name": "Parallel transcription with no post-processing",
            "args": ["-o", "/tmp/large_parallel_no_post.txt", "--parallel", "-P", "-v"]
        },
        {
            "name": "Full parallel (transcription and post-processing)",
            "args": ["-o", "/tmp/large_full_parallel.txt", "--parallel", "--max-parallel-workers", "4", "-v"]
        }
    ]
    
    # Run all tests
    results = []
    for test in tests:
        result = run_test(test["name"], large_audio, test["args"])
        results.append(result)
        # Add a small delay between tests
        time.sleep(1)
    
    # Display results summary
    logger.info("\n--- Test Results Summary ---")
    for result in results:
        logger.info(f"{result['test_name']}:")
        logger.info(f"  Total time: {result['elapsed_time']:.2f} seconds")
        logger.info(f"  Exit code: {result['exit_code']}")
        if "transcription_workers" in result:
            logger.info(f"  {result['transcription_workers']}")
        if "post_processing_workers" in result:
            logger.info(f"  {result['post_processing_workers']}")
        if "transcription_time" in result:
            logger.info(f"  {result['transcription_time']}")
        if "post_processing_time" in result:
            logger.info(f"  {result['post_processing_time']}")
        logger.info("")

if __name__ == "__main__":
    run_tests()

#fin