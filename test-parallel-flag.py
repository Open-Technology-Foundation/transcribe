#!/usr/bin/env python3
"""
Test script to validate the --parallel flag implementation.

This script tests the parallel transcription feature with different configurations
to ensure the --parallel flag correctly enables parallel processing for both 
transcription and post-processing.

It runs a series of tests with different combinations of flags to validate
that the fix for the parallel flag issue is working correctly.
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
        
    # Log result
    if result.returncode == 0:
        logger.info(f"{test_name} completed in {elapsed_time:.2f} seconds")
    else:
        logger.error(f"{test_name} failed with exit code {result.returncode}")
        logger.error(f"Stderr: {result.stderr}")
    
    return test_result

def run_all_tests():
    """Run all the test configurations."""
    # Use the small test audio for faster tests
    small_audio = "test_audio/small_test_audio.mp3"
    
    # Define the test matrix
    tests = [
        {
            "name": "No parallel (baseline)",
            "args": ["-o", "/tmp/test_no_parallel.txt", "-v"]
        },
        {
            "name": "Parallel flag only",
            "args": ["-o", "/tmp/test_parallel_flag.txt", "--parallel", "-v"]
        },
        {
            "name": "Explicit workers=2",
            "args": ["-o", "/tmp/test_workers_2.txt", "-w", "2", "-v"]
        },
        {
            "name": "Parallel flag with post-processing parallel",
            "args": ["-o", "/tmp/test_both_parallel.txt", "--parallel", "--max-parallel-workers", "2", "-v"]
        },
        {
            "name": "Parallel flag with post-processing disabled",
            "args": ["-o", "/tmp/test_parallel_no_post.txt", "--parallel", "-P", "-v"]
        }
    ]
    
    # Run all tests
    results = []
    for test in tests:
        result = run_test(test["name"], small_audio, test["args"])
        results.append(result)
        # Add a small delay between tests
        time.sleep(1)
    
    # Display results summary
    logger.info("\n--- Test Results Summary ---")
    for result in results:
        logger.info(f"{result['test_name']}:")
        logger.info(f"  Time: {result['elapsed_time']:.2f} seconds")
        logger.info(f"  Exit code: {result['exit_code']}")
        if "transcription_workers" in result:
            logger.info(f"  {result['transcription_workers']}")
        if "post_processing_workers" in result:
            logger.info(f"  {result['post_processing_workers']}")
        logger.info("")

if __name__ == "__main__":
    run_all_tests()

#fin