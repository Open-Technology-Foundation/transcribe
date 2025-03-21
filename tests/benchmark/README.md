# Transcribe Package Benchmarks and Tests

This directory contains scripts for benchmarking and testing the parallel processing features of the transcribe package.

## Available Scripts

- `benchmark-parallel.sh`: Benchmark script to compare performance of different parallel processing configurations
- `test-parallel-flag.py`: Test script to verify the correct implementation of the `--parallel` flag
- `test-medium-file.py`: Test script for medium-sized audio files with parallel processing
- `test-large-file.py`: Test script for large audio files with parallel processing
- `test-new-features`: Shell script for testing multiple transcript processing features

## Usage

### Running the Benchmark

```bash
./benchmark-parallel.sh path/to/audio_file [--skip-large]
```

### Running Tests

```bash
# Test parallel flag implementation
./test-parallel-flag.py

# Test with medium-sized file
./test-medium-file.py

# Test with large file
./test-large-file.py

# Test multiple features
./test-new-features --medium --parallel --verbose
```

For more options, run each script with the `--help` flag.

## Performance Considerations

These scripts are designed to measure and validate the performance improvements gained from parallel processing. Refer to the main `/PARALLEL_PROCESSING.md` documentation for detailed information about optimization strategies.