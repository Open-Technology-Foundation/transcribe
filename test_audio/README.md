# Test Audio Files

This directory contains audio files and reference transcripts used for testing and benchmarking the transcribe package.

## Contents

- **small_test_audio.mp3**: Short audio clip (approximately 30 seconds)
- **medium_test_audio.mp3**: Medium-length audio clip (approximately 2-5 minutes)
- **large_test_audio.mp3**: Long audio clip (approximately 10+ minutes)

## Usage

These files are referenced by the test and benchmark scripts in the `tests/benchmark/` directory. They provide consistent audio samples for:

1. Verifying transcription accuracy
2. Measuring performance improvements with parallel processing
3. Testing subtitle generation
4. Validating post-processing capabilities

## Note for Contributors

When adding new test audio files, please ensure they are:
- Free from copyright restrictions or properly licensed
- Appropriately sized for the intended test case
- Accompanied by a reference transcript when possible

