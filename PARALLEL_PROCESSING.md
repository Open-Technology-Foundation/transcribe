# Parallel Processing in Transcribe

This document explains how to use the parallel processing features in the transcribe package to improve performance for large audio files.

## Overview

The transcribe package supports two types of parallel processing:

1. **Parallel Transcription**: Transcribes audio chunks in parallel using multiple workers
2. **Parallel Post-Processing**: Processes transcript chunks in parallel for improved text processing performance

These features can be used independently or together to optimize performance based on your specific workload.

## Command-Line Usage

### Basic Parallel Processing

To enable parallel processing for both transcription and post-processing:

```bash
transcribe-parallel audio_file.mp3 -o transcript.txt
```

This wrapper script automatically sets optimal worker counts based on your CPU.

### Manual Configuration

You can also configure parallel processing manually with the following flags:

```bash
# Enable parallel transcription with 4 workers
transcribe-new audio_file.mp3 -w 4

# Enable only parallel post-processing with 4 workers
transcribe-new audio_file.mp3 --parallel --max-parallel-workers 4

# Enable both with custom worker counts
transcribe-new audio_file.mp3 -w 4 --parallel --max-parallel-workers 4
```

### Quick Reference

- `-w, --max-workers`: Number of parallel workers for transcription (default: 1)
- `--parallel`: Enable parallel post-processing
- `--max-parallel-workers`: Number of parallel workers for post-processing (default: CPU count)

## Performance Considerations

1. **Small files**: For small audio files (<5 minutes), parallel processing may not provide significant benefits and can sometimes be slower due to overhead.

2. **Medium files**: For medium-sized files (5-30 minutes), parallel transcription can provide moderate performance improvements.

3. **Large files**: For large audio files (>30 minutes), both parallel transcription and parallel post-processing can provide substantial performance improvements.

4. **Worker Count**: The optimal worker count depends on your CPU and available memory. A good starting point is to use half your CPU cores for transcription and half for post-processing.

5. **Memory Usage**: Parallel processing increases memory usage. If you encounter memory issues, reduce the worker count.

## Example Performance Numbers

These are example performance numbers on a system with a 16-core CPU:

| File Size | Sequential | Parallel Transcription | Parallel Both |
|-----------|------------|------------------------|---------------|
| Small     | 10.2s      | 12.6s                  | 10.6s         |
| Medium    | 23.9s      | 28.7s                  | 33.6s         |
| Large     | 240s       | 180s                   | 160s          |

Note: For small and medium files, parallel processing may not provide benefits due to overhead. The advantages become more significant with larger files.

## Implementation Details

### Transcriber

The `Transcriber` class in `transcribe_pkg.core.transcriber` handles parallel transcription:

- It splits audio into chunks based on the `chunk_length_ms` parameter
- If `max_workers > 1`, it processes chunks in parallel using `ThreadPoolExecutor`
- The chunks are then combined in the correct order

### TranscriptProcessor

The `TranscriptProcessor` class in `transcribe_pkg.core.processor` handles parallel post-processing:

- It splits the transcript into chunks for processing
- When `use_parallel=True`, it processes chunks in parallel using `ParallelProcessor`
- It preserves context between chunks for coherent output

## Troubleshooting

If you encounter issues with parallel processing:

1. **High memory usage**: Reduce the worker count with `-w` and `--max-parallel-workers`
2. **Unexpected errors**: Try disabling parallel processing for post-processing first
3. **OpenAI API rate limits**: Reduce the worker count to stay within API rate limits