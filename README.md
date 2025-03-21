# Transcribe

A robust Python package for high-quality audio transcription using OpenAI's Whisper and post-processing with GPT models.

## Features

- **High-Quality Transcription**: Leverages OpenAI's Whisper model for accurate audio transcription
- **Smart Post-Processing**: Uses OpenAI's GPT models to clean and format transcripts
- **Timestamp Support**: Include precise timing information for each segment of speech
- **Subtitle Generation**: Create SRT and VTT subtitle files for videos
- **Parallel Processing**: Efficiently processes large audio files in parallel
- **Context-Aware**: Apply domain-specific context to improve transcription quality
- **Multi-language Support**: Automatic language detection and support for numerous languages
- **Configurable**: Extensive configuration options via command line, environment variables, or config files
- **Caching**: Intelligent caching to avoid redundant API calls and reduce costs
- **Robust Error Handling**: Graceful degradation with automatic retry mechanisms

## Installation

### Using the Installation Script (Recommended)

The easiest way to install is using the provided installation script:

```bash
# Clone the repository
git clone https://github.com/yourusername/transcribe.git
cd transcribe

# Local installation (in the current directory)
./install.sh --local

# OR System-wide installation (requires sudo)
sudo ./install.sh --system
```

After installation, you may need to create a `.env` file with your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Manual Installation

If you prefer to install manually, follow these steps:

#### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/transcribe.git
cd transcribe

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x transcribe clean-transcript create-sentences
```

#### System-wide Installation

For system-wide installation, you can install the package and create symlinks to the commands:

```bash
# Clone the repository
sudo git clone https://github.com/yourusername/transcribe.git /usr/share/transcribe
cd /usr/share/transcribe

# Create and activate a virtual environment
sudo python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
sudo pip install -r requirements.txt

# Make scripts executable
sudo chmod +x transcribe clean-transcript create-sentences

# Create symlinks in /usr/local/bin
sudo ln -sf /usr/share/transcribe/transcribe /usr/local/bin/transcribe
sudo ln -sf /usr/share/transcribe/clean-transcript /usr/local/bin/clean-transcript
sudo ln -sf /usr/share/transcribe/create-sentences /usr/local/bin/create-sentences
```

## Quick Start

### Transcribe an Audio File

```bash
# Basic transcription
transcribe audio_file.mp3 -o output.txt -c "context about the audio" -m gpt-4o -v

# Transcription with timestamps
transcribe audio_file.mp3 -o output.txt -T -v

# Generate SRT subtitles
transcribe audio_file.mp3 -o output.txt --srt -v

# Generate VTT subtitles
transcribe audio_file.mp3 -o output.txt --vtt -v
```

### Clean a Transcript

```bash
clean-transcript input.txt -o output.txt -c "context information" -m gpt-4o
```

### Create Well-Formed Sentences

```bash
create-sentences input.txt -o output.txt
```

## Command Line Options

### Transcribe

```
Usage: transcribe [options] audio_file

Output Options:
  -o, --output FILE      Output file (default: input filename with .txt extension)
  -O, --output-to-stdout Output the transcription to stdout

Processing Options:
  -l, --chunk-length MS  Length of audio chunks in milliseconds (default: 600000)
  -L, --input-language   Define the language used in the input audio (auto-detects if not specified)
  -c, --context TEXT     Context information to improve transcription
  -W, --transcribe-model MODEL Model for transcription (default: whisper-1)
  -m, --model MODEL      GPT model for post-processing (default: gpt-4o)
  -s, --max-chunk-size N Maximum chunk size for post-processing (default: 3000)
  -t, --temperature N    Temperature for generation (default: 0.05)
  -p, --prompt TEXT      Provide a prompt to guide the initial transcription
  -P, --no-post-processing Skip post-processing step
  -w, --max-workers N    Maximum number of parallel workers for transcription (default: 1)

Advanced Post-Processing Options:
  --summary-model MODEL  OpenAI model to use for context summarization (default: gpt-4o-mini)
  --auto-context         Automatically determine domain context from transcript content
  --raw                  Save the raw transcript before post-processing
  --auto-language        Auto-detect language from transcript content
  --parallel             Enable parallel processing for large transcripts
  --max-parallel-workers N Maximum number of parallel workers for post-processing
  --cache                Enable caching of processing results for better performance
  --content-aware        Enable content-aware specialized processing
  --clear-cache          Clear any cached processing results before starting

Timestamp and Subtitle Options:
  -T, --timestamps       Include timestamp information in the output
  --srt                  Generate SRT subtitle file (enables timestamps automatically)
  --vtt                  Generate VTT subtitle file (enables timestamps automatically)

Logging Options:
  -v, --verbose          Verbose output
  -d, --debug            Debug output

Examples:
  # Basic transcription
  transcribe-new audio_file.mp3
  
  # Advanced transcription with processing options
  transcribe-new audio_file.mp3 -o transcript.txt --content-aware --parallel --cache
  
  # Transcription with context hints
  transcribe-new audio_file.mp3 -c "philosophy,science" -m gpt-4o
  
  # Subtitles generation
  transcribe-new audio_file.mp3 --srt
  
  # Use custom models
  transcribe-new audio_file.mp3 -W gpt-4o-mini-transcribe -m gpt-4o
```

### Clean Transcript

```
Usage: clean-transcript [options] input_file

Options:
  -o, --output FILE      Output file (default: stdout)
  -c, --context TEXT     Context information to improve cleaning
  -m, --model MODEL      GPT model for processing (default: gpt-4o)
  -t, --temperature N    Temperature for generation (default: 0.1)
  -v, --verbose          Verbose output
  --debug                Debug output
  --config FILE          Configuration file
```

### Create Sentences

```
Usage: create-sentences [options] input_file

Options:
  -o, --output FILE      Output file (default: stdout)
  --min-sentences N      Minimum sentences per paragraph (default: 2)
  --max-sentences N      Maximum sentences per paragraph (default: 8)
  -v, --verbose          Verbose output
  --debug                Debug output
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_COMPLETION_MODEL`: Default model for completions
- `OPENAI_SUMMARY_MODEL`: Default model for summaries
- `TRANSCRIBE_CHUNK_LENGTH_MS`: Chunk length in milliseconds (default: 600000)
- `TRANSCRIBE_MAX_CHUNK_SIZE`: Maximum chunk size for processing
- `TRANSCRIBE_TEMP_DIR`: Directory for temporary files
- `TRANSCRIBE_LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `TRANSCRIBE_NO_CACHE`: Disable caching (set to "1")
- `TRANSCRIBE_CACHE_DIR`: Custom cache directory
- `TRANSCRIBE_CACHE_SIZE_MB`: Maximum cache size in MB
- `TRANSCRIBE_DEFAULT_SUBTITLE_FORMAT`: Default subtitle format (srt or vtt)
- `TRANSCRIBE_MAX_LINE_LENGTH`: Maximum line length for subtitles (default: 42)
- `TRANSCRIBE_MAX_SUBTITLE_DURATION`: Maximum duration for a single subtitle (default: 5.0)

## Configuration File

You can specify configuration options in a JSON file:

```json
{
  "openai": {
    "models": {
      "completion": "gpt-4o",
      "summary": "gpt-4o-mini"
    }
  },
  "transcription": {
    "chunk_length_ms": 600000
  },
  "processing": {
    "max_chunk_size": 3000,
    "temperature": 0.1
  },
  "subtitles": {
    "default_format": "srt",
    "max_line_length": 42,
    "max_duration": 5.0
  }
}
```

## Advanced Usage

### Parallel Processing

For optimal performance with large audio files, you can use the `transcribe-parallel` script which automatically configures parallel processing:

```bash
# Use the optimized parallel wrapper script
transcribe-parallel audio_file.mp3 -o output.txt
```

For detailed information on parallel processing options and performance considerations, see [PARALLEL_PROCESSING.md](PARALLEL_PROCESSING.md).

### Advanced Command Line Features

```bash
# Enable parallel transcription with specific worker count
transcribe audio_file.mp3 -w 4

# Enable parallel post-processing 
transcribe audio_file.mp3 --parallel --max-parallel-workers 4

# Enable both parallel transcription and post-processing
transcribe audio_file.mp3 -w 4 --parallel

# Use content-aware processing for better handling of specific content types
transcribe audio_file.mp3 --content-aware

# Enable caching to avoid redundant API calls
transcribe audio_file.mp3 --cache

# All advanced features combined
transcribe audio_file.mp3 --parallel --content-aware --cache --auto-context
```

### Using as a Python Package

```python
from transcribe_pkg.core.transcriber import Transcriber
from transcribe_pkg.core.processor import TranscriptProcessor
from transcribe_pkg.utils.subtitle_utils import save_subtitles

# Initialize components
transcriber = Transcriber(model="whisper-1")
processor = TranscriptProcessor(
    model="gpt-4o",
    summary_model="gpt-4o-mini",
    max_chunk_size=3000,
    cache_enabled=True,
    content_aware=True
)

# Transcribe audio file
transcript = transcriber.transcribe(
    audio_path="lecture.mp3",
    prompt="University lecture on quantum physics",
    with_timestamps=True,
    max_workers=4  # Enable parallel transcription
)

# Save subtitles
save_subtitles(transcript, "lecture.srt", format_type="srt")

# Process the transcript with advanced features
processed_text = processor.process(
    text=transcript["text"],
    context="University lecture on quantum physics",
    use_parallel=True,
    content_analysis=True
)

# Save processed text
with open("lecture_processed.txt", "w") as f:
    f.write(processed_text)
```

## Development

### Running Tests

```bash
python run_tests.py
python run_tests.py --coverage  # Generate coverage report
python run_tests.py --pattern "test_config"  # Run specific tests
```

### Benchmarking

For performance testing and benchmarking of parallel processing features, check out the scripts in the `tests/benchmark` directory:

```bash
# Run the benchmark script with a sample audio file
tests/benchmark/benchmark-parallel.sh path/to/audio_file

# Test parallel flag implementation
tests/benchmark/test-parallel-flag.py
```

See [tests/benchmark/README.md](tests/benchmark/README.md) for more details on benchmarking tools.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- Gary Dean
- Claude Code 0.2.30

---

*This README was last updated on March 21st, 2025*

## Recent Changes
- Added optimized parallel processing wrapper script `transcribe-parallel`
- Fixed HTTP request logs appearing at INFO level (now only shown in debug mode)
- Improved parallel processing implementation for better performance