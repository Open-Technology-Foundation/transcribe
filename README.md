# Transcribe

A robust Python package for high-quality audio transcription using OpenAI's Whisper API with intelligent post-processing via GPT models.

## Features

- **High-Quality Transcription**: Uses OpenAI's Whisper API for accurate audio-to-text conversion
- **Intelligent Post-Processing**: GPT models clean and format transcripts for readability
- **Timestamp Support**: Precise timing information for each speech segment
- **Subtitle Generation**: Create SRT and VTT subtitle files with proper formatting
- **Parallel Processing**: Efficiently process large audio files using multiple workers
- **Content-Aware Processing**: Specialized handling for different content types (technical, medical, etc.)
- **Multi-language Support**: Automatic language detection and processing
- **Flexible Configuration**: Configure via command line, environment variables, or JSON files
- **Smart Caching**: Avoid redundant API calls and reduce costs
- **Robust Error Handling**: Automatic retry mechanisms with graceful degradation

## Installation

### Using the Installation Script (Recommended)

The simplest way to install is using the provided installation script:

```bash
# Clone the repository
git clone https://github.com/Open-Technology-Foundation/transcribe.git
cd transcribe

# Local installation (recommended - installs in current directory)
./install.sh --local

# OR System-wide installation (requires sudo)
sudo ./install.sh --system
```

After installation, create a `.env` file with your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Manual Installation

If you prefer manual installation:

#### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/Open-Technology-Foundation/transcribe.git
cd transcribe

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and package in development mode
pip install -r requirements.txt
pip install -e .

# Make wrapper scripts executable
chmod +x transcribe transcribe-parallel clean-transcript create-sentences
```

#### System-wide Installation

```bash
# Clone to system location
sudo git clone https://github.com/Open-Technology-Foundation/transcribe.git /usr/share/transcribe
cd /usr/share/transcribe

# Create virtual environment and install
sudo python3 -m venv .venv
sudo .venv/bin/pip install -r requirements.txt
sudo .venv/bin/pip install -e .

# Make scripts executable
sudo chmod +x transcribe transcribe-parallel clean-transcript create-sentences

# Create system-wide symlinks
sudo ln -sf /usr/share/transcribe/transcribe /usr/local/bin/transcribe
sudo ln -sf /usr/share/transcribe/transcribe-parallel /usr/local/bin/transcribe-parallel
sudo ln -sf /usr/share/transcribe/clean-transcript /usr/local/bin/clean-transcript
sudo ln -sf /usr/share/transcribe/create-sentences /usr/local/bin/create-sentences
```

## Quick Start

### Basic Transcription

```bash
# Simple transcription
transcribe audio_file.mp3 -o transcript.txt

# With context for better accuracy
transcribe audio_file.mp3 -o transcript.txt -c "medical,technical" -v

# Generate timestamps
transcribe audio_file.mp3 -o transcript.txt -T

# Create SRT subtitles
transcribe audio_file.mp3 --srt

# Create VTT subtitles  
transcribe audio_file.mp3 --vtt
```

### Large File Processing

```bash
# Use optimized parallel processing for large files
transcribe-parallel large_audio.mp3 -o transcript.txt

# Manual parallel configuration
transcribe audio_file.mp3 -w 4 --parallel --max-parallel-workers 4
```

### Post-Processing Tools

```bash
# Clean existing transcript
clean-transcript raw_transcript.txt -o clean_transcript.txt -c "context"

# Format text into sentences
create-sentences transcript.txt -o formatted.txt
```

## Command Line Options

### Transcribe

```
Usage: transcribe [options] audio_file

Output Options:
  -o, --output FILE      Output file (default: input filename with .txt extension)
  -O, --output-to-stdout Output transcription to stdout

Processing Options:
  -P, --no-post-processing   Disable post-processing cleanups (default: enabled)
  -l, --chunk-length MS      Audio chunk length in milliseconds (default: 600000)
  -L, --input-language LANG  Input audio language (default: auto-detect)
  -c, --context TEXT         Context for post-processing (e.g., medical,legal,technical)
  -W, --transcribe-model MODEL Whisper model for transcription (default: whisper-1)
  -m, --model MODEL          GPT model for post-processing (default: gpt-4o)
  -s, --max-chunk-size N     Maximum chunk size for post-processing (default: 3000)
  -t, --temperature N        Temperature for generation (default: 0.05)
  -p, --prompt TEXT          Prompt to guide initial transcription
  -w, --max-workers N        Parallel workers for transcription (default: 1)

Advanced Post-Processing Options:
  --summary-model MODEL      Model for context summarization (default: gpt-4o-mini)
  --auto-context            Auto-determine domain context from content
  --raw                     Save raw transcript before post-processing
  --auto-language           Auto-detect language from transcript content
  --parallel                Enable parallel processing for large transcripts
  --max-parallel-workers N  Parallel workers for post-processing (default: CPU count)
  --cache                   Enable caching for better performance
  --content-aware           Enable content-aware specialized processing
  --clear-cache             Clear cached results before starting

Timestamp and Subtitle Options:
  -T, --timestamps          Include timestamp information in output
  --srt                     Generate SRT subtitle file (enables timestamps)
  --vtt                     Generate VTT subtitle file (enables timestamps)

Logging Options:
  -v, --verbose             Enable verbose output
  -d, --debug               Enable debug output

Examples:
  # Basic transcription
  transcribe audio_file.mp3
  
  # Advanced transcription with all features
  transcribe audio_file.mp3 -o transcript.txt --content-aware --parallel --cache
  
  # Transcription with context
  transcribe audio_file.mp3 -c "medical,technical" -m gpt-4o
  
  # Generate subtitles
  transcribe audio_file.mp3 --srt
  
  # Parallel processing
  transcribe audio_file.mp3 -w 4 --parallel --max-parallel-workers 4
```

### Clean Transcript

```
Usage: clean-transcript [options] input_file

Options:
  -L, --input-language LANG    Input language (translate to English if specified)
  -c, --context TEXT           Domain-specific context (e.g., medical,legal)
  -m, --model MODEL            OpenAI model to use (default: gpt-4o)
  -M, --max-tokens N           Maximum tokens (default: 4096)
  -s, --max-chunk-size N       Maximum chunk size for processing (default: 3000)
  -t, --temperature N          Temperature for generation (default: 0.05)
  -o, --output FILE            Output file (default: stdout)
  -v, --verbose                Enable verbose output
  -d, --debug                  Enable debug output
```

### Create Sentences

```
Usage: create-sentences [options] input_file

Options:
  -o, --output FILE            Output file (default: stdout)
  -p, --paragraphs             Create paragraphs instead of sentences
  -m, --min-sentences N        Minimum sentences per paragraph (default: 2)
  -M, --max-sentences N        Maximum sentences per paragraph (default: 8)
  -s, --max-sentence-length N  Maximum sentence length in bytes (default: 3000)
  -v, --verbose                Enable verbose output
  -d, --debug                  Enable debug output
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_COMPLETION_MODEL`: Default model for post-processing (default: gpt-4o)
- `OPENAI_SUMMARY_MODEL`: Default model for summaries (default: gpt-4o-mini)
- `TRANSCRIBE_CHUNK_LENGTH_MS`: Audio chunk length in milliseconds (default: 600000)
- `TRANSCRIBE_MAX_CHUNK_SIZE`: Maximum text chunk size for processing (default: 3000)
- `TRANSCRIBE_PARALLEL`: Enable parallel processing (true/false, default: true)
- `TRANSCRIBE_MAX_WORKERS`: Maximum number of workers (default: 4)
- `TRANSCRIBE_TEMP_DIR`: Directory for temporary files (default: system temp)
- `TRANSCRIBE_LOG_LEVEL`: Logging level (INFO, DEBUG, etc., default: INFO)

## Configuration File

You can specify configuration options in a JSON file:

```json
{
  "openai": {
    "models": {
      "transcription": "whisper-1",
      "completion": "gpt-4o",
      "summary": "gpt-4o-mini"
    }
  },
  "transcription": {
    "chunk_length_ms": 600000,
    "temperature": 0.05,
    "parallel": true,
    "max_workers": 4
  },
  "processing": {
    "max_chunk_size": 3000,
    "temperature": 0.1,
    "max_tokens": 4096,
    "post_processing": true
  },
  "output": {
    "save_raw": true,
    "create_paragraphs": true,
    "min_sentences_per_paragraph": 2,
    "max_sentences_per_paragraph": 8
  },
  "system": {
    "temp_dir": "",
    "log_level": "INFO"
  }
}
```

## Advanced Usage

### Parallel Processing

For large audio files (>30 minutes), use the optimized parallel wrapper:

```bash
# Automatically configures optimal parallel settings
transcribe-parallel large_audio.mp3 -o transcript.txt
```

Manual parallel configuration:

```bash
# Parallel transcription workers
transcribe audio_file.mp3 -w 4

# Parallel post-processing 
transcribe audio_file.mp3 --parallel --max-parallel-workers 4

# Combined parallel processing
transcribe audio_file.mp3 -w 4 --parallel --max-parallel-workers 4
```

### Advanced Features

```bash
# Content-aware processing (automatically detects content type)
transcribe audio_file.mp3 --content-aware

# Enable caching to reduce API costs
transcribe audio_file.mp3 --cache

# Auto-detect domain context from content
transcribe audio_file.mp3 --auto-context

# All advanced features combined
transcribe audio_file.mp3 --content-aware --parallel --cache --auto-context

# Save raw transcript before post-processing
transcribe audio_file.mp3 --raw --content-aware
```

### Using as a Python Package

```python
from transcribe_pkg.core.transcriber import Transcriber
from transcribe_pkg.core.processor import TranscriptProcessor
from transcribe_pkg.utils.subtitle_utils import save_subtitles

# Initialize transcriber
transcriber = Transcriber(
    model="whisper-1",
    language="en",
    temperature=0.05,
    chunk_length_ms=600000
)

# Initialize processor with advanced features
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
    max_workers=4
)

# Generate subtitles if timestamps are available
if isinstance(transcript, dict) and "segments" in transcript:
    save_subtitles(transcript, "lecture.srt", format_type="srt")

# Process transcript with content-aware analysis
processed_text = processor.process(
    text=transcript["text"] if isinstance(transcript, dict) else transcript,
    context="scientific,physics,education",
    use_parallel=True,
    content_analysis=True
)

# Save results
with open("lecture_processed.txt", "w", encoding="utf-8") as f:
    f.write(processed_text)
```

## Development

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run with coverage report
python run_tests.py -c

# Run specific test module
python run_tests.py -p tests.test_text_utils

# Run specific test method
python run_tests.py -p tests.test_text_utils.TestTextUtils.test_create_sentences_normal
```

### Benchmarking

Performance testing scripts are available in `tests/benchmark/`:

```bash
# Benchmark parallel processing
./tests/benchmark/benchmark-parallel.sh sample_audio.mp3

# Test specific parallel features
python tests/benchmark/test-parallel-flag.py
python tests/benchmark/test-medium-file.py
python tests/benchmark/test-large-file.py
```

For detailed setup and development information, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Follow the coding standards in [DEVELOPMENT.md](DEVELOPMENT.md)
5. Submit a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- Gary Dean
- AI Transcription Team

---

*This README was comprehensively updated to reflect the current codebase*