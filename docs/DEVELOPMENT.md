# Transcribe Development Guide

This document provides comprehensive information for developers contributing to the Transcribe project.

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- OpenAI API key

### Installation for Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Open-Technology-Foundation/transcribe.git
   cd transcribe
   ```

2. **Set up local development environment**:
   ```bash
   # Install using the local installation script
   ./install.sh --local
   
   # Activate the virtual environment
   source .venv/bin/activate
   
   # Install in development mode (editable install)
   pip install -e .
   ```

3. **Configure environment**:
   ```bash
   # Create .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

### Manual Development Setup

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Make scripts executable
chmod +x transcribe clean-transcript create-sentences transcribe-parallel
```

## Project Architecture

### Core Components

- **`transcribe_pkg/core/`**: Core transcription and processing logic
  - `transcriber.py`: Main transcription engine with audio processing
  - `processor.py`: Post-processing and text improvement
  - `parallel.py`: Parallel processing implementation
  - `analyzer.py`: Content analysis and optimization
  - `examples.py`: Example configurations and usage patterns

- **`transcribe_pkg/utils/`**: Utility modules
  - `api_utils.py`: OpenAI API client and wrapper functions
  - `audio_utils.py`: Audio file processing and chunking
  - `text_utils.py`: Text processing and manipulation utilities
  - `config.py`: Configuration management
  - `logging_utils.py`: Logging setup and utilities
  - `subtitle_utils.py`: SRT and VTT subtitle file generation
  - `cache.py`: Caching system for API responses
  - `progress.py`: Progress display utilities
  - `prompts.py`: Prompt management and templates
  - `language_utils.py`: Language detection and utilities

- **`transcribe_pkg/cli/`**: Command-line interface
  - `commands.py`: CLI command implementations

### Command-Line Scripts

- `transcribe`: Main transcription command
- `transcribe-parallel`: Optimized parallel processing wrapper
- `clean-transcript`: Post-process existing transcripts
- `create-sentences`: Format text into well-formed sentences

## Testing

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run specific test pattern
python run_tests.py -p tests.test_text_utils

# Run with coverage report
python run_tests.py -c

# Run specific test method
python run_tests.py -p tests.test_text_utils.TestTextUtils.test_create_sentences
```

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **API Tests**: Test OpenAI API integration (with mocking)

### Adding New Tests

1. Create test files in the `tests/` directory
2. Follow the naming convention: `test_<module_name>.py`
3. Use the `unittest` framework
4. Include proper docstrings and meaningful test method names
5. Mock external dependencies (OpenAI API calls)

## Code Style and Standards

### Python Code Style

- **Indentation**: 2 spaces (not tabs)
- **Shebang**: Always use `#!/usr/bin/env python3`
- **Import Order**: 
  1. Standard library imports
  2. Third-party imports
  3. Local module imports
- **Constants**: Use UPPER_CASE for constants defined at module level
- **Type Hints**: Use for function parameters and return values
- **Docstrings**: Required for all public functions and classes
- **Script Ending**: End all scripts with `\n#fin\n`

### Example Function Format

```python
def process_audio_chunk(
    audio_path: str,
    chunk_size: int = 3000,
    language: Optional[str] = None
) -> Dict[str, Any]:
  """
  Process an audio chunk for transcription.
  
  Args:
      audio_path: Path to the audio file
      chunk_size: Maximum size of audio chunk in bytes
      language: Language code for transcription (auto-detect if None)
      
  Returns:
      Dictionary containing transcription results and metadata
      
  Raises:
      APIError: If transcription API call fails
      FileNotFoundError: If audio file doesn't exist
  """
  # Implementation here
  pass
```

### Shell Script Style

- **Shebang**: `#!/bin/bash`
- **Error Handling**: Always start with `set -euo pipefail`
- **Indentation**: 2 spaces
- **Variables**: Always declare with `declare` statements
- **Conditionals**: Prefer `[[` over `[`
- **Script Ending**: End all scripts with `\n#fin\n`

## Error Handling

### Exception Hierarchy

- `APIError`: Base exception for API-related errors
- `APIRateLimitError`: Rate limit exceeded
- `APIAuthenticationError`: Authentication issues
- `APIConnectionError`: Connection problems
- `EmptyResponseError`: Empty API responses
- `AudioTranscriptionError`: Audio transcription failures

### Best Practices

- Use specific exception types for different error conditions
- Include meaningful error messages with context
- Log errors appropriately (use the logging utilities)
- Implement retry mechanisms for transient failures (using tenacity)

## Configuration Management

The project uses a flexible configuration system supporting:

- Command-line arguments
- Environment variables  
- JSON configuration files
- Default values

### Configuration Precedence

1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration file values
4. Default values (lowest priority)

### Environment Variables

Key environment variables:

- `OPENAI_API_KEY`: OpenAI API key (required)
- `OPENAI_COMPLETION_MODEL`: Default model for post-processing (default: gpt-4o)
- `OPENAI_SUMMARY_MODEL`: Default model for summaries (default: gpt-4o-mini)
- `TRANSCRIBE_CHUNK_LENGTH_MS`: Audio chunk length in milliseconds (default: 600000)
- `TRANSCRIBE_MAX_CHUNK_SIZE`: Text chunk size for processing (default: 3000)
- `TRANSCRIBE_PARALLEL`: Enable parallel processing (true/false, default: true)
- `TRANSCRIBE_MAX_WORKERS`: Maximum number of workers (default: 4)
- `TRANSCRIBE_TEMP_DIR`: Directory for temporary files (default: system temp)
- `TRANSCRIBE_LOG_LEVEL`: Logging level (INFO, DEBUG, etc., default: INFO)

## Performance Optimization

### Parallel Processing

The project supports parallel processing for:

- Audio transcription (multiple workers for different audio chunks)
- Text post-processing (parallel processing of text chunks)

Use `transcribe-parallel` for large files (>30 minutes) or enable manually:

```bash
# Parallel transcription
transcribe audio.mp3 -w 4

# Parallel post-processing  
transcribe audio.mp3 --parallel --max-parallel-workers 4

# Both
transcribe audio.mp3 -w 4 --parallel
```

### Caching

Enable caching to avoid redundant API calls:

```bash
transcribe audio.mp3 --cache
```

## Debugging

### Logging Levels

- `--verbose (-v)`: INFO level logging
- `--debug (-d)`: DEBUG level logging

### Common Issues

1. **API Rate Limits**: Use retry mechanisms and reasonable request rates
2. **Large Files**: Use parallel processing and appropriate chunk sizes
3. **Memory Usage**: Monitor memory consumption with large audio files
4. **API Key Issues**: Ensure valid OpenAI API key is configured

## Contributing

### Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes following the code style guidelines
4. Add tests for new functionality
5. Run the test suite: `python run_tests.py`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

### Pull Request Guidelines

- Include a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Update documentation if needed
- Follow the established code style

### Commit Message Format

Use clear, concise commit messages:

```
type: brief description

Optional longer description explaining the change.

- List any breaking changes
- Reference issues: Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Release Process

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Create release branch
5. Tag release: `git tag v1.0.0`
6. Push tags: `git push --tags`

## Troubleshooting

### Common Development Issues

1. **Import Errors**: Ensure package is installed in development mode (`pip install -e .`)
2. **Test Failures**: Check that virtual environment is activated and dependencies are installed
3. **API Errors**: Verify OpenAI API key is valid and has sufficient credits
4. **Permission Issues**: Ensure scripts are executable (`chmod +x script_name`)

### Performance Testing

Use the benchmark scripts in `tests/benchmark/` for performance testing:

```bash
# Run benchmark with sample audio
tests/benchmark/benchmark-parallel.sh path/to/audio_file

# Test parallel flag implementation
tests/benchmark/test-parallel-flag.py
```

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Support

For development questions:
- Open an issue on GitHub
- Check existing documentation
- Review test examples for usage patterns

---

*This development guide was updated for accuracy during comprehensive documentation overhaul*

#fin