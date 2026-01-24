# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python package for high-quality audio transcription using OpenAI's Whisper API with intelligent post-processing via LLM models. Features multi-provider LLM support (OpenAI, Anthropic, Gemini, Ollama), parallel processing, subtitle generation, content-aware processing, and multi-language support.

## Build/Installation/Test Commands

- Install: `./install.sh --local` (local) or `sudo ./install.sh --system` (system-wide)
- Activate venv: `source .venv/bin/activate` (for local install)
- Development install: `pip install -e .` (within activated venv)
- Run all tests: `python run_tests.py`
- Run specific test: `python run_tests.py -p tests.test_text_utils.TestTextUtils.test_create_sentences_normal`
- Run tests with coverage: `python run_tests.py -c`
- Basic usage: `./transcribe audio_file.mp3 -o output.txt`
- Clean transcript: `./clean-transcript raw.txt -o clean.txt`

## Default Models

Defined in `transcribe_pkg/constants.py`:

| Constant | Model | Purpose |
|----------|-------|---------|
| `DEFAULT_LLM_MODEL` | `claude-sonnet-4-5` | Main text processing |
| `DEFAULT_SUMMARY_MODEL` | `claude-haiku-4-5` | Language detection, context extraction |
| `DEFAULT_WHISPER_MODEL` | `whisper-1` | Audio transcription (OpenAI) |

## Multi-Provider LLM Support

The package supports multiple LLM providers via automatic model prefix routing:

| Prefix | Provider |
|--------|----------|
| `gpt-*`, `o1-*`, `o3-*` | OpenAI |
| `claude-*` | Anthropic |
| `gemini-*` | Google Gemini |
| `ollama/*`, `llama*`, `mistral*` | Ollama (local) |

### Provider System Files

Located in `transcribe_pkg/utils/providers/`:
- `base.py` - LLMClientProtocol interface
- `registry.py` - Provider routing and client caching
- `openai_client.py`, `anthropic_client.py`, `gemini_client.py`, `ollama_client.py`

### Usage Examples

```bash
# Anthropic Claude (default)
clean-transcript raw.txt -m claude-sonnet-4-5

# OpenAI
clean-transcript raw.txt -m gpt-4o

# Google Gemini
clean-transcript raw.txt -m gemini-1.5-pro

# Local Ollama
clean-transcript raw.txt -m ollama/llama3.2

# Override provider detection
clean-transcript raw.txt -m custom-model --provider anthropic
```

## Architecture

### Core Processing Pipeline

1. **Transcription** (`transcribe_pkg/core/transcriber.py`):
   - Splits audio into chunks using AudioProcessor
   - Transcribes via OpenAI Whisper API with retry mechanisms
   - Supports parallel transcription with ThreadPoolExecutor

2. **Post-Processing** (`transcribe_pkg/core/processor.py`):
   - Breaks text into processable chunks with context preservation
   - Cleans/improves text using LLM models (multi-provider)
   - Supports parallel processing via ParallelProcessor
   - Accepts `provider` parameter for explicit provider override

3. **Content Analysis** (`transcribe_pkg/core/analyzer.py`):
   - Auto-detects content type (technical, dialogue, speech, lecture)
   - Applies specialized processing strategies per content type
   - Uses `DEFAULT_SUMMARY_MODEL` for fast classification

4. **Parallel Processing** (`transcribe_pkg/core/parallel.py`):
   - Distributes work across multiple threads/processes
   - Handles chunk overlap to maintain context

### Key Utilities

- `api_utils.py`: Multi-provider LLM interface via `call_llm()`, OpenAI client for audio
- `providers/registry.py`: Provider routing based on model prefix
- `audio_utils.py`: Audio splitting/processing with pydub
- `text_utils.py`: Text chunking, sentence/paragraph creation
- `subtitle_utils.py`: SRT/VTT subtitle file generation
- `cache.py`: API response caching to reduce costs
- `prompts.py`: Prompt template management (provider-aware)
- `config.py`: Configuration via CLI/env vars/JSON files
- `constants.py`: Centralized constants including default models

### Entry Points

- CLI interfaces defined in `transcribe_pkg/cli/commands.py`
- Main entry point: `transcribe_pkg/main.py`
- Wrapper scripts: `transcribe`, `transcribe-parallel`, `clean-transcript`, `create-sentences`

## Environment Variables

```bash
# Required for audio transcription
OPENAI_API_KEY=sk-...

# For LLM providers (based on model used)
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434  # Optional, defaults to localhost
```

## Code Style

### Python
- 2-space indentation, shebang `#!/usr/bin/env python3`
- Type hints: Python 3.12+ syntax (`list[T]`, `X | None`)
- Constants in `constants.py` with UPPER_CASE names
- End files with `#fin` comment
- Use tenacity retry for API calls

### Shell Scripts
- Shebang `#!/usr/bin/env bash`
- Always `set -euo pipefail`
- 2-space indentation
- End with `#fin` comment

## Important Implementation Details

### The `call_llm()` Function

All LLM calls route through `transcribe_pkg/utils/api_utils.py:call_llm()`:

```python
call_llm(
    user_prompt="...",
    system_prompt="...",
    model="claude-sonnet-4-5",  # Provider auto-detected
    temperature=0.1,
    max_tokens=4096,
    provider=None  # Optional override
)
```

### Configuration Priority

1. Default values from `constants.py`
2. Environment variables
3. JSON config file (if specified)
4. Command-line arguments (highest priority)

### API Retry Logic

OpenAI API calls use tenacity with exponential backoff:
- Max 3 attempts
- Exponential backoff factor 2
- Initial delay 1 second

#fin
