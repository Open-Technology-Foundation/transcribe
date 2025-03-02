#!/usr/bin/env python3
"""
Setup file for the transcribe package.
"""

from setuptools import setup, find_packages

setup(
  name="transcribe",
  version="0.1.0",
  packages=find_packages(),
  install_requires=[
    "colorlog>=6.9.0",
    "langcodes>=3.4.1",
    "nltk>=3.9.1",
    "openai>=1.54.3",
    "pydub>=0.25.1",
    "python-dotenv>=1.0.1",
    "tenacity>=9.0.0",
    "tqdm>=4.67.0",
  ],
  entry_points={
    "console_scripts": [
      "transcribe=transcribe_pkg.main:transcribe_main",
      "clean-transcript=transcribe_pkg.main:clean_transcript_main",
      "create-sentences=transcribe_pkg.main:create_sentences_main",
      "language-codes=transcribe_pkg.main:language_codes_main",
    ],
  },
  python_requires=">=3.8",
  author="AI Transcription Team",
  author_email="your.email@example.com",
  description="A package for transcribing audio files using OpenAI's Whisper API",
  keywords="transcribe, audio, openai, whisper",
)

#fin