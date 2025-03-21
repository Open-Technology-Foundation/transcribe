#!/usr/bin/env python3
"""
Main entry point for the transcription package.

This module provides the entry point for running the transcription package
as a Python module with `python -m transcribe_pkg`.
"""

import sys
from transcribe_pkg.cli.commands import transcribe_command

if __name__ == "__main__":
    sys.exit(transcribe_command())

#fin