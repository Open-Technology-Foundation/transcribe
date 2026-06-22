#!/usr/bin/env python3
"""
Main entry point for the transcription package.

This module provides the entry point for running the transcription package
as a Python module with `python -m transcribe_pkg`.
"""

import sys
# Route through main.transcribe_main so the SIGINT/Ctrl-C handler defined at
# import time in transcribe_pkg.main is installed (clean exit with code 130).
from transcribe_pkg.main import transcribe_main

if __name__ == "__main__":
    sys.exit(transcribe_main())

#fin