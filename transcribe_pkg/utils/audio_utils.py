#!/usr/bin/env python3
"""
Audio utility functions for the transcription package.

This module provides utilities for audio file manipulation, including
splitting audio into smaller chunks for processing large files efficiently.
"""
import os
import logging
import tempfile
from pydub import AudioSegment
from tqdm import tqdm
from typing import List, Optional, Tuple

from transcribe_pkg.utils.logging_utils import get_logger

class AudioProcessor:
    """
    Handles audio file operations including format conversion and splitting.
    """
    
    SUPPORTED_EXTENSIONS = {
        '.mp3': 'mp3',
        '.wav': 'wav',
        '.m4a': 'm4a',
        '.flac': 'flac',
        '.ogg': 'ogg',
        '.wma': 'wma',
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize with optional logger.
        
        Args:
            logger: Logger instance for output logging
        """
        self.logger = logger or get_logger(__name__)
        self.temp_files = []
    
    def __del__(self):
        """Clean up any temporary files on object destruction."""
        self.cleanup()
    
    def cleanup(self):
        """Remove any temporary files created during processing."""
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.logger.debug(f"Removed temporary file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary file {file_path}: {str(e)}")
        self.temp_files = []
    
    def get_audio_format(self, audio_path: str) -> str:
        """
        Determine audio format from file extension.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            The audio format (e.g., 'mp3', 'wav')
            
        Raises:
            ValueError: If file format is not supported
        """
        _, ext = os.path.splitext(audio_path.lower())
        if ext not in self.SUPPORTED_EXTENSIONS:
            supported = ', '.join(self.SUPPORTED_EXTENSIONS.keys())
            raise ValueError(f"Unsupported audio format: {ext}. Supported formats: {supported}")
        return self.SUPPORTED_EXTENSIONS[ext]
    
    def load_audio(self, audio_path: str) -> AudioSegment:
        """
        Load audio file and return AudioSegment.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            AudioSegment object containing the loaded audio
            
        Raises:
            FileNotFoundError: If the audio file does not exist
            ValueError: If the audio file is empty or invalid
            Exception: For other loading errors
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Check if file is empty
        if os.path.getsize(audio_path) == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
            
        self.logger.info(f"Loading audio file: {audio_path}")
        try:
            audio_format = self.get_audio_format(audio_path)
            if audio_format == 'mp3':
                return AudioSegment.from_mp3(audio_path)
            elif audio_format == 'wav':
                return AudioSegment.from_wav(audio_path)
            elif audio_format == 'flac':
                return AudioSegment.from_file(audio_path, format="flac")
            elif audio_format == 'ogg':
                return AudioSegment.from_ogg(audio_path)
            else:
                # Default fallback
                return AudioSegment.from_file(audio_path, format=audio_format)
        except Exception as e:
            self.logger.error(f"Failed to load audio file: {str(e)}")
            if "ffmpeg returned error code" in str(e):
                raise ValueError(f"Invalid audio file or format not supported: {audio_path}. Error: {str(e)}")
            raise
    
    def split_audio(self, audio_path: str, chunk_length_ms: int = 600000) -> List[str]:
        """
        Split audio into chunks of specified length.
        
        Args:
            audio_path: Path to the audio file
            chunk_length_ms: Length of each chunk in milliseconds (default: 10 minutes)
            
        Returns:
            List of paths to temporary chunk files
            
        Raises:
            Various exceptions for file handling and processing errors
        """
        try:
            audio = self.load_audio(audio_path)
            total_length_ms = len(audio)
            chunks = []
            temp_dir = tempfile.mkdtemp()
            
            self.logger.info(f"Splitting audio into {chunk_length_ms/1000:.1f}-second chunks")
            
            # Use tqdm for progress tracking if logging level is INFO or lower
            iterator = tqdm(
                range(0, total_length_ms, chunk_length_ms),
                desc="Splitting audio",
                disable=not self.logger.isEnabledFor(logging.INFO)
            )
            
            for i in iterator:
                chunk = audio[i:i+chunk_length_ms]
                chunk_path = os.path.join(temp_dir, f"chunk_{i//chunk_length_ms}.mp3")
                with open(chunk_path, 'wb') as f:
                    chunk.export(f, format="mp3")
                chunks.append(chunk_path)
                self.temp_files.append(chunk_path)
                
            self.logger.info(f"Audio split into {len(chunks)} chunks")
            return chunks
                
        except Exception as e:
            self.logger.error(f"Error splitting audio: {str(e)}")
            self.cleanup()
            raise
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        audio = self.load_audio(audio_path)
        return len(audio) / 1000.0  # pydub uses milliseconds

#fin