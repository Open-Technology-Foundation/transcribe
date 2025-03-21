#!/usr/bin/env python3
"""
API utility functions for interacting with OpenAI's services.

This module provides classes and functions for interacting with OpenAI's APIs,
including the Whisper API for audio transcription and GPT models for text processing.
"""
import os
import logging
import json
from typing import Dict, Any, Optional, Union, BinaryIO, List
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from transcribe_pkg.utils.logging_utils import get_logger

class APIError(Exception):
    """Base exception for API-related errors."""
    pass

class APIRateLimitError(APIError):
    """Exception raised when API rate limits are exceeded."""
    pass

class APIAuthenticationError(APIError):
    """Exception raised for authentication issues."""
    pass

class APIConnectionError(APIError):
    """Exception raised for connection issues."""
    pass

class OpenAIClient:
    """
    Client for interacting with OpenAI's APIs with retry logic and error handling.
    """
    
    def __init__(self, api_key: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize client with API key and logger.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            logger: Logger instance for output logging
            
        Raises:
            ValueError: If no API key is provided and none is found in environment
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass as parameter.")
            
        self.logger = logger or get_logger(__name__)
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Configure OpenAI client's HTTP logger to use DEBUG level instead of INFO
        # Try multiple possible logger names used by different OpenAI library versions
        for logger_name in ["openai._client", "openai.http_client", "_client"]:
            openai_logger = logging.getLogger(logger_name)
            if openai_logger:
                openai_logger.setLevel(logging.DEBUG)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((APIRateLimitError, APIConnectionError)),
        reraise=True
    )
    def transcribe_audio(
        self,
        audio_file: Union[str, BinaryIO],
        model: str = "whisper-1",
        prompt: str = "",
        language: str = 'en',
        response_format: str = "text",
        temperature: float = 0.05
    ) -> Any:
        """
        Transcribe audio using OpenAI's Whisper API.
        
        Args:
            audio_file: Path to audio file or file-like object
            model: Model to use for transcription (e.g., "whisper-1")
            prompt: Optional context prompt to guide transcription
            language: Language code (e.g., 'en', 'fr')
            response_format: Format of response ("text" or "verbose_json")
            temperature: Temperature for generation
            
        Returns:
            Transcription result in requested format
            
        Raises:
            APIError: For API-related errors
            FileNotFoundError: If audio file path doesn't exist
            IOError: For file reading errors
        """
        try:
            # Handle file path vs file object
            if isinstance(audio_file, str):
                if not os.path.exists(audio_file):
                    raise FileNotFoundError(f"Audio file not found: {audio_file}")
                    
                with open(audio_file, "rb") as f:
                    return self._transcribe_with_file(f, model, prompt, language, response_format, temperature)
            else:
                # Assume it's already a file-like object
                return self._transcribe_with_file(audio_file, model, prompt, language, response_format, temperature)
                
        except openai.RateLimitError as e:
            self.logger.warning(f"Rate limit exceeded: {str(e)}")
            raise APIRateLimitError(f"OpenAI API rate limit exceeded: {str(e)}")
            
        except openai.AuthenticationError as e:
            self.logger.error(f"Authentication error: {str(e)}")
            raise APIAuthenticationError(f"OpenAI API authentication error: {str(e)}")
            
        except openai.APIConnectionError as e:
            self.logger.warning(f"API connection error: {str(e)}")
            raise APIConnectionError(f"OpenAI API connection error: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {str(e)}")
            raise APIError(f"OpenAI API error: {str(e)}")
    
    def _transcribe_with_file(
        self,
        file_obj: BinaryIO,
        model: str,
        prompt: str,
        language: str,
        response_format: str,
        temperature: float
    ) -> Any:
        """
        Internal method to perform transcription with a file object.
        
        Args:
            file_obj: File-like object containing audio data
            model: Model to use for transcription
            prompt: Context prompt
            language: Language code
            response_format: Format of response
            temperature: Temperature for generation
            
        Returns:
            Transcription result
        """
        try:
            transcription = self.client.audio.transcriptions.create(
                model=model,
                file=file_obj,
                temperature=temperature,
                prompt=prompt,
                language=language,
                response_format=response_format
            )
            
            self.logger.debug(f"HTTP Request: POST https://api.openai.com/v1/audio/transcriptions \"HTTP/1.1 200 OK\"")
            return transcription
        except Exception as e:
            self.logger.error(f"Error in transcription API call: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((APIRateLimitError, APIConnectionError)),
        reraise=True
    )
    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate text using OpenAI's chat completion API.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            model: Model to use (e.g., "gpt-4o", "gpt-4o-mini")
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            APIError: For API-related errors
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None
            )
            
            self.logger.debug(f"HTTP Request: POST https://api.openai.com/v1/chat/completions \"HTTP/1.1 200 OK\"")
            
            if not response.choices or not response.choices[0].message.content.strip():
                self.logger.warning(f"Empty response from API: user_prompt='{user_prompt[:128]}...'")
                return ""
                
            return response.choices[0].message.content.strip()
            
        except openai.RateLimitError as e:
            self.logger.warning(f"Rate limit exceeded: {str(e)}")
            raise APIRateLimitError(f"OpenAI API rate limit exceeded: {str(e)}")
            
        except openai.AuthenticationError as e:
            self.logger.error(f"Authentication error: {str(e)}")
            raise APIAuthenticationError(f"OpenAI API authentication error: {str(e)}")
            
        except openai.APIConnectionError as e:
            self.logger.warning(f"API connection error: {str(e)}")
            raise APIConnectionError(f"OpenAI API connection error: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in chat completion: {str(e)}")
            raise APIError(f"OpenAI API error: {str(e)}")

#fin