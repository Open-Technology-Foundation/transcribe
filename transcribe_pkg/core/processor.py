#!/usr/bin/env python3
"""
Transcript post-processing module.

This module provides the core functionality for post-processing transcripts
using AI models to improve quality and readability. It handles:

- Breaking text into processable chunks
- Processing chunks with AI models for grammar and clarity improvements
- Maintaining context across chunks
- Combining processed chunks into a cohesive final transcript

The post-processing improves readability, fixes grammar, removes hesitations,
and creates logical paragraphs while preserving the original meaning.
"""

import logging
from typing import Any
import sys

from transcribe_pkg.utils.api_utils import OpenAIClient, APIError, call_llm
from transcribe_pkg.utils.logging_utils import get_logger
from transcribe_pkg.utils.text_utils import create_sentences, create_paragraphs
from transcribe_pkg.utils.prompts import PromptManager
from transcribe_pkg.utils.progress import ProgressDisplay
from transcribe_pkg.utils.cache import CacheManager, cached
from transcribe_pkg.core.parallel import ParallelProcessor
from transcribe_pkg.core.analyzer import ContentAnalyzer, SpecializedProcessor

class TranscriptProcessor:
    """
    Process transcripts using AI models to improve quality and readability.
    
    This class handles the complete post-processing workflow including:
    - Breaking text into processable chunks
    - Cleaning and improving text with AI
    - Context generation for continuing coherence
    - Reconstruction into a final polished transcript
    """
    
    def __init__(
        self,
        api_client: OpenAIClient | None = None,
        model: str = "gpt-4o",
        summary_model: str = "gpt-4o-mini",
        temperature: float = 0.05,
        max_tokens: int = 4096,
        max_chunk_size: int = 3000,
        max_workers: int | None = None,
        cache_enabled: bool = True,
        content_aware: bool = True,
        logger: logging.Logger | None = None,
        prompt_manager: PromptManager | None = None
    ):
        """
        Initialize transcript processor with configuration.
        
        Args:
            api_client: OpenAI client for API calls (creates new one if None)
            model: Main model for transcript processing
            summary_model: Model for generating context summaries
            temperature: Temperature for text generation
            max_tokens: Maximum tokens for API response
            max_chunk_size: Maximum size for each chunk in bytes
            max_workers: Maximum number of parallel workers (default: CPU count)
            cache_enabled: Enable caching of API responses
            content_aware: Enable content-aware processing
            logger: Logger instance
            prompt_manager: PromptManager instance for handling prompts
        """
        self.logger = logger or get_logger(__name__)
        self.api_client = api_client or OpenAIClient(logger=self.logger)
        self.model = model
        self.summary_model = summary_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_chunk_size = max_chunk_size
        self.cache_enabled = cache_enabled
        self.content_aware = content_aware
        
        # Set up prompt manager
        self.prompt_manager = prompt_manager or PromptManager(api_client=self.api_client, logger=self.logger)
        
        # Set up content analyzer if content-aware processing is enabled
        if self.content_aware:
            self.content_analyzer = ContentAnalyzer(
                api_client=self.api_client,
                prompt_manager=self.prompt_manager,
                model=self.summary_model,
                logger=self.logger
            )
            self.specialized_processor = SpecializedProcessor(
                api_client=self.api_client,
                analyzer=self.content_analyzer,
                logger=self.logger
            )
        
        # Set up cache manager if caching is enabled
        if self.cache_enabled:
            self.cache_manager = CacheManager(logger=self.logger)
            
        # Set up parallel processor
        self.parallel_processor = ParallelProcessor(
            max_workers=max_workers,
            use_processes=False,  # Use threads for better memory sharing
            chunk_size=max_chunk_size,
            overlap=500,  # Overlap between chunks to maintain context
            logger=self.logger
        )
        
    def process(
        self,
        text: str,
        context: str = "",
        language: str | None = None,
        use_parallel: bool = True,
        content_analysis: bool = True
    ) -> str:
        """
        Process the transcript text.
        
        Args:
            text: The raw transcript text
            context: Domain context hints (e.g., "medical,technical")
            language: Language code, auto-detected if None
            use_parallel: Use parallel processing for large texts
            content_analysis: Use content analysis for specialized processing
            
        Returns:
            Processed and improved transcript text
        """
        text = text.rstrip()
        total_length = len(text)
        
        self.logger.info(f"Processing transcript of {total_length} bytes")
        self.logger.info(f"Using context: {context if context else 'None'}")
        self.logger.info(f"Using language: {language if language else 'Auto-detect'}")
        
        # Detect language if not specified
        if language is None:
            # Use prompt manager to detect language
            sample_text = text[:1000]  # Use first 1000 chars to detect language
            
            # Try to get from cache first
            if self.cache_enabled:
                cache_key = f"language_detection:{hash(sample_text)}"
                cached_language = self.cache_manager.get(cache_key)
                if cached_language:
                    language = cached_language
                    self.logger.info(f"Using cached language detection: {language}")
                else:
                    language = self.prompt_manager.detect_language(sample_text)
                    self.cache_manager.set(cache_key, language)
                    self.logger.info(f"Auto-detected language: {language}")
            else:
                language = self.prompt_manager.detect_language(sample_text)
                self.logger.info(f"Auto-detected language: {language}")
        
        # Check if parallel processing is explicitly requested
        if use_parallel:
            self.logger.info(f"Using parallel processing with {self.parallel_processor.max_workers} workers")
            
            # For very small texts, parallel processing might not be efficient
            # But if user explicitly requested it, we'll honor that request
            if total_length <= self.max_chunk_size:
                self.logger.info("Text is small, but using parallel processing as requested")
                # Split the text into at least 2 chunks for parallel processing
                # Create an artificial split point at around 40% of the text
                split_point = max(1, int(total_length * 0.4))
                chunks = [text[:split_point], text[split_point:]]
                
                # Process chunks in parallel
                results = []
                for i, chunk in enumerate(chunks):
                    if content_analysis and self.content_aware:
                        results.append(self._process_with_content_analysis(chunk, context, language))
                    else:
                        results.append(self._process_chunk(chunk, context, None, language))
                
                # Combine results
                result = "\n\n".join(results)
            else:
                # For larger texts, use regular parallel processing
                result = self._process_parallel(text, context, language, content_analysis)
        
        # If parallel processing is not requested, or text is small enough
        elif total_length <= self.max_chunk_size:
            self.logger.info(f"Text is small enough for single-chunk processing")
            
            if content_analysis and self.content_aware:
                # Use content-aware processing
                self.logger.info("Using content-aware processing")
                result = self._process_with_content_analysis(text, context, language)
            else:
                # Use standard processing
                self.logger.info("Using standard processing")
                result = self._process_chunk(text, context, None, language)
        
        # For larger texts with sequential processing
        else:
            self.logger.info("Using sequential processing")
            result = self._process_sequential(text, context, language, content_analysis)
            
        self.logger.info("Transcript processing complete")
        return result
        
    def _process_with_content_analysis(self, text: str, context: str, language: str) -> str:
        """
        Process text using content-aware specialized processing.
        
        Args:
            text: Text to process
            context: Domain context
            language: Language code
            
        Returns:
            Processed text
        """
        # Use the specialized processor with content analysis
        return self.specialized_processor.process_content(
            text=text,
            context=context,
            language=language,
            model=self.model,
            temperature=self.temperature
        )
        
    def _process_parallel(self, text: str, context: str, language: str, content_analysis: bool) -> str:
        """
        Process text in parallel chunks.
        
        Args:
            text: Text to process
            context: Domain context
            language: Language code
            content_analysis: Use content analysis for specialized processing
            
        Returns:
            Processed text
        """
        # Analyze content if enabled
        content_type = None
        specialized_prompt = None
        
        if content_analysis and self.content_aware:
            # Analyze a sample of the text
            sample_length = min(len(text), 3000)
            sample = text[:sample_length]
            
            analysis = self.content_analyzer.analyze_content(sample)
            content_type = analysis.get("content_type", "general")
            
            # Get specialized prompt based on content type
            specialized_prompt = self.content_analyzer.get_specialized_prompt(
                analysis, context=context, purpose="clean"
            )
            
            self.logger.info(f"Content analysis: type={content_type}, language={analysis.get('language', language)}")
            
            # Use detected language if none provided
            if language is None:
                language = analysis.get("language", "en")
        
        # Define the chunk processing function
        # This function is passed to the parallel processor
        def process_chunk(chunk: str, kwargs: dict[str, Any]) -> str:
            # Extract any additional kwargs
            chunk_context = kwargs.get("context", context)
            chunk_index = kwargs.get("chunk_index", 0)
            prev_context = kwargs.get("prev_context")
            
            # Create a cache key for this chunk if caching is enabled
            if self.cache_enabled:
                cache_key = f"chunk_processing:{hash(chunk)}:{hash(chunk_context)}:{language}:{self.model}:{self.temperature}"
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    self.logger.debug(f"Using cached result for chunk {chunk_index}")
                    return cached_result
            
            # Process chunk based on content type
            if specialized_prompt and content_analysis:
                try:
                    result = call_llm(
                        user_prompt=chunk,
                        system_prompt=specialized_prompt,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    # Cache the result if enabled
                    if self.cache_enabled:
                        self.cache_manager.set(cache_key, result)
                        
                    return result
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
                    return chunk  # Fallback to original text on error
            else:
                # Use standard processing
                result = self._process_chunk(chunk, chunk_context, prev_context, language)
                
                # Cache the result if enabled
                if self.cache_enabled:
                    self.cache_manager.set(cache_key, result)
                    
                return result
        
        # Use parallel processor to process chunks
        chunks = self.parallel_processor.process_text(
            text=text,
            process_func=process_chunk,
            # Define a function to combine chunks
            combine_func=lambda chunks: "\n\n".join([c.strip() for c in chunks]),
            show_progress=True,
            context=context,
            prev_context=None
        )
        
        return chunks
    
    def _process_sequential(self, text: str, context: str, language: str, content_analysis: bool) -> str:
        """
        Process text sequentially chunk by chunk.
        
        Args:
            text: Text to process
            context: Domain context
            language: Language code
            content_analysis: Use content analysis for specialized processing
            
        Returns:
            Processed text
        """
        text = text.rstrip()
        total_length = len(text)
        processed_length = 0
        generated_text = ""
        remaining_text = text
        iterations = 0
        iteration_limit = int((total_length / self.max_chunk_size) * 2)
        context_summary = None
        
        # Analyze content if enabled
        if content_analysis and self.content_aware:
            # Analyze a sample of the text
            sample_length = min(len(text), 3000)
            sample = text[:sample_length]
            
            analysis = self.content_analyzer.analyze_content(sample)
            content_type = analysis.get("content_type", "general")
            
            self.logger.info(f"Content analysis: type={content_type}, language={analysis.get('language', language)}")
            
            # Use detected language if none provided
            if language is None:
                language = analysis.get("language", "en")
        
        # Create progress display
        progress = ProgressDisplay(
            total=total_length,
            description="Processing transcript",
            unit="bytes",
            logger=None  # Don't use logger for progress updates - use direct console output
        )
        
        while remaining_text:
            # Store the initial length of the remaining text
            initial_length = len(remaining_text)
            
            # Get the next chunk with complete sentences
            chunk, remaining_text = self._get_chunk_with_complete_sentences(
                remaining_text, self.max_chunk_size
            )
            
            # Process the chunk
            self.logger.debug(f"Processing chunk of {len(chunk)} bytes")
            
            if content_analysis and self.content_aware:
                # Use specialized processing
                processed_chunk = self.specialized_processor.process_content(
                    text=chunk,
                    context=context,
                    language=language,
                    model=self.model,
                    temperature=self.temperature
                )
            else:
                # Use standard processing
                processed_chunk = self._process_chunk(
                    chunk, context, context_summary, language
                )
            
            # Add the processed chunk to the generated text
            if generated_text and generated_text[-1] in '.,?!`"':
                generated_text += ' '
            generated_text += processed_chunk
            
            # Update context summary from the processed chunk
            # Use last few paragraphs to create a summary for context continuity
            paragraphs = processed_chunk.strip().split('\n\n')
            context_text = '\n\n'.join(paragraphs[-7:])
            context_summary = self.prompt_manager.generate_summary(context_text, model=self.summary_model)
            
            # Update the processed length
            processed_length += initial_length - len(remaining_text)
            
            # Update progress
            progress.update(processed_length)
            self.logger.debug(f"Processed {processed_length}/{total_length} bytes")
            
            # Break if finished or hitting iteration limit
            if processed_length >= total_length or remaining_text == '' or processed_length / total_length >= 0.9985:
                break
                
            iterations += 1
            if iterations > iteration_limit:
                self.logger.error(f"Too many iterations! Stopping at {processed_length}/{total_length} bytes complete")
                break
        
        # Ensure progress display is complete
        progress.complete()
        
        return generated_text
    
    def _add_oxford_comma(self, context_str: str) -> str:
        """
        Add Oxford comma to a comma-separated list of items.
        
        Args:
            context_str: Comma-separated string
            
        Returns:
            String with Oxford comma added if applicable
        """
        if not context_str:
            return context_str
            
        words = context_str.split(', ')
        if len(words) <= 1:
            return context_str
        elif len(words) == 2:
            return ' and '.join(words)
            
        return ', '.join(words[:-1]) + ', and ' + words[-1]
    
    def _process_chunk(
        self,
        chunk: str,
        context: str,
        context_summary: str | None,
        language: str
    ) -> str:
        """
        Process a transcript chunk with AI for improvement.
        
        Args:
            chunk: Text chunk to process
            context: Domain-specific context
            context_summary: Summary of previous chunks for continuity
            language: Language code
            
        Returns:
            Processed text
        """
        # Get system prompt from the prompt manager
        system_prompt = self.prompt_manager.get_system_prompt(
            template_name='transcript_processing',
            context=context,
            language=language,
            context_summary=context_summary
        )
        
        try:
            # Call the OpenAI API
            response = call_llm(
                user_prompt=chunk,
                system_prompt=system_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Return the processed text
            return response
            
        except APIError as e:
            self.logger.error(f"API error processing chunk: {str(e)}")
            # Return original chunk on error, to ensure we don't lose content
            return chunk
        except Exception as e:
            self.logger.error(f"Unexpected error processing chunk: {str(e)}")
            return chunk
    
    # Method is replaced by prompt_manager.generate_summary
    
    def _get_chunk_with_complete_sentences(self, text: str, max_chunk_size: int) -> tuple[str, str]:
        """
        Extract a chunk of text with complete sentences up to max_chunk_size.
        
        Args:
            text: Input text to chunk
            max_chunk_size: Maximum size of the chunk in bytes
            
        Returns:
            Tuple of (chunk, remaining_text)
        """
        sentences = create_sentences(text, max_sentence_length=max_chunk_size-1)
        chunk = ''
        
        for sentence in sentences:
            proposed_chunk = chunk + sentence.strip() + ' '
            
            if len(proposed_chunk.encode('utf-8')) <= max_chunk_size:
                chunk = proposed_chunk
            else:
                # If we already have some content, stop here
                if chunk:
                    break
                # Handle case when an individual sentence exceeds max_chunk_size
                self.logger.warning(f"Sentence exceeds max_chunk_size and will be truncated: {sentence[:50]}...")
                # Take as much as we can (should be handled better by create_sentences)
                chunk = sentence[:max_chunk_size]
                break
        
        # Find the remaining text after the chunk
        # First get the chunk without trailing space for accurate position calculation
        clean_chunk = chunk.strip()
        chunk_position = text.find(clean_chunk) + len(clean_chunk)
        remaining_text = text[chunk_position:] if chunk_position < len(text) else ''
        
        return clean_chunk, remaining_text

def process_transcript(
    input_text: str,
    model: str = "gpt-4o",
    max_chunk_size: int = 3000,
    temperature: float = 0.1,
    context: str = "",
    language: str = "en"
) -> str:
    """
    High-level function to process a transcript text.
    
    Args:
        input_text: Raw transcript text to process
        model: Model to use for processing
        max_chunk_size: Maximum chunk size for processing
        temperature: Temperature for generation
        context: Context information for better processing
        language: Language code
        
    Returns:
        Processed transcript text
        
    Raises:
        APIError: For API-related errors
    """
    processor = TranscriptProcessor(
        model=model,
        max_chunk_size=max_chunk_size,
        temperature=temperature
    )
    
    return processor.process(
        text=input_text,
        context=context,
        language=language
    )

def _generate_text_with_continuation(
    text: str,
    model: str = "gpt-4o", 
    max_tokens: int = 1000,
    temperature: float = 0.1,
    context: str = ""
) -> str:
    """
    Internal function for text generation with continuation (used by tests).
    
    Args:
        text: Input text to continue
        model: Model to use for generation
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation  
        context: Additional context for generation
        
    Returns:
        Generated continuation text
        
    Raises:
        APIError: For API-related errors
    """
    from transcribe_pkg.utils.api_utils import get_openai_client
    
    client = get_openai_client()
    
    # Build prompt
    system_prompt = f"Continue the following text naturally. Context: {context}" if context else "Continue the following text naturally."
    
    response = call_llm(
        user_prompt=text,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response

#fin