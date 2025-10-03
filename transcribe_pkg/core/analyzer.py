#!/usr/bin/env python3
"""
Content analysis and specialized processing.

This module provides advanced analysis and specialized processing of transcripts,
detecting content characteristics and applying appropriate strategies.
"""
import re
import logging
from typing import Any
import json

from transcribe_pkg.utils.logging_utils import get_logger
from transcribe_pkg.utils.api_utils import OpenAIClient, APIError, call_llm
from transcribe_pkg.utils.prompts import PromptManager

class ContentAnalyzer:
    """
    Analyze transcript content for characteristics and specialized processing.
    
    This class provides advanced content analysis capabilities, detecting
    content types, speaker patterns, topic shifts, and other characteristics
    to enable specialized content-aware processing.
    """
    
    def __init__(
        self,
        api_client: OpenAIClient | None = None,
        prompt_manager: PromptManager | None = None,
        model: str = "gpt-4o-mini",
        logger: logging.Logger | None = None
    ):
        """
        Initialize the content analyzer.
        
        Args:
            api_client: OpenAI client for API calls
            prompt_manager: Prompt manager for generating prompts
            model: Model to use for analysis
            logger: Logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.api_client = api_client or OpenAIClient(logger=self.logger)
        self.prompt_manager = prompt_manager or PromptManager(
            api_client=self.api_client,
            logger=self.logger
        )
        self.model = model
    
    def analyze_content(self, text: str) -> dict[str, Any]:
        """
        Analyze content characteristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of content characteristics
        """
        # Prepare sample (use first part of text for efficiency)
        sample_length = min(len(text), 5000)
        sample = text[:sample_length]
        
        # Perform basic analysis
        analysis = {
            "length": len(text),
            "content_type": self._detect_content_type(sample),
            "language": self._detect_language(sample),
            "technical_level": self._measure_technical_level(sample),
            "dialogue_ratio": self._measure_dialogue_ratio(sample),
            "domains": self._extract_domains(sample),
            "structure": self._analyze_structure(sample),
        }
        
        return analysis
    
    def get_specialized_prompt(
        self,
        analysis: dict[str, Any],
        context: str = "",
        purpose: str = "clean"
    ) -> str:
        """
        Get a specialized prompt based on content analysis.
        
        Args:
            analysis: Content analysis results
            context: Additional context information
            purpose: Purpose of the prompt ("clean", "summarize", etc.)
            
        Returns:
            Specialized prompt
        """
        # Use the prompt template appropriate for the content type
        content_type = analysis.get("content_type", "general")
        
        # Set up template mapping for different purposes
        template_mapping = {
            "clean": {
                "dialogue": "dialogue_cleaning",
                "technical": "technical_cleaning",
                "speech": "speech_cleaning",
                "lecture": "lecture_cleaning",
                "general": "transcript_processing"
            },
            "summarize": {
                "dialogue": "dialogue_summary",
                "technical": "technical_summary",
                "speech": "speech_summary",
                "lecture": "lecture_summary",
                "general": "context_summary"
            }
        }
        
        # Get the template name to use
        purpose_templates = template_mapping.get(purpose, {})
        template_name = purpose_templates.get(content_type, purpose_templates.get("general"))
        
        # If the template doesn't exist, fall back to default
        if template_name not in self.prompt_manager._templates:
            if purpose == "clean":
                template_name = "transcript_processing"
            else:
                template_name = "context_summary"
        
        # Incorporate domains into context
        domains = analysis.get("domains", [])
        if domains and not context:
            context = ", ".join(domains)
            self.logger.info(f"Auto-generated context from content: {context}")
        elif domains:
            original_context = context
            context = context + ", " + ", ".join(domains)
            self.logger.info(f"Enhanced context: {original_context} + {', '.join(domains)} = {context}")

        # Get language from analysis
        language = analysis.get("language", "en")

        # Generate the specialized prompt
        return self.prompt_manager.get_system_prompt(
            template_name=template_name,
            context=context,
            language=language
        )
    
    def _detect_content_type(self, text: str) -> str:
        """
        Detect the type of content in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Content type ("dialogue", "technical", "speech", "lecture", "general")
        """
        # Try to use API for detection
        try:
            system_prompt = """Classify the text into exactly ONE category.

Categories:
- dialogue: Conversation between two or more speakers with turns and speaker indicators
- technical: Technical content with specialized terminology, concepts, or procedures
- speech: Monologue or address delivered to an audience
- lecture: Educational content with explanations of concepts and instructional tone
- general: General narrative or explanatory content

Respond with ONLY ONE WORD - the category name in lowercase (dialogue, technical, speech, lecture, or general). No explanation, no punctuation, just the single word."""

            # Use minimal reasoning effort for GPT-5 models for faster classification
            from transcribe_pkg.utils.api_utils import _is_reasoning_model
            reasoning_effort = "minimal" if _is_reasoning_model(self.model) else None

            response = call_llm(
                user_prompt=text[:3000],
                system_prompt=system_prompt,
                model=self.model,
                temperature=0.0,
                max_tokens=100,
                reasoning_effort=reasoning_effort
            )

            # Extract content type from response
            content_type = response.strip().lower()

            # Handle empty response
            if not content_type:
                self.logger.debug("Empty response from API for content type detection, using basic detection")
                # Fall through to basic pattern detection
            else:
                # Validate content type
                valid_types = ["dialogue", "technical", "speech", "lecture", "general"]
                if content_type in valid_types:
                    return content_type

                # Default to basic detection if API result is invalid
                self.logger.debug(f"Invalid content type from API: {content_type}, using basic detection")
        except Exception as e:
            self.logger.warning(f"Error detecting content type with API: {str(e)}")
        
        # Fall back to basic pattern detection
        # Check for dialogue patterns
        dialogue_markers = [":", '"', "'", "said", "asked", "replied"]
        dialogue_count = sum(text.count(marker) for marker in dialogue_markers)
        
        # Check for technical content
        technical_markers = ["Figure", "Table", "algorithm", "equation", "function"]
        technical_count = sum(text.count(marker) for marker in technical_markers)
        
        # Check for speech patterns
        speech_markers = ["thank you", "ladies and gentlemen", "my fellow", "address", "speech"]
        speech_count = sum(text.count(marker) for marker in speech_markers)
        
        # Check for lecture patterns
        lecture_markers = ["as we can see", "in this lecture", "let's examine", "concept", "theory"]
        lecture_count = sum(text.count(marker) for marker in lecture_markers)
        
        # Determine content type based on markers
        if dialogue_count > len(text.split()) / 20:
            return "dialogue"
        elif technical_count > len(text.split()) / 100:
            return "technical"
        elif speech_count > 3:
            return "speech"
        elif lecture_count > 3:
            return "lecture"
        else:
            return "general"
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'fr')
        """
        # Use prompt manager's language detection
        try:
            return self.prompt_manager.detect_language(text[:1000])
        except Exception as e:
            self.logger.warning(f"Error detecting language: {str(e)}")
            return "en"  # Default to English
    
    def _measure_technical_level(self, text: str) -> float:
        """
        Measure the technical level of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Technical level score (0.0 to 1.0)
        """
        # Simple heuristic based on common technical terms
        technical_terms = [
            "algorithm", "function", "system", "process", "analyze",
            "component", "interface", "module", "parameter", "variable",
            "implementation", "architecture", "framework", "methodology", 
            "procedure", "protocol", "specification", "structure", "technique"
        ]
        
        # Count occurrences of technical terms
        term_count = sum(text.lower().count(term) for term in technical_terms)
        
        # Normalize by text length and cap at 1.0
        words = len(text.split())
        score = min(1.0, term_count / (words * 0.05))
        
        return score
    
    def _measure_dialogue_ratio(self, text: str) -> float:
        """
        Measure the ratio of dialogue to narrative in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dialogue ratio (0.0 to 1.0)
        """
        # Look for dialogue indicators
        dialogue_markers = [":", '"', "'", "said", "asked", "replied"]
        
        # Count lines with dialogue markers
        lines = text.split("\n")
        dialogue_lines = sum(
            1 for line in lines 
            if any(marker in line for marker in dialogue_markers)
        )
        
        # Calculate ratio (cap at 1.0)
        ratio = min(1.0, dialogue_lines / max(1, len(lines)))
        
        return ratio
    
    def _extract_domains(self, text: str) -> list[str]:
        """
        Extract domain-specific knowledge areas from the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of domain areas
        """
        # Use prompt manager to extract context
        try:
            context_str = self.prompt_manager.extract_context(text[:3000])
            if not context_str:
                self.logger.debug("Empty context string from extract_context, no domains extracted")
                return []
            domains = [domain.strip() for domain in context_str.split(",") if domain.strip()]
            self.logger.debug(f"Extracted {len(domains)} domains: {domains}")
            return domains
        except Exception as e:
            self.logger.warning(f"Error extracting domains: {str(e)}")
            return []
    
    def _analyze_structure(self, text: str) -> dict[str, Any]:
        """
        Analyze the structural characteristics of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of structural characteristics
        """
        lines = text.split("\n")
        paragraphs = text.split("\n\n")
        
        # Analyze structure
        structure = {
            "avg_paragraph_length": sum(len(p.split()) for p in paragraphs) / max(1, len(paragraphs)),
            "avg_line_length": sum(len(l.split()) for l in lines) / max(1, len(lines)),
            "paragraph_count": len(paragraphs),
            "line_count": len(lines),
        }
        
        return structure

class SpecializedProcessor:
    """
    Provide specialized processing based on content analysis.
    
    This class applies different processing strategies based on content
    characteristics, optimizing the processing for different types of content.
    """
    
    def __init__(
        self,
        api_client: OpenAIClient | None = None,
        analyzer: ContentAnalyzer | None = None,
        logger: logging.Logger | None = None
    ):
        """
        Initialize the specialized processor.
        
        Args:
            api_client: OpenAI client for API calls
            analyzer: Content analyzer instance
            logger: Logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.api_client = api_client or OpenAIClient(logger=self.logger)
        self.analyzer = analyzer or ContentAnalyzer(
            api_client=self.api_client,
            logger=self.logger
        )
    
    def process_content(
        self,
        text: str,
        context: str = "",
        language: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.05
    ) -> str:
        """
        Process content with specialized handling based on content type.
        
        Args:
            text: Text to process
            context: Domain-specific context
            language: Language code
            model: Model to use for processing
            temperature: Temperature for text generation
            
        Returns:
            Processed text
        """
        # Analyze content to determine characteristics
        analysis = self.analyzer.analyze_content(text)
        self.logger.info(f"Content analysis: type={analysis['content_type']}, language={analysis['language']}")
        
        # Override language if explicitly provided
        if language:
            analysis['language'] = language
        
        # Get specialized prompt for this content type
        system_prompt = self.analyzer.get_specialized_prompt(
            analysis,
            context=context,
            purpose="clean"
        )
        
        # Process text with specialized prompt
        try:
            response = call_llm(
                user_prompt=text,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=4096
            )
            
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            return text  # Return original text on error
    
    def summarize_content(
        self,
        text: str,
        context: str = "",
        language: str | None = None,
        model: str = "gpt-4o-mini",
        max_length: int | None = None
    ) -> str:
        """
        Generate a summary of content with specialized handling.
        
        Args:
            text: Text to summarize
            context: Domain-specific context
            language: Language code
            model: Model to use for summarization
            max_length: Maximum summary length in characters
            
        Returns:
            Summary of the content
        """
        # Analyze content to determine characteristics
        analysis = self.analyzer.analyze_content(text)
        
        # Override language if explicitly provided
        if language:
            analysis['language'] = language
        
        # Get specialized prompt for summarization
        system_prompt = self.analyzer.get_specialized_prompt(
            analysis,
            context=context,
            purpose="summarize"
        )
        
        # Add max length instruction if specified
        if max_length:
            system_prompt += f"\n\nYour summary must be no longer than {max_length} characters."
        
        # Generate summary
        try:
            response = call_llm(
                user_prompt=text,
                system_prompt=system_prompt,
                model=model,
                temperature=0.0,
                max_tokens=1000
            )
            
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error summarizing content: {str(e)}")
            # Generate a minimal summary on error
            return text[:100] + "..." if len(text) > 100 else text

#fin