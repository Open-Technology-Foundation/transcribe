#!/usr/bin/env python3
"""
Prompt management for AI models.

This module provides a centralized way to manage prompts used with AI models
throughout the transcription toolkit. It includes:

- Standard prompts for text processing
- Context extraction
- Language detection
- Text summary generation
- Template management for customizing prompts
"""
import logging
from typing import Any
import re

from transcribe_pkg.utils.api_utils import OpenAIClient, APIError, call_llm
from transcribe_pkg.utils.logging_utils import get_logger

class PromptManager:
    """
    Manage and generate prompts for AI interactions.
    
    This class centralizes prompt management, providing:
    - Standard system prompts for various tasks
    - Customization options
    - Context-specific prompt generation
    """
    
    def __init__(self, api_client: OpenAIClient | None = None, logger: logging.Logger | None = None):
        """
        Initialize the prompt manager.
        
        Args:
            api_client: OpenAI client instance (creates new if None)
            logger: Logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.api_client = api_client or OpenAIClient(logger=self.logger)
        
        # Initialize template storage
        self._templates = {
            # Transcript processing - General
            'transcript_processing': """
# Translation/Transcription Correction and Formatting Editor

You are an expert translation/transcription editor{context}.

Your task is to review and correct text -- which could be transcriptions, or other text -- focusing on domain-specific terms and concepts{language_task}. Follow these guidelines:

1. **Grammar and Clarity**:
  - Fix poor grammar **only where necessary** for clarity.
  - Remove hesitation words (e.g., "um", "uh") and repetitions.

2. **Relevance**:
  - Some text/transcripts may contain sentences or paragraphs requesting that viewers subscribe to their channel, or join their Patreon. These sentences and paragraphs must be removed.
  - Some text/transcripts may contain sentences or paragraphs that are promotions for products or services unrelated to the topic of the text. These sentences and paragraphs must be removed.

3. **Transcription Formatting**:
  - Do **not** use third-person references (e.g., "The speaker said...").
  - Create logical sentences that are properly capitalized and punctuated.
  - Create logical paragraphs from the sentences that are neither too long nor too short.

4. **Content Integrity**:
  - Do **not** change the meaning of the text.
  - Do **not** add any new information or preambles. Just the corrected text.
  - Reformat the text as needed without altering the original content.

5. **Examples**:
  - Correct: "The brain's prefrontal cortex is responsible for decision-making."
  - Incorrect: "The speaker said that the brain's prefrontal cortex is responsible for decision-making."
{context_summary}
## Input/Output

Your only goal is to reformat the text for readability and clarity while preserving the original meaning and context. Output only the corrected and formatted text. Do not include any additional preamble, commentary or explanations.

Input: Raw transcript text

Output: Corrected and formatted translation/transcript, in English
""",

            # Transcript processing - Dialogue
            'dialogue_cleaning': """
# Dialogue Transcript Editor

You are an expert dialogue editor and transcript specialist{context}.

Your task is to clean up and format dialogue-based transcripts while preserving the natural flow of conversation{language_task}. Follow these guidelines:

1. **Conversation Flow**:
  - Preserve speaker turns and the natural back-and-forth of conversation.
  - Keep the informal tone when appropriate, but remove excessive filler words.
  - Maintain the personality and speaking style of each participant.

2. **Speaker Attribution**:
  - Maintain any existing speaker labels if present (e.g., "Speaker 1:", "John:").
  - Do NOT add speaker labels if none exist in the original.
  - If speakers can be identified by context, ensure consistency in attribution.

3. **Readability Improvements**:
  - Fix grammar only where necessary for clarity, but preserve conversational speech patterns.
  - Remove excessive hesitations, false starts, and repetitions (like "um", "uh", "you know").
  - Format dialogue exchanges with proper line breaks between speakers.

4. **Content Integrity**:
  - Remove irrelevant promotional content (e.g., "subscribe to our channel").
  - Do NOT change the meaning of what was said.
  - Do NOT add any new information or commentary.

5. **Examples**:
  - Original: "Yeah, um, I think that, you know, the whole thing with quantum physics is, like, it's really weird, right?"
  - Improved: "Yeah, I think the whole thing with quantum physics is really weird, right?"
{context_summary}
## Input/Output

Your only goal is to improve readability while preserving the natural conversational quality. Output only the cleaned dialogue. Do not include any additional commentary or explanations.

Input: Raw dialogue transcript

Output: Cleaned and formatted dialogue
""",

            # Transcript processing - Technical
            'technical_cleaning': """
# Technical Content Editor

You are an expert editor specializing in technical and scientific content{context}.

Your task is to review and correct technical transcripts, ensuring accuracy of terminology and clarity of complex concepts{language_task}. Follow these guidelines:

1. **Technical Accuracy**:
  - Preserve all technical terms, ensuring they are correctly spelled and used.
  - Maintain the precision of scientific and technical descriptions.
  - Correct obvious technical errors only if they are clearly mistakes in transcription, not errors in the original speech.

2. **Structural Clarity**:
  - Format procedural steps and processes clearly.
  - Preserve the logical flow of technical explanations.
  - Use consistent formatting for equations, variables, and specialized notation.

3. **Clarity Improvements**:
  - Fix grammar issues that obscure technical meaning.
  - Remove verbal hesitations and filler words.
  - Break down overly long technical explanations into manageable paragraphs.

4. **Content Integrity**:
  - Do NOT simplify complex concepts or technical terminology.
  - Do NOT add explanations beyond what was in the original text.
  - Preserve all substantive technical content, even if complex.

5. **Examples**:
  - Original: "So um when we talk about the uh algorithm complexity we mean like big O of n squared right?"
  - Improved: "When we talk about the algorithm complexity, we mean Big O of n-squared, right?"
{context_summary}
## Input/Output

Your goal is to enhance the readability of technical content while preserving all technical information and terminology. Output only the corrected technical content without commentary.

Input: Raw technical transcript

Output: Corrected and formatted technical content
""",

            # Summary generation
            'context_summary': """
# Summary Editor

You are a Summary Editor, expert in editing texts into very brief, concise summaries.

Your role is to create a summary of the main points of the text, focussing only on the most salient information, in no more than three paragraphs.

Follow these guidelines:

  - **Do not** use third-person references (e.g., "The speaker said...", etc).
  - **Do not** add any new information or preambles. Just the summarized text.
  - **Only** Output the summary paragraphs; *no* preambles or commentary.
  - NEVER include **any** additional preamble to the summary paragraphs, such as 'Here is the detailed summary ...', etc.
  - Your only task is to create summary paragraphs, and to output those paragraphs.

Examples:

  - Incorrect: "Here is the brief summary of the main points from the text"
  - Incorrect: "The speaker said that the brain's prefrontal cortex is responsible for decision-making."
""",

            # Context extraction
            'context_extraction': """Identify the 3-5 most relevant academic or professional fields for this text.

Output ONLY a comma-separated list of fields, ordered by relevance. Use lowercase, no articles.

Examples:
- neuroscience,psychology,biology
- economics,sociology,political science
- physics,astronomy,mathematics
- literature,cultural studies,history

Respond with ONLY the comma-separated list. No explanation, no preamble, no punctuation except commas.""",

            # Language detection
            'language_detection': """Determine the language of the text.

Respond with ONLY a two-character ISO 639-1 language code in lowercase.

Examples: en, es, fr, de, zh, ja, ko, ar, ru

Output ONLY the two-letter code. No explanation, no preamble, no punctuation."""
        }
    
    def get_template(self, template_name: str) -> str:
        """
        Get a prompt template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template string
            
        Raises:
            ValueError: If template name is not found
        """
        if template_name not in self._templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self._templates[template_name]
    
    def set_template(self, template_name: str, template: str) -> None:
        """
        Set or update a prompt template.
        
        Args:
            template_name: Name of the template
            template: Template string
        """
        self._templates[template_name] = template
    
    def get_system_prompt(
        self,
        template_name: str,
        context: str = "",
        language: str = "en",
        context_summary: str | None = None
    ) -> str:
        """
        Get a formatted system prompt for a specific task.
        
        Args:
            template_name: Name of the template
            context: Domain-specific context
            language: Language code
            context_summary: Optional context summary
            
        Returns:
            Formatted system prompt
        """
        # Get the template
        template = self.get_template(template_name)
        
        # Format context
        context_str = ""
        if context:
            context_str = f", with extensive knowledge in {self._add_oxford_comma(context)}"
        
        # Set up language-specific handling
        language_task = ""
        if language != "en":
            language_task = f", and accurately translate/interpret the text from {language.upper()} into English"
        
        # Prepare context summary
        context_summary_text = ""
        if context_summary:
            context_summary_text = f"\n\n## Context Summary:\n\n{context_summary}\n\n"
        
        # Format the prompt
        return template.format(
            context=context_str,
            language_task=language_task,
            context_summary=context_summary_text
        )
    
    def extract_context(self, text: str, model: str = "gpt-4o-mini") -> str:
        """
        Extract domain context from text.
        
        Args:
            text: Text to analyze
            model: Model to use
            
        Returns:
            Comma-separated context string
        """
        system_prompt = self.get_template('context_extraction')
        try:
            # Use minimal reasoning effort for GPT-5 models for faster classification
            from transcribe_pkg.utils.api_utils import _is_reasoning_model
            reasoning_effort = "minimal" if _is_reasoning_model(model) else None

            response = call_llm(
                user_prompt=text,
                system_prompt=system_prompt,
                model=model,
                temperature=0.0,
                max_tokens=200,
                reasoning_effort=reasoning_effort
            )
            result = response.strip()
            if not result:
                self.logger.debug("Empty response from context extraction")
            else:
                self.logger.debug(f"Context extraction result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error extracting context: {str(e)}")
            return ""
    
    def detect_language(self, text: str, model: str = "gpt-4o-mini") -> str:
        """
        Detect the language of text.
        
        Args:
            text: Text to analyze
            model: Model to use
            
        Returns:
            ISO language code
        """
        system_prompt = self.get_template('language_detection')
        try:
            # Use minimal reasoning effort for GPT-5 models for faster classification
            from transcribe_pkg.utils.api_utils import _is_reasoning_model
            reasoning_effort = "minimal" if _is_reasoning_model(model) else None

            response = call_llm(
                user_prompt=text,
                system_prompt=system_prompt,
                model=model,
                temperature=0.0,
                max_tokens=50,
                reasoning_effort=reasoning_effort
            )
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error detecting language: {str(e)}")
            return "en"  # Default to English on error
    
    def generate_summary(self, text: str, model: str = "gpt-4o-mini") -> str:
        """
        Generate a summary of text.
        
        Args:
            text: Text to summarize
            model: Model to use
            
        Returns:
            Text summary
        """
        system_prompt = self.get_template('context_summary')
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
            self.logger.error(f"Error generating summary: {str(e)}")
            return ""
    
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

#fin