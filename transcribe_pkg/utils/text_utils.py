#!/usr/bin/env python3
"""
Text processing utilities for the transcribe package.

This module provides functions for processing and manipulating text data in the 
transcription pipeline. It handles common text processing tasks including:

- Sentence tokenization and length management
- Paragraph creation with logical chunking
- Text chunking for efficient processing while preserving sentence boundaries
- Byte length awareness for API limitations

The module builds on NLTK's tokenization capabilities while adding additional
functionality specific to transcript processing needs, with special attention
to API limitations and readability concerns.
"""

import logging
import nltk
import re
from typing import Any
from collections import Counter
from nltk.tokenize import sent_tokenize

from transcribe_pkg.utils.logging_utils import get_logger

# This function will ensure NLTK data is downloaded only once
def _ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

def create_sentences(text: str, *, max_sentence_length: int = 3000) -> list[str]:
    """
    Create logical sentences from text, respecting maximum byte length.
    
    Args:
        text (str): Input text to process
        max_sentence_length (int, optional): Maximum byte length of sentences. Defaults to 3000.
        
    Returns:
        list: List of sentences
    """
    # Ensure NLTK data is available
    _ensure_nltk_data()
        
    sentences = []
    
    # Normalize line endings and tokenize text
    normalized_text = text.replace('\r\n', '\n').replace('\r', '')
    sents = sent_tokenize(normalized_text)
    
    for sentence in sents:
        # Check if sentence is within length limit
        if len(sentence.encode('utf-8')) <= max_sentence_length:
            # Replace newlines with spaces and ensure proper spacing
            sentences.append(sentence.replace('\n', ' ').rstrip())
        else:
            # Handle oversized sentences by splitting into words
            current_words = []
            words = sentence.replace('\n', ' ').split(' ')

            for word in words:
                # Check if adding this word would exceed the limit
                test_sent = ' '.join(current_words + [word])
                if len(test_sent.encode('utf-8')) >= max_sentence_length:
                    if current_words:  # Only append if we have accumulated words
                        sentences.append(' '.join(current_words))
                        current_words = []

                # Add word to the current fragment
                current_words.append(word)

            # Don't forget any remaining text
            if current_words:
                sentences.append(' '.join(current_words))
    
    return sentences

def create_paragraphs(
    text: str, 
    *, 
    min_sentences: int = 2, 
    max_sentences: int = 8, 
    max_sentence_length: int = 3000
) -> str:
    """
    Create logical paragraphs from text by grouping sentences.
    
    Args:
        text (str): Input text to organize into paragraphs
        min_sentences (int, optional): Minimum sentences per paragraph. Defaults to 2.
        max_sentences (int, optional): Maximum sentences per paragraph. Defaults to 8.
        max_sentence_length (int, optional): Maximum sentence length in bytes. Defaults to 3000.
        
    Returns:
        str: Text organized into paragraphs
    """
    # Create sentences from the text
    sentences = create_sentences(text, max_sentence_length=max_sentence_length)
    
    # If no sentences, return empty string
    if not sentences:
        return ""
    
    paragraphs = []
    current_paragraph = []
    
    for i, sentence in enumerate(sentences):
        current_paragraph.append(sentence)
        
        # Check if we have enough sentences to form a paragraph
        if len(current_paragraph) >= max_sentences:
            paragraphs.append(" ".join(current_paragraph))
            current_paragraph = []
    
    # Add any remaining sentences as a paragraph
    if current_paragraph:
        # If there's only one paragraph and it's short, add it to the last paragraph
        if paragraphs and len(current_paragraph) < min_sentences:
            last_paragraph = paragraphs.pop()
            paragraphs.append(last_paragraph + " " + " ".join(current_paragraph))
        else:
            paragraphs.append(" ".join(current_paragraph))
    
    # Join paragraphs with double newlines
    return '\n\n'.join(paragraphs)

def split_text_for_processing(
    text: str,
    max_chunk_size: int = 3000,
    overlap: int = 200
) -> list[str]:
    """
    Split text into chunks for efficient processing, preserving sentence boundaries.
    
    Optimized O(n) algorithm that processes text incrementally without redundant calculations.
    
    Args:
        text (str): Text to split
        max_chunk_size (int, optional): Maximum chunk size in bytes. Defaults to 3000.
        overlap (int, optional): Overlap between chunks in bytes. Defaults to 200.
        
    Returns:
        List[str]: List of text chunks
    """
    if not text:
        return []
        
    # Ensure NLTK data is available
    _ensure_nltk_data()
    
    # Pre-compute all sentences once at the beginning - O(n) operation
    all_sentences = create_sentences(text, max_sentence_length=max_chunk_size//2)
    
    if not all_sentences:
        # No sentences could be created, split by raw chunks
        chunks = []
        for i in range(0, len(text), max_chunk_size):
            chunks.append(text[i:i + max_chunk_size])
        return chunks
    
    # Pre-compute sentence lengths to avoid recalculation - O(n) operation
    sentence_lengths = [len(sentence) for sentence in all_sentences]
    sentence_byte_lengths = [len(sentence.encode('utf-8')) for sentence in all_sentences]
    
    chunks = []
    sentence_index = 0
    total_sentences = len(all_sentences)
    
    while sentence_index < total_sentences:
        current_sentences = []
        current_byte_size = 0
        chunk_start_index = sentence_index

        # Build chunk by adding sentences until we hit the size limit
        while sentence_index < total_sentences:
            sentence = all_sentences[sentence_index]
            sentence_byte_size = sentence_byte_lengths[sentence_index]

            # Check if adding this sentence would exceed the limit
            if current_byte_size + sentence_byte_size <= max_chunk_size:
                current_sentences.append(sentence)
                current_byte_size += sentence_byte_size
                sentence_index += 1
            else:
                break

        # If we couldn't fit any sentence, force-add the first one to avoid infinite loop
        if not current_sentences and sentence_index < total_sentences:
            logger = get_logger(__name__)
            logger.warning(f"Sentence exceeds max_chunk_size, force-adding to avoid infinite loop")
            current_sentences.append(all_sentences[sentence_index])
            sentence_index += 1

        if current_sentences:
            chunks.append(''.join(current_sentences))
        
        # Handle overlap by backing up some sentences
        if overlap > 0 and len(chunks) > 1:
            # Calculate how many sentences to back up for overlap
            overlap_bytes = 0
            overlap_sentences = 0
            
            # Work backwards from current position to find overlap
            for i in range(sentence_index - 1, chunk_start_index - 1, -1):
                if overlap_bytes + sentence_byte_lengths[i] <= overlap:
                    overlap_bytes += sentence_byte_lengths[i]
                    overlap_sentences += 1
                else:
                    break
            
            # Back up by the overlap amount, but ensure we always make forward progress
            # At minimum, advance by 1 sentence to prevent infinite loops
            max_backup = sentence_index - chunk_start_index - 1
            actual_backup = min(overlap_sentences, max_backup)
            sentence_index -= actual_backup
    
    return chunks

def clean_transcript_text(text: str) -> str:
    """
    Perform basic cleaning operations on transcript text.
    
    Args:
        text (str): Raw transcript text
        
    Returns:
        str: Cleaned transcript text
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common hesitation markers
    text = re.sub(r'\b(um|uh|er|ah|like,|you know,)\b', '', text, flags=re.IGNORECASE)
    
    # Remove repeated words (e.g., "I I I think")
    text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)
    
    # Remove common filler phrases
    text = re.sub(r'\b(sort of|kind of|I mean,|basically,)\b', '', text, flags=re.IGNORECASE)
    
    # Fix spacing after punctuation
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_key_topics(text: str, max_topics: int = 5) -> list[str]:
    """
    Extract key topics from text based on frequency and relevance.
    
    Note: This is a simplified implementation. In a production version,
    this would use more sophisticated NLP techniques.
    
    Args:
        text (str): Text to analyze
        max_topics (int, optional): Maximum number of topics to extract. Defaults to 5.
        
    Returns:
        List[str]: Key topics
    """
    # Ensure NLTK data is available
    _ensure_nltk_data()
    
    # Convert to lowercase and tokenize
    words = re.findall(r'\b\w{3,}\b', text.lower())

    # Count word frequencies using Counter (efficient implementation)
    word_counts = Counter(words)

    # Filter out common stopwords (simplified approach)
    stopwords = {'the', 'and', 'that', 'this', 'with', 'for', 'from', 'have', 'you',
                'are', 'was', 'were', 'they', 'will', 'would', 'could', 'should',
                'what', 'when', 'where', 'which', 'there', 'their', 'then', 'than'}

    # Remove stopwords from counter
    for stopword in stopwords:
        word_counts.pop(stopword, None)

    # Use Counter's most_common method for efficient sorting
    return [word for word, count in word_counts.most_common(max_topics)]

def detect_language(text: str) -> str:
    """
    Detect the language of the text.
    
    This is a placeholder for a more sophisticated language detection implementation.
    In a production version, this would use a library like langid or langdetect.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: ISO 639-1 language code (e.g., 'en', 'fr')
    """
    # In a real implementation, we would use a proper language detection library
    # For now, we'll default to English
    return "en"

#fin"""