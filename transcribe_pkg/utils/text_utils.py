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
from typing import List, Optional, Dict, Any, Tuple
from nltk.tokenize import sent_tokenize

from transcribe_pkg.utils.logging_utils import get_logger

# This function will ensure NLTK data is downloaded only once
def _ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

def create_sentences(text: str, *, max_sentence_length: int = 3000) -> List[str]:
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
            newsent = ''
            words = sentence.replace('\n', ' ').split(' ')
            
            for word in words:
                # Check if adding this word would exceed the limit
                if len((newsent + word).encode('utf-8')) >= max_sentence_length:
                    sentences.append(newsent.rstrip())
                    newsent = ''
                    
                # Add word to the current sentence fragment
                newsent += word + ' '
                
            # Don't forget any remaining text
            if newsent.strip():
                sentences.append(newsent.rstrip())
    
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
) -> List[str]:
    """
    Split text into chunks for efficient processing, preserving sentence boundaries.
    
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
    
    chunks = []
    remaining_text = text
    
    while remaining_text:
        # Extract sentences from the current position
        sentences = create_sentences(remaining_text, max_sentence_length=max_chunk_size//2)
        
        if not sentences:
            # Handle case where no complete sentences could be created
            chunks.append(remaining_text[:max_chunk_size])
            remaining_text = remaining_text[max_chunk_size:]
            continue
            
        # Build a chunk up to max_chunk_size
        current_chunk = ""
        used_sentences = 0
        
        for sentence in sentences:
            if len((current_chunk + sentence).encode('utf-8')) <= max_chunk_size:
                current_chunk += sentence
                used_sentences += 1
            else:
                break
                
        if not current_chunk:
            # If we couldn't fit any sentence, take a raw chunk
            logger = get_logger(__name__)
            logger.warning(f"Could not fit any sentence within chunk size limit. Using raw chunk.")
            chunks.append(remaining_text[:max_chunk_size])
            remaining_text = remaining_text[max_chunk_size:]
        else:
            # Add the chunk and remove processed sentences from remaining text
            chunks.append(current_chunk)
            
            # Calculate how much text we've processed
            processed_text_length = 0
            for i in range(used_sentences):
                processed_text_length += len(sentences[i])
                
            # Move past the processed text, considering overlap
            if processed_text_length > overlap:
                move_to = processed_text_length - overlap
                remaining_text = remaining_text[move_to:]
            elif len(remaining_text) > processed_text_length:
                # If we can't create a proper overlap, just advance a bit
                remaining_text = remaining_text[processed_text_length//2:]
            else:
                # We're done
                remaining_text = ""
    
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

def extract_key_topics(text: str, max_topics: int = 5) -> List[str]:
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
    
    # Count word frequencies
    word_counts = {}
    for word in words:
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    
    # Filter out common stopwords (simplified approach)
    stopwords = {'the', 'and', 'that', 'this', 'with', 'for', 'from', 'have', 'you', 
                'are', 'was', 'were', 'they', 'will', 'would', 'could', 'should', 
                'what', 'when', 'where', 'which', 'there', 'their', 'then', 'than'}
    
    for stopword in stopwords:
        if stopword in word_counts:
            del word_counts[stopword]
    
    # Sort by frequency and return top topics
    topics = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [topic[0] for topic in topics[:max_topics]]

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