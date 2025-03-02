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
from nltk.tokenize import sent_tokenize

def create_sentences(text, *, max_sentence_length=3000):
  """
  Create logical sentences from text, respecting maximum byte length.
  
  Args:
    text (str): Input text to process
    max_sentence_length (int, optional): Maximum byte length of sentences. Defaults to 3000.
    
  Returns:
    list: List of sentences
  """
  # Download necessary NLTK data
  if not hasattr(create_sentences, "initialized"):
    nltk.download('punkt', quiet=True)
    create_sentences.initialized = True
    
  sentences = []
  sents = sent_tokenize(text.replace('\r\n', '\n').replace('\r', ''))
  
  for sentence in sents:
    # check for overlong sentences
    if len(sentence.encode('utf-8')) <= max_sentence_length:
      sentences.append(sentence.replace('\n', ' ').rstrip() + ' ')
    else:
      newsent = ''
      words = sentence.replace('\n', ' ').split(' ')
      for word in words:
        if len((newsent + word).encode('utf-8')) >= max_sentence_length:
          sentences.append(newsent.rstrip())
          newsent = ''
        newsent += word + ' '
      if newsent:
        sentences.append(newsent.rstrip())
        
  return sentences

def create_paragraphs(text, *, min_sentences=3, max_sentences=8, max_sentence_length=3000):
  """
  Create logical paragraphs from text.
  
  Args:
    text (str): Input text to process
    min_sentences (int, optional): Minimum sentences per paragraph. Defaults to 3.
    max_sentences (int, optional): Maximum sentences per paragraph. Defaults to 8.
    max_sentence_length (int, optional): Maximum sentence length in bytes. Defaults to 3000.
    
  Returns:
    str: Formatted text with paragraphs
  """
  SENTENCE_ENDINGS = {'.', '!', '?'}
  
  sentences = create_sentences(text, max_sentence_length=max_sentence_length)
  
  paragraphs = []
  current_paragraph = []
  
  for sentence in sentences:
    current_paragraph.append(sentence)
    if len(current_paragraph) >= min_sentences and (
        len(current_paragraph) >= max_sentences or
        sentence[-1] in SENTENCE_ENDINGS):
      paragraphs.append(' '.join(current_paragraph))
      current_paragraph = []
      
  # Add any remaining sentences as a paragraph
  if current_paragraph:
    paragraphs.append(' '.join(current_paragraph))
    
  return '\n\n'.join(paragraphs)

def get_chunk_with_complete_sentences(text, max_chunk_size):
  """
  Extract a chunk of text with complete sentences up to max_chunk_size.
  
  This function breaks text into chunks by keeping sentences together, ensuring
  that the chunk size does not exceed the specified maximum byte size. This is
  particularly useful for API calls with context length limitations.
  
  Algorithm:
  1. Split the text into sentences using create_sentences
  2. Add sentences to the chunk until adding another would exceed max_chunk_size
  3. If a single sentence exceeds max_chunk_size, it will be split at word boundaries
  
  Args:
    text (str): Input text to chunk
    max_chunk_size (int): Maximum size of chunk in bytes
    
  Returns:
    tuple: (chunk, remaining_text) where:
      - chunk: A text chunk containing complete sentences up to max_chunk_size
      - remaining_text: The remaining text not included in the chunk
      
  Example:
    text = "First sentence. Second very long sentence. Third sentence."
    chunk, remaining = get_chunk_with_complete_sentences(text, 30)
    # chunk might be "First sentence. "
    # remaining would be "Second very long sentence. Third sentence."
  """
  sentences = create_sentences(text, max_sentence_length=max_chunk_size-1)
  chunk = ''
  
  for sentence in sentences:
    proposed_chunk = chunk + sentence.strip() + ' '
    
    if len(proposed_chunk.encode('utf-8')) <= max_chunk_size:
      chunk = proposed_chunk
    else:
      # Handle case when an individual sentence exceeds max_chunk_size
      if not chunk:
        logging.warning(f"Sentence exceeds max_chunk_size and will be skipped: {sentence}")
      break
  
  # The remaining text should start after the successful chunk
  remaining_text = text[len(chunk):] if len(chunk) < len(text) else ''
  
  return chunk.strip(), remaining_text

#fin