#!/usr/bin/env python3
"""
Transcript processing functionality for the transcribe package.
"""

import logging
from openai import OpenAI

from transcribe_pkg.utils.api_utils import openai_client, call_llm
from transcribe_pkg.utils.language_utils import get_language_name
from transcribe_pkg.utils.text_utils import get_chunk_with_complete_sentences

# Default values
DEF_MODEL = 'gpt-4o'
DEF_SUMMARY_MODEL = 'gpt-4o-mini'
DEF_MAX_CHUNK_SIZE = 3000
DEF_TEMPERATURE = 0.05
DEF_MAX_TOKENS = 4096

def add_and_before_last(s):
  """
  Add 'and' before the last item in a comma-separated list.
  
  Args:
    s (str): Comma-separated string
    
  Returns:
    str: Formatted string with 'and' before last item
  """
  words = s.split(', ')
  if len(words) <= 1:   
    return s
  elif len(words) == 2: 
    return ' and '.join(words)
  return ', '.join(words[:-1]) + ', and ' + words[-1]

def create_context_summary(input_text, model=DEF_SUMMARY_MODEL):
  """
  Create a summary of the context for processing.
  
  Args:
    input_text (str): Input text to summarize
    model (str, optional): Model to use. Defaults to DEF_SUMMARY_MODEL.
    
  Returns:
    str: Summary of input text
  """
  systemprompt="""
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

"""
  return call_llm(systemprompt, input_text, model, 0, 1000)

def _generate_text_with_continuation(input_text, model, max_tokens, temperature, context, lang, context_summary):
  """
  Generate corrected and formatted text using OpenAI's API.
  
  Args:
    input_text (str): Input text to process
    model (str): Model to use
    max_tokens (int): Maximum tokens to generate
    temperature (float): Temperature for generation
    context (str): Domain context
    lang (str): Language code
    context_summary (str): Context summary for continuation
    
  Returns:
    object: OpenAI API response
  """
  if context:
    context = f", with extensive knowledge in {add_and_before_last(context)}"

  language = ''
  language_task = ''

  if lang is None:
    lang = 'en'
    language = ''
  elif lang != 'en':
    language = get_language_name(lang)
    if 'Unknown' in language:
      lang = 'en'
      language = ''
    else:
      context += f", and you are an expert {language}-English translator."
      language_task = f', and accurately translate/interpret the text from {language} into English'

  logging.info(f"Context: {context}")

  if context_summary:
    context_summary = f"\n\n## Context Summary:\n\n{context_summary}\n\n"
  else:
    context_summary = ''

  systemprompt = f"""
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

"""

  messages = [
    {"role": "system", "content": systemprompt},
    {"role": "user", "content": input_text}
  ]
  
  try:
    response = openai_client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=1,
      n=1,
      stop='',
      frequency_penalty=0,
      presence_penalty=0
    )
    
    if not response.choices or not response.choices[0].message.content.strip():
      logging.warning(f"Empty response from API: input_text='{input_text[:128]}...'")
      logging.debug(f"Response: {response}")
      response.choices[0].message.content = ''
      
    logging.debug(f"Response preview: {response.choices[0].message.content[:80]}")
    return response
    
  except Exception as e:
    logging.critical(f"Error in API call: {str(e)}")
    raise

def process_transcript(input_text, *, model=DEF_MODEL, max_chunk_size=DEF_MAX_CHUNK_SIZE, 
                     temperature=DEF_TEMPERATURE, context='', language='en', 
                     max_tokens=DEF_MAX_TOKENS):
  """
  Process transcript text in chunks, preserving context between chunks.
  
  Args:
    input_text (str): Input transcript text
    model (str, optional): OpenAI model to use. Defaults to DEF_MODEL.
    max_chunk_size (int, optional): Maximum chunk size. Defaults to DEF_MAX_CHUNK_SIZE.
    temperature (float, optional): Temperature for generation. Defaults to DEF_TEMPERATURE.
    context (str, optional): Domain context. Defaults to ''.
    language (str, optional): Language code. Defaults to 'en'.
    max_tokens (int, optional): Maximum tokens. Defaults to DEF_MAX_TOKENS.
    
  Returns:
    str: Processed transcript
  """
  input_text = input_text.rstrip()
  total_length = len(input_text)
  processed_length = 0
  generated_text = ""
  remaining_text = input_text
  iterations = 0
  iteration_limit = int((total_length / max_chunk_size) * 2)
  context_summary = None

  max_iterations_warning = False
  
  try:
    while remaining_text:
      # Store the initial length of the remaining text
      initial_length = len(remaining_text)
      
      # Get the next chunk with complete sentences
      chunk, remaining_text = get_chunk_with_complete_sentences(remaining_text, max_chunk_size)
      
      # Safety check: if no progress is being made
      if initial_length == len(remaining_text):
        logging.warning("No progress made in processing the text. Breaking loop.")
        break
      
      # Generate the text
      try:
        response = _generate_text_with_continuation(chunk, model, max_tokens, temperature, 
                                                context, language, context_summary)
                                                
        # Add to generated text with proper spacing
        if generated_text and generated_text[-1] in '.,?!`"':
          generated_text += ' '
        generated_text += response.choices[0].message.content
  
        # Update context summary for next chunk
        paragraphs = response.choices[0].message.content.strip().split('\n\n')
        context_summary = create_context_summary('\n\n'.join(paragraphs[-7:]))
      except Exception as e:
        logging.error(f"Error during text generation: {str(e)}")
        logging.info("Continuing with partial results...")
        # If we have partial generated text, we'll return that rather than failing completely
        if generated_text:
          break
        else:
          # Otherwise, re-raise the exception
          raise
  
      # Update progress tracking
      processed_length += initial_length - len(remaining_text)
      percent = processed_length/total_length
      logging.info(f"Progress: {percent:.1%} Iteration: {iterations}/{iteration_limit}")
      
      # Break the loop if complete
      if processed_length >= total_length or remaining_text == '' or percent >= 99.85:
        break
        
      iterations += 1
      if iterations > iteration_limit:
        if not max_iterations_warning:
          logging.warning(f'Maximum iterations ({iteration_limit}) reached. Processing may be incomplete.')
          max_iterations_warning = True
        # Instead of breaking, try to process one more chunk with double the chunk size
        if max_chunk_size * 2 <= DEF_MAX_CHUNK_SIZE * 4:  # Don't let it grow too large
          max_chunk_size *= 2
          logging.info(f"Increasing chunk size to {max_chunk_size} to attempt to complete processing")
        else:
          logging.error("Failed to complete processing after multiple attempts with increased chunk sizes")
          break
  except Exception as e:
    logging.error(f"Error in transcript processing: {str(e)}")
    # Return what we have so far rather than nothing
    if not generated_text:
      raise
      
  return generated_text

#fin