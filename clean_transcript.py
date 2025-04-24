#!/usr/bin/env python3
"""
Translator/Transcript Cleaner and Formatter

This script processes raw transcripts using OpenAI's API to correct grammar,
remove irrelevant content, and improve formatting. It can handle large transcripts
by splitting them into chunks and supports various languages.

Usage:
  clean-transcript raw_transcript.txt -c "context" -m model -o clean_transcript.txt
"""

import argparse
import sys
import os
from openai import OpenAI
from language_codes import get_language_name
import re
from tenacity import retry, stop_after_attempt, wait_exponential

DEF_MODEL = 'gpt-4o'
DEF_SUMMARY_MODEL = 'gpt-4o-mini'
DEF_MAX_CHUNK_SIZE = 3000
DEF_TEMPERATURE = 0.05
DEF_MAX_TOKENS = 4096

import logging
import colorlog
def setup_logging(verbose, debug=False):
  """Set up logging configuration with color."""
  logger = logging.getLogger()
  handler = colorlog.StreamHandler()
  if verbose or debug:
    if debug:
      logger.setLevel(logging.DEBUG)
    else:
      logger.setLevel(logging.INFO)
    datefmt="%H:%M:%S"
    logformat=f"%(log_color)s%(asctime)s:%(module)s:%(levelname)s: %(message)s"
  else:
    logger.setLevel(logging.ERROR)
    datefmt=None
    logformat=f"%(log_color)s%(module)s:%(levelname)s: %(message)s"
  formatter = colorlog.ColoredFormatter(
    logformat,
    datefmt=datefmt,
    reset=True,
    log_colors={ 'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white' },
    secondary_log_colors={},
    style='%'
  )
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  return logger

# Initialize OpenAI client with API key from environment variables
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
if not os.getenv('OPENAI_API_KEY'):
  logging.error("OPENAI_API_KEY environment variable not set")
  sys.exit(1)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_LLM(systemprompt, input_text, model='gpt-4o', temperature=0, max_tokens=1000):
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
      n=1,
      stop=''
    )
    if not response.choices or not response.choices[0].message.content.strip():
      logging.warning(f"Empty response from API: input_text='{input_text[:128]}...'")
      response.choices[0].message.content = ''
    return response.choices[0].message.content.strip()
  except Exception as e:
    logging.error(f"API error: {str(e)}")
    sys.exit(1)

# CONTEXT --------------------------------------------------------------------------
def create_context_summary(input_text, model=DEF_SUMMARY_MODEL):
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
  return call_LLM(systemprompt, input_text, model, 0, 1000)


def add_and_before_last(s):
  words = s.split(', ')
  if len(words) <= 1:   return s
  elif len(words) == 2: return ' and '.join(words)
  return ', '.join(words[:-1]) + ', and ' + words[-1]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _generate_text_with_continuation(input_text, model, max_tokens, temperature, context, lang, context_summary):
  """
  Generate corrected and formatted text using OpenAI's API with retry mechanism.

  Args:
    input_text (str): The input text to process
    model (str): The OpenAI model to use
    max_tokens (int): Maximum number of tokens for the API response
    temperature (float): Temperature for text generation
    context (str): Domain-specific context for the transcript
    lang (str): Language code of the input text

  Returns:
    OpenAI API response object
  """
  if context:
    context = f", with extensive knowledge in {add_and_before_last(context)}"

  Language = ''
  Language_Task = ''

  if lang is None:
    lang='en'
    Language = ''
  elif lang != 'en':
    Language = get_language_name(lang)
    if 'Unknown' in Language:
      lang = 'en'
      Language = ''
    else:
      context += f", and you are an expert {Language}-English translator."
      Language_Task = f', and accurately translate/interpret the text from {Language} into English'

  logging.info(f"{context=}")

  if context_summary:
    context_summary = f"\n\n## Context Summary:\n\n{context_summary}\n\n"
  else:
    context_summary = ''

  systemprompt = f"""
# Translation/Transcription Correction and Formatting Editor

You are an expert translation/transcription editor{context}.

Your task is to review and correct text -- which could be transcriptions, or other text -- focusing on domain-specific terms and concepts{Language_Task}. Follow these guidelines:

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
        presence_penalty=0)
    if not response.choices or not response.choices[0].message.content.strip():
      logging.warning(f"Empty response from API: input_text='{input_text[:128]}...'")
      logging.debug(f'{response=}')
      response.choices[0].message.content = ''
    logging.debug(f'{response.choices[0].message.content[:80]=}')
    return response
  except Exception as e:
    logging.debug(f'{response=}')
    logging.critical(f"Error in API call: {str(e)}")
    sys.exit(1)


from create_sentences import create_sentences
def _get_chunk_with_complete_sentences(text, max_chunk_size):
  """
  Extract a chunk of text with complete sentences up to max_chunk_size.

  Args:
      text (str): Input text to chunk.
      max_chunk_size (int): Maximum size of the chunk.

  Returns:
      tuple: (chunk, remaining_text)
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
  remaining_text_index = len(chunk.encode('utf-8')) # Ensures UTF-8 byte consideration
  remaining_text = text[len(chunk):] if remaining_text_index < len(text) else ''

  return chunk.strip(), remaining_text


def process_transcript(input_text, *, model=DEF_MODEL, max_chunk_size=DEF_MAX_CHUNK_SIZE, temperature=DEF_TEMPERATURE, context='', language='en', max_tokens=DEF_MAX_TOKENS):
  """
  Process the entire transcript by splitting it into chunks and generating corrected text.

  Args:
    input_text (str): The full input transcript
    model (str): OpenAI model to use
    max_chunk_size (int): Maximum size of each chunk
    temperature (float): Temperature for text generation
    context (str): Domain-specific context
    language (str): Language code of the input text
    max_tokens (int): Maximum number of tokens for API response

  Returns:
    str: Processed and corrected transcript
  """
  input_text = input_text.rstrip()
  total_length = len(input_text)
  processed_length = 0
  generated_text = ""
  remaining_text = input_text
  iterations = 0
  iteration_limit = int((total_length / max_chunk_size) * 2)
  context_summary = None

  while remaining_text:
    # Store the initial length of the remaining text
    initial_length = len(remaining_text)

    # Get the next chunk with complete sentences
    chunk, remaining_text = _get_chunk_with_complete_sentences(remaining_text, max_chunk_size)

    # Generate the text
    response = _generate_text_with_continuation(chunk, model, max_tokens, temperature, context, language, context_summary)
    if generated_text and generated_text[-1] in '.,?!`"':
      generated_text += ' '
    generated_text += response.choices[0].message.content

    paragraphs = response.choices[0].message.content.strip().split('\n\n')
    context_summary = create_context_summary('\n\n'.join(paragraphs[-7:]))

    # Update the processed length by the difference in length before and after processing the chunk
    processed_length += initial_length - len(remaining_text)
    percent = processed_length/total_length
    logging.info(f"Progress: {percent:.1%} Iteration: {iterations}/{iteration_limit}")
    # Break the loop if all text has been processed
    if processed_length >= total_length or remaining_text == '' or percent >= 99.85:
      break
    iterations+=1
    if iterations > iteration_limit:
      logging.error(f'Too many iterations!')
  return generated_text


def main():
  """
  Parse command-line arguments and orchestrate the transcript cleaning process.
  """
  # Set up the signal handler
  import signal
  def signal_handler(sig, frame):
    print('\033[0m^C\n')
    sys.exit(130)
  signal.signal(signal.SIGINT, signal_handler)

  parser = argparse.ArgumentParser(
    description="Fix and clean up transcripts using OpenAI API.",
    epilog="Example: clean-transcript raw_transcript.txt -c \"neuroscience, free will\" -m gpt-4o -o clean_transcript.txt"
  )
  parser.add_argument("input_file",
      help="Path to the raw text/transcript file")
  parser.add_argument('-L', '--input-language', default=None,
      help='Define the language of the text. If this is specified, then the text is translated into English (def: None))')
  parser.add_argument("-c", "--context", default=None,
      help="Domain-specific context for the transcript (default: none)")
  parser.add_argument("-m", "--model", default=DEF_MODEL,
      help=f"OpenAI model to use (default: {DEF_MODEL})")
  parser.add_argument("-M", "--max-tokens", type=int, default=DEF_MAX_TOKENS,
      help=f"Maximum tokens (default: {DEF_MAX_TOKENS})")
  parser.add_argument("-s", "--max-chunk-size", type=int, default=DEF_MAX_CHUNK_SIZE,
      help=f"Maximum chunk size for processing (default: {DEF_MAX_CHUNK_SIZE})")
  parser.add_argument("-t", "--temperature", type=float, default=DEF_TEMPERATURE,
      help=f"Temperature for text generation, 0.0 - 1.0 (default: {DEF_TEMPERATURE})")
  parser.add_argument("-o", "--output",
      help="Output file path (default: stdout)")
  parser.add_argument('-v', '--verbose', default=False, action='store_true',
      help='Enable verbose output')
  parser.add_argument('-d', '--debug', default=False, action='store_true',
      help='Enable debug output')

  args = parser.parse_args()

  # Set up logging based on verbose/debug options
  logger = setup_logging(args.verbose, args.debug)

  try:
    with open(args.input_file, 'r') as file:
      input_text = file.read()
  except IOError as e:
    logging.error(f"Error reading input file: {str(e)}")
    sys.exit(1)

  generated_text = process_transcript(input_text,
      model=args.model,
      max_tokens=args.max_tokens,
      temperature=args.temperature,
      context=args.context,
      language=args.input_language,
      max_chunk_size=args.max_chunk_size,
    )

  if args.output:
    try:
      with open(args.output, 'w') as file:
        file.write(generated_text)
    except IOError as e:
      logging.error(f"Error writing to output file: {str(e)}")
      sys.exit(1)
  else:
    print(generated_text)

if __name__ == "__main__":
  main()

#fin
