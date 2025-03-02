#!/usr/bin/env python3
"""
Language code utilities for the transcribe package.
"""

from typing import List, Tuple
import langcodes
from transcribe_pkg.utils.api_utils import call_llm

def get_language_codes() -> List[Tuple[str, str]]:
  """
  Retrieve a sorted list of two-letter language codes and their names.

  Returns:
    List[Tuple[str, str]]: A list of tuples containing (code, name) pairs.
  """
  languages = []
  for lang_code in langcodes.LANGUAGE_ALPHA3:
    lang = langcodes.Language.make(language=lang_code)
    code = lang.language
    if len(code) == 2:  # Only process if we have a 2-letter code
      name = lang.display_name()
      languages.append((code, name))
  return sorted(languages, key=lambda x: x[1])

def display_all_languages() -> None:
  """Display all languages with their two-letter codes."""
  languages = get_language_codes()
  for code, name in languages:
    print(f"{code} {name}")

def get_language_name(code: str) -> str:
  """
  Get the language name for a given two-letter language code.

  Args:
    code (str): A two-letter language code.

  Returns:
    str: The language name or an error message if the code is invalid.
  """
  try:
    lang = langcodes.Language.make(language=code[:2].lower())
    return lang.display_name()
  except langcodes.tag_parser.LanguageTagError:
    return f"!!Unknown language [{code}]"

def determine_language(input_text, model='gpt-4o-mini'):
  """
  Determine the language of the input text using AI.
  
  Args:
    input_text (str): Text to determine language for
    model (str, optional): Model to use. Defaults to 'gpt-4o-mini'.
    
  Returns:
    str: Two-letter language code (e.g., 'en', 'fr')
  """
  systemprompt="""
  You are an expert in determining what language a text is written in.
  
  You return one two-character language code that corresponds to the language of the text; eg: en, id, ko, zh, ja.
  
  Make no other commentary or preamble. Just output the language code.
  """
  lang = call_llm(systemprompt, input_text, model, 0, 50)
  if 'Unknown' in get_language_name(lang):
    lang = 'en'
  return lang

#fin