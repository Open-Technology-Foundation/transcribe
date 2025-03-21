#!/usr/bin/env python
from typing import List, Tuple
import langcodes

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

def main() -> None:
  """Main function to run when the script is executed directly."""
  import argparse
  parser = argparse.ArgumentParser(
    description="Display language codes and names."
  )
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
    "-l",
    "--list",
    action="store_true",
    help="List all two-letter language codes and their names."
  )
  group.add_argument(
    "-c",
    "--code",
    metavar="CODE",
    type=str,
    help="Get the language name for the specified two-letter code."
  )

  args = parser.parse_args()

  if args.list:
    display_all_languages()
    print(f"{len(langcodes.LANGUAGE_ALPHA3)} languages")
  elif args.code:
    name = get_language_name(args.code)
    print(name)
  else:
    display_all_languages()
    print(f"{len(langcodes.LANGUAGE_ALPHA3)} languages")

if __name__ == "__main__":
  main()

#fin
