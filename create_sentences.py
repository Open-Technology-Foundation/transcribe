#!/usr/bin/env python3
import nltk
from nltk.tokenize import sent_tokenize

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

def create_sentences(text, *, max_sentence_length=3000):
  """
  Create logical sentences from text, using byte length.
  """
  # Download necessary NLTK data
  if not hasattr(create_sentences, "initialized"):
    nltk.download('punkt', quiet=True)
    create_sentences.initialized = True
  Sentences = []
  sents = sent_tokenize(text.replace('\r\n', '\n').replace('\r', ''))
  for sentence in sents:
    # check for overlong sentences
    if len(sentence.encode('utf-8')) <= max_sentence_length:
      Sentences.append(sentence.replace('\n', ' ').rstrip() + ' ')
    else:
      newsent = ''
      Words = sentence.replace('\n', ' ').split(' ')
      for word in Words:
        word_bytes = (word + ' ').encode('utf-8')
        if len((newsent + word).encode('utf-8')) >= max_sentence_length:
          Sentences.append(newsent.rstrip())
          newsent = ''
        newsent += word + ' '
      if newsent:
        Sentences.append(newsent.rstrip())
  return Sentences

def main():
  import sys
  import logging
  import argparse

  # Set up the signal handler
  import signal
  def signal_handler(sig, frame):
    print('\033[0m^C\n')
    sys.exit(130)
  signal.signal(signal.SIGINT, signal_handler)

  """
  Create sentences from text
  """
  parser = argparse.ArgumentParser(
    description="Create logical sentences from unstructured text.",
    epilog='Example: create_sentences textile.txt -s 2000 -S "|"'
  )
  parser.add_argument("input_file",
      help="Path to the text file")
  parser.add_argument('-s', '--max-sentence-size', type=int,
      help='Max size of sentence in bytes (default: 3000)', default=3000)
  parser.add_argument('-S', '--suffix', default='',
      help='Suffix to append to each printed line (def: "")')
  parser.add_argument('-v', '--verbose', default=False, action='store_true',
      help='Enable verbose output')
  parser.add_argument('-d', '--debug', default=False, action='store_true',
      help='Enable debug output')
  args = parser.parse_args()

  # Set up logging based on verbose/quiet options
  logger = setup_logging(args.verbose, args.debug)

  try:
    with open(args.input_file, 'r') as file:
      input_text = file.read()
  except IOError as e:
    logging.error(f"Error reading input file: {str(e)}")
    sys.exit(1)

  sentences = create_sentences(input_text.rstrip(), max_sentence_length=args.max_sentence_size)
  for sentence in sentences:
    print(f"{sentence}{args.suffix}")

if __name__ == "__main__":
  main()

#fin
