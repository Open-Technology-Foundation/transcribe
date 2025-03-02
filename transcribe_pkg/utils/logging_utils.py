#!/usr/bin/env python3
"""
Logging utilities for the transcribe package.
"""

import logging
import colorlog

def setup_logging(verbose, debug=False):
  """
  Set up logging configuration with color.
  
  Args:
    verbose (bool): Enable verbose logging (INFO level)
    debug (bool): Enable debug logging (DEBUG level)
    
  Returns:
    logging.Logger: Configured logger instance
  """
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
    log_colors={ 
      'DEBUG': 'cyan', 
      'INFO': 'green', 
      'WARNING': 'yellow', 
      'ERROR': 'red', 
      'CRITICAL': 'red,bg_white' 
    },
    secondary_log_colors={},
    style='%'
  )
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  return logger

#fin