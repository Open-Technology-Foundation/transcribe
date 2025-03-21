#!/usr/bin/env python3
"""
Logging utilities for the transcription package.

This module provides centralized logging configuration and utilities
to ensure consistent logging across all components of the package.
"""
import logging
import colorlog

# Create a filter to suppress HTTP request logs at INFO level
class HttpRequestFilter(logging.Filter):
    """Filter to suppress HTTP request log messages."""
    
    def filter(self, record):
        """Filter out HTTP Request messages at INFO level."""
        if record.levelno == logging.INFO and "HTTP Request" in record.getMessage():
            return False
        return True

def setup_logging(verbose=False, debug=False, log_file=None):
    """
    Set up logging configuration with color and optional file output.
    
    Args:
        verbose (bool): Enable verbose output at INFO level
        debug (bool): Enable debug output (overrides verbose)
        log_file (str, optional): Path to log file for file-based logging
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger()
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set appropriate log level
    if debug:
        logger.setLevel(logging.DEBUG)
        datefmt = "%H:%M:%S"
        logformat = f"%(log_color)s%(asctime)s:%(module)s:%(levelname)s: %(message)s"
    elif verbose:
        logger.setLevel(logging.INFO)
        datefmt = "%H:%M:%S"
        logformat = f"%(log_color)s%(asctime)s:%(module)s:%(levelname)s: %(message)s"
    else:
        logger.setLevel(logging.ERROR)
        datefmt = None
        logformat = f"%(log_color)s%(module)s:%(levelname)s: %(message)s"
    
    # Configure console handler with color
    console_handler = colorlog.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        logformat,
        datefmt=datefmt,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    console_handler.setFormatter(console_formatter)
    # Add the HTTP request filter to suppress INFO level HTTP request logs
    console_handler.addFilter(HttpRequestFilter())
    logger.addHandler(console_handler)
    
    # Also add the filter to the OpenAI client loggers to ensure they don't show HTTP requests at INFO level
    for logger_name in ["openai._client", "openai.http_client", "_client"]:
        openai_logger = logging.getLogger(logger_name)
        if openai_logger:
            openai_logger.addFilter(HttpRequestFilter())
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s:%(module)s:%(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name=None):
    """
    Get a named logger instance.
    
    Args:
        name (str, optional): Logger name for module-specific logging
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

#fin