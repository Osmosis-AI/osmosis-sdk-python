"""
Logging utilities for Osmosis Wrap
"""

import sys
import logging
import os
from typing import Optional
from .consts import log_destination

# Create logger
logger = logging.getLogger('osmosis_wrap')
logger.setLevel(logging.INFO)

# Default configuration - no logging
logger.propagate = False

def configure_logger() -> None:
    """
    Configure the logger based on the log_destination setting in consts.py
    """
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Configure based on log_destination
    if log_destination == "none":
        # No logging
        return
    elif log_destination == "stdout":
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
    elif log_destination == "stderr":
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
    elif log_destination == "file":
        # Log to a file in the current directory
        log_file = os.environ.get("OSMOSIS_LOG_FILE", "osmosis_wrap.log")
        handler = logging.FileHandler(log_file)
        logger.addHandler(handler)
    else:
        # Invalid setting, log to stderr as fallback
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
        logger.warning(f"Invalid log_destination: {log_destination}, using stderr")
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in logger.handlers:
        handler.setFormatter(formatter)

# Initialize the logger on module import
configure_logger()

# Function to force reconfiguration if log_destination is changed
def reconfigure_logger() -> None:
    """
    Reconfigure the logger, should be called if log_destination changes
    """
    configure_logger() 