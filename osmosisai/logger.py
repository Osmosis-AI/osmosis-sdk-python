"""
Logging utilities for Osmosis AI
"""

import sys
import logging
import os
from .consts import log_destination as default_log_destination

# Create logger
logger = logging.getLogger('osmosisai')
logger.setLevel(logging.INFO)

# Default configuration - no logging
logger.propagate = False

def configure_logger(destination: str = None) -> None:
    """
    Configure the logger based on the log_destination setting
    
    Args:
        destination: Optional override for log destination
    """
    # Use provided destination or get from utils
    if destination is None:
        # Import here to avoid circular import
        from .utils import get_log_destination
        log_dest = get_log_destination()
    else:
        log_dest = destination
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Configure based on log_destination
    if log_dest == "none":
        # No logging
        return
    elif log_dest == "stdout":
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
    elif log_dest == "stderr":
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
    elif log_dest == "file":
        # Log to a file in the current directory
        log_file = os.environ.get("OSMOSIS_LOG_FILE", "osmosisai.log")
        handler = logging.FileHandler(log_file)
        logger.addHandler(handler)
    else:
        # Invalid setting, log to stderr as fallback
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
        logger.warning(f"Invalid log_destination: {log_dest}, using stderr")
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in logger.handlers:
        handler.setFormatter(formatter)

# Initialize the logger on module import with default settings
configure_logger(default_log_destination)

# Function to force reconfiguration if log_destination is changed
def reconfigure_logger() -> None:
    """
    Reconfigure the logger, should be called if log_destination changes
    """
    configure_logger() 