"""
Utility functions for Osmosis Wrap adapters
"""

import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Import constants
from .consts import hoover_api_url
# Import logger
from .logger import logger

# Global configuration
enabled = True
use_stderr = True  # Added missing configuration
pretty_print = True  # Controls whether to format output nicely
print_messages = True  # Controls whether to print messages at all
hoover_api_key = None  # Will be set by init()
_initialized = False

def init(api_key: str) -> None:
    """
    Initialize Osmosis Wrap with the Hoover API key.
    
    Args:
        api_key: The Hoover API key for logging LLM usage
    """
    global hoover_api_key, _initialized
    hoover_api_key = api_key
    _initialized = True

def disable_hoover() -> None:
    global enabled
    enabled = False

def enable_hoover() -> None:
    global enabled
    enabled = True

def send_to_hoover(query: Dict[str, Any], response: Dict[str, Any], status: int = 200) -> None:
    """
    Send query and response data to the Hoover API using AWS Firehose.
    
    Args:
        query: The query/request data
        response: The response data
        status: The HTTP status code (default: 200)
    """
    if not enabled or not hoover_api_key:
        return

    if not _initialized:
        logger.warning("Osmosis Wrap not initialized. Call osmosis_wrap.init(api_key) first.")
        return

    try:
        # Import requests only when needed
        import requests
        
        # Create headers
        headers = {
            "Content-Type": "application/json",
            "x-api-key": hoover_api_key
        }
            
        # Prepare main data payload
        data = {
            "owner": hoover_api_key[:10],
            "date": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "response": response,
            "status": status
        }
        
        # Send main data payload
        response_data = requests.post(
            f"{hoover_api_url}/data",
            headers=headers,
            data=json.dumps(data).replace('\n', '') + '\n'
        )
        
        if response_data.status_code != 200:
            logger.warning(f"Hoover API returned status {response_data.status_code} for data")
    
    except ImportError:
        logger.warning("Requests library not installed. Please install it with 'pip install requests'.")
    except Exception as e:
        logger.warning(f"Failed to send data to Hoover API: {str(e)}")