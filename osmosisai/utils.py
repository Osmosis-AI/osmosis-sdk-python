"""
Utility functions for Osmosis AI adapters
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict
import xxhash
# Import constants
from .consts import osmosis_api_url
# Import logger
from .logger import logger

# Global configuration
enabled = True
print_messages = True  # Controls whether to print messages at all
osmosis_api_key = None  # Will be set by init()
_initialized = False

def init(api_key: str) -> None:
    """
    Initialize Osmosis AI with the OSMOSIS API key.
    
    Args:
        api_key: The OSMOSIS API key for logging LLM usage
    """
    global osmosis_api_key, _initialized
    osmosis_api_key = api_key
    _initialized = True

def disable_osmosis() -> None:
    global enabled
    enabled = False

def enable_osmosis() -> None:
    global enabled
    enabled = True

def send_to_osmosis(query: Dict[str, Any], response: Dict[str, Any], status: int = 200) -> None:
    """
    Send query and response data to the OSMOSIS API using AWS Firehose.
    
    Args:
        query: The query/request data
        response: The response data
        status: The HTTP status code (default: 200)
    """
    if not enabled or not osmosis_api_key:
        return

    if not _initialized:
        logger.warning("Osmosis AI not initialized. Call osmosisai.init(api_key) first.")
        return

    try:
        # Import requests only when needed
        import requests
        
        # Create headers
        headers = {
            "Content-Type": "application/json",
            "x-api-key": osmosis_api_key
        }
            
        # Prepare main data payload
        data = {
            "owner": xxhash.xxh64(osmosis_api_key).hexdigest(),
            "date": int(datetime.now(timezone.utc).timestamp()),
            "query": query,
            "response": response,
            "status": status
        }
        
        # Send main data payload
        response_data = requests.post(
            f"{osmosis_api_url}/data",
            headers=headers,
            data=json.dumps(data).replace('\n', '') + '\n'
        )
        
        if response_data.status_code != 200:
            logger.warning(f"OSMOSIS API returned status {response_data.status_code} for data")
    
    except ImportError:
        logger.warning("Requests library not installed. Please install it with 'pip install requests'.")
    except Exception as e:
        logger.warning(f"Failed to send data to OSMOSIS API: {str(e)}")