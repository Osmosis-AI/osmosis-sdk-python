"""
Utility functions for Osmosis Wrap adapters
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import urllib.request
import urllib.error

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use environment variables as is

# Global configuration
enabled = True
use_stderr = True  # Added missing configuration
pretty_print = True  # Controls whether to format output nicely
print_messages = True  # Controls whether to print messages at all
indent = 2  # Number of spaces to use for indentation in pretty print
hoover_api_key = None  # Will be set by init()
hoover_api_url = "https://ha2udfkbnh.execute-api.us-west-2.amazonaws.com/store-usage"
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

def send_to_hoover(query: Dict[str, Any], response: Dict[str, Any], status: int = 200) -> None:
    """
    Send query and response data to the Hoover API.
    
    Args:
        query: The query/request data
        response: The response data
        status: The HTTP status code (default: 200)
    """
    if not enabled or not hoover_api_key:
        if not _initialized:
            print("Warning: Osmosis Wrap not initialized. Call osmosis_wrap.init(api_key) first.", file=sys.stderr)
        return

    try:
        data = {
            "date": datetime.now(timezone.utc).astimezone(timezone.utc).isoformat(),
            "query": json.dumps(query),
            "response": json.dumps(response),
            "status": status,
            "apiKey": hoover_api_key
        }

        # Create request
        headers = {
            "Content-Type": "application/json"
        }
        
        # Convert data to JSON and encode as bytes
        data_bytes = json.dumps(data).encode('utf-8')
        
        # Create request object
        req = urllib.request.Request(
            hoover_api_url,
            data=data_bytes,
            headers=headers,
            method='POST'
        )
        
        # Send request
        with urllib.request.urlopen(req) as response:
            if response.status != 200:
                print(f"Warning: Hoover API returned status {response.status}", file=sys.stderr)
    
    except Exception as e:
        print(f"Warning: Failed to send data to Hoover API: {str(e)}", file=sys.stderr)

def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a specific provider from environment variables.
    
    Args:
        provider: The name of the provider (e.g., 'anthropic', 'openai', 'hoover')
        
    Returns:
        The API key for the specified provider, or None if not found.
    """
    key_name = f"{provider.upper()}_API_KEY"
    return os.environ.get(key_name) 