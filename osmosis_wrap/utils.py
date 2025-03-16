"""
Utility functions for Osmosis Wrap adapters
"""

import json
import os
import sys
from typing import Any, Dict

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use environment variables as is

# Global configuration flags (with environment variable fallbacks)
enabled = True
use_stderr = os.environ.get("OSMOSIS_USE_STDERR", "").lower() in ("true", "1", "yes")
pretty_print = os.environ.get("OSMOSIS_PRETTY_PRINT", "true").lower() not in ("false", "0", "no")
print_messages = os.environ.get("OSMOSIS_PRINT_RESPONSES", "true").lower() not in ("false", "0", "no")
indent = int(os.environ.get("OSMOSIS_INDENT", "2"))

def _get_output_stream():
    """Get the output stream to print to."""
    return sys.stderr if use_stderr else sys.stdout

def _print_json(data: Dict[str, Any]) -> None:
    """Print a dictionary as JSON."""
    if pretty_print:
        print(json.dumps(data, indent=indent), file=_get_output_stream())
    else:
        print(json.dumps(data), file=_get_output_stream())

def get_api_key(provider: str) -> str:
    """
    Get API key for a specific provider from environment variables.
    
    Args:
        provider: The name of the provider (e.g., 'anthropic', 'openai')
        
    Returns:
        The API key for the specified provider, or None if not found.
    """
    key_name = f"{provider.upper()}_API_KEY"
    return os.environ.get(key_name) 