import os
from typing import Optional


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