"""
Anthropic adapter for Osmosis Wrap

This module provides monkey patching for the Anthropic Python client.
"""

import functools
import inspect
import sys
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from osmosis_wrap.utils import _send_to_hoover

# Flag to control whether sending to Hoover is enabled
enabled = True

def wrap_anthropic() -> None:
    """
    Monkey patch Anthropic's client to send all prompts and responses to Hoover.
    
    This function should be called before creating any Anthropic client instances.
    """
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package is not installed.", file=sys.stderr)
        return

    # Store original methods
    original_create = getattr(anthropic.Client, "messages", None)
    original_complete = getattr(anthropic.Client, "completions", None)

    # Monkey patch the messages method
    if original_create and not hasattr(original_create, "_osmosis_wrapped"):
        @functools.wraps(original_create)
        def wrapped_messages(self, *args, **kwargs):
            response = original_create(self, *args, **kwargs)
            
            if enabled:
                _send_to_hoover(
                    query=kwargs,
                    response=response.model_dump() if hasattr(response, 'model_dump') else response,
                    status=200
                )
                
            return response
        
        wrapped_messages._osmosis_wrapped = True
        setattr(anthropic.Client, "messages", wrapped_messages)
        
    # Monkey patch the completions method if it exists
    if original_complete and not hasattr(original_complete, "_osmosis_wrapped"):
        @functools.wraps(original_complete)
        def wrapped_completions(self, *args, **kwargs):
            response = original_complete(self, *args, **kwargs)
            
            if enabled:
                _send_to_hoover(
                    query=kwargs,
                    response=response.model_dump() if hasattr(response, 'model_dump') else response,
                    status=200
                )
                
            return response
        
        wrapped_completions._osmosis_wrapped = True
        setattr(anthropic.Client, "completions", wrapped_completions)

    # For newer Anthropic client versions, we need to patch the async methods too
    for name, method in inspect.getmembers(anthropic.Client):
        if (name.startswith("a") and (name.endswith("messages") or name.endswith("completions")) 
            and inspect.iscoroutinefunction(method) 
            and not hasattr(method, "_osmosis_wrapped")):
            
            original_method = method
            
            @functools.wraps(original_method)
            async def wrapped_async_method(self, *args, **kwargs):
                response = await original_method(self, *args, **kwargs)
                
                if enabled:
                    _send_to_hoover(
                        query=kwargs,
                        response=response.model_dump() if hasattr(response, 'model_dump') else response,
                        status=200
                    )
                    
                return response
            
            wrapped_async_method._osmosis_wrapped = True
            setattr(anthropic.Client, name, wrapped_async_method)

    print("Anthropic client has been wrapped by osmosis-wrap.", file=sys.stderr) 