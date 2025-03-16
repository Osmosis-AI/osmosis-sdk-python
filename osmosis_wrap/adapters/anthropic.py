"""
Anthropic adapter for Osmosis Wrap

This module provides monkey patching for the Anthropic Python client.
"""

import functools
import json
import sys
import inspect
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from osmosis_wrap.utils import _get_output_stream, _print_json

# Flag to control whether printing is enabled
enabled = True
# Flag to control printing of messages
print_messages = True

def wrap_anthropic() -> None:
    """
    Monkey patch Anthropic's client to print all prompts and responses.
    
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
            if enabled:
                print("\n=== ANTHROPIC REQUEST ===", file=_get_output_stream())
                _print_json(kwargs)
                
            response = original_create(self, *args, **kwargs)
            
            if enabled and print_messages:
                print("\n=== ANTHROPIC RESPONSE ===", file=_get_output_stream())
                _print_json(response)
                
            return response
        
        wrapped_messages._osmosis_wrapped = True
        setattr(anthropic.Client, "messages", wrapped_messages)
        
    # Monkey patch the completions method if it exists
    if original_complete and not hasattr(original_complete, "_osmosis_wrapped"):
        @functools.wraps(original_complete)
        def wrapped_completions(self, *args, **kwargs):
            if enabled:
                print("\n=== ANTHROPIC COMPLETIONS REQUEST ===", file=_get_output_stream())
                _print_json(kwargs)
                
            response = original_complete(self, *args, **kwargs)
            
            if enabled and print_messages:
                print("\n=== ANTHROPIC COMPLETIONS RESPONSE ===", file=_get_output_stream())
                _print_json(response)
                
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
                method_name = original_method.__name__
                
                if enabled:
                    print(f"\n=== ANTHROPIC ASYNC {method_name.upper()} REQUEST ===", file=_get_output_stream())
                    _print_json(kwargs)
                    
                response = await original_method(self, *args, **kwargs)
                
                if enabled and print_messages:
                    print(f"\n=== ANTHROPIC ASYNC {method_name.upper()} RESPONSE ===", file=_get_output_stream())
                    _print_json(response)
                    
                return response
            
            wrapped_async_method._osmosis_wrapped = True
            setattr(anthropic.Client, name, wrapped_async_method)

    print("Anthropic client has been wrapped by osmosis-wrap.", file=_get_output_stream()) 