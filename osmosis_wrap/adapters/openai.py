"""
OpenAI adapter for Osmosis Wrap

This module provides monkey patching for the OpenAI Python client.
"""

import functools
import inspect
import sys
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from osmosis_wrap.utils import _send_to_hoover

# Flag to control whether sending to Hoover is enabled
enabled = True

def wrap_openai() -> None:
    """
    Monkey patch OpenAI's client to send all prompts and responses to Hoover.
    
    This function should be called before creating any OpenAI client instances.
    """
    try:
        import openai
    except ImportError:
        print("Error: openai package is not installed.", file=sys.stderr)
        return

    # Check if using v1 of the OpenAI client
    try:
        from openai import OpenAI
        _wrap_openai_v1()
    except (ImportError, AttributeError):
        # Fall back to legacy client
        _wrap_openai_legacy()

def _wrap_openai_v1() -> None:
    """Monkey patch the OpenAI v1 client."""
    from openai import OpenAI
    from openai.resources.chat import completions
    from openai.resources import completions as text_completions
    
    # Patch the chat completions create method
    original_chat_create = completions.Completions.create
    if not hasattr(original_chat_create, "_osmosis_wrapped"):
        @functools.wraps(original_chat_create)
        def wrapped_chat_create(self, *args, **kwargs):
            response = original_chat_create(self, *args, **kwargs)
            
            if enabled:
                _send_to_hoover(
                    query=kwargs,
                    response=response.model_dump() if hasattr(response, 'model_dump') else response,
                    status=200
                )
                
            return response
        
        wrapped_chat_create._osmosis_wrapped = True
        completions.Completions.create = wrapped_chat_create
    
    # Patch the completions create method
    original_completions_create = text_completions.Completions.create
    if not hasattr(original_completions_create, "_osmosis_wrapped"):
        @functools.wraps(original_completions_create)
        def wrapped_completions_create(self, *args, **kwargs):
            response = original_completions_create(self, *args, **kwargs)
            
            if enabled:
                _send_to_hoover(
                    query=kwargs,
                    response=response.model_dump() if hasattr(response, 'model_dump') else response,
                    status=200
                )
                
            return response
        
        wrapped_completions_create._osmosis_wrapped = True
        text_completions.Completions.create = wrapped_completions_create
    
    # Find and wrap async methods
    for module in [completions, text_completions]:
        for name, method in inspect.getmembers(module.Completions):
            if (name.startswith("a") and name.endswith("create") 
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
                setattr(module.Completions, name, wrapped_async_method)
    
    print("OpenAI v1 client has been wrapped by osmosis-wrap.", file=sys.stderr)

def _wrap_openai_legacy() -> None:
    """Monkey patch the legacy OpenAI client."""
    import openai
    
    # Patch the Completion.create method
    original_completion_create = openai.Completion.create
    if not hasattr(original_completion_create, "_osmosis_wrapped"):
        @functools.wraps(original_completion_create)
        def wrapped_completion_create(*args, **kwargs):
            response = original_completion_create(*args, **kwargs)
            
            if enabled:
                _send_to_hoover(
                    query=kwargs,
                    response=response,
                    status=200
                )
                
            return response
        
        wrapped_completion_create._osmosis_wrapped = True
        openai.Completion.create = wrapped_completion_create
    
    # Patch the ChatCompletion.create method
    if hasattr(openai, "ChatCompletion"):
        original_chat_create = openai.ChatCompletion.create
        if not hasattr(original_chat_create, "_osmosis_wrapped"):
            @functools.wraps(original_chat_create)
            def wrapped_chat_create(*args, **kwargs):
                response = original_chat_create(*args, **kwargs)
                
                if enabled:
                    _send_to_hoover(
                        query=kwargs,
                        response=response,
                        status=200
                    )
                    
                return response
            
            wrapped_chat_create._osmosis_wrapped = True
            openai.ChatCompletion.create = wrapped_chat_create
            
    # Patch the async methods if they exist
    for obj in [openai.Completion, getattr(openai, "ChatCompletion", None)]:
        if obj is None:
            continue
            
        if hasattr(obj, "acreate"):
            original_acreate = obj.acreate
            if not hasattr(original_acreate, "_osmosis_wrapped"):
                @functools.wraps(original_acreate)
                async def wrapped_acreate(*args, **kwargs):
                    response = await original_acreate(*args, **kwargs)
                    
                    if enabled:
                        _send_to_hoover(
                            query=kwargs,
                            response=response,
                            status=200
                        )
                        
                    return response
                
                wrapped_acreate._osmosis_wrapped = True
                obj.acreate = wrapped_acreate
    
    print("OpenAI legacy client has been wrapped by osmosis-wrap.", file=sys.stderr) 