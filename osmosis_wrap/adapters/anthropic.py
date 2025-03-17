"""
Anthropic adapter for Osmosis Wrap

This module provides monkey patching for the Anthropic Python client.
"""

import functools
import sys

from osmosis_wrap.utils import send_to_hoover, enabled
from osmosis_wrap import utils

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
    
    # Get the resources.messages module and class
    messages_module = anthropic.resources.messages
    messages_class = messages_module.Messages

    # Patch the Messages.create method
    original_messages_create = messages_class.create
    if not hasattr(original_messages_create, "_osmosis_wrapped"):
        @functools.wraps(original_messages_create)
        def wrapped_messages_create(self, *args, **kwargs):
            response = original_messages_create(self, *args, **kwargs)
            
            if utils.enabled:
                send_to_hoover(
                    query=kwargs,
                    response=response.model_dump() if hasattr(response, 'model_dump') else response,
                    status=200
                )
                
            return response
        
        wrapped_messages_create._osmosis_wrapped = True
        messages_class.create = wrapped_messages_create
    
    # Patch the async create method if it exists
    if hasattr(messages_class, "acreate"):
        original_acreate = messages_class.acreate
        if not hasattr(original_acreate, "_osmosis_wrapped"):
            @functools.wraps(original_acreate)
            async def wrapped_acreate(self, *args, **kwargs):
                response = await original_acreate(self, *args, **kwargs)
                
                if utils.enabled:
                    send_to_hoover(
                        query=kwargs,
                        response=response.model_dump() if hasattr(response, 'model_dump') else response,
                        status=200
                    )
                    
                return response
            
            wrapped_acreate._osmosis_wrapped = True
            messages_class.acreate = wrapped_acreate

    # Check if Completions exists and wrap it if it does
    try:
        completions_module = anthropic.resources.completions
        completions_class = completions_module.Completions
        
        original_completions_create = completions_class.create
        if not hasattr(original_completions_create, "_osmosis_wrapped"):
            @functools.wraps(original_completions_create)
            def wrapped_completions_create(self, *args, **kwargs):
                response = original_completions_create(self, *args, **kwargs)
                
                if utils.enabled:
                    send_to_hoover(
                        query=kwargs,
                        response=response.model_dump() if hasattr(response, 'model_dump') else response,
                        status=200
                    )
                    
                return response
            
            wrapped_completions_create._osmosis_wrapped = True
            completions_class.create = wrapped_completions_create
            
            # Patch the async create method if it exists
            if hasattr(completions_class, "acreate"):
                original_completions_acreate = completions_class.acreate
                if not hasattr(original_completions_acreate, "_osmosis_wrapped"):
                    @functools.wraps(original_completions_acreate)
                    async def wrapped_completions_acreate(self, *args, **kwargs):
                        response = await original_completions_acreate(self, *args, **kwargs)
                        
                        if utils.enabled:
                            send_to_hoover(
                                query=kwargs,
                                response=response.model_dump() if hasattr(response, 'model_dump') else response,
                                status=200
                            )
                            
                        return response
                    
                    wrapped_completions_acreate._osmosis_wrapped = True
                    completions_class.acreate = wrapped_completions_acreate
    except (ImportError, AttributeError):
        # Completions module may not exist in this version
        pass

    print("Anthropic client has been wrapped by osmosis-wrap.", file=sys.stderr) 