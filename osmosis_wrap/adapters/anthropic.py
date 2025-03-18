"""
Anthropic adapter for Osmosis Wrap

This module provides monkey patching for the Anthropic Python client.
"""

import functools
import sys

from osmosis_wrap.utils import send_to_hoover
from osmosis_wrap import utils

def wrap_anthropic() -> None:
    """
    Monkey patch Anthropic's client to send all prompts and responses to Hoover.
    
    This function should be called before creating any Anthropic client instances.
    
    Features supported:
    - Basic message completions
    - Async message completions
    - Tool use (function calling)
    - Tool responses
    - Streaming (if available)
    """
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package is not installed.", file=sys.stderr)
        return
    
    print(f"Wrapping Anthropic client, package version: {anthropic.__version__}", file=sys.stderr)
    
    # Get the resources.messages module and class
    messages_module = anthropic.resources.messages
    messages_class = messages_module.Messages
    
    print(f"Found Anthropic messages class: {messages_class}", file=sys.stderr)

    # Patch the Messages.create method
    original_messages_create = messages_class.create
    print(f"Original create method: {original_messages_create}", file=sys.stderr)
    
    if not hasattr(original_messages_create, "_osmosis_wrapped"):
        @functools.wraps(original_messages_create)
        def wrapped_messages_create(self, *args, **kwargs):
            print(f"Wrapped create called with args: {args}, kwargs: {kwargs}", file=sys.stderr)
            try:
                # Check if the call includes tool use parameters
                has_tools = "tools" in kwargs
                if has_tools:
                    print(f"Tool use detected with {len(kwargs['tools'])} tools", file=sys.stderr)
                
                # Check if this is a tool response message
                if "messages" in kwargs:
                    for message in kwargs["messages"]:
                        if message.get("role") == "user" and isinstance(message.get("content"), list):
                            for content_item in message["content"]:
                                if isinstance(content_item, dict) and content_item.get("type") == "tool_response":
                                    print("Tool response detected in messages", file=sys.stderr)
                                    break
                
                response = original_messages_create(self, *args, **kwargs)
                
                if utils.enabled:
                    print("Sending success to Hoover (success)", file=sys.stderr)
                    send_to_hoover(
                        query=kwargs,
                        response=response.model_dump() if hasattr(response, 'model_dump') else response,
                        status=200
                    )
                    
                return response
            except Exception as e:
                print(f"Error in wrapped create: {e}", file=sys.stderr)
                if utils.enabled:
                    error_response = {"error": str(e)}
                    send_to_hoover(
                        query=kwargs,
                        response=error_response,
                        status=400
                    )
                    print("Sending error to Hoover (success)", file=sys.stderr)
                raise  # Re-raise the exception
        
        wrapped_messages_create._osmosis_wrapped = True
        messages_class.create = wrapped_messages_create
        print("Successfully wrapped Messages.create method", file=sys.stderr)
    
    # Directly wrap the AsyncAnthropic client
    try:
        # Get the AsyncAnthropic class
        AsyncAnthropicClass = anthropic.AsyncAnthropic
        print(f"Found AsyncAnthropic class: {AsyncAnthropicClass}", file=sys.stderr)
        
        # Store the original __init__ to keep track of created instances
        original_async_init = AsyncAnthropicClass.__init__
        
        if not hasattr(original_async_init, "_osmosis_wrapped"):
            @functools.wraps(original_async_init)
            def wrapped_async_init(self, *args, **kwargs):
                # Call the original init
                result = original_async_init(self, *args, **kwargs)
                
                print("Wrapping new AsyncAnthropic instance's messages.create method", file=sys.stderr)
                
                # Get the messages client from this instance
                async_messages = self.messages
                
                # Store and patch the create method if not already wrapped
                if hasattr(async_messages, "create") and not hasattr(async_messages.create, "_osmosis_wrapped"):
                    original_async_messages_create = async_messages.create
                    
                    @functools.wraps(original_async_messages_create)
                    async def wrapped_async_messages_create(*args, **kwargs):
                        print(f"AsyncAnthropic.messages.create called with args: {args}, kwargs: {kwargs}", file=sys.stderr)
                        try:
                            response = await original_async_messages_create(*args, **kwargs)
                            
                            if utils.enabled:
                                print("Sending AsyncAnthropic response to Hoover (success)", file=sys.stderr)
                                send_to_hoover(
                                    query=kwargs,
                                    response=response.model_dump() if hasattr(response, 'model_dump') else response,
                                    status=200
                                )
                            
                            return response
                        except Exception as e:
                            print(f"Error in wrapped AsyncAnthropic.messages.create: {e}", file=sys.stderr)
                            if utils.enabled:
                                print("Sending AsyncAnthropic error to Hoover", file=sys.stderr)
                                error_response = {"error": str(e)}
                                send_to_hoover(
                                    query=kwargs,
                                    response=error_response,
                                    status=400
                                )
                            raise  # Re-raise the exception
                    
                    wrapped_async_messages_create._osmosis_wrapped = True
                    async_messages.create = wrapped_async_messages_create
                    print("Successfully wrapped AsyncAnthropic.messages.create method", file=sys.stderr)
                
                return result
            
            wrapped_async_init._osmosis_wrapped = True
            AsyncAnthropicClass.__init__ = wrapped_async_init
            print("Successfully wrapped AsyncAnthropic.__init__ to patch message methods on new instances", file=sys.stderr)
    except (ImportError, AttributeError) as e:
        print(f"AsyncAnthropic class not found or has unexpected structure: {e}", file=sys.stderr)
    
    # For compatibility, still try to patch the old-style acreate method if it exists
    if hasattr(messages_class, "acreate"):
        original_acreate = messages_class.acreate
        if not hasattr(original_acreate, "_osmosis_wrapped"):
            @functools.wraps(original_acreate)
            async def wrapped_acreate(self, *args, **kwargs):
                print(f"Wrapped async create called with args: {args}, kwargs: {kwargs}", file=sys.stderr)
                try:
                    # Check if the async call includes tool use parameters
                    has_tools = "tools" in kwargs
                    if has_tools:
                        print(f"Async tool use detected with {len(kwargs['tools'])} tools", file=sys.stderr)
                    
                    if "messages" in kwargs:
                        for message in kwargs["messages"]:
                            if message.get("role") == "user" and isinstance(message.get("content"), list):
                                for content_item in message["content"]:
                                    if isinstance(content_item, dict) and content_item.get("type") == "tool_response":
                                        print("Async tool response detected in messages", file=sys.stderr)
                                        break
                    
                    response = await original_acreate(self, *args, **kwargs)
                    
                    if utils.enabled:
                        print("Sending async response to Hoover (success)", file=sys.stderr)
                        send_to_hoover(
                            query=kwargs,
                            response=response.model_dump() if hasattr(response, 'model_dump') else response,
                            status=200
                        )
                        
                    return response
                except Exception as e:
                    print(f"Error in wrapped async create: {e}", file=sys.stderr)
                    if utils.enabled:
                        print("Sending async error to Hoover", file=sys.stderr)
                        error_response = {"error": str(e)}
                        send_to_hoover(
                            query=kwargs,
                            response=error_response,
                            status=400
                        )
                    raise  # Re-raise the exception
            
            wrapped_acreate._osmosis_wrapped = True
            messages_class.acreate = wrapped_acreate
            print("Successfully wrapped Messages.acreate method", file=sys.stderr)
    else:
        print("No async acreate method found in Messages class", file=sys.stderr)

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
                        print(f"Wrapped Completions async create called with args: {args}, kwargs: {kwargs}", file=sys.stderr)
                        try:
                            response = await original_completions_acreate(self, *args, **kwargs)
                            
                            if utils.enabled:
                                print("Sending Completions async response to Hoover (success)", file=sys.stderr)
                                send_to_hoover(
                                    query=kwargs,
                                    response=response.model_dump() if hasattr(response, 'model_dump') else response,
                                    status=200
                                )
                                
                            return response
                        except Exception as e:
                            print(f"Error in wrapped Completions async create: {e}", file=sys.stderr)
                            if utils.enabled:
                                print("Sending Completions async error to Hoover", file=sys.stderr)
                                error_response = {"error": str(e)}
                                send_to_hoover(
                                    query=kwargs,
                                    response=error_response,
                                    status=400
                                )
                            raise  # Re-raise the exception
                    
                    wrapped_completions_acreate._osmosis_wrapped = True
                    completions_class.acreate = wrapped_completions_acreate
                    print("Successfully wrapped Completions.acreate method", file=sys.stderr)
                else:
                    print("Completions.acreate already wrapped", file=sys.stderr)
            else:
                print("No async acreate method found in Completions class", file=sys.stderr)
        else:
            print("Completions.create already wrapped", file=sys.stderr)
    except (ImportError, AttributeError) as e:
        # Completions module may not exist in this version
        print(f"Completions module not found or has an unexpected structure: {e}", file=sys.stderr)

    print("Anthropic client has been wrapped by osmosis-wrap.", file=sys.stderr) 