"""
Anthropic adapter for Osmosis Wrap

This module provides monkey patching for the Anthropic Python client.
"""

import functools
import sys

from osmosis_wrap.utils import send_to_hoover
from osmosis_wrap import utils
from osmosis_wrap.logger import logger

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
        logger.error("anthropic package is not installed.")
        return
    
    logger.info(f"Wrapping Anthropic client, package version: {anthropic.__version__}")
    
    # Get the resources.messages module and class
    messages_module = anthropic.resources.messages
    messages_class = messages_module.Messages
    
    logger.info(f"Found Anthropic messages class: {messages_class}")

    # Patch the Messages.create method
    original_messages_create = messages_class.create
    logger.info(f"Original create method: {original_messages_create}")
    
    if not hasattr(original_messages_create, "_osmosis_wrapped"):
        @functools.wraps(original_messages_create)
        def wrapped_messages_create(self, *args, **kwargs):
            logger.debug(f"Wrapped create called with args: {args}, kwargs: {kwargs}")
            try:
                # Check if the call includes tool use parameters
                has_tools = "tools" in kwargs
                if has_tools:
                    logger.debug(f"Tool use detected with {len(kwargs['tools'])} tools")
                
                # Check if this is a tool response message
                if "messages" in kwargs:
                    for message in kwargs["messages"]:
                        if message.get("role") == "user" and isinstance(message.get("content"), list):
                            for content_item in message["content"]:
                                if isinstance(content_item, dict) and content_item.get("type") == "tool_response":
                                    logger.debug("Tool response detected in messages")
                                    break
                
                response = original_messages_create(self, *args, **kwargs)
                
                if utils.enabled:
                    logger.debug("Sending success to Hoover (success)")
                    send_to_hoover(
                        query=kwargs,
                        response=response.model_dump() if hasattr(response, 'model_dump') else response,
                        status=200
                    )
                    
                return response
            except Exception as e:
                logger.error(f"Error in wrapped create: {e}")
                if utils.enabled:
                    error_response = {"error": str(e)}
                    send_to_hoover(
                        query=kwargs,
                        response=error_response,
                        status=400
                    )
                    logger.debug("Sending error to Hoover (success)")
                raise  # Re-raise the exception
        
        wrapped_messages_create._osmosis_wrapped = True
        messages_class.create = wrapped_messages_create
        logger.info("Successfully wrapped Messages.create method")
    
    # Directly wrap the AsyncAnthropic client
    try:
        # Get the AsyncAnthropic class
        AsyncAnthropicClass = anthropic.AsyncAnthropic
        logger.info(f"Found AsyncAnthropic class: {AsyncAnthropicClass}")
        
        # Store the original __init__ to keep track of created instances
        original_async_init = AsyncAnthropicClass.__init__
        
        if not hasattr(original_async_init, "_osmosis_wrapped"):
            @functools.wraps(original_async_init)
            def wrapped_async_init(self, *args, **kwargs):
                # Call the original init
                result = original_async_init(self, *args, **kwargs)
                
                logger.info("Wrapping new AsyncAnthropic instance's messages.create method")
                
                # Get the messages client from this instance
                async_messages = self.messages
                
                # Store and patch the create method if not already wrapped
                if hasattr(async_messages, "create") and not hasattr(async_messages.create, "_osmosis_wrapped"):
                    original_async_messages_create = async_messages.create
                    
                    @functools.wraps(original_async_messages_create)
                    async def wrapped_async_messages_create(*args, **kwargs):
                        logger.debug(f"AsyncAnthropic.messages.create called with args: {args}, kwargs: {kwargs}")
                        try:
                            response = await original_async_messages_create(*args, **kwargs)
                            
                            if utils.enabled:
                                logger.debug("Sending AsyncAnthropic response to Hoover (success)")
                                send_to_hoover(
                                    query=kwargs,
                                    response=response.model_dump() if hasattr(response, 'model_dump') else response,
                                    status=200
                                )
                            
                            return response
                        except Exception as e:
                            logger.error(f"Error in wrapped AsyncAnthropic.messages.create: {e}")
                            if utils.enabled:
                                logger.debug("Sending AsyncAnthropic error to Hoover")
                                error_response = {"error": str(e)}
                                send_to_hoover(
                                    query=kwargs,
                                    response=error_response,
                                    status=400
                                )
                            raise  # Re-raise the exception
                    
                    wrapped_async_messages_create._osmosis_wrapped = True
                    async_messages.create = wrapped_async_messages_create
                    logger.info("Successfully wrapped AsyncAnthropic.messages.create method")
                
                return result
            
            wrapped_async_init._osmosis_wrapped = True
            AsyncAnthropicClass.__init__ = wrapped_async_init
            logger.info("Successfully wrapped AsyncAnthropic.__init__ to patch message methods on new instances")
    except (ImportError, AttributeError) as e:
        logger.warning(f"AsyncAnthropic class not found or has unexpected structure: {e}")
    
    # For compatibility, still try to patch the old-style acreate method if it exists
    if hasattr(messages_class, "acreate"):
        original_acreate = messages_class.acreate
        if not hasattr(original_acreate, "_osmosis_wrapped"):
            @functools.wraps(original_acreate)
            async def wrapped_acreate(self, *args, **kwargs):
                logger.debug(f"Wrapped async create called with args: {args}, kwargs: {kwargs}")
                try:
                    # Check if the async call includes tool use parameters
                    has_tools = "tools" in kwargs
                    if has_tools:
                        logger.debug(f"Async tool use detected with {len(kwargs['tools'])} tools")
                    
                    if "messages" in kwargs:
                        for message in kwargs["messages"]:
                            if message.get("role") == "user" and isinstance(message.get("content"), list):
                                for content_item in message["content"]:
                                    if isinstance(content_item, dict) and content_item.get("type") == "tool_response":
                                        logger.debug("Async tool response detected in messages")
                                        break
                    
                    response = await original_acreate(self, *args, **kwargs)
                    
                    if utils.enabled:
                        logger.debug("Sending async response to Hoover (success)")
                        send_to_hoover(
                            query=kwargs,
                            response=response.model_dump() if hasattr(response, 'model_dump') else response,
                            status=200
                        )
                        
                    return response
                except Exception as e:
                    logger.error(f"Error in wrapped async create: {e}")
                    if utils.enabled:
                        logger.debug("Sending async error to Hoover")
                        error_response = {"error": str(e)}
                        send_to_hoover(
                            query=kwargs,
                            response=error_response,
                            status=400
                        )
                    raise  # Re-raise the exception
            
            wrapped_acreate._osmosis_wrapped = True
            messages_class.acreate = wrapped_acreate
            logger.info("Successfully wrapped Messages.acreate method")
    else:
        logger.info("No async acreate method found in Messages class")

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
                        logger.debug(f"Wrapped Completions async create called with args: {args}, kwargs: {kwargs}")
                        try:
                            response = await original_completions_acreate(self, *args, **kwargs)
                            
                            if utils.enabled:
                                logger.debug("Sending Completions async response to Hoover (success)")
                                send_to_hoover(
                                    query=kwargs,
                                    response=response.model_dump() if hasattr(response, 'model_dump') else response,
                                    status=200
                                )
                                
                            return response
                        except Exception as e:
                            logger.error(f"Error in wrapped Completions async create: {e}")
                            if utils.enabled:
                                logger.debug("Sending Completions async error to Hoover")
                                error_response = {"error": str(e)}
                                send_to_hoover(
                                    query=kwargs,
                                    response=error_response,
                                    status=400
                                )
                            raise  # Re-raise the exception
                    
                    wrapped_completions_acreate._osmosis_wrapped = True
                    completions_class.acreate = wrapped_completions_acreate
                    logger.info("Successfully wrapped Completions.acreate method")
                else:
                    logger.info("Completions.acreate already wrapped")
            else:
                logger.info("No async acreate method found in Completions class")
        else:
            logger.info("Completions.create already wrapped")
    except (ImportError, AttributeError) as e:
        # Completions module may not exist in this version
        logger.warning(f"Completions module not found or has an unexpected structure: {e}")

    logger.info("Anthropic client has been wrapped by osmosis-wrap.") 