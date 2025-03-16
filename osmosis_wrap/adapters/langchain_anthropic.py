"""
Langchain-Anthropic adapter for Osmosis Wrap

This module provides monkey patching for the langchain-anthropic package.
"""

import functools
import sys

from osmosis_wrap import utils
from osmosis_wrap.utils import send_to_hoover
from osmosis_wrap.logger import logger

def wrap_langchain_anthropic() -> None:
    """
    Monkey patch langchain-anthropic's models to send all prompts and responses to Hoover.
    
    This function should be called before using any langchain-anthropic models.
    """
    try:
        import langchain_anthropic
    except ImportError:
        logger.error("langchain-anthropic package is not installed.")
        return
    
    _patch_anthropic_chat_models()
    
    logger.info("langchain-anthropic has been wrapped by osmosis-wrap.")

def _patch_anthropic_chat_models() -> None:
    """Patch langchain-anthropic chat model classes to send data to Hoover."""
    try:
        # Try to import ChatAnthropic class
        try:
            from langchain_anthropic import ChatAnthropic
            logger.info("Successfully imported ChatAnthropic from langchain_anthropic")
        except ImportError:
            # Handle older versions if needed
            try:
                from langchain.chat_models.anthropic import ChatAnthropic
                logger.info("Found ChatAnthropic in langchain.chat_models.anthropic")
            except ImportError:
                logger.warning("Could not find ChatAnthropic class in any expected location.")
                return
        
        # Patch the _generate method if it exists
        if hasattr(ChatAnthropic, "_generate"):
            original_generate = ChatAnthropic._generate
            
            if not hasattr(original_generate, "_osmosis_wrapped"):
                @functools.wraps(original_generate)
                def wrapped_generate(self, messages, stop=None, run_manager=None, **kwargs):
                    # Get the response
                    response = original_generate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
                    
                    # Send to Hoover if enabled
                    if utils.enabled:
                        # Create payload
                        payload = {
                            "model_type": "ChatAnthropic",
                            "model_name": self.model,
                            "messages": [str(msg) for msg in messages],  # Convert to strings for serialization
                            "response": str(response),  # Convert to string since it may not be serializable
                            "kwargs": {"stop": stop, **kwargs}
                        }
                        
                        send_to_hoover(
                            query={"type": "langchain_anthropic_generate", "messages": [str(msg) for msg in messages], "model": self.model},
                            response=payload,
                            status=200
                        )
                    
                    return response
                
                wrapped_generate._osmosis_wrapped = True
                ChatAnthropic._generate = wrapped_generate
            else:
                logger.info("ChatAnthropic._generate already wrapped.")
        
        # Patch the _agenerate method if it exists
        if hasattr(ChatAnthropic, "_agenerate"):
            original_agenerate = ChatAnthropic._agenerate
            
            if not hasattr(original_agenerate, "_osmosis_wrapped"):
                @functools.wraps(original_agenerate)
                async def wrapped_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                    # Get the response
                    response = await original_agenerate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
                    
                    # Send to Hoover if enabled
                    if utils.enabled:
                        # Create payload
                        payload = {
                            "model_type": "ChatAnthropic",
                            "model_name": self.model,
                            "messages": [str(msg) for msg in messages],  # Convert to strings for serialization
                            "response": str(response),  # Convert to string since it may not be serializable
                            "kwargs": {"stop": stop, **kwargs}
                        }
                        
                        send_to_hoover(
                            query={"type": "langchain_anthropic_agenerate", "messages": [str(msg) for msg in messages], "model": self.model},
                            response=payload,
                            status=200
                        )
                    
                    return response
                
                wrapped_agenerate._osmosis_wrapped = True
                ChatAnthropic._agenerate = wrapped_agenerate
            else:
                logger.info("ChatAnthropic._agenerate already wrapped.")
        
        # Patch _call method if it exists (used in newer versions)
        if hasattr(ChatAnthropic, "_call"):
            original_call = ChatAnthropic._call
            
            if not hasattr(original_call, "_osmosis_wrapped"):
                @functools.wraps(original_call)
                def wrapped_call(self, messages, stop=None, run_manager=None, **kwargs):
                    # Get the response
                    response = original_call(self, messages, stop=stop, run_manager=run_manager, **kwargs)
                    
                    # Send to Hoover if enabled
                    if utils.enabled:
                        # Create payload
                        payload = {
                            "model_type": "ChatAnthropic",
                            "model_name": self.model,
                            "messages": [str(msg) for msg in messages],  # Convert to strings for serialization
                            "response": str(response),
                            "kwargs": {"stop": stop, **kwargs}
                        }
                        
                        send_to_hoover(
                            query={"type": "langchain_anthropic_call", "messages": [str(msg) for msg in messages], "model": self.model},
                            response=payload,
                            status=200
                        )
                    
                    return response
                
                wrapped_call._osmosis_wrapped = True
                ChatAnthropic._call = wrapped_call
            else:
                logger.info("ChatAnthropic._call already wrapped.")
        
        # Patch _acall method if it exists
        if hasattr(ChatAnthropic, "_acall"):
            original_acall = ChatAnthropic._acall
            
            if not hasattr(original_acall, "_osmosis_wrapped"):
                @functools.wraps(original_acall)
                async def wrapped_acall(self, messages, stop=None, run_manager=None, **kwargs):
                    # Get the response
                    response = await original_acall(self, messages, stop=stop, run_manager=run_manager, **kwargs)
                    
                    # Send to Hoover if enabled
                    if utils.enabled:
                        # Create payload
                        payload = {
                            "model_type": "ChatAnthropic",
                            "model_name": self.model,
                            "messages": [str(msg) for msg in messages],  # Convert to strings for serialization
                            "response": str(response),
                            "kwargs": {"stop": stop, **kwargs}
                        }
                        
                        send_to_hoover(
                            query={"type": "langchain_anthropic_acall", "messages": [str(msg) for msg in messages], "model": self.model},
                            response=payload,
                            status=200
                        )
                    
                    return response
                
                wrapped_acall._osmosis_wrapped = True
                ChatAnthropic._acall = wrapped_acall
            else:
                logger.info("ChatAnthropic._acall already wrapped.")
                
    except Exception as e:
        logger.error(f"Failed to patch langchain-anthropic chat model classes: {e}") 