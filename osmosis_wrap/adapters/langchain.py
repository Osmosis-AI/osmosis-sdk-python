"""
Langchain adapter for Osmosis Wrap

This module provides monkey patching for the LangChain Python library.
"""

import functools
import sys

from osmosis_wrap import utils
from osmosis_wrap.utils import send_to_hoover

def wrap_langchain() -> None:
    """
    Monkey patch LangChain's components to send all prompts and responses to Hoover.
    
    This function should be called before using any LangChain components.
    """
    try:
        import langchain
    except ImportError:
        print("Error: langchain package is not installed.", file=sys.stderr)
        return
    
    # Patch LLM classes
    _patch_langchain_llms()
    
    # Patch Chat model classes
    _patch_langchain_chat_models()
    
    # Patch prompt templates
    _patch_langchain_prompts()

def _patch_langchain_llms() -> None:
    """Patch LangChain LLM classes to send requests and responses to Hoover."""
    try:
        # Try to import BaseLLM from different possible locations in LangChain
        try:
            from langchain_core.language_models.llms import LLM as BaseLLM
            print(f"Found BaseLLM in langchain_core.language_models.llms", file=sys.stderr)
        except ImportError:
            try:
                from langchain.llms.base import BaseLLM
                print(f"Found BaseLLM in langchain.llms.base", file=sys.stderr)
            except ImportError:
                try:
                    from langchain.llms import BaseLLM
                    print(f"Found BaseLLM in langchain.llms", file=sys.stderr)
                except ImportError:
                    try:
                        from langchain_core.language_models import BaseLLM
                        print(f"Found BaseLLM in langchain_core.language_models", file=sys.stderr)
                    except ImportError:
                        print("Warning: Could not find LangChain BaseLLM class.", file=sys.stderr)
                        return
        
        print(f"BaseLLM methods: {dir(BaseLLM)}", file=sys.stderr)
        
        # In Pydantic v2, we need to patch class methods, not instance methods
        
        # Patch the invoke method at the class level
        if hasattr(BaseLLM, "invoke"):
            # Save original class method
            original_invoke = BaseLLM.invoke
            
            # Check if already wrapped
            if hasattr(original_invoke, "_osmosis_wrapped"):
                print("BaseLLM.invoke already wrapped", file=sys.stderr)
            else:
                # Create wrapper function
                @functools.wraps(original_invoke)
                def wrapped_invoke(self, prompt, *args, **kwargs):
                    """Wrapped invoke method to send data to Hoover"""
                    # Get the model name if available
                    model_name = getattr(self, "model_name", None) or self.__class__.__name__
                    
                    print(f"LangChain LLM invoke called with prompt: {prompt}", file=sys.stderr)
                    
                    # Create query data
                    query_data = {
                        "provider": "langchain",
                        "component": "llm",
                        "model": model_name,
                        "messages": prompt,
                        "parameters": kwargs
                    }
                    
                    # Call the original method
                    try:
                        response = original_invoke(self, prompt, *args, **kwargs)
                        
                        print(f"LangChain LLM invoke response: {response}", file=sys.stderr)
                        
                        # Create response data
                        response_data = {
                            "completion": str(response),
                            "model": model_name,
                            "provider": "langchain"
                        }
                        
                        # Send to Hoover
                        send_to_hoover(query_data, response_data)
                        
                        # Optional print to console
                        if utils.print_messages and utils.enabled:
                            output = sys.stderr if utils.use_stderr else sys.stdout
                            print(f"\n--- LangChain LLM Invoke ({model_name}) ---", file=output)
                            print(f"Prompt: {prompt}", file=output)
                            print(f"\n--- Response ---", file=output)
                            print(f"{response}", file=output)
                            print("-------------------------------\n", file=output)
                        
                        return response
                    except Exception as e:
                        print(f"Error in LangChain LLM invoke: {e}", file=sys.stderr)
                        # Log errors too
                        response_data = {
                            "error": str(e),
                            "model": model_name,
                            "provider": "langchain"
                        }
                        send_to_hoover(query_data, response_data, status=500)
                        raise
                
                # Mark the function
                wrapped_invoke._osmosis_wrapped = True
                
                # Replace method in the class
                try:
                    # For property descriptors or decorated methods
                    if isinstance(original_invoke, property):
                        print("BaseLLM.invoke is a property, handling specially", file=sys.stderr)
                        # Get original property getter
                        original_getter = original_invoke.fget
                        
                        # Create new property getter that wraps the result function
                        def new_getter(self):
                            func = original_getter(self)
                            
                            # Only wrap if it's not already wrapped
                            if not hasattr(func, "_osmosis_wrapped"):
                                @functools.wraps(func)
                                def wrapped_func(*args, **kwargs):
                                    # Get the model name if available
                                    model_name = getattr(self, "model_name", None) or self.__class__.__name__
                                    
                                    print(f"LangChain LLM invoke property called", file=sys.stderr)
                                    
                                    # Create query data
                                    query_data = {
                                        "provider": "langchain",
                                        "component": "llm",
                                        "model": model_name,
                                        "messages": args[0] if args else kwargs.get("prompt", ""),
                                        "parameters": kwargs
                                    }
                                    
                                    # Call the original function
                                    try:
                                        response = func(*args, **kwargs)
                                        
                                        # Create response data
                                        response_data = {
                                            "completion": str(response),
                                            "model": model_name,
                                            "provider": "langchain"
                                        }
                                        
                                        # Send to Hoover
                                        send_to_hoover(query_data, response_data)
                                        
                                        return response
                                    except Exception as e:
                                        print(f"Error in LangChain LLM invoke property: {e}", file=sys.stderr)
                                        # Log errors too
                                        response_data = {
                                            "error": str(e),
                                            "model": model_name,
                                            "provider": "langchain"
                                        }
                                        send_to_hoover(query_data, response_data, status=500)
                                        raise
                                
                                # Mark as wrapped
                                wrapped_func._osmosis_wrapped = True
                                return wrapped_func
                            
                            return func
                        
                        # Create new property with our getter
                        new_property = property(new_getter, original_invoke.fset, original_invoke.fdel)
                        setattr(BaseLLM, "invoke", new_property)
                        print("Successfully replaced BaseLLM.invoke property", file=sys.stderr)
                    else:
                        # Regular method replacement
                        setattr(BaseLLM, "invoke", wrapped_invoke)
                        print("Successfully replaced BaseLLM.invoke method", file=sys.stderr)
                except Exception as e:
                    print(f"Error replacing BaseLLM.invoke: {e}", file=sys.stderr)
                    
        # Also patch __call__ method
        if hasattr(BaseLLM, "__call__") and callable(getattr(BaseLLM, "__call__")):
            original_call = BaseLLM.__call__
            
            if not hasattr(original_call, "_osmosis_wrapped"):
                @functools.wraps(original_call)
                def wrapped_call(self, prompt, *args, **kwargs):
                    """Wrapped __call__ method to send data to Hoover"""
                    # Get the model name if available
                    model_name = getattr(self, "model_name", None) or self.__class__.__name__
                    
                    print(f"LangChain LLM __call__ with prompt: {prompt}", file=sys.stderr)
                    
                    # Create query data
                    query_data = {
                        "provider": "langchain",
                        "component": "llm",
                        "model": model_name,
                        "messages": prompt,
                        "parameters": kwargs
                    }
                    
                    # Call the original method
                    try:
                        response = original_call(self, prompt, *args, **kwargs)
                        
                        # Create response data
                        response_data = {
                            "completion": str(response),
                            "model": model_name,
                            "provider": "langchain"
                        }
                        
                        # Send to Hoover
                        send_to_hoover(query_data, response_data)
                        
                        return response
                    except Exception as e:
                        print(f"Error in LangChain LLM __call__: {e}", file=sys.stderr)
                        # Log errors too
                        response_data = {
                            "error": str(e),
                            "model": model_name,
                            "provider": "langchain"
                        }
                        send_to_hoover(query_data, response_data, status=500)
                        raise
                
                # Mark wrapped
                wrapped_call._osmosis_wrapped = True
                
                # Replace class method
                try:
                    BaseLLM.__call__ = wrapped_call
                    print("Successfully replaced BaseLLM.__call__", file=sys.stderr)
                except Exception as e:
                    print(f"Error replacing BaseLLM.__call__: {e}", file=sys.stderr)
        
        # Patch descriptor methods if needed
        # For Pydantic, we need to handle subclass methods by monkey patching the base class itself
        # This is needed because Pydantic models handle method resolution differently
        
        # Patch _generate if it exists for handling other methods
        if hasattr(BaseLLM, "_call") and callable(getattr(BaseLLM, "_call")):
            original_call_method = BaseLLM._call
            
            if not hasattr(original_call_method, "_osmosis_wrapped"):
                @functools.wraps(original_call_method)
                def wrapped_call_method(self, prompt, stop=None, run_manager=None, **kwargs):
                    """Wrapped _call method to send data to Hoover"""
                    # Get the model name if available
                    model_name = getattr(self, "model_name", None) or self.__class__.__name__
                    
                    print(f"LangChain LLM _call with prompt: {prompt}", file=sys.stderr)
                    
                    # Create query data
                    query_data = {
                        "provider": "langchain",
                        "component": "llm",
                        "model": model_name,
                        "messages": prompt,
                        "parameters": kwargs
                    }
                    
                    # Call the original method
                    try:
                        response = original_call_method(self, prompt, stop=stop, run_manager=run_manager, **kwargs)
                        
                        # Create response data
                        response_data = {
                            "completion": str(response),
                            "model": model_name,
                            "provider": "langchain"
                        }
                        
                        # Send to Hoover
                        send_to_hoover(query_data, response_data)
                        
                        return response
                    except Exception as e:
                        print(f"Error in LangChain LLM _call: {e}", file=sys.stderr)
                        # Log errors too
                        response_data = {
                            "error": str(e),
                            "model": model_name,
                            "provider": "langchain"
                        }
                        send_to_hoover(query_data, response_data, status=500)
                        raise
                
                # Mark wrapped
                wrapped_call_method._osmosis_wrapped = True
                
                # Replace class method
                try:
                    BaseLLM._call = wrapped_call_method
                    print("Successfully replaced BaseLLM._call", file=sys.stderr)
                except Exception as e:
                    print(f"Error replacing BaseLLM._call: {e}", file=sys.stderr)
                    
    except Exception as e:
        print(f"Warning: Could not patch LangChain LLMs: {str(e)}", file=sys.stderr)

def _patch_langchain_chat_models() -> None:
    """Patch LangChain Chat model classes to send requests and responses to Hoover."""
    try:
        # Try different import paths for BaseChatModel
        try:
            from langchain.chat_models.base import BaseChatModel
        except ImportError:
            try:
                from langchain_core.language_models.chat_models import BaseChatModel
            except ImportError:
                try:
                    from langchain.chat_models import BaseChatModel
                except ImportError:
                    print("Warning: Could not find LangChain BaseChatModel", file=sys.stderr)
                    return

        # Check which methods exist
        has_generate = hasattr(BaseChatModel, "_generate")
        has_invoke = hasattr(BaseChatModel, "invoke")
        
        if has_generate:
            # Store original _generate method
            original_generate = BaseChatModel._generate
            
            @functools.wraps(original_generate)
            def wrapped_generate(self, messages, stop=None, **kwargs):
                """Wrapped _generate method to send data to Hoover"""
                # Get the model name if available
                model_name = getattr(self, "model_name", None) or self.__class__.__name__
                
                # Create query data
                formatted_messages = []
                for message in messages:
                    # Handle different message schemas
                    content = getattr(message, "content", None)
                    message_type = getattr(message, "type", None)
                    
                    if message_type is None:
                        # Try to infer type from class name
                        message_type = message.__class__.__name__.lower().replace("message", "")
                    
                    formatted_messages.append({
                        "role": message_type,
                        "content": content
                    })
                
                query_data = {
                    "provider": "langchain",
                    "model": model_name,
                    "messages": formatted_messages,
                    "stop": stop,
                    "parameters": kwargs
                }
                
                # Call the original method
                try:
                    result = original_generate(self, messages, stop=stop, **kwargs)
                    
                    # Format response - handle different result schemas
                    response_content = []
                    
                    # Try to extract the message from the generations
                    try:
                        for generation in result.generations:
                            if isinstance(generation, list):
                                message = generation[0].message
                            else:
                                message = generation.message
                                
                            # Get message content and type
                            content = getattr(message, "content", None)
                            message_type = getattr(message, "type", None)
                            
                            if message_type is None:
                                # Try to infer type from class name
                                message_type = message.__class__.__name__.lower().replace("message", "")
                                
                            response_content.append({
                                "role": message_type,
                                "content": content
                            })
                    except (AttributeError, IndexError) as e:
                        # Fallback for different response structures
                        response_content = [{
                            "role": "ai",
                            "content": str(result)
                        }]
                    
                    # Create response data
                    response_data = {
                        "completion": response_content,
                        "model": model_name,
                        "provider": "langchain"
                    }
                    
                    # Send to Hoover
                    send_to_hoover(query_data, response_data)
                    
                    # Optional print to console
                    if utils.print_messages and utils.enabled:
                        output = sys.stderr if utils.use_stderr else sys.stdout
                        print(f"\n--- LangChain Chat Request ({model_name}) ---", file=output)
                        for i, msg in enumerate(formatted_messages):
                            print(f"[{msg['role']}]: {msg['content']}", file=output)
                        print(f"\n--- Response ---", file=output)
                        for resp in response_content:
                            print(f"[{resp['role']}]: {resp['content']}", file=output)
                        print("-------------------------------\n", file=output)
                    
                    return result
                except Exception as e:
                    # Log errors too
                    response_data = {
                        "error": str(e),
                        "model": model_name,
                        "provider": "langchain"
                    }
                    send_to_hoover(query_data, response_data, status=500)
                    raise
            
            # Replace the method
            BaseChatModel._generate = wrapped_generate
            
            # Patch async method if it exists
            if hasattr(BaseChatModel, "_agenerate"):
                original_agenerate = BaseChatModel._agenerate
                
                @functools.wraps(original_agenerate)
                async def wrapped_agenerate(self, messages, stop=None, **kwargs):
                    """Wrapped _agenerate method to send data to Hoover"""
                    # Get the model name if available
                    model_name = getattr(self, "model_name", None) or self.__class__.__name__
                    
                    # Create query data
                    formatted_messages = []
                    for message in messages:
                        # Handle different message schemas
                        content = getattr(message, "content", None)
                        message_type = getattr(message, "type", None)
                        
                        if message_type is None:
                            # Try to infer type from class name
                            message_type = message.__class__.__name__.lower().replace("message", "")
                        
                        formatted_messages.append({
                            "role": message_type,
                            "content": content
                        })
                    
                    query_data = {
                        "provider": "langchain",
                        "model": model_name,
                        "messages": formatted_messages,
                        "stop": stop,
                        "parameters": kwargs
                    }
                    
                    # Call the original method
                    try:
                        result = await original_agenerate(self, messages, stop=stop, **kwargs)
                        
                        # Format response - handle different result schemas
                        response_content = []
                        
                        # Try to extract the message from the generations
                        try:
                            for generation in result.generations:
                                if isinstance(generation, list):
                                    message = generation[0].message
                                else:
                                    message = generation.message
                                    
                                # Get message content and type
                                content = getattr(message, "content", None)
                                message_type = getattr(message, "type", None)
                                
                                if message_type is None:
                                    # Try to infer type from class name
                                    message_type = message.__class__.__name__.lower().replace("message", "")
                                    
                                response_content.append({
                                    "role": message_type,
                                    "content": content
                                })
                        except (AttributeError, IndexError) as e:
                            # Fallback for different response structures
                            response_content = [{
                                "role": "ai",
                                "content": str(result)
                            }]
                        
                        # Create response data
                        response_data = {
                            "completion": response_content,
                            "model": model_name,
                            "provider": "langchain"
                        }
                        
                        # Send to Hoover
                        send_to_hoover(query_data, response_data)
                        
                        # Optional print to console
                        if utils.print_messages and utils.enabled:
                            output = sys.stderr if utils.use_stderr else sys.stdout
                            print(f"\n--- LangChain Chat Async Request ({model_name}) ---", file=output)
                            for i, msg in enumerate(formatted_messages):
                                print(f"[{msg['role']}]: {msg['content']}", file=output)
                            print(f"\n--- Response ---", file=output)
                            for resp in response_content:
                                print(f"[{resp['role']}]: {resp['content']}", file=output)
                            print("-------------------------------\n", file=output)
                        
                        return result
                    except Exception as e:
                        # Log errors too
                        response_data = {
                            "error": str(e),
                            "model": model_name,
                            "provider": "langchain"
                        }
                        send_to_hoover(query_data, response_data, status=500)
                        raise
                
                # Replace the method
                BaseChatModel._agenerate = wrapped_agenerate
        
        # For newer LangChain versions with invoke
        if has_invoke:
            original_invoke = BaseChatModel.invoke
            
            @functools.wraps(original_invoke)
            def wrapped_invoke(self, messages, *args, **kwargs):
                """Wrapped invoke method to send data to Hoover"""
                # Get the model name if available
                model_name = getattr(self, "model_name", None) or self.__class__.__name__
                
                # Format messages for logging
                formatted_messages = []
                if isinstance(messages, list):
                    for message in messages:
                        content = getattr(message, "content", None)
                        message_type = getattr(message, "type", None)
                        
                        if message_type is None:
                            # Try to infer type from class name
                            message_type = message.__class__.__name__.lower().replace("message", "")
                            
                        formatted_messages.append({
                            "role": message_type,
                            "content": content
                        })
                else:
                    # Single message
                    content = getattr(messages, "content", str(messages))
                    formatted_messages.append({
                        "role": "user",
                        "content": content
                    })
                
                query_data = {
                    "provider": "langchain",
                    "model": model_name,
                    "messages": formatted_messages,
                    "parameters": kwargs
                }
                
                # Call the original method
                try:
                    result = original_invoke(self, messages, *args, **kwargs)
                    
                    # Format response - newer LangChain versions may return different formats
                    response_content = []
                    
                    # Handle different result types
                    if hasattr(result, "content"):
                        # Message-like object
                        content = result.content
                        message_type = getattr(result, "type", "ai")
                        
                        response_content.append({
                            "role": message_type,
                            "content": content
                        })
                    else:
                        # Default fallback
                        response_content.append({
                            "role": "ai",
                            "content": str(result)
                        })
                    
                    # Create response data
                    response_data = {
                        "completion": response_content,
                        "model": model_name,
                        "provider": "langchain"
                    }
                    
                    # Send to Hoover
                    send_to_hoover(query_data, response_data)
                    
                    # Optional print to console
                    if utils.print_messages and utils.enabled:
                        output = sys.stderr if utils.use_stderr else sys.stdout
                        print(f"\n--- LangChain Chat Invoke ({model_name}) ---", file=output)
                        for i, msg in enumerate(formatted_messages):
                            print(f"[{msg['role']}]: {msg['content']}", file=output)
                        print(f"\n--- Response ---", file=output)
                        for resp in response_content:
                            print(f"[{resp['role']}]: {resp['content']}", file=output)
                        print("-------------------------------\n", file=output)
                    
                    return result
                except Exception as e:
                    # Log errors too
                    response_data = {
                        "error": str(e),
                        "model": model_name,
                        "provider": "langchain"
                    }
                    send_to_hoover(query_data, response_data, status=500)
                    raise
            
            # Replace the method
            BaseChatModel.invoke = wrapped_invoke
            
            # Patch async invoke if it exists
            if hasattr(BaseChatModel, "ainvoke"):
                original_ainvoke = BaseChatModel.ainvoke
                
                @functools.wraps(original_ainvoke)
                async def wrapped_ainvoke(self, messages, *args, **kwargs):
                    """Wrapped ainvoke method to send data to Hoover"""
                    # Get the model name if available
                    model_name = getattr(self, "model_name", None) or self.__class__.__name__
                    
                    # Format messages for logging
                    formatted_messages = []
                    if isinstance(messages, list):
                        for message in messages:
                            content = getattr(message, "content", None)
                            message_type = getattr(message, "type", None)
                            
                            if message_type is None:
                                # Try to infer type from class name
                                message_type = message.__class__.__name__.lower().replace("message", "")
                                
                            formatted_messages.append({
                                "role": message_type,
                                "content": content
                            })
                    else:
                        # Single message
                        content = getattr(messages, "content", str(messages))
                        formatted_messages.append({
                            "role": "user",
                            "content": content
                        })
                    
                    query_data = {
                        "provider": "langchain",
                        "model": model_name,
                        "messages": formatted_messages,
                        "parameters": kwargs
                    }
                    
                    # Call the original method
                    try:
                        result = await original_ainvoke(self, messages, *args, **kwargs)
                        
                        # Format response - newer LangChain versions may return different formats
                        response_content = []
                        
                        # Handle different result types
                        if hasattr(result, "content"):
                            # Message-like object
                            content = result.content
                            message_type = getattr(result, "type", "ai")
                            
                            response_content.append({
                                "role": message_type,
                                "content": content
                            })
                        else:
                            # Default fallback
                            response_content.append({
                                "role": "ai",
                                "content": str(result)
                            })
                        
                        # Create response data
                        response_data = {
                            "completion": response_content,
                            "model": model_name,
                            "provider": "langchain"
                        }
                        
                        # Send to Hoover
                        send_to_hoover(query_data, response_data)
                        
                        # Optional print to console
                        if utils.print_messages and utils.enabled:
                            output = sys.stderr if utils.use_stderr else sys.stdout
                            print(f"\n--- LangChain Chat Async Invoke ({model_name}) ---", file=output)
                            for i, msg in enumerate(formatted_messages):
                                print(f"[{msg['role']}]: {msg['content']}", file=output)
                            print(f"\n--- Response ---", file=output)
                            for resp in response_content:
                                print(f"[{resp['role']}]: {resp['content']}", file=output)
                            print("-------------------------------\n", file=output)
                        
                        return result
                    except Exception as e:
                        # Log errors too
                        response_data = {
                            "error": str(e),
                            "model": model_name,
                            "provider": "langchain"
                        }
                        send_to_hoover(query_data, response_data, status=500)
                        raise
                
                # Replace the method
                BaseChatModel.ainvoke = wrapped_ainvoke
    
    except Exception as e:
        print(f"Warning: Could not patch LangChain Chat Models: {str(e)}", file=sys.stderr)

def _patch_langchain_prompts() -> None:
    """Patch LangChain prompt templates to capture template usage."""
    try:
        # Try different import paths
        try:
            from langchain_core.prompts import BasePromptTemplate
            print("Found BasePromptTemplate in langchain_core.prompts")
        except ImportError:
            try:
                from langchain.prompts.base import BasePromptTemplate
                print("Found BasePromptTemplate in langchain.prompts.base")
            except ImportError:
                try:
                    from langchain.prompts import BasePromptTemplate
                    print("Found BasePromptTemplate in langchain.prompts")
                except ImportError:
                    print("Warning: Could not find LangChain BasePromptTemplate", file=sys.stderr)
                    return
        
        # Check if format method exists
        if not hasattr(BasePromptTemplate, "format"):
            print("Warning: LangChain BasePromptTemplate does not have format method", file=sys.stderr)
            return
        
        # Store original format method
        original_format = BasePromptTemplate.format
        print(f"Original format method: {original_format}")
        
        # Check if it's already wrapped
        if hasattr(original_format, "_osmosis_wrapped"):
            print("Format method already wrapped")
            return
        
        @functools.wraps(original_format)
        def wrapped_format(self, **kwargs):
            """Wrapped format method to log prompt template usage"""
            # Get template name
            template_name = getattr(self, "template_name", None) or self.__class__.__name__
            
            # Call the original method
            try:
                print(f"Calling original format method: {original_format}")
                result = original_format(self, **kwargs)
                
                # Create query data (simplified for prompt templates)
                query_data = {
                    "provider": "langchain",
                    "component": "prompt_template",
                    "template_name": template_name,
                    "template": getattr(self, "template", ""),
                    "input_variables": getattr(self, "input_variables", []),
                    "parameters": kwargs
                }
                
                # Create response data
                response_data = {
                    "formatted_prompt": result
                }
                
                # Send to Hoover (with a low priority as this is just template usage)
                print(f"Sending to Hoover: {query_data}, {response_data}")
                send_to_hoover(query_data, response_data)
                
                return result
            except Exception as e:
                # Just continue if there's an error in our monitoring code
                # We don't want to break the actual functionality
                print(f"Warning: Error monitoring LangChain prompt template: {str(e)}", file=sys.stderr)
                return original_format(self, **kwargs)
        
        # Mark our function as wrapped to avoid double-wrapping
        wrapped_format._osmosis_wrapped = True
        
        # For LangChain using Pydantic v2, we need a different approach
        try:
            # Try to access the pydantic Field descriptor
            if isinstance(original_format, property):
                print("Format method is a property, using property-based patching")
                # It's a property, patch it differently
                original_getter = original_format.fget
                
                @property
                @functools.wraps(original_getter)
                def wrapped_format_property(self):
                    orig_method = original_getter(self)
                    
                    @functools.wraps(orig_method)
                    def wrapped_inner_format(**kwargs):
                        # Get template name
                        template_name = getattr(self, "template_name", None) or self.__class__.__name__
                        
                        # Call the original method
                        try:
                            result = orig_method(**kwargs)
                            
                            # Create query data (simplified for prompt templates)
                            query_data = {
                                "provider": "langchain",
                                "component": "prompt_template",
                                "template_name": template_name,
                                "template": getattr(self, "template", ""),
                                "input_variables": getattr(self, "input_variables", []),
                                "parameters": kwargs
                            }
                            
                            # Create response data
                            response_data = {
                                "formatted_prompt": result
                            }
                            
                            # Send to Hoover
                            print(f"Sending to Hoover (property): {query_data}, {response_data}")
                            send_to_hoover(query_data, response_data)
                            
                            return result
                        except Exception as e:
                            # Just continue if there's an error in our monitoring code
                            # We don't want to break the actual functionality
                            print(f"Warning: Error monitoring LangChain prompt template: {str(e)}", file=sys.stderr)
                            return orig_method(**kwargs)
                    
                    # Mark as wrapped
                    wrapped_inner_format._osmosis_wrapped = True
                    
                    return wrapped_inner_format
                
                # Replace the property
                wrapped_format_property._osmosis_wrapped = True
                BasePromptTemplate.format = wrapped_format_property
                print(f"Replaced format property: {BasePromptTemplate.format}")
            else:
                # It's a regular method
                print(f"Replacing standard format method: {original_format}")
                BasePromptTemplate.format = wrapped_format
                print(f"New format method: {BasePromptTemplate.format}")
        except Exception as e:
            print(f"Error while patching format method: {str(e)}", file=sys.stderr)
        
        # Patch the StringPromptTemplate class separately
        try:
            try:
                from langchain_core.prompts.string import StringPromptTemplate
            except ImportError:
                try:
                    from langchain.prompts.string import StringPromptTemplate
                except ImportError:
                    StringPromptTemplate = None
            
            if StringPromptTemplate and StringPromptTemplate != BasePromptTemplate:
                print(f"Found StringPromptTemplate: {StringPromptTemplate}")
                if hasattr(StringPromptTemplate, "format") and not hasattr(StringPromptTemplate.format, "_osmosis_wrapped"):
                    string_original_format = StringPromptTemplate.format
                    
                    @functools.wraps(string_original_format)
                    def wrapped_string_format(self, **kwargs):
                        """Wrapped format method for StringPromptTemplate"""
                        # Get template name
                        template_name = getattr(self, "template_name", None) or self.__class__.__name__
                        
                        # Call the original method
                        try:
                            result = string_original_format(self, **kwargs)
                            
                            # Create query data
                            query_data = {
                                "provider": "langchain",
                                "component": "string_prompt_template",
                                "template_name": template_name,
                                "template": getattr(self, "template", ""),
                                "input_variables": getattr(self, "input_variables", []),
                                "parameters": kwargs
                            }
                            
                            # Create response data
                            response_data = {
                                "formatted_prompt": result
                            }
                            
                            # Send to Hoover
                            print(f"Sending to Hoover (StringPromptTemplate): {query_data}, {response_data}")
                            send_to_hoover(query_data, response_data)
                            
                            return result
                        except Exception as e:
                            print(f"Warning: Error monitoring StringPromptTemplate: {str(e)}", file=sys.stderr)
                            return string_original_format(self, **kwargs)
                    
                    # Mark as wrapped
                    wrapped_string_format._osmosis_wrapped = True
                    StringPromptTemplate.format = wrapped_string_format
                    print(f"Replaced StringPromptTemplate.format: {StringPromptTemplate.format}")
        except Exception as e:
            print(f"Error while patching StringPromptTemplate: {str(e)}", file=sys.stderr)
        
        # Finally, patch the PromptTemplate directly
        try:
            try:
                from langchain_core.prompts import PromptTemplate
            except ImportError:
                try:
                    from langchain.prompts import PromptTemplate
                except ImportError:
                    PromptTemplate = None
            
            if PromptTemplate and hasattr(PromptTemplate, "format") and not hasattr(PromptTemplate.format, "_osmosis_wrapped"):
                print(f"Found PromptTemplate: {PromptTemplate}")
                prompt_original_format = PromptTemplate.format
                
                # Check if it's a different instance than what we've already wrapped
                if (not hasattr(prompt_original_format, "_osmosis_wrapped") and 
                    prompt_original_format != original_format):
                    
                    @functools.wraps(prompt_original_format)
                    def wrapped_prompt_format(self, **kwargs):
                        """Wrapped format method for PromptTemplate"""
                        # Get template name
                        template_name = getattr(self, "template_name", None) or self.__class__.__name__
                        
                        # Call the original method
                        try:
                            result = prompt_original_format(self, **kwargs)
                            
                            # Create query data
                            query_data = {
                                "provider": "langchain",
                                "component": "prompt_template",
                                "template_name": template_name,
                                "template": getattr(self, "template", ""),
                                "input_variables": getattr(self, "input_variables", []),
                                "parameters": kwargs
                            }
                            
                            # Create response data
                            response_data = {
                                "formatted_prompt": result
                            }
                            
                            # Send to Hoover
                            print(f"Sending to Hoover (PromptTemplate): {query_data}, {response_data}")
                            send_to_hoover(query_data, response_data)
                            
                            return result
                        except Exception as e:
                            print(f"Warning: Error monitoring PromptTemplate: {str(e)}", file=sys.stderr)
                            return prompt_original_format(self, **kwargs)
                    
                    # Mark as wrapped
                    wrapped_prompt_format._osmosis_wrapped = True
                    
                    # Replace the method
                    PromptTemplate.format = wrapped_prompt_format
                    print(f"Replaced PromptTemplate.format: {PromptTemplate.format}")
        except Exception as e:
            print(f"Error while patching PromptTemplate: {str(e)}", file=sys.stderr)
        
    except Exception as e:
        print(f"Warning: Could not patch LangChain Prompt Templates: {str(e)}", file=sys.stderr) 