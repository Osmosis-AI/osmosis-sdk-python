"""
Langchain adapter for Osmosis

This module provides monkey patching for the LangChain Python library.
"""

import functools
import sys

from osmosis_ai import utils
from osmosis_ai.utils import send_to_osmosis
from osmosis_ai.logger import logger


def wrap_langchain() -> None:
    """
    Monkey patch LangChain's components to send all prompts and responses to OSMOSIS.

    This function should be called before using any LangChain components.
    """
    try:
        import langchain
    except ImportError:
        logger.debug("langchain package is not installed.")
        return

    # Patch LLM classes
    _patch_langchain_llms()

    # Patch Chat model classes
    _patch_langchain_chat_models()

    # Patch prompt templates
    _patch_langchain_prompts()

    logger.info("LangChain has been wrapped by osmosis-ai.")


def _patch_langchain_llms() -> None:
    """Patch LangChain LLM classes to send data to OSMOSIS."""
    try:
        # Try to import LangChain LLM base classes
        # First try langchain_core (newer versions)
        try:
            from langchain_core.language_models.llms import BaseLLM

            logger.info(
                f"Successfully imported BaseLLM from langchain_core.language_models.llms"
            )
        except ImportError:
            # Then try other possible locations (older versions)
            try:
                from langchain.llms.base import BaseLLM

                logger.info(f"Found BaseLLM in langchain.llms.base")
            except ImportError:
                try:
                    from langchain.llms import BaseLLM

                    logger.info(f"Found BaseLLM in langchain.llms")
                except ImportError:
                    try:
                        from langchain_core.language_models import BaseLLM

                        logger.info(f"Found BaseLLM in langchain_core.language_models")
                    except ImportError:
                        logger.warning(
                            "Could not find BaseLLM class in any expected location."
                        )
                        return

        logger.info("Starting to wrap LangChain LLM methods...")

        # Get all available methods to understand which API we're working with
        llm_methods = [
            method
            for method in dir(BaseLLM)
            if not method.startswith("_") or method in ["_call", "__call__"]
        ]
        logger.info(f"Found the following methods on BaseLLM: {llm_methods}")

        # Patch the _call method if it exists
        if hasattr(BaseLLM, "_call"):
            original_call = BaseLLM._call

            if not hasattr(original_call, "_osmosis_aiped"):

                @functools.wraps(original_call)
                def wrapped_call(self, prompt, *args, **kwargs):
                    # Get the response
                    response = original_call(self, prompt, *args, **kwargs)

                    # Send to OSMOSIS if enabled
                    if utils.enabled:
                        # Try to get model name
                        model_name = "unknown_model"
                        if hasattr(self, "model_name"):
                            model_name = self.model_name

                        # Create payload
                        payload = {
                            "llm_type": self.__class__.__name__,
                            "model_name": model_name,
                            "prompt": prompt,
                            "response": response,
                            "kwargs": kwargs,
                        }

                        send_to_osmosis(
                            query={
                                "type": "langchain_llm",
                                "prompt": prompt,
                                "model": model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_call._osmosis_aiped = True
                BaseLLM._call = wrapped_call
                logger.info("Successfully wrapped BaseLLM._call method")
            else:
                logger.info("LangChain BaseLLM._call already wrapped.")
        else:
            logger.info("LangChain BaseLLM does not have a _call method, skipping.")

        # Also patch invoke method if it exists
        if hasattr(BaseLLM, "invoke"):
            original_invoke = BaseLLM.invoke

            if not hasattr(original_invoke, "_osmosis_aiped"):

                @functools.wraps(original_invoke)
                def wrapped_invoke(self, prompt, *args, **kwargs):
                    # Call original
                    response = original_invoke(self, prompt, *args, **kwargs)

                    # Send to OSMOSIS if enabled
                    if utils.enabled:
                        # Try to get model name
                        model_name = "unknown_model"
                        if hasattr(self, "model_name"):
                            model_name = self.model_name

                        # Create payload
                        payload = {
                            "llm_type": self.__class__.__name__,
                            "model_name": model_name,
                            "prompt": prompt,
                            "response": response,
                            "kwargs": kwargs,
                        }

                        send_to_osmosis(
                            query={
                                "type": "langchain_llm_invoke",
                                "prompt": prompt,
                                "model": model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_invoke._osmosis_aiped = True
                BaseLLM.invoke = wrapped_invoke
                logger.info("Successfully wrapped BaseLLM.invoke method")
            else:
                logger.info("LangChain BaseLLM.invoke already wrapped.")
        else:
            logger.info("LangChain BaseLLM does not have an invoke method, skipping.")

        # Patch the generate method if it exists
        if hasattr(BaseLLM, "generate"):
            original_generate = BaseLLM.generate

            if not hasattr(original_generate, "_osmosis_aiped"):

                @functools.wraps(original_generate)
                def wrapped_generate(self, prompts, *args, **kwargs):
                    # Get the response
                    response = original_generate(self, prompts, *args, **kwargs)

                    # Send to OSMOSIS if enabled
                    if utils.enabled:
                        # Try to get model name
                        model_name = "unknown_model"
                        if hasattr(self, "model_name"):
                            model_name = self.model_name

                        # Create payload
                        payload = {
                            "llm_type": self.__class__.__name__,
                            "model_name": model_name,
                            "prompts": prompts,
                            "response": str(
                                response
                            ),  # Convert to string since it may not be serializable
                            "kwargs": kwargs,
                        }

                        send_to_osmosis(
                            query={
                                "type": "langchain_llm_generate",
                                "prompts": prompts,
                                "model": model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_generate._osmosis_aiped = True
                BaseLLM.generate = wrapped_generate
                logger.info("Successfully wrapped BaseLLM.generate method")
            else:
                logger.info("LangChain BaseLLM.generate already wrapped.")
        else:
            logger.info("LangChain BaseLLM does not have a generate method, skipping.")

        # For modern LangChain, patch __call__ which could be the Model.__call__ method
        if hasattr(BaseLLM, "__call__") and callable(getattr(BaseLLM, "__call__")):
            # Get the method, not the descriptor
            original_call_method = BaseLLM.__call__

            if not hasattr(original_call_method, "_osmosis_aiped"):

                @functools.wraps(original_call_method)
                def wrapped_call_method(
                    self, prompt, stop=None, run_manager=None, **kwargs
                ):
                    try:
                        # Get the response
                        response = original_call_method(
                            self, prompt, stop=stop, run_manager=run_manager, **kwargs
                        )

                        # Send to OSMOSIS if enabled
                        if utils.enabled:
                            # Try to get model name
                            model_name = "unknown_model"
                            if hasattr(self, "model_name"):
                                model_name = self.model_name

                            # Create payload
                            payload = {
                                "llm_type": self.__class__.__name__,
                                "model_name": model_name,
                                "prompt": prompt,
                                "response": response,
                                "kwargs": {"stop": stop, **kwargs},
                            }

                            send_to_osmosis(
                                query={
                                    "type": "langchain_llm_call",
                                    "prompt": prompt,
                                    "model": model_name,
                                },
                                response=payload,
                                status=200,
                            )

                        return response
                    except TypeError as e:
                        # Handle parameter mismatch gracefully
                        logger.warning(
                            f"TypeError in wrapped __call__: {e}, trying without run_manager"
                        )
                        # Try calling without run_manager (older versions)
                        response = original_call_method(
                            self, prompt, stop=stop, **kwargs
                        )

                        # Send to OSMOSIS if enabled
                        if utils.enabled:
                            model_name = getattr(self, "model_name", "unknown_model")
                            payload = {
                                "llm_type": self.__class__.__name__,
                                "model_name": model_name,
                                "prompt": prompt,
                                "response": response,
                                "kwargs": {"stop": stop, **kwargs},
                            }

                            send_to_osmosis(
                                query={
                                    "type": "langchain_llm_call_fallback",
                                    "prompt": prompt,
                                    "model": model_name,
                                },
                                response=payload,
                                status=200,
                            )

                        return response

                wrapped_call_method._osmosis_aiped = True
                BaseLLM.__call__ = wrapped_call_method
                logger.info("Successfully wrapped BaseLLM.__call__ method")
            else:
                logger.info("LangChain BaseLLM.__call__ already wrapped.")
        else:
            logger.info(
                "LangChain BaseLLM does not have a callable __call__ method, skipping."
            )

    except Exception as e:
        logger.error(f"Failed to patch LangChain LLM classes: {e}")


def _patch_langchain_chat_models() -> None:
    """Patch LangChain Chat model classes to send data to OSMOSIS."""
    try:
        # Try to import BaseChatModel from different possible locations
        # First try langchain_core (newer versions)
        try:
            from langchain_core.language_models.chat_models import BaseChatModel

            logger.info(f"Successfully imported BaseChatModel from langchain_core")
        except ImportError:
            # Then try other possible locations (older versions)
            try:
                from langchain.chat_models.base import BaseChatModel

                logger.info(f"Found BaseChatModel in langchain.chat_models.base")
            except ImportError:
                try:
                    from langchain.chat_models import BaseChatModel

                    logger.info(f"Found BaseChatModel in langchain.chat_models")
                except ImportError:
                    logger.warning(
                        "Could not find BaseChatModel class in any expected location."
                    )
                    return

        logger.info("Calling wrap_langchain()...")

        # Patch the generate method
        if hasattr(BaseChatModel, "generate"):
            original_generate = BaseChatModel.generate

            if not hasattr(original_generate, "_osmosis_aiped"):

                @functools.wraps(original_generate)
                def wrapped_generate(self, messages, stop=None, **kwargs):
                    # Get the response
                    response = original_generate(self, messages, stop=stop, **kwargs)

                    # Send to OSMOSIS if enabled
                    if utils.enabled:
                        # Try to get model name
                        model_name = "unknown_model"
                        if hasattr(self, "model_name"):
                            model_name = self.model_name

                        # Create payload
                        payload = {
                            "chat_model_type": self.__class__.__name__,
                            "model_name": model_name,
                            "messages": [
                                str(msg) for msg in messages
                            ],  # Convert to strings for serialization
                            "response": str(
                                response
                            ),  # Convert to string since it may not be serializable
                            "kwargs": {"stop": stop, **kwargs},
                        }

                        send_to_osmosis(
                            query={
                                "type": "langchain_chat_generate",
                                "messages": [str(msg) for msg in messages],
                                "model": model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_generate._osmosis_aiped = True
                BaseChatModel.generate = wrapped_generate
            else:
                logger.info("LangChain BaseChatModel.generate already wrapped.")

        # Patch agenerate method if it exists
        if hasattr(BaseChatModel, "agenerate"):
            original_agenerate = BaseChatModel.agenerate

            if not hasattr(original_agenerate, "_osmosis_aiped"):

                @functools.wraps(original_agenerate)
                async def wrapped_agenerate(self, messages, stop=None, **kwargs):
                    # Get the response
                    response = await original_agenerate(
                        self, messages, stop=stop, **kwargs
                    )

                    # Send to OSMOSIS if enabled
                    if utils.enabled:
                        # Try to get model name
                        model_name = "unknown_model"
                        if hasattr(self, "model_name"):
                            model_name = self.model_name

                        # Create payload
                        payload = {
                            "chat_model_type": self.__class__.__name__,
                            "model_name": model_name,
                            "messages": [
                                str(msg) for msg in messages
                            ],  # Convert to strings for serialization
                            "response": str(
                                response
                            ),  # Convert to string since it may not be serializable
                            "kwargs": {"stop": stop, **kwargs},
                        }

                        send_to_osmosis(
                            query={
                                "type": "langchain_chat_agenerate",
                                "messages": [str(msg) for msg in messages],
                                "model": model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_agenerate._osmosis_aiped = True
                BaseChatModel.agenerate = wrapped_agenerate
            else:
                logger.info("LangChain BaseChatModel.agenerate already wrapped.")

        # Patch the invoke method if it exists
        if hasattr(BaseChatModel, "invoke"):
            original_invoke = BaseChatModel.invoke

            if not hasattr(original_invoke, "_osmosis_aiped"):

                @functools.wraps(original_invoke)
                def wrapped_invoke(self, messages, *args, **kwargs):
                    # Call original
                    response = original_invoke(self, messages, *args, **kwargs)

                    # Send to OSMOSIS if enabled
                    if utils.enabled:
                        # Try to get model name
                        model_name = "unknown_model"
                        if hasattr(self, "model_name"):
                            model_name = self.model_name

                        # Create payload
                        payload = {
                            "chat_model_type": self.__class__.__name__,
                            "model_name": model_name,
                            "messages": [
                                str(msg) for msg in messages
                            ],  # Convert to strings for serialization
                            "response": str(
                                response
                            ),  # Convert to string since it may not be serializable
                            "kwargs": kwargs,
                        }

                        send_to_osmosis(
                            query={
                                "type": "langchain_chat_invoke",
                                "messages": [str(msg) for msg in messages],
                                "model": model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_invoke._osmosis_aiped = True
                BaseChatModel.invoke = wrapped_invoke
            else:
                logger.info("LangChain BaseChatModel.invoke already wrapped.")

        # Patch ainvoke method if it exists
        if hasattr(BaseChatModel, "ainvoke"):
            original_ainvoke = BaseChatModel.ainvoke

            if not hasattr(original_ainvoke, "_osmosis_aiped"):

                @functools.wraps(original_ainvoke)
                async def wrapped_ainvoke(self, messages, *args, **kwargs):
                    # Call original
                    response = await original_ainvoke(self, messages, *args, **kwargs)

                    # Send to OSMOSIS if enabled
                    if utils.enabled:
                        # Try to get model name
                        model_name = "unknown_model"
                        if hasattr(self, "model_name"):
                            model_name = self.model_name

                        # Create payload
                        payload = {
                            "chat_model_type": self.__class__.__name__,
                            "model_name": model_name,
                            "messages": [
                                str(msg) for msg in messages
                            ],  # Convert to strings for serialization
                            "response": str(
                                response
                            ),  # Convert to string since it may not be serializable
                            "kwargs": kwargs,
                        }

                        send_to_osmosis(
                            query={
                                "type": "langchain_chat_ainvoke",
                                "messages": [str(msg) for msg in messages],
                                "model": model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_ainvoke._osmosis_aiped = True
                BaseChatModel.ainvoke = wrapped_ainvoke
            else:
                logger.info("LangChain BaseChatModel.ainvoke already wrapped.")

        # For modern LangChain, patch __call__ which could be the Model.__call__ method
        if hasattr(BaseChatModel, "__call__"):
            # Get the method, not the descriptor
            original_call_method = BaseChatModel.__call__

            if not hasattr(original_call_method, "_osmosis_aiped"):

                @functools.wraps(original_call_method)
                def wrapped_call_method(self, messages, stop=None, **kwargs):
                    # Get the response
                    response = original_call_method(self, messages, stop=stop, **kwargs)

                    # Send to OSMOSIS if enabled
                    if utils.enabled:
                        # Try to get model name
                        model_name = "unknown_model"
                        if hasattr(self, "model_name"):
                            model_name = self.model_name

                        # Create payload
                        payload = {
                            "chat_model_type": self.__class__.__name__,
                            "model_name": model_name,
                            "messages": [
                                str(msg) for msg in messages
                            ],  # Convert to strings for serialization
                            "response": str(
                                response
                            ),  # Convert to string since it may not be serializable
                            "kwargs": {"stop": stop, **kwargs},
                        }

                        send_to_osmosis(
                            query={
                                "type": "langchain_chat_call",
                                "messages": [str(msg) for msg in messages],
                                "model": model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_call_method._osmosis_aiped = True
                BaseChatModel.__call__ = wrapped_call_method
            else:
                logger.info("LangChain BaseChatModel.__call__ already wrapped.")

    except Exception as e:
        logger.error(f"Failed to patch LangChain Chat model classes: {e}")


def _patch_langchain_prompts() -> None:
    """Patch LangChain prompt templates to send data to OSMOSIS."""
    try:
        # Try to import BasePromptTemplate from different possible locations
        try:
            # Try from langchain_core first (newer versions)
            from langchain_core.prompts import BasePromptTemplate

            logger.info(f"Successfully imported from langchain_core.prompts")
            import_path = "langchain_core.prompts"
        except ImportError:
            # Try from langchain for older versions
            try:
                from langchain.prompts import BasePromptTemplate

                logger.info(f"Found prompt templates via langchain.prompts")
                import_path = "langchain.prompts"
            except ImportError:
                # Last attempt
                try:
                    from langchain.prompts.base import BasePromptTemplate

                    logger.info(f"Found prompt templates via langchain.prompts.base")
                    import_path = "langchain.prompts.base"
                except ImportError:
                    logger.warning(
                        "Could not import BasePromptTemplate from any expected location."
                    )
                    return

        # Patch the format method
        original_format = BasePromptTemplate.format
        logger.debug(f"Original format method: {original_format}")

        # Only patch if not already patched
        if not hasattr(original_format, "_osmosis_aiped"):
            logger.info("Calling wrap_langchain()...")

            @functools.wraps(original_format)
            def wrapped_format(self, **kwargs):
                # Call the original format method
                formatted_prompt = original_format(self, **kwargs)

                # Send to OSMOSIS if enabled
                if utils.enabled:
                    # Create payload
                    payload = {
                        "prompt_type": self.__class__.__name__,
                        "template": getattr(self, "template", None),
                        "input_variables": getattr(self, "input_variables", []),
                        "template_format": getattr(self, "template_format", None),
                        "kwargs": kwargs,
                        "formatted_prompt": formatted_prompt,
                    }

                    send_to_osmosis(
                        query={
                            "type": "langchain_prompt",
                            "template": getattr(self, "template", str(self)),
                        },
                        response=payload,
                        status=200,
                    )

                return formatted_prompt

            # Mark the method as wrapped to avoid double wrapping
            wrapped_format._osmosis_aiped = True
            BasePromptTemplate.format = wrapped_format
        else:
            logger.info("LangChain BasePromptTemplate.format already wrapped.")

    except Exception as e:
        logger.error(f"Failed to patch LangChain prompt templates: {e}")
        # If format method patching failed but the class exists, try direct patching
        try:
            if "BasePromptTemplate" in locals():
                logger.debug("Format method wasn't patched, patching manually...")
                BasePromptTemplate.format = wrapped_format
                logger.debug(
                    f"After manual patch: {BasePromptTemplate.format != original_format}"
                )
        except Exception as inner_e:
            logger.error(f"Manual patching also failed: {inner_e}")
