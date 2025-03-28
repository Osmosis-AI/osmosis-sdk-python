"""
Langchain-Anthropic adapter for Osmosis

This module provides monkey patching for the langchain-anthropic package.
"""

import functools
import sys

from osmosis_ai import utils
from osmosis_ai.utils import send_to_osmosis
from osmosis_ai.logger import logger


def wrap_langchain_anthropic() -> None:
    """
    Monkey patch langchain-anthropic's models to send all prompts and responses to OSMOSIS.

    This function should be called before using any langchain-anthropic models.
    """
    try:
        import langchain_anthropic
    except ImportError:
        logger.debug("langchain-anthropic package is not installed.")
        return

    _patch_anthropic_chat_models()

    logger.info("langchain-anthropic has been wrapped by osmosis-ai.")


def _patch_anthropic_chat_models() -> None:
    """Patch langchain-anthropic chat model classes to send data to OSMOSIS."""
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
                logger.warning(
                    "Could not find ChatAnthropic class in any expected location."
                )
                return

        # Log available methods on ChatAnthropic for debugging
        chat_methods = [
            method
            for method in dir(ChatAnthropic)
            if not method.startswith("__")
            or method in ["_generate", "_agenerate", "_call", "_acall"]
        ]
        logger.info(f"Found the following methods on ChatAnthropic: {chat_methods}")

        # Try to get the model attribute name - could be model or model_name
        model_attr = None
        for attr in ["model", "model_name"]:
            if hasattr(ChatAnthropic, attr):
                model_attr = attr
                logger.info(f"ChatAnthropic uses '{attr}' attribute for model name")
                break

        if not model_attr:
            model_attr = "model"  # Default to 'model' if we can't determine it
            logger.info(
                f"Could not determine model attribute name, defaulting to '{model_attr}'"
            )

        # Patch the _generate method if it exists
        if hasattr(ChatAnthropic, "_generate"):
            original_generate = ChatAnthropic._generate

            if not hasattr(original_generate, "_osmosis_aiped"):

                @functools.wraps(original_generate)
                def wrapped_generate(
                    self, messages, stop=None, run_manager=None, **kwargs
                ):
                    # Get the response
                    response = original_generate(
                        self, messages, stop=stop, run_manager=run_manager, **kwargs
                    )

                    # Send to OSMOSIS if enabled
                    if utils.enabled:
                        # Create payload
                        model_name = getattr(self, model_attr, "unknown_model")
                        payload = {
                            "model_type": "ChatAnthropic",
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
                                "type": "langchain_anthropic_generate",
                                "messages": [str(msg) for msg in messages],
                                "model": model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_generate._osmosis_aiped = True
                ChatAnthropic._generate = wrapped_generate
                logger.info("Successfully wrapped ChatAnthropic._generate method")
            else:
                logger.info("ChatAnthropic._generate already wrapped.")
        else:
            logger.info("ChatAnthropic does not have a _generate method, skipping.")

        # Patch the _agenerate method if it exists
        if hasattr(ChatAnthropic, "_agenerate"):
            original_agenerate = ChatAnthropic._agenerate

            if not hasattr(original_agenerate, "_osmosis_aiped"):

                @functools.wraps(original_agenerate)
                async def wrapped_agenerate(
                    self, messages, stop=None, run_manager=None, **kwargs
                ):
                    # Get the response
                    response = await original_agenerate(
                        self, messages, stop=stop, run_manager=run_manager, **kwargs
                    )

                    # Send to OSMOSIS if enabled
                    if utils.enabled:
                        # Create payload
                        model_name = getattr(self, model_attr, "unknown_model")
                        payload = {
                            "model_type": "ChatAnthropic",
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
                                "type": "langchain_anthropic_agenerate",
                                "messages": [str(msg) for msg in messages],
                                "model": model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_agenerate._osmosis_aiped = True
                ChatAnthropic._agenerate = wrapped_agenerate
                logger.info("Successfully wrapped ChatAnthropic._agenerate method")
            else:
                logger.info("ChatAnthropic._agenerate already wrapped.")
        else:
            logger.info("ChatAnthropic does not have a _agenerate method, skipping.")

        # Patch _call method if it exists (used in newer versions)
        if hasattr(ChatAnthropic, "_call"):
            original_call = ChatAnthropic._call

            if not hasattr(original_call, "_osmosis_aiped"):

                @functools.wraps(original_call)
                def wrapped_call(self, messages, stop=None, run_manager=None, **kwargs):
                    try:
                        # Get the response
                        response = original_call(
                            self, messages, stop=stop, run_manager=run_manager, **kwargs
                        )

                        # Send to OSMOSIS if enabled
                        if utils.enabled:
                            # Create payload
                            model_name = getattr(self, model_attr, "unknown_model")
                            payload = {
                                "model_type": "ChatAnthropic",
                                "model_name": model_name,
                                "messages": [
                                    str(msg) for msg in messages
                                ],  # Convert to strings for serialization
                                "response": str(response),
                                "kwargs": {"stop": stop, **kwargs},
                            }

                            send_to_osmosis(
                                query={
                                    "type": "langchain_anthropic_call",
                                    "messages": [str(msg) for msg in messages],
                                    "model": model_name,
                                },
                                response=payload,
                                status=200,
                            )

                        return response
                    except TypeError as e:
                        # Handle parameter mismatch gracefully
                        logger.warning(
                            f"TypeError in wrapped _call: {e}, trying without run_manager"
                        )
                        # Try calling without run_manager (older versions)
                        response = original_call(self, messages, stop=stop, **kwargs)

                        # Send to OSMOSIS if enabled
                        if utils.enabled:
                            model_name = getattr(self, model_attr, "unknown_model")
                            payload = {
                                "model_type": "ChatAnthropic",
                                "model_name": model_name,
                                "messages": [str(msg) for msg in messages],
                                "response": str(response),
                                "kwargs": {"stop": stop, **kwargs},
                            }

                            send_to_osmosis(
                                query={
                                    "type": "langchain_anthropic_call_fallback",
                                    "messages": [str(msg) for msg in messages],
                                    "model": model_name,
                                },
                                response=payload,
                                status=200,
                            )

                        return response

                wrapped_call._osmosis_aiped = True
                ChatAnthropic._call = wrapped_call
                logger.info("Successfully wrapped ChatAnthropic._call method")
            else:
                logger.info("ChatAnthropic._call already wrapped.")
        else:
            logger.info("ChatAnthropic does not have a _call method, skipping.")

        # Patch _acall method if it exists
        if hasattr(ChatAnthropic, "_acall"):
            original_acall = ChatAnthropic._acall

            if not hasattr(original_acall, "_osmosis_aiped"):

                @functools.wraps(original_acall)
                async def wrapped_acall(
                    self, messages, stop=None, run_manager=None, **kwargs
                ):
                    try:
                        # Get the response
                        response = await original_acall(
                            self, messages, stop=stop, run_manager=run_manager, **kwargs
                        )

                        # Send to OSMOSIS if enabled
                        if utils.enabled:
                            # Create payload
                            model_name = getattr(self, model_attr, "unknown_model")
                            payload = {
                                "model_type": "ChatAnthropic",
                                "model_name": model_name,
                                "messages": [
                                    str(msg) for msg in messages
                                ],  # Convert to strings for serialization
                                "response": str(response),
                                "kwargs": {"stop": stop, **kwargs},
                            }

                            send_to_osmosis(
                                query={
                                    "type": "langchain_anthropic_acall",
                                    "messages": [str(msg) for msg in messages],
                                    "model": model_name,
                                },
                                response=payload,
                                status=200,
                            )

                        return response
                    except TypeError as e:
                        # Handle parameter mismatch gracefully
                        logger.warning(
                            f"TypeError in wrapped _acall: {e}, trying without run_manager"
                        )
                        # Try calling without run_manager (older versions)
                        response = await original_acall(
                            self, messages, stop=stop, **kwargs
                        )

                        # Send to OSMOSIS if enabled
                        if utils.enabled:
                            model_name = getattr(self, model_attr, "unknown_model")
                            payload = {
                                "model_type": "ChatAnthropic",
                                "model_name": model_name,
                                "messages": [str(msg) for msg in messages],
                                "response": str(response),
                                "kwargs": {"stop": stop, **kwargs},
                            }

                            send_to_osmosis(
                                query={
                                    "type": "langchain_anthropic_acall_fallback",
                                    "messages": [str(msg) for msg in messages],
                                    "model": model_name,
                                },
                                response=payload,
                                status=200,
                            )

                        return response

                wrapped_acall._osmosis_aiped = True
                ChatAnthropic._acall = wrapped_acall
                logger.info("Successfully wrapped ChatAnthropic._acall method")
            else:
                logger.info("ChatAnthropic._acall already wrapped.")
        else:
            logger.info("ChatAnthropic does not have a _acall method, skipping.")

    except Exception as e:
        logger.error(f"Failed to patch langchain-anthropic chat model classes: {e}")
