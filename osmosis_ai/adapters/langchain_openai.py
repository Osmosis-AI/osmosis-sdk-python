"""
Langchain-OpenAI adapter for Osmosis

This module provides monkey patching for the langchain-openai package.
"""

import functools
import sys

from osmosis_ai import utils
from osmosis_ai.utils import send_to_osmosis
from osmosis_ai.logger import logger


def wrap_langchain_openai() -> None:
    """
    Monkey patch langchain-openai's models to send all prompts and responses to OSMOSIS.

    This function should be called before using any langchain-openai models.
    """
    try:
        import langchain_openai
    except ImportError:
        logger.debug("langchain-openai package is not installed.")
        return

    _patch_openai_chat_models()
    _patch_openai_llm_models()

    logger.info("langchain-openai has been wrapped by osmosis-ai.")


def _patch_openai_chat_models() -> None:
    """Patch langchain-openai chat model classes to send data to OSMOSIS."""
    try:
        # Try to import ChatOpenAI class
        try:
            from langchain_openai import ChatOpenAI

            logger.info("Successfully imported ChatOpenAI from langchain_openai")
        except ImportError:
            # Handle older versions if needed
            try:
                from langchain.chat_models.openai import ChatOpenAI

                logger.info("Found ChatOpenAI in langchain.chat_models.openai")
            except ImportError:
                logger.warning(
                    "Could not find ChatOpenAI class in any expected location."
                )
                return

        # Log available methods on ChatOpenAI for debugging
        chat_methods = [
            method
            for method in dir(ChatOpenAI)
            if not method.startswith("__")
            or method in ["_generate", "_agenerate", "_call", "_acall"]
        ]
        logger.info(f"Found the following methods on ChatOpenAI: {chat_methods}")

        # Try to get the model attribute name - should be model_name but might differ
        model_attr = None
        for attr in ["model_name", "model"]:
            if hasattr(ChatOpenAI, attr):
                model_attr = attr
                logger.info(f"ChatOpenAI uses '{attr}' attribute for model name")
                break

        if not model_attr:
            model_attr = (
                "model_name"  # Default to 'model_name' if we can't determine it
            )
            logger.info(
                f"Could not determine model attribute name, defaulting to '{model_attr}'"
            )

        # Patch the _generate method if it exists
        if hasattr(ChatOpenAI, "_generate"):
            original_generate = ChatOpenAI._generate

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
                            "model_type": "ChatOpenAI",
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
                                "type": "langchain_openai_generate",
                                "messages": [str(msg) for msg in messages],
                                "model": model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_generate._osmosis_aiped = True
                ChatOpenAI._generate = wrapped_generate
                logger.info("Successfully wrapped ChatOpenAI._generate method")
            else:
                logger.info("ChatOpenAI._generate already wrapped.")
        else:
            logger.info("ChatOpenAI does not have a _generate method, skipping.")

        # Patch the _agenerate method if it exists
        if hasattr(ChatOpenAI, "_agenerate"):
            original_agenerate = ChatOpenAI._agenerate

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
                            "model_type": "ChatOpenAI",
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
                                "type": "langchain_openai_agenerate",
                                "messages": [str(msg) for msg in messages],
                                "model": model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_agenerate._osmosis_aiped = True
                ChatOpenAI._agenerate = wrapped_agenerate
                logger.info("Successfully wrapped ChatOpenAI._agenerate method")
            else:
                logger.info("ChatOpenAI._agenerate already wrapped.")
        else:
            logger.info("ChatOpenAI does not have a _agenerate method, skipping.")

        # Patch _call method if it exists (used in newer versions)
        if hasattr(ChatOpenAI, "_call"):
            original_call = ChatOpenAI._call

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
                                "model_type": "ChatOpenAI",
                                "model_name": model_name,
                                "messages": [
                                    str(msg) for msg in messages
                                ],  # Convert to strings for serialization
                                "response": str(response),
                                "kwargs": {"stop": stop, **kwargs},
                            }

                            send_to_osmosis(
                                query={
                                    "type": "langchain_openai_call",
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
                                "model_type": "ChatOpenAI",
                                "model_name": model_name,
                                "messages": [str(msg) for msg in messages],
                                "response": str(response),
                                "kwargs": {"stop": stop, **kwargs},
                            }

                            send_to_osmosis(
                                query={
                                    "type": "langchain_openai_call_fallback",
                                    "messages": [str(msg) for msg in messages],
                                    "model": model_name,
                                },
                                response=payload,
                                status=200,
                            )

                        return response

                wrapped_call._osmosis_aiped = True
                ChatOpenAI._call = wrapped_call
                logger.info("Successfully wrapped ChatOpenAI._call method")
            else:
                logger.info("ChatOpenAI._call already wrapped.")
        else:
            logger.info("ChatOpenAI does not have a _call method, skipping.")

        # Patch _acall method if it exists
        if hasattr(ChatOpenAI, "_acall"):
            original_acall = ChatOpenAI._acall

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
                                "model_type": "ChatOpenAI",
                                "model_name": model_name,
                                "messages": [
                                    str(msg) for msg in messages
                                ],  # Convert to strings for serialization
                                "response": str(response),
                                "kwargs": {"stop": stop, **kwargs},
                            }

                            send_to_osmosis(
                                query={
                                    "type": "langchain_openai_acall",
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
                                "model_type": "ChatOpenAI",
                                "model_name": model_name,
                                "messages": [str(msg) for msg in messages],
                                "response": str(response),
                                "kwargs": {"stop": stop, **kwargs},
                            }

                            send_to_osmosis(
                                query={
                                    "type": "langchain_openai_acall_fallback",
                                    "messages": [str(msg) for msg in messages],
                                    "model": model_name,
                                },
                                response=payload,
                                status=200,
                            )

                        return response

                wrapped_acall._osmosis_aiped = True
                ChatOpenAI._acall = wrapped_acall
                logger.info("Successfully wrapped ChatOpenAI._acall method")
            else:
                logger.info("ChatOpenAI._acall already wrapped.")
        else:
            logger.info("ChatOpenAI does not have a _acall method, skipping.")

    except Exception as e:
        logger.error(f"Failed to patch langchain-openai chat model classes: {e}")


def _patch_openai_llm_models() -> None:
    """Patch langchain-openai LLM classes to send data to OSMOSIS."""
    try:
        # Try to import OpenAI class
        try:
            from langchain_openai import OpenAI

            logger.info("Successfully imported OpenAI from langchain_openai")
        except ImportError:
            # Handle older versions if needed
            try:
                from langchain.llms.openai import OpenAI

                logger.info("Found OpenAI in langchain.llms.openai")
            except ImportError:
                logger.warning("Could not find OpenAI class in any expected location.")
                return

        # Patch the _call method if it exists
        if hasattr(OpenAI, "_call"):
            original_call = OpenAI._call

            if not hasattr(original_call, "_osmosis_aiped"):

                @functools.wraps(original_call)
                def wrapped_call(self, prompt, stop=None, run_manager=None, **kwargs):
                    # Get the response
                    response = original_call(
                        self, prompt, stop=stop, run_manager=run_manager, **kwargs
                    )

                    # Send to OSMOSIS if enabled
                    if utils.enabled:
                        # Create payload
                        payload = {
                            "model_type": "OpenAI",
                            "model_name": self.model_name,
                            "prompt": prompt,
                            "response": response,
                            "kwargs": {"stop": stop, **kwargs},
                        }

                        send_to_osmosis(
                            query={
                                "type": "langchain_openai_llm_call",
                                "prompt": prompt,
                                "model": self.model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_call._osmosis_aiped = True
                OpenAI._call = wrapped_call
            else:
                logger.info("OpenAI._call already wrapped.")

        # Patch the _acall method if it exists
        if hasattr(OpenAI, "_acall"):
            original_acall = OpenAI._acall

            if not hasattr(original_acall, "_osmosis_aiped"):

                @functools.wraps(original_acall)
                async def wrapped_acall(
                    self, prompt, stop=None, run_manager=None, **kwargs
                ):
                    # Get the response
                    response = await original_acall(
                        self, prompt, stop=stop, run_manager=run_manager, **kwargs
                    )

                    # Send to OSMOSIS if enabled
                    if utils.enabled:
                        # Create payload
                        payload = {
                            "model_type": "OpenAI",
                            "model_name": self.model_name,
                            "prompt": prompt,
                            "response": response,
                            "kwargs": {"stop": stop, **kwargs},
                        }

                        send_to_osmosis(
                            query={
                                "type": "langchain_openai_llm_acall",
                                "prompt": prompt,
                                "model": self.model_name,
                            },
                            response=payload,
                            status=200,
                        )

                    return response

                wrapped_acall._osmosis_aiped = True
                OpenAI._acall = wrapped_acall
            else:
                logger.info("OpenAI._acall already wrapped.")

        # Also try to patch AzureOpenAI if available
        try:
            from langchain_openai import AzureOpenAI

            logger.info("Found AzureOpenAI class, patching...")

            # Patch the _call method if it exists
            if hasattr(AzureOpenAI, "_call"):
                original_call = AzureOpenAI._call

                if not hasattr(original_call, "_osmosis_aiped"):

                    @functools.wraps(original_call)
                    def wrapped_call(
                        self, prompt, stop=None, run_manager=None, **kwargs
                    ):
                        # Get the response
                        response = original_call(
                            self, prompt, stop=stop, run_manager=run_manager, **kwargs
                        )

                        # Send to OSMOSIS if enabled
                        if utils.enabled:
                            # Create payload
                            payload = {
                                "model_type": "AzureOpenAI",
                                "model_name": self.deployment_name,
                                "prompt": prompt,
                                "response": response,
                                "kwargs": {"stop": stop, **kwargs},
                            }

                            send_to_osmosis(
                                query={
                                    "type": "langchain_azure_openai_llm_call",
                                    "prompt": prompt,
                                    "model": self.deployment_name,
                                },
                                response=payload,
                                status=200,
                            )

                        return response

                    wrapped_call._osmosis_aiped = True
                    AzureOpenAI._call = wrapped_call
                else:
                    logger.info("AzureOpenAI._call already wrapped.")

            # Patch the _acall method if it exists
            if hasattr(AzureOpenAI, "_acall"):
                original_acall = AzureOpenAI._acall

                if not hasattr(original_acall, "_osmosis_aiped"):

                    @functools.wraps(original_acall)
                    async def wrapped_acall(
                        self, prompt, stop=None, run_manager=None, **kwargs
                    ):
                        # Get the response
                        response = await original_acall(
                            self, prompt, stop=stop, run_manager=run_manager, **kwargs
                        )

                        # Send to OSMOSIS if enabled
                        if utils.enabled:
                            # Create payload
                            payload = {
                                "model_type": "AzureOpenAI",
                                "model_name": self.deployment_name,
                                "prompt": prompt,
                                "response": response,
                                "kwargs": {"stop": stop, **kwargs},
                            }

                            send_to_osmosis(
                                query={
                                    "type": "langchain_azure_openai_llm_acall",
                                    "prompt": prompt,
                                    "model": self.deployment_name,
                                },
                                response=payload,
                                status=200,
                            )

                        return response

                    wrapped_acall._osmosis_aiped = True
                    AzureOpenAI._acall = wrapped_acall
                else:
                    logger.info("AzureOpenAI._acall already wrapped.")
        except ImportError:
            logger.info("AzureOpenAI not found, skipping.")

        # Also try to patch AzureChatOpenAI if available
        try:
            from langchain_openai import AzureChatOpenAI

            logger.info("Found AzureChatOpenAI class, patching...")

            # Patch the _generate method if it exists
            if hasattr(AzureChatOpenAI, "_generate"):
                original_generate = AzureChatOpenAI._generate

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
                            payload = {
                                "model_type": "AzureChatOpenAI",
                                "model_name": self.deployment_name,
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
                                    "type": "langchain_azure_chat_openai",
                                    "messages": [str(msg) for msg in messages],
                                    "model": self.deployment_name,
                                },
                                response=payload,
                                status=200,
                            )

                        return response

                    wrapped_generate._osmosis_aiped = True
                    AzureChatOpenAI._generate = wrapped_generate
                else:
                    logger.info("AzureChatOpenAI._generate already wrapped.")
        except ImportError:
            logger.info("AzureChatOpenAI not found, skipping.")

    except Exception as e:
        logger.error(f"Failed to patch langchain-openai LLM classes: {e}")
