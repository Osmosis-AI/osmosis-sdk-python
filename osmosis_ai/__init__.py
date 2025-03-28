"""
osmosis-ai: A library for monkey patching LLM APIs to print all prompts and responses.

This module patches various LLM client libraries to send all prompts and responses
to the OSMOSIS API for logging and monitoring.

Currently supported adapters:
- Anthropic (both sync and async clients)
- OpenAI (both sync and async clients, v1 and v2 API versions)
- LangChain (LLMs, Chat Models, and Prompt Templates)
"""


# Use lazy imports to avoid importing modules during setup
def _import_modules():
    global utils, logger, reconfigure_logger
    global set_log_destination, log_destination, LogDestination
    global init, enabled, disable_osmosis, enable_osmosis

    from . import utils
    from .logger import logger, reconfigure_logger, set_log_destination, log_destination
    from .consts import LogDestination

    # Re-export configuration flags for easy access
    enabled = utils.enabled

    # Export disable and enable functions
    disable_osmosis = utils.disable_osmosis
    enable_osmosis = utils.enable_osmosis

    # Re-export initialization function
    init = utils.init

    # Initialize wrappers as None
    global wrap_anthropic, wrap_openai, wrap_langchain
    global wrap_langchain_openai, wrap_langchain_anthropic

    wrap_anthropic = None
    wrap_openai = None
    wrap_langchain = None
    wrap_langchain_openai = None
    wrap_langchain_anthropic = None

    # Lazily set up anthropic wrapper if available
    try:
        from .adapters import anthropic

        wrap_anthropic = anthropic.wrap_anthropic
        # Apply the wrapper, but don't fail if it doesn't work
        try:
            wrap_anthropic()
        except Exception as e:
            logger.warning(f"Failed to wrap Anthropic: {str(e)}")
    except ImportError:

        def wrap_anthropic():
            logger.warning(
                "Anthropic support not available. Install Anthropic to use this feature."
            )

    # Lazily set up OpenAI wrapper if available
    try:
        from .adapters import openai

        wrap_openai = openai.wrap_openai
        # Apply the wrapper, but don't fail if it doesn't work
        try:
            wrap_openai()
        except Exception as e:
            logger.warning(f"Failed to wrap OpenAI: {str(e)}")
    except ImportError:

        def wrap_openai():
            logger.warning(
                "OpenAI support not available. Install OpenAI to use this feature."
            )

    # Lazily set up LangChain wrapper if available
    try:
        from .adapters import langchain

        wrap_langchain = langchain.wrap_langchain
        # Apply the wrapper, but don't fail if it doesn't work
        try:
            wrap_langchain()
        except Exception as e:
            logger.warning(f"Failed to wrap LangChain: {str(e)}")
    except ImportError:

        def wrap_langchain():
            logger.warning(
                "LangChain support not available. Install LangChain to use this feature."
            )

    # Lazily set up LangChain-OpenAI wrapper if available
    try:
        from .adapters import langchain_openai

        wrap_langchain_openai = langchain_openai.wrap_langchain_openai
        # Apply the wrapper, but don't fail if it doesn't work
        try:
            wrap_langchain_openai()
        except Exception as e:
            logger.warning(f"Failed to wrap LangChain-OpenAI: {str(e)}")
    except ImportError:

        def wrap_langchain_openai():
            logger.warning(
                "LangChain-OpenAI support not available. Install LangChain and OpenAI to use this feature."
            )

    # Lazily set up LangChain-Anthropic wrapper if available
    try:
        from .adapters import langchain_anthropic

        wrap_langchain_anthropic = langchain_anthropic.wrap_langchain_anthropic
        # Apply the wrapper, but don't fail if it doesn't work
        try:
            wrap_langchain_anthropic()
        except Exception as e:
            logger.warning(f"Failed to wrap LangChain-Anthropic: {str(e)}")
    except ImportError:

        def wrap_langchain_anthropic():
            logger.warning(
                "LangChain-Anthropic support not available. Install LangChain and Anthropic to use this feature."
            )


# Initialize the module on first import, but not during installation
import sys

if "pip" not in sys.modules:
    _import_modules()
