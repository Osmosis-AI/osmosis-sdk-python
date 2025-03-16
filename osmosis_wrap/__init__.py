"""
osmosis-wrap: A library for monkey patching LLM APIs to print all prompts and responses.

This module patches various LLM client libraries to send all prompts and responses
to the Hoover API for logging and monitoring.

To use this library:

1. Initialize with your Hoover API key:
   osmosis_wrap.init("your-hoover-api-key")

2. Import and use your LLM clients as usual:
   from anthropic import Anthropic
   client = Anthropic()
   
   # Or with async clients:
   from anthropic import AsyncAnthropic
   async_client = AsyncAnthropic()
   
   from openai import OpenAI, AsyncOpenAI
   openai_client = OpenAI()
   async_openai_client = AsyncOpenAI()
   
   # With LangChain:
   from langchain.llms import OpenAI as LangChainOpenAI
   from langchain.chat_models import ChatOpenAI
   
   llm = LangChainOpenAI()
   chat_model = ChatOpenAI()

Currently supported adapters:
- Anthropic (both sync and async clients)
- OpenAI (both sync and async clients, v1 and v2 API versions)
- LangChain (LLMs, Chat Models, and Prompt Templates)
"""

# Import consts directly to avoid cyclic imports
from .consts import package_name, package_version, indent

# Use lazy imports to avoid importing modules during setup
def _import_modules():
    global utils, logger, reconfigure_logger, anthropic, openai, langchain
    global enabled, use_stderr, pretty_print, print_messages
    global init, wrap_anthropic, wrap_openai, disable_hoover, enable_hoover, wrap_langchain
    
    from . import utils
    from .logger import logger, reconfigure_logger
    from .adapters import anthropic, openai
    
    # Re-export configuration flags for easy access
    enabled = utils.enabled
    use_stderr = utils.use_stderr
    pretty_print = utils.pretty_print
    print_messages = utils.print_messages
    
    # Re-export initialization function
    init = utils.init
    
    # Export adapter functions
    wrap_anthropic = anthropic.wrap_anthropic
    wrap_openai = openai.wrap_openai
    
    # Export disable and enable functions
    disable_hoover = utils.disable_hoover
    enable_hoover = utils.enable_hoover
    
    # Automatically apply patches when the module is imported
    wrap_anthropic()
    wrap_openai()
    
    # Conditionally load and apply LangChain wrapper
    try:
        from .adapters import langchain
        wrap_langchain = langchain.wrap_langchain
        # Apply the wrapper, but don't fail if it doesn't work
        try:
            wrap_langchain()
        except Exception as e:
            logger.warning(f"Failed to wrap LangChain: {str(e)}")
    except ImportError:
        # LangChain not installed or not compatible
        def wrap_langchain():
            logger.warning("LangChain support not available. Install LangChain to use this feature.")

# Initialize the module on first import, but not during installation
import sys
if 'pip' not in sys.modules:
    _import_modules() 