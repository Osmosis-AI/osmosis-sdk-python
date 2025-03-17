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

Currently supported adapters:
- Anthropic
- OpenAI
"""

from . import utils
from .adapters import anthropic, openai

# Re-export configuration flags for easy access
enabled = utils.enabled
use_stderr = utils.use_stderr
pretty_print = utils.pretty_print
print_messages = utils.print_messages
indent = utils.indent

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