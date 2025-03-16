"""
osmosis-wrap: A library for monkey patching LLM APIs to print all prompts and responses.

This module patches various LLM client libraries to print all prompts sent to the API
and all responses received, which can be helpful for debugging and monitoring.

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

# Export adapter functions
wrap_anthropic = anthropic.wrap_anthropic
wrap_openai = openai.wrap_openai

# Automatically apply patches when the module is imported
wrap_anthropic()
wrap_openai() 