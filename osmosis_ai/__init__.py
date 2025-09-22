"""
osmosis-ai: A Python library for reward function validation with strict type enforcement.

This library provides the @osmosis_reward decorator that enforces standardized
function signatures for reward functions used in LLM applications.

Features:
- Type-safe reward function decoration
- Parameter name and type validation
- Support for optional configuration parameters
"""

from .utils import osmosis_reward

__all__ = ["osmosis_reward"]
