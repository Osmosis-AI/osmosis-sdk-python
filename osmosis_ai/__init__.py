"""
osmosis-ai: A Python library for reward function validation with strict type enforcement.

This library provides decorators such as @osmosis_reward and @osmosis_rubric that
enforce standardized function signatures for LLM-centric workflows.

Features:
- Type-safe reward function decoration
- Parameter name and type validation
- Support for optional configuration parameters
"""

from .rubric_eval import MissingAPIKeyError, evaluate_rubric
from .rubric_types import ModelNotFoundError, ProviderRequestError
from .utils import osmosis_reward, osmosis_rubric

__all__ = [
    "osmosis_reward",
    "osmosis_rubric",
    "evaluate_rubric",
    "MissingAPIKeyError",
    "ProviderRequestError",
    "ModelNotFoundError",
]
