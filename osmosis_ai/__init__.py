"""
osmosis-ai: A Python library for LLM training workflows.

Features:
- Rubric evaluation via LLM-as-judge (evaluate_rubric)
- Type-safe interfaces for LLM-centric workflows

Remote rollout uses ``osmosis_ai.rollout`` and is not re-exported at package
top level.
"""

from typing import TYPE_CHECKING

from .consts import PACKAGE_VERSION as __version__

if TYPE_CHECKING:
    from .eval.rubric import (
        MissingAPIKeyError,
        ModelNotFoundError,
        ProviderRequestError,
        RubricResult,
        evaluate_rubric,
    )

# ---------------------------------------------------------------------------
# Lazy-loaded exports: these names are resolved on first access so that
# importing ``osmosis_ai`` does not pull in heavy dependencies (litellm,
# openai, …) unless actually needed.
# ---------------------------------------------------------------------------

_RUBRIC_EXPORTS: frozenset[str] = frozenset(
    {
        "MissingAPIKeyError",
        "ModelNotFoundError",
        "ProviderRequestError",
        "RubricResult",
        "evaluate_rubric",
    }
)


def __getattr__(name: str) -> object:
    if name in _RUBRIC_EXPORTS:
        try:
            from .eval import rubric
        except ImportError:
            raise ImportError(
                f"'{name}' requires additional dependencies: "
                "pip install osmosis-ai[cli]"
            ) from None

        value = getattr(rubric, name)
        globals()[name] = value  # cache so future access skips __getattr__
        return value
    raise AttributeError(f"module 'osmosis_ai' has no attribute {name!r}")


__all__ = [
    "MissingAPIKeyError",
    "ModelNotFoundError",
    "ProviderRequestError",
    "RubricResult",
    "__version__",
    "evaluate_rubric",
]
