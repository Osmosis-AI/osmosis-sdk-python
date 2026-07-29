"""
osmosis-ai: A Python SDK and CLI for building agent rollouts and managing LLM post-training workflows on Osmosis.

Features:
- Rubric evaluation via LLM-as-judge (evaluate_rubric)
- Type-safe interfaces for LLM-centric workflows

Remote rollout uses ``osmosis_ai.rollout`` and is not re-exported at package
top level.
"""

from typing import TYPE_CHECKING

from ._imports import public_module_dir, resolve_lazy_export
from .consts import PACKAGE_VERSION as __version__

if TYPE_CHECKING:
    from .eval.rubric import (
        MissingAPIKeyError,
        ModelNotFoundError,
        ProviderRequestError,
        RubricResult,
    )
    from .eval.rubric import (
        evaluate_rubric as evaluate_rubric,
    )

# ---------------------------------------------------------------------------
# Lazy-loaded exports: these names are resolved on first access so that
# importing ``osmosis_ai`` does not pull in heavy dependencies (litellm,
# openai, …) unless actually needed.
# ---------------------------------------------------------------------------

_EXPORTS: dict[str, tuple[str, str]] = {
    "MissingAPIKeyError": ("osmosis_ai.eval.rubric.types", "MissingAPIKeyError"),
    "ModelNotFoundError": ("osmosis_ai.eval.rubric.types", "ModelNotFoundError"),
    "ProviderRequestError": ("osmosis_ai.eval.rubric.types", "ProviderRequestError"),
    "RubricResult": ("osmosis_ai.eval.rubric.types", "RubricResult"),
    "evaluate_rubric": ("osmosis_ai.eval.rubric", "evaluate_rubric"),
}


def __getattr__(name: str) -> object:
    return resolve_lazy_export(
        name,
        module_name=__name__,
        namespace=globals(),
        exports=_EXPORTS,
    )


def __dir__() -> list[str]:
    return public_module_dir(globals(), _EXPORTS)


__all__ = [
    "MissingAPIKeyError",
    "ModelNotFoundError",
    "ProviderRequestError",
    "RubricResult",
    "__version__",
]
