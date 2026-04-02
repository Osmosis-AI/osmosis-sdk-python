"""
osmosis-ai: A Python library for LLM training workflows.

Features:
- Remote rollout SDK for integrating agent frameworks with Osmosis training
- Rubric evaluation via LLM-as-judge (evaluate_rubric)
- Type-safe interfaces for LLM-centric workflows
"""

from .consts import PACKAGE_VERSION as __version__

# ---------------------------------------------------------------------------
# Lazy-loaded exports: these names are resolved on first access so that
# importing ``osmosis_ai`` does not pull in heavy dependencies (litellm,
# openai, fastapi, …) unless actually needed.
# ---------------------------------------------------------------------------

_ROLLOUT_EXPORTS: set[str] = {
    "CompletionsResult",
    "InitResponse",
    "OpenAIFunctionToolSchema",
    "OsmosisLLMClient",
    "OsmosisRolloutError",
    "OsmosisServerError",
    "OsmosisTimeoutError",
    "OsmosisTransportError",
    "OsmosisValidationError",
    "RolloutAgentLoop",
    "RolloutContext",
    "RolloutMetrics",
    "RolloutRequest",
    "RolloutResponse",
    "RolloutResult",
    "create_app",
    "get_agent_loop",
    "list_agent_loops",
    "register_agent_loop",
}

_RUBRIC_EXPORTS: set[str] = {
    "MissingAPIKeyError",
    "ModelNotFoundError",
    "ProviderRequestError",
    "RubricResult",
    "evaluate_rubric",
}


def __getattr__(name: str) -> object:
    if name in _ROLLOUT_EXPORTS:
        from . import rollout

        value = getattr(rollout, name)
        globals()[name] = value  # cache so future access skips __getattr__
        return value
    if name in _RUBRIC_EXPORTS:
        from .rollout.eval import rubric

        value = getattr(rubric, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'osmosis_ai' has no attribute {name!r}")


__all__ = [
    "__version__",
    *sorted(_ROLLOUT_EXPORTS),
    *sorted(_RUBRIC_EXPORTS),
]
