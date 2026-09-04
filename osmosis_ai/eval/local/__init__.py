"""Local evaluation runner: supervisor, durable state, and result projections."""

from typing import TYPE_CHECKING

from osmosis_ai._imports import (
    public_module_dir,
    raise_optional_dependency_error,
    resolve_lazy_export,
)

if TYPE_CHECKING:
    from osmosis_ai.eval.local.listener import LlmBridgeListener
    from osmosis_ai.eval.local.llm_bridge import LiteLLMBridge

_EXPORTS: dict[str, tuple[str, str]] = {
    "LlmBridgeListener": (
        "osmosis_ai.eval.local.listener",
        "LlmBridgeListener",
    ),
    "LiteLLMBridge": (
        "osmosis_ai.eval.local.llm_bridge",
        "LiteLLMBridge",
    ),
}


def __getattr__(name: str) -> object:
    try:
        return resolve_lazy_export(
            name,
            module_name=__name__,
            namespace=globals(),
            exports=_EXPORTS,
        )
    except ModuleNotFoundError as exc:
        raise_optional_dependency_error(
            exc,
            extra="eval",
            feature="Local evaluation",
        )


def __dir__() -> list[str]:
    return public_module_dir(globals(), _EXPORTS)


__all__ = [
    "LiteLLMBridge",
    "LlmBridgeListener",
]
