"""Generic rollout callback controller primitives.

Optional FastAPI/uvicorn symbols load on first access and require the
``eval-run`` extra. ``CallbackStore`` is in-process only and has no extra.
"""

from typing import TYPE_CHECKING

from osmosis_ai._imports import (
    public_module_dir,
    raise_optional_dependency_error,
    resolve_lazy_export,
)

if TYPE_CHECKING:
    from osmosis_ai.rollout.controller.listener import (
        CallbackListener,
        create_callback_app,
    )
    from osmosis_ai.rollout.controller.llm_bridge import (
        LiteLLMBridge,
        create_bridge_router,
    )
    from osmosis_ai.rollout.controller.store import (
        CallbackStore,
        DuplicateRegistrationError,
        TerminalCallbackResult,
        UnknownRolloutIdError,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "CallbackListener": (
        "osmosis_ai.rollout.controller.listener",
        "CallbackListener",
    ),
    "CallbackStore": ("osmosis_ai.rollout.controller.store", "CallbackStore"),
    "DuplicateRegistrationError": (
        "osmosis_ai.rollout.controller.store",
        "DuplicateRegistrationError",
    ),
    "LiteLLMBridge": (
        "osmosis_ai.rollout.controller.llm_bridge",
        "LiteLLMBridge",
    ),
    "TerminalCallbackResult": (
        "osmosis_ai.rollout.controller.store",
        "TerminalCallbackResult",
    ),
    "UnknownRolloutIdError": (
        "osmosis_ai.rollout.controller.store",
        "UnknownRolloutIdError",
    ),
    "create_bridge_router": (
        "osmosis_ai.rollout.controller.llm_bridge",
        "create_bridge_router",
    ),
    "create_callback_app": (
        "osmosis_ai.rollout.controller.listener",
        "create_callback_app",
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
            extra="eval-run",
            feature="Local evaluation",
        )


def __dir__() -> list[str]:
    return public_module_dir(globals(), _EXPORTS)


__all__ = [
    "CallbackListener",
    "CallbackStore",
    "DuplicateRegistrationError",
    "LiteLLMBridge",
    "TerminalCallbackResult",
    "UnknownRolloutIdError",
    "create_bridge_router",
    "create_callback_app",
]
