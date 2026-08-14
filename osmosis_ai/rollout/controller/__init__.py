"""Generic rollout callback controller primitives.

Optional FastAPI/aiohttp symbols load on first access and require the
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
    from osmosis_ai.rollout.controller.proxy_client import (
        EvalProxyClient,
        EvalProxyError,
        EvalProxySession,
        EvalProxyStubUpstream,
        create_eval_proxy_stub_app,
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
    "EvalProxyClient": (
        "osmosis_ai.rollout.controller.proxy_client",
        "EvalProxyClient",
    ),
    "EvalProxyError": (
        "osmosis_ai.rollout.controller.proxy_client",
        "EvalProxyError",
    ),
    "EvalProxySession": (
        "osmosis_ai.rollout.controller.proxy_client",
        "EvalProxySession",
    ),
    "EvalProxyStubUpstream": (
        "osmosis_ai.rollout.controller.proxy_client",
        "EvalProxyStubUpstream",
    ),
    "TerminalCallbackResult": (
        "osmosis_ai.rollout.controller.store",
        "TerminalCallbackResult",
    ),
    "UnknownRolloutIdError": (
        "osmosis_ai.rollout.controller.store",
        "UnknownRolloutIdError",
    ),
    "create_callback_app": (
        "osmosis_ai.rollout.controller.listener",
        "create_callback_app",
    ),
    "create_eval_proxy_stub_app": (
        "osmosis_ai.rollout.controller.proxy_client",
        "create_eval_proxy_stub_app",
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
    "EvalProxyClient",
    "EvalProxyError",
    "EvalProxySession",
    "EvalProxyStubUpstream",
    "TerminalCallbackResult",
    "UnknownRolloutIdError",
    "create_callback_app",
    "create_eval_proxy_stub_app",
]
