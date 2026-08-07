"""Public rollout server API, loaded only when a symbol is requested."""

from typing import TYPE_CHECKING

from osmosis_ai._imports import (
    public_module_dir,
    raise_optional_dependency_error,
    resolve_lazy_export,
)

if TYPE_CHECKING:
    from osmosis_ai.rollout.server.app import create_rollout_server
    from osmosis_ai.rollout.server.auth import ControllerAuth

_EXPORTS: dict[str, tuple[str, str]] = {
    "ControllerAuth": ("osmosis_ai.rollout.server.auth", "ControllerAuth"),
    "create_rollout_server": (
        "osmosis_ai.rollout.server.app",
        "create_rollout_server",
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
            extra="server",
            feature="The rollout server",
        )


def __dir__() -> list[str]:
    return public_module_dir(globals(), _EXPORTS)


__all__ = [
    "ControllerAuth",
    "create_rollout_server",
]
