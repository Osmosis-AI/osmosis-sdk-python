"""Public Harbor backend API, loaded only when a symbol is requested."""

from typing import TYPE_CHECKING

from osmosis_ai._imports import (
    public_module_dir,
    raise_optional_dependency_error,
    resolve_lazy_export,
)

if TYPE_CHECKING:
    from osmosis_ai.rollout.backend.harbor.backend import HarborBackend
    from osmosis_ai.rollout.backend.harbor.tasks import TaskMode

_EXPORTS: dict[str, tuple[str, str]] = {
    "HarborBackend": ("osmosis_ai.rollout.backend.harbor.backend", "HarborBackend"),
    "TaskMode": ("osmosis_ai.rollout.backend.harbor.tasks", "TaskMode"),
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
            extra="harbor",
            feature="The Harbor backend",
        )


def __dir__() -> list[str]:
    return public_module_dir(globals(), _EXPORTS)


__all__ = [
    "HarborBackend",
    "TaskMode",
]
