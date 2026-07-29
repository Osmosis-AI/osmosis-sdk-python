"""Public rubric API with the optional engine loaded on demand."""

from typing import TYPE_CHECKING

from osmosis_ai._imports import (
    public_module_dir,
    raise_optional_dependency_error,
    resolve_lazy_export,
)

if TYPE_CHECKING:
    from .engine import evaluate_rubric
    from .types import (
        MissingAPIKeyError,
        ModelNotFoundError,
        ProviderRequestError,
        RubricResult,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "MissingAPIKeyError": ("osmosis_ai.eval.rubric.types", "MissingAPIKeyError"),
    "ModelNotFoundError": ("osmosis_ai.eval.rubric.types", "ModelNotFoundError"),
    "ProviderRequestError": ("osmosis_ai.eval.rubric.types", "ProviderRequestError"),
    "RubricResult": ("osmosis_ai.eval.rubric.types", "RubricResult"),
    "evaluate_rubric": ("osmosis_ai.eval.rubric.engine", "evaluate_rubric"),
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
        if name != "evaluate_rubric":
            raise
        raise_optional_dependency_error(
            exc,
            extra="rubric",
            expected_modules=frozenset({"litellm", "orjson"}),
            feature="Rubric evaluation",
        )


def __dir__() -> list[str]:
    return public_module_dir(globals(), _EXPORTS)


__all__ = [
    "MissingAPIKeyError",
    "ModelNotFoundError",
    "ProviderRequestError",
    "RubricResult",
    "evaluate_rubric",
]
