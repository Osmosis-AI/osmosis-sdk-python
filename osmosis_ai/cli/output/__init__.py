"""Public surface of the CLI output package.

Serializer re-exports are resolved lazily so importing this package (or any
submodule) does not create platform API model classes at CLI startup.
"""

from typing import TYPE_CHECKING

from osmosis_ai._imports import public_module_dir, resolve_lazy_export

from .context import (
    OutputContext,
    OutputFormat,
    default_output_context,
    get_output_context,
    install_output_context,
    override_output_context,
    resolve_format_selectors,
)
from .error import (
    classify_error,
    command_path_for_error,
    emit_structured_error_to_stderr,
)
from .renderer import render, render_command_result, verify_output_emitted
from .result import (
    CommandResult,
    DetailField,
    DetailResult,
    DetailSection,
    ListColumn,
    ListResult,
    ListSection,
    MessageResult,
    OperationResult,
    SectionedListResult,
    detail_fields,
)

if TYPE_CHECKING:
    from .serializers import (
        serialize_benchmark_run,
        serialize_checkpoint,
        serialize_dataset,
        serialize_dev_rollout_server,
        serialize_environment_secret,
        serialize_eval_run,
        serialize_lora_model,
        serialize_model,
        serialize_rollout,
        serialize_training_run,
    )

_SERIALIZER_EXPORTS: dict[str, tuple[str, str]] = {
    "serialize_benchmark_run": (
        "osmosis_ai.cli.output.serializers",
        "serialize_benchmark_run",
    ),
    "serialize_checkpoint": (
        "osmosis_ai.cli.output.serializers",
        "serialize_checkpoint",
    ),
    "serialize_dataset": ("osmosis_ai.cli.output.serializers", "serialize_dataset"),
    "serialize_dev_rollout_server": (
        "osmosis_ai.cli.output.serializers",
        "serialize_dev_rollout_server",
    ),
    "serialize_environment_secret": (
        "osmosis_ai.cli.output.serializers",
        "serialize_environment_secret",
    ),
    "serialize_eval_run": ("osmosis_ai.cli.output.serializers", "serialize_eval_run"),
    "serialize_lora_model": (
        "osmosis_ai.cli.output.serializers",
        "serialize_lora_model",
    ),
    "serialize_model": ("osmosis_ai.cli.output.serializers", "serialize_model"),
    "serialize_rollout": ("osmosis_ai.cli.output.serializers", "serialize_rollout"),
    "serialize_training_run": (
        "osmosis_ai.cli.output.serializers",
        "serialize_training_run",
    ),
}


def __getattr__(name: str) -> object:
    return resolve_lazy_export(
        name,
        module_name=__name__,
        namespace=globals(),
        exports=_SERIALIZER_EXPORTS,
    )


def __dir__() -> list[str]:
    return public_module_dir(globals(), _SERIALIZER_EXPORTS)


__all__ = [
    "CommandResult",
    "DetailField",
    "DetailResult",
    "DetailSection",
    "ListColumn",
    "ListResult",
    "ListSection",
    "MessageResult",
    "OperationResult",
    "OutputContext",
    "OutputFormat",
    "SectionedListResult",
    "classify_error",
    "command_path_for_error",
    "default_output_context",
    "detail_fields",
    "emit_structured_error_to_stderr",
    "get_output_context",
    "install_output_context",
    "override_output_context",
    "render",
    "render_command_result",
    "resolve_format_selectors",
    "serialize_benchmark_run",
    "serialize_checkpoint",
    "serialize_dataset",
    "serialize_dev_rollout_server",
    "serialize_environment_secret",
    "serialize_eval_run",
    "serialize_lora_model",
    "serialize_model",
    "serialize_rollout",
    "serialize_training_run",
    "verify_output_emitted",
]
