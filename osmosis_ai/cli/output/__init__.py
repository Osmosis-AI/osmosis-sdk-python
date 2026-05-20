"""Public surface of the CLI output package."""

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
    MessageResult,
    OperationResult,
)
from .serializers import (
    serialize_checkpoint,
    serialize_dataset,
    serialize_deployment,
    serialize_eval_cache_entry,
    serialize_model,
    serialize_rollout,
    serialize_training_run,
)

__all__ = [
    "CommandResult",
    "DetailField",
    "DetailResult",
    "DetailSection",
    "ListColumn",
    "ListResult",
    "MessageResult",
    "OperationResult",
    "OutputContext",
    "OutputFormat",
    "classify_error",
    "command_path_for_error",
    "default_output_context",
    "emit_structured_error_to_stderr",
    "get_output_context",
    "install_output_context",
    "override_output_context",
    "render",
    "render_command_result",
    "resolve_format_selectors",
    "serialize_checkpoint",
    "serialize_dataset",
    "serialize_deployment",
    "serialize_eval_cache_entry",
    "serialize_model",
    "serialize_rollout",
    "serialize_training_run",
    "verify_output_emitted",
]
