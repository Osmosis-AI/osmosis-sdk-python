"""Shared common building blocks for rollout evaluation workflows."""

from osmosis_ai.rollout.eval.common.cli import (
    build_completion_params,
    create_llm_client,
    format_duration,
    format_tokens,
    load_agent,
    load_dataset_rows,
)
from osmosis_ai.rollout.eval.common.dataset import (
    REQUIRED_COLUMNS,
    DatasetReader,
    DatasetRow,
    dataset_row_to_request,
)
from osmosis_ai.rollout.eval.common.errors import (
    DatasetParseError,
    DatasetValidationError,
    LocalExecutionError,
    ProviderError,
    ToolValidationError,
)
from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient
from osmosis_ai.rollout.eval.common.runner import (
    LocalBatchResult,
    LocalLLMClientProtocol,
    LocalRolloutRunner,
    LocalRunResult,
    validate_tools,
)

__all__ = [
    "build_completion_params",
    "create_llm_client",
    "format_duration",
    "format_tokens",
    "load_agent",
    "load_dataset_rows",
    "REQUIRED_COLUMNS",
    "DatasetReader",
    "DatasetRow",
    "dataset_row_to_request",
    "LocalExecutionError",
    "DatasetValidationError",
    "DatasetParseError",
    "ToolValidationError",
    "ProviderError",
    "ExternalLLMClient",
    "LocalBatchResult",
    "LocalRunResult",
    "LocalLLMClientProtocol",
    "LocalRolloutRunner",
    "validate_tools",
]
