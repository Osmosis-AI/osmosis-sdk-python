"""Shared helpers for local-execution CLI commands (`test` and `bench`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from osmosis_ai.rollout.cli_utils import CLIError, load_agent_loop
from osmosis_ai.rollout.console import Console
from osmosis_ai.rollout.eval.common.dataset import DatasetReader
from osmosis_ai.rollout.eval.common.errors import (
    DatasetParseError,
    DatasetValidationError,
    ProviderError,
)

if TYPE_CHECKING:
    from osmosis_ai.rollout.core.base import RolloutAgentLoop
    from osmosis_ai.rollout.eval.common.dataset import DatasetRow
    from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient


def format_duration(ms: float) -> str:
    """Format duration in human-readable format."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60000:
        return f"{ms/1000:.1f}s"
    minutes = int(ms // 60000)
    seconds = (ms % 60000) / 1000
    return f"{minutes}m{seconds:.1f}s"


def format_tokens(tokens: int) -> str:
    """Format token count with comma separators."""
    return f"{tokens:,}"


def load_agent(
    module: str,
    quiet: bool,
    console: Console,
) -> Tuple[Optional["RolloutAgentLoop"], Optional[str]]:
    """Load agent loop from module path."""
    if not quiet:
        console.print(f"Loading agent: {module}")

    try:
        agent_loop = load_agent_loop(module)
    except CLIError as e:
        return None, str(e)

    if not quiet:
        console.print(f"  Agent name: {agent_loop.name}")

    return agent_loop, None


def load_dataset_rows(
    dataset_path: str,
    limit: Optional[int],
    offset: int,
    quiet: bool,
    console: Console,
    empty_error: str,
    action_label: str,
) -> Tuple[Optional[List["DatasetRow"]], Optional[str]]:
    """Load rows from dataset and provide command-specific messaging."""
    if not quiet:
        console.print(f"Loading dataset: {dataset_path}")

    try:
        reader = DatasetReader(dataset_path)
        total_rows = len(reader)
        rows = reader.read(limit=limit, offset=offset)
    except FileNotFoundError as e:
        return None, str(e)
    except (DatasetParseError, DatasetValidationError) as e:
        return None, str(e)

    if not rows:
        return None, empty_error

    if not quiet:
        if limit:
            console.print(f"  Total rows: {total_rows} ({action_label} {len(rows)})")
        else:
            console.print(f"  Total rows: {len(rows)}")

    return rows, None


def create_llm_client(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    quiet: bool,
    console: Console,
) -> Tuple[Optional["ExternalLLMClient"], Optional[str]]:
    """Initialize ExternalLLMClient with consistent messaging and errors."""
    from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

    if not quiet:
        if base_url:
            console.print(f"Connecting to endpoint: {base_url}")
        else:
            provider_name = model.split("/")[0] if "/" in model else "openai"
            console.print(f"Initializing provider: {provider_name}")

    try:
        llm_client = ExternalLLMClient(
            model=model,
            api_key=api_key,
            api_base=base_url,
        )
    except ProviderError as e:
        return None, str(e)

    if not quiet:
        model_name = getattr(llm_client, "model", model)
        console.print(f"  Model: {model_name}")

    return llm_client, None


def build_completion_params(
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    """Build completion params dict from CLI options."""
    params: Dict[str, Any] = {}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    return params

__all__ = [
    "build_completion_params",
    "create_llm_client",
    "format_duration",
    "format_tokens",
    "load_agent",
    "load_dataset_rows",
]
