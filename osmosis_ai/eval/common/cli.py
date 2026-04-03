"""Shared helpers for local-execution CLI commands (`test` and `eval`)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.eval.common.dataset import DatasetReader
from osmosis_ai.eval.common.errors import (
    DatasetParseError,
    DatasetValidationError,
    ProviderError,
    SystemicProviderError,
)

# Mapping of LiteLLM provider prefixes to their expected environment variables.
_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "fireworks_ai": "FIREWORKS_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "perplexity": "PERPLEXITYAI_API_KEY",
    "replicate": "REPLICATE_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "ai21": "AI21_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "nvidia_nim": "NVIDIA_NIM_API_KEY",
    "github": "GITHUB_API_KEY",
}

if TYPE_CHECKING:
    from osmosis_ai.eval.common.dataset import DatasetRow
    from osmosis_ai.eval.common.llm_client import ExternalLLMClient


def format_duration(ms: float) -> str:
    """Format duration in human-readable format."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60000:
        return f"{ms / 1000:.1f}s"
    minutes = int(ms // 60000)
    seconds = (ms % 60000) / 1000
    return f"{minutes}m{seconds:.1f}s"


def format_tokens(tokens: int) -> str:
    """Format token count with comma separators."""
    return f"{tokens:,}"


def truncate_error(text: str, max_len: int = 50) -> str:
    """Truncate a single-line error string with ellipsis if too long."""
    flat = text.replace("\n", " ")
    return flat[: max_len - 3] + "..." if len(flat) > max_len else flat


def _resolve_workflow(module_path: str) -> tuple[type, Any]:
    """Resolve a module:attribute path to an AgentWorkflow subclass and its config.

    Returns (workflow_cls, config) where config may be None if no
    AgentWorkflowConfig instance is found in the module namespace.
    """
    import sys

    from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
    from osmosis_ai.rollout_v2.types import AgentWorkflowConfig
    from osmosis_ai.rollout_v2.utils.imports import resolve_object

    # Ensure cwd is on sys.path so local modules can be imported from CLI.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    obj = resolve_object(module_path)

    if not (isinstance(obj, type) and issubclass(obj, AgentWorkflow)):
        raise TypeError(
            f"'{module_path}' must be an AgentWorkflow subclass, "
            f"got {type(obj).__name__}"
        )

    # Auto-discover config: scan the module for AgentWorkflowConfig instances.
    module_name = module_path.rsplit(":", 1)[0]
    mod = sys.modules[module_name]
    config = None
    for val in vars(mod).values():
        if isinstance(val, AgentWorkflowConfig):
            config = val
            break

    return obj, config


def load_workflow(
    module: str,
    quiet: bool,
    console: Console,
) -> tuple[type | None, Any, str | None]:
    """Load an AgentWorkflow class and its config from a module path.

    Returns (workflow_cls, workflow_config, error).
    """
    if not quiet:
        console.print(f"Loading workflow: {module}")

    try:
        workflow_cls, workflow_config = _resolve_workflow(module)
    except (CLIError, ImportError, ValueError, TypeError) as e:
        return None, None, str(e)

    if not quiet:
        console.print(f"  Workflow: {workflow_cls.__name__}")

    return workflow_cls, workflow_config, None


def load_dataset_rows(
    dataset_path: str,
    limit: int | None,
    offset: int,
    quiet: bool,
    console: Console,
    empty_error: str,
    action_label: str,
) -> tuple[list[DatasetRow] | None, str | None]:
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


def _check_api_key(
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> str | None:
    """Return an error message if the required API key is missing, else None."""
    if api_key or base_url:
        return None

    # LITELLM_API_KEY is a global fallback accepted by LiteLLM for any provider
    if os.environ.get("LITELLM_API_KEY"):
        return None

    provider = model.split("/")[0].lower() if "/" in model else "openai"
    env_var = _PROVIDER_ENV_KEYS.get(provider)
    if env_var is None:
        return None

    if not os.environ.get(env_var):
        return (
            f"Missing API key for provider '{provider}'. "
            f"Set {env_var} or LITELLM_API_KEY, or pass --api-key."
        )
    return None


def _format_model_error(model: str, base_url: str | None, detail: str) -> str:
    """Format a concise model validation error with actionable guidance."""
    if base_url:
        return (
            f"Invalid model for --base-url. Received model='{model}'. Details: {detail}"
        )
    return (
        "Invalid LiteLLM model format. Use 'provider/model' "
        "(for example: openai/gpt-5-mini). "
        f"Received model='{model}'. Details: {detail}"
    )


def _check_model(model: str, base_url: str | None) -> str | None:
    """Return an error message if model format is invalid, else None."""
    from osmosis_ai.eval.common.llm_client import _first_line

    candidate = model.strip()
    if not candidate:
        return "Model cannot be empty. Pass a value with --model."

    # When base_url is provided, any non-empty model name is valid.
    # The ExternalLLMClient will route through openai/ provider internally.
    if base_url:
        return None

    normalized_model = candidate if "/" in candidate else f"openai/{candidate}"
    provider, model_name = normalized_model.split("/", 1)
    if not provider.strip() or not model_name.strip():
        return _format_model_error(
            candidate, base_url, "missing provider or model name"
        )

    try:
        import litellm
    except ImportError:
        # Dependency errors are handled by ExternalLLMClient initialization.
        return None

    # Prevent litellm from registering its buggy atexit cleanup handler
    # before the first lazy-attribute access triggers __getattr__.
    # We handle async client cleanup explicitly in ExternalLLMClient.close().
    if hasattr(litellm, "_async_client_cleanup_registered"):
        litellm._async_client_cleanup_registered = True

    litellm.suppress_debug_info = True

    try:
        litellm.get_llm_provider(model=normalized_model, api_base=base_url)
    except Exception as exc:
        provider_message = getattr(exc, "message", str(exc))
        normalized_message = provider_message.lower()
        if (
            "llm provider not provided" in normalized_message
            or "pass in the llm provider you are trying to call" in normalized_message
        ):
            return _format_model_error(
                candidate,
                base_url,
                _first_line(provider_message),
            )
        return (
            "Invalid model configuration. "
            f"Received model='{candidate}'. Details: {_first_line(provider_message)}"
        )

    return None


def create_llm_client(
    model: str,
    api_key: str | None,
    base_url: str | None,
    quiet: bool,
    console: Console,
) -> tuple[ExternalLLMClient | None, str | None]:
    """Initialize ExternalLLMClient with consistent messaging and errors."""
    if error := _check_model(model, base_url):
        return None, error

    if error := _check_api_key(model, api_key, base_url):
        return None, error

    from osmosis_ai.eval.common.llm_client import ExternalLLMClient

    if not quiet:
        if base_url:
            console.print(f"Connecting to endpoint: {base_url}")
        else:
            provider_name = model.split("/")[0].lower() if "/" in model else "openai"
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
        model_name = getattr(llm_client, "display_name", model)
        console.print(f"  Model: {model_name}")

    return llm_client, None


async def verify_llm_client(
    llm_client: ExternalLLMClient,
    quiet: bool,
    console: Console,
) -> str | None:
    """Run a preflight check against the LLM provider.

    Returns an error message string on failure, or None on success.
    """
    if not quiet:
        console.print("Verifying provider connectivity...")

    try:
        await llm_client.preflight_check()
    except SystemicProviderError as e:
        return str(e)

    return None


def build_completion_params(
    temperature: float | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Build completion params dict from CLI options."""
    params: dict[str, Any] = {}
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
    "load_dataset_rows",
    "load_workflow",
    "truncate_error",
    "verify_llm_client",
]
