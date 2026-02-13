"""Shared helpers for local-execution CLI commands (`test` and `eval`)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from osmosis_ai.rollout.cli_utils import CLIError, load_agent_loop
from osmosis_ai.rollout.console import Console
from osmosis_ai.rollout.eval.common.dataset import DatasetReader
from osmosis_ai.rollout.eval.common.errors import (
    DatasetParseError,
    DatasetValidationError,
    ProviderError,
    SystemicProviderError,
)

# Mapping of LiteLLM provider prefixes to their expected environment variables.
_PROVIDER_ENV_KEYS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "fireworks_ai": "FIREWORKS_AI_API_KEY",
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


def load_mcp_agent(
    tools_path: str,
    quiet: bool,
    console: Console,
) -> Tuple[Optional["RolloutAgentLoop"], Optional[str]]:
    """Load an MCPAgentLoop from a tools directory.

    The directory must contain a ``main.py`` with a FastMCP instance and
    registered ``@mcp.tool()`` functions.
    """
    try:
        from osmosis_ai.rollout.mcp import MCPAgentLoop, MCPLoadError, load_mcp_server
    except ImportError:
        return None, (
            "MCP support requires fastmcp. "
            "Install it with: pip install osmosis-ai[mcp]"
        )

    if not quiet:
        console.print(f"Loading MCP tools: {tools_path}")

    try:
        mcp_server = load_mcp_server(tools_path)
    except MCPLoadError as e:
        return None, str(e)

    agent_loop = MCPAgentLoop(mcp_server)

    if not quiet:
        tool_names = [t.function.name for t in agent_loop.get_tools(None)]  # type: ignore[arg-type]
        console.print(f"  Discovered {len(tool_names)} tool(s): {', '.join(tool_names)}")

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


def _check_api_key(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
) -> Optional[str]:
    """Return an error message if the required API key is missing, else None."""
    if api_key or base_url:
        return None

    provider = model.split("/")[0] if "/" in model else "openai"
    env_var = _PROVIDER_ENV_KEYS.get(provider)
    if env_var is None:
        return None

    if not os.environ.get(env_var):
        return (
            f"Missing API key for provider '{provider}'. "
            f"Set the {env_var} environment variable or pass --api-key."
        )
    return None


def _first_line(message: str) -> str:
    """Return first non-empty line for concise error details."""
    for line in message.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return message.strip()


def _format_model_error(model: str, base_url: Optional[str], detail: str) -> str:
    """Format a concise model validation error with actionable guidance."""
    if base_url:
        return (
            "Invalid model/provider format for --base-url. "
            "Use an OpenAI-compatible model identifier with the 'openai/' prefix "
            "(for example: --model openai/<model-id>). "
            f"Received model='{model}'. Details: {detail}"
        )
    return (
        "Invalid LiteLLM model format. Use 'provider/model' "
        "(for example: openai/gpt-5-mini). "
        f"Received model='{model}'. Details: {detail}"
    )


def _check_model(model: str, base_url: Optional[str]) -> Optional[str]:
    """Return an error message if model format is invalid, else None."""
    candidate = model.strip()
    if not candidate:
        return "Model cannot be empty. Pass a value with --model."

    normalized_model = candidate if "/" in candidate else f"openai/{candidate}"
    provider, model_name = normalized_model.split("/", 1)
    if not provider.strip() or not model_name.strip():
        return _format_model_error(candidate, base_url, "missing provider or model name")

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

    previous_debug_state = getattr(litellm, "suppress_debug_info", None)
    if previous_debug_state is not None:
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
    finally:
        if previous_debug_state is not None:
            litellm.suppress_debug_info = previous_debug_state

    return None


def create_llm_client(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    quiet: bool,
    console: Console,
) -> Tuple[Optional["ExternalLLMClient"], Optional[str]]:
    """Initialize ExternalLLMClient with consistent messaging and errors."""
    if error := _check_model(model, base_url):
        return None, error

    if error := _check_api_key(model, api_key, base_url):
        return None, error

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


async def verify_llm_client(
    llm_client: "ExternalLLMClient",
    quiet: bool,
    console: Console,
) -> Optional[str]:
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
    "load_mcp_agent",
    "verify_llm_client",
]
