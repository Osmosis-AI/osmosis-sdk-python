"""TOML config loading and validation for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.shared_config import (
    SECRET_NAME_RE,
    config_issues_error,
    read_toml_file,
    read_toml_table,
    validate_env_var_keys,
    validation_issue_to_config_issue,
)

_BENCHMARK_CONFIG_LABEL = "benchmark"
_HARNESS_API_KEY_ENV = {
    "cursor-cli": "CURSOR_API_KEY",
    "mini-swe-agent": "MSWEA_API_KEY",
}
_RESERVED_MODEL_API_KEY_SECRET_NAMES = frozenset(
    {
        "HF_TOKEN",
        "DAYTONA_API_KEY",
        "DAYTONA_API_URL",
        "SKYPILOT_SERVICE_ACCOUNT_TOKEN",
        "SKYPILOT_API_SERVER_ENDPOINT",
    }
)


class _StrictSection(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class BenchmarkExperimentSection(_StrictSection):
    benchmark: str


_NonEmptyTaskSelector = Annotated[str, Field(min_length=1, pattern=r"\S")]


class BenchmarkTasksSection(_StrictSection):
    categories: list[_NonEmptyTaskSelector] | None = None
    task_names: list[_NonEmptyTaskSelector] | None = None
    task_set: Literal["parity"] | None = None


class BenchmarkProviderModel(_StrictSection):
    type: Literal["provider"]
    model: str
    api_key_secret: str


class BenchmarkEndpointModel(_StrictSection):
    type: Literal["endpoint"]
    base_url: str
    model: str
    api_key_secret: str
    extra_headers: dict[str, str] | None = None

    @field_validator("extra_headers")
    @classmethod
    def reject_authorization_header(
        cls, extra_headers: dict[str, str] | None
    ) -> dict[str, str] | None:
        if extra_headers is not None and any(
            name.casefold() == "authorization" for name in extra_headers
        ):
            raise ValueError(
                "extra_headers must not include an Authorization header; "
                "use api_key_secret for endpoint authentication"
            )
        return extra_headers


class BenchmarkHostedModel(_StrictSection):
    type: Literal["hosted"]
    base_model: str
    lora_model_name: str


BenchmarkModel = Annotated[
    BenchmarkProviderModel | BenchmarkEndpointModel | BenchmarkHostedModel,
    Field(discriminator="type"),
]


class BenchmarkAgentSection(_StrictSection):
    harness: str | None = None
    harness_api_key_secret: str | None = None
    model: BenchmarkModel
    env: dict[str, str] = Field(default_factory=dict)


class BenchmarkExecutionSection(_StrictSection):
    attempts_per_task: Any = None
    max_concurrent_attempts: Any = None
    timeout_multiplier: Any = None
    max_retries: Any = None
    pass_threshold: Any = None
    judge_model: Any = None
    judge_api_key_secret: str | None = None


class BenchmarkSubmitConfig(_StrictSection):
    """Parsed benchmark run TOML configuration."""

    experiment: BenchmarkExperimentSection
    tasks: BenchmarkTasksSection = Field(default_factory=BenchmarkTasksSection)
    agents: list[BenchmarkAgentSection] = Field(min_length=1, max_length=8)
    execution: BenchmarkExecutionSection = Field(
        default_factory=BenchmarkExecutionSection
    )
    env: dict[str, str] = Field(default_factory=dict)

    @property
    def experiment_config(self) -> dict[str, Any]:
        return self.experiment.model_dump()

    @property
    def tasks_config(self) -> dict[str, Any]:
        return self.tasks.model_dump(exclude_none=True)

    @property
    def agents_config(self) -> list[dict[str, Any]]:
        return [agent.model_dump(exclude_none=True) for agent in self.agents]

    @property
    def execution_config(self) -> dict[str, Any]:
        return self.execution.model_dump(exclude_none=True)

    @property
    def required_secrets(self) -> list[str]:
        names = [
            agent.model.api_key_secret
            for agent in self.agents
            if isinstance(agent.model, BenchmarkProviderModel | BenchmarkEndpointModel)
        ]
        names.extend(
            agent.harness_api_key_secret
            for agent in self.agents
            if agent.harness_api_key_secret is not None
        )
        judge_secret = self.execution.judge_api_key_secret
        if isinstance(judge_secret, str):
            names.append(judge_secret)
        # HLE's managed adapter reads the dataset through a fixed Platform
        # secret, even though the name is not repeated in the submit config.
        if self.experiment.benchmark.strip() == "HLE":
            names.append("HF_TOKEN")
        return list(dict.fromkeys(names))


def _env_source_label(name: str, agent_index: int, agent_env: dict[str, str]) -> str:
    if name in agent_env:
        return f"[agents.env] of agent {agent_index}"
    return "[env]"


def _validate_secret_references(config: BenchmarkSubmitConfig, path: Path) -> None:
    """Validate secret record names and per-agent env collisions.

    The platform injects an agent model's ``api_key_secret`` value (and any
    judge secret) as an env var of the same name into that agent's runtime env,
    so a literal env var with that name would be silently overwritten.
    Collisions are scoped per agent: one agent's secret name may still be
    another agent's literal env var. Model secrets also cannot use names the
    runner removes before model-key aliasing. Harness credentials travel
    through a separate reserved channel; a credentialed harness must reference
    a secret record named exactly for the variable it reads, and cannot also
    set that name as a literal env var. ``HF_TOKEN`` is always runner-reserved
    as a literal env.
    """
    for name in config.required_secrets:
        if not SECRET_NAME_RE.match(name):
            raise CLIError(
                f"Invalid secret name '{name}' in {path}: use uppercase "
                "letters, digits, and underscores, starting with a letter "
                "(e.g. MY_SECRET). Must match ^[A-Z][A-Z0-9_]*$."
            )

    judge_secret = config.execution.judge_api_key_secret
    judge_name = (
        judge_secret if isinstance(judge_secret, str) and judge_secret else None
    )
    for index, agent in enumerate(config.agents, start=1):
        effective_env = {**config.env, **agent.env}
        model = agent.model
        if isinstance(model, BenchmarkProviderModel | BenchmarkEndpointModel):
            if model.api_key_secret in _RESERVED_MODEL_API_KEY_SECRET_NAMES:
                raise CLIError(
                    f"Agent {index}'s api_key_secret '{model.api_key_secret}' "
                    f"in {path} uses a name reserved by the benchmark runner. "
                    "Store the model credential under a different Platform "
                    "secret record name."
                )
            if model.api_key_secret in effective_env:
                source = _env_source_label(model.api_key_secret, index, agent.env)
                raise CLIError(
                    f"'{model.api_key_secret}' appears in {source} and as agent "
                    f"{index}'s api_key_secret in {path}. The platform injects "
                    "the secret value under that name; remove the env var or "
                    "rename it."
                )
        if judge_name and judge_name in effective_env:
            source = _env_source_label(judge_name, index, agent.env)
            raise CLIError(
                f"'{judge_name}' appears in {source} and as "
                f"judge_api_key_secret in {path}. The platform injects the "
                "judge secret value under that name; remove the env var or "
                "rename it."
            )
        if "HF_TOKEN" in effective_env:
            source = _env_source_label("HF_TOKEN", index, agent.env)
            raise CLIError(
                f"'HF_TOKEN' appears in {source} but is reserved by the "
                f"benchmark runner in {path}. The runner removes this literal "
                "env var before starting the agent; remove it from the config. "
                "For HLE, store the dataset credential in the HF_TOKEN Platform "
                "secret record instead."
            )
        harness_env_name = _HARNESS_API_KEY_ENV.get(agent.harness or "")
        if harness_env_name and harness_env_name in effective_env:
            source = _env_source_label(harness_env_name, index, agent.env)
            raise CLIError(
                f"'{harness_env_name}' appears in {source} but is managed by "
                f"agent {index}'s {agent.harness} harness API key secret in "
                f"{path}. Remove the env var and set harness_api_key_secret "
                "to a Platform secret record name."
            )
        if harness_env_name and agent.harness_api_key_secret is None:
            raise CLIError(
                f"Agent {index}'s {agent.harness} harness requires "
                f"harness_api_key_secret in {path}. Set it to the name of a "
                f"Platform secret record containing {harness_env_name}."
            )
        if harness_env_name and agent.harness_api_key_secret != harness_env_name:
            raise CLIError(
                f"Agent {index}'s harness_api_key_secret "
                f"'{agent.harness_api_key_secret}' in {path} does not match "
                f"the variable the {agent.harness} harness reads. Store the "
                f"credential in a Platform secret record named exactly "
                f"{harness_env_name} and reference that name."
            )


def load_benchmark_submit_config(path: Path) -> BenchmarkSubmitConfig:
    """Load and validate TOML config for benchmark run submit."""
    raw = read_toml_file(path)
    read_toml_table(raw, "experiment", path, required=True)
    if "agents" not in raw:
        raise CLIError(f"Missing [[agents]] section in {path}")
    if not isinstance(raw["agents"], list):
        raise CLIError(f"[[agents]] must be an array of tables in {path}")

    try:
        config = BenchmarkSubmitConfig.model_validate(raw)
    except ValidationError as exc:
        raise config_issues_error(
            issues=[
                validation_issue_to_config_issue(error=error, section_name="")
                for error in exc.errors()
            ],
            config_label=_BENCHMARK_CONFIG_LABEL,
        ) from exc

    validate_env_var_keys(env=config.env, path=path)
    for index, agent in enumerate(config.agents, start=1):
        validate_env_var_keys(
            env=agent.env,
            path=path,
            source_label=f"agent {index}'s [agents.env]",
        )
    _validate_secret_references(config, path)
    return config


__all__ = ["BenchmarkSubmitConfig", "load_benchmark_submit_config"]
