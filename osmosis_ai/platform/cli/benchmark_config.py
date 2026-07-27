"""TOML config loading and validation for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


class _StrictSection(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class BenchmarkExperimentSection(_StrictSection):
    benchmark: str


class BenchmarkTasksSection(_StrictSection):
    categories: Any = None
    task_names: Any = None
    task_set: Any = None


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


class BenchmarkHostedModel(_StrictSection):
    type: Literal["hosted"]
    base_model: str
    checkpoint_name: str


BenchmarkModel = Annotated[
    BenchmarkProviderModel | BenchmarkEndpointModel | BenchmarkHostedModel,
    Field(discriminator="type"),
]


class BenchmarkAgentSection(_StrictSection):
    harness: str | None = None
    model: BenchmarkModel
    env: dict[str, str] = Field(default_factory=dict)


class BenchmarkExecutionSection(_StrictSection):
    attempts_per_task: Any = None
    max_concurrent_attempts: Any = None
    timeout_multiplier: Any = None
    max_retries: Any = None
    pass_threshold: Any = None
    judge_model: Any = None
    judge_api_key_secret: Any = None


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
        judge_secret = self.execution.judge_api_key_secret
        if isinstance(judge_secret, str) and judge_secret:
            names.append(judge_secret)
        return list(dict.fromkeys(names))


def _env_source_label(name: str, agent_index: int, agent_env: dict[str, str]) -> str:
    if name in agent_env:
        return f"[agents.env] of agent {agent_index}"
    return "[env]"


def _validate_secret_references(config: BenchmarkSubmitConfig, path: Path) -> None:
    """Validate secret record names and per-agent env collisions.

    The platform injects an agent's ``api_key_secret`` value (and any judge
    secret) as an env var of the same name into that agent's runtime env, so a
    literal env var with that name would be silently overwritten. Collisions
    are scoped per agent: one agent's secret name may still be another agent's
    literal env var.
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
        if (
            isinstance(model, BenchmarkProviderModel | BenchmarkEndpointModel)
            and model.api_key_secret in effective_env
        ):
            source = _env_source_label(model.api_key_secret, index, agent.env)
            raise CLIError(
                f"'{model.api_key_secret}' appears in {source} and as agent "
                f"{index}'s api_key_secret in {path}. The platform injects the "
                "secret value under that name; remove the env var or rename it."
            )
        if judge_name and judge_name in effective_env:
            source = _env_source_label(judge_name, index, agent.env)
            raise CLIError(
                f"'{judge_name}' appears in {source} and as "
                f"judge_api_key_secret in {path}. The platform injects the "
                "judge secret value under that name; remove the env var or "
                "rename it."
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
    for agent in config.agents:
        validate_env_var_keys(env=agent.env, path=path)
    _validate_secret_references(config, path)
    return config


__all__ = ["BenchmarkSubmitConfig", "load_benchmark_submit_config"]
