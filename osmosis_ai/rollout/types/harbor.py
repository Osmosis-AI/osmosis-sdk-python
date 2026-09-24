"""Configuration for the managed native Harbor gateway, independent of task source."""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class HarborHealthcheckConfig(BaseModel):
    """Portable subset of Harbor's native HealthcheckConfig for the bare CLI."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    command: str = Field(min_length=1)
    interval_sec: float = Field(default=5, gt=0)
    timeout_sec: float = Field(default=30, gt=0)
    start_period_sec: float = Field(default=0, ge=0)
    start_interval_sec: float = Field(default=5, gt=0)
    retries: int = Field(default=3, ge=1)


class HarborGatewayConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    agent: str = Field(default="opencode", pattern=r"^[a-z][a-z0-9-]*$")
    native_agent_kwargs: dict[str, Any] = Field(default_factory=dict)
    concurrency: int = Field(default=4, ge=1, le=256)
    environment_kwargs: dict[str, Any] = Field(default_factory=dict)
    environment_healthcheck: HarborHealthcheckConfig | None = None
    cleanup_successful_trials: bool = True
