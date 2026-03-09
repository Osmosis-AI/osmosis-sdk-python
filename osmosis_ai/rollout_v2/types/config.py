from pydantic import BaseModel, ConfigDict, Field


class ConcurrencyConfig(BaseModel):
    max_concurrent: int | None = Field(default=None, ge=1)


class BaseConfig(BaseModel):
    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for flexibility
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    name: str
    description: str | None = None


class AgentWorkflowConfig(BaseConfig):
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)


class GraderConfig(BaseConfig):
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
