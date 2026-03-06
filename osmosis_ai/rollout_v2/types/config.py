from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

class BaseConfig(BaseModel):
    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for flexibility
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    name: str
    description: Optional[str] = None

class AgentWorkflowConfig(BaseConfig):
    ...

class GraderConfig(BaseConfig):
    ...


class WorkflowConcurrencyConfig(BaseModel):
    max_concurrent: int | None = Field(default=None, ge=1)


class GraderConcurrencyConfig(BaseModel):
    max_concurrent: int | None = Field(default=None, ge=1)


class RolloutServerConfig(BaseModel):
    workflow: WorkflowConcurrencyConfig = Field(
        default_factory=WorkflowConcurrencyConfig
    )
    grader: GraderConcurrencyConfig = Field(default_factory=GraderConcurrencyConfig)