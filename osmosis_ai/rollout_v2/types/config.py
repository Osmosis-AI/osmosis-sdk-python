from pydantic import BaseModel, ConfigDict
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