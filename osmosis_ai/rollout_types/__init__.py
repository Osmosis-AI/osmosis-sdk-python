"""Lightweight rollout type definitions.

This package contains only Pydantic models and enums — no runtime
dependencies beyond ``pydantic``.  Import from here when you need
rollout types without pulling in the full rollout SDK.

    from osmosis_ai.rollout_types import RolloutSample, RolloutStatus
"""

from osmosis_ai.rollout_types.config import (
    AgentWorkflowConfig,
    BaseConfig,
    ConcurrencyConfig,
    GraderConfig,
)
from osmosis_ai.rollout_types.protocol import (
    GraderCompleteRequest,
    GraderInitRequest,
    GraderInitResponse,
    GraderStatus,
    RolloutCompleteRequest,
    RolloutInitRequest,
    RolloutInitResponse,
)
from osmosis_ai.rollout_types.sample import (
    ExecutionRequest,
    ExecutionResult,
    MessageDict,
    MultiTurnMode,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)

__all__ = [
    "AgentWorkflowConfig",
    "BaseConfig",
    "ConcurrencyConfig",
    "ExecutionRequest",
    "ExecutionResult",
    "GraderCompleteRequest",
    "GraderConfig",
    "GraderInitRequest",
    "GraderInitResponse",
    "GraderStatus",
    "MessageDict",
    "MultiTurnMode",
    "RolloutCompleteRequest",
    "RolloutErrorCategory",
    "RolloutInitRequest",
    "RolloutInitResponse",
    "RolloutSample",
    "RolloutStatus",
]
