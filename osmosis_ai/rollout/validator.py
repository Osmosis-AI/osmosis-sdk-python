"""Static startup validation for v2 ``AgentWorkflow`` and ``Grader`` components.

Mirrors LocalBackend instantiation: ``workflow_cls(workflow_config)`` and
``grader_cls(grader_config)`` with no zero-argument fallback. Intended for use
before serving so invalid classes, async contracts, names, or instantiation
issues are reported together: ``validate_backend`` aggregates every applicable
error into one ``ValidationResult`` instead of stopping at the first failure.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
from osmosis_ai.rollout_v2.grader import Grader
from osmosis_ai.rollout_v2.types import AgentWorkflowConfig, GraderConfig

__all__ = [
    "ValidationError",
    "ValidationResult",
    "resolved_agent_name",
    "validate_backend",
]


@dataclass
class ValidationError:
    code: str
    message: str
    field: str | None = None


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationError] = field(default_factory=list)


def resolved_agent_name(
    workflow_cls: type,
    workflow_config: AgentWorkflowConfig | None,
) -> str:
    return (
        workflow_config.name.strip() if workflow_config else ""
    ) or workflow_cls.__name__


def validate_backend(
    workflow_cls: type,
    workflow_config: AgentWorkflowConfig | None,
    grader_cls: type[Grader] | type | None = None,
    grader_config: GraderConfig | None = None,
) -> ValidationResult:
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []

    workflow_ok = (
        isinstance(workflow_cls, type)
        and issubclass(workflow_cls, AgentWorkflow)
        and workflow_cls is not AgentWorkflow
    )
    if not workflow_ok:
        errors.append(
            ValidationError(
                code="INVALID_WORKFLOW_CLASS",
                message="workflow_cls must be a concrete subclass of AgentWorkflow",
                field="workflow_cls",
            )
        )
    else:
        if not asyncio.iscoroutinefunction(workflow_cls.run):
            errors.append(
                ValidationError(
                    code="WORKFLOW_RUN_NOT_ASYNC",
                    message="AgentWorkflow.run must be an async method",
                    field="workflow_cls.run",
                )
            )
        try:
            workflow_cls(workflow_config)
        except Exception as e:
            errors.append(
                ValidationError(
                    code="WORKFLOW_INIT_FAILED",
                    message=f"Failed to instantiate workflow: {e}",
                    field="workflow_cls",
                )
            )

    if isinstance(workflow_cls, type):
        resolved = resolved_agent_name(workflow_cls, workflow_config)
        if not (1 <= len(resolved) <= 256):
            errors.append(
                ValidationError(
                    code="INVALID_AGENT_NAME",
                    message="Resolved agent name must be between 1 and 256 characters",
                    field="name",
                )
            )

    if grader_cls is None:
        errors.append(
            ValidationError(
                code="MISSING_GRADER",
                message="grader_cls is required for LocalBackend validation",
                field="grader_cls",
            )
        )
    else:
        grader_ok = (
            isinstance(grader_cls, type)
            and issubclass(grader_cls, Grader)
            and grader_cls is not Grader
        )
        if not grader_ok:
            errors.append(
                ValidationError(
                    code="INVALID_GRADER_CLASS",
                    message="grader_cls must be a concrete subclass of Grader",
                    field="grader_cls",
                )
            )
        else:
            if not asyncio.iscoroutinefunction(grader_cls.grade):
                errors.append(
                    ValidationError(
                        code="GRADER_GRADE_NOT_ASYNC",
                        message="Grader.grade must be an async method",
                        field="grader_cls.grade",
                    )
                )
            try:
                grader_cls(grader_config)
            except Exception as e:
                errors.append(
                    ValidationError(
                        code="GRADER_INIT_FAILED",
                        message=f"Failed to instantiate grader: {e}",
                        field="grader_cls",
                    )
                )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
