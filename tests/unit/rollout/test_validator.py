"""Tests for rollout_v2 static backend validation."""

from __future__ import annotations

from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
from osmosis_ai.rollout_v2.context import AgentWorkflowContext, GraderContext
from osmosis_ai.rollout_v2.grader import Grader
from osmosis_ai.rollout_v2.types import (
    AgentWorkflowConfig,
    ConcurrencyConfig,
    GraderConfig,
)
from osmosis_ai.rollout_v2.validator import (
    ValidationError,
    ValidationResult,
    validate_backend,
)


def _wf_config(name: str = "wf") -> AgentWorkflowConfig:
    return AgentWorkflowConfig(name=name, concurrency=ConcurrencyConfig())


def _grader_config(name: str = "gr") -> GraderConfig:
    return GraderConfig(name=name, concurrency=ConcurrencyConfig())


class ValidWorkflow(AgentWorkflow[AgentWorkflowConfig]):
    async def run(self, ctx: AgentWorkflowContext[AgentWorkflowConfig]) -> None:
        return None


class ValidGrader(Grader):
    async def grade(self, ctx: GraderContext) -> None:
        return None


class NotAWorkflow:
    pass


class SyncRunWorkflow(AgentWorkflow[AgentWorkflowConfig]):
    def run(self, ctx: AgentWorkflowContext[AgentWorkflowConfig]) -> None:
        return None


class InitFailWorkflow(AgentWorkflow[AgentWorkflowConfig]):
    def __init__(self, config: AgentWorkflowConfig | None = None) -> None:
        super().__init__(config)
        raise RuntimeError("workflow init failed")

    async def run(self, ctx: AgentWorkflowContext[AgentWorkflowConfig]) -> None:
        return None


class NotAGrader:
    pass


class SyncGradeGrader(Grader):
    def grade(self, ctx: GraderContext) -> None:
        return None


class InitFailGrader(Grader):
    def __init__(self, config: GraderConfig | None = None) -> None:
        super().__init__(config)
        raise RuntimeError("grader init failed")

    async def grade(self, ctx: GraderContext) -> None:
        return None


class WhitespaceNameWorkflow(AgentWorkflow[AgentWorkflowConfig]):
    async def run(self, ctx: AgentWorkflowContext[AgentWorkflowConfig]) -> None:
        return None


def _codes(result: ValidationResult) -> list[str]:
    return [e.code for e in result.errors]


def test_valid_workflow_and_grader():
    result = validate_backend(
        ValidWorkflow,
        _wf_config("agent"),
        ValidGrader,
        _grader_config("grader"),
    )
    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []


def test_invalid_workflow_class():
    result = validate_backend(NotAWorkflow, _wf_config(), ValidGrader, _grader_config())
    assert result.valid is False
    assert "INVALID_WORKFLOW_CLASS" in _codes(result)


def test_agent_workflow_abc_rejected():
    result = validate_backend(
        AgentWorkflow, _wf_config(), ValidGrader, _grader_config()
    )
    assert result.valid is False
    assert "INVALID_WORKFLOW_CLASS" in _codes(result)


def test_sync_run_rejected():
    result = validate_backend(
        SyncRunWorkflow, _wf_config(), ValidGrader, _grader_config()
    )
    assert result.valid is False
    assert "WORKFLOW_RUN_NOT_ASYNC" in _codes(result)


def test_workflow_init_failure():
    result = validate_backend(
        InitFailWorkflow, _wf_config(), ValidGrader, _grader_config()
    )
    assert result.valid is False
    assert "WORKFLOW_INIT_FAILED" in _codes(result)


def test_invalid_grader_class():
    result = validate_backend(ValidWorkflow, _wf_config(), NotAGrader, _grader_config())
    assert result.valid is False
    assert "INVALID_GRADER_CLASS" in _codes(result)


def test_grader_abc_rejected():
    result = validate_backend(ValidWorkflow, _wf_config(), Grader, _grader_config())
    assert result.valid is False
    assert "INVALID_GRADER_CLASS" in _codes(result)


def test_sync_grade_rejected():
    result = validate_backend(
        ValidWorkflow, _wf_config(), SyncGradeGrader, _grader_config()
    )
    assert result.valid is False
    assert "GRADER_GRADE_NOT_ASYNC" in _codes(result)


def test_grader_init_failure():
    result = validate_backend(
        ValidWorkflow, _wf_config(), InitFailGrader, _grader_config()
    )
    assert result.valid is False
    assert "GRADER_INIT_FAILED" in _codes(result)


def test_missing_grader():
    result = validate_backend(ValidWorkflow, _wf_config(), None, _grader_config())
    assert result.valid is False
    assert "MISSING_GRADER" in _codes(result)


def test_whitespace_workflow_name_falls_back_to_class_name():
    cfg = _wf_config("   ")
    result = validate_backend(
        WhitespaceNameWorkflow,
        cfg,
        ValidGrader,
        _grader_config(),
    )
    assert result.valid is True
    assert "INVALID_AGENT_NAME" not in _codes(result)


def test_too_long_agent_name():
    long_name = "x" * 257
    result = validate_backend(
        ValidWorkflow,
        _wf_config(long_name),
        ValidGrader,
        _grader_config(),
    )
    assert result.valid is False
    assert "INVALID_AGENT_NAME" in _codes(result)


def test_explicit_config_name_valid():
    result = validate_backend(
        ValidWorkflow,
        _wf_config("my-explicit-agent"),
        ValidGrader,
        _grader_config(),
    )
    assert result.valid is True
    assert "INVALID_AGENT_NAME" not in _codes(result)


def test_fallback_name_when_config_none():
    result = validate_backend(ValidWorkflow, None, ValidGrader, _grader_config())
    assert result.valid is True
    assert "INVALID_AGENT_NAME" not in _codes(result)


def test_multiple_errors_collected():
    result = validate_backend(
        SyncRunWorkflow,
        _wf_config(),
        InitFailGrader,
        _grader_config(),
    )
    assert result.valid is False
    codes = set(_codes(result))
    assert "WORKFLOW_RUN_NOT_ASYNC" in codes
    assert "GRADER_INIT_FAILED" in codes
    assert len(result.errors) >= 2


def test_validation_error_dataclass_fields():
    err = ValidationError(code="X", message="m", field="f")
    assert err.code == "X"
    assert err.message == "m"
    assert err.field == "f"
    err2 = ValidationError(code="Y", message="n")
    assert err2.field is None


def test_validation_result_structure():
    r = ValidationResult(valid=False, errors=[], warnings=[])
    assert r.valid is False
    assert r.errors == []
    assert r.warnings == []
