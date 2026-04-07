"""Tests for hard-fail when multiple workflow/grader/config candidates are found."""

import sys
import types
from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.eval.common.cli import _resolve_workflow, auto_discover_grader
from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
from osmosis_ai.rollout_v2.context import AgentWorkflowContext
from osmosis_ai.rollout_v2.grader import Grader
from osmosis_ai.rollout_v2.types import AgentWorkflowConfig, GraderConfig


@pytest.fixture
def tmp_rollout_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """chdir to tmp_path and create rollouts/<name> for _ensure_rollout_on_path."""
    monkeypatch.chdir(tmp_path)
    rollout = "r"
    (tmp_path / "rollouts" / rollout).mkdir(parents=True)
    yield rollout


def _make_workflow_module_two_classes(name: str) -> types.ModuleType:
    class W1(AgentWorkflow):
        async def run(self, ctx: AgentWorkflowContext) -> None:
            del ctx

    class W2(AgentWorkflow):
        async def run(self, ctx: AgentWorkflowContext) -> None:
            del ctx

    mod = types.ModuleType(name)
    mod.W1 = W1
    mod.W2 = W2
    return mod


def test_resolve_workflow_multiple_agent_workflow_subclasses(
    tmp_rollout_layout: str,
):
    mod_name = "_test_two_wf"
    mod = _make_workflow_module_two_classes(mod_name)
    sys.modules[mod_name] = mod
    try:
        with pytest.raises(CLIError) as exc_info:
            _resolve_workflow(
                rollout=tmp_rollout_layout,
                entrypoint="_test_two_wf.py",
            )
        assert "Multiple AgentWorkflow" in str(exc_info.value)
    finally:
        del sys.modules[mod_name]


def test_auto_discover_grader_multiple_grader_config_instances():
    from osmosis_ai.rollout_v2.context import GraderContext

    mod_name = "_test_two_gc"

    class OnlyGrader(Grader):
        async def grade(self, ctx: GraderContext):
            del ctx

    cfg_x = GraderConfig(name="x")
    cfg_y = GraderConfig(name="y")

    mod = types.ModuleType(mod_name)
    mod.OnlyGrader = OnlyGrader
    mod.cfg_x = cfg_x
    mod.cfg_y = cfg_y
    sys.modules[mod_name] = mod
    try:
        with pytest.raises(CLIError) as exc_info:
            auto_discover_grader("_test_two_gc.py")
        msg = str(exc_info.value)
        assert "Multiple GraderConfig" in msg
    finally:
        del sys.modules[mod_name]


def test_resolve_workflow_same_workflow_class_two_bindings_not_ambiguous(
    tmp_rollout_layout: str,
):
    """Re-exporting the same class under two names must not count as two workflows."""

    class OnlyWorkflow(AgentWorkflow):
        async def run(self, ctx: AgentWorkflowContext) -> None:
            del ctx

    mod_name = "_test_wf_alias"
    mod = types.ModuleType(mod_name)
    mod.OnlyWorkflow = OnlyWorkflow
    mod.OnlyWorkflowAlias = OnlyWorkflow
    sys.modules[mod_name] = mod
    try:
        wf_cls, cfg = _resolve_workflow(
            rollout=tmp_rollout_layout,
            entrypoint="_test_wf_alias.py",
        )
        assert wf_cls is OnlyWorkflow
        assert cfg is None
    finally:
        del sys.modules[mod_name]


def test_auto_discover_grader_same_grader_config_two_bindings_ok():
    from osmosis_ai.rollout_v2.context import GraderContext

    mod_name = "_test_gc_alias"

    class G(Grader):
        async def grade(self, ctx: GraderContext):
            del ctx

    cfg = GraderConfig(name="solo")
    mod = types.ModuleType(mod_name)
    mod.G = G
    mod.grader_cfg = cfg
    mod.grader_cfg_dup = cfg
    sys.modules[mod_name] = mod
    try:
        grader_cls, grader_config = auto_discover_grader("_test_gc_alias.py")
        assert grader_cls is G
        assert grader_config is cfg
    finally:
        del sys.modules[mod_name]


def test_auto_discover_grader_multiple_grader_subclasses():
    from osmosis_ai.rollout_v2.context import GraderContext

    mod_name = "_test_two_gr"

    class G1(Grader):
        async def grade(self, ctx: GraderContext):
            del ctx

    class G2(Grader):
        async def grade(self, ctx: GraderContext):
            del ctx

    mod = types.ModuleType(mod_name)
    mod.G1 = G1
    mod.G2 = G2
    sys.modules[mod_name] = mod
    try:
        with pytest.raises(CLIError) as exc_info:
            auto_discover_grader("_test_two_gr.py")
        assert "Multiple Grader" in str(exc_info.value)
    finally:
        del sys.modules[mod_name]


def test_resolve_workflow_multiple_agent_workflow_config_instances(
    tmp_rollout_layout: str,
):
    class OnlyWorkflow(AgentWorkflow):
        async def run(self, ctx: AgentWorkflowContext) -> None:
            del ctx

    cfg_a = AgentWorkflowConfig(name="a")
    cfg_b = AgentWorkflowConfig(name="b")

    mod_name = "_test_two_cfg"
    mod = types.ModuleType(mod_name)
    mod.OnlyWorkflow = OnlyWorkflow
    mod.cfg_a = cfg_a
    mod.cfg_b = cfg_b
    sys.modules[mod_name] = mod
    try:
        with pytest.raises(CLIError) as exc_info:
            _resolve_workflow(
                rollout=tmp_rollout_layout,
                entrypoint="_test_two_cfg.py",
            )
        assert "Multiple AgentWorkflowConfig" in str(exc_info.value)
    finally:
        del sys.modules[mod_name]
