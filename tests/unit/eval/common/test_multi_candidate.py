"""Tests for hard-fail when multiple workflow/grader/config candidates are found."""

import sys
import textwrap
import types
from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.eval.common.cli import _resolve_workflow, auto_discover_grader
from osmosis_ai.rollout.grader import Grader
from osmosis_ai.rollout.types import GraderConfig


@pytest.fixture
def tmp_rollout_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """chdir to tmp_path and create rollouts/<name> for _ensure_rollout_on_path."""
    monkeypatch.chdir(tmp_path)
    rollout = "r"
    (tmp_path / "rollouts" / rollout).mkdir(parents=True)
    yield rollout


def _write_entrypoint(tmp_path: Path, rollout: str, filename: str, code: str) -> None:
    """Write a Python entrypoint file under rollouts/<rollout>/."""
    ep = tmp_path / "rollouts" / rollout / filename
    ep.write_text(textwrap.dedent(code), encoding="utf-8")


def test_resolve_workflow_multiple_agent_workflow_subclasses(
    tmp_rollout_layout: str,
    tmp_path: Path,
):
    _write_entrypoint(
        tmp_path,
        tmp_rollout_layout,
        "two_wf.py",
        """\
        from osmosis_ai.rollout.agent_workflow import AgentWorkflow

        class W1(AgentWorkflow):
            async def run(self, ctx):
                pass

        class W2(AgentWorkflow):
            async def run(self, ctx):
                pass
        """,
    )
    with pytest.raises(CLIError) as exc_info:
        _resolve_workflow(
            rollout=tmp_rollout_layout,
            entrypoint="two_wf.py",
        )
    assert "Multiple AgentWorkflow" in str(exc_info.value)


def test_auto_discover_grader_multiple_grader_config_instances():
    from osmosis_ai.rollout.context import GraderContext

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
            auto_discover_grader(mod_name)
        msg = str(exc_info.value)
        assert "Multiple GraderConfig" in msg
    finally:
        del sys.modules[mod_name]


def test_resolve_workflow_same_workflow_class_two_bindings_not_ambiguous(
    tmp_rollout_layout: str,
    tmp_path: Path,
):
    """Re-exporting the same class under two names must not count as two workflows."""
    _write_entrypoint(
        tmp_path,
        tmp_rollout_layout,
        "wf_alias.py",
        """\
        from osmosis_ai.rollout.agent_workflow import AgentWorkflow

        class OnlyWorkflow(AgentWorkflow):
            async def run(self, ctx):
                pass

        OnlyWorkflowAlias = OnlyWorkflow
        """,
    )
    wf_cls, cfg, _mod_name = _resolve_workflow(
        rollout=tmp_rollout_layout,
        entrypoint="wf_alias.py",
    )
    assert wf_cls is not None
    assert wf_cls.__name__ == "OnlyWorkflow"
    assert cfg is None


def test_auto_discover_grader_same_grader_config_two_bindings_ok():
    from osmosis_ai.rollout.context import GraderContext

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
        grader_cls, grader_config = auto_discover_grader(mod_name)
        assert grader_cls is G
        assert grader_config is cfg
    finally:
        del sys.modules[mod_name]


def test_auto_discover_grader_multiple_grader_subclasses():
    from osmosis_ai.rollout.context import GraderContext

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
            auto_discover_grader(mod_name)
        assert "Multiple Grader" in str(exc_info.value)
    finally:
        del sys.modules[mod_name]


def test_resolve_workflow_multiple_agent_workflow_config_instances(
    tmp_rollout_layout: str,
    tmp_path: Path,
):
    _write_entrypoint(
        tmp_path,
        tmp_rollout_layout,
        "two_cfg.py",
        """\
        from osmosis_ai.rollout.agent_workflow import AgentWorkflow
        from osmosis_ai.rollout.types import AgentWorkflowConfig

        class OnlyWorkflow(AgentWorkflow):
            async def run(self, ctx):
                pass

        cfg_a = AgentWorkflowConfig(name="a")
        cfg_b = AgentWorkflowConfig(name="b")
        """,
    )
    with pytest.raises(CLIError) as exc_info:
        _resolve_workflow(
            rollout=tmp_rollout_layout,
            entrypoint="two_cfg.py",
        )
    assert "Multiple AgentWorkflowConfig" in str(exc_info.value)
