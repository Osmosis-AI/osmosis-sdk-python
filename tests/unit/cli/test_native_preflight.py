"""Submit-preflight support for native rollouts.

A native rollout entrypoint declares a ``NativeHarborBackend`` (no AgentWorkflow)
and derives its reward from the harbor task's own verifier, so it carries no
Python Grader. ``validate_rollout_backend`` must therefore detect it and skip the
Grader requirement instead of failing with "No AgentWorkflow subclass found".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.eval.common.cli import discover_native_backend
from osmosis_ai.platform.cli.workspace_directory_contract import (
    validate_rollout_backend,
)

NATIVE_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def main():
    backend = NativeHarborBackend()
    return create_rollout_server(backend=backend)
"""

# No AgentWorkflow and no NativeHarborBackend -> a genuinely broken entrypoint.
EMPTY_ENTRYPOINT = "VALUE = 1\n"


def _make_rollout(workspace: Path, name: str, source: str) -> None:
    rollout_dir = workspace / "rollouts" / name
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "main.py").write_text(source, encoding="utf-8")


class TestDiscoverNativeBackend:
    def test_finds_native_backend(self, tmp_path):
        _make_rollout(tmp_path, "native-rollout", NATIVE_ENTRYPOINT)
        cls = discover_native_backend(
            rollout="native-rollout",
            entrypoint="main.py",
            workspace_directory=tmp_path,
        )
        assert cls is not None
        assert cls.__name__ == "NativeHarborBackend"

    def test_none_for_non_native(self, tmp_path):
        _make_rollout(tmp_path, "empty-rollout", EMPTY_ENTRYPOINT)
        assert (
            discover_native_backend(
                rollout="empty-rollout",
                entrypoint="main.py",
                workspace_directory=tmp_path,
            )
            is None
        )

    def test_none_on_missing_entrypoint(self, tmp_path):
        _make_rollout(tmp_path, "native-rollout", NATIVE_ENTRYPOINT)
        # A load failure (wrong filename) is swallowed to None; the workflow path
        # surfaces the real error.
        assert (
            discover_native_backend(
                rollout="native-rollout",
                entrypoint="nope.py",
                workspace_directory=tmp_path,
            )
            is None
        )


class TestValidateRolloutBackendNative:
    def test_native_passes_without_grader(self, tmp_path):
        _make_rollout(tmp_path, "native-rollout", NATIVE_ENTRYPOINT)
        # Should NOT raise: native rollouts need no Python Grader.
        validate_rollout_backend(
            workspace_directory=tmp_path,
            rollout="native-rollout",
            entrypoint="main.py",
            command_label="Test",
        )

    def test_neither_workflow_nor_native_raises(self, tmp_path):
        _make_rollout(tmp_path, "empty-rollout", EMPTY_ENTRYPOINT)
        with pytest.raises(CLIError, match="preflight failed"):
            validate_rollout_backend(
                workspace_directory=tmp_path,
                rollout="empty-rollout",
                entrypoint="main.py",
                command_label="Test",
            )
