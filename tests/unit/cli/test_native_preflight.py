"""Submit-preflight support for module-level native rollout apps.

Native rollouts have no Python ``AgentWorkflow`` or ``Grader``. Their entrypoint
therefore exposes an ``app`` built by ``create_rollout_server`` so preflight can
inspect the backend actually bound to that ASGI app without interpreting source.
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
from osmosis_ai.rollout.backend.native_harbor import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


backend = NativeHarborBackend()
app = create_rollout_server(backend=backend)
"""

NATIVE_SUBCLASS_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


class CustomNativeBackend(NativeHarborBackend):
    pass


app = create_rollout_server(backend=CustomNativeBackend())
"""

MAIN_ONLY_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def main():
    app = create_rollout_server(backend=NativeHarborBackend())
    return app
"""

IMPORT_ONLY_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor import NativeHarborBackend
"""

UNREACHABLE_NATIVE_APP_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


if False:
    app = create_rollout_server(backend=NativeHarborBackend())
"""

NON_NATIVE_APP_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.server import create_rollout_server


class OtherBackend(ExecutionBackend):
    async def execute(self, request, on_workflow_complete, on_grader_complete=None):
        return None


app = create_rollout_server(backend=OtherBackend())
"""

UNWIRED_NATIVE_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.backend.native_harbor import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


class OtherBackend(ExecutionBackend):
    async def execute(self, request, on_workflow_complete, on_grader_complete=None):
        return None


unused_native_backend = NativeHarborBackend()
app = create_rollout_server(backend=OtherBackend())
"""

EMPTY_ENTRYPOINT = "VALUE = 1\n"


def _make_rollout(workspace: Path, name: str, source: str) -> None:
    rollout_dir = workspace / "rollouts" / name
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "main.py").write_text(source, encoding="utf-8")


class TestDiscoverNativeBackend:
    @pytest.mark.parametrize(
        "source, expected_name",
        [
            (NATIVE_ENTRYPOINT, "NativeHarborBackend"),
            (NATIVE_SUBCLASS_ENTRYPOINT, "CustomNativeBackend"),
        ],
        ids=["native", "native-subclass"],
    )
    def test_finds_backend_bound_to_module_app(
        self, tmp_path: Path, source: str, expected_name: str
    ) -> None:
        _make_rollout(tmp_path, "native-rollout", source)

        cls = discover_native_backend(
            rollout="native-rollout",
            entrypoint="main.py",
            workspace_directory=tmp_path,
        )

        assert cls is not None
        assert cls.__name__ == expected_name

    @pytest.mark.parametrize(
        "source",
        [
            MAIN_ONLY_ENTRYPOINT,
            IMPORT_ONLY_ENTRYPOINT,
            UNREACHABLE_NATIVE_APP_ENTRYPOINT,
            NON_NATIVE_APP_ENTRYPOINT,
            UNWIRED_NATIVE_ENTRYPOINT,
            EMPTY_ENTRYPOINT,
        ],
        ids=[
            "main-only",
            "import-only",
            "unreachable-native-app",
            "non-native-app",
            "unwired-native",
            "empty",
        ],
    )
    def test_none_without_module_level_native_app(
        self, tmp_path: Path, source: str
    ) -> None:
        _make_rollout(tmp_path, "native-rollout", source)

        assert (
            discover_native_backend(
                rollout="native-rollout",
                entrypoint="main.py",
                workspace_directory=tmp_path,
            )
            is None
        )

    def test_none_on_missing_entrypoint(self, tmp_path: Path) -> None:
        _make_rollout(tmp_path, "native-rollout", NATIVE_ENTRYPOINT)

        assert (
            discover_native_backend(
                rollout="native-rollout",
                entrypoint="nope.py",
                workspace_directory=tmp_path,
            )
            is None
        )


class TestValidateRolloutBackendNative:
    def test_native_app_passes_without_grader(self, tmp_path: Path) -> None:
        _make_rollout(tmp_path, "native-rollout", NATIVE_ENTRYPOINT)

        validate_rollout_backend(
            workspace_directory=tmp_path,
            rollout="native-rollout",
            entrypoint="main.py",
            command_label="Test",
        )

    @pytest.mark.parametrize(
        "source",
        [IMPORT_ONLY_ENTRYPOINT, MAIN_ONLY_ENTRYPOINT, NON_NATIVE_APP_ENTRYPOINT],
        ids=["import-only", "main-only", "non-native-app"],
    )
    def test_non_native_contract_fails_preflight(
        self, tmp_path: Path, source: str
    ) -> None:
        _make_rollout(tmp_path, "native-rollout", source)

        with pytest.raises(CLIError, match="preflight failed"):
            validate_rollout_backend(
                workspace_directory=tmp_path,
                rollout="native-rollout",
                entrypoint="main.py",
                command_label="Test",
            )
