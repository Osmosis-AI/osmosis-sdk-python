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

INLINE_NATIVE_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend as Native
from osmosis_ai.rollout.server import create_rollout_server as make_server


def main():
    return make_server(backend=Native())
"""

QUALIFIED_NATIVE_ENTRYPOINT = """\
import osmosis_ai.rollout.backend.native_harbor as native_harbor
import osmosis_ai.rollout.server as server


def main():
    backend = native_harbor.NativeHarborBackend()
    return server.create_rollout_server(backend=backend)
"""

HELPER_NATIVE_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def build_app():
    backend = NativeHarborBackend()
    return create_rollout_server(backend=backend)


def main():
    return build_app()
"""

HELPER_ARGUMENT_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def build_app(backend):
    return create_rollout_server(backend=backend)


def main():
    backend = NativeHarborBackend()
    return build_app(backend)
"""

DESTRUCTURED_NATIVE_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def main():
    ignored, backend = [object(), NativeHarborBackend()]
    return create_rollout_server(backend=backend)
"""

CALLED_LAMBDA_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


build_app = lambda backend: create_rollout_server(backend=backend)


def main():
    return build_app(NativeHarborBackend())
"""

UNCALLED_LAMBDA_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


unused_build_app = lambda: create_rollout_server(backend=NativeHarborBackend())


def main():
    return None
"""

VAR_POSITIONAL_HELPER_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def forward_positional(*args):
    return create_rollout_server(backend=args[0])


def main():
    arguments = (NativeHarborBackend(),)
    return forward_positional(*arguments)
"""

NEGATIVE_VAR_POSITIONAL_HELPER_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def forward_positional(*args):
    return create_rollout_server(backend=args[-1])


def main():
    arguments = (object(), NativeHarborBackend())
    return forward_positional(*arguments)
"""

VAR_KEYWORD_HELPER_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def forward_keywords(**kwargs):
    return create_rollout_server(**kwargs)


def main():
    options = {"backend": NativeHarborBackend()}
    return forward_keywords(**options)
"""

DEFAULT_ARGUMENT_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def build_app(backend=NativeHarborBackend()):
    return create_rollout_server(backend=backend)


def main():
    return build_app()
"""

OVERRIDDEN_DEFAULT_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def build_app(backend=NativeHarborBackend()):
    return create_rollout_server(backend=backend)


def main():
    return build_app(object())
"""

DYNAMIC_KEYWORDS_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def build_app(**kwargs):
    return create_rollout_server(**kwargs)


def options():
    return {"backend": NativeHarborBackend()}


def main():
    return build_app(**options())
"""

DEAD_HELPER_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def unused_build_app():
    return create_rollout_server(backend=NativeHarborBackend())


def main():
    return None
"""

IMPORT_ONLY_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
"""

UNWIRED_NATIVE_ENTRYPOINT = """\
from osmosis_ai.rollout.backend.native_harbor.backend import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def main():
    unused = NativeHarborBackend()
    return create_rollout_server(backend=object())
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

    def test_finds_inline_aliased_native_backend(self, tmp_path):
        _make_rollout(tmp_path, "native-rollout", INLINE_NATIVE_ENTRYPOINT)
        cls = discover_native_backend(
            rollout="native-rollout",
            entrypoint="main.py",
            workspace_directory=tmp_path,
        )
        assert cls is not None
        assert cls.__name__ == "NativeHarborBackend"

    def test_finds_qualified_native_backend(self, tmp_path):
        _make_rollout(tmp_path, "native-rollout", QUALIFIED_NATIVE_ENTRYPOINT)
        cls = discover_native_backend(
            rollout="native-rollout",
            entrypoint="main.py",
            workspace_directory=tmp_path,
        )
        assert cls is not None
        assert cls.__name__ == "NativeHarborBackend"

    @pytest.mark.parametrize(
        "source",
        [
            HELPER_NATIVE_ENTRYPOINT,
            HELPER_ARGUMENT_ENTRYPOINT,
            DESTRUCTURED_NATIVE_ENTRYPOINT,
            CALLED_LAMBDA_ENTRYPOINT,
            VAR_POSITIONAL_HELPER_ENTRYPOINT,
            NEGATIVE_VAR_POSITIONAL_HELPER_ENTRYPOINT,
            VAR_KEYWORD_HELPER_ENTRYPOINT,
            DEFAULT_ARGUMENT_ENTRYPOINT,
        ],
        ids=[
            "helper-constructs-backend",
            "main-passes-backend",
            "destructured-backend",
            "called-lambda",
            "var-positional-forwarding",
            "negative-var-positional-forwarding",
            "var-keyword-forwarding",
            "default-argument",
        ],
    )
    def test_follows_wired_helper_from_main(self, tmp_path, source):
        _make_rollout(tmp_path, "native-rollout", source)
        cls = discover_native_backend(
            rollout="native-rollout",
            entrypoint="main.py",
            workspace_directory=tmp_path,
        )
        assert cls is not None
        assert cls.__name__ == "NativeHarborBackend"

    @pytest.mark.parametrize(
        "source",
        [
            IMPORT_ONLY_ENTRYPOINT,
            UNWIRED_NATIVE_ENTRYPOINT,
            DEAD_HELPER_ENTRYPOINT,
            UNCALLED_LAMBDA_ENTRYPOINT,
            OVERRIDDEN_DEFAULT_ENTRYPOINT,
            DYNAMIC_KEYWORDS_ENTRYPOINT,
        ],
        ids=[
            "import-only",
            "constructed-but-not-wired",
            "unreachable-helper",
            "uncalled-lambda",
            "overridden-default",
            "dynamic-keywords",
        ],
    )
    def test_none_when_native_backend_is_not_wired(self, tmp_path, source):
        _make_rollout(tmp_path, "native-rollout", source)
        assert (
            discover_native_backend(
                rollout="native-rollout",
                entrypoint="main.py",
                workspace_directory=tmp_path,
            )
            is None
        )

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

    def test_import_only_native_backend_fails_preflight(self, tmp_path):
        _make_rollout(tmp_path, "native-rollout", IMPORT_ONLY_ENTRYPOINT)
        with pytest.raises(CLIError, match="preflight failed"):
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
