from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import osmosis_ai.platform.cli.shared_submit as shared_submit_module
from osmosis_ai.cli.errors import CLIError, CLIErrorCode
from osmosis_ai.platform.api.models import WorkspaceSummary
from osmosis_ai.platform.workspace_scope import override_workspace_name


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        workspace_directory=Path("/repo"),
        git_identity="acme/rollouts",
        repo_url="https://github.com/acme/rollouts.git",
        credentials=object(),
    )


def test_source_submit_workspace_requires_absolute_config_path() -> None:
    with (
        override_workspace_name("acme"),
        pytest.raises(CLIError, match="requires an absolute config path") as exc_info,
    ):
        shared_submit_module._resolve_source_submit_context(
            Path("configs/eval/default.toml")
        )

    assert exc_info.value.code == CLIErrorCode.VALIDATION


def test_source_submit_workspace_matches_config_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "repo" / "configs" / "eval" / "default.toml"
    context = _context()
    resolved_from: list[Path] = []

    monkeypatch.setattr(
        shared_submit_module,
        "resolve_git_workspace_directory_context",
        lambda *, cwd: (resolved_from.append(cwd), context)[1],
    )

    class FakeClient:
        def list_workspaces(self, *, credentials: object) -> list[WorkspaceSummary]:
            assert credentials is context.credentials
            return [
                WorkspaceSummary(
                    id="workspace-1",
                    name="acme",
                    connected_repo_full_name="Acme/Rollouts",
                )
            ]

    monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)

    with override_workspace_name("acme"):
        actual_context, actual_path, request_git_identity = (
            shared_submit_module._resolve_source_submit_context(config_path)
        )

    assert actual_context is context
    assert actual_path == config_path.resolve()
    assert request_git_identity is None
    assert resolved_from == [config_path.parent.resolve()]


@pytest.mark.parametrize(
    ("workspace_name", "connected_repo", "match"),
    [
        ("missing", "acme/rollouts", "not available to this account"),
        ("acme", None, "has no connected repository"),
        ("acme", "other/repo", "config file belongs to acme/rollouts"),
    ],
)
def test_source_submit_workspace_rejects_unmatched_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_name: str,
    connected_repo: str | None,
    match: str,
) -> None:
    context = _context()
    monkeypatch.setattr(
        shared_submit_module,
        "resolve_git_workspace_directory_context",
        lambda *, cwd: context,
    )

    class FakeClient:
        def list_workspaces(self, *, credentials: object) -> list[WorkspaceSummary]:
            return [
                WorkspaceSummary(
                    id="workspace-1",
                    name="acme",
                    connected_repo_full_name=connected_repo,
                )
            ]

    monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)

    with (
        override_workspace_name(workspace_name),
        pytest.raises(CLIError, match=match),
    ):
        shared_submit_module._resolve_source_submit_context(
            tmp_path / "repo" / "configs" / "eval" / "default.toml"
        )
