from __future__ import annotations

from pathlib import Path

from osmosis_ai.cli.main import main


def _make_workspace(root: Path) -> Path:
    for rel_path in (
        ".osmosis/research",
        "rollouts",
        "configs/training",
        "configs/eval",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / ".osmosis" / "workspace.toml").write_text(
        "[workspace]\nsetup_source = 'test'\n",
        encoding="utf-8",
    )
    return root


def test_workspace_validate_success(tmp_path, capsys) -> None:
    workspace_root = _make_workspace(tmp_path)

    rc = main(["workspace", "validate", str(workspace_root)])

    capsys.readouterr()
    assert rc == 0


def test_workspace_validate_reports_missing_required_path(tmp_path, capsys) -> None:
    workspace_root = _make_workspace(tmp_path)
    (workspace_root / "configs" / "eval").rmdir()

    rc = main(["workspace", "validate", str(workspace_root)])

    captured = capsys.readouterr()
    assert rc != 0
    assert "configs/eval" in captured.err
