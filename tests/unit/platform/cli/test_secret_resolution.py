from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.secret_resolution import resolve_run_secrets


def test_secrets_file_wins_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("A_KEY", "from-env")
    path = tmp_path / ".env"
    path.write_text("A_KEY=from-file\n", encoding="utf-8")

    assert resolve_run_secrets(
        names=["A_KEY"], secrets_file=str(path), stored_names=set(), is_tty=False
    ) == {"A_KEY": "from-file"}


def test_environment_used_when_no_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A_KEY", "from-env")

    assert resolve_run_secrets(
        names=["A_KEY"], secrets_file=None, stored_names=set(), is_tty=False
    ) == {"A_KEY": "from-env"}


def test_stored_names_are_left_to_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("A_KEY", raising=False)

    assert (
        resolve_run_secrets(
            names=["A_KEY"], secrets_file=None, stored_names={"A_KEY"}, is_tty=False
        )
        == {}
    )


def test_non_tty_lists_every_missing_name_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)

    with pytest.raises(CLIError, match=r"A_KEY.*B_KEY"):
        resolve_run_secrets(
            names=["A_KEY", "B_KEY"],
            secrets_file=None,
            stored_names=set(),
            is_tty=False,
        )


def test_comments_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("# a comment\n\nA_KEY=value\n", encoding="utf-8")

    assert resolve_run_secrets(
        names=["A_KEY"], secrets_file=str(path), stored_names=set(), is_tty=False
    ) == {"A_KEY": "value"}


def test_a_malformed_secrets_file_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("NOT_AN_ASSIGNMENT\n", encoding="utf-8")

    with pytest.raises(CLIError, match="Invalid line"):
        resolve_run_secrets(
            names=["A_KEY"],
            secrets_file=str(path),
            stored_names=set(),
            is_tty=False,
        )
