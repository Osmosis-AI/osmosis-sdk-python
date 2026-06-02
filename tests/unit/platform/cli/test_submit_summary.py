"""Tests for submit summary table builders."""

from __future__ import annotations

from osmosis_ai.platform.cli.shared_config import (
    build_env_table_rows,
    build_secret_table_rows,
    build_submit_summary_rows,
)


def _rows(**kwargs) -> dict[str, str]:
    defaults = {
        "rollout": "r",
        "entrypoint": "e.py",
        "model": "m",
        "dataset": "d",
        "commit_sha": None,
        "env": {},
        "secrets": [],
    }
    defaults.update(kwargs)
    return dict(build_submit_summary_rows(**defaults))


def test_summary_rows_no_env_or_secrets() -> None:
    rows = _rows(env={"A": "1"}, secrets=["X"])
    assert not any(label.startswith("Rollout env") for label in rows)
    assert not any(label.startswith("Rollout secrets") for label in rows)


def test_summary_rows_includes_core_fields() -> None:
    rows = _rows()
    assert rows["Rollout"] == "r"
    assert rows["Entrypoint"] == "e.py"
    assert rows["Model"] == "m"
    assert rows["Dataset"] == "d"


def test_summary_rows_includes_commit_sha() -> None:
    rows = _rows(commit_sha="abc123")
    assert rows["Commit"] == "abc123"


# --- build_env_table_rows ---


def test_env_table_rows_sorted_by_name() -> None:
    rows = build_env_table_rows({"Z_VAR": "z", "A_VAR": "a"})
    assert rows == [("A_VAR", "a"), ("Z_VAR", "z")]


def test_env_table_rows_empty() -> None:
    assert build_env_table_rows({}) == []


# --- build_secret_table_rows ---


def test_secret_table_rows_workspace_scope() -> None:
    rows = build_secret_table_rows(
        ["WANDB_API_KEY"],
        user_secret_names=set(),
        workspace_secret_names={"WANDB_API_KEY"},
    )
    assert rows == [("WANDB_API_KEY", "Workspace")]


def test_secret_table_rows_personal_override_scope() -> None:
    rows = build_secret_table_rows(
        ["OPENAI_API_KEY"],
        user_secret_names={"OPENAI_API_KEY"},
        workspace_secret_names={"OPENAI_API_KEY"},
    )
    assert rows == [("OPENAI_API_KEY", "Personal (overrides workspace)")]


def test_secret_table_rows_personal_only_is_not_an_override() -> None:
    rows = build_secret_table_rows(
        ["OPENAI_API_KEY"],
        user_secret_names={"OPENAI_API_KEY"},
        workspace_secret_names=set(),
    )
    assert rows == [("OPENAI_API_KEY", "Personal")]


def test_secret_table_rows_mixed_scopes_sorted() -> None:
    rows = build_secret_table_rows(
        ["OPENAI_API_KEY", "GITHUB_TOKEN", "WANDB_API_KEY"],
        user_secret_names={"OPENAI_API_KEY"},
        workspace_secret_names={"OPENAI_API_KEY", "GITHUB_TOKEN"},
    )
    assert rows == [
        ("GITHUB_TOKEN", "Workspace"),
        ("OPENAI_API_KEY", "Personal (overrides workspace)"),
        ("WANDB_API_KEY", "Workspace"),
    ]


def test_secret_table_rows_empty() -> None:
    assert (
        build_secret_table_rows(
            [], user_secret_names=set(), workspace_secret_names=set()
        )
        == []
    )
