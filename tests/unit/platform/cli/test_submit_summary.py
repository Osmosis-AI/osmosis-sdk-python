"""Tests for build_submit_summary_rows secrets rendering."""

from __future__ import annotations

from osmosis_ai.platform.cli.shared_config import build_submit_summary_rows


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


def test_secrets_row_omitted_when_empty() -> None:
    rows = _rows(secrets=[])
    assert not any(label.startswith("Rollout secrets") for label in rows)


def test_secrets_row_lists_names_sorted_with_count() -> None:
    rows = _rows(secrets=["OPENAI_API_KEY", "DATABASE_URL"])
    assert rows["Rollout secrets (2)"] == "DATABASE_URL, OPENAI_API_KEY"
