"""Root callback tests for --json/--plain parsing."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.dataset as dataset_module
from osmosis_ai.cli import main as cli
from osmosis_ai.platform.api.models import DatasetFile, PaginatedDatasets


def test_help_is_plain_text_even_with_json(capsys) -> None:
    exit_code = cli.main(["--json", "--help"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Osmosis" in captured.out
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.out)


def test_version_is_plain_text_even_with_json(capsys) -> None:
    exit_code = cli.main(["--json", "--version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip().startswith("osmosis-ai")


def test_conflicting_selectors_emit_validation_error(capsys) -> None:
    exit_code = cli.main(["--json", "--plain", "dataset", "list"])
    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    assert "conflict" in captured.err.lower() or "format" in captured.err.lower()


def test_json_mode_contextvar_resets_after_eager_help(capsys) -> None:
    from osmosis_ai.cli.output.context import _output_context_var

    cli.main(["--json", "--help"])
    capsys.readouterr()
    assert _output_context_var.get() is None


def _stub_dataset_list(monkeypatch: pytest.MonkeyPatch) -> None:
    context = SimpleNamespace(
        workspace_directory="/repo",
        git_identity="acme/rollouts",
        repo_url="https://github.com/acme/rollouts.git",
        credentials=object(),
    )
    monkeypatch.setattr(
        dataset_module,
        "require_git_workspace_directory_context",
        lambda: context,
    )

    class FakeClient:
        def list_datasets(self, *, limit, offset, git_identity, credentials=None):
            return PaginatedDatasets(
                datasets=[
                    DatasetFile(
                        id="ds_1",
                        file_name="train.jsonl",
                        file_size=100,
                        status="uploaded",
                        created_at="2026-04-26T00:00:00Z",
                    )
                ],
                total_count=1,
                has_more=False,
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)


def test_postfix_json_works_for_list_command(monkeypatch, capsys) -> None:
    _stub_dataset_list(monkeypatch)

    exit_code = cli.main(["dataset", "list", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["items"][0]["id"] == "ds_1"


def test_postfix_json_output_matches_prefix(monkeypatch, capsys) -> None:
    _stub_dataset_list(monkeypatch)
    assert cli.main(["--json", "dataset", "list"]) == 0
    prefix_out = capsys.readouterr().out

    _stub_dataset_list(monkeypatch)
    assert cli.main(["dataset", "list", "--json"]) == 0
    postfix_out = capsys.readouterr().out

    assert json.loads(postfix_out) == json.loads(prefix_out)


def test_postfix_plain_output_matches_prefix(monkeypatch, capsys) -> None:
    _stub_dataset_list(monkeypatch)
    assert cli.main(["--plain", "dataset", "list"]) == 0
    prefix_out = capsys.readouterr().out

    _stub_dataset_list(monkeypatch)
    assert cli.main(["dataset", "list", "--plain"]) == 0
    postfix_out = capsys.readouterr().out

    assert postfix_out == prefix_out
    assert "train.jsonl" in postfix_out


def test_postfix_json_works_for_argument_taking_command(
    tmp_path, monkeypatch, capsys
) -> None:
    data = tmp_path / "data.jsonl"
    row = json.dumps({"system_prompt": "s", "user_prompt": "u", "ground_truth": "g"})
    data.write_text((row + "\n") * 4, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert cli.main(["--json", "dataset", "validate", str(data)]) == 0
    prefix_out = capsys.readouterr().out

    assert cli.main(["dataset", "validate", str(data), "--json"]) == 0
    postfix_out = capsys.readouterr().out

    assert json.loads(postfix_out) == json.loads(prefix_out)


def test_postfix_conflicting_selectors_emit_validation_error(capsys) -> None:
    exit_code = cli.main(["dataset", "list", "--json", "--plain"])
    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    assert "conflict" in captured.err.lower() or "format" in captured.err.lower()


def test_json_after_double_dash_is_a_literal_argument(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["dataset", "validate", "--", "--json"])
    captured = capsys.readouterr()

    assert exit_code != 0
    # "--json" was consumed as the path argument, so the failure renders as
    # human text rather than a JSON envelope.
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.err)
