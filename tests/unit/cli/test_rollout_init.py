"""End-to-end CLI contracts for ``osmosis rollout init``.

The scaffold source lives in ``osmosis_ai/templates/_scaffolds/rollout/`` as
SDK-bundled package data, so these tests run against the same ``.tpl`` files
that production users will receive. Mocking the source is deliberately limited
to the wheel-corruption edge case below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osmosis_ai.cli import main as cli


def _make_project(root: Path) -> Path:
    """Create a fully valid Osmosis project layout (matches validate_project_contract)."""
    (root / ".osmosis").mkdir(parents=True, exist_ok=True)
    (root / ".osmosis" / "project.toml").write_text(
        "[project]\nsetup_source = 'test'\n",
        encoding="utf-8",
    )
    for rel in ("rollouts", "configs/eval", "configs/training", "data"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


# ── success path ─────────────────────────────────────────────────


def test_rollout_init_json_writes_full_scaffold(monkeypatch, tmp_path, capsys) -> None:
    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    rc = cli.main(["--json", "rollout", "init", "my-agent"])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["status"] == "success"
    assert payload["operation"] == "rollout.init"
    assert payload["resource"]["name"] == "my-agent"
    assert payload["resource"]["project_root"] == str(project_root)
    assert Path(payload["resource"]["project_root"]).is_absolute()
    assert payload["resource"]["rollout_dir"] == "rollouts/my-agent"
    assert not Path(payload["resource"]["rollout_dir"]).is_absolute()
    assert payload["resource"]["configs"] == {
        "eval": "configs/eval/my-agent.toml",
        "training": "configs/training/my-agent.toml",
    }

    files = payload["resource"]["files"]
    assert "rollouts/my-agent/main.py" in files
    assert "rollouts/my-agent/pyproject.toml" in files
    assert "rollouts/my-agent/README.md" in files
    assert "configs/eval/my-agent.toml" in files
    assert "configs/training/my-agent.toml" in files

    rollout_dir = project_root / "rollouts" / "my-agent"
    assert (rollout_dir / "main.py").is_file()
    assert (rollout_dir / "pyproject.toml").is_file()
    assert (rollout_dir / "README.md").is_file()
    assert (project_root / "configs" / "eval" / "my-agent.toml").is_file()
    assert (project_root / "configs" / "training" / "my-agent.toml").is_file()


def test_rollout_init_substitutes_rollout_name_in_emitted_files(
    monkeypatch, tmp_path
) -> None:
    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    rc = cli.main(["--json", "rollout", "init", "my-agent"])
    assert rc == 0

    rollout_dir = project_root / "rollouts" / "my-agent"
    pyproject = (rollout_dir / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "my-agent"' in pyproject
    assert "<your-rollout>" not in pyproject

    main_py = (rollout_dir / "main.py").read_text(encoding="utf-8")
    # main.py.tpl opens with a docstring containing the rollout name token.
    assert "my-agent" in main_py
    assert "<your-rollout>" not in main_py

    readme = (rollout_dir / "README.md").read_text(encoding="utf-8")
    assert "my-agent" in readme
    assert "<your-rollout>" not in readme


def test_rollout_init_plain_next_steps_use_existing_commands(
    monkeypatch, tmp_path, capsys
) -> None:
    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    rc = cli.main(["--plain", "rollout", "init", "my-agent"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "osmosis eval run configs/eval/my-agent.toml --limit 1" in captured.out
    assert "osmosis train submit configs/training/my-agent.toml" in captured.out
    assert "osmosis rollout validate" not in captured.out


def test_rollout_init_main_py_is_a_runnable_rollout_server(
    monkeypatch, tmp_path
) -> None:
    """Placeholder main.py must wire LocalBackend + uvicorn so users can run it."""
    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    rc = cli.main(["--json", "rollout", "init", "my-agent"])
    assert rc == 0

    main_py = (project_root / "rollouts" / "my-agent" / "main.py").read_text(
        encoding="utf-8"
    )
    assert "from osmosis_ai.rollout.backend.local import LocalBackend" in main_py
    assert "from osmosis_ai.rollout.server import create_rollout_server" in main_py
    assert "osmosis_ai.rollout.integrations.agents.openai_agents" in main_py
    assert "osmosis_ai.rollout.integrations.agents.openai " not in main_py
    assert "osmosis rollout serve" not in main_py
    assert "def main()" in main_py
    assert 'if __name__ == "__main__":' in main_py
    assert "uvicorn.run(" in main_py


def test_rollout_init_substitutes_rollout_name_in_configs(
    monkeypatch, tmp_path
) -> None:
    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    rc = cli.main(["--json", "rollout", "init", "my-agent"])
    assert rc == 0

    eval_toml = (project_root / "configs" / "eval" / "my-agent.toml").read_text(
        encoding="utf-8"
    )
    assert 'rollout = "my-agent"' in eval_toml
    # Dataset placeholder must remain (user must pick a dataset themselves).
    assert "<your-dataset>" in eval_toml

    training_toml = (project_root / "configs" / "training" / "my-agent.toml").read_text(
        encoding="utf-8"
    )
    assert 'rollout = "my-agent"' in training_toml
    # Entrypoint is a literal in the source TOML (not substituted), so it flows
    # through unchanged.
    assert 'entrypoint = "main.py"' in training_toml
    # Non-rollout placeholders the user must still fill in remain literal.
    assert "<your-model-path>" in training_toml
    assert "<your-dataset-name>" in training_toml
    assert "Qwen/Qwen3.6-35B-A3B" in training_toml
    assert "Qwen/Qwen3.5-122B-A10B" in training_toml


def test_rollout_init_training_config_loads_with_template_defaults(
    monkeypatch, tmp_path
) -> None:
    from osmosis_ai.platform.cli.training_config import load_training_config

    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    rc = cli.main(["--json", "rollout", "init", "my-agent"])
    assert rc == 0

    training_toml = project_root / "configs" / "training" / "my-agent.toml"
    config = load_training_config(training_toml)

    assert config.training_n_samples_per_prompt == 8
    assert config.training_rollout_batch_size == 64


# ── conflict handling ───────────────────────────────────────────


def test_rollout_init_refuses_overwrite_without_force(
    monkeypatch, tmp_path, capsys
) -> None:
    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    rollout_dir = project_root / "rollouts" / "my-agent"
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "main.py").write_text("# user code\n", encoding="utf-8")

    rc = cli.main(["--json", "rollout", "init", "my-agent"])
    captured = capsys.readouterr()

    assert rc != 0
    err = json.loads(captured.err)
    assert err["error"]["code"] == "CONFLICT"
    assert "rollouts/my-agent/" in err["error"]["message"]
    assert "--force" in err["error"]["message"]
    # User code intact.
    assert (rollout_dir / "main.py").read_text(encoding="utf-8") == "# user code\n"


def test_rollout_init_force_overwrites_rollout_and_configs(
    monkeypatch, tmp_path
) -> None:
    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    rollout_dir = project_root / "rollouts" / "my-agent"
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "STALE.txt").write_text("stale", encoding="utf-8")
    (project_root / "configs" / "eval" / "my-agent.toml").write_text(
        "[eval]\nstale = true\n", encoding="utf-8"
    )

    rc = cli.main(["--json", "rollout", "init", "my-agent", "--force"])

    assert rc == 0
    assert not (rollout_dir / "STALE.txt").exists()
    assert (rollout_dir / "main.py").is_file()
    fresh = (project_root / "configs" / "eval" / "my-agent.toml").read_text(
        encoding="utf-8"
    )
    assert "stale" not in fresh
    assert 'rollout = "my-agent"' in fresh


def test_rollout_init_force_refuses_non_directory_rollout_path(
    monkeypatch, tmp_path, capsys
) -> None:
    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    # `rollouts/my-agent` exists as a regular file, not a directory.
    bogus = project_root / "rollouts" / "my-agent"
    bogus.write_text("oops", encoding="utf-8")

    rc = cli.main(["--json", "rollout", "init", "my-agent", "--force"])
    captured = capsys.readouterr()

    assert rc != 0
    err = json.loads(captured.err)
    assert err["error"]["code"] == "CONFLICT"
    assert "non-directory" in err["error"]["message"]
    assert bogus.read_text(encoding="utf-8") == "oops"


def test_rollout_init_force_refuses_config_symlink(
    monkeypatch, tmp_path, capsys
) -> None:
    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    outside = tmp_path / "outside.toml"
    outside.write_text("original", encoding="utf-8")
    symlink = project_root / "configs" / "eval" / "my-agent.toml"
    symlink.symlink_to(outside)

    rc = cli.main(["--json", "rollout", "init", "my-agent", "--force"])
    captured = capsys.readouterr()

    assert rc != 0
    err = json.loads(captured.err)
    assert err["error"]["code"] == "CONFLICT"
    assert "configs/eval/my-agent.toml" in err["error"]["message"]
    assert outside.read_text(encoding="utf-8") == "original"


def test_rollout_init_force_refuses_config_directory_before_resetting_rollout(
    monkeypatch, tmp_path, capsys
) -> None:
    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    rollout_dir = project_root / "rollouts" / "my-agent"
    rollout_dir.mkdir(parents=True)
    user_code = rollout_dir / "main.py"
    user_code.write_text("# user code\n", encoding="utf-8")
    (project_root / "configs" / "eval" / "my-agent.toml").mkdir()

    rc = cli.main(["--json", "rollout", "init", "my-agent", "--force"])
    captured = capsys.readouterr()

    assert rc != 0
    err = json.loads(captured.err)
    assert err["error"]["code"] == "CONFLICT"
    assert "configs/eval/my-agent.toml" in err["error"]["message"]
    assert user_code.read_text(encoding="utf-8") == "# user code\n"


# ── name validation ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_name",
    ["My-Agent", "_starter", "1agent", "a/b", "default"],
)
def test_rollout_init_rejects_invalid_or_reserved_names(
    monkeypatch, tmp_path, capsys, bad_name: str
) -> None:
    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    rc = cli.main(["--json", "rollout", "init", bad_name])
    captured = capsys.readouterr()

    assert rc != 0
    err = json.loads(captured.err)
    assert err["error"]["code"] == "VALIDATION"


# ── project-contract guard ──────────────────────────────────────


def test_rollout_init_outside_project_errors(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    rc = cli.main(["rollout", "init", "my-agent"])
    captured = capsys.readouterr()

    assert rc != 0
    assert "Not in an Osmosis project" in captured.err


def test_rollout_init_incomplete_project_errors(monkeypatch, tmp_path, capsys) -> None:
    # `.osmosis/project.toml` exists but canonical directories are missing.
    project_root = tmp_path / "proj"
    (project_root / ".osmosis").mkdir(parents=True)
    (project_root / ".osmosis" / "project.toml").write_text(
        "[project]\nsetup_source = 'test'\n", encoding="utf-8"
    )
    monkeypatch.chdir(project_root)

    rc = cli.main(["rollout", "init", "my-agent"])
    captured = capsys.readouterr()

    assert rc != 0
    assert "missing required Osmosis paths" in captured.err


# ── SDK package-data guard ──────────────────────────────────────


def test_scaffold_tpl_files_ship_with_package() -> None:
    """Sanity: every entry in ``_SCAFFOLD_LAYOUT`` resolves under the package.

    Guards against pyproject.toml's ``package-data`` regressing and dropping the
    ``.tpl`` files from the wheel/editable install.
    """
    from importlib.resources import files

    from osmosis_ai.templates.init import _SCAFFOLD_LAYOUT, _SCAFFOLD_PACKAGE

    scaffold_root = files(_SCAFFOLD_PACKAGE)
    for tpl_name, _ in _SCAFFOLD_LAYOUT:
        assert scaffold_root.joinpath(tpl_name).is_file(), (
            f"missing scaffold: {tpl_name}"
        )


def test_rollout_init_missing_source_files_returns_not_found(
    monkeypatch, tmp_path, capsys
) -> None:
    """If the SDK wheel is missing a bundled scaffold .tpl (e.g. corrupted install),
    ``rollout init`` should surface a NOT_FOUND error instead of an uncaught
    FileNotFoundError.
    """
    from osmosis_ai.templates import init as init_module

    monkeypatch.setattr(
        init_module,
        "_SCAFFOLD_LAYOUT",
        (("nonexistent-scaffold.tpl", "rollouts/{name}/oops"),),
    )

    project_root = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project_root)

    rc = cli.main(["--json", "rollout", "init", "my-agent"])
    captured = capsys.readouterr()

    assert rc != 0
    err = json.loads(captured.err)
    assert err["error"]["code"] == "NOT_FOUND"
    assert "nonexistent-scaffold.tpl" in err["error"]["message"]


# ── help ────────────────────────────────────────────────────────


def test_rollout_help_lists_init(capfd) -> None:
    rc = cli.main(["rollout", "--help"])
    out = capfd.readouterr().out

    assert rc == 0
    assert "init" in out
    assert "list" in out
