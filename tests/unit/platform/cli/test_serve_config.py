"""Tests for serve TOML config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli.serve_config import ServeConfig, load_serve_config


def test_load_serve_config_full(tmp_path: Path) -> None:
    path = tmp_path / "serve.toml"
    path.write_text(
        """
[serve]
rollout = "my_rollout"
entrypoint = "pkg.mod:agent"

[server]
port = 8080
host = "127.0.0.1"
log_level = "debug"

[debug]
no_validate = true
trace_dir = "/tmp/traces"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_serve_config(path)
    assert isinstance(cfg, ServeConfig)
    assert cfg.serve_rollout == "my_rollout"
    assert cfg.serve_entrypoint == "pkg.mod:agent"
    assert cfg.server_port == 8080
    assert cfg.server_host == "127.0.0.1"
    assert cfg.server_log_level == "debug"
    assert cfg.debug_no_validate is True
    assert cfg.debug_trace_dir == "/tmp/traces"


def test_load_serve_config_minimal_defaults(tmp_path: Path) -> None:
    path = tmp_path / "minimal.toml"
    path.write_text(
        """
[serve]
rollout = "r"
entrypoint = "m:a"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_serve_config(path)
    assert cfg.serve_rollout == "r"
    assert cfg.serve_entrypoint == "m:a"
    assert cfg.server_port == 9000
    assert cfg.server_host == "0.0.0.0"
    assert cfg.server_log_level == "info"
    assert cfg.debug_no_validate is False
    assert cfg.debug_trace_dir is None


def test_missing_serve_section(tmp_path: Path) -> None:
    path = tmp_path / "no_serve.toml"
    path.write_text("""[server]\nport = 9000\n""", encoding="utf-8")

    with pytest.raises(CLIError) as exc_info:
        load_serve_config(path)
    assert "[serve]" in str(exc_info.value)


def test_serve_must_be_table_not_primitive(tmp_path: Path) -> None:
    """`serve` must be a TOML table; a scalar value must not leak TypeError."""
    path = tmp_path / "serve_primitive.toml"
    path.write_text("serve = 123\n", encoding="utf-8")

    with pytest.raises(CLIError) as exc_info:
        load_serve_config(path)
    assert "[serve]" in str(exc_info.value)


def test_missing_rollout(tmp_path: Path) -> None:
    path = tmp_path / "no_rollout.toml"
    path.write_text(
        """
[serve]
entrypoint = "m:a"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(CLIError) as exc_info:
        load_serve_config(path)
    assert "rollout" in str(exc_info.value)


def test_missing_entrypoint(tmp_path: Path) -> None:
    path = tmp_path / "no_entrypoint.toml"
    path.write_text(
        """
[serve]
rollout = "r"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(CLIError) as exc_info:
        load_serve_config(path)
    assert "entrypoint" in str(exc_info.value)


def test_invalid_toml(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[[[not valid", encoding="utf-8")

    with pytest.raises(CLIError) as exc_info:
        load_serve_config(path)
    assert "Invalid TOML" in str(exc_info.value)


def test_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"

    with pytest.raises(CLIError) as exc_info:
        load_serve_config(missing)
    assert "not found" in str(exc_info.value)


def test_load_serve_config_directory_path_raises_cli_error(tmp_path: Path) -> None:
    """Paths that exist but are not readable files (e.g. directories) must raise CLIError."""
    dir_path = tmp_path / "config_dir"
    dir_path.mkdir()

    with pytest.raises(CLIError) as exc_info:
        load_serve_config(dir_path)
    assert "Cannot read config file" in str(exc_info.value)


def test_invalid_port(tmp_path: Path) -> None:
    path = tmp_path / "bad_port.toml"
    path.write_text(
        """
[serve]
rollout = "r"
entrypoint = "m:a"

[server]
port = 99999
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(CLIError) as exc_info:
        load_serve_config(path)
    assert "Invalid config" in str(exc_info.value)


def test_invalid_log_level(tmp_path: Path) -> None:
    path = tmp_path / "bad_log.toml"
    path.write_text(
        """
[serve]
rollout = "r"
entrypoint = "m:a"

[server]
log_level = "verbose"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(CLIError) as exc_info:
        load_serve_config(path)
    assert "Invalid config" in str(exc_info.value)
