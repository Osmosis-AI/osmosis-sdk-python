"""CLI shell tests for ``osmosis rollout serve``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.cli import main as cli
from osmosis_ai.rollout import serve as serve_module


def test_serve_forwards_config_and_runtime_options(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    config_path = tmp_path / "rollout.toml"
    config_path.write_text('backend = "harbor"\n[harbor]\nagent = "mini-swe-agent"\n')
    captured: dict[str, Any] = {}

    def fake_serve(path: Path, **kwargs: Any) -> None:
        captured["path"] = path
        captured.update(kwargs)

    monkeypatch.setattr(serve_module, "serve", fake_serve)

    rc = cli.main(
        [
            "--json",
            "rollout",
            "serve",
            str(config_path),
            "--harbor-dataset",
            "./tasks",
            "--host",
            "127.0.0.1",
            "--port",
            "8123",
        ]
    )

    assert rc == 0
    assert captured == {
        "path": config_path,
        "harbor_dataset": "./tasks",
        "host": "127.0.0.1",
        "port": 8123,
    }
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
