from __future__ import annotations

from pathlib import Path

import pytest

import osmosis_ai.cli.commands.benchmark as benchmark_commands
import osmosis_ai.platform.cli.benchmark as benchmark_handler


def test_benchmark_submit_delegates_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()

    def fake_submit(config_path: Path, *, yes: bool) -> object:
        captured.update(config_path=config_path, yes=yes)
        return expected

    monkeypatch.setattr(benchmark_handler, "submit", fake_submit)

    result = benchmark_commands.benchmark_submit(
        Path("configs/benchmark/smoke.toml"), yes=True
    )

    assert result is expected
    assert captured == {
        "config_path": Path("configs/benchmark/smoke.toml"),
        "yes": True,
    }
