from __future__ import annotations

from pathlib import Path

import pytest

import osmosis_ai.cli.commands.benchmark as benchmark_commands
import osmosis_ai.platform.cli.benchmark as benchmark_handler


def test_benchmark_list_delegates_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()

    def fake_list_benchmarks(*, limit: int, all_: bool) -> object:
        captured.update(limit=limit, all_=all_)
        return expected

    monkeypatch.setattr(benchmark_handler, "list_benchmarks", fake_list_benchmarks)

    result = benchmark_commands.benchmark_list(limit=25, all_=True)

    assert result is expected
    assert captured == {"limit": 25, "all_": True}


def test_benchmark_info_delegates_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()

    def fake_info(name_or_id: str) -> object:
        captured["name_or_id"] = name_or_id
        return expected

    monkeypatch.setattr(benchmark_handler, "info", fake_info)

    result = benchmark_commands.benchmark_info("Terminal-Bench 2.1")

    assert result is expected
    assert captured == {"name_or_id": "Terminal-Bench 2.1"}


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
