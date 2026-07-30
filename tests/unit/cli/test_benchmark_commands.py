from __future__ import annotations

from pathlib import Path

import pytest

import osmosis_ai.cli.commands.benchmark as benchmark_commands
import osmosis_ai.platform.cli.benchmark as benchmark_handler


def test_benchmark_catalog_list_delegates_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()

    def fake_list_benchmarks(*, limit: int, all_: bool) -> object:
        captured.update(limit=limit, all_=all_)
        return expected

    monkeypatch.setattr(benchmark_handler, "list_benchmarks", fake_list_benchmarks)

    result = benchmark_commands.benchmark_catalog_list(limit=25, all_=True)

    assert result is expected
    assert captured == {"limit": 25, "all_": True}


def test_benchmark_catalog_info_delegates_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()

    def fake_info(name_or_id: str) -> object:
        captured["name_or_id"] = name_or_id
        return expected

    monkeypatch.setattr(benchmark_handler, "catalog_info", fake_info)

    result = benchmark_commands.benchmark_catalog_info("Terminal-Bench 2.1")

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


def test_benchmark_run_commands_delegate_to_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    expected = object()

    monkeypatch.setattr(
        benchmark_handler,
        "list_benchmark_runs",
        lambda *, limit, all_: calls.append(("list", (limit, all_))) or expected,
    )
    monkeypatch.setattr(
        benchmark_handler,
        "run_info",
        lambda name_or_id: calls.append(("info", name_or_id)) or expected,
    )
    monkeypatch.setattr(
        benchmark_handler,
        "logs",
        lambda name_or_id, *, limit, cursor: (
            calls.append(("logs", (name_or_id, limit, cursor))) or expected
        ),
    )
    monkeypatch.setattr(
        benchmark_handler,
        "stop",
        lambda name_or_id, *, yes: (
            calls.append(("stop", (name_or_id, yes))) or expected
        ),
    )
    monkeypatch.setattr(
        benchmark_handler,
        "download",
        lambda name_or_id, *, output, types, overwrite, yes: (
            calls.append(("download", (name_or_id, output, types, overwrite, yes)))
            or expected
        ),
    )

    assert benchmark_commands.benchmark_list(limit=10, all_=False) is expected
    assert benchmark_commands.benchmark_info("run-1") is expected
    assert (
        benchmark_commands.benchmark_logs("run-1", limit=25, cursor="older") is expected
    )
    assert benchmark_commands.benchmark_stop("run-1", yes=True) is expected
    assert (
        benchmark_commands.benchmark_download(
            "run-1",
            output="out",
            types="all",
            overwrite=True,
            yes=True,
        )
        is expected
    )
    assert calls == [
        ("list", (10, False)),
        ("info", "run-1"),
        ("logs", ("run-1", 25, "older")),
        ("stop", ("run-1", True)),
        ("download", ("run-1", "out", "all", True, True)),
    ]
