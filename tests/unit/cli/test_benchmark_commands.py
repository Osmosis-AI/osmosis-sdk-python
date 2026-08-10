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

    def fake_info(key: str, *, limit: int, all_: bool) -> object:
        captured.update(key=key, limit=limit, all_=all_)
        return expected

    monkeypatch.setattr(benchmark_handler, "benchmark_info", fake_info)

    result = benchmark_commands.benchmark_info(
        "terminal-bench-2-1", limit=15, all_=False
    )

    assert result is expected
    assert captured == {
        "key": "terminal-bench-2-1",
        "limit": 15,
        "all_": False,
    }


def test_benchmark_submit_delegates_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()

    def fake_submit(
        config_path: Path, *, yes: bool, secrets_file: str | None
    ) -> object:
        captured.update(config_path=config_path, yes=yes, secrets_file=secrets_file)
        return expected

    monkeypatch.setattr(benchmark_handler, "submit", fake_submit)

    result = benchmark_commands.benchmark_submit(
        Path("configs/benchmark/smoke.toml"), yes=True, secrets_file=None
    )

    assert result is expected
    assert captured == {
        "config_path": Path("configs/benchmark/smoke.toml"),
        "yes": True,
        "secrets_file": None,
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
        lambda name: calls.append(("info", name)) or expected,
    )
    monkeypatch.setattr(
        benchmark_handler,
        "logs",
        lambda name, *, limit, cursor: (
            calls.append(("logs", (name, limit, cursor))) or expected
        ),
    )
    monkeypatch.setattr(
        benchmark_handler,
        "stop",
        lambda name, *, yes: calls.append(("stop", (name, yes))) or expected,
    )
    monkeypatch.setattr(
        benchmark_handler,
        "download",
        lambda name, *, output, types, overwrite, yes: (
            calls.append(("download", (name, output, types, overwrite, yes)))
            or expected
        ),
    )

    assert benchmark_commands.benchmark_runs_list(limit=10, all_=False) is expected
    assert benchmark_commands.benchmark_runs_info("run-1") is expected
    assert (
        benchmark_commands.benchmark_runs_logs("run-1", limit=25, cursor="older")
        is expected
    )
    assert benchmark_commands.benchmark_runs_stop("run-1", yes=True) is expected
    assert (
        benchmark_commands.benchmark_runs_download(
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
