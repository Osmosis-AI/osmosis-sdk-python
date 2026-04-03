"""Tests for EvalProxy and RequestMetrics."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from osmosis_ai.eval.proxy import EvalProxy, RequestMetrics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_client(
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
) -> AsyncMock:
    """Return an AsyncMock that mimics ExternalLLMClient.chat_completions."""
    mock_client = AsyncMock()
    mock_client.chat_completions.return_value = {
        "message": {"role": "assistant", "content": "hi"},
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "finish_reason": "stop",
    }
    return mock_client


# ---------------------------------------------------------------------------
# RequestMetrics
# ---------------------------------------------------------------------------


class TestRequestMetrics:
    def test_request_metrics_defaults(self) -> None:
        """All fields default to zero."""
        m = RequestMetrics()
        assert m.prompt_tokens == 0
        assert m.completion_tokens == 0
        assert m.num_calls == 0
        assert m.total_latency_ms == 0.0


# ---------------------------------------------------------------------------
# collect_metrics
# ---------------------------------------------------------------------------


class TestCollectMetrics:
    def test_collect_metrics_returns_and_removes(self) -> None:
        """collect_metrics pops metrics keyed by rollout_id."""
        proxy = EvalProxy(client=AsyncMock())
        # Seed internal state directly.
        proxy._metrics["run-1"] = RequestMetrics(
            prompt_tokens=20,
            completion_tokens=10,
            num_calls=2,
            total_latency_ms=500.0,
        )

        result = proxy.collect_metrics("run-1")

        assert result.prompt_tokens == 20
        assert result.completion_tokens == 10
        assert result.num_calls == 2
        assert result.total_latency_ms == 500.0
        # Key should have been removed.
        assert "run-1" not in proxy._metrics

    def test_collect_metrics_missing_returns_empty(self) -> None:
        """Collecting metrics for an unknown id returns zeroed RequestMetrics."""
        proxy = EvalProxy(client=AsyncMock())
        result = proxy.collect_metrics("does-not-exist")

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.num_calls == 0
        assert result.total_latency_ms == 0.0


# ---------------------------------------------------------------------------
# _handle_chat_completions
# ---------------------------------------------------------------------------


class TestHandleChatCompletions:
    async def test_handle_chat_completions_tracks_metrics(self) -> None:
        """Metrics accumulate across multiple calls with the same rollout_id."""
        mock_client = _make_mock_client(prompt_tokens=10, completion_tokens=5)
        proxy = EvalProxy(client=mock_client)

        rollout_id = "run-42"

        # First call
        resp1 = await proxy._handle_chat_completions(
            rollout_id,
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp1["message"]["content"] == "hi"

        # Second call — same rollout_id
        await proxy._handle_chat_completions(
            rollout_id,
            {"messages": [{"role": "user", "content": "world"}]},
        )

        metrics = proxy.collect_metrics(rollout_id)
        assert metrics.prompt_tokens == 20  # 10 + 10
        assert metrics.completion_tokens == 10  # 5 + 5
        assert metrics.num_calls == 2
        assert metrics.total_latency_ms > 0

    async def test_handle_chat_completions_forwards_params(self) -> None:
        """Only whitelisted keys (tools, tool_choice) are forwarded."""
        mock_client = _make_mock_client()
        proxy = EvalProxy(client=mock_client)

        await proxy._handle_chat_completions(
            "run-99",
            {
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.7,
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "f"}}],
            },
        )

        mock_client.chat_completions.assert_awaited_once()
        call_kwargs = mock_client.chat_completions.call_args
        # Sampling params and stream are NOT forwarded.
        assert "temperature" not in call_kwargs.kwargs
        assert "stream" not in call_kwargs.kwargs
        # Semantic keys ARE forwarded.
        assert call_kwargs.kwargs["tools"] == [
            {"type": "function", "function": {"name": "f"}}
        ]

    async def test_handle_chat_completions_without_trace(self) -> None:
        """No trace file is written when trace_dir is None."""
        mock_client = _make_mock_client()
        proxy = EvalProxy(client=mock_client, trace_dir=None)

        await proxy._handle_chat_completions(
            "run-0",
            {"messages": [{"role": "user", "content": "hi"}]},
        )

        # Proxy should have metrics but no trace_dir set.
        assert proxy.trace_dir is None
        metrics = proxy.collect_metrics("run-0")
        assert metrics.num_calls == 1


# ---------------------------------------------------------------------------
# Trace logging
# ---------------------------------------------------------------------------


class TestTraceLogging:
    async def test_trace_logging(self, tmp_path: object) -> None:
        """Writes JSONL to trace_dir/{rollout_id}.jsonl with correct format."""
        from pathlib import Path

        trace_dir = Path(str(tmp_path)) / "traces"
        mock_client = _make_mock_client()
        proxy = EvalProxy(client=mock_client, trace_dir=str(trace_dir))

        rollout_id = "trace-run-1"
        request_body = {"messages": [{"role": "user", "content": "ping"}]}

        await proxy._handle_chat_completions(rollout_id, request_body)

        trace_file = trace_dir / f"{rollout_id}.jsonl"
        assert trace_file.exists()

        lines = trace_file.read_text().strip().splitlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["type"] == "llm_call"
        assert "ts" in entry
        assert isinstance(entry["ts"], float)
        assert entry["request"] == request_body
        assert entry["response"]["message"]["content"] == "hi"
        assert isinstance(entry["latency_ms"], float)
        assert entry["latency_ms"] >= 0

    async def test_trace_multiple_calls_appends(self, tmp_path: object) -> None:
        """Multiple calls append separate lines to the same trace file."""
        from pathlib import Path

        trace_dir = Path(str(tmp_path)) / "traces"
        mock_client = _make_mock_client()
        proxy = EvalProxy(client=mock_client, trace_dir=str(trace_dir))

        rollout_id = "trace-run-2"
        for i in range(3):
            await proxy._handle_chat_completions(
                rollout_id,
                {"messages": [{"role": "user", "content": f"msg-{i}"}]},
            )

        trace_file = trace_dir / f"{rollout_id}.jsonl"
        lines = trace_file.read_text().strip().splitlines()
        assert len(lines) == 3

        # Verify each line is valid JSON with monotonically non-decreasing ts.
        prev_ts = 0.0
        for line in lines:
            entry = json.loads(line)
            assert entry["ts"] >= prev_ts
            prev_ts = entry["ts"]
