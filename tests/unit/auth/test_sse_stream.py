"""Tests for SSE parsing in osmosis_ai.platform.auth.platform_client."""

from __future__ import annotations

from osmosis_ai.platform.auth.platform_client import iter_sse_data


class TestIterSSEData:
    def test_yields_one_dict_per_event(self) -> None:
        lines = [
            'data: {"timestamp": "t1", "message": "a"}\n',
            "\n",
            'data: {"timestamp": "t2", "message": "b"}\n',
            "\n",
        ]
        assert list(iter_sse_data(lines)) == [
            {"timestamp": "t1", "message": "a"},
            {"timestamp": "t2", "message": "b"},
        ]

    def test_skips_comment_heartbeats(self) -> None:
        lines = [
            ": keepalive\n",
            'data: {"message": "x"}\n',
            "\n",
            ": keepalive\n",
        ]
        assert list(iter_sse_data(lines)) == [{"message": "x"}]

    def test_concatenates_multiline_data(self) -> None:
        lines = ["data: {\n", 'data: "message": "y"}\n', "\n"]
        assert list(iter_sse_data(lines)) == [{"message": "y"}]

    def test_ignores_non_data_fields(self) -> None:
        lines = ["event: log\n", "id: 5\n", 'data: {"message": "z"}\n', "\n"]
        assert list(iter_sse_data(lines)) == [{"message": "z"}]

    def test_flushes_trailing_event_without_blank_line(self) -> None:
        lines = ['data: {"message": "last"}\n']
        assert list(iter_sse_data(lines)) == [{"message": "last"}]

    def test_skips_non_json_data(self) -> None:
        lines = ["data: not-json\n", "\n", 'data: {"message": "ok"}\n', "\n"]
        assert list(iter_sse_data(lines)) == [{"message": "ok"}]
