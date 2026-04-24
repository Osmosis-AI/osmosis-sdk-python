"""Tests for osmosis_ai.platform.api.download."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest

import osmosis_ai.platform.api.download as download_module


class _FakeStreamResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        body: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks or []
        self._body = body

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *_args: Any) -> None:
        pass

    def iter_bytes(self, _chunk_size: int):
        yield from self._chunks

    def read(self) -> bytes:
        return self._body


def test_download_file_uses_content_disposition_filename(monkeypatch, tmp_path) -> None:
    progress_updates: list[tuple[int, int]] = []

    monkeypatch.setattr(
        download_module.httpx,
        "stream",
        lambda *args, **kwargs: _FakeStreamResponse(
            headers={
                "content-disposition": 'attachment; filename="data.jsonl"',
                "content-length": "11",
            },
            chunks=[b"hello ", b"world"],
        ),
    )
    monkeypatch.setattr(
        download_module,
        "make_progress_bar",
        lambda total, *, description: (
            nullcontext(),
            lambda completed, total: progress_updates.append((completed, total)),
        ),
    )

    destination = download_module.download_file(
        "https://example.com/signed",
        output=tmp_path,
        default_filename="fallback",
        expected_size=11,
    )

    assert destination == tmp_path / "data.jsonl"
    assert destination.read_bytes() == b"hello world"
    assert progress_updates[-1] == (11, 11)
    assert not list(tmp_path.glob("*.tmp"))


def test_download_file_follows_https_redirect(monkeypatch, tmp_path) -> None:
    responses = iter(
        [
            _FakeStreamResponse(
                status_code=302,
                headers={"location": "/download/data.jsonl"},
            ),
            _FakeStreamResponse(
                headers={"content-length": "11"},
                chunks=[b"hello ", b"world"],
            ),
        ]
    )
    requests: list[tuple[str, str, bool]] = []

    def fake_stream(method, url, **kwargs):
        requests.append((method, url, kwargs["follow_redirects"]))
        return next(responses)

    monkeypatch.setattr(download_module.httpx, "stream", fake_stream)
    monkeypatch.setattr(
        download_module,
        "make_progress_bar",
        lambda total, *, description: (nullcontext(), lambda *_args: None),
    )

    destination = download_module.download_file(
        "https://example.com/signed",
        output=tmp_path,
        default_filename="data.jsonl",
        expected_size=11,
    )

    assert destination == tmp_path / "data.jsonl"
    assert destination.read_bytes() == b"hello world"
    assert requests == [
        ("GET", "https://example.com/signed", False),
        ("GET", "https://example.com/download/data.jsonl", False),
    ]


def test_download_file_rejects_insecure_redirect(monkeypatch, tmp_path) -> None:
    requests: list[str] = []

    def fake_stream(_method, url, **_kwargs):
        requests.append(url)
        return _FakeStreamResponse(
            status_code=302,
            headers={"location": "http://example.com/data.jsonl"},
        )

    monkeypatch.setattr(download_module.httpx, "stream", fake_stream)

    with pytest.raises(RuntimeError, match="Download redirect URL must use HTTPS"):
        download_module.download_file(
            "https://example.com/signed",
            output=tmp_path,
            default_filename="data.jsonl",
            expected_size=11,
        )

    assert requests == ["https://example.com/signed"]
    assert not list(tmp_path.iterdir())


def test_download_file_refuses_to_overwrite_existing_file(
    monkeypatch, tmp_path
) -> None:
    existing = tmp_path / "data.jsonl"
    existing.write_text("old", encoding="utf-8")

    monkeypatch.setattr(
        download_module.httpx,
        "stream",
        lambda *args, **kwargs: _FakeStreamResponse(
            headers={
                "content-disposition": 'attachment; filename="data.jsonl"',
                "content-length": "3",
            },
            chunks=[b"new"],
        ),
    )

    with pytest.raises(FileExistsError, match="File already exists"):
        download_module.download_file(
            "https://example.com/signed",
            output=tmp_path,
            default_filename="fallback",
            expected_size=3,
        )

    assert existing.read_text(encoding="utf-8") == "old"


def test_resolve_download_destination_rejects_missing_directory_intent(
    tmp_path,
) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(RuntimeError, match="Output directory not found"):
        download_module._resolve_download_destination(
            missing_dir,
            "data.jsonl",
            overwrite=False,
            output_is_directory=True,
        )

    assert not missing_dir.exists()


def test_download_file_reports_http_errors(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        download_module.httpx,
        "stream",
        lambda *args, **kwargs: _FakeStreamResponse(
            status_code=403,
            body=b"expired",
        ),
    )

    with pytest.raises(RuntimeError, match=r"HTTP 403.*expired"):
        download_module.download_file(
            "https://example.com/signed",
            output=tmp_path,
            default_filename="data.jsonl",
            expected_size=10,
        )
