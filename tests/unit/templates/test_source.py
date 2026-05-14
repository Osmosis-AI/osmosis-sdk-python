"""Unit tests for template source resolution."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import pytest
import requests

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.templates import source


def test_workspace_template_root_refresh_redownloads_cached_checkout(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("OSMOSIS_WORKSPACE_TEMPLATE_PATH", raising=False)
    monkeypatch.setattr(source, "CACHE_DIR", tmp_path)

    calls: list[Path] = []

    def fake_download_workspace_template(
        repo: str, ref: str, destination: Path
    ) -> None:
        del repo, ref
        calls.append(destination)
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        (destination / "version.txt").write_text(str(len(calls)), encoding="utf-8")

    monkeypatch.setattr(
        source, "_download_workspace_template", fake_download_workspace_template
    )

    first = source.workspace_template_root()
    assert (first / "version.txt").read_text(encoding="utf-8") == "1"

    (first / "version.txt").write_text("stale", encoding="utf-8")
    second = source.workspace_template_root(refresh=True)

    assert second == first
    assert calls == [first, first]
    assert (second / "version.txt").read_text(encoding="utf-8") == "2"


def test_safe_extract_uses_data_filter(tmp_path: Path, monkeypatch) -> None:
    archive_path = tmp_path / "template.tar"
    payload = tmp_path / "payload.txt"
    payload.write_text("ok", encoding="utf-8")
    with tarfile.open(archive_path, "w") as archive:
        archive.add(payload, arcname="workspace-template-main/payload.txt")

    calls: list[object] = []

    def fake_extractall(
        self, path=".", members=None, *, numeric_owner=False, filter=None
    ):
        del self, path, members, numeric_owner
        calls.append(filter)

    monkeypatch.setattr(tarfile.TarFile, "extractall", fake_extractall)

    with tarfile.open(archive_path, "r") as archive:
        source._safe_extract(archive, tmp_path / "extract")

    assert calls == ["data"]


def test_missing_override_path_uses_user_facing_template_terms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv("OSMOSIS_WORKSPACE_TEMPLATE_PATH", str(missing))

    with pytest.raises(CLIError) as exc_info:
        source.workspace_template_root()

    message = str(exc_info.value).lower()
    assert "configured template path does not exist" in message
    assert "workspace template" not in message


def test_download_failure_uses_user_facing_template_terms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_get(*args, **kwargs):
        del args, kwargs
        raise requests.RequestException("offline")

    monkeypatch.setattr(source.requests, "get", fail_get)

    with pytest.raises(CLIError) as exc_info:
        source._download_workspace_template(
            "Osmosis-AI/workspace-template", "main", tmp_path / "templates"
        )

    message = str(exc_info.value).lower()
    assert "unable to fetch starter templates" in message
    assert "workspace template" not in message
