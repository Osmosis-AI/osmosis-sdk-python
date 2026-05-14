"""Unit tests for workspace template source resolution."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

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
