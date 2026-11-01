"""Archive-mode eval import: ZIP64 layout, file list, and streamed verification."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from osmosis_ai.eval.local.upload import (
    EvalUploadFile,
    LocalEvalUploadError,
    _hash_file,
)
from osmosis_ai.eval.local.upload_archive import (
    ArchiveLayout,
    RunFileReader,
    build_run_archive,
)


@dataclass(frozen=True)
class _Entry:
    path: str
    size: int
    crc32: int


class _MemorySource:
    def __init__(self, bodies: list[tuple[str, bytes]]) -> None:
        self.bodies = [body for _, body in bodies]
        self.entries = [
            _Entry(path, len(body), zlib.crc32(body)) for path, body in bodies
        ]
        self.bytes_read = 0

    def read(self, index: int, offset: int, length: int) -> bytes:
        self.bytes_read += length
        return self.bodies[index][offset : offset + length]


def _stream(layout: ArchiveLayout, source: _MemorySource) -> bytes:
    return b"".join(layout.iter_bytes(0, layout.size, source.read))


def _contract_size(bodies: list[tuple[str, bytes]]) -> int:
    names = [len(path.encode()) for path, _ in bodies]
    return (
        sum(30 + n + 20 + len(body) for n, (_, body) in zip(names, bodies, strict=True))
        + sum(46 + n + 28 for n in names)
        + 98
    )


BODIES = [
    ("a/empty-first.txt", b""),
    ("dataset_rows.jsonl", b'{"row_index": 0}\n'),
    ("index.jsonl", b'{"rollout_id": "x"}\n' * 40),
    ("logs.txt", b""),
    ("progress.json", b'{"total_runs": 1}\n'),
    ("rollout_trials/r/artifacts/café 文件.bin", bytes(range(256)) * 9),
    ("rollout_trials/r/trajectory.json", b"{}"),
    ("z/empty-last.txt", b""),
]


def test_archive_has_the_contract_layout_and_opens_with_zipfile() -> None:
    source = _MemorySource(BODIES)
    layout = ArchiveLayout(source.entries)
    blob = _stream(layout, source)

    assert layout.size == len(blob) == _contract_size(BODIES)
    archive = zipfile.ZipFile(io.BytesIO(blob))
    assert archive.testzip() is None
    assert archive.namelist() == [path for path, _ in BODIES]
    for info, (_, body) in zip(archive.infolist(), BODIES, strict=True):
        assert archive.read(info) == body
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.flag_bits == 0x0800
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert (info.create_system, info.create_version) == (3, 45)
        assert info.extract_version == 45
        assert info.external_attr == 0o100644 << 16
        assert zlib.crc32(body) == info.CRC
        assert info.comment == b""
        local = struct.unpack_from("<IHHHHHIIIHH", blob, info.header_offset)
        assert local[7:9] == (0xFFFFFFFF, 0xFFFFFFFF)
        name_end = info.header_offset + 30 + local[9]
        assert blob[name_end : name_end + 20] == struct.pack(
            "<HHQQ", 1, 16, len(body), len(body)
        )
        assert info.extra == struct.pack(
            "<HHQQQ", 1, 24, len(body), len(body), info.header_offset
        )
    assert blob[-22:] == struct.pack(
        "<IHHHHIIH", 0x06054B50, 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0
    )


def test_archive_with_more_than_65535_entries_opens_with_zipfile() -> None:
    bodies = [(f"e/{index:06d}", bytes([index % 251])) for index in range(65_540)]
    source = _MemorySource(bodies)
    layout = ArchiveLayout(source.entries)
    blob = _stream(layout, source)

    assert len(blob) == layout.size == _contract_size(bodies)
    archive = zipfile.ZipFile(io.BytesIO(blob))
    infos = archive.infolist()
    assert len(infos) == len(bodies)
    for index in (0, 65_534, 65_535, 65_536, len(bodies) - 1):
        assert infos[index].filename == bodies[index][0]
        assert archive.read(infos[index]) == bodies[index][1]


@pytest.mark.parametrize("part_size", [1, 2, 49, 50, 51, 97, 4096])
def test_any_part_regenerated_alone_equals_its_slice_of_the_stream(
    part_size: int,
) -> None:
    full_source = _MemorySource(BODIES)
    layout = ArchiveLayout(full_source.entries)
    full = _stream(layout, full_source)

    for start in range(0, layout.size, part_size):
        stop = min(layout.size, start + part_size)
        source = _MemorySource(BODIES)
        part = b"".join(layout.iter_bytes(start, stop, source.read))
        assert part == full[start:stop]
        assert source.bytes_read <= stop - start


def test_parts_spanning_large_entries_read_only_their_segments() -> None:
    bodies = [("big.bin", bytes(range(256)) * 10_000), ("small.txt", b"tail")]
    full_source = _MemorySource(bodies)
    layout = ArchiveLayout(full_source.entries)
    full = _stream(layout, full_source)
    part_size = 1024 * 1024 + 7

    for start in range(0, layout.size, part_size):
        stop = min(layout.size, start + part_size)
        source = _MemorySource(bodies)
        assert b"".join(layout.iter_bytes(start, stop, source.read)) == full[start:stop]
        assert source.bytes_read <= stop - start


def _planned(tmp_path: Path, bodies: list[tuple[str, bytes]]) -> list[EvalUploadFile]:
    files = []
    for name, body in bodies:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        files.append(_hash_file(path, relative=name))
    return files


def test_file_list_is_sorted_path_size_sha256_lines(tmp_path: Path) -> None:
    files = _planned(tmp_path, BODIES)

    archive = build_run_archive(files)

    assert archive.file_list.endswith(b"\n")
    rows = [json.loads(line) for line in archive.file_list.decode().splitlines()]
    assert [list(row) for row in rows] == [["path", "size", "sha256"]] * len(BODIES)
    assert rows == [
        {"path": path, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        for path, body in BODIES
    ]
    assert archive.to_request() == {
        "size": _contract_size(BODIES),
        "file_count": len(BODIES),
        "total_bytes": sum(len(body) for _, body in BODIES),
        "file_list": {
            "size": len(archive.file_list),
            "sha256": hashlib.sha256(archive.file_list).hexdigest(),
        },
    }


def test_reader_streams_real_files_and_regenerates_a_part_alone(
    tmp_path: Path,
) -> None:
    files = _planned(tmp_path, BODIES)
    layout = ArchiveLayout(files)
    with RunFileReader(files) as reader:
        full = b"".join(layout.iter_bytes(0, layout.size, reader.read))
        reader.finish()

    archive = zipfile.ZipFile(io.BytesIO(full))
    assert [archive.read(path) for path, _ in BODIES] == [body for _, body in BODIES]
    start, stop = 100, layout.size // 2
    with RunFileReader(files) as reader:
        assert b"".join(layout.iter_bytes(start, stop, reader.read)) == full[start:stop]


@pytest.mark.parametrize("field", ["sha256", "crc32"])
def test_reader_rejects_bytes_that_differ_from_the_plan(
    tmp_path: Path, field: str
) -> None:
    files = _planned(tmp_path, BODIES)
    index = next(i for i, file in enumerate(files) if file.path == "index.jsonl")
    wrong = "0" * 64 if field == "sha256" else files[index].crc32 ^ 1
    files[index] = dataclasses.replace(files[index], **{field: wrong})
    layout = ArchiveLayout(files)

    with RunFileReader(files) as reader:
        with pytest.raises(LocalEvalUploadError, match=r"index\.jsonl changed after"):
            for _ in layout.iter_bytes(0, layout.size, reader.read):
                pass


def test_finish_requires_every_file_to_be_verified_in_order(tmp_path: Path) -> None:
    files = _planned(tmp_path, BODIES)
    layout = ArchiveLayout(files)

    with RunFileReader(files) as reader:
        for _ in layout.iter_bytes(layout.size // 2, layout.size, reader.read):
            pass
        with pytest.raises(LocalEvalUploadError, match="did not verify every"):
            reader.finish()
