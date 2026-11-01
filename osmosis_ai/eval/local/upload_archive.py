"""Deterministic ``files.jsonl`` and ZIP64 ``run.zip`` for an eval import.

Every header is computed from the upload plan, so the archive's exact size is
known before upload and any byte range can be regenerated on its own. The byte
layout is a contract with the platform's extraction Lambda: stored entries in
path order, every entry ZIP64, a fixed 1980-01-01 timestamp, no comments.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from types import TracebackType
from typing import IO, Any, Protocol

from osmosis_ai.eval.local.upload import EvalUploadFile, LocalEvalUploadError

FILE_LIST_PATH = "files.jsonl"
ARCHIVE_PATH = "run.zip"

_READ_CHUNK_SIZE = 1024 * 1024

_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_ZIP64_EXTRA = struct.Struct("<HHQQ")
_CENTRAL_HEADER = struct.Struct("<IHHHHHHIIIHHHHHII")
_CENTRAL_ZIP64_EXTRA = struct.Struct("<HHQQQ")
_ZIP64_END = struct.Struct("<IQHHIIQQQQ")
_ZIP64_LOCATOR = struct.Struct("<IIQI")
_END = struct.Struct("<IHHHHIIH")

_LOCAL_SIGNATURE = 0x04034B50
_CENTRAL_SIGNATURE = 0x02014B50
_ZIP64_END_SIGNATURE = 0x06064B50
_ZIP64_LOCATOR_SIGNATURE = 0x07064B50
_END_SIGNATURE = 0x06054B50
_ZIP64_EXTRA_TAG = 0x0001
_VERSION_MADE_BY = 0x032D
_VERSION_NEEDED = 45
_UTF8_FLAG = 0x0800
_STORED = 0
_DOS_TIME = 0
_DOS_DATE = 0x0021
_EXTERNAL_ATTRIBUTES = 0o100644 << 16
_MAX_16 = 0xFFFF
_MAX_32 = 0xFFFFFFFF

_LOCAL_OVERHEAD = _LOCAL_HEADER.size + _LOCAL_ZIP64_EXTRA.size
_CENTRAL_OVERHEAD = _CENTRAL_HEADER.size + _CENTRAL_ZIP64_EXTRA.size

EntryReader = Callable[[int, int, int], bytes]
"""``(entry index, offset in entry, length) -> exactly that many bytes``."""


class ArchiveEntry(Protocol):
    @property
    def path(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def crc32(self) -> int: ...


class ArchiveLayout:
    """Byte layout of the archive for entries already sorted by path."""

    def __init__(self, entries: Sequence[ArchiveEntry]) -> None:
        self._entries = tuple(entries)
        self._names = tuple(entry.path.encode("utf-8") for entry in self._entries)
        self._local_offsets: list[int] = []
        offset = 0
        for entry, name in zip(self._entries, self._names, strict=True):
            if len(name) > _MAX_16:
                raise ValueError(f"Archive entry name is too long: {entry.path!r}")
            self._local_offsets.append(offset)
            offset += _LOCAL_OVERHEAD + len(name) + entry.size
        self._central_offset = offset
        self._central_offsets: list[int] = []
        for name in self._names:
            self._central_offsets.append(offset)
            offset += _CENTRAL_OVERHEAD + len(name)
        self._trailer_offset = offset
        self._trailer = self._build_trailer()
        self.size: int = offset + len(self._trailer)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def iter_bytes(self, start: int, stop: int, read: EntryReader) -> Iterator[bytes]:
        """Yield archive bytes ``[start, stop)``, reading only the entry data in range.

        An empty entry yields no data, but ``read(index, 0, 0)`` is still called in
        the range holding the end of its header so a verifying reader sees it.
        """
        if not 0 <= start <= stop <= self.size:
            raise ValueError(f"Archive range [{start}, {stop}) is out of bounds")
        position = start
        if position < self._central_offset:
            first = bisect_right(self._local_offsets, position) - 1
            for index in range(first, len(self._entries)):
                header_start = self._local_offsets[index]
                if header_start >= stop:
                    break
                header = self._local_header(index)
                data_start = header_start + len(header)
                if position < data_start:
                    end = min(stop, data_start)
                    yield header[position - header_start : end - header_start]
                    position = end
                    if position < data_start:
                        break
                size = self._entries[index].size
                if size == 0:
                    read(index, 0, 0)
                    continue
                end = min(stop, data_start + size)
                while position < end:
                    length = min(_READ_CHUNK_SIZE, end - position)
                    chunk = read(index, position - data_start, length)
                    if len(chunk) != length:
                        raise RuntimeError(
                            f"Archive entry {self._entries[index].path!r} returned "
                            f"{len(chunk)} bytes; expected {length}"
                        )
                    yield chunk
                    position += length
        if self._central_offset <= position < min(stop, self._trailer_offset):
            index = bisect_right(self._central_offsets, position) - 1
            while index < len(self._entries) and position < stop:
                record_start = self._central_offsets[index]
                record = self._central_header(index)
                end = min(stop, record_start + len(record))
                yield record[position - record_start : end - record_start]
                position = end
                index += 1
        if position < stop:
            yield self._trailer[
                position - self._trailer_offset : stop - self._trailer_offset
            ]

    def _local_header(self, index: int) -> bytes:
        entry = self._entries[index]
        name = self._names[index]
        return (
            _LOCAL_HEADER.pack(
                _LOCAL_SIGNATURE,
                _VERSION_NEEDED,
                _UTF8_FLAG,
                _STORED,
                _DOS_TIME,
                _DOS_DATE,
                entry.crc32,
                _MAX_32,
                _MAX_32,
                len(name),
                _LOCAL_ZIP64_EXTRA.size,
            )
            + name
            + _LOCAL_ZIP64_EXTRA.pack(
                _ZIP64_EXTRA_TAG,
                _LOCAL_ZIP64_EXTRA.size - 4,
                entry.size,
                entry.size,
            )
        )

    def _central_header(self, index: int) -> bytes:
        entry = self._entries[index]
        name = self._names[index]
        return (
            _CENTRAL_HEADER.pack(
                _CENTRAL_SIGNATURE,
                _VERSION_MADE_BY,
                _VERSION_NEEDED,
                _UTF8_FLAG,
                _STORED,
                _DOS_TIME,
                _DOS_DATE,
                entry.crc32,
                _MAX_32,
                _MAX_32,
                len(name),
                _CENTRAL_ZIP64_EXTRA.size,
                0,
                0,
                0,
                _EXTERNAL_ATTRIBUTES,
                _MAX_32,
            )
            + name
            + _CENTRAL_ZIP64_EXTRA.pack(
                _ZIP64_EXTRA_TAG,
                _CENTRAL_ZIP64_EXTRA.size - 4,
                entry.size,
                entry.size,
                self._local_offsets[index],
            )
        )

    def _build_trailer(self) -> bytes:
        count = len(self._entries)
        central_size = self._trailer_offset - self._central_offset
        return (
            _ZIP64_END.pack(
                _ZIP64_END_SIGNATURE,
                _ZIP64_END.size - 12,
                _VERSION_MADE_BY,
                _VERSION_NEEDED,
                0,
                0,
                count,
                count,
                central_size,
                self._central_offset,
            )
            + _ZIP64_LOCATOR.pack(_ZIP64_LOCATOR_SIGNATURE, 0, self._trailer_offset, 1)
            + _END.pack(_END_SIGNATURE, 0, 0, _MAX_16, _MAX_16, _MAX_32, _MAX_32, 0)
        )


@dataclass(frozen=True)
class RunArchive:
    """The two uploads that carry one run: its file list and its archive."""

    file_list: bytes
    file_list_sha256: str
    layout: ArchiveLayout
    total_bytes: int

    def to_request(self) -> dict[str, Any]:
        return {
            "size": self.layout.size,
            "file_count": self.layout.entry_count,
            "total_bytes": self.total_bytes,
            "file_list": {
                "size": len(self.file_list),
                "sha256": self.file_list_sha256,
            },
        }


def build_run_archive(files: Sequence[EvalUploadFile]) -> RunArchive:
    """Describe *files*, already sorted by path, as ``files.jsonl`` and ``run.zip``."""
    file_list = b"".join(
        json.dumps(file.to_request(), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
        for file in files
    )
    return RunArchive(
        file_list=file_list,
        file_list_sha256=hashlib.sha256(file_list).hexdigest(),
        layout=ArchiveLayout(files),
        total_bytes=sum(file.size for file in files),
    )


class RunFileReader:
    """Read planned files for the archive, verifying them as they stream.

    Entries read in order are checked against the plan's size, SHA-256, and
    CRC-32; :meth:`finish` fails unless every entry was verified that way.
    """

    def __init__(self, files: Sequence[EvalUploadFile]) -> None:
        self._files = tuple(files)
        self._stack = ExitStack()
        self._open_index: int | None = None
        self._handle: IO[bytes] | None = None
        self._next = 0
        self._verified = 0
        self._sha256 = hashlib.sha256()
        self._crc32 = 0

    def __enter__(self) -> RunFileReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._open_index = None
        self._handle = None
        self._stack.__exit__(exc_type, exc, tb)

    def read(self, index: int, offset: int, length: int) -> bytes:
        file = self._files[index]
        if self._open_index != index or self._handle is None:
            self._close()
            self._handle = self._stack.enter_context(file.open_verified())
            self._open_index = index
        self._handle.seek(offset)
        data = self._handle.read(length)
        if len(data) != length:
            raise _changed(file)
        if index == self._next and offset == self._verified:
            self._sha256.update(data)
            self._crc32 = zlib.crc32(data, self._crc32)
            self._verified += length
            if self._verified == file.size:
                if self._sha256.hexdigest() != file.sha256 or self._crc32 != file.crc32:
                    raise _changed(file)
                self._next += 1
                self._verified = 0
                self._sha256 = hashlib.sha256()
                self._crc32 = 0
        return data

    def finish(self) -> None:
        """Close the open file and require that every planned file was verified."""
        self._close()
        if self._next != len(self._files):
            raise LocalEvalUploadError(
                "The archive upload did not verify every planned file; retry"
            )

    def _close(self) -> None:
        self._open_index = None
        self._handle = None
        self._stack.close()


def _changed(file: EvalUploadFile) -> LocalEvalUploadError:
    return LocalEvalUploadError(
        f"{file.path} changed after the upload plan was built; retry"
    )
