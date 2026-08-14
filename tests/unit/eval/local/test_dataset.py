"""Dataset resolver, cache, and row-normalization contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.eval.local.dataset import (
    DatasetCache,
    DatasetDescription,
    DatasetResolutionError,
    normalize_row,
    parse_row_selector,
    resolve_explicit_dataset_file,
    resolve_platform_dataset,
    select_rows,
    sha256_of_file,
)

PROMPT_ROWS = [
    {"system_prompt": "be terse", "user_prompt": "2*3", "ground_truth": "6"},
    {"system_prompt": "be terse", "user_prompt": "4*5", "ground_truth": "20"},
    {"system_prompt": "be terse", "user_prompt": "6*7", "ground_truth": "42"},
    {"system_prompt": "be terse", "user_prompt": "8*9", "ground_truth": "72"},
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# --rows selector
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3", (3,)),
        ("3,7", (3, 7)),
        ("10-13", (10, 11, 12, 13)),
        ("7,3", (3, 7)),
        ("3,3,3", (3,)),
        ("5-6,6-7", (5, 6, 7)),
        (" 1,2 ", (1, 2)),
        ("4-4", (4,)),
    ],
)
def test_row_selector_dedupes_and_sorts(value: str, expected: tuple[int, ...]) -> None:
    assert parse_row_selector(value) == expected


@pytest.mark.parametrize("value", ["", "a", "1,", "-1", "1--2", "1,,2", "1 2", "1.5"])
def test_row_selector_rejects_malformed_input(value: str) -> None:
    with pytest.raises(DatasetResolutionError, match="--rows"):
        parse_row_selector(value)


def test_row_selector_rejects_an_inverted_range() -> None:
    with pytest.raises(DatasetResolutionError, match="inverted"):
        parse_row_selector("9-3")


# --------------------------------------------------------------------------- #
# Row normalization
# --------------------------------------------------------------------------- #


def test_prompt_mode_builds_system_then_user_messages() -> None:
    row = normalize_row(PROMPT_ROWS[0], row_index=0, source_row_index=3)
    assert row.initial_messages == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "2*3"},
    ]
    assert row.label == "6"
    assert row.metadata is None
    assert (row.row_index, row.source_row_index) == (0, 3)


def test_system_prompt_is_optional() -> None:
    row = normalize_row(
        {"user_prompt": "hi", "ground_truth": "1"}, row_index=0, source_row_index=0
    )
    assert row.initial_messages == [{"role": "user", "content": "hi"}]


def test_label_is_an_accepted_alias_for_ground_truth() -> None:
    row = normalize_row(
        {"user_prompt": "hi", "label": "7"}, row_index=0, source_row_index=0
    )
    assert row.label == "7"


def test_ground_truth_wins_over_label_when_both_are_present() -> None:
    raw = {"user_prompt": "hi", "ground_truth": "6", "label": "9"}
    assert normalize_row(raw, row_index=0, source_row_index=0).label == "6"


def test_prompt_mode_requires_user_prompt() -> None:
    with pytest.raises(DatasetResolutionError, match="user_prompt is required"):
        normalize_row({"ground_truth": "6"}, row_index=0, source_row_index=2)


def test_prompt_mode_requires_a_label() -> None:
    with pytest.raises(DatasetResolutionError, match="ground_truth"):
        normalize_row({"user_prompt": "hi"}, row_index=0, source_row_index=0)


def test_metadata_mode_makes_prompt_columns_optional() -> None:
    row = normalize_row(
        {"metadata": {"task": "multiply", "left": 15}}, row_index=0, source_row_index=0
    )
    assert row.initial_messages == []
    assert row.label is None
    assert row.metadata == {"task": "multiply", "left": 15}


def test_metadata_mode_accepts_an_encoded_json_object() -> None:
    row = normalize_row(
        {"metadata": '{"task": "multiply"}'}, row_index=0, source_row_index=0
    )
    assert row.metadata == {"task": "multiply"}


@pytest.mark.parametrize("value", ["[1, 2]", '"text"', "17"])
def test_metadata_must_decode_to_an_object(value: str) -> None:
    with pytest.raises(DatasetResolutionError, match="metadata must be a JSON object"):
        normalize_row({"metadata": value}, row_index=0, source_row_index=0)


def test_metadata_with_broken_json_is_reported_with_the_row() -> None:
    with pytest.raises(
        DatasetResolutionError, match="row 4: metadata is not valid JSON"
    ):
        normalize_row({"metadata": "{oops"}, row_index=0, source_row_index=4)


def test_columns_outside_the_contract_are_ignored() -> None:
    raw = {"user_prompt": "hi", "ground_truth": "6", "notes": "ignore me"}
    row = normalize_row(raw, row_index=0, source_row_index=0)
    assert row.initial_messages == [{"role": "user", "content": "hi"}]
    assert row.metadata is None


# --------------------------------------------------------------------------- #
# Row selection
# --------------------------------------------------------------------------- #


def test_no_limit_selects_every_row_in_order(tmp_path: Path) -> None:
    selection = select_rows(_write_jsonl(tmp_path / "d.jsonl", PROMPT_ROWS))
    assert selection.total_dataset_rows == 4
    assert selection.source_row_indices == (0, 1, 2, 3)
    assert [row.row_index for row in selection.rows] == [0, 1, 2, 3]


def test_positive_limit_takes_the_first_n_rows(tmp_path: Path) -> None:
    selection = select_rows(_write_jsonl(tmp_path / "d.jsonl", PROMPT_ROWS), limit=2)
    assert selection.source_row_indices == (0, 1)
    # total_dataset_rows still reports the whole file, for progress.json.
    assert selection.total_dataset_rows == 4


@pytest.mark.parametrize("limit", [0, -1, None])
def test_non_positive_limit_selects_every_row(
    tmp_path: Path, limit: int | None
) -> None:
    selection = select_rows(
        _write_jsonl(tmp_path / "d.jsonl", PROMPT_ROWS), limit=limit
    )
    assert selection.source_row_indices == (0, 1, 2, 3)


def test_limit_larger_than_the_dataset_is_harmless(tmp_path: Path) -> None:
    selection = select_rows(_write_jsonl(tmp_path / "d.jsonl", PROMPT_ROWS), limit=99)
    assert selection.source_row_indices == (0, 1, 2, 3)


def test_row_selector_renumbers_row_index_within_the_selected_set(
    tmp_path: Path,
) -> None:
    # The shared contract: row_index is the position in the selected set, and
    # source_row_index preserves the dataset offset for local UX.
    selection = select_rows(
        _write_jsonl(tmp_path / "d.jsonl", PROMPT_ROWS), row_selector=(1, 3)
    )
    assert [(r.row_index, r.source_row_index) for r in selection.rows] == [
        (0, 1),
        (1, 3),
    ]
    assert selection.rows[0].label == "20"
    assert selection.rows[1].label == "72"


def test_row_selector_overrides_limit(tmp_path: Path) -> None:
    selection = select_rows(
        _write_jsonl(tmp_path / "d.jsonl", PROMPT_ROWS), limit=1, row_selector=(2, 3)
    )
    assert selection.source_row_indices == (2, 3)


def test_out_of_range_row_selector_fails_before_the_run_exists(tmp_path: Path) -> None:
    with pytest.raises(DatasetResolutionError, match=r"selects \[9\].*has 4 rows"):
        select_rows(
            _write_jsonl(tmp_path / "d.jsonl", PROMPT_ROWS), row_selector=(0, 9)
        )


def test_an_empty_selection_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetResolutionError, match="selected no rows"):
        select_rows(_write_jsonl(tmp_path / "d.jsonl", []))


# --------------------------------------------------------------------------- #
# Format parity
# --------------------------------------------------------------------------- #


def test_jsonl_and_csv_normalize_identically(tmp_path: Path) -> None:
    jsonl = select_rows(_write_jsonl(tmp_path / "d.jsonl", PROMPT_ROWS))
    csv_path = tmp_path / "d.csv"
    header = "system_prompt,user_prompt,ground_truth\n"
    body = "".join(
        f"{row['system_prompt']},{row['user_prompt']},{row['ground_truth']}\n"
        for row in PROMPT_ROWS
    )
    csv_path.write_text(header + body, encoding="utf-8")
    assert select_rows(csv_path).rows == jsonl.rows


def test_parquet_normalizes_identically(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    jsonl = select_rows(_write_jsonl(tmp_path / "d.jsonl", PROMPT_ROWS))
    table = pa.table(
        {
            key: [row[key] for row in PROMPT_ROWS]
            for key in ("system_prompt", "user_prompt", "ground_truth")
        }
    )
    parquet_path = tmp_path / "d.parquet"
    pq.write_table(table, parquet_path)
    assert select_rows(parquet_path).rows == jsonl.rows


def test_unknown_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "d.txt"
    path.write_text("nope")
    with pytest.raises(DatasetResolutionError, match="expected one of"):
        select_rows(path)


def test_malformed_jsonl_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text('{"user_prompt": "a", "ground_truth": "1"}\n{oops\n')
    with pytest.raises(DatasetResolutionError, match=r"d\.jsonl:2: invalid JSON"):
        select_rows(path)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def test_explicit_dataset_file_wins_and_is_hashed(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "dev.jsonl", PROMPT_ROWS)
    resolved = resolve_explicit_dataset_file(path)
    assert resolved.source == "explicit"
    assert resolved.sha256 == sha256_of_file(path)
    assert resolved.extension == "jsonl"


def test_explicit_dataset_file_must_exist(tmp_path: Path) -> None:
    with pytest.raises(DatasetResolutionError, match="is not a file"):
        resolve_explicit_dataset_file(tmp_path / "absent.jsonl")


def test_dataset_fingerprint_ignores_the_path(tmp_path: Path) -> None:
    first = _write_jsonl(tmp_path / "a.jsonl", PROMPT_ROWS)
    second = _write_jsonl(tmp_path / "b.jsonl", PROMPT_ROWS)
    assert sha256_of_file(first) == sha256_of_file(second)


class FakeFetcher:
    """Records platform calls so cache hits can be asserted as zero downloads."""

    def __init__(self, *, rows: list[dict[str, Any]], version: str = "v1") -> None:
        self.rows = rows
        self.version = version
        self.describe_calls = 0
        self.download_calls = 0
        self.describe_error: Exception | None = None

    def describe(self, dataset_name: str) -> DatasetDescription:
        self.describe_calls += 1
        if self.describe_error is not None:
            raise self.describe_error
        return DatasetDescription(
            dataset_id="ds-1",
            dataset_name=dataset_name,
            extension="jsonl",
            version=self.version,
            row_count=len(self.rows),
            organization_id="org-1",
        )

    def download_to(self, dataset_name: str, destination: Path) -> None:
        self.download_calls += 1
        _write_jsonl(destination, self.rows)


def test_platform_dataset_downloads_then_caches(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path / "cache", git_identity="github.com/acme/repo")
    fetcher = FakeFetcher(rows=PROMPT_ROWS)

    first = resolve_platform_dataset("multiply", cache=cache, fetcher=fetcher)
    assert first.source == "download"
    assert fetcher.download_calls == 1
    assert select_rows(first.path).total_dataset_rows == 4

    second = resolve_platform_dataset("multiply", cache=cache, fetcher=fetcher)
    assert second.source == "cache"
    assert second.sha256 == first.sha256
    assert fetcher.download_calls == 1


def test_cache_is_content_addressed(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path / "cache", git_identity="github.com/acme/repo")
    resolved = resolve_platform_dataset(
        "multiply", cache=cache, fetcher=FakeFetcher(rows=PROMPT_ROWS)
    )
    assert resolved.path.name == f"{resolved.sha256}.jsonl"
    metadata = json.loads((resolved.path.parent / "metadata.json").read_text())
    assert metadata["sha256"] == resolved.sha256
    assert metadata["row_count"] == 4
    assert metadata["organization_id"] == "org-1"


def test_a_new_platform_version_invalidates_the_cache(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path / "cache", git_identity="github.com/acme/repo")
    fetcher = FakeFetcher(rows=PROMPT_ROWS)
    resolve_platform_dataset("multiply", cache=cache, fetcher=fetcher)

    fetcher.version = "v2"
    fetcher.rows = [*PROMPT_ROWS[:2], {"user_prompt": "1*1", "ground_truth": "1"}]
    refreshed = resolve_platform_dataset("multiply", cache=cache, fetcher=fetcher)
    assert refreshed.source == "download"
    assert fetcher.download_calls == 2
    assert select_rows(refreshed.path).total_dataset_rows == 3


def test_a_corrupted_cache_entry_is_not_a_hit(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path / "cache", git_identity="github.com/acme/repo")
    fetcher = FakeFetcher(rows=PROMPT_ROWS)
    resolved = resolve_platform_dataset("multiply", cache=cache, fetcher=fetcher)
    resolved.path.write_text("truncated")

    again = resolve_platform_dataset("multiply", cache=cache, fetcher=fetcher)
    assert again.source == "download"
    assert fetcher.download_calls == 2
    assert sha256_of_file(again.path) == again.sha256


def test_an_unreachable_platform_falls_back_to_a_verified_cache(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path / "cache", git_identity="github.com/acme/repo")
    fetcher = FakeFetcher(rows=PROMPT_ROWS)
    first = resolve_platform_dataset("multiply", cache=cache, fetcher=fetcher)

    fetcher.describe_error = RuntimeError("connection refused")
    notes: list[str] = []
    fallback = resolve_platform_dataset(
        "multiply", cache=cache, fetcher=fetcher, on_event=notes.append
    )
    assert fallback.sha256 == first.sha256
    assert any("platform unreachable" in note for note in notes)


def test_an_unreachable_platform_with_no_cache_asks_for_dataset_file(
    tmp_path: Path,
) -> None:
    cache = DatasetCache(tmp_path / "cache", git_identity="github.com/acme/repo")
    fetcher = FakeFetcher(rows=PROMPT_ROWS)
    fetcher.describe_error = RuntimeError("connection refused")
    with pytest.raises(DatasetResolutionError, match="--dataset-file"):
        resolve_platform_dataset("multiply", cache=cache, fetcher=fetcher)


def test_caches_are_scoped_per_workspace(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    first = DatasetCache(root, git_identity="github.com/acme/one")
    second = DatasetCache(root, git_identity="github.com/acme/two")
    assert first.directory_for("ds-1") != second.directory_for("ds-1")


def test_a_dataset_resolution_error_from_describe_is_not_masked(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path / "cache", git_identity="github.com/acme/repo")
    fetcher = FakeFetcher(rows=PROMPT_ROWS)
    fetcher.describe_error = DatasetResolutionError("still processing")
    with pytest.raises(DatasetResolutionError, match="still processing"):
        resolve_platform_dataset("multiply", cache=cache, fetcher=fetcher)
