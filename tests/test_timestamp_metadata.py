import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from liq.store import timestamp_metadata
from liq.store.timestamp_metadata import (
    PinnedParquetFile,
    TimestampMetadataRefused,
    inspect_timestamp_metadata,
)


@pytest.fixture
def empty_file(tmp_path: Path) -> tuple[Path, PinnedParquetFile]:
    path = tmp_path / "metadata/empty.parquet"
    path.parent.mkdir()
    pq.write_table(pa.table({"timestamp": pa.array([], type=pa.timestamp("us", tz="UTC"))}), path)
    return tmp_path, PinnedParquetFile(
        "metadata/empty.parquet", hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_empty_schema_file_is_unverified_without_page_reads(empty_file, monkeypatch) -> None:
    root, identity = empty_file

    def forbidden(*_args, **_kwargs):
        pytest.fail("footer inspection attempted data pages")

    monkeypatch.setattr(pq, "read_table", forbidden)
    with pytest.raises(TimestampMetadataRefused, match="empty"):
        inspect_timestamp_metadata(
            root, (identity,), max_files=1, max_file_bytes=100000, max_row_groups=10
        )


def test_wrong_digest_refuses_before_footer_parser(empty_file, monkeypatch) -> None:
    root, identity = empty_file
    wrong = PinnedParquetFile(identity.path, "0" * 64)
    monkeypatch.setattr(pq, "read_metadata", lambda *_: pytest.fail("changed bytes reached parser"))
    with pytest.raises(TimestampMetadataRefused, match="digest"):
        inspect_timestamp_metadata(
            root, (wrong,), max_files=1, max_file_bytes=100000, max_row_groups=10
        )


@pytest.mark.parametrize("path", ["../outside.parquet", "/outside.parquet", "./empty.parquet"])
def test_noncanonical_identity_refuses_before_file_read(tmp_path: Path, path: str) -> None:
    with pytest.raises(TimestampMetadataRefused, match="path"):
        inspect_timestamp_metadata(
            tmp_path,
            (PinnedParquetFile(path, "0" * 64),),
            max_files=1,
            max_file_bytes=100000,
            max_row_groups=10,
        )


def test_symlink_file_is_refused(empty_file) -> None:
    root, identity = empty_file
    link = root / "metadata/linked.parquet"
    link.symlink_to(root / identity.path)
    with pytest.raises((OSError, TimestampMetadataRefused)):
        inspect_timestamp_metadata(
            root,
            (PinnedParquetFile("metadata/linked.parquet", identity.sha256),),
            max_files=1,
            max_file_bytes=100000,
            max_row_groups=10,
        )


def test_file_size_budget_refuses_before_parser(empty_file, monkeypatch) -> None:
    root, identity = empty_file
    monkeypatch.setattr(
        pq, "read_metadata", lambda *_: pytest.fail("oversized bytes reached parser")
    )
    with pytest.raises(TimestampMetadataRefused, match="size"):
        inspect_timestamp_metadata(
            root, (identity,), max_files=1, max_file_bytes=1, max_row_groups=10
        )


@pytest.mark.parametrize("field", ["max_files", "max_file_bytes", "max_row_groups"])
def test_invalid_budget_refuses(empty_file, field: str) -> None:
    root, identity = empty_file
    budgets = {"max_files": 1, "max_file_bytes": 100000, "max_row_groups": 10}
    budgets[field] = 0
    with pytest.raises(TimestampMetadataRefused, match="budget"):
        inspect_timestamp_metadata(root, (identity,), **budgets)


def test_unparseable_footer_is_a_typed_refusal(tmp_path: Path) -> None:
    payload = b"Nonfinancial invalid-format sentinel"
    (tmp_path / "invalid.parquet").write_bytes(payload)
    with pytest.raises(TimestampMetadataRefused, match="footer"):
        inspect_timestamp_metadata(
            tmp_path,
            (PinnedParquetFile("invalid.parquet", hashlib.sha256(payload).hexdigest()),),
            max_files=1,
            max_file_bytes=100000,
            max_row_groups=10,
        )


@pytest.fixture
def metadata_objects(monkeypatch):
    stats = SimpleNamespace(
        has_min_max=True,
        has_null_count=True,
        null_count=0,
        num_values=1,
        min_raw=1,
        max_raw=2,
        physical_type="INT64",
    )
    column = SimpleNamespace(
        path_in_schema="timestamp", file_path="", statistics=stats, num_values=1
    )
    group = SimpleNamespace(num_rows=1, column=lambda _index: column)
    logical = {"Type": "Timestamp", "isAdjustedToUTC": True, "timeUnit": "nanoseconds"}
    column_schema = SimpleNamespace(
        path="timestamp",
        physical_type="INT64",
        logical_type=SimpleNamespace(to_json=lambda: json.dumps(logical)),
    )
    schema = SimpleNamespace(
        names=["timestamp"],
        column=lambda _index: column_schema,
        to_arrow_schema=lambda: pa.schema([("timestamp", pa.timestamp("ns", tz="UTC"))]),
    )
    metadata = SimpleNamespace(
        num_rows=1, num_row_groups=1, schema=schema, row_group=lambda _index: group
    )
    monkeypatch.setattr(pq, "read_metadata", lambda *_: metadata)
    return metadata, group, column, stats, logical


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("has_min_max", False, "missing"),
        ("has_null_count", False, "missing"),
        ("null_count", None, "null"),
        ("null_count", 1, "null"),
        ("num_values", 2, "count"),
        ("min_raw", 3, "extrema"),
        ("min_raw", True, "extrema"),
        ("physical_type", "INT96", "extrema"),
    ],
)
def test_incomplete_or_inconsistent_metadata_objects_refuse(
    metadata_objects, field, value, reason
) -> None:
    _, _, _, stats, _ = metadata_objects
    setattr(stats, field, value)
    with pytest.raises(TimestampMetadataRefused, match=reason):
        timestamp_metadata._inspect_file(
            b"nonfinancial metadata test", PinnedParquetFile("metadata.parquet", "0" * 64), 10
        )


@pytest.mark.parametrize(
    "case,reason",
    [
        ("no_statistics", "missing"),
        ("empty_group", "empty"),
        ("external", "external"),
        ("file_total", "count"),
        ("unknown_unit", "encoding"),
        ("not_utc", "encoding"),
        ("group_budget", "budget"),
        ("ambiguous", "ambiguous"),
    ],
)
def test_unverified_metadata_structure_refuses(metadata_objects, case, reason) -> None:
    metadata, group, column, _, logical = metadata_objects
    if case == "no_statistics":
        column.statistics = None
    elif case == "empty_group":
        group.num_rows = 0
    elif case == "external":
        column.file_path = "external.parquet"
    elif case == "file_total":
        metadata.num_rows = 2
    elif case == "unknown_unit":
        logical["timeUnit"] = "unknown"
    elif case == "not_utc":
        logical["isAdjustedToUTC"] = False
    elif case == "group_budget":
        metadata.num_row_groups = 11
    elif case == "ambiguous":
        metadata.schema.names = ["timestamp", "timestamp"]
    with pytest.raises(TimestampMetadataRefused, match=reason):
        timestamp_metadata._inspect_file(
            b"nonfinancial metadata test", PinnedParquetFile("metadata.parquet", "0" * 64), 10
        )
