"""Helper operations: gap detection and consolidation."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from liq.store.parquet import ParquetStore


def test_get_gaps_detects_missing_intervals(temp_storage_path: Path) -> None:
    store = ParquetStore(str(temp_storage_path))
    ts0 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    ts1 = datetime(2024, 1, 1, 11, 0, tzinfo=UTC)
    ts3 = datetime(2024, 1, 1, 13, 0, tzinfo=UTC)  # gap at 12:00
    df = pl.DataFrame({
        "timestamp": [ts0, ts1, ts3],
        "symbol": ["EUR_USD"] * 3,
        "value": [1.0, 2.0, 4.0],
    })
    store.write("forex/EUR_USD", df)

    gaps = store.get_gaps(
        "forex/EUR_USD",
        start=ts0,
        end=ts3,
        expected_interval=timedelta(hours=1),
    )
    assert len(gaps) == 1
    gap_start, gap_end = gaps[0]
    assert gap_start == datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    assert gap_end == datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


def test_get_gaps_no_gaps_returns_empty(temp_storage_path: Path) -> None:
    store = ParquetStore(str(temp_storage_path))
    ts0 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    df = pl.DataFrame({
        "timestamp": [ts0, ts0 + timedelta(hours=1)],
        "value": [1.0, 2.0],
    })
    store.write("test/no_gaps", df)

    gaps = store.get_gaps(
        "test/no_gaps",
        start=ts0,
        end=ts0 + timedelta(hours=1),
        expected_interval=timedelta(hours=1),
    )
    assert gaps == []


def test_get_gaps_head_and_tail(temp_storage_path: Path) -> None:
    store = ParquetStore(str(temp_storage_path))
    ts1 = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
    ts2 = datetime(2024, 1, 2, 13, 0, tzinfo=UTC)
    df = pl.DataFrame({"timestamp": [ts1, ts2], "value": [1.0, 2.0]})
    store.write("test/head_tail", df)

    start = datetime(2024, 1, 2, 10, 0, tzinfo=UTC)
    end = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
    gaps = store.get_gaps("test/head_tail", start=start, end=end, expected_interval=timedelta(hours=1))
    # Expect gap before first and after last
    assert gaps[0][0] == start
    assert gaps[-1][0] == datetime(2024, 1, 2, 14, 0, tzinfo=UTC)


def test_consolidate_reduces_file_count(temp_storage_path: Path) -> None:
    store = ParquetStore(str(temp_storage_path))
    # Use small chunk size to create multiple files
    store.config = store.config.__class__(
        target_rows_per_file=1,
        compression=store.config.compression,
        compression_level=store.config.compression_level,
        lock_timeout_seconds=store.config.lock_timeout_seconds,
    )
    ts0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    df = pl.DataFrame({
        "timestamp": [ts0 + timedelta(minutes=i) for i in range(5)],
        "value": list(range(5)),
    })
    store.write("test/chunks", df)
    files_before = len(list((temp_storage_path / "test" / "chunks").rglob("*.parquet")))

    stats = store.consolidate("test/chunks", target_rows_per_file=4)
    files_after = len(list((temp_storage_path / "test" / "chunks").rglob("*.parquet")))

    assert stats["files_before"] == files_before
    assert files_after <= files_before
    assert stats["rows_processed"] == len(df)

    # Data preserved
    result = store.read("test/chunks")
    assert len(result) == len(df)


def test_consolidate_empty_key(temp_storage_path: Path) -> None:
    store = ParquetStore(str(temp_storage_path))
    stats = store.consolidate("missing/key", target_rows_per_file=10)
    assert stats["rows_processed"] == 0
    assert stats["files_before"] == 0
    assert stats["files_after"] == 0
