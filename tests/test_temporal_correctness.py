"""Tests for temporal correctness and data integrity.

Following TDD: Tests verify timestamp handling, UTC enforcement, and data integrity.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from liq.store.naming import generate_filename, parse_filename
from liq.store.parquet import ParquetStore


class TestTimestampUTCEnforcement:
    """Tests verifying UTC timezone handling in storage operations."""

    def test_read_filter_uses_utc_for_start_date(
        self, temp_storage_path: Path, sample_timestamp: datetime
    ) -> None:
        """Date filter for start should use UTC midnight."""
        store = ParquetStore(str(temp_storage_path))

        # Create data with specific UTC timestamps
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC),   # Midnight UTC
                datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),  # Noon UTC
                datetime(2024, 1, 16, 0, 0, 0, tzinfo=UTC),   # Next day midnight UTC
            ],
            "value": [1.0, 2.0, 3.0],
        })
        store.write("test/utc", df)

        # Filter for date 2024-01-15 should include both midnight and noon
        result = store.read("test/utc", start=date(2024, 1, 15), end=date(2024, 1, 15))

        assert len(result) == 2
        timestamps = result["timestamp"].to_list()
        assert timestamps[0] == datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)
        assert timestamps[1] == datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

    def test_read_filter_end_includes_full_day(
        self, temp_storage_path: Path
    ) -> None:
        """End date filter should include data up to end of day UTC."""
        store = ParquetStore(str(temp_storage_path))

        # Create data with timestamps spanning multiple days
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 23, 59, 59, tzinfo=UTC),  # End of day
                datetime(2024, 1, 16, 0, 0, 1, tzinfo=UTC),     # Just past midnight
            ],
            "value": [1.0, 2.0, 3.0],
        })
        store.write("test/eod", df)

        # End date Jan 15 should include 23:59:59 but not 00:00:01 of Jan 16
        result = store.read("test/eod", end=date(2024, 1, 15))

        assert len(result) == 2

    def test_date_range_returns_utc_dates(
        self, temp_storage_path: Path
    ) -> None:
        """get_date_range should return dates in UTC context."""
        store = ParquetStore(str(temp_storage_path))

        # Store data with UTC timestamps
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 20, 23, 59, 59, tzinfo=UTC),
            ],
            "value": [1.0, 2.0],
        })
        store.write("test/range", df)

        result = store.get_date_range("test/range")

        assert result is not None
        start_date, end_date = result
        assert start_date == date(2024, 1, 15)
        assert end_date == date(2024, 1, 20)


class TestFilenameTimestampParsing:
    """Tests verifying filename timestamp parsing returns UTC."""

    def test_parse_filename_returns_utc_timestamps(self) -> None:
        """Parsed timestamps from filename should be UTC-aware."""
        filename = "20240115T103000-20240115T120000.parquet"

        result = parse_filename(filename)

        assert result is not None
        start_ts, end_ts = result
        assert start_ts.tzinfo == UTC
        assert end_ts.tzinfo == UTC

    def test_generate_filename_uses_local_time_correctly(self) -> None:
        """Filename generation should handle timezone-aware timestamps."""
        # Create timestamps in UTC
        start = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        end = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

        filename = generate_filename(start, end)

        # Should contain the UTC times
        assert "20240115T103000" in filename
        assert "20240115T120000" in filename

    def test_filename_roundtrip_preserves_utc(self) -> None:
        """Generate then parse should preserve UTC timezone."""
        original_start = datetime(2024, 3, 15, 8, 30, 45, tzinfo=UTC)
        original_end = datetime(2024, 3, 20, 16, 45, 30, tzinfo=UTC)

        filename = generate_filename(original_start, original_end)
        parsed = parse_filename(filename)

        assert parsed is not None
        parsed_start, parsed_end = parsed
        assert parsed_start == original_start
        assert parsed_end == original_end
        assert parsed_start.tzinfo == UTC
        assert parsed_end.tzinfo == UTC


class TestTimestampSortingOnWrite:
    """Tests verifying data is sorted by timestamp on write."""

    def test_write_sorts_by_timestamp(
        self, temp_storage_path: Path
    ) -> None:
        """Data should be sorted by timestamp after write."""
        store = ParquetStore(str(temp_storage_path))

        # Write out-of-order data
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),  # Middle
                datetime(2024, 1, 15, 8, 0, 0, tzinfo=UTC),   # First
                datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC),  # Last
            ],
            "value": [2.0, 1.0, 3.0],
        })
        store.write("test/sort", df)

        result = store.read("test/sort")
        timestamps = result["timestamp"].to_list()

        # Should be sorted ascending
        assert timestamps[0] < timestamps[1] < timestamps[2]
        assert timestamps[0] == datetime(2024, 1, 15, 8, 0, 0, tzinfo=UTC)
        assert timestamps[2] == datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC)

    def test_append_maintains_sorted_order(
        self, temp_storage_path: Path
    ) -> None:
        """Appended data should be merged and sorted."""
        store = ParquetStore(str(temp_storage_path))

        # Write initial data
        df1 = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            ],
            "value": [1.0, 2.0],
        })
        store.write("test/append", df1)

        # Append data that should interleave
        df2 = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 8, 0, 0, tzinfo=UTC),   # Before existing
                datetime(2024, 1, 15, 11, 0, 0, tzinfo=UTC),  # Between existing
            ],
            "value": [0.0, 1.5],
        })
        store.write("test/append", df2, mode="append")

        result = store.read("test/append")
        timestamps = result["timestamp"].to_list()

        # Verify sorted order
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1]


class TestDeduplicationBehavior:
    """Tests verifying timestamp-based deduplication."""

    def test_append_deduplicates_by_timestamp(
        self, temp_storage_path: Path
    ) -> None:
        """Duplicate timestamps should keep last value."""
        store = ParquetStore(str(temp_storage_path))

        ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)

        # Write initial data
        df1 = pl.DataFrame({
            "timestamp": [ts],
            "value": [1.0],
        })
        store.write("test/dedup", df1)

        # Append same timestamp with different value
        df2 = pl.DataFrame({
            "timestamp": [ts],
            "value": [2.0],  # New value for same timestamp
        })
        store.write("test/dedup", df2, mode="append")

        result = store.read("test/dedup")

        # Should have only one row (deduplicated)
        assert len(result) == 1
        # Should keep last value
        assert result["value"][0] == 2.0

    def test_overwrite_does_not_merge_with_existing(
        self, temp_storage_path: Path
    ) -> None:
        """Overwrite mode should replace, not merge."""
        store = ParquetStore(str(temp_storage_path))

        # Write initial data
        df1 = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 11, 0, 0, tzinfo=UTC),
            ],
            "value": [1.0, 2.0],
        })
        store.write("test/overwrite", df1)

        # Overwrite with single row
        df2 = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)],
            "value": [3.0],
        })
        store.write("test/overwrite", df2, mode="overwrite")

        result = store.read("test/overwrite")

        # Should only have the new data
        assert len(result) == 1
        assert result["value"][0] == 3.0


class TestTimestampColumnHandling:
    """Tests for handling of timestamp columns."""

    def test_no_timestamp_column_skips_sorting(
        self, temp_storage_path: Path
    ) -> None:
        """Data without timestamp column should be stored as-is."""
        store = ParquetStore(str(temp_storage_path))

        df = pl.DataFrame({
            "id": [3, 1, 2],
            "value": [30.0, 10.0, 20.0],
        })
        store.write("test/no_ts", df)

        result = store.read("test/no_ts")

        # Order should be preserved (no timestamp sorting)
        assert result["id"].to_list() == [3, 1, 2]

    def test_no_timestamp_column_skips_date_filter(
        self, temp_storage_path: Path
    ) -> None:
        """Date filters should be ignored for data without timestamp."""
        store = ParquetStore(str(temp_storage_path))

        df = pl.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
        })
        store.write("test/no_ts_filter", df)

        # Date filters should have no effect
        result = store.read(
            "test/no_ts_filter",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31)
        )

        assert len(result) == 3


class TestTimestampPrecision:
    """Tests for timestamp precision preservation."""

    def test_microsecond_precision_preserved(
        self, temp_storage_path: Path
    ) -> None:
        """Microsecond precision should be preserved in storage."""
        store = ParquetStore(str(temp_storage_path))

        ts_with_micros = datetime(2024, 1, 15, 10, 30, 45, 123456, tzinfo=UTC)
        df = pl.DataFrame({
            "timestamp": [ts_with_micros],
            "value": [1.0],
        })
        store.write("test/precision", df)

        result = store.read("test/precision")

        # Note: Parquet stores timestamps with microsecond precision
        # The microseconds should be preserved
        stored_ts = result["timestamp"][0]
        assert stored_ts.microsecond == 123456

    def test_nanosecond_truncated_to_microsecond(
        self, temp_storage_path: Path
    ) -> None:
        """Nanosecond precision may be truncated to microsecond."""
        store = ParquetStore(str(temp_storage_path))

        # Polars supports nanosecond precision
        df = pl.DataFrame({
            "timestamp": pl.Series(
                [datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)],
                dtype=pl.Datetime("ns", "UTC")
            ),
            "value": [1.0],
        })
        store.write("test/nano", df)

        result = store.read("test/nano")

        # Should still be readable, precision may be reduced
        assert len(result) == 1
        assert result["timestamp"].dtype.time_zone == "UTC"


class TestChunkedWriteTimestamps:
    """Tests for chunked write timestamp handling."""

    def test_chunk_filenames_reflect_timestamp_range(
        self, temp_storage_path: Path
    ) -> None:
        """Chunk filenames should contain correct timestamp ranges."""
        from liq.store.parquet import ParquetStoreConfig

        # Use small target_rows_per_file to force multiple chunks
        config = ParquetStoreConfig(target_rows_per_file=2)
        store = ParquetStore(str(temp_storage_path), config=config)

        # Create data with 5 rows
        timestamps = [
            datetime(2024, 1, 15, i, 0, 0, tzinfo=UTC)
            for i in range(5)
        ]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "value": list(range(5)),
        })
        store.write("test/chunks", df)

        # Check that multiple parquet files were created
        key_path = temp_storage_path / "test" / "chunks"
        parquet_files = list(key_path.rglob("*.parquet"))

        # Should have multiple files due to chunking
        assert len(parquet_files) >= 2

        # Verify filenames contain timestamp information
        for f in parquet_files:
            parsed = parse_filename(f.name)
            # Some files might be timestamp-based
            if parsed is not None:
                start_ts, end_ts = parsed
                assert start_ts.tzinfo == UTC
                assert end_ts.tzinfo == UTC


class TestEmptyDataframeDateRange:
    """Tests for edge cases with empty data."""

    def test_empty_dataframe_date_range_returns_none(
        self, temp_storage_path: Path
    ) -> None:
        """Empty DataFrame should return None for date range."""
        store = ParquetStore(str(temp_storage_path))

        # Write non-empty data first, then check empty case
        # Empty writes don't create files
        result = store.get_date_range("nonexistent/key")

        assert result is None

    def test_read_empty_returns_empty_dataframe(
        self, temp_storage_path: Path
    ) -> None:
        """Reading nonexistent key should return empty DataFrame."""
        store = ParquetStore(str(temp_storage_path))

        result = store.read("nonexistent/key")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0


class TestTimezoneAwareDataIntegrity:
    """Tests ensuring timezone awareness is maintained throughout."""

    def test_utc_timestamps_preserved_through_roundtrip(
        self, temp_storage_path: Path
    ) -> None:
        """UTC timestamps should be preserved through write/read."""
        store = ParquetStore(str(temp_storage_path))

        original_ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        df = pl.DataFrame({
            "timestamp": [original_ts],
            "value": [1.0],
        })
        store.write("test/utc_roundtrip", df)

        result = store.read("test/utc_roundtrip")

        assert result["timestamp"][0] == original_ts
        # Verify timezone is preserved
        assert result["timestamp"].dtype.time_zone == "UTC"

    def test_filter_date_boundaries_are_inclusive(
        self, temp_storage_path: Path
    ) -> None:
        """Both start and end dates should be inclusive."""
        store = ParquetStore(str(temp_storage_path))

        # Create data spanning 3 days
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 16, 12, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 17, 12, 0, 0, tzinfo=UTC),
            ],
            "value": [1.0, 2.0, 3.0],
        })
        store.write("test/inclusive", df)

        # Filter for middle day only
        result = store.read(
            "test/inclusive",
            start=date(2024, 1, 16),
            end=date(2024, 1, 16)
        )

        assert len(result) == 1
        assert result["value"][0] == 2.0
