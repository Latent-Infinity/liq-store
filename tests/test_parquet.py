"""Tests for liq.store.parquet module."""

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from liq.store.exceptions import StorageError
from liq.store.parquet import ParquetStore


class TestParquetStoreCreation:
    """Tests for ParquetStore instantiation."""

    def test_creates_data_directory(self, temp_storage_path: Path) -> None:
        ParquetStore(str(temp_storage_path / "data"))
        assert (temp_storage_path / "data").exists()

    def test_implements_protocol(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        # Verify it has all protocol methods
        assert hasattr(store, "write")
        assert hasattr(store, "read")
        assert hasattr(store, "exists")
        assert hasattr(store, "delete")
        assert hasattr(store, "list_keys")
        assert hasattr(store, "get_date_range")


class TestParquetStoreWrite:
    """Tests for ParquetStore.write method."""

    def test_write_creates_parquet_file(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)

        # Check that parquet files were created
        parquet_files = list(temp_storage_path.rglob("*.parquet"))
        assert len(parquet_files) > 0

    def test_write_and_read_roundtrip(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)
        result = store.read("forex/EUR_USD")

        assert len(result) == len(sample_ohlcv_df)
        assert set(result.columns) == set(sample_ohlcv_df.columns)

    def test_write_append_mode(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)
        store.write("forex/EUR_USD", sample_ohlcv_df, mode="append")
        result = store.read("forex/EUR_USD")

        # Should have double the rows (after deduplication if timestamps differ)
        assert len(result) >= len(sample_ohlcv_df)

    def test_write_overwrite_mode(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)

        # Create different data
        new_df = sample_ohlcv_df.with_columns(pl.col("close") + 0.001)
        store.write("forex/EUR_USD", new_df, mode="overwrite")

        result = store.read("forex/EUR_USD")
        assert len(result) == len(new_df)

    def test_write_empty_dataframe(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        empty_df = pl.DataFrame({"timestamp": [], "value": []})
        store.write("test/empty", empty_df)

        # No files should be created for empty data
        parquet_files = list((temp_storage_path / "test" / "empty").rglob("*.parquet"))
        assert len(parquet_files) == 0

    def test_write_without_timestamp_column(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        df = pl.DataFrame({"symbol": ["EUR_USD", "GBP_USD"], "value": [1.0, 2.0]})
        store.write("test/no_timestamp", df)

        result = store.read("test/no_timestamp")
        assert len(result) == 2
        # Should create data.parquet file
        assert (temp_storage_path / "test" / "no_timestamp" / "data.parquet").exists()

    def test_write_append_without_timestamp_column(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        df1 = pl.DataFrame({"symbol": ["EUR_USD"], "value": [1.0]})
        df2 = pl.DataFrame({"symbol": ["GBP_USD"], "value": [2.0]})

        store.write("test/no_ts", df1)
        store.write("test/no_ts", df2, mode="append")

        result = store.read("test/no_ts")
        assert len(result) == 2

    def test_write_raises_storage_error_on_os_error(
        self, temp_storage_path: Path
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        df = pl.DataFrame({"value": [1.0]})

        # Create a file where directory should be to trigger OSError
        (temp_storage_path / "blocked").write_text("blocking")

        with pytest.raises(StorageError, match="Failed to write data"):
            store.write("blocked/key", df)


class TestParquetStoreRead:
    """Tests for ParquetStore.read method."""

    def test_read_nonexistent_returns_empty(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        result = store.read("nonexistent/key")
        assert len(result) == 0

    def test_read_empty_directory_returns_empty(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        # Create directory but no parquet files
        (temp_storage_path / "empty" / "key").mkdir(parents=True)
        result = store.read("empty/key")
        assert len(result) == 0

    def test_read_with_date_filter(
        self, temp_storage_path: Path, sample_timestamp: datetime
    ) -> None:
        store = ParquetStore(str(temp_storage_path))

        # Create data spanning multiple dates
        timestamps = [
            sample_timestamp,
            sample_timestamp + timedelta(days=1),
            sample_timestamp + timedelta(days=2),
        ]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "symbol": ["EUR_USD"] * 3,
            "value": [1.0, 2.0, 3.0],
        })
        store.write("forex/EUR_USD", df)

        # Filter to just the first day (exclusive of day+1)
        start = sample_timestamp.date()
        end = sample_timestamp.date()  # Same day, should get only first row
        result = store.read("forex/EUR_USD", start=start, end=end)

        assert len(result) == 1

    def test_read_with_start_filter_only(
        self, temp_storage_path: Path, sample_timestamp: datetime
    ) -> None:
        store = ParquetStore(str(temp_storage_path))

        timestamps = [
            sample_timestamp,
            sample_timestamp + timedelta(days=1),
            sample_timestamp + timedelta(days=2),
        ]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "symbol": ["EUR_USD"] * 3,
            "value": [1.0, 2.0, 3.0],
        })
        store.write("forex/EUR_USD", df)

        # Filter with start only
        start = (sample_timestamp + timedelta(days=1)).date()
        result = store.read("forex/EUR_USD", start=start)

        assert len(result) == 2  # Should get days 1 and 2

    def test_read_with_end_filter_only(
        self, temp_storage_path: Path, sample_timestamp: datetime
    ) -> None:
        store = ParquetStore(str(temp_storage_path))

        timestamps = [
            sample_timestamp,
            sample_timestamp + timedelta(days=1),
            sample_timestamp + timedelta(days=2),
        ]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "symbol": ["EUR_USD"] * 3,
            "value": [1.0, 2.0, 3.0],
        })
        store.write("forex/EUR_USD", df)

        # Filter with end only
        end = sample_timestamp.date()
        result = store.read("forex/EUR_USD", end=end)

        assert len(result) == 1  # Should get only day 0

    def test_read_without_timestamp_column_ignores_filters(
        self, temp_storage_path: Path
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        df = pl.DataFrame({"symbol": ["EUR_USD", "GBP_USD"], "value": [1.0, 2.0]})
        store.write("test/no_ts", df)

        # Date filters should be ignored for data without timestamp
        result = store.read("test/no_ts", start=date(2024, 1, 1), end=date(2024, 1, 31))
        assert len(result) == 2


class TestParquetStoreExists:
    """Tests for ParquetStore.exists method."""

    def test_exists_false_for_nonexistent(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        assert not store.exists("nonexistent/key")

    def test_exists_true_after_write(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)
        assert store.exists("forex/EUR_USD")


class TestParquetStoreDelete:
    """Tests for ParquetStore.delete method."""

    def test_delete_returns_false_for_nonexistent(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        result = store.delete("nonexistent/key")
        assert result is False

    def test_delete_returns_false_for_empty_directory(
        self, temp_storage_path: Path
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        # Create directory but no parquet files
        (temp_storage_path / "empty" / "key").mkdir(parents=True)
        result = store.delete("empty/key")
        assert result is False

    def test_delete_returns_true_and_removes_data(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)
        assert store.exists("forex/EUR_USD")

        result = store.delete("forex/EUR_USD")
        assert result is True
        assert not store.exists("forex/EUR_USD")

    def test_delete_cleans_up_parent_directories(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("deep/nested/path/key", sample_ohlcv_df)

        result = store.delete("deep/nested/path/key")
        assert result is True

        # Empty parent directories should be cleaned up
        assert not (temp_storage_path / "deep" / "nested" / "path" / "key").exists()
        assert not (temp_storage_path / "deep" / "nested" / "path").exists()
        assert not (temp_storage_path / "deep" / "nested").exists()
        assert not (temp_storage_path / "deep").exists()

    def test_delete_preserves_sibling_directories(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)
        store.write("forex/GBP_USD", sample_ohlcv_df)

        result = store.delete("forex/EUR_USD")
        assert result is True

        # Parent forex directory should still exist (sibling data present)
        assert (temp_storage_path / "forex").exists()
        assert store.exists("forex/GBP_USD")


class TestParquetStoreListKeys:
    """Tests for ParquetStore.list_keys method."""

    def test_list_keys_empty_store(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        keys = store.list_keys()
        assert keys == []

    def test_list_keys_returns_written_keys(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)
        store.write("forex/GBP_USD", sample_ohlcv_df)
        store.write("crypto/BTC-USD", sample_ohlcv_df)

        keys = store.list_keys()
        assert len(keys) == 3
        assert "forex/EUR_USD" in keys
        assert "forex/GBP_USD" in keys
        assert "crypto/BTC-USD" in keys

    def test_list_keys_with_prefix(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)
        store.write("forex/GBP_USD", sample_ohlcv_df)
        store.write("crypto/BTC-USD", sample_ohlcv_df)

        forex_keys = store.list_keys(prefix="forex/")
        assert len(forex_keys) == 2
        assert all(k.startswith("forex/") for k in forex_keys)


class TestParquetStoreGetDateRange:
    """Tests for ParquetStore.get_date_range method."""

    def test_get_date_range_nonexistent_returns_none(
        self, temp_storage_path: Path
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        result = store.get_date_range("nonexistent/key")
        assert result is None

    def test_get_date_range_empty_directory_returns_none(
        self, temp_storage_path: Path
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        # Create directory but no parquet files
        (temp_storage_path / "empty" / "key").mkdir(parents=True)
        result = store.get_date_range("empty/key")
        assert result is None

    def test_get_date_range_no_timestamp_column_returns_none(
        self, temp_storage_path: Path
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        df = pl.DataFrame({"symbol": ["EUR_USD", "GBP_USD"], "value": [1.0, 2.0]})
        store.write("test/no_ts", df)

        result = store.get_date_range("test/no_ts")
        assert result is None

    def test_get_date_range_empty_dataframe_returns_none(
        self, temp_storage_path: Path
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        # Write some data first, then overwrite with non-empty but all-null timestamps
        df = pl.DataFrame({
            "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "value": pl.Series([], dtype=pl.Float64),
        })
        # Need to manually write a parquet file with empty data
        key_path = temp_storage_path / "test" / "empty_ts"
        key_path.mkdir(parents=True)
        df.write_parquet(key_path / "data.parquet")

        result = store.get_date_range("test/empty_ts")
        assert result is None

    def test_get_date_range_returns_correct_range(
        self, temp_storage_path: Path, sample_timestamp: datetime
    ) -> None:
        store = ParquetStore(str(temp_storage_path))

        timestamps = [
            sample_timestamp,
            sample_timestamp + timedelta(days=5),
            sample_timestamp + timedelta(days=10),
        ]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "symbol": ["EUR_USD"] * 3,
            "value": [1.0, 2.0, 3.0],
        })
        store.write("forex/EUR_USD", df)

        result = store.get_date_range("forex/EUR_USD")
        assert result is not None
        start, end = result
        assert start == sample_timestamp.date()
        assert end == (sample_timestamp + timedelta(days=10)).date()
