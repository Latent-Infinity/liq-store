"""Tests for liq.store.parquet module."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from liq.store.exceptions import (
    ConcurrentWriteError,
    PathTraversalError,
    SchemaCompatibilityError,
    StorageError,
)
from liq.store.parquet import ParquetStore, ParquetStoreConfig


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

    def test_default_config(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        assert store.config.target_rows_per_file == 150_000
        assert store.config.compression == "zstd"
        assert store.config.compression_level == 3
        assert store.config.lock_timeout_seconds == 30

    def test_custom_config(self, temp_storage_path: Path) -> None:
        config = ParquetStoreConfig(
            target_rows_per_file=100_000,
            compression="snappy",
            compression_level=5,
            lock_timeout_seconds=60,
        )
        store = ParquetStore(str(temp_storage_path), config=config)
        assert store.config.target_rows_per_file == 100_000
        assert store.config.compression == "snappy"
        assert store.config.compression_level == 5
        assert store.config.lock_timeout_seconds == 60


class TestParquetStoreConfig:
    """Tests for ParquetStoreConfig."""

    def test_default_values(self) -> None:
        config = ParquetStoreConfig()
        assert config.target_rows_per_file == 150_000
        assert config.compression == "zstd"
        assert config.compression_level == 3
        assert config.lock_timeout_seconds == 30

    def test_custom_values(self) -> None:
        config = ParquetStoreConfig(
            target_rows_per_file=50_000,
            compression="lz4",
            compression_level=10,
            lock_timeout_seconds=120,
        )
        assert config.target_rows_per_file == 50_000
        assert config.compression == "lz4"
        assert config.compression_level == 10
        assert config.lock_timeout_seconds == 120

    def test_config_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        config = ParquetStoreConfig()
        with pytest.raises(FrozenInstanceError):
            config.target_rows_per_file = 999  # type: ignore[misc]


class TestPathTraversal:
    """Tests for path traversal protection."""

    def test_rejects_dotdot_traversal(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        df = pl.DataFrame({"value": [1.0]})

        with pytest.raises(PathTraversalError, match="traverses outside storage root"):
            store.write("../escape", df)

    def test_rejects_complex_traversal(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        df = pl.DataFrame({"value": [1.0]})

        with pytest.raises(PathTraversalError, match="traverses outside storage root"):
            store.write("foo/../../escape", df)

    def test_rejects_absolute_path(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))
        df = pl.DataFrame({"value": [1.0]})

        with pytest.raises(PathTraversalError, match="traverses outside storage root"):
            store.write("/etc/passwd", df)

    def test_read_rejects_path_traversal(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))

        with pytest.raises(PathTraversalError):
            store.read("../escape")

    def test_exists_rejects_path_traversal(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))

        with pytest.raises(PathTraversalError):
            store.exists("../escape")

    def test_delete_rejects_path_traversal(self, temp_storage_path: Path) -> None:
        store = ParquetStore(str(temp_storage_path))

        with pytest.raises(PathTraversalError):
            store.delete("../escape")

    def test_get_date_range_rejects_path_traversal(
        self, temp_storage_path: Path
    ) -> None:
        store = ParquetStore(str(temp_storage_path))

        with pytest.raises(PathTraversalError):
            store.get_date_range("../escape")

    def test_valid_nested_key_allowed(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        # Valid nested keys should work fine
        store.write("forex/EUR_USD", sample_ohlcv_df)
        assert store.exists("forex/EUR_USD")


class TestSchemaCompatibility:
    """Tests for schema compatibility checking on append."""

    def test_append_compatible_schema_succeeds(
        self, temp_storage_path: Path, sample_timestamp: datetime
    ) -> None:
        store = ParquetStore(str(temp_storage_path))

        df1 = pl.DataFrame({
            "timestamp": [sample_timestamp],
            "value": [1.0],
        })
        df2 = pl.DataFrame({
            "timestamp": [sample_timestamp + timedelta(minutes=1)],
            "value": [2.0],
        })

        store.write("test/key", df1)
        store.write("test/key", df2, mode="append")

        result = store.read("test/key")
        assert len(result) == 2

    def test_append_new_columns_succeeds(
        self, temp_storage_path: Path, sample_timestamp: datetime
    ) -> None:
        """Schema evolution: adding new columns should work."""
        store = ParquetStore(str(temp_storage_path))

        df1 = pl.DataFrame({
            "timestamp": [sample_timestamp],
            "value": [1.0],
        })
        df2 = pl.DataFrame({
            "timestamp": [sample_timestamp + timedelta(minutes=1)],
            "value": [2.0],
            "new_column": ["extra"],  # New column
        })

        store.write("test/key", df1)
        store.write("test/key", df2, mode="append")

        result = store.read("test/key")
        assert len(result) == 2
        assert "new_column" in result.columns

    def test_append_incompatible_types_raises_error(
        self, temp_storage_path: Path, sample_timestamp: datetime
    ) -> None:
        store = ParquetStore(str(temp_storage_path))

        df1 = pl.DataFrame({
            "timestamp": [sample_timestamp],
            "value": [1.0],  # Float
        })
        df2 = pl.DataFrame({
            "timestamp": [sample_timestamp + timedelta(minutes=1)],
            "value": ["string"],  # String - incompatible!
        })

        store.write("test/key", df1)

        with pytest.raises(SchemaCompatibilityError, match="Schema incompatible"):
            store.write("test/key", df2, mode="append")

    def test_append_numeric_widening_allowed(
        self, temp_storage_path: Path, sample_timestamp: datetime
    ) -> None:
        """Numeric type widening (Int32 -> Int64) should be allowed."""
        store = ParquetStore(str(temp_storage_path))

        df1 = pl.DataFrame({
            "timestamp": [sample_timestamp],
            "value": pl.Series([1], dtype=pl.Int32),
        })
        df2 = pl.DataFrame({
            "timestamp": [sample_timestamp + timedelta(minutes=1)],
            "value": pl.Series([2], dtype=pl.Int64),
        })

        store.write("test/key", df1)
        store.write("test/key", df2, mode="append")

        result = store.read("test/key")
        assert len(result) == 2


class TestConcurrentWrites:
    """Tests for concurrent write protection."""

    def test_concurrent_write_raises_error(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Test that concurrent writes to same partition raise ConcurrentWriteError."""
        store = ParquetStore(str(temp_storage_path))

        # First, create the partition so locks are needed
        store.write("forex/EUR_USD", sample_ohlcv_df)

        # We need to test the locking mechanism by trying to acquire lock twice
        key_path = store._key_to_path("forex/EUR_USD")

        # Manually acquire lock
        from liq.store.parquet import FcntlPartitionLock

        lock1 = FcntlPartitionLock(key_path)
        lock1.acquire()

        try:
            # Second lock should fail
            lock2 = FcntlPartitionLock(key_path)
            with pytest.raises(ConcurrentWriteError, match="locked by another writer"):
                lock2.acquire()
        finally:
            lock1.release()

    def test_lock_released_after_write(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Test that lock is released after write completes."""
        store = ParquetStore(str(temp_storage_path))

        # Write should acquire and release lock
        store.write("forex/EUR_USD", sample_ohlcv_df)

        # Second write should succeed (lock released)
        store.write("forex/EUR_USD", sample_ohlcv_df, mode="append")

    def test_lock_released_on_error(self, temp_storage_path: Path) -> None:
        """Test that lock is released even when write fails."""
        store = ParquetStore(str(temp_storage_path))

        # Create a situation that will cause write to fail
        key_path = temp_storage_path / "blocked"
        key_path.write_text("blocking file")

        df = pl.DataFrame({"value": [1.0]})

        with pytest.raises(StorageError):
            store.write("blocked/key", df)

        # Lock should be released, so we should be able to write elsewhere
        store.write("other/key", df)


class TestAtomicWrites:
    """Tests for atomic write behavior."""

    def test_temp_cleaned_on_failure(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Test that temp directory is cleaned up on write failure."""
        store = ParquetStore(str(temp_storage_path))

        # First write to create partition
        store.write("forex/EUR_USD", sample_ohlcv_df)

        # Create incompatible data that will fail schema check
        bad_df = pl.DataFrame({
            "timestamp": [datetime.now(UTC)],
            "symbol": [123],  # Wrong type - was string before
        })

        with pytest.raises(SchemaCompatibilityError):
            store.write("forex/EUR_USD", bad_df, mode="append")

        # No temp directories should remain
        temp_dirs = list(temp_storage_path.glob("**/.*.tmp.*"))
        assert len(temp_dirs) == 0

    def test_original_preserved_on_failure(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Test that original data is preserved when write fails."""
        store = ParquetStore(str(temp_storage_path))

        # First write
        store.write("forex/EUR_USD", sample_ohlcv_df)
        original_count = len(store.read("forex/EUR_USD"))

        # Incompatible append should fail
        bad_df = pl.DataFrame({
            "timestamp": [datetime.now(UTC)],
            "open": ["not_a_number"],  # Wrong type
        })

        with pytest.raises(SchemaCompatibilityError):
            store.write("forex/EUR_USD", bad_df, mode="append")

        # Original data should be unchanged
        assert len(store.read("forex/EUR_USD")) == original_count


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

    def test_write_append_without_timestamp_column(
        self, temp_storage_path: Path
    ) -> None:
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

    def test_read_with_columns_subset(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)

        result = store.read("forex/EUR_USD", columns=["timestamp", "close"])
        assert set(result.columns) == {"timestamp", "close"}

    def test_read_streaming_mode(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)

        result = store.read("forex/EUR_USD", streaming=True)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_ohlcv_df)

    def test_read_batch_mode_returns_iterator(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)

        result = store.read("forex/EUR_USD", batch_size=1)
        # Should return an iterator
        batches = list(result)
        assert len(batches) == 3  # 3 rows, batch_size=1

    def test_read_batch_iteration_complete(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)

        # Read in batches and verify all data is returned
        total_rows = 0
        for batch in store.read("forex/EUR_USD", batch_size=2):
            total_rows += len(batch)

        assert total_rows == len(sample_ohlcv_df)


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

    def test_delete_returns_false_for_nonexistent(
        self, temp_storage_path: Path
    ) -> None:
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

    def test_delete_removes_lock_file_if_present(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)

        # Manually create a lock file to test cleanup
        # (Lock is released after write, but file may persist on some systems)
        key_path = temp_storage_path / "forex" / "EUR_USD"
        lock_file = key_path / ".lock"
        lock_file.touch()  # Create lock file
        assert lock_file.exists()

        store.delete("forex/EUR_USD")
        # Lock file should be removed along with data
        assert not lock_file.exists()


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


class TestLogging:
    """Tests for logging behavior."""

    def test_write_logs_on_success(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.INFO):
            store = ParquetStore(str(temp_storage_path))
            store.write("forex/EUR_USD", sample_ohlcv_df)

        assert any("Wrote" in record.message and "forex/EUR_USD" in record.message for record in caplog.records)

    def test_read_logs_debug_info(
        self, temp_storage_path: Path, sample_ohlcv_df: pl.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        store = ParquetStore(str(temp_storage_path))
        store.write("forex/EUR_USD", sample_ohlcv_df)

        with caplog.at_level(logging.DEBUG):
            store.read("forex/EUR_USD")

        assert any("Reading" in record.message for record in caplog.records)
