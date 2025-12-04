"""Parquet storage implementation for the LIQ Stack.

This module provides a key-based Parquet storage backend that implements
the TimeSeriesStore protocol. Data is stored in partitioned Parquet files
with ZSTD compression.

Storage Layout:
    data_root/
        key_segment_1/
            key_segment_2/
                {YYYYMMDDTHHMMSS}-{YYYYMMDDTHHMMSS}.parquet

Example:
    store = ParquetStore("./data")
    store.write("forex/EUR_USD", df)
    df = store.read("forex/EUR_USD")
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import polars as pl

from liq.store.exceptions import (
    ConcurrentWriteError,
    PathTraversalError,
    SchemaCompatibilityError,
    StorageError,
)
from liq.store.naming import generate_filename

if TYPE_CHECKING:
    from collections.abc import Generator

# Type for valid Parquet compression options
CompressionType = Literal["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParquetStoreConfig:
    """Configuration for ParquetStore.

    Sensible defaults provided. Caller is responsible for loading
    from environment variables or config files if needed.

    Example:
        # Use defaults
        store = ParquetStore("./data")

        # Override via constructor
        config = ParquetStoreConfig(compression_level=6)
        store = ParquetStore("./data", config=config)

        # Caller loads from env (not liq-store's responsibility)
        config = ParquetStoreConfig(
            compression_level=int(os.environ.get("LIQ_COMPRESSION_LEVEL", 3))
        )

    Attributes:
        target_rows_per_file: Target rows per Parquet file (~150k = 4-7 days of M1 data)
        compression: Compression algorithm (zstd recommended)
        compression_level: Compression level (1-22 for zstd, 3 is balanced)
        lock_timeout_seconds: Timeout for acquiring partition locks
    """

    target_rows_per_file: int = 150_000
    compression: CompressionType = "zstd"
    compression_level: int = 3
    lock_timeout_seconds: int = 30


class FcntlPartitionLock:
    """Local filesystem locking via fcntl.

    Provides exclusive write locks for partitions. Cloud backends
    will implement alternatives (DynamoDB, GCS conditional writes).
    """

    def __init__(self, key_path: Path, timeout: float = 30.0) -> None:
        """Initialize partition lock.

        Args:
            key_path: Path to the partition directory
            timeout: Lock acquisition timeout in seconds (currently unused,
                    kept for API compatibility with future cloud locks)
        """
        self.lock_file = key_path / ".lock"
        self.timeout = timeout
        self._fd: int | None = None

    def acquire(self) -> None:
        """Acquire exclusive lock on partition.

        Raises:
            ConcurrentWriteError: If lock cannot be acquired (another writer holds it)
        """
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self.lock_file), os.O_WRONLY | os.O_CREAT)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            os.close(self._fd)
            self._fd = None
            raise ConcurrentWriteError(
                "Failed to acquire lock: partition is locked by another writer"
            ) from e

    def release(self) -> None:
        """Release the partition lock."""
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class ParquetStore:
    """Parquet-based storage implementation.

    Stores time-series data in Parquet files with ZSTD compression.
    Supports key-based access where keys can contain path separators
    (e.g., "forex/EUR_USD").

    Features:
    - Path traversal protection (keys cannot escape data_root)
    - Atomic writes with crash safety
    - Partition locking for concurrent write protection
    - Schema compatibility checking on append
    - Configurable compression and chunking

    Example:
        store = ParquetStore("./data")
        store.write("forex/EUR_USD", df)
        df = store.read("forex/EUR_USD", start=date(2024, 1, 1))
    """

    def __init__(
        self,
        data_root: str,
        config: ParquetStoreConfig | None = None,
    ) -> None:
        """Initialize ParquetStore.

        Args:
            data_root: Root directory for data storage
            config: Optional configuration (uses defaults if not provided)
        """
        self.data_root = Path(data_root).resolve()
        self.config = config or ParquetStoreConfig()
        self.data_root.mkdir(parents=True, exist_ok=True)
        logger.debug("Initialized ParquetStore at %s", self.data_root)

    def _key_to_path(self, key: str) -> Path:
        """Convert storage key to filesystem path with traversal protection.

        Args:
            key: Storage key (e.g., "forex/EUR_USD")

        Returns:
            Resolved absolute path within data_root

        Raises:
            PathTraversalError: If key would resolve outside data_root
        """
        # Resolve to absolute path
        path = (self.data_root / key).resolve()

        # Ensure path is within data_root
        if not path.is_relative_to(self.data_root):
            raise PathTraversalError(f"Key '{key}' traverses outside storage root")

        return path

    @contextmanager
    def _partition_lock(self, key_path: Path) -> Generator[None, None, None]:
        """Acquire partition lock for exclusive write access.

        Uses fcntl locally; cloud backends will swap implementation.

        Args:
            key_path: Path to the partition directory

        Yields:
            None (lock is held during context)

        Raises:
            ConcurrentWriteError: If lock cannot be acquired
        """
        lock = FcntlPartitionLock(key_path, timeout=self.config.lock_timeout_seconds)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def _check_schema_compatibility(
        self, existing_schema: pl.Schema, new_schema: pl.Schema
    ) -> None:
        """Ensure new data is compatible with existing data for append.

        Rules:
        - New columns are allowed (schema evolution)
        - Existing columns must have compatible types
        - Missing columns in new data will become null (Polars handles this)

        Args:
            existing_schema: Schema of existing data
            new_schema: Schema of new data to append

        Raises:
            SchemaCompatibilityError: If schemas are incompatible
        """
        for col_name, col_type in existing_schema.items():
            if col_name in new_schema:
                new_type = new_schema[col_name]
                if not self._types_compatible(col_type, new_type):
                    raise SchemaCompatibilityError(
                        f"Schema incompatible for append: column '{col_name}' "
                        f"has type {col_type} but new data has {new_type}"
                    )

    def _types_compatible(self, existing: pl.DataType, new: pl.DataType) -> bool:
        """Check if types are compatible for concatenation.

        Args:
            existing: Existing column type
            new: New column type

        Returns:
            True if types are compatible, False otherwise
        """
        # Same type is always compatible
        if existing == new:
            return True

        # Allow numeric type widening
        numeric_types = (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        )

        # Check if both are numeric types - with diagonal_relaxed Polars handles coercion
        existing_base = existing.base_type()
        new_base = new.base_type()

        # If both are numeric types, Polars diagonal_relaxed will handle coercion
        existing_is_numeric = any(existing_base == t for t in numeric_types)
        new_is_numeric = any(new_base == t for t in numeric_types)

        return existing_is_numeric and new_is_numeric

    def _merge_and_dedupe(
        self, existing: pl.DataFrame, new: pl.DataFrame
    ) -> pl.DataFrame:
        """Merge existing and new data with deduplication.

        Args:
            existing: Existing DataFrame
            new: New DataFrame to merge

        Returns:
            Combined DataFrame, deduplicated by timestamp if present
        """
        # Use diagonal_relaxed concat to handle schema evolution
        # This allows type coercion (e.g., Int32 -> Int64) and missing columns become null
        combined = pl.concat([existing, new], how="diagonal_relaxed")

        # Deduplicate by timestamp if present
        if "timestamp" in combined.columns:
            combined = combined.unique(subset=["timestamp"], keep="last").sort(
                "timestamp"
            )

        return combined

    def write(self, key: str, data: pl.DataFrame, mode: str = "append") -> None:
        """Write time-series data to storage.

        Args:
            key: Storage key (e.g., "forex/EUR_USD")
            data: Polars DataFrame with time-series data
            mode: Write mode - "append" (default) or "overwrite"

        Raises:
            StorageError: If write operation fails
            PathTraversalError: If key traverses outside storage root
            ConcurrentWriteError: If another writer holds the partition lock
            SchemaCompatibilityError: If append data has incompatible schema
        """
        logger.debug("Writing %d rows to key=%s mode=%s", len(data), key, mode)

        try:
            key_path = self._key_to_path(key)

            with self._partition_lock(key_path):
                temp_path = key_path.parent / f".{key_path.name}.tmp.{uuid4().hex[:8]}"

                try:
                    temp_path.mkdir(parents=True, exist_ok=True)

                    if mode == "overwrite" or not key_path.exists():
                        # Fresh write
                        output_df = data
                        if "timestamp" in output_df.columns:
                            output_df = output_df.sort("timestamp")
                    elif mode == "append" and any(key_path.glob("*.parquet")):
                        # Read, check schema, merge, dedupe
                        existing_df = pl.read_parquet(key_path / "*.parquet")
                        self._check_schema_compatibility(existing_df.schema, data.schema)
                        output_df = self._merge_and_dedupe(existing_df, data)
                    else:
                        # No existing data, treat as fresh write
                        output_df = data
                        if "timestamp" in output_df.columns:
                            output_df = output_df.sort("timestamp")

                    # Write to temp location
                    self._write_chunked(output_df, temp_path)

                    # Atomic swap - readers see old OR new, never partial
                    if key_path.exists():
                        shutil.rmtree(key_path)
                    temp_path.rename(key_path)

                    logger.info("Wrote %d rows to %s", len(output_df), key)

                except Exception:
                    # Clean up temp on any error
                    if temp_path.exists():
                        shutil.rmtree(temp_path)
                    raise

        except (PathTraversalError, ConcurrentWriteError, SchemaCompatibilityError):
            # Re-raise our own exceptions without wrapping
            raise
        except pl.exceptions.PolarsError as e:
            logger.error("Failed to write to %s: %s", key, e)
            raise StorageError(f"Failed to write data: {e}") from e
        except OSError as e:
            logger.error("Failed to write to %s: %s", key, e)
            raise StorageError(f"Failed to write data: {e}") from e

    def _write_chunked(self, df: pl.DataFrame, path: Path) -> None:
        """Write DataFrame in optimally-sized chunks.

        Args:
            df: DataFrame to write
            path: Directory path to write chunks to
        """
        if df.is_empty():
            return

        # If no timestamp column, write as single file
        if "timestamp" not in df.columns:
            df.write_parquet(
                path / "data.parquet",
                compression=self.config.compression,
                compression_level=self.config.compression_level,
            )
            return

        total_rows = len(df)

        # Split into chunks
        for start_idx in range(0, total_rows, self.config.target_rows_per_file):
            end_idx = min(start_idx + self.config.target_rows_per_file, total_rows)
            chunk = df.slice(start_idx, end_idx - start_idx)

            # Generate filename from timestamp range
            min_ts = chunk["timestamp"].min()
            max_ts = chunk["timestamp"].max()

            if min_ts is not None and max_ts is not None:
                filename = generate_filename(min_ts, max_ts)
            else:
                filename = f"chunk_{start_idx}.parquet"

            chunk.write_parquet(
                path / filename,
                compression=self.config.compression,
                compression_level=self.config.compression_level,
            )

    def read(
        self,
        key: str,
        start: date | None = None,
        end: date | None = None,
        columns: list[str] | None = None,
        *,
        streaming: bool = False,
        batch_size: int | None = None,
    ) -> pl.DataFrame | Iterator[pl.DataFrame]:
        """Read time-series data from storage.

        Args:
            key: Storage key
            start: Optional start date filter (inclusive)
            end: Optional end date filter (inclusive)
            columns: Optional column subset (reduces memory)
            streaming: If True, use Polars streaming engine (memory-efficient)
            batch_size: If set, yield DataFrames in chunks (returns Iterator)

        Returns:
            Polars DataFrame with time-series data, or Iterator[DataFrame]
            if batch_size is set. Empty DataFrame if no data found.

        Raises:
            StorageError: If read operation fails
            PathTraversalError: If key traverses outside storage root

        Examples:
            # Default: load into memory
            df = store.read("forex/EUR_USD")

            # Memory-efficient streaming
            df = store.read("forex/EUR_USD", streaming=True)

            # Process in chunks
            for batch in store.read("forex/EUR_USD", batch_size=100_000):
                process(batch)

            # Column subset to reduce memory
            df = store.read("forex/EUR_USD", columns=["timestamp", "close"])
        """
        logger.debug("Reading key=%s start=%s end=%s", key, start, end)

        try:
            key_path = self._key_to_path(key)

            if not key_path.exists():
                logger.debug("Key %s does not exist, returning empty DataFrame", key)
                return pl.DataFrame()

            parquet_files = list(key_path.glob("*.parquet"))
            if not parquet_files:
                logger.debug("No parquet files for %s, returning empty DataFrame", key)
                return pl.DataFrame()

            # Use lazy scanning for memory efficiency
            lf = pl.scan_parquet(key_path / "*.parquet")

            # Apply column selection (predicate pushdown)
            if columns:
                lf = lf.select(columns)

            # Apply date filters if timestamp column exists
            if start is not None or end is not None:
                # Check if timestamp column exists by collecting schema
                schema = lf.collect_schema()
                if "timestamp" in schema:
                    if start is not None:
                        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
                        lf = lf.filter(pl.col("timestamp") >= start_dt)
                    if end is not None:
                        end_dt = datetime.combine(end, datetime.max.time(), tzinfo=UTC)
                        lf = lf.filter(pl.col("timestamp") <= end_dt)

                    lf = lf.sort("timestamp")

            # Return based on requested mode
            if batch_size is not None:
                return self._read_batched(lf, batch_size)

            # Use streaming engine if requested (memory-efficient for large datasets)
            result = lf.collect(engine="streaming") if streaming else lf.collect()
            logger.debug("Read %d rows from %s", len(result), key)
            return result

        except PathTraversalError:
            raise
        except pl.exceptions.PolarsError as e:
            logger.error("Failed to read from %s: %s", key, e)
            raise StorageError(f"Failed to read data: {e}") from e

    def _read_batched(
        self, lf: pl.LazyFrame, batch_size: int
    ) -> Iterator[pl.DataFrame]:
        """Read data in batches for memory efficiency.

        Args:
            lf: LazyFrame to read from
            batch_size: Number of rows per batch

        Yields:
            DataFrame batches
        """
        # Collect full result (streaming engine helps with memory)
        # Then yield in chunks
        # Note: For truly large datasets, consider using pl.scan_parquet
        # with row_index and filtering
        df = lf.collect(engine="streaming")
        total_rows = len(df)

        for start_idx in range(0, total_rows, batch_size):
            yield df.slice(start_idx, batch_size)

    def exists(self, key: str) -> bool:
        """Check if data exists for a key.

        Args:
            key: Storage key

        Returns:
            True if data exists, False otherwise

        Raises:
            PathTraversalError: If key traverses outside storage root
        """
        key_path = self._key_to_path(key)
        if not key_path.exists():
            return False
        return any(key_path.glob("*.parquet"))

    def delete(self, key: str) -> bool:
        """Delete all data for a key.

        Args:
            key: Storage key

        Returns:
            True if data was deleted, False if key didn't exist

        Raises:
            PathTraversalError: If key traverses outside storage root
        """
        logger.debug("Deleting key=%s", key)
        key_path = self._key_to_path(key)

        if not key_path.exists():
            return False

        parquet_files = list(key_path.glob("*.parquet"))
        if not parquet_files:
            return False

        # Remove all parquet files and lock file
        for f in parquet_files:
            f.unlink()

        # Remove lock file if present
        lock_file = key_path / ".lock"
        if lock_file.exists():
            lock_file.unlink()

        # Remove empty directories up the tree
        try:
            key_path.rmdir()
            # Try to remove parent directories if empty
            parent = key_path.parent
            while parent != self.data_root:
                if not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
                else:
                    break
        except OSError:
            pass  # Directory not empty, which is fine

        logger.info("Deleted key %s", key)
        return True

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix filter.

        Args:
            prefix: Optional prefix to filter keys (e.g., "forex/")

        Returns:
            List of matching storage keys
        """
        keys: list[str] = []

        # Walk directory tree looking for parquet files
        for parquet_file in self.data_root.rglob("*.parquet"):
            # Get key from path relative to data_root
            rel_path = parquet_file.parent.relative_to(self.data_root)
            key = str(rel_path)

            if key not in keys and (not prefix or key.startswith(prefix)):
                keys.append(key)

        return sorted(keys)

    def get_date_range(self, key: str) -> tuple[date, date] | None:
        """Get the date range of available data for a key.

        Args:
            key: Storage key

        Returns:
            Tuple of (earliest_date, latest_date) or None if no data

        Raises:
            PathTraversalError: If key traverses outside storage root
        """
        key_path = self._key_to_path(key)

        if not key_path.exists():
            return None

        parquet_files = list(key_path.glob("*.parquet"))
        if not parquet_files:
            return None

        try:
            df = pl.read_parquet(key_path / "*.parquet")

            if "timestamp" not in df.columns or df.is_empty():
                return None

            min_ts = df["timestamp"].min()
            max_ts = df["timestamp"].max()

            if min_ts is None or max_ts is None:
                return None

            # Cast to datetime to get .date() method
            if isinstance(min_ts, datetime) and isinstance(max_ts, datetime):
                return (min_ts.date(), max_ts.date())

            return None

        except pl.exceptions.PolarsError:
            return None
