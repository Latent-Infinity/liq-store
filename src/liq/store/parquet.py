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
import pyarrow.dataset as ds

from liq.store.exceptions import (
    ConcurrentWriteError,
    PathTraversalError,
    SchemaCompatibilityError,
    StorageError,
)
from liq.store.naming import generate_filename, parse_filename

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
        fd = os.open(str(self.lock_file), os.O_WRONLY | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            os.close(fd)
            raise ConcurrentWriteError(
                "Failed to acquire lock: partition is locked by another writer"
            ) from e
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

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

        subset: list[str] = []
        if "timestamp" in combined.columns:
            subset.append("timestamp")
        if "symbol" in combined.columns:
            subset.append("symbol")
        if "provider" in combined.columns:
            subset.append("provider")

        if subset:
            combined = combined.unique(subset=subset, keep="last")
            if "timestamp" in combined.columns:
                combined = combined.sort("timestamp")

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

                write_succeeded = False
                try:
                    temp_path.mkdir(parents=True, exist_ok=True)

                    if data.is_empty():
                        logger.info("No data to write for key=%s (empty DataFrame)", key)
                        return

                    # Global schema check against any existing file for append mode
                    if mode == "append" and key_path.exists():
                        first_existing = next(key_path.rglob("*.parquet"), None)
                        if first_existing:
                            existing_df = pl.read_parquet(first_existing)
                            self._check_schema_compatibility(
                                existing_df.schema, data.schema
                            )

                    # Determine partitions from data
                    if "timestamp" in data.columns:
                        data_with_parts = data.with_columns(
                            [
                                pl.col("timestamp").dt.year().alias("_year"),
                                pl.col("timestamp").dt.month().alias("_month"),
                            ]
                        )
                        partitions = data_with_parts.partition_by(
                            ["_year", "_month"], maintain_order=True
                        )
                    else:
                        partitions = [data]

                    touched_partitions: set[tuple[int | None, int | None]] = set()
                    if "timestamp" in data.columns:
                        for part in partitions:
                            year = int(part["_year"][0])
                            month = int(part["_month"][0])
                            touched_partitions.add((year, month))
                    else:
                        touched_partitions.add((None, None))

                    # Copy untouched existing partitions for append
                    if mode == "append" and key_path.exists():
                        if "timestamp" in data.columns:
                            for year_dir in key_path.glob("year=*"):
                                year_value = year_dir.name.replace("year=", "")
                                for month_dir in year_dir.glob("month=*"):
                                    month_value = month_dir.name.replace("month=", "")
                                    try:
                                        year_int = int(year_value)
                                        month_int = int(month_value)
                                    except ValueError:
                                        continue
                                    if (year_int, month_int) in touched_partitions:
                                        continue
                                    dest_dir = (
                                        temp_path / year_dir.name / month_dir.name
                                    )
                                    dest_dir.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copytree(month_dir, dest_dir)
                        else:
                            # No timestamp partitioning; copy everything
                            shutil.copytree(key_path, temp_path, dirs_exist_ok=True)

                    # Handle touched partitions
                    if "timestamp" in data.columns:
                        for part in partitions:
                            year = int(part["_year"][0])
                            month = int(part["_month"][0])
                            part_df = part.drop(["_year", "_month"])
                            part_dest = temp_path / f"year={year}" / f"month={month:02d}"
                            existing_df = pl.DataFrame()
                            existing_path = (
                                key_path / f"year={year}" / f"month={month:02d}"
                            )
                            if mode != "overwrite" and existing_path.exists() and any(
                                existing_path.glob("*.parquet")
                            ):
                                existing_df = pl.read_parquet(
                                    existing_path / "*.parquet"
                                )
                                self._check_schema_compatibility(
                                    existing_df.schema, part_df.schema
                                )
                            merged = self._merge_and_dedupe(existing_df, part_df)
                            if "timestamp" in merged.columns:
                                merged = merged.sort("timestamp")
                            self._write_chunked(merged, part_dest)
                    else:
                        # No timestamp partitioning; treat as single partition
                        existing_df = pl.DataFrame()
                        if mode == "append" and key_path.exists():
                            existing_glob = list(key_path.glob("*.parquet"))
                            if existing_glob:
                                existing_df = pl.read_parquet(existing_glob)
                                self._check_schema_compatibility(
                                    existing_df.schema, data.schema
                                )
                        merged = self._merge_and_dedupe(existing_df, data)
                        self._write_chunked(merged, temp_path)

                    # Atomic swap
                    if key_path.exists():
                        shutil.rmtree(key_path)
                    temp_path.rename(key_path)

                    logger.info("Wrote %d rows to %s", len(data), key)
                    write_succeeded = True

                finally:
                    if not write_succeeded and temp_path.exists():
                        shutil.rmtree(temp_path)

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

        path.mkdir(parents=True, exist_ok=True)

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

            parquet_files = sorted(key_path.rglob("*.parquet"))
            if not parquet_files:
                logger.debug("No parquet files for %s, returning empty DataFrame", key)
                return pl.DataFrame()

            # Use lazy scanning for memory efficiency
            lf = pl.scan_parquet(parquet_files)

            schema = lf.collect_schema()
            has_timestamp = "timestamp" in schema

            # Apply date filters if timestamp column exists
            if (start is not None or end is not None) and has_timestamp:
                if start is not None:
                    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
                    lf = lf.filter(pl.col("timestamp") >= start_dt)
                if end is not None:
                    end_dt = datetime.combine(end, datetime.max.time(), tzinfo=UTC)
                    lf = lf.filter(pl.col("timestamp") <= end_dt)

                lf = lf.sort("timestamp")

            # Apply column selection after filtering/sorting
            if columns:
                lf = lf.select(columns)

            # Return based on requested mode
            if batch_size is not None:
                return self._read_batched(parquet_files, columns, start, end, batch_size)

            # Use streaming engine if requested (memory-efficient for large datasets)
            result = lf.collect(engine="streaming") if streaming else lf.collect()
            logger.debug("Read %d rows from %s", len(result), key)
            return result

        except PathTraversalError:
            raise
        except pl.exceptions.PolarsError as e:
            logger.error("Failed to read from %s: %s", key, e)
            raise StorageError(f"Failed to read data: {e}") from e

    def read_latest(self, key: str, n: int = 1) -> pl.DataFrame:
        """Read the most recent rows for a key."""
        df = self.read(key)
        if df.is_empty() or "timestamp" not in df.columns:
            return pl.DataFrame()
        return df.sort("timestamp").tail(n)

    def get_gaps(
        self,
        key: str,
        start: datetime,
        end: datetime,
        expected_interval: timedelta,
    ) -> list[tuple[datetime, datetime]]:
        """Detect gaps in a time-series based on expected interval."""
        df = self.read(key, start=start.date(), end=end.date(), columns=["timestamp"])
        if df.is_empty() or "timestamp" not in df.columns:
            return [(start, end)]

        df_sorted = df.sort("timestamp")
        diffs = df_sorted.with_columns(pl.col("timestamp").diff().alias("_delta"))

        tolerance = expected_interval.total_seconds() * 1.1
        gaps: list[tuple[datetime, datetime]] = []

        for row in diffs.iter_rows(named=True):
            delta = row["_delta"]
            ts = row["timestamp"]
            if delta is None:
                continue
            if delta.total_seconds() > tolerance:
                gap_start = ts - delta + expected_interval
                gap_end = ts - expected_interval
                if gap_end < gap_start:
                    gap_end = gap_start
                gaps.append((gap_start, gap_end))

        # Check head/tail coverage
        first_ts = df_sorted["timestamp"][0]
        last_ts = df_sorted["timestamp"][-1]
        if first_ts > start:
            gap_end = min(first_ts - expected_interval, end)
            if gap_end >= start:
                gaps.insert(0, (start, gap_end))
        if last_ts < end:
            gap_start = last_ts + expected_interval
            if gap_start <= end:
                gaps.append((gap_start, end))

        return gaps

    def _read_batched(
        self,
        parquet_files: list[Path],
        columns: list[str] | None,
        start: date | None,
        end: date | None,
        batch_size: int,
    ) -> Iterator[pl.DataFrame]:
        """Read data in batches for memory efficiency.

        Args:
            parquet_files: List of parquet files to scan
            columns: Optional list of columns to return
            start: Optional start date filter (inclusive)
            end: Optional end date filter (inclusive)
            batch_size: Number of rows per batch

        Yields:
            DataFrame batches
        """
        dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")

        filter_expr = None
        if "timestamp" in dataset.schema.names and (start is not None or end is not None):
            field = ds.field("timestamp")
            if start is not None:
                start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
                filter_expr = field >= start_dt
            if end is not None:
                end_dt = datetime.combine(end, datetime.max.time(), tzinfo=UTC)
                end_expr = field <= end_dt
                filter_expr = end_expr if filter_expr is None else (filter_expr & end_expr)

        scanner = dataset.scanner(
            columns=columns,
            filter=filter_expr,
            batch_size=batch_size,
        )

        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            df = pl.from_arrow(batch)
            if "timestamp" in df.columns:
                df = df.sort("timestamp")
            yield df

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
        return any(key_path.rglob("*.parquet"))

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

        if not any(key_path.rglob("*.parquet")):
            return False

        shutil.rmtree(key_path)
        logger.info("Deleted key %s", key)
        return True

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix filter.

        Args:
            prefix: Optional prefix to filter keys (e.g., "forex/")

        Returns:
            List of matching storage keys
        """
        keys: set[str] = set()

        for parquet_file in self.data_root.rglob("*.parquet"):
            rel_parts = parquet_file.parent.relative_to(self.data_root).parts
            trimmed_parts = []
            for part in rel_parts:
                if part.startswith("year=") or part.startswith("month="):
                    break
                trimmed_parts.append(part)
            if not trimmed_parts:
                continue
            key = "/".join(trimmed_parts)
            if not prefix or key.startswith(prefix):
                keys.add(key)

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

        parquet_files = list(key_path.rglob("*.parquet"))
        if not parquet_files:
            return None

        try:
            min_dates: list[date] = []
            max_dates: list[date] = []

            for file in parquet_files:
                parsed = parse_filename(file.name)
                if parsed:
                    start_ts, end_ts = parsed
                    min_dates.append(start_ts.date())
                    max_dates.append(end_ts.date())
                else:
                    df = pl.read_parquet(file)
                    if "timestamp" not in df.columns or df.is_empty():
                        continue
                    min_ts = df["timestamp"].min()
                    max_ts = df["timestamp"].max()
                    if isinstance(min_ts, datetime) and isinstance(max_ts, datetime):
                        min_dates.append(min_ts.date())
                        max_dates.append(max_ts.date())

            if not min_dates or not max_dates:
                return None

            return (min(min_dates), max(max_dates))

        except pl.exceptions.PolarsError:
            return None

    def consolidate(
        self, key: str, target_rows_per_file: int | None = None
    ) -> dict[str, int]:
        """Consolidate parquet files for a key into larger chunks.

        Args:
            key: Storage key
            target_rows_per_file: Optional override for chunk sizing

        Returns:
            Stats dict with files_before, files_after, rows_processed
        """
        key_path = self._key_to_path(key)
        parquet_files = list(key_path.rglob("*.parquet"))
        files_before = len(parquet_files)

        df = self.read(key)
        rows = len(df)

        if rows == 0:
            return {"files_before": files_before, "files_after": files_before, "rows_processed": 0}

        original_target = self.config.target_rows_per_file
        if target_rows_per_file is not None:
            self.config = ParquetStoreConfig(
                target_rows_per_file=target_rows_per_file,
                compression=self.config.compression,
                compression_level=self.config.compression_level,
                lock_timeout_seconds=self.config.lock_timeout_seconds,
            )

        self.write(key, df, mode="overwrite")

        if target_rows_per_file is not None:
            self.config = ParquetStoreConfig(
                target_rows_per_file=original_target,
                compression=self.config.compression,
                compression_level=self.config.compression_level,
                lock_timeout_seconds=self.config.lock_timeout_seconds,
            )

        files_after = len(list(self._key_to_path(key).rglob("*.parquet")))

        return {
            "files_before": files_before,
            "files_after": files_after,
            "rows_processed": rows,
        }
