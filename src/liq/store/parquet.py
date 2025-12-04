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

from datetime import date, datetime
from pathlib import Path

import polars as pl

from liq.store.exceptions import StorageError
from liq.store.naming import generate_filename

# Target rows per file: ~150k rows = ~4-7 days of 1-minute data
TARGET_ROWS_PER_FILE = 150_000


class ParquetStore:
    """Parquet-based storage implementation.

    Stores time-series data in Parquet files with ZSTD compression.
    Supports key-based access where keys can contain path separators
    (e.g., "forex/EUR_USD").

    Example:
        store = ParquetStore("./data")
        store.write("forex/EUR_USD", df)
        df = store.read("forex/EUR_USD", start=date(2024, 1, 1))
    """

    def __init__(self, data_root: str) -> None:
        """Initialize ParquetStore.

        Args:
            data_root: Root directory for data storage
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert storage key to filesystem path."""
        return self.data_root / key

    def write(self, key: str, data: pl.DataFrame, mode: str = "append") -> None:
        """Write time-series data to storage.

        Args:
            key: Storage key (e.g., "forex/EUR_USD")
            data: Polars DataFrame with time-series data
            mode: Write mode - "append" (default) or "overwrite"

        Raises:
            StorageError: If write operation fails
        """
        try:
            key_path = self._key_to_path(key)
            key_path.mkdir(parents=True, exist_ok=True)

            if mode == "overwrite":
                # Remove existing files
                for f in key_path.glob("*.parquet"):
                    f.unlink()
                output_df = data
            elif mode == "append" and any(key_path.glob("*.parquet")):
                # Read existing data and combine
                existing_df = pl.read_parquet(key_path / "*.parquet")
                combined_df = pl.concat([existing_df, data])

                # Deduplicate by timestamp if timestamp column exists
                if "timestamp" in combined_df.columns:
                    output_df = combined_df.unique(subset=["timestamp"], keep="last").sort(
                        "timestamp"
                    )
                else:
                    output_df = combined_df

                # Remove old files before writing new
                for f in key_path.glob("*.parquet"):
                    f.unlink()
            else:
                output_df = data
                if "timestamp" in output_df.columns:
                    output_df = output_df.sort("timestamp")

            # Write data
            self._write_chunked(output_df, key_path)

        except pl.exceptions.PolarsError as e:
            raise StorageError(f"Failed to write data: {e}") from e
        except OSError as e:
            raise StorageError(f"Failed to write data: {e}") from e

    def _write_chunked(self, df: pl.DataFrame, path: Path) -> None:
        """Write DataFrame in optimally-sized chunks."""
        if df.is_empty():
            return

        # If no timestamp column, write as single file
        if "timestamp" not in df.columns:
            df.write_parquet(
                path / "data.parquet",
                compression="zstd",
                compression_level=3,
            )
            return

        total_rows = len(df)

        # Split into chunks
        for start_idx in range(0, total_rows, TARGET_ROWS_PER_FILE):
            end_idx = min(start_idx + TARGET_ROWS_PER_FILE, total_rows)
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
                compression="zstd",
                compression_level=3,
            )

    def read(
        self, key: str, start: date | None = None, end: date | None = None
    ) -> pl.DataFrame:
        """Read time-series data from storage.

        Args:
            key: Storage key
            start: Optional start date filter (inclusive)
            end: Optional end date filter (inclusive)

        Returns:
            Polars DataFrame with time-series data
            Empty DataFrame if no data found
        """
        try:
            key_path = self._key_to_path(key)

            if not key_path.exists():
                return pl.DataFrame()

            parquet_files = list(key_path.glob("*.parquet"))
            if not parquet_files:
                return pl.DataFrame()

            # Read all parquet files
            df = pl.read_parquet(key_path / "*.parquet")

            # Apply date filters if timestamp column exists
            if "timestamp" in df.columns and (start is not None or end is not None):
                from datetime import UTC

                if start is not None:
                    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
                    df = df.filter(pl.col("timestamp") >= start_dt)
                if end is not None:
                    end_dt = datetime.combine(end, datetime.max.time(), tzinfo=UTC)
                    df = df.filter(pl.col("timestamp") <= end_dt)

                df = df.sort("timestamp")

            return df

        except pl.exceptions.PolarsError as e:
            raise StorageError(f"Failed to read data: {e}") from e

    def exists(self, key: str) -> bool:
        """Check if data exists for a key."""
        key_path = self._key_to_path(key)
        if not key_path.exists():
            return False
        return any(key_path.glob("*.parquet"))

    def delete(self, key: str) -> bool:
        """Delete all data for a key.

        Returns:
            True if data was deleted, False if key didn't exist
        """
        key_path = self._key_to_path(key)

        if not key_path.exists():
            return False

        parquet_files = list(key_path.glob("*.parquet"))
        if not parquet_files:
            return False

        # Remove all parquet files
        for f in parquet_files:
            f.unlink()

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

        return True

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix filter."""
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

        Returns:
            Tuple of (earliest_date, latest_date) or None if no data
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

            return (min_ts.date(), max_ts.date())

        except pl.exceptions.PolarsError:
            return None
