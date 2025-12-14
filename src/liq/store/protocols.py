"""Storage protocols for the LIQ Stack.

This module defines the Protocol-based interfaces for storage backends.
Using Protocol instead of ABC enables structural subtyping (duck typing).
"""

from datetime import date
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class TimeSeriesStore(Protocol):
    """Protocol for backend-agnostic time-series storage.

    Implementations must provide methods for reading, writing, and managing
    time-series data. This protocol uses structural subtyping, so any class
    with the correct method signatures will satisfy the protocol.

    Key design principles:
    - Key-based access (e.g., "forex/EUR_USD" or "crypto/BTC-USD")
    - Polars DataFrame as the data interchange format
    - Optional date filtering for reads
    - Append vs overwrite modes for writes

    Example:
        class ParquetStore:
            def write(self, key: str, data: pl.DataFrame, mode: str = "append") -> None:
                ...

            def read(self, key: str, start: date | None = None, end: date | None = None) -> pl.DataFrame:
                ...

            # ... other required methods

        # ParquetStore satisfies TimeSeriesStore without explicit inheritance
        store: TimeSeriesStore = ParquetStore(...)
    """

    def write(self, key: str, data: pl.DataFrame, mode: str = "append") -> None:
        """Write time-series data to storage.

        Args:
            key: Storage key (e.g., "forex/EUR_USD", "crypto/BTC-USD")
            data: Polars DataFrame with time-series data
                  Expected columns: timestamp, and domain-specific columns
            mode: Write mode - "append" (default) or "overwrite"

        Raises:
            StorageError: If write operation fails
        """
        ...

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

        Raises:
            StorageError: If read operation fails
            DataCorruptionError: If stored data is corrupted
        """
        ...

    def read_latest(self, key: str, n: int = 1) -> pl.DataFrame:
        """Read the most recent rows for a key.

        Args:
            key: Storage key
            n: Number of most recent rows to return (default: 1)

        Returns:
            Polars DataFrame with the most recent n rows (sorted ascending by timestamp)

        Raises:
            StorageError: If read operation fails
            DataCorruptionError: If stored data is corrupted
        """
        ...

    def exists(self, key: str) -> bool:
        """Check if data exists for a key.

        Args:
            key: Storage key

        Returns:
            True if data exists, False otherwise
        """
        ...

    def delete(self, key: str) -> bool:
        """Delete all data for a key.

        Args:
            key: Storage key

        Returns:
            True if data was deleted, False if key didn't exist
        """
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix filter.

        Args:
            prefix: Optional prefix to filter keys (e.g., "forex/")

        Returns:
            List of matching storage keys
        """
        ...

    def get_date_range(self, key: str) -> tuple[date, date] | None:
        """Get the date range of available data for a key.

        Args:
            key: Storage key

        Returns:
            Tuple of (earliest_date, latest_date) or None if no data
        """
        ...
