"""Storage protocols for the LIQ Stack.

This module defines the Protocol-based interfaces for storage backends.
Using Protocol instead of ABC enables structural subtyping (duck typing).
"""

from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import polars as pl

if TYPE_CHECKING:
    from liq.store.parquet import MultiReadResult


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
            def write(
                self,
                key: str,
                data: pl.DataFrame,
                mode: str = "append",
                *,
                dedupe_subset: Sequence[str] | None = None,
            ) -> None:
                ...

            def read(self, key: str, start: date | None = None, end: date | None = None) -> pl.DataFrame:
                ...

            # ... other required methods

        # ParquetStore satisfies TimeSeriesStore without explicit inheritance
        store: TimeSeriesStore = ParquetStore(...)
    """

    def write(
        self,
        key: str,
        data: pl.DataFrame,
        mode: str = "append",
        *,
        dedupe_subset: Sequence[str] | None = None,
    ) -> None:
        """Write time-series data to storage.

        Args:
            key: Storage key (e.g., "forex/EUR_USD", "crypto/BTC-USD")
            data: Polars DataFrame with time-series data
                  Expected columns: timestamp, and domain-specific columns
            mode: Write mode - "append" (default) or "overwrite"
            dedupe_subset: Optional explicit uniqueness key for tabular data.

        Raises:
            StorageError: If write operation fails
        """
        ...

    def read(self, key: str, start: date | None = None, end: date | None = None) -> pl.DataFrame:
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

    def read_multi(
        self,
        keys: Sequence[str],
        start: datetime,
        end: datetime,
        *,
        columns: Sequence[str] | None = None,
    ) -> "MultiReadResult":
        """Read one time window across N keys in a single call.

        Returns a ``MultiReadResult(data, missing_keys)`` so the
        caller can decide whether partial coverage is a fail-loud
        condition. The half-open window ``[start, end)`` is applied
        identically to every key.

        Args:
            keys: Storage keys to read; each must address a bar/time-
                series partition. The symbol is derived from the
                segment immediately preceding ``bars`` in each key
                (e.g., ``"databento/AAPL/bars/1m" → "AAPL"``).
            start: Inclusive lower bound on ``timestamp``.
            end: Exclusive upper bound on ``timestamp``.
            columns: Optional column subset (excluding ``symbol``;
                always included in the output).

        Returns:
            ``MultiReadResult`` whose ``data`` carries the selected
            columns plus ``symbol`` and whose ``missing_keys`` is
            the sorted tuple of input keys that had no data on disk.
        """
        ...
