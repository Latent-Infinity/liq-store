"""
liq-store: Storage abstraction layer for the LIQ Stack.

This package provides storage backends for time-series financial data,
including Parquet-based storage with ZSTD compression.
"""

from liq.store.exceptions import (
    ConcurrentWriteError,
    DataCorruptionError,
    DataNotFoundError,
    PathTraversalError,
    SchemaCompatibilityError,
    StorageError,
)
from liq.store.naming import generate_filename, is_timestamp_filename, parse_filename
from liq.store.parquet import ParquetStore, ParquetStoreConfig
from liq.store.protocols import TimeSeriesStore

__all__ = [
    # Protocols
    "TimeSeriesStore",
    # Implementations
    "ParquetStore",
    # Configuration
    "ParquetStoreConfig",
    # Naming utilities
    "generate_filename",
    "parse_filename",
    "is_timestamp_filename",
    # Exceptions
    "StorageError",
    "DataNotFoundError",
    "DataCorruptionError",
    "ConcurrentWriteError",
    "PathTraversalError",
    "SchemaCompatibilityError",
]

__version__ = "0.1.0"
