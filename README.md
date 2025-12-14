# liq-store

Storage abstraction layer for the LIQ (Latent Infinity Quant) Stack. Provides backend-agnostic storage for time-series and related tabular data using Parquet with ZSTD compression.

## Features

- **Path Traversal Protection**: Keys are validated to prevent escaping the data root
- **Atomic Writes**: Crash-safe writes using temp directories and atomic rename
- **Partition Locking**: `fcntl`-based locking prevents concurrent write corruption
- **Schema Evolution**: Append operations support adding new columns
- **Schema Validation**: Type compatibility checked on append to prevent silent corruption
- **Configurable Compression**: ZSTD compression with tunable levels
- **Memory-Efficient Reading**: Optional streaming and batch modes for large datasets
- **Standardized Keys**: Helper builders for bars/features/indicators/fundamentals so every producer shares the same layout

## Installation

```bash
pip install liq-store
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Write and Read

```python
from datetime import UTC, datetime
from decimal import Decimal

import polars as pl

from liq.store import ParquetStore, key_builder

# Initialize store
store = ParquetStore("./data")

# Create sample OHLCV data
df = pl.DataFrame({
    "timestamp": [
        datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 10, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 10, 2, tzinfo=UTC),
    ],
    "symbol": ["EUR_USD"] * 3,
    "open": [1.1000, 1.1005, 1.1010],
    "high": [1.1010, 1.1015, 1.1020],
    "low": [1.0995, 1.1000, 1.1005],
    "close": [1.1005, 1.1010, 1.1015],
    "volume": [1000, 1500, 1200],
})

# Write data (provider-prefixed + standardized bars key)
store.write(f"oanda/{key_builder.bars('EUR_USD', '1m')}", df)

# Read data back
result = store.read(f"oanda/{key_builder.bars('EUR_USD', '1m')}")
print(result)
```

### Append Mode

```python
# Append new data (automatically deduplicates by timestamp/symbol/provider when present)
new_data = pl.DataFrame({
    "timestamp": [datetime(2024, 1, 1, 10, 3, tzinfo=UTC)],
    "symbol": ["EUR_USD"],
    "open": [1.1015],
    "high": [1.1025],
    "low": [1.1010],
    "close": [1.1020],
    "volume": [1100],
})

store.write(f"oanda/{key_builder.bars('EUR_USD', '1m')}", new_data, mode="append")
```

### Date Filtering

```python
from datetime import date

# Read specific date range
df = store.read(
    f"oanda/{key_builder.bars('EUR_USD', '1m')}",
    start=date(2024, 1, 1),
    end=date(2024, 1, 31),
)
```

### Memory-Efficient Reading

```python
# Column subset (reduces memory)
df = store.read(f"oanda/{key_builder.bars('EUR_USD', '1m')}", columns=["timestamp", "close"])

# Streaming mode (memory-efficient for large datasets)
df = store.read(f"oanda/{key_builder.bars('EUR_USD', '1m')}", streaming=True)

# Batch processing (very large datasets)
for batch in store.read(f"oanda/{key_builder.bars('EUR_USD', '1m')}", batch_size=100_000):
    process(batch)
```

### Custom Configuration

```python
from liq.store import ParquetStore, ParquetStoreConfig

# Custom configuration
config = ParquetStoreConfig(
    target_rows_per_file=100_000,  # Rows per Parquet file
    compression="zstd",
    compression_level=6,           # Higher = more compression, slower
    lock_timeout_seconds=60,
)

store = ParquetStore("./data", config=config)
```

### Other Operations

```python
# Check if data exists
if store.exists(f"oanda/{key_builder.bars('EUR_USD', '1m')}"):
    print("Data available")

# Get date range of available data
date_range = store.get_date_range(f"oanda/{key_builder.bars('EUR_USD', '1m')}")
if date_range:
    start, end = date_range
    print(f"Data from {start} to {end}")

# List all keys
keys = store.list_keys()
print(keys)  # ['oanda/EUR_USD/bars/1m', 'binance/BTC_USDT/bars/1h']

# List keys with prefix filter
oanda_keys = store.list_keys(prefix="oanda/")
print(oanda_keys)  # ['oanda/EUR_USD/bars/1m']

# Delete data
store.delete(f"oanda/{key_builder.bars('EUR_USD', '1m')}")
```

## Storage Layout

Data is stored in a hierarchical directory structure:

```
data_root/
    oanda/
        EUR_USD/
            bars/
                1m/
                    year=2024/month=01/20240101T100000-20240101T235959.parquet
                    year=2024/month=02/20240201T000000-20240201T235959.parquet
            features/
                alpha_v1/data.parquet
            fundamentals/data.parquet
```

Filenames encode the timestamp range of contained data in ISO 8601 compact format. Timestamped data is partitioned by year/month; non-timestamp data stays unpartitioned in `data.parquet` chunks under the key directory. Empty writes are ignored to avoid clutter.

### Key schema helpers

Use the helpers in `liq.store.key_builder` to stay consistent across services:

- `key_builder.bars(symbol, timeframe)` → `"EUR_USD/bars/1m"`
- `key_builder.features(symbol, feature_set)` → `"EUR_USD/features/alpha_v1"`
- `key_builder.indicators(symbol, indicator, params_id)` → `"EUR_USD/indicators/rsi/14"`
- `key_builder.fundamentals(symbol)` → `"EUR_USD/fundamentals"`

Keys are typically prefixed by provider (e.g., `oanda/{bars_key}`).

## Exception Handling

```python
from liq.store import (
    ParquetStore,
    StorageError,
    PathTraversalError,
    ConcurrentWriteError,
    SchemaCompatibilityError,
    key_builder,
)

store = ParquetStore("./data")

try:
    store.write(f"oanda/{key_builder.bars('EUR_USD', '1m')}", df)
except PathTraversalError:
    # Key attempted to escape data root (e.g., "../secrets")
    print("Invalid key: path traversal detected")
except ConcurrentWriteError:
    # Another process is writing to this partition
    print("Partition is locked by another writer")
except SchemaCompatibilityError as e:
    # Append data has incompatible column types
    print(f"Schema mismatch: {e}")
except StorageError as e:
    # General storage error (I/O, Polars, etc.)
    print(f"Storage error: {e}")
```

## API Reference

### ParquetStore

| Method | Description |
|--------|-------------|
| `write(key, data, mode="append")` | Write DataFrame to storage |
| `read(key, start=None, end=None, columns=None, streaming=False, batch_size=None)` | Read DataFrame from storage |
| `exists(key)` | Check if data exists |
| `delete(key)` | Delete all data for key |
| `list_keys(prefix="")` | List all keys with optional prefix |
| `get_date_range(key)` | Get (start, end) dates of data |

### ParquetStoreConfig

| Field | Default | Description |
|-------|---------|-------------|
| `target_rows_per_file` | 150,000 | Target rows per Parquet file |
| `compression` | "zstd" | Compression algorithm |
| `compression_level` | 3 | Compression level (1-22 for zstd) |
| `lock_timeout_seconds` | 30 | Lock acquisition timeout |

### Exceptions

| Exception | When Raised |
|-----------|-------------|
| `StorageError` | Base exception for storage errors |
| `PathTraversalError` | Key contains path traversal (e.g., `../`) |
| `ConcurrentWriteError` | Partition is locked by another writer |
| `SchemaCompatibilityError` | Append data has incompatible schema |
| `DataNotFoundError` | Requested data not found |
| `DataCorruptionError` | Stored data is corrupted |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=liq.store --cov-report=term-missing

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

## License

MIT License - see [LICENSE](LICENSE) for details.
