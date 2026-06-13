"""Schema-stability round-trip guard.

The contract being defended: round-tripping a DataFrame through
``write`` + ``read`` returns identical column names, identical column
order, identical dtypes, and identical row count. A regression here
(accidental column reordering, dtype widening, dropped columns) would
not break a single-symbol read but would corrupt cross-sectional
``read_multi`` UNIONs the moment a schema-naive consumer relies on
positional access.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from liq.store.parquet import ParquetStore


def _bars() -> pl.DataFrame:
    base = datetime(2024, 6, 3, 14, 30, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(10)],
            "open": [100.0 + i for i in range(10)],
            "high": [100.5 + i for i in range(10)],
            "low": [99.5 + i for i in range(10)],
            "close": [100.25 + i for i in range(10)],
            "volume": [1000 + i for i in range(10)],
        }
    )


@pytest.fixture
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(str(tmp_path))


class TestSingleKeyRoundTrip:
    def test_columns_dtypes_and_order_survive_write_read(self, store: ParquetStore) -> None:
        original = _bars()
        store.write("databento/AAPL/bars/1m", original)

        roundtripped = store.read("databento/AAPL/bars/1m")

        assert roundtripped.columns == original.columns
        assert roundtripped.schema == original.schema
        assert roundtripped.height == original.height

    def test_two_distinct_keys_serialize_with_identical_schema(self, store: ParquetStore) -> None:
        original = _bars()
        store.write("databento/AAPL/bars/1m", original)
        store.write("databento/MSFT/bars/1m", original)

        a = store.read("databento/AAPL/bars/1m")
        b = store.read("databento/MSFT/bars/1m")
        assert a.schema == b.schema
        assert a.columns == b.columns


class TestReadMultiSchemaStability:
    def test_cross_sectional_read_carries_consistent_schema(self, store: ParquetStore) -> None:
        """The added ``symbol`` column is the only difference vs. a single-key
        read. Every other column survives identically."""
        original = _bars()
        for sym in ("AAPL", "MSFT"):
            store.write(f"databento/{sym}/bars/1m", original)

        result = store.read_multi(
            ["databento/AAPL/bars/1m", "databento/MSFT/bars/1m"],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )

        df = result.data
        # symbol may be added at the end (DuckDB UNION ALL convention).
        non_symbol_cols = [c for c in df.columns if c != "symbol"]
        assert non_symbol_cols == original.columns
        for col in original.columns:
            assert df.schema[col] == original.schema[col], (
                f"dtype drift on '{col}': original={original.schema[col]} "
                f"roundtripped={df.schema[col]}"
            )
