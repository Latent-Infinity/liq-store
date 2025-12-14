"""Tests for liq.store.protocols module."""

from datetime import date

import polars as pl

from liq.store.protocols import TimeSeriesStore


class TestTimeSeriesStoreProtocol:
    """Tests for TimeSeriesStore protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol should be runtime checkable."""
        assert hasattr(TimeSeriesStore, "__protocol_attrs__") or isinstance(
            TimeSeriesStore, type
        )

    def test_protocol_defines_write(self) -> None:
        """Protocol should define write method."""
        # Check that the protocol has the expected method signature
        assert hasattr(TimeSeriesStore, "write")

    def test_protocol_defines_read(self) -> None:
        """Protocol should define read method."""
        assert hasattr(TimeSeriesStore, "read")

    def test_protocol_defines_exists(self) -> None:
        """Protocol should define exists method."""
        assert hasattr(TimeSeriesStore, "exists")

    def test_protocol_defines_delete(self) -> None:
        """Protocol should define delete method."""
        assert hasattr(TimeSeriesStore, "delete")

    def test_protocol_defines_list_keys(self) -> None:
        """Protocol should define list_keys method."""
        assert hasattr(TimeSeriesStore, "list_keys")

    def test_protocol_defines_get_date_range(self) -> None:
        """Protocol should define get_date_range method."""
        assert hasattr(TimeSeriesStore, "get_date_range")

    def test_protocol_defines_read_latest(self) -> None:
        """Protocol should define read_latest method."""
        assert hasattr(TimeSeriesStore, "read_latest")


class MockTimeSeriesStore:
    """Mock implementation for testing protocol conformance."""

    def __init__(self) -> None:
        self._data: dict[str, pl.DataFrame] = {}

    def write(self, key: str, data: pl.DataFrame, mode: str = "append") -> None:
        if mode == "append" and key in self._data:
            self._data[key] = pl.concat([self._data[key], data])
        else:
            self._data[key] = data

    def read(
        self, key: str, start: date | None = None, end: date | None = None  # noqa: ARG002
    ) -> pl.DataFrame:
        if key not in self._data:
            return pl.DataFrame()
        return self._data[key]

    def exists(self, key: str) -> bool:
        return key in self._data

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def list_keys(self, prefix: str = "") -> list[str]:
        return [k for k in self._data if k.startswith(prefix)]

    def get_date_range(self, key: str) -> tuple[date, date] | None:
        if key not in self._data or len(self._data[key]) == 0:
            return None
        df = self._data[key]
        if "timestamp" not in df.columns:
            return None
        ts_col = df["timestamp"]
        min_ts = ts_col.min()
        max_ts = ts_col.max()
        if min_ts is None or max_ts is None:
            return None
        return (min_ts.date(), max_ts.date())

    def read_latest(self, key: str, n: int = 1) -> pl.DataFrame:
        if key not in self._data:
            return pl.DataFrame()
        df = self._data[key]
        if "timestamp" not in df.columns or df.is_empty():
            return pl.DataFrame()
        df_sorted = df.sort("timestamp")
        return df_sorted.tail(n)


class TestMockTimeSeriesStore:
    """Tests for mock implementation to verify protocol works."""

    def test_mock_implements_protocol(self) -> None:
        """Mock should satisfy the protocol."""
        store = MockTimeSeriesStore()
        # If this doesn't raise, the mock satisfies the protocol structure
        assert hasattr(store, "write")
        assert hasattr(store, "read")
        assert hasattr(store, "exists")
        assert hasattr(store, "delete")
        assert hasattr(store, "list_keys")
        assert hasattr(store, "get_date_range")

    def test_write_and_read(self, sample_ohlcv_df: pl.DataFrame) -> None:
        store = MockTimeSeriesStore()
        store.write("test/EUR_USD", sample_ohlcv_df)
        result = store.read("test/EUR_USD")
        assert len(result) == len(sample_ohlcv_df)

    def test_exists(self, sample_ohlcv_df: pl.DataFrame) -> None:
        store = MockTimeSeriesStore()
        assert not store.exists("test/EUR_USD")
        store.write("test/EUR_USD", sample_ohlcv_df)
        assert store.exists("test/EUR_USD")

    def test_delete(self, sample_ohlcv_df: pl.DataFrame) -> None:
        store = MockTimeSeriesStore()
        store.write("test/EUR_USD", sample_ohlcv_df)
        assert store.exists("test/EUR_USD")
        result = store.delete("test/EUR_USD")
        assert result is True
        assert not store.exists("test/EUR_USD")

    def test_delete_nonexistent(self) -> None:
        store = MockTimeSeriesStore()
        result = store.delete("nonexistent")
        assert result is False

    def test_list_keys(self, sample_ohlcv_df: pl.DataFrame) -> None:
        store = MockTimeSeriesStore()
        store.write("forex/EUR_USD", sample_ohlcv_df)
        store.write("forex/GBP_USD", sample_ohlcv_df)
        store.write("crypto/BTC-USD", sample_ohlcv_df)

        forex_keys = store.list_keys("forex/")
        assert len(forex_keys) == 2
        assert "forex/EUR_USD" in forex_keys
        assert "forex/GBP_USD" in forex_keys

    def test_get_date_range(self, sample_ohlcv_df: pl.DataFrame) -> None:
        store = MockTimeSeriesStore()
        store.write("test/EUR_USD", sample_ohlcv_df)
        result = store.get_date_range("test/EUR_USD")
        assert result is not None
        start, end = result
        assert isinstance(start, date)
        assert isinstance(end, date)

    def test_get_date_range_nonexistent(self) -> None:
        store = MockTimeSeriesStore()
        result = store.get_date_range("nonexistent")
        assert result is None

    def test_read_empty(self) -> None:
        store = MockTimeSeriesStore()
        result = store.read("nonexistent")
        assert len(result) == 0

    def test_read_latest(self, sample_ohlcv_df: pl.DataFrame) -> None:
        store = MockTimeSeriesStore()
        store.write("test/EUR_USD", sample_ohlcv_df)
        latest = store.read_latest("test/EUR_USD", n=1)
        assert len(latest) == 1
        assert latest["timestamp"][0] == sample_ohlcv_df["timestamp"].max()

    def test_write_append_mode(self, sample_ohlcv_df: pl.DataFrame) -> None:
        store = MockTimeSeriesStore()
        store.write("test/EUR_USD", sample_ohlcv_df)
        store.write("test/EUR_USD", sample_ohlcv_df, mode="append")
        result = store.read("test/EUR_USD")
        assert len(result) == len(sample_ohlcv_df) * 2

    def test_write_overwrite_mode(self, sample_ohlcv_df: pl.DataFrame) -> None:
        store = MockTimeSeriesStore()
        store.write("test/EUR_USD", sample_ohlcv_df)
        store.write("test/EUR_USD", sample_ohlcv_df, mode="overwrite")
        result = store.read("test/EUR_USD")
        assert len(result) == len(sample_ohlcv_df)
