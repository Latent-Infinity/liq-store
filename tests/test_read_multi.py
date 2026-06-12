"""TDD tests for ``ParquetStore.read_multi``.

Contract under test:

* Cross-sectional read across N bar keys returns a long-format
  ``pl.DataFrame`` with one ``symbol`` column populated from each
  key (last segment before ``bars/...``).
* Half-open ``[start, end)`` window applied identically per key.
* Missing keys contribute zero rows (no exception — the store is not
  the scanner's coverage gate).
* Backwards compat: a key whose old data sits under a year-only
  partition (``year=YYYY/data.parquet``) reads alongside new
  monthly-partitioned data and yields a single unioned frame.
* Optional column subset reduces I/O; ``symbol`` is always added.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from liq.store.key_layout import is_1m_bar_key, symbol_from_bar_key, use_month_partitions
from liq.store.parquet import ParquetStore


def _bars(symbol: str, day: int, n_minutes: int = 5, year: int = 2024) -> pl.DataFrame:
    base = datetime(year, 6, day, 14, 30, tzinfo=UTC)
    rows = []
    for i in range(n_minutes):
        rows.append(
            {
                "timestamp": base + timedelta(minutes=i),
                "open": 100.0 + i,
                "high": 100.5 + i,
                "low": 99.5 + i,
                "close": 100.25 + i,
                "volume": 1000 + i,
                "symbol_meta": symbol,  # noise column to confirm column subset works
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(str(tmp_path))


def _key(symbol: str) -> str:
    return f"databento/{symbol}/bars/1m"


def _seed(store: ParquetStore, symbols: list[str]) -> None:
    for sym in symbols:
        store.write(_key(sym), _bars(sym, 3))


# ----- correctness ----------------------------------------------------------


class TestReadMultiCorrectness:
    def test_five_symbols_returns_long_frame_with_symbol_column(self, store: ParquetStore) -> None:
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]
        _seed(store, symbols)

        df = store.read_multi(
            [_key(s) for s in symbols],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )

        assert "symbol" in df.columns
        assert set(df["symbol"].unique().to_list()) == set(symbols)
        # 5 rows per symbol × 5 symbols
        assert df.height == 25

    def test_time_window_filters_rows(self, store: ParquetStore) -> None:
        _seed(store, ["AAPL"])
        # Only the first 2 minutes (14:30, 14:31).
        df = store.read_multi(
            [_key("AAPL")],
            start=datetime(2024, 6, 3, 14, 30, tzinfo=UTC),
            end=datetime(2024, 6, 3, 14, 32, tzinfo=UTC),
        )
        assert df.height == 2

    def test_column_subset_returns_requested_plus_symbol(self, store: ParquetStore) -> None:
        _seed(store, ["AAPL"])
        df = store.read_multi(
            [_key("AAPL")],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
            columns=["timestamp", "close"],
        )
        assert set(df.columns) == {"timestamp", "close", "symbol"}


# ----- missing-key tolerance ------------------------------------------------


class TestReadMultiMissingKeys:
    def test_missing_key_returns_other_symbols_only(self, store: ParquetStore) -> None:
        _seed(store, ["AAPL"])  # no MSFT
        df = store.read_multi(
            [_key("AAPL"), _key("MSFT")],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )
        assert set(df["symbol"].unique().to_list()) == {"AAPL"}

    def test_all_missing_returns_empty_frame(self, store: ParquetStore) -> None:
        df = store.read_multi(
            [_key("NONE")],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )
        assert df.is_empty()

    def test_empty_keys_list_returns_empty_frame(self, store: ParquetStore) -> None:
        df = store.read_multi(
            [],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )
        assert df.is_empty()


# ----- backwards-compat (legacy year-only layout) ---------------------------


class TestBackwardsCompatLayout:
    """A symbol whose history straddles the layout change reads as one
    unioned frame — the operator does not need to migrate before
    scanning."""

    def test_unions_year_only_and_year_month_layouts(
        self, tmp_path: Path, store: ParquetStore
    ) -> None:
        sym = "LEGACY"
        # First: write fresh, new-layout data.
        store.write(_key(sym), _bars(sym, 4))

        # Now synthesize a legacy year-only partition under the same key.
        # Reuse the write helper for shape, then move the file out of the
        # month=MM subdir into the year=YYYY dir.
        legacy_df = _bars(sym, 3, year=2023)
        legacy_root = tmp_path / "databento" / sym / "bars" / "1m" / "year=2023"
        legacy_root.mkdir(parents=True, exist_ok=True)
        legacy_df.write_parquet(legacy_root / "data.parquet")

        df = store.read_multi(
            [_key(sym)],
            start=datetime(2023, 1, 1, tzinfo=UTC),
            end=datetime(2025, 1, 1, tzinfo=UTC),
        )
        ts = sorted(t.date() for t in df["timestamp"].to_list())
        assert date(2023, 6, 3) in ts  # legacy year-only data surfaced
        assert date(2024, 6, 4) in ts  # new monthly-partition data surfaced


# ----- half-open window semantics -------------------------------------------


class TestHalfOpenWindow:
    def test_end_is_exclusive(self, store: ParquetStore) -> None:
        _seed(store, ["AAPL"])
        # The bar at 14:34 (5th row, index 4) is included only if end
        # is strictly after it. Setting end=14:34 should exclude it.
        df = store.read_multi(
            [_key("AAPL")],
            start=datetime(2024, 6, 3, 14, 30, tzinfo=UTC),
            end=datetime(2024, 6, 3, 14, 34, tzinfo=UTC),
        )
        assert df.height == 4


# ----- robustness ----------------------------------------------------------


class TestSymbolExtraction:
    """The extractor must accept the documented key shape and tolerate
    the segment before ``bars/...`` being the last identifier."""

    def test_provider_prefixed_key(self, store: ParquetStore) -> None:
        sym = "AAPL"
        store.write(f"databento/{sym}/bars/1m", _bars(sym, 3))
        df = store.read_multi(
            [f"databento/{sym}/bars/1m"],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )
        assert df["symbol"].unique().to_list() == ["AAPL"]

    def test_unprefixed_key(self, store: ParquetStore, tmp_path: Path) -> None:
        sym = "MSFT"
        store.write(f"{sym}/bars/1m", _bars(sym, 3))
        df = store.read_multi(
            [f"{sym}/bars/1m"],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )
        assert df["symbol"].unique().to_list() == ["MSFT"]

    def test_non_bar_key_raises(self, store: ParquetStore) -> None:
        # Universe symbology / fundamentals keys can't be cross-sectioned
        # by this method — fail loud rather than guess.
        with pytest.raises(ValueError, match="bars"):
            store.read_multi(
                ["reference/databento/symbology"],
                start=datetime(2024, 6, 3, tzinfo=UTC),
                end=datetime(2024, 6, 4, tzinfo=UTC),
            )

    def test_missing_symbol_segment_raises(self) -> None:
        with pytest.raises(ValueError, match="missing a symbol"):
            symbol_from_bar_key("bars/1m")


class TestPartitionLayoutHelpers:
    def test_is_1m_bar_key_false_for_non_bar_key(self) -> None:
        assert is_1m_bar_key("reference/databento/symbology") is False

    def test_use_month_partitions_keeps_non_bar_legacy_layout(self) -> None:
        assert use_month_partitions("forex/EUR_USD") is True

    def test_use_month_partitions_false_for_daily_bar_key(self) -> None:
        assert use_month_partitions("databento/SPY/bars/1d") is False
