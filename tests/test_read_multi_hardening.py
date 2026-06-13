"""TDD tests for hardened ``ParquetStore.read_multi`` behaviour.

Covers:

* ``MultiReadResult`` wrapper — the scanner needs an *explicit*
  missing-keys signal so it can fail loud on partial coverage. A
  bare ``pl.DataFrame`` return type drops that information silently.
* Structured logging — a single ``event=read_multi`` record per call
  with ``keys_count``, ``start``, ``end``, ``latency_ms``,
  ``missing_count`` so an operator can reconstruct a sweep.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from liq.store import MultiReadResult
from liq.store.parquet import ParquetStore


def _bars(symbol: str, n: int = 5) -> pl.DataFrame:
    base = datetime(2024, 6, 3, 14, 30, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(n)],
            "open": [100.0 + i for i in range(n)],
            "high": [100.5 + i for i in range(n)],
            "low": [99.5 + i for i in range(n)],
            "close": [100.25 + i for i in range(n)],
            "volume": [1000 + i for i in range(n)],
        }
    )


def _key(symbol: str) -> str:
    return f"databento/{symbol}/bars/1m"


@pytest.fixture
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(str(tmp_path))


# ----- MultiReadResult wrapper ----------------------------------------------


class TestMultiReadResult:
    def test_multi_read_result_exported_from_package(self) -> None:
        from liq.store import MultiReadResult as Exported

        assert Exported is MultiReadResult

    def test_returns_multi_read_result_with_data_and_missing(self, store: ParquetStore) -> None:
        store.write(_key("AAPL"), _bars("AAPL"))

        result = store.read_multi(
            [_key("AAPL"), _key("MSFT")],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )

        assert isinstance(result, MultiReadResult)
        assert isinstance(result.data, pl.DataFrame)
        assert result.missing_keys == (_key("MSFT"),)
        # Iterable / tuple-style access remains available for ergonomics.
        data, missing = result
        assert data.height == 5
        assert missing == (_key("MSFT"),)

    def test_all_present_yields_empty_missing_tuple(self, store: ParquetStore) -> None:
        for sym in ("AAPL", "MSFT"):
            store.write(_key(sym), _bars(sym))

        result = store.read_multi(
            [_key("AAPL"), _key("MSFT")],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )
        assert result.missing_keys == ()
        assert set(result.data["symbol"].unique().to_list()) == {"AAPL", "MSFT"}

    def test_all_missing_yields_empty_data_and_full_missing_list(self, store: ParquetStore) -> None:
        result = store.read_multi(
            [_key("NONE1"), _key("NONE2")],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )
        assert result.data.is_empty()
        assert sorted(result.missing_keys) == sorted([_key("NONE1"), _key("NONE2")])

    def test_empty_keys_list_returns_empty_result(self, store: ParquetStore) -> None:
        result = store.read_multi(
            [],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )
        assert result.data.is_empty()
        assert result.missing_keys == ()


# ----- structured logging ----------------------------------------------------


class TestReadMultiLogging:
    def test_emits_single_read_multi_event_with_required_fields(
        self,
        store: ParquetStore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        for sym in ("AAPL", "MSFT"):
            store.write(_key(sym), _bars(sym))

        caplog.set_level(logging.INFO, logger="liq.store.parquet")
        store.read_multi(
            [_key("AAPL"), _key("MSFT"), _key("MISSING")],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )

        events = [r for r in caplog.records if getattr(r, "event", None) == "read_multi"]
        assert len(events) == 1
        record = events[0]
        assert record.keys_count == 3
        assert record.missing_count == 1
        assert record.start == "2024-06-03T00:00:00+00:00"
        assert record.end == "2024-06-04T00:00:00+00:00"
        latency = record.latency_ms
        assert isinstance(latency, int | float)
        assert latency >= 0

    def test_empty_keys_still_emits_event(
        self,
        store: ParquetStore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO, logger="liq.store.parquet")
        store.read_multi(
            [],
            start=datetime(2024, 6, 3, tzinfo=UTC),
            end=datetime(2024, 6, 4, tzinfo=UTC),
        )
        events = [r for r in caplog.records if getattr(r, "event", None) == "read_multi"]
        assert len(events) == 1
        assert events[0].keys_count == 0
        assert events[0].missing_count == 0

    def test_invalid_key_still_emits_event_before_raising(
        self,
        store: ParquetStore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO, logger="liq.store.parquet")

        with pytest.raises(ValueError):
            store.read_multi(
                ["reference/databento/symbology"],
                start=datetime(2024, 6, 3, tzinfo=UTC),
                end=datetime(2024, 6, 4, tzinfo=UTC),
            )

        events = [r for r in caplog.records if getattr(r, "event", None) == "read_multi"]
        assert len(events) == 1
        assert events[0].keys_count == 1
        assert events[0].missing_count == 0
