"""Perf benchmark for ``ParquetStore.read_multi``.

Plan §2.2 budgets (p95 over 10 timed runs after a warm-up):

* 500-symbol 1-session window: < 1.0 s
* 5000-symbol 1-session window: < 5.0 s — gated behind
  ``RUN_LARGE_PERF=1`` because writing 5000 synthetic keys to a
  tmp_path is slow and disk-heavy.

The benchmark is opt-in (``@pytest.mark.perf``) — the default
pytest invocation excludes ``-m perf`` so coverage and CI runs stay
fast. Run explicitly with ``pytest -m perf``.

Reproducible numbers land under ``artifacts/phase-3/perf.json`` after
the phase gate. The test prints p50/p95/budget so the operator can
capture the exact run without making the benchmark mutate repository
state.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from liq.store.parquet import ParquetStore

ONE_SESSION_BARS = 390  # NYSE regular-hours minutes per day
WARMUP_RUNS = 2
TIMED_RUNS = 10


def _session_bars(symbol: str, day: datetime, n: int = ONE_SESSION_BARS) -> pl.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": day + timedelta(minutes=i),
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.25,
                "volume": 1000,
            }
        )
    return pl.DataFrame(rows)


def _seed_universe(store: ParquetStore, symbols: list[str], day: datetime) -> list[str]:
    """Write one session per symbol; return the bar keys."""
    keys: list[str] = []
    for sym in symbols:
        key = f"databento/{sym}/bars/1m"
        store.write(key, _session_bars(sym, day))
        keys.append(key)
    return keys


def _percentiles(values: list[float]) -> dict[str, float]:
    sorted_v = sorted(values)
    return {
        "p50": statistics.median(sorted_v),
        "p95": sorted_v[int(0.95 * len(sorted_v)) - 1],
        "min": sorted_v[0],
        "max": sorted_v[-1],
    }


@pytest.mark.perf
def test_read_multi_500_symbols_under_1s(tmp_path: Path) -> None:
    store = ParquetStore(str(tmp_path))
    day = datetime(2024, 6, 3, 13, 30, tzinfo=UTC)
    symbols = [f"SYM{i:04d}" for i in range(500)]
    keys = _seed_universe(store, symbols, day)

    start = day
    end = day + timedelta(minutes=ONE_SESSION_BARS)

    # Warm-up — load the cold caches.
    for _ in range(WARMUP_RUNS):
        store.read_multi(keys, start=start, end=end)

    timings = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        df = store.read_multi(keys, start=start, end=end).data
        elapsed = time.perf_counter() - t0
        assert df.height == 500 * ONE_SESSION_BARS
        timings.append(elapsed)

    stats = _percentiles(timings)
    print(json.dumps({"symbols": 500, "budget_s": 1.0, **stats}))
    assert stats["p95"] < 1.0, f"p95={stats['p95']:.3f}s exceeds 1.0s budget"


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_LARGE_PERF") != "1",
    reason="set RUN_LARGE_PERF=1 to run the 5000-symbol benchmark (slow seed)",
)
def test_read_multi_5000_symbols_under_5s(tmp_path: Path) -> None:
    store = ParquetStore(str(tmp_path))
    day = datetime(2024, 6, 3, 13, 30, tzinfo=UTC)
    symbols = [f"SYM{i:05d}" for i in range(5000)]
    keys = _seed_universe(store, symbols, day)

    start = day
    end = day + timedelta(minutes=ONE_SESSION_BARS)

    for _ in range(WARMUP_RUNS):
        store.read_multi(keys, start=start, end=end)

    timings = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        store.read_multi(keys, start=start, end=end)
        timings.append(time.perf_counter() - t0)

    stats = _percentiles(timings)
    print(json.dumps({"symbols": 5000, "budget_s": 5.0, **stats}))
    assert stats["p95"] < 5.0, f"p95={stats['p95']:.3f}s exceeds 5.0s budget"
