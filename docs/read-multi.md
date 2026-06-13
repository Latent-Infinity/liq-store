# `ParquetStore.read_multi` — cross-sectional reads

`read_multi` is the one-call read path for scanning data across many
symbols within a single time window. The scanner reads one window at
a time across the universe; without `read_multi` it would have to
issue one `read(...)` per symbol and pay the per-call overhead N
times.

## Contract

```python
def read_multi(
    self,
    keys: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    columns: Sequence[str] | None = None,
) -> MultiReadResult
```

`MultiReadResult` is a NamedTuple:

```python
class MultiReadResult(NamedTuple):
    data: pl.DataFrame
    missing_keys: tuple[str, ...]
```

* `keys` — bar keys (e.g. `"databento/AAPL/bars/1m"`). The symbol is
  parsed from the segment immediately preceding `bars`. Non-bar keys
  raise `ValueError` — `read_multi` only handles time-series bars.
* `start` / `end` — half-open `[start, end)` window applied
  identically to every key.
* `columns` — optional column subset (e.g. `["timestamp", "close"]`).
  `symbol` is always added to the output.
* `result.data` — long-format `pl.DataFrame` sorted by `(symbol,
  timestamp)`. Empty `DataFrame` when every key is missing or
  produces zero rows in the window.
* `result.missing_keys` — sorted tuple of input keys whose on-disk
  partition was empty. Useful for the scanner to fail loud on
  partial coverage without re-checking `exists()` per key.

`MultiReadResult` is a NamedTuple, so destructuring also works:

```python
data, missing = store.read_multi(keys, start, end)
```

## Missing-key semantics

Missing keys contribute zero rows but are **not** errors. The store
is data-availability-agnostic — failing loud when coverage is
incomplete is the scanner's job, not the store's. Callers that need
"all keys must have data" can inspect `result.missing_keys` directly
and raise if it is non-empty.

## Backwards-compat layout

A `read_multi` call automatically unions across both partition
layouts a 1m bar key might have on disk:

* 1m current: `{key}/year=YYYY/month=MM/*.parquet`
* Coarser bars and legacy 1m: `{key}/year=YYYY/*.parquet` (no month subdir)

The unification happens via
`liq.store.key_layout.partition_files(key_dir)`, which the read
path calls before handing files to DuckDB. Operators do not need to
migrate before scanning; see `docs/migration-1m-monthly.md` for the
forward path.

## Performance

Plan §2.2 budgets (p95 over 10 timed runs after a warm-up, on the
reference hardware):

| Universe size | Window | Budget |
| --- | --- | --- |
| 500 symbols | 1 session | <1.0 s |
| 5000 symbols | 1 session | <5.0 s |

`tests/perf/test_read_multi.py` enforces both. The 5000-symbol case
is gated behind `RUN_LARGE_PERF=1` because seeding 5000 synthetic
keys is itself slow. Run with `pytest -m perf`. Default `pytest`
runs do not include the perf benchmarks.

## Why DuckDB

The implementation issues one DuckDB `read_parquet([...],
filename=true)` scan over the full file list, then joins each row to a
temporary filename-to-symbol table. DuckDB's `read_parquet` is
significantly faster than per-key `polars.scan_parquet` when N keys
are involved, because file-open and predicate-pushdown overhead is
amortized across one vectorized scan rather than paid per Python call.

DuckDB sits in-memory only — no on-disk database is created and the
connection is closed at the end of every call.
