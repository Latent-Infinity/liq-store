# `liq-store` structured log catalog

`liq-store` emits structured log records via the standard `logging`
module. Fields are attached via the `extra=` kwarg so they land as
queryable attributes on `LogRecord` rather than being baked into the
message string. This file catalogs the events worth filtering on.

Each subsection's event name matches the `event=` field literally.

## `read_multi`

Emitted once per call to `ParquetStore.read_multi(...)`. The record
lets an operator reconstruct a sweep without inspecting application-
level state — keys count + window + latency + missing-key count is
enough to triage "why was this sweep slow" or "why did this scan
return short."

Logger: `liq.store.parquet` (INFO).

| Field | Type | Description |
| --- | --- | --- |
| `event` | `str` | Always `"read_multi"` |
| `keys_count` | `int` | Number of input keys passed to the call |
| `start` | `str` | ISO timestamp of the half-open window's lower bound |
| `end` | `str` | ISO timestamp of the half-open window's upper bound |
| `latency_ms` | `int` | Wall-clock duration of the entire call, including the DuckDB plan + filesystem walk |
| `missing_count` | `int` | Count of input keys with no parquet files on disk; mirrors `len(MultiReadResult.missing_keys)` |

Example:

```
2026-06-12T18:42:11Z INFO  read_multi
  event=read_multi keys_count=500 start=2024-06-03T13:30:00+00:00
  end=2024-06-03T20:00:00+00:00 latency_ms=412 missing_count=3
```

The event is emitted exactly once even for empty `keys` lists, so
"this scanner stopped calling read_multi" is observable via gap in
the event stream rather than ambiguity around zero records.

## Reconstructing a sweep

Filter the log stream by the run-level correlation id the caller
emits (e.g., a `scan_run_id` or `sync_run_id`) and join with the
read_multi events that fired during the same time window:

```bash
jq 'select(.event == "read_multi" and .ts >= "2026-06-12T18:00:00Z")' \
  liq-store.log
```

Application-level correlation keys live in the *caller's* records —
liq-store stays storage-agnostic and does not invent its own
sync/scan ids.
