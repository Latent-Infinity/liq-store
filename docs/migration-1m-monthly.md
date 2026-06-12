# Operator runbook: migrating legacy 1m partitions

## Background

Two partition layouts exist for 1m bar keys on disk:

* **1m current** — `{key}/year=YYYY/month=MM/*.parquet`
* **1m legacy** — `{key}/year=YYYY/*.parquet` (no month subdir)

Coarser bar timeframes remain yearly by design; `1d`, `1h`, and other
non-1m bar keys are not migration candidates.

Reads via `ParquetStore.read` and `ParquetStore.read_multi` already
union across both layouts via
`liq.store.key_layout.partition_files`, so **there is no functional
reason to migrate** for the scanner to work. Migration is a
maintenance task that improves write-time partition pruning for
high-cardinality 1m keys.

## Identifying legacy keys

```bash
liq-store migrate-1m-partitions --dry-run --root /path/to/data
```

Output (JSON, stdout):

```json
{
  "dry_run": true,
  "root": "/path/to/data",
  "legacy_keys": [
    "databento/AAPL/bars/1m",
    "databento/MSFT/bars/1m"
  ],
  "count": 2
}
```

Pipe through `jq` for selective filtering. The command never moves
files — running without `--dry-run` exits non-zero with a clear
message. The actual migration is deliberately not implemented in
this phase; see "Why the move command is not implemented yet"
below.

## Manual migration (when actually needed)

For each key reported by the dry-run:

1. For every `{key}/year=YYYY/*.parquet` file, derive its month from
   the timestamps inside.
2. Create `{key}/year=YYYY/month=MM/` if missing.
3. Atomically move (or split, if the file straddles months) the
   file into the new directory.
4. Verify a `read_multi` round-trip still returns the same rows.

The `ParquetStoreConfig.target_rows_per_file` setting controls
chunk sizing — splitting on month boundaries is naturally aligned
with the chunking strategy in the current write path.

## Why the move command is not implemented yet

The read path already handles both layouts, so a migration is a
storage hygiene exercise rather than a correctness gate. Designing
an in-place rewrite that respects ongoing writers (the partition
lock in `parquet.py::_partition_lock` is per-key, but a migration
walks many keys) is a meaningful piece of work and is out of scope
for the phase that introduced the new layout. The dry-run command
exists so an operator can quantify the work without writing the
mover prematurely.
