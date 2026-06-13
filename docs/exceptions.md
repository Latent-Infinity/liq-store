# `liq-store` exception catalog

All custom exceptions live in `liq.store.exceptions` and derive from
`StorageError` so consumers can catch the entire class hierarchy with
one `except` clause. The hierarchy below mirrors the source file —
keep them in sync when a new class is added.

## Hierarchy

```
StorageError                       — root for every liq-store error
├── PathTraversalError             — key resolves outside data_root
├── ConcurrentWriteError           — partition lock could not be acquired
├── SchemaCompatibilityError       — append would change column dtypes
├── DataNotFoundError              — write_overwrite called with require_exists=True against a missing key
└── DataCorruptionError            — stored data failed an integrity check
```

## When each is raised

| Exception | Where | Retry-eligible? | Recovery |
| --- | --- | --- | --- |
| `StorageError` | Base; concrete subclasses below | n/a | — |
| `PathTraversalError` | `_key_to_path` rejects any key whose resolved path falls outside `data_root` (e.g., `"../escape"`, absolute paths). | **No** — programmer error, not a race. | Fix the caller's key construction. |
| `ConcurrentWriteError` | `write(...)` cannot acquire the per-partition `fcntl` lock because another writer holds it. | **Yes** — bounded retry with backoff is appropriate; the lock auto-releases when the other writer exits. | The caller can wait and retry, or surface the contention to the operator. |
| `SchemaCompatibilityError` | `write(..., mode="append")` detects an existing partition whose column dtype differs from the appending DataFrame in a non-coercible way. | **No** — appending would silently widen or truncate the column on disk. | Re-write with `mode="overwrite"`, or coerce the appending data to the on-disk dtype before calling. |
| `DataNotFoundError` | `write_overwrite(..., require_exists=True)` is asked to overwrite a key that does not exist. | **No** — caller asserted "I know this exists." | Either drop `require_exists` or seed the key before overwriting. |
| `DataCorruptionError` | Read paths detect truncated, unreadable, or schema-misaligned parquet files. | **No** — file-level data integrity failure. | Inspect the affected parquet files; consider re-ingesting from source. |

## Cross-sectional reads

`ParquetStore.read_multi` does **not** raise on missing keys. Missing
keys surface in `MultiReadResult.missing_keys` so the scanner can fail
loud or proceed at its own discretion — losing that signal in a
generic exception would force every caller to re-check `exists()` per
key.

The only exception `read_multi` raises is `ValueError` when an input
key does not contain a `bars` segment (the symbol-extraction
contract). Non-bar keys (symbology, fundamentals, reference data) are
not cross-sectionable by this method.
