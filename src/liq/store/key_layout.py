"""Storage-layout helpers shared by ``ParquetStore`` and migration tools.

These are isolated from ``parquet.py`` so the read-multi path, the
future ``migrate-1m-partitions`` CLI, and any out-of-band recovery
script all reason about the on-disk shape through one set of helpers.
"""

from __future__ import annotations

from pathlib import Path

_BARS_SEGMENT = "bars"
_NEW_LAYOUT_GLOB = "year=*/month=*/*.parquet"
_LEGACY_LAYOUT_GLOB = "year=*/*.parquet"


def symbol_from_bar_key(key: str) -> str:
    """Extract the symbol from a bars key.

    Convention: bar keys end with ``.../{symbol}/bars/{timeframe}``.
    The symbol is the segment immediately preceding ``bars``.

    Raises ``ValueError`` for any key that doesn't carry the ``bars``
    marker — non-bar keys (symbology, fundamentals, etc.) can't be
    cross-sectioned by ``read_multi`` and we'd rather fail loud than
    guess the wrong segment.
    """
    parts = [p for p in key.split("/") if p]
    if _BARS_SEGMENT not in parts:
        raise ValueError(
            f"key {key!r} does not contain a '{_BARS_SEGMENT}' segment; "
            "read_multi only supports bar keys (e.g. 'provider/SYM/bars/1m')"
        )
    idx = parts.index(_BARS_SEGMENT)
    if idx == 0:
        raise ValueError(f"key {key!r} is missing a symbol segment before '{_BARS_SEGMENT}'")
    return parts[idx - 1]


def is_1m_bar_key(key: str) -> bool:
    """Return True when ``key`` follows ``.../bars/1m``."""
    parts = [p for p in key.split("/") if p]
    if _BARS_SEGMENT not in parts:
        return False
    idx = parts.index(_BARS_SEGMENT)
    return idx + 1 < len(parts) and parts[idx + 1] == "1m"


def use_month_partitions(key: str) -> bool:
    """Return whether new writes for ``key`` should use month subdirs.

    Phase 3 introduces monthly partitioning for 1m bar keys only.
    Non-bar time-series keys keep the historical year/month layout for
    backwards compatibility with existing callers and tests. Bar keys
    at coarser timeframes remain yearly.
    """
    parts = [p for p in key.split("/") if p]
    if _BARS_SEGMENT not in parts:
        return True
    return is_1m_bar_key(key)


def partition_files(key_dir: Path) -> list[str]:
    """Return parquet files for a key, across both partition layouts.

    The 1m bar write path emits ``year=YYYY/month=MM/*.parquet``.
    Coarser bar keys and legacy 1m data may use ``year=YYYY/*.parquet``
    (no month subdir).
    Returning the actual file list — not globs — lets DuckDB read
    across either layout without an operator-run migration AND avoids
    DuckDB's "no files match" error when one layout is empty.
    """
    return sorted(
        {
            *(str(p) for p in key_dir.glob(_NEW_LAYOUT_GLOB)),
            *(str(p) for p in key_dir.glob(_LEGACY_LAYOUT_GLOB)),
        }
    )


__all__ = ["is_1m_bar_key", "partition_files", "symbol_from_bar_key", "use_month_partitions"]
