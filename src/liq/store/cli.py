"""Minimal CLI surface for ``liq-store`` maintenance commands.

Currently exposes one subcommand:

* ``migrate-1m-partitions --dry-run`` — scans a data root for 1m bar
  keys whose on-disk partitions include legacy ``year=YYYY/*.parquet``
  files (no ``month=MM`` subdir) and prints them as a JSON plan.
  **Dry-run-only.** Running without ``--dry-run`` exits non-zero;
  the actual file moves are deliberately not implemented in this
  phase — the read path already unions both layouts via
  :func:`liq.store.key_layout.partition_files`, so the migration is
  a future operator concern.

Argparse keeps the dependency footprint small. If the CLI grows
multiple subcommands the next step would be ``typer`` — for now
this is fine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from liq.store.key_layout import is_1m_bar_key


def _find_legacy_keys(root: Path) -> list[str]:
    """Return 1m bar keys whose on-disk layout is year-only."""
    if not root.exists():
        raise FileNotFoundError(f"data root {root} does not exist")

    legacy: set[str] = set()
    for parquet_file in root.rglob("*.parquet"):
        parent = parquet_file.parent
        if parent.name.startswith("month="):
            continue
        if not parent.name.startswith("year="):
            continue
        # Year-only directory — walk up to the key path.
        # Convention: {root}/{key_segments}/year=YYYY/*.parquet
        key_dir = parent.parent
        if not key_dir.is_relative_to(root):
            continue
        key_parts = key_dir.relative_to(root).parts
        if "bars" not in key_parts:
            continue
        key = "/".join(key_parts)
        if not is_1m_bar_key(key):
            continue
        legacy.add(key)
    return sorted(legacy)


def _cmd_migrate_1m_partitions(args: argparse.Namespace) -> int:
    if not args.dry_run:
        print(
            "ERROR: real migration not implemented; pass --dry-run to preview.",
            file=sys.stderr,
        )
        return 2

    root = Path(args.root)
    try:
        legacy_keys = _find_legacy_keys(root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = {
        "dry_run": True,
        "root": str(root),
        "legacy_keys": legacy_keys,
        "count": len(legacy_keys),
    }
    print(json.dumps(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="liq-store",
        description="Maintenance commands for liq-store data roots.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    migrate = sub.add_parser(
        "migrate-1m-partitions",
        help="Preview migration of legacy year-only 1m partitions.",
    )
    migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only; required (real migration not implemented).",
    )
    migrate.add_argument("--root", required=True, help="Data root containing bar keys.")
    migrate.set_defaults(func=_cmd_migrate_1m_partitions)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


__all__ = ["build_parser", "main"]
