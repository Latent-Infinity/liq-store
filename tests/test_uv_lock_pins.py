"""Pin assertions for load-bearing runtime dependencies.

The :func:`ParquetStore.read_multi` path delegates to DuckDB; an
accidental downgrade or removal would silently change predicate
pushdown behavior and the union-vs-mismatch semantics that
``read_multi`` relies on. Reading ``uv.lock`` directly catches a
mis-edit before it lands as a runtime surprise.
"""

from __future__ import annotations

import re
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_locked_versions(lockfile: Path) -> dict[str, str]:
    """Return ``{package_name: version}`` from a ``uv.lock`` file.

    The format we care about is the ``[[package]]`` array of tables:

        [[package]]
        name = "duckdb"
        version = "1.3.0"

    A tiny regex is cheaper than pulling tomli for one lookup.
    """
    text = lockfile.read_text(encoding="utf-8")
    versions: dict[str, str] = {}
    for match in re.finditer(
        r'\[\[package\]\]\s+name\s*=\s*"([^"]+)"\s+version\s*=\s*"([^"]+)"',
        text,
    ):
        versions[match.group(1)] = match.group(2)
    return versions


def _version_tuple(version: str) -> tuple[int, ...]:
    """Convert ``1.3.0`` → ``(1, 3, 0)`` for ordered comparison."""
    parts: list[int] = []
    for piece in version.split("."):
        m = re.match(r"(\d+)", piece)
        if not m:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


class TestUvLockPins:
    def test_duckdb_pinned_to_at_least_1_0(self) -> None:
        lock = _project_root() / "uv.lock"
        assert lock.exists(), "uv.lock missing from project root"
        versions = _parse_locked_versions(lock)
        assert "duckdb" in versions, "duckdb is not pinned in uv.lock"
        assert _version_tuple(versions["duckdb"]) >= (1, 0), (
            f"duckdb {versions['duckdb']} is below the >=1.0 pin"
        )

    def test_polars_pinned_to_at_least_1_20(self) -> None:
        versions = _parse_locked_versions(_project_root() / "uv.lock")
        assert "polars" in versions
        assert _version_tuple(versions["polars"]) >= (1, 20), (
            f"polars {versions['polars']} is below the >=1.20 pin"
        )
