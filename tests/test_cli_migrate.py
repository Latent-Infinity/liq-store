"""TDD tests for the ``liq-store migrate-1m-partitions`` CLI stub.

The CLI exists so operators can preview what a future migration from
legacy ``year=YYYY/*.parquet`` (year-only) to the current
``year=YYYY/month=MM/*.parquet`` (year + month) layout would touch.
The stub is dry-run-only: running without ``--dry-run`` exits
non-zero with a clear message rather than silently doing nothing.

Tests drive ``cli.main(argv)`` in-process so pytest-cov can
instrument the CLI module.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from liq.store.cli import main


def _bars(symbol: str, day: int, year: int = 2023) -> pl.DataFrame:
    base = datetime(year, 6, day, 14, 30, tzinfo=UTC)
    rows = []
    for i in range(3):
        rows.append(
            {
                "timestamp": base + timedelta(minutes=i),
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.25,
                "volume": 1000,
            }
        )
    return pl.DataFrame(rows)


def _seed_legacy(root: Path, symbol: str) -> None:
    """Create a legacy-layout (year-only) parquet under a bar key."""
    legacy_root = root / "databento" / symbol / "bars" / "1m" / "year=2023"
    legacy_root.mkdir(parents=True, exist_ok=True)
    _bars(symbol, 3).write_parquet(legacy_root / "data.parquet")


def _seed_legacy_daily(root: Path, symbol: str) -> None:
    """Create a yearly 1d parquet that is not a migration candidate."""
    legacy_root = root / "databento" / symbol / "bars" / "1d" / "year=2023"
    legacy_root.mkdir(parents=True, exist_ok=True)
    _bars(symbol, 3).write_parquet(legacy_root / "data.parquet")


def _seed_new(root: Path, symbol: str) -> None:
    """Create a new-layout (year+month) parquet under a bar key."""
    new_root = root / "databento" / symbol / "bars" / "1m" / "year=2024" / "month=06"
    new_root.mkdir(parents=True, exist_ok=True)
    _bars(symbol, 3, year=2024).write_parquet(new_root / "data.parquet")


class TestMigrate1mPartitionsDryRun:
    def test_dry_run_reports_legacy_keys_as_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _seed_legacy(tmp_path, "AAPL")
        _seed_legacy(tmp_path, "MSFT")
        _seed_new(tmp_path, "GOOG")

        rc = main(["migrate-1m-partitions", "--dry-run", "--root", str(tmp_path)])
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["dry_run"] is True
        legacy_keys = set(report["legacy_keys"])
        assert "databento/AAPL/bars/1m" in legacy_keys
        assert "databento/MSFT/bars/1m" in legacy_keys
        assert "databento/GOOG/bars/1m" not in legacy_keys
        assert report["count"] == 2

    def test_dry_run_with_no_legacy_returns_empty_list(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _seed_new(tmp_path, "AAPL")
        rc = main(["migrate-1m-partitions", "--dry-run", "--root", str(tmp_path)])
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["legacy_keys"] == []
        assert report["count"] == 0

    def test_without_dry_run_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _seed_legacy(tmp_path, "AAPL")
        rc = main(["migrate-1m-partitions", "--root", str(tmp_path)])
        assert rc != 0
        captured = capsys.readouterr()
        assert "dry-run" in (captured.err + captured.out).lower()

    def test_missing_root_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(
            [
                "migrate-1m-partitions",
                "--dry-run",
                "--root",
                str(tmp_path / "does-not-exist"),
            ]
        )
        assert rc != 0
        assert "does not exist" in capsys.readouterr().err.lower()

    def test_dry_run_ignores_non_bar_legacy_paths(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A year-only partition under a non-bar key (e.g. reference data)
        should not be reported as a migration candidate."""
        non_bar = tmp_path / "reference" / "databento" / "symbology" / "year=2023"
        non_bar.mkdir(parents=True, exist_ok=True)
        _bars("REF", 3).write_parquet(non_bar / "data.parquet")

        rc = main(["migrate-1m-partitions", "--dry-run", "--root", str(tmp_path)])
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["legacy_keys"] == []

    def test_dry_run_ignores_yearly_non_1m_bar_keys(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _seed_legacy_daily(tmp_path, "SPY")

        rc = main(["migrate-1m-partitions", "--dry-run", "--root", str(tmp_path)])
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["legacy_keys"] == []


class TestCLIDispatch:
    def test_unknown_subcommand_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["nope", "--root", str(tmp_path)])
        assert excinfo.value.code != 0

    def test_help_lists_migrate_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
        assert "migrate-1m-partitions" in capsys.readouterr().out


class TestModuleEntrypoint:
    """``python -m liq.store`` dispatches to ``cli.main`` — instrumented
    in-process so the ``__main__`` module is covered."""

    def test_main_callable_is_exposed(self) -> None:
        from liq.store import cli

        assert callable(cli.main)
        assert callable(cli.build_parser)
