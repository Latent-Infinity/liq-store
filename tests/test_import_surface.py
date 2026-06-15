"""Tests for package import side effects."""

import subprocess
import sys


def test_protocol_import_does_not_load_duckdb() -> None:
    """Lightweight store imports should not initialize the DuckDB backend."""
    import liq.store
    from liq.store.protocols import TimeSeriesStore

    assert TimeSeriesStore.__name__ == "TimeSeriesStore"
    assert liq.store.TimeSeriesStore is TimeSeriesStore

    code = "\n".join(
        [
            "import sys",
            "import liq.store",
            "from liq.store.protocols import TimeSeriesStore",
            "assert TimeSeriesStore.__name__ == 'TimeSeriesStore'",
            "assert 'duckdb' not in sys.modules",
            "assert '_duckdb' not in sys.modules",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)
