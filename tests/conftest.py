"""Pytest configuration and fixtures for liq-store tests."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def sample_timestamp() -> datetime:
    """Provide a sample timezone-aware timestamp for tests."""
    return datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)


@pytest.fixture
def sample_ohlcv_df(sample_timestamp: datetime) -> pl.DataFrame:
    """Provide a sample OHLCV DataFrame for tests."""
    return pl.DataFrame({
        "timestamp": [
            sample_timestamp,
            sample_timestamp.replace(minute=31),
            sample_timestamp.replace(minute=32),
        ],
        "symbol": ["EUR_USD", "EUR_USD", "EUR_USD"],
        "open": [1.1000, 1.1005, 1.1010],
        "high": [1.1010, 1.1015, 1.1020],
        "low": [1.0995, 1.1000, 1.1005],
        "close": [1.1005, 1.1010, 1.1015],
        "volume": [1000.0, 1500.0, 1200.0],
    })


@pytest.fixture
def temp_storage_path(tmp_path: Path) -> Path:
    """Provide a temporary directory for storage tests."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    return storage_dir
