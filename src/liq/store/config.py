"""Configuration helpers for ParquetStore built from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from liq.store.parquet import ParquetStore, ParquetStoreConfig


def load_parquet_config_from_env(env: Mapping[str, str] | None = None) -> ParquetStoreConfig:
    """Create ParquetStoreConfig from environment variables.

    Supported variables (all optional):
    - LIQ_STORAGE_TARGET_ROWS
    - LIQ_STORAGE_COMPRESSION
    - LIQ_STORAGE_COMPRESSION_LEVEL
    - LIQ_STORAGE_LOCK_TIMEOUT

    Args:
        env: Optional mapping to read from (defaults to os.environ).

    Returns:
        ParquetStoreConfig with values from env or defaults.

    Raises:
        ValueError: If numeric env values cannot be parsed.
    """
    env = env or os.environ

    def _int(var: str, default: int) -> int:
        val = env.get(var)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError as exc:
            raise ValueError(f"{var} must be an integer") from exc

    target_rows = _int("LIQ_STORAGE_TARGET_ROWS", ParquetStoreConfig.target_rows_per_file)
    compression = env.get("LIQ_STORAGE_COMPRESSION", ParquetStoreConfig.compression)
    compression_level = _int(
        "LIQ_STORAGE_COMPRESSION_LEVEL", ParquetStoreConfig.compression_level
    )
    lock_timeout = _int(
        "LIQ_STORAGE_LOCK_TIMEOUT", ParquetStoreConfig.lock_timeout_seconds
    )

    return ParquetStoreConfig(
        target_rows_per_file=target_rows,
        compression=compression,  # type: ignore[arg-type]
        compression_level=compression_level,
        lock_timeout_seconds=lock_timeout,
    )


def create_parquet_store_from_env(env: Mapping[str, str] | None = None) -> ParquetStore:
    """Create a ParquetStore using env for data root and config.

    Environment variables:
    - LIQ_DATA_ROOT (defaults to "./data")
    - See load_parquet_config_from_env for additional knobs
    """
    env = env or os.environ
    data_root = env.get("LIQ_DATA_ROOT", "./data")
    config = load_parquet_config_from_env(env)
    return ParquetStore(str(Path(data_root).expanduser()), config=config)
