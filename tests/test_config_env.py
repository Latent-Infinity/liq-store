"""Tests for environment-based configuration helpers."""

import os
from pathlib import Path

import pytest

from liq.store.config import create_parquet_store_from_env, load_parquet_config_from_env
from liq.store.parquet import ParquetStoreConfig


def test_load_parquet_config_defaults() -> None:
    cfg = load_parquet_config_from_env(env={})
    assert cfg.target_rows_per_file == ParquetStoreConfig.target_rows_per_file
    assert cfg.compression == ParquetStoreConfig.compression
    assert cfg.compression_level == ParquetStoreConfig.compression_level
    assert cfg.lock_timeout_seconds == ParquetStoreConfig.lock_timeout_seconds


def test_load_parquet_config_overrides() -> None:
    cfg = load_parquet_config_from_env(
        env={
            "LIQ_STORAGE_TARGET_ROWS": "1000",
            "LIQ_STORAGE_COMPRESSION": "snappy",
            "LIQ_STORAGE_COMPRESSION_LEVEL": "9",
            "LIQ_STORAGE_LOCK_TIMEOUT": "15",
        }
    )
    assert cfg.target_rows_per_file == 1000
    assert cfg.compression == "snappy"
    assert cfg.compression_level == 9
    assert cfg.lock_timeout_seconds == 15


def test_load_parquet_config_invalid_int() -> None:
    with pytest.raises(ValueError):
        load_parquet_config_from_env(env={"LIQ_STORAGE_TARGET_ROWS": "not_an_int"})


def test_create_parquet_store_from_env(tmp_path) -> None:  # type: ignore[annotation-unchecked]
    env = {
        "DATA_ROOT": str(tmp_path),
        "LIQ_STORAGE_TARGET_ROWS": "123",
    }
    store = create_parquet_store_from_env(env=env)
    assert store.data_root == Path(tmp_path).resolve()
    assert store.config.target_rows_per_file == 123


def test_create_parquet_store_uses_default_root(tmp_path, monkeypatch):  # type: ignore[annotation-unchecked]
    monkeypatch.chdir(tmp_path)
    store = create_parquet_store_from_env(env={})
    assert store.data_root == Path("./data").resolve()


def test_create_parquet_store_rejects_empty_root() -> None:
    with pytest.raises(ValueError, match="DATA_ROOT must be set"):
        create_parquet_store_from_env(env={"DATA_ROOT": " "})
