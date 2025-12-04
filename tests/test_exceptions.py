"""Tests for liq.store.exceptions module."""

import pytest

from liq.store.exceptions import DataCorruptionError, DataNotFoundError, StorageError


class TestStorageError:
    """Tests for StorageError base exception."""

    def test_storage_error_is_exception(self) -> None:
        error = StorageError("test error")
        assert isinstance(error, Exception)

    def test_storage_error_message(self) -> None:
        error = StorageError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_can_catch_storage_error(self) -> None:
        with pytest.raises(StorageError):
            raise StorageError("test")


class TestDataNotFoundError:
    """Tests for DataNotFoundError exception."""

    def test_inherits_from_storage_error(self) -> None:
        error = DataNotFoundError("not found")
        assert isinstance(error, StorageError)

    def test_can_catch_as_storage_error(self) -> None:
        with pytest.raises(StorageError):
            raise DataNotFoundError("symbol not found")

    def test_can_catch_specifically(self) -> None:
        with pytest.raises(DataNotFoundError):
            raise DataNotFoundError("symbol not found")


class TestDataCorruptionError:
    """Tests for DataCorruptionError exception."""

    def test_inherits_from_storage_error(self) -> None:
        error = DataCorruptionError("corrupted")
        assert isinstance(error, StorageError)

    def test_can_catch_as_storage_error(self) -> None:
        with pytest.raises(StorageError):
            raise DataCorruptionError("data corrupted")

    def test_can_catch_specifically(self) -> None:
        with pytest.raises(DataCorruptionError):
            raise DataCorruptionError("data corrupted")
