"""Storage exception hierarchy for the LIQ Stack.

This module defines the exception classes used throughout the storage layer.
All storage-specific exceptions inherit from StorageError for consistent handling.
"""


class StorageError(Exception):
    """Base exception for storage-related errors.

    All storage-specific exceptions should inherit from this class
    to allow for consistent error handling.
    """

    pass


class DataNotFoundError(StorageError):
    """Exception raised when requested data is not found.

    This occurs when trying to read data for a symbol/date range
    that doesn't exist in storage.
    """

    pass


class DataCorruptionError(StorageError):
    """Exception raised when stored data is corrupted or invalid.

    This includes situations where data cannot be read, parsed,
    or validated successfully.
    """

    pass
