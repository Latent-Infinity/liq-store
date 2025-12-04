"""Timestamp-based file naming utilities for Parquet storage.

This module provides utilities for generating and parsing timestamp-based
Parquet filenames in ISO 8601 compact format.

Format: {YYYYMMDDTHHMMSS}-{YYYYMMDDTHHMMSS}.parquet

Example: 20240101T000000-20240104T235959.parquet

This enables sub-partition pruning by allowing queries to skip files
that don't overlap with the requested time range.
"""

import re
from datetime import UTC, datetime
from pathlib import Path

# ISO 8601 compact format: YYYYMMDDTHHMMSS
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"

# Regex pattern for parsing timestamp filenames
# Matches: 20240101T000000-20240104T235959.parquet
FILENAME_PATTERN = re.compile(r"(\d{8}T\d{6})-(\d{8}T\d{6})\.parquet$")


def generate_filename(start_ts: datetime, end_ts: datetime) -> str:
    """Generate timestamp-based filename from start and end datetimes.

    Args:
        start_ts: Start timestamp (timezone-aware)
        end_ts: End timestamp (timezone-aware)

    Returns:
        Filename string in format: {start}-{end}.parquet
        Example: 20240101T000000-20240104T235959.parquet
    """
    start_str = start_ts.strftime(TIMESTAMP_FORMAT)
    end_str = end_ts.strftime(TIMESTAMP_FORMAT)
    return f"{start_str}-{end_str}.parquet"


def parse_filename(filename: str) -> tuple[datetime, datetime] | None:
    """Parse start and end timestamps from a timestamp-based filename.

    Args:
        filename: Filename or full path to parse

    Returns:
        Tuple of (start_datetime, end_datetime) if valid timestamp filename,
        None if legacy format (data.parquet) or invalid format.
        Returned datetimes are timezone-aware (UTC).
    """
    # Extract just the filename from path if full path provided
    name = Path(filename).name

    match = FILENAME_PATTERN.match(name)
    if not match:
        return None

    try:
        start_ts = datetime.strptime(match.group(1), TIMESTAMP_FORMAT).replace(tzinfo=UTC)
        end_ts = datetime.strptime(match.group(2), TIMESTAMP_FORMAT).replace(tzinfo=UTC)
        return (start_ts, end_ts)
    except ValueError:
        return None


def is_timestamp_filename(filename: str) -> bool:
    """Check if filename is in timestamp format.

    Args:
        filename: Filename or full path to check

    Returns:
        True if timestamp format, False otherwise
    """
    return parse_filename(filename) is not None
