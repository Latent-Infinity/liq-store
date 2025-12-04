"""Tests for liq.store.naming module."""

from datetime import UTC, datetime

from liq.store.naming import (
    generate_filename,
    is_timestamp_filename,
    parse_filename,
)


class TestGenerateFilename:
    """Tests for generate_filename function."""

    def test_generates_correct_format(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 4, 23, 59, 59, tzinfo=UTC)
        result = generate_filename(start, end)
        assert result == "20240101T000000-20240104T235959.parquet"

    def test_includes_parquet_extension(self) -> None:
        start = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        end = datetime(2024, 6, 15, 14, 45, 30, tzinfo=UTC)
        result = generate_filename(start, end)
        assert result.endswith(".parquet")

    def test_same_start_end(self) -> None:
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = generate_filename(ts, ts)
        assert result == "20240101T120000-20240101T120000.parquet"


class TestParseFilename:
    """Tests for parse_filename function."""

    def test_parses_valid_filename(self) -> None:
        filename = "20240101T000000-20240104T235959.parquet"
        result = parse_filename(filename)
        assert result is not None
        start, end = result
        assert start == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert end == datetime(2024, 1, 4, 23, 59, 59, tzinfo=UTC)

    def test_parses_full_path(self) -> None:
        path = "/data/symbol=EUR_USD/year=2024/month=1/20240101T000000-20240104T235959.parquet"
        result = parse_filename(path)
        assert result is not None

    def test_returns_none_for_legacy_format(self) -> None:
        result = parse_filename("data.parquet")
        assert result is None

    def test_returns_none_for_invalid_format(self) -> None:
        result = parse_filename("invalid_filename.parquet")
        assert result is None

    def test_returns_none_for_invalid_dates(self) -> None:
        # Regex matches but strptime fails (invalid month 99)
        result = parse_filename("20249901T000000-20249901T235959.parquet")
        assert result is None

    def test_roundtrip(self) -> None:
        start = datetime(2024, 3, 15, 8, 30, 45, tzinfo=UTC)
        end = datetime(2024, 3, 20, 16, 45, 30, tzinfo=UTC)
        filename = generate_filename(start, end)
        result = parse_filename(filename)
        assert result is not None
        parsed_start, parsed_end = result
        assert parsed_start == start
        assert parsed_end == end


class TestIsTimestampFilename:
    """Tests for is_timestamp_filename function."""

    def test_returns_true_for_timestamp_format(self) -> None:
        assert is_timestamp_filename("20240101T000000-20240104T235959.parquet")

    def test_returns_false_for_legacy_format(self) -> None:
        assert not is_timestamp_filename("data.parquet")

    def test_returns_false_for_invalid_format(self) -> None:
        assert not is_timestamp_filename("invalid.parquet")

    def test_handles_full_path(self) -> None:
        path = "/data/year=2024/20240101T000000-20240104T235959.parquet"
        assert is_timestamp_filename(path)
