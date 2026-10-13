from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

from liq.store.exceptions import DataCorruptionError


class TimestampMetadataRefused(DataCorruptionError):
    pass


@dataclass(frozen=True, slots=True)
class PinnedParquetFile:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class TimestampRowGroupMetadata:
    index: int
    row_count: int
    minimum_ns: int
    maximum_ns: int


@dataclass(frozen=True, slots=True)
class TimestampFileMetadata:
    identity: PinnedParquetFile
    row_count: int
    row_groups: tuple[TimestampRowGroupMetadata, ...]


def _parts(value: str) -> tuple[str, ...]:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or "\\" in value
        or not value.endswith(".parquet")
    ):
        raise TimestampMetadataRefused("noncanonical parquet path")
    return path.parts


def _stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _pinned_bytes(root: Path, identity: PinnedParquetFile, maximum: int) -> bytes:
    parts = root.parts[1:] + _parts(identity.path)
    parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = next_fd
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise TimestampMetadataRefused("file must be regular and singly linked")
            if not 0 < before.st_size <= maximum:
                raise TimestampMetadataRefused("file size budget")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(fd, min(remaining, 1024 * 1024))
                if not chunk:
                    raise TimestampMetadataRefused("file truncated")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise TimestampMetadataRefused("file grew")
            after = os.fstat(fd)
            located = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            if _stat_identity(before) != _stat_identity(after) or _stat_identity(
                before
            ) != _stat_identity(located):
                raise TimestampMetadataRefused("file identity changed")
        finally:
            os.close(fd)
    finally:
        os.close(parent)
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != identity.sha256:
        raise TimestampMetadataRefused("file digest mismatch")
    return payload


def _inspect_file(
    payload: bytes, identity: PinnedParquetFile, remaining_groups: int
) -> TimestampFileMetadata:
    try:
        metadata = pq.read_metadata(io.BytesIO(payload))
    except (pa.ArrowInvalid, pa.ArrowIOError, NotImplementedError) as exc:
        raise TimestampMetadataRefused("unreadable timestamp footer") from exc
    if metadata.num_rows <= 0 or metadata.num_row_groups <= 0:
        raise TimestampMetadataRefused("empty file metadata is unverified")
    if metadata.num_row_groups > remaining_groups:
        raise TimestampMetadataRefused("row-group budget")
    schema = metadata.schema.to_arrow_schema()
    if schema.names.count("timestamp") != 1 or metadata.schema.names.count("timestamp") != 1:
        raise TimestampMetadataRefused("missing or ambiguous timestamp schema")
    field = schema.field("timestamp")
    if not pa.types.is_timestamp(field.type) or field.type.tz != "UTC":
        raise TimestampMetadataRefused("timestamp schema must declare UTC")
    index = metadata.schema.names.index("timestamp")
    column_schema = metadata.schema.column(index)
    logical = json.loads(column_schema.logical_type.to_json())
    units = {
        "milliseconds": ("ms", 1_000_000),
        "microseconds": ("us", 1_000),
        "nanoseconds": ("ns", 1),
    }
    unit = units.get(logical.get("timeUnit"))
    if (
        column_schema.path != "timestamp"
        or column_schema.physical_type != "INT64"
        or logical.get("Type") != "Timestamp"
        or logical.get("isAdjustedToUTC") is not True
        or unit is None
        or field.type.unit != unit[0]
    ):
        raise TimestampMetadataRefused("unsupported timestamp encoding")
    groups: list[TimestampRowGroupMetadata] = []
    for group_index in range(metadata.num_row_groups):
        group = metadata.row_group(group_index)
        if group.num_rows <= 0:
            raise TimestampMetadataRefused("empty row group metadata is unverified")
        column = group.column(index)
        if column.path_in_schema != "timestamp" or column.file_path:
            raise TimestampMetadataRefused("external or ambiguous timestamp column")
        stats = column.statistics
        if stats is None or not stats.has_min_max or not stats.has_null_count:
            raise TimestampMetadataRefused("missing timestamp statistics")
        if (
            stats.null_count != 0
            or stats.num_values != group.num_rows
            or column.num_values != group.num_rows
        ):
            raise TimestampMetadataRefused("timestamp row count or null inconsistency")
        minimum, maximum = stats.min_raw, stats.max_raw
        if (
            type(minimum) is not int
            or type(maximum) is not int
            or minimum > maximum
            or stats.physical_type != "INT64"
        ):
            raise TimestampMetadataRefused("invalid timestamp extrema")
        groups.append(
            TimestampRowGroupMetadata(
                group_index, group.num_rows, minimum * unit[1], maximum * unit[1]
            )
        )
    if sum(group.row_count for group in groups) != metadata.num_rows:
        raise TimestampMetadataRefused("file row count inconsistency")
    return TimestampFileMetadata(identity, metadata.num_rows, tuple(groups))


def inspect_timestamp_metadata(
    root: Path,
    files: tuple[PinnedParquetFile, ...],
    *,
    max_files: int,
    max_file_bytes: int,
    max_row_groups: int,
) -> tuple[TimestampFileMetadata, ...]:
    """Inspect pinned footer declarations only; callers must obtain guarded source admission."""
    if any(
        type(value) is not int or value <= 0
        for value in (max_files, max_file_bytes, max_row_groups)
    ):
        raise TimestampMetadataRefused("metadata budgets must be positive integers")
    if not root.is_absolute() or ".." in root.parts:
        raise TimestampMetadataRefused("root path must be absolute")
    if not files or len(files) > max_files or len({file.path for file in files}) != len(files):
        raise TimestampMetadataRefused("file budget or duplicate inventory")
    for identity in files:
        _parts(identity.path)
        if re.fullmatch(r"[0-9a-f]{64}", identity.sha256) is None:
            raise TimestampMetadataRefused("invalid digest")
    result: list[TimestampFileMetadata] = []
    remaining = max_row_groups
    for identity in files:
        payload = _pinned_bytes(root, identity, max_file_bytes)
        item = _inspect_file(payload, identity, remaining)
        result.append(item)
        remaining -= len(item.row_groups)
    return tuple(result)
