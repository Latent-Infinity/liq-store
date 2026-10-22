from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

from liq.store.bounded_parquet import PinnedReadRefused, _PinnedReadBudget
from liq.store.exceptions import DataCorruptionError

_PINNED_CLEANUP_ERRORS = object()


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


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _pinned_bytes(
    root: Path,
    identity: PinnedParquetFile,
    maximum: int,
    *,
    budget: _PinnedReadBudget | None = None,
) -> bytes:
    with _pinned_source(root, identity, maximum, budget=budget) as source:
        if (
            budget is not None
            and source.size > budget.limits.max_total_read_bytes - budget.total_read_bytes
        ):
            raise PinnedReadRefused("resource_limit")
        chunks: list[bytes] = []
        remaining = source.size
        while remaining:
            request_size = (
                budget.before_read(remaining) if budget is not None else min(remaining, 1024 * 1024)
            )
            chunk = os.read(source.borrow_fd(), request_size)
            if budget is not None:
                budget.record_read(len(chunk))
            if not chunk:
                raise TimestampMetadataRefused("file truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if budget is None and os.read(source.borrow_fd(), 1):
            raise TimestampMetadataRefused("file grew")
    payload = b"".join(chunks)
    if budget is not None:
        budget.check_deadline()
    digest = hashlib.sha256(payload).hexdigest()
    if budget is not None:
        budget.check_deadline()
    if digest != identity.sha256:
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


_PinnedSourceIdentity = tuple[int, int, int, int, int]


class _PinnedSource:
    __slots__ = ("_fd", "_parent_fd", "_leaf_name", "_identity", "_budget", "_closed")

    def __init__(
        self,
        fd: int,
        parent_fd: int,
        leaf_name: str,
        identity: _PinnedSourceIdentity,
        budget: _PinnedReadBudget | None,
    ) -> None:
        self._fd = fd
        self._parent_fd = parent_fd
        self._leaf_name = leaf_name
        self._identity = identity
        self._budget = budget
        self._closed = False

    @property
    def size(self) -> int:
        return self._identity[2]

    @property
    def identity(self) -> _PinnedSourceIdentity:
        return self._identity

    @property
    def budget(self) -> _PinnedReadBudget | None:
        return self._budget

    def borrow_fd(self) -> int:
        if self._closed:
            raise PinnedReadRefused("invalid_contract")
        return self._fd


def _validate_pinned_source(
    root: Path,
    identity: PinnedParquetFile,
    maximum: int,
    budget: _PinnedReadBudget | None,
) -> tuple[str, ...]:
    if budget is None:
        return root.parts[1:] + _parts(identity.path)
    if (
        type(budget) is not _PinnedReadBudget
        or not isinstance(root, Path)
        or not root.is_absolute()
        or ".." in root.parts
        or not isinstance(identity, PinnedParquetFile)
        or type(identity.path) is not str
        or type(identity.sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", identity.sha256) is None
        or type(maximum) is not int
        or maximum <= 0
    ):
        raise PinnedReadRefused("invalid_contract")
    try:
        leaf_parts = _parts(identity.path)
    except TimestampMetadataRefused as exc:
        raise PinnedReadRefused("invalid_contract") from exc
    budget.limits.__post_init__()
    return root.parts[1:] + leaf_parts


def _close_pinned_source(fd: int) -> BaseException | None:
    try:
        os.close(fd)
    except BaseException as exc:
        return exc
    return None


def _retain_pinned_cleanup_error(
    primary: BaseException, cleanup_errors: list[BaseException]
) -> None:
    if not cleanup_errors:
        return
    state = BaseException.__dict__["__dict__"].__get__(primary, BaseException)
    previously_retained = _PINNED_CLEANUP_ERRORS in state
    prior = state.get(_PINNED_CLEANUP_ERRORS, ())
    state[_PINNED_CLEANUP_ERRORS] = prior + tuple(cleanup_errors)
    if previously_retained:
        return

    participants = (primary, *cleanup_errors)
    if len({id(error) for error in participants}) != len(participants):
        return
    cause = BaseException.__dict__["__cause__"]
    context = BaseException.__dict__["__context__"]
    if any(
        cause.__get__(error, BaseException) is not None
        or context.__get__(error, BaseException) is not None
        for error in participants
    ):
        return
    for earlier, later in zip(participants[:-1], participants[1:], strict=True):
        cause.__set__(earlier, later)


@contextmanager
def _pinned_source(
    root: Path,
    identity: PinnedParquetFile,
    maximum: int,
    *,
    budget: _PinnedReadBudget | None,
) -> Iterator[_PinnedSource]:
    parts = _validate_pinned_source(root, identity, maximum, budget)
    parent: int | None = None
    fd: int | None = None
    source: _PinnedSource | None = None
    primary: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        if budget is not None:
            budget.before_open()
        parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if budget is not None:
            budget.check_deadline()
        for part in parts[:-1]:
            if budget is not None:
                budget.before_open()
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            previous = parent
            parent = child
            close_error = _close_pinned_source(previous)
            if close_error is not None:
                raise close_error
            if budget is not None:
                budget.check_deadline()
        if budget is not None:
            budget.before_open()
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        if budget is not None:
            budget.check_deadline()
        before = os.fstat(fd)
        if budget is not None:
            budget.check_deadline()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TimestampMetadataRefused("file must be regular singly linked")
        if budget is not None and before.st_size > min(
            maximum, budget.limits.max_source_size_bytes
        ):
            raise PinnedReadRefused("resource_limit")
        if not 0 < before.st_size <= maximum:
            raise TimestampMetadataRefused("file size budget")
        source = _PinnedSource(fd, parent, parts[-1], _stat_identity(before), budget)
        yield source
        if budget is not None:
            budget.check_deadline()
        after = os.fstat(fd)
        if budget is not None:
            budget.check_deadline()
        located = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        if budget is not None:
            budget.check_deadline()
        if source.identity != _stat_identity(after) or source.identity != _stat_identity(located):
            raise TimestampMetadataRefused("file identity changed")
    except BaseException as exc:
        primary = exc
    finally:
        if source is not None:
            source._closed = True
        if fd is not None:
            close_error = _close_pinned_source(fd)
            if close_error is not None:
                if primary is None:
                    primary = close_error
                else:
                    cleanup_errors.append(close_error)
        if parent is not None:
            close_error = _close_pinned_source(parent)
            if close_error is not None:
                if primary is None:
                    primary = close_error
                else:
                    cleanup_errors.append(close_error)
    if primary is not None:
        _retain_pinned_cleanup_error(primary, cleanup_errors)
        raise primary


def _read_pinned_range(source: _PinnedSource, *, offset: int, length: int) -> bytes:
    if (
        type(source) is not _PinnedSource
        or source._closed
        or type(source.budget) is not _PinnedReadBudget
        or type(offset) is not int
        or type(length) is not int
        or offset < 0
        or length <= 0
        or offset + length > source.size
    ):
        raise PinnedReadRefused("invalid_contract")
    budget = source.budget
    if length > budget.limits.max_total_read_bytes - budget.total_read_bytes:
        raise PinnedReadRefused("resource_limit")
    if budget.before_read(length) != length:
        raise PinnedReadRefused("resource_limit")
    payload = os.pread(source.borrow_fd(), length, offset)
    budget.record_read(len(payload))
    if len(payload) != length:
        raise TimestampMetadataRefused("incomplete bounded range")
    return payload
