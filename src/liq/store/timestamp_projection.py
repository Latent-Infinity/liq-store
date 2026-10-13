import re
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from liq.store.timestamp_metadata import (
    PinnedParquetFile,
    TimestampFileMetadata,
    TimestampMetadataRefused,
    _inspect_file,
    _pinned_bytes,
)


@dataclass(frozen=True, slots=True)
class TimestampProjectionLimits:
    max_file_bytes: int
    max_rows: int
    max_row_groups: int
    batch_rows: int

    def __post_init__(self) -> None:
        values = (self.max_file_bytes, self.max_rows, self.max_row_groups, self.batch_rows)
        if any(type(value) is not int or value <= 0 for value in values):
            raise TimestampMetadataRefused("projection budgets must be positive integers")


@dataclass(frozen=True, slots=True)
class TimestampProjectionRefused(TimestampMetadataRefused):
    identity: PinnedParquetFile
    group: int | None
    row_offset: int
    reason: str
    timestamp_ns: int | None = None

    def __str__(self) -> str:
        return (
            f"{self.identity.path} group={self.group} row={self.row_offset} "
            f"timestamp_ns={self.timestamp_ns}: {self.reason}"
        )


@dataclass(frozen=True, slots=True)
class PinnedTimestampBatch:
    identity: PinnedParquetFile
    group: int
    row_offset: int
    timestamps_ns: tuple[int, ...]


def iter_pinned_timestamps(
    root: Path,
    expected: TimestampFileMetadata,
    limits: TimestampProjectionLimits,
) -> Generator[PinnedTimestampBatch, None, None]:
    """Decode only after caller authorization; exhaust to verify all footer declarations."""
    identity = expected.identity
    group_index: int | None = None
    offset = 0

    def refuse(reason: str) -> TimestampProjectionRefused:
        return TimestampProjectionRefused(identity, group_index, offset, reason)

    if not root.is_absolute() or ".." in root.parts:
        raise refuse("root path must be absolute")
    if re.fullmatch(r"[0-9a-f]{64}", identity.sha256) is None:
        raise refuse("invalid digest")
    if not 0 < expected.row_count <= limits.max_rows:
        raise refuse("declared row budget")
    if not 0 < len(expected.row_groups) <= limits.max_row_groups:
        raise refuse("declared row-group budget")
    try:
        payload = _pinned_bytes(root, identity, limits.max_file_bytes)
        actual = _inspect_file(payload, identity, limits.max_row_groups)
    except (TimestampMetadataRefused, OSError) as exc:
        raise refuse(str(exc)) from exc
    if actual != expected:
        raise refuse("footer differs from admitted declaration")

    try:
        with pq.ParquetFile(pa.BufferReader(payload)) as source:
            expected_type = source.schema_arrow.field("timestamp").type
            scale = {"ms": 1_000_000, "us": 1_000, "ns": 1}[expected_type.unit]
            total_rows = 0
            for group in actual.row_groups:
                group_index = group.index
                offset = 0
                minimum: int | None = None
                maximum: int | None = None
                for batch in source.iter_batches(
                    batch_size=limits.batch_rows,
                    row_groups=[group.index],
                    columns=["timestamp"],
                    use_threads=False,
                    use_pandas_metadata=False,
                ):
                    if batch.schema.names != ["timestamp"] or batch.column(0).type != expected_type:
                        raise refuse("projected timestamp schema mismatch")
                    if not 0 < batch.num_rows <= limits.batch_rows:
                        raise refuse("physical batch row budget")
                    if (
                        offset + batch.num_rows > group.row_count
                        or total_rows + batch.num_rows > limits.max_rows
                    ):
                        raise refuse("physical row budget")
                    column = batch.column(0)
                    if column.null_count:
                        raise refuse("null physical timestamp")
                    values = tuple(value * scale for value in column.cast(pa.int64()).to_pylist())
                    low, high = min(values), max(values)
                    if low < group.minimum_ns or high > group.maximum_ns:
                        for index, value in enumerate(values):
                            if not group.minimum_ns <= value <= group.maximum_ns:
                                raise TimestampProjectionRefused(
                                    identity,
                                    group.index,
                                    offset + index,
                                    "physical timestamp outside declared bounds",
                                    value,
                                )
                    minimum = low if minimum is None else min(minimum, low)
                    maximum = high if maximum is None else max(maximum, high)
                    yield PinnedTimestampBatch(identity, group.index, offset, values)
                    offset += len(values)
                    total_rows += len(values)
                if (offset, minimum, maximum) != (
                    group.row_count,
                    group.minimum_ns,
                    group.maximum_ns,
                ):
                    raise refuse("physical row-group declaration mismatch")
            if total_rows != actual.row_count:
                raise refuse("physical file declaration mismatch")
    except (pa.ArrowInvalid, pa.ArrowIOError, NotImplementedError) as exc:
        raise refuse("unreadable timestamp pages") from exc
