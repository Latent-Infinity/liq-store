"""Immutable contracts for pinned reads whose bounds must be enforceable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from dataclasses import field as _footer_record_field
from datetime import datetime, timedelta
from struct import unpack_from
from typing import TYPE_CHECKING, Literal

from liq.store.exceptions import DataCorruptionError, StorageError

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from liq.store.timestamp_metadata import PinnedParquetFile, TimestampFileMetadata


class PinnedReadRefused(StorageError):
    """A requested read contract is invalid or cannot currently be enforced."""

    def __init__(
        self,
        reason_code: Literal[
            "invalid_contract", "unsupported_mode", "resource_limit", "deadline_exceeded"
        ],
    ) -> None:
        if reason_code not in (
            "invalid_contract",
            "unsupported_mode",
            "resource_limit",
            "deadline_exceeded",
        ):
            reason_code = "invalid_contract"
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class ParquetProjection:
    """Unique scalar column names and an explicit half-open UTC interval."""

    columns: tuple[str, ...]
    start_inclusive: datetime
    end_exclusive: datetime

    def __post_init__(self) -> None:
        if (
            type(self.columns) is not tuple
            or not self.columns
            or any(type(column) is not str or not column for column in self.columns)
            or len(set(self.columns)) != len(self.columns)
            or "timestamp" not in self.columns
        ):
            raise PinnedReadRefused("invalid_contract")
        for bound in (self.start_inclusive, self.end_exclusive):
            if not isinstance(bound, datetime) or bound.utcoffset() != timedelta(0):
                raise PinnedReadRefused("invalid_contract")
        if self.start_inclusive >= self.end_exclusive:
            raise PinnedReadRefused("invalid_contract")


@dataclass(frozen=True, slots=True)
class PinnedReadLimits:
    """Required positive integer bounds in the caller's monotonic clock domain."""

    max_source_size_bytes: int
    max_total_read_bytes: int
    max_source_opens: int
    max_decoded_rows: int
    max_output_rows: int
    max_row_groups: int
    max_row_group_uncompressed_bytes: int
    max_total_uncompressed_bytes: int
    batch_rows: int
    deadline_ns: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise PinnedReadRefused("invalid_contract")


class _PinnedReadBudget:
    """Per-invocation mutable accounting alongside immutable read limits."""

    def __init__(self, limits: PinnedReadLimits, *, clock_ns: Callable[[], int]) -> None:
        if type(limits) is not PinnedReadLimits or not callable(clock_ns):
            raise PinnedReadRefused("invalid_contract")
        limits.__post_init__()
        self.limits = limits
        self.clock_ns = clock_ns
        self.source_opens = 0
        self.total_read_bytes = 0

    def before_open(self) -> None:
        self.check_deadline()
        if self.source_opens >= self.limits.max_source_opens:
            raise PinnedReadRefused("resource_limit")
        self.source_opens += 1

    def before_read(self, remaining_bytes: int) -> int:
        if type(remaining_bytes) is not int or remaining_bytes <= 0:
            raise PinnedReadRefused("invalid_contract")
        self.check_deadline()
        available = self.limits.max_total_read_bytes - self.total_read_bytes
        if available <= 0:
            raise PinnedReadRefused("resource_limit")
        return min(remaining_bytes, available, 1024 * 1024)

    def record_read(self, actual_bytes: int) -> None:
        if type(actual_bytes) is not int or actual_bytes < 0:
            raise PinnedReadRefused("invalid_contract")
        self.total_read_bytes += actual_bytes
        if self.total_read_bytes > self.limits.max_total_read_bytes:
            raise PinnedReadRefused("resource_limit")
        self.check_deadline()

    def check_deadline(self) -> None:
        now = self.clock_ns()
        if type(now) is not int or now < 0:
            raise PinnedReadRefused("invalid_contract")
        if now >= self.limits.deadline_ns:
            raise PinnedReadRefused("deadline_exceeded")


@dataclass(frozen=True, slots=True)
class _CompactWireLimits:
    """Caller-declared logical wire, structural and payload-copy allowances."""

    max_input_bytes: int
    max_object_bytes: int
    max_header_bytes: int
    max_depth: int
    max_fields_per_struct: int
    max_total_fields: int
    max_items_per_container: int
    max_total_items: int
    max_string_bytes: int
    max_total_string_bytes: int
    max_varint_bytes: int
    max_total_wire_bytes: int
    max_work_units: int
    max_copied_payload_bytes: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise PinnedReadRefused("invalid_contract")
        if self.max_varint_bytes > 10:
            raise PinnedReadRefused("invalid_contract")


class _CompactWireBudget:
    """Shared mutable accounting for cursors in one invocation."""

    def __init__(self, limits: _CompactWireLimits) -> None:
        if type(limits) is not _CompactWireLimits:
            raise PinnedReadRefused("invalid_contract")
        limits.__post_init__()
        self.limits = limits
        self.total_wire_bytes = 0
        self.work_units = 0
        self.total_fields = 0
        self.total_items = 0
        self.total_string_bytes = 0
        self.copied_payload_bytes = 0

    def _reserve(
        self,
        *,
        wire_bytes: int = 0,
        work_units: int = 0,
        field_count: int = 0,
        items: int = 0,
        string_bytes: int = 0,
        copied_bytes: int = 0,
    ) -> None:
        """Reserve all counters together before an internal bounded operation."""
        changes = (
            (self.total_wire_bytes, wire_bytes, self.limits.max_total_wire_bytes),
            (self.work_units, work_units, self.limits.max_work_units),
            (self.total_fields, field_count, self.limits.max_total_fields),
            (self.total_items, items, self.limits.max_total_items),
            (self.total_string_bytes, string_bytes, self.limits.max_total_string_bytes),
            (self.copied_payload_bytes, copied_bytes, self.limits.max_copied_payload_bytes),
        )
        if any(increment > maximum - current for current, increment, maximum in changes):
            raise PinnedReadRefused("resource_limit")
        self.total_wire_bytes += wire_bytes
        self.work_units += work_units
        self.total_fields += field_count
        self.total_items += items
        self.total_string_bytes += string_bytes
        self.copied_payload_bytes += copied_bytes


class _CompactWireMalformed(DataCorruptionError):
    """Wire refusal with a bounded message that contains no source values."""

    def __init__(self, reason_code: str) -> None:
        if reason_code not in (
            "truncated_wire",
            "noncanonical_wire",
            "invalid_wire",
            "schema_mismatch",
        ):
            reason_code = "invalid_wire"
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class _CompactWireField:
    """One field identity, logical wire type and optional embedded boolean."""

    field_id: int
    wire_type: str
    boolean: bool | None


class _CompactWireStruct:
    """Mutable field identity and termination state for one bounded frame."""

    def __init__(self) -> None:
        self.last_id = 0
        self.seen_ids: set[int] = set()
        self.stopped = False


def _compact_wire_type(tag: int) -> str:
    match tag:
        case 1 | 2:
            return "bool"
        case 3:
            return "byte"
        case 4:
            return "i16"
        case 5:
            return "i32"
        case 6:
            return "i64"
        case 7:
            return "double"
        case 8:
            return "binary"
        case 9:
            return "list"
        case 12:
            return "struct"
        case _:
            raise _CompactWireMalformed("invalid_wire")


def _validate_compact_wire_type(wire_type: str) -> None:
    if type(wire_type) is not str or wire_type not in (
        "bool",
        "byte",
        "i16",
        "i32",
        "i64",
        "double",
        "binary",
        "list",
        "struct",
    ):
        raise PinnedReadRefused("invalid_contract")


class _CompactWireCursor:
    """Mutable cursor and structural state over one retained immutable buffer."""

    def __init__(
        self,
        payload: bytes,
        start: int,
        stop: int,
        budget: _CompactWireBudget,
        *,
        kind: str,
    ) -> None:
        if (
            type(payload) is not bytes
            or type(start) is not int
            or type(stop) is not int
            or type(budget) is not _CompactWireBudget
            or type(kind) is not str
            or kind not in ("footer", "page_header")
            or not 0 <= start <= stop <= len(payload)
        ):
            raise PinnedReadRefused("invalid_contract")
        if len(payload) > budget.limits.max_input_bytes:
            raise PinnedReadRefused("resource_limit")
        self.payload = payload
        self.start = start
        self.position = start
        self.stop = stop
        self.budget = budget
        self.kind = kind
        self._structs: list[_CompactWireStruct] = []
        self._pending_bool: bool | None = None

    def _ensure_available(self, size: int) -> None:
        if size > self.stop - self.position:
            raise _CompactWireMalformed("truncated_wire")

    def _consume(self, size: int, *, binary: bool = False, copy_binary: bool = True) -> int:
        """Reserve an entire read before advancing or copying any payload bytes."""
        self._ensure_available(size)
        consumed = self.position - self.start
        limits = self.budget.limits
        if (
            size > limits.max_object_bytes - consumed
            or self.kind == "page_header"
            and size > limits.max_header_bytes - consumed
            or binary
            and size > limits.max_string_bytes
        ):
            raise PinnedReadRefused("resource_limit")
        self.budget._reserve(
            wire_bytes=size,
            work_units=size,
            string_bytes=size if binary else 0,
            copied_bytes=size if binary and copy_binary else 0,
        )
        previous = self.position
        self.position += size
        return previous

    def _unsigned_byte(self) -> int:
        return self.payload[self._consume(1)]

    def _varint(self, width: int) -> int:
        value = 0
        for index in range((width + 6) // 7):
            if index >= self.budget.limits.max_varint_bytes:
                raise PinnedReadRefused("resource_limit")
            byte = self._unsigned_byte()
            payload = byte & 0x7F
            remaining = width - index * 7
            if remaining < 7 and (payload >= 1 << remaining or byte & 0x80):
                raise _CompactWireMalformed("invalid_wire")
            value |= payload << (index * 7)
            if not byte & 0x80:
                if index and payload == 0:
                    raise _CompactWireMalformed("noncanonical_wire")
                return value
        raise _CompactWireMalformed("invalid_wire")

    def _signed_integer(self, width: int) -> int:
        value = self._varint(width)
        return (value >> 1) ^ -(value & 1)

    def begin_struct(self) -> None:
        if self._pending_bool is not None or self._structs and self._structs[-1].stopped:
            raise _CompactWireMalformed("invalid_wire")
        self.budget._reserve(work_units=1)
        if len(self._structs) >= self.budget.limits.max_depth:
            raise PinnedReadRefused("resource_limit")
        self._structs.append(_CompactWireStruct())

    def read_field(self, allowed_fields: dict[int, str]) -> _CompactWireField | None:
        if type(allowed_fields) is not dict:
            raise PinnedReadRefused("invalid_contract")
        if len(allowed_fields) > self.budget.limits.max_fields_per_struct:
            raise PinnedReadRefused("resource_limit")
        self.budget._reserve(work_units=len(allowed_fields))
        for field_id, wire_type in allowed_fields.items():
            if type(field_id) is not int or not 0 < field_id <= 32767:
                raise PinnedReadRefused("invalid_contract")
            _validate_compact_wire_type(wire_type)
        if not self._structs or self._pending_bool is not None or self._structs[-1].stopped:
            raise _CompactWireMalformed("invalid_wire")
        self.budget._reserve(work_units=1)
        header = self._unsigned_byte()
        tag = header & 0x0F
        frame = self._structs[-1]
        if tag == 0:
            if header != 0:
                raise _CompactWireMalformed("noncanonical_wire")
            frame.stopped = True
            return None
        if len(frame.seen_ids) >= self.budget.limits.max_fields_per_struct:
            raise PinnedReadRefused("resource_limit")
        self.budget._reserve(field_count=1)
        wire_type = _compact_wire_type(tag)
        delta = header >> 4
        if delta:
            field_id = frame.last_id + delta
        else:
            field_id = self.read_i16()
            if 1 <= field_id - frame.last_id <= 15:
                raise _CompactWireMalformed("noncanonical_wire")
        if not 0 < field_id <= 32767:
            raise _CompactWireMalformed("invalid_wire")
        if field_id in frame.seen_ids or allowed_fields.get(field_id) != wire_type:
            raise _CompactWireMalformed("schema_mismatch")
        frame.last_id = field_id
        frame.seen_ids.add(field_id)
        boolean = tag == 1 if wire_type == "bool" else None
        self._pending_bool = boolean
        return _CompactWireField(field_id, wire_type, boolean)

    def end_struct(self) -> None:
        if not self._structs or not self._structs[-1].stopped or self._pending_bool is not None:
            raise _CompactWireMalformed("invalid_wire")
        self._structs.pop()

    def read_bool(self) -> bool:
        if self._pending_bool is not None:
            value = self._pending_bool
            self._pending_bool = None
            return value
        match self._unsigned_byte():
            case 1:
                return True
            case 2:
                return False
            case _:
                raise _CompactWireMalformed("invalid_wire")

    def read_byte(self) -> int:
        value = self._unsigned_byte()
        return value - 256 if value >= 128 else value

    def read_i16(self) -> int:
        return self._signed_integer(16)

    def read_i32(self) -> int:
        return self._signed_integer(32)

    def read_i64(self) -> int:
        return self._signed_integer(64)

    def read_double(self) -> float:
        return unpack_from("<d", self.payload, self._consume(8))[0]

    def read_binary(self) -> bytes:
        length = self._varint(31)
        start = self._consume(length, binary=True)
        return self.payload[start : self.position]

    def read_binary_span(self) -> tuple[int, int]:
        """Return the retained binary body interval without copying its bytes."""
        length = self._varint(31)
        start = self._consume(length, binary=True, copy_binary=False)
        return start, self.position

    def read_list_header(self, expected_element_type: str) -> tuple[str, int]:
        _validate_compact_wire_type(expected_element_type)
        header = self._unsigned_byte()
        if header & 0x0F == 2:
            raise _CompactWireMalformed("noncanonical_wire")
        wire_type = _compact_wire_type(header & 0x0F)
        if wire_type != expected_element_type:
            raise _CompactWireMalformed("schema_mismatch")
        count = header >> 4
        if count == 15:
            count = self._varint(31)
            if count < 15:
                raise _CompactWireMalformed("noncanonical_wire")
        if count > self.budget.limits.max_items_per_container:
            raise PinnedReadRefused("resource_limit")
        self.budget._reserve(items=count, work_units=count)
        return wire_type, count

    def finish(self, *, require_exhausted: bool) -> None:
        if type(require_exhausted) is not bool:
            raise PinnedReadRefused("invalid_contract")
        if self._structs or self._pending_bool is not None:
            raise _CompactWireMalformed("invalid_wire")
        if require_exhausted and self.position != self.stop:
            raise _CompactWireMalformed("invalid_wire")


@dataclass(frozen=True, slots=True)
class PinnedReadReceipt:
    """Completed I/O and decode counters; no live accumulator is returned."""

    source_size_bytes: int
    total_read_bytes: int
    source_opens: int
    decoded_rows: int
    output_rows: int
    row_groups: int
    total_uncompressed_bytes: int
    completed_ns: int


@dataclass(frozen=True, slots=True)
class PinnedParquetRead:
    """Completed read envelope, published only after source-wide verification.

    The envelope and receipt are immutable; the frame remains a Polars value.
    The refusal-only reader currently never constructs this result.
    """

    frame: pl.DataFrame
    identity: PinnedParquetFile
    receipt: PinnedReadReceipt
    projection: ParquetProjection
    complete: Literal[True]


def read_pinned_parquet(
    root: Path,
    expected: TimestampFileMetadata,
    projection: ParquetProjection,
    limits: PinnedReadLimits,
    *,
    clock_ns: Callable[[], int],
) -> PinnedParquetRead:
    """Refuse unsupported enforcement before source access or callback execution."""
    # This refusal-only slice never inspects the source or consults the clock.
    _ = (root, expected, clock_ns)
    if type(projection) is not ParquetProjection or type(limits) is not PinnedReadLimits:
        raise PinnedReadRefused("invalid_contract")
    projection.__post_init__()
    limits.__post_init__()
    raise PinnedReadRefused("unsupported_mode")


_FooterKind = Literal[
    "FileMetaData",
    "SchemaElement",
    "RowGroup",
    "ColumnChunk",
    "ColumnMetaData",
    "KeyValue",
    "SortingColumn",
    "PageEncodingStats",
    "ColumnOrder",
    "TypeDefinedOrder",
    "LogicalType",
    "DecimalType",
    "TimeType",
    "TimestampType",
    "TimeUnit",
    "IntType",
    "VariantType",
    "GeometryType",
    "GeographyType",
    "StringType",
    "MapType",
    "ListType",
    "EnumType",
    "DateType",
    "NullType",
    "JsonType",
    "BsonType",
    "UUIDType",
    "Float16Type",
    "MilliSeconds",
    "MicroSeconds",
    "NanoSeconds",
    "ColumnCryptoMetaData",
    "EncryptionWithFooterKey",
    "EncryptionWithColumnKey",
    "EncryptionAlgorithm",
    "AesGcmV1",
    "AesGcmCtrV1",
    "FileCryptoMetaData",
    "Statistics",
    "SizeStatistics",
    "GeospatialStatistics",
    "BoundingBox",
]
_FooterWireType = Literal["bool", "byte", "i16", "i32", "i64", "double", "binary", "struct", "list"]


@dataclass(frozen=True, slots=True, repr=False)
class _FooterLimits:
    max_schema_nodes: int
    max_schema_depth: int
    max_leaf_columns: int
    max_row_groups: int
    max_total_column_chunks: int
    max_total_containers: int
    max_total_metadata_entries: int
    max_total_retained_records: int
    max_semantic_work_units: int
    max_span_comparison_bytes: int

    def __post_init__(self) -> None:
        for declaration in fields(self):
            value = getattr(self, declaration.name)
            if type(value) is not int or value <= 0:
                raise PinnedReadRefused("invalid_contract")


@dataclass(frozen=True, slots=True, repr=False)
class _FooterCharge:
    schema_nodes: int = 0
    leaf_columns: int = 0
    row_groups: int = 0
    total_column_chunks: int = 0
    total_containers: int = 0
    total_metadata_entries: int = 0
    total_retained_records: int = 0
    semantic_work_units: int = 0
    span_comparison_bytes: int = 0

    def __post_init__(self) -> None:
        for declaration in fields(self):
            value = getattr(self, declaration.name)
            if type(value) is not int or value < 0:
                raise PinnedReadRefused("invalid_contract")


class _FooterBudget:
    """Invocation-owned mutable footer accounting."""

    def __init__(self, *, wire_budget: _CompactWireBudget, limits: _FooterLimits) -> None:
        if type(wire_budget) is not _CompactWireBudget or type(limits) is not _FooterLimits:
            raise PinnedReadRefused("invalid_contract")
        limits.__post_init__()
        self.wire_budget = wire_budget
        self.limits = limits
        self.schema_nodes = 0
        self.leaf_columns = 0
        self.row_groups = 0
        self.total_column_chunks = 0
        self.total_containers = 0
        self.total_metadata_entries = 0
        self.total_retained_records = 0
        self.semantic_work_units = 0
        self.span_comparison_bytes = 0
        self.schema_depth_peak = 0
        self.active_schema_depth = 0

    def reserve(self, charge: _FooterCharge) -> None:
        if type(charge) is not _FooterCharge:
            raise PinnedReadRefused("invalid_contract")
        charge.__post_init__()
        for axis in fields(charge):
            current = getattr(self, axis.name)
            increment = getattr(charge, axis.name)
            limit = getattr(self.limits, "max_" + axis.name)
            if increment > limit - current:
                raise PinnedReadRefused("resource_limit")
        for axis in fields(charge):
            setattr(self, axis.name, getattr(self, axis.name) + getattr(charge, axis.name))

    def push_schema(self) -> None:
        if self.active_schema_depth >= self.limits.max_schema_depth:
            raise PinnedReadRefused("resource_limit")
        self.reserve(_FooterCharge(total_retained_records=1, semantic_work_units=1))
        self.active_schema_depth += 1
        self.schema_depth_peak = max(self.schema_depth_peak, self.active_schema_depth)

    def pop_schema(self) -> None:
        if self.active_schema_depth == 0:
            raise PinnedReadRefused("invalid_contract")
        self.reserve(_FooterCharge(semantic_work_units=1))
        self.active_schema_depth -= 1


@dataclass(frozen=True, slots=True, repr=False)
class _FooterExtent:
    source_size: int
    footer_start: int
    footer_end: int

    def __post_init__(self) -> None:
        if (
            type(self.source_size) is not int
            or type(self.footer_start) is not int
            or type(self.footer_end) is not int
            or not 0 <= self.footer_start < self.footer_end <= self.source_size
        ):
            raise PinnedReadRefused("invalid_contract")


@dataclass(frozen=True, slots=True, repr=False)
class _FooterSpan:
    start: int
    end: int


@dataclass(frozen=True, slots=True, repr=False)
class _FooterField:
    field_id: int
    value: _FooterValue


@dataclass(frozen=True, slots=True, repr=False)
class _FooterNode:
    kind: _FooterKind
    fields: tuple[_FooterField, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _FooterList:
    element_type: _FooterWireType
    values: tuple[_FooterValue, ...]


_FooterValue = bool | int | _FooterSpan | _FooterNode | _FooterList | Literal["opaque"]


@dataclass(frozen=True, slots=True, repr=False)
class _FooterDocument:
    root: _FooterNode
    payload: bytes
    extent: _FooterExtent
    budget: _FooterBudget = _footer_record_field(repr=False, compare=False)
    read_eligible: Literal[False] = False


@dataclass(frozen=True, slots=True)
class _FooterFieldSpec:
    field_id: int
    wire_type: _FooterWireType
    required: bool
    element_type: _FooterWireType | None
    child_kind: _FooterKind | None
    enum_domain: tuple[int, ...] | None
    opaque: bool


@dataclass(frozen=True, slots=True)
class _FooterStructSpec:
    kind: _FooterKind
    fields: tuple[_FooterFieldSpec, ...]
    union: bool = False


def _footer_field(
    field_id: int,
    wire_type: _FooterWireType,
    *,
    required: bool = False,
    element_type: _FooterWireType | None = None,
    child_kind: _FooterKind | None = None,
    enum_domain: tuple[int, ...] | None = None,
    opaque: bool = False,
) -> _FooterFieldSpec:
    return _FooterFieldSpec(
        field_id, wire_type, required, element_type, child_kind, enum_domain, opaque
    )


_FOOTER_TYPES = tuple(range(8))
_FOOTER_ENCODINGS = (0, 2, 3, 4, 5, 6, 7, 8, 9)
_FOOTER_SPECS = (
    _FooterStructSpec(
        "FileMetaData",
        (
            _footer_field(1, "i32", required=True),
            _footer_field(
                2, "list", required=True, element_type="struct", child_kind="SchemaElement"
            ),
            _footer_field(3, "i64", required=True),
            _footer_field(4, "list", required=True, element_type="struct", child_kind="RowGroup"),
            _footer_field(5, "list", element_type="struct", child_kind="KeyValue"),
            _footer_field(6, "binary", opaque=True),
            _footer_field(7, "list", element_type="struct", child_kind="ColumnOrder"),
            _footer_field(8, "struct", child_kind="EncryptionAlgorithm"),
            _footer_field(9, "binary", opaque=True),
        ),
    ),
    _FooterStructSpec(
        "SchemaElement",
        (
            _footer_field(1, "i32", enum_domain=_FOOTER_TYPES),
            _footer_field(2, "i32"),
            _footer_field(3, "i32", enum_domain=(0, 1, 2)),
            _footer_field(4, "binary", required=True),
            _footer_field(5, "i32"),
            _footer_field(6, "i32", enum_domain=tuple(range(22))),
            _footer_field(7, "i32"),
            _footer_field(8, "i32"),
            _footer_field(9, "i32"),
            _footer_field(10, "struct", child_kind="LogicalType"),
        ),
    ),
    _FooterStructSpec(
        "RowGroup",
        (
            _footer_field(
                1, "list", required=True, element_type="struct", child_kind="ColumnChunk"
            ),
            _footer_field(2, "i64", required=True),
            _footer_field(3, "i64", required=True),
            _footer_field(4, "list", element_type="struct", child_kind="SortingColumn"),
            _footer_field(5, "i64"),
            _footer_field(6, "i64"),
            _footer_field(7, "i16"),
        ),
    ),
    _FooterStructSpec(
        "ColumnChunk",
        (
            _footer_field(1, "binary"),
            _footer_field(2, "i64", required=True),
            _footer_field(3, "struct", child_kind="ColumnMetaData"),
            _footer_field(4, "i64"),
            _footer_field(5, "i32"),
            _footer_field(6, "i64"),
            _footer_field(7, "i32"),
            _footer_field(8, "struct", child_kind="ColumnCryptoMetaData"),
            _footer_field(9, "binary", opaque=True),
        ),
    ),
    _FooterStructSpec(
        "ColumnMetaData",
        (
            _footer_field(1, "i32", required=True, enum_domain=_FOOTER_TYPES),
            _footer_field(
                2, "list", required=True, element_type="i32", enum_domain=_FOOTER_ENCODINGS
            ),
            _footer_field(3, "list", required=True, element_type="binary"),
            _footer_field(4, "i32", required=True, enum_domain=tuple(range(8))),
            _footer_field(5, "i64", required=True),
            _footer_field(6, "i64", required=True),
            _footer_field(7, "i64", required=True),
            _footer_field(8, "list", element_type="struct", child_kind="KeyValue"),
            _footer_field(9, "i64", required=True),
            _footer_field(10, "i64"),
            _footer_field(11, "i64"),
            _footer_field(12, "struct", child_kind="Statistics", opaque=True),
            _footer_field(13, "list", element_type="struct", child_kind="PageEncodingStats"),
            _footer_field(14, "i64"),
            _footer_field(15, "i32"),
            _footer_field(16, "struct", child_kind="SizeStatistics", opaque=True),
            _footer_field(17, "struct", child_kind="GeospatialStatistics", opaque=True),
        ),
    ),
    _FooterStructSpec(
        "KeyValue",
        (
            _footer_field(1, "binary", required=True),
            _footer_field(2, "binary", opaque=True),
        ),
    ),
    _FooterStructSpec(
        "SortingColumn",
        (
            _footer_field(1, "i32", required=True),
            _footer_field(2, "bool", required=True),
            _footer_field(3, "bool", required=True),
        ),
    ),
    _FooterStructSpec(
        "PageEncodingStats",
        (
            _footer_field(1, "i32", required=True, enum_domain=(0, 1, 2, 3)),
            _footer_field(2, "i32", required=True, enum_domain=_FOOTER_ENCODINGS),
            _footer_field(3, "i32", required=True),
        ),
    ),
    _FooterStructSpec(
        "ColumnOrder", (_footer_field(1, "struct", child_kind="TypeDefinedOrder"),), union=True
    ),
    _FooterStructSpec(
        "LogicalType",
        (
            _footer_field(1, "struct", child_kind="StringType"),
            _footer_field(2, "struct", child_kind="MapType"),
            _footer_field(3, "struct", child_kind="ListType"),
            _footer_field(4, "struct", child_kind="EnumType"),
            _footer_field(5, "struct", child_kind="DecimalType"),
            _footer_field(6, "struct", child_kind="DateType"),
            _footer_field(7, "struct", child_kind="TimeType"),
            _footer_field(8, "struct", child_kind="TimestampType"),
            _footer_field(10, "struct", child_kind="IntType"),
            _footer_field(11, "struct", child_kind="NullType"),
            _footer_field(12, "struct", child_kind="JsonType"),
            _footer_field(13, "struct", child_kind="BsonType"),
            _footer_field(14, "struct", child_kind="UUIDType"),
            _footer_field(15, "struct", child_kind="Float16Type"),
            _footer_field(16, "struct", child_kind="VariantType"),
            _footer_field(17, "struct", child_kind="GeometryType"),
            _footer_field(18, "struct", child_kind="GeographyType"),
        ),
        union=True,
    ),
    _FooterStructSpec(
        "DecimalType",
        (
            _footer_field(1, "i32", required=True),
            _footer_field(2, "i32", required=True),
        ),
    ),
    _FooterStructSpec(
        "TimeType",
        (
            _footer_field(1, "bool", required=True),
            _footer_field(2, "struct", required=True, child_kind="TimeUnit"),
        ),
    ),
    _FooterStructSpec(
        "TimestampType",
        (
            _footer_field(1, "bool", required=True),
            _footer_field(2, "struct", required=True, child_kind="TimeUnit"),
        ),
    ),
    _FooterStructSpec(
        "TimeUnit",
        (
            _footer_field(1, "struct", child_kind="MilliSeconds"),
            _footer_field(2, "struct", child_kind="MicroSeconds"),
            _footer_field(3, "struct", child_kind="NanoSeconds"),
        ),
        union=True,
    ),
    _FooterStructSpec(
        "IntType",
        (
            _footer_field(1, "byte", required=True),
            _footer_field(2, "bool", required=True),
        ),
    ),
    _FooterStructSpec("VariantType", (_footer_field(1, "byte"),)),
    _FooterStructSpec("GeometryType", (_footer_field(1, "binary", opaque=True),)),
    _FooterStructSpec(
        "GeographyType",
        (
            _footer_field(1, "binary", opaque=True),
            _footer_field(2, "i32", enum_domain=(0, 1, 2, 3, 4)),
        ),
    ),
    _FooterStructSpec(
        "ColumnCryptoMetaData",
        (
            _footer_field(1, "struct", child_kind="EncryptionWithFooterKey"),
            _footer_field(2, "struct", child_kind="EncryptionWithColumnKey"),
        ),
        union=True,
    ),
    _FooterStructSpec(
        "EncryptionWithColumnKey",
        (
            _footer_field(1, "list", required=True, element_type="binary"),
            _footer_field(2, "binary", opaque=True),
        ),
    ),
    _FooterStructSpec(
        "EncryptionAlgorithm",
        (
            _footer_field(1, "struct", child_kind="AesGcmV1"),
            _footer_field(2, "struct", child_kind="AesGcmCtrV1"),
        ),
        union=True,
    ),
    _FooterStructSpec(
        "AesGcmV1",
        (
            _footer_field(1, "binary", opaque=True),
            _footer_field(2, "binary", opaque=True),
            _footer_field(3, "bool"),
        ),
    ),
    _FooterStructSpec(
        "AesGcmCtrV1",
        (
            _footer_field(1, "binary", opaque=True),
            _footer_field(2, "binary", opaque=True),
            _footer_field(3, "bool"),
        ),
    ),
    _FooterStructSpec(
        "Statistics",
        (
            _footer_field(1, "binary", opaque=True),
            _footer_field(2, "binary", opaque=True),
            _footer_field(3, "i64", opaque=True),
            _footer_field(4, "i64", opaque=True),
            _footer_field(5, "binary", opaque=True),
            _footer_field(6, "binary", opaque=True),
            _footer_field(7, "bool", opaque=True),
            _footer_field(8, "bool", opaque=True),
        ),
    ),
    _FooterStructSpec(
        "SizeStatistics",
        (
            _footer_field(1, "i64", opaque=True),
            _footer_field(2, "list", element_type="i64", opaque=True),
            _footer_field(3, "list", element_type="i64", opaque=True),
        ),
    ),
    _FooterStructSpec(
        "GeospatialStatistics",
        (
            _footer_field(1, "struct", child_kind="BoundingBox", opaque=True),
            _footer_field(2, "list", element_type="i32", opaque=True),
        ),
    ),
    _FooterStructSpec(
        "BoundingBox",
        (
            _footer_field(1, "double", required=True, opaque=True),
            _footer_field(2, "double", required=True, opaque=True),
            _footer_field(3, "double", required=True, opaque=True),
            _footer_field(4, "double", required=True, opaque=True),
            _footer_field(5, "double", opaque=True),
            _footer_field(6, "double", opaque=True),
            _footer_field(7, "double", opaque=True),
            _footer_field(8, "double", opaque=True),
        ),
    ),
    _FooterStructSpec("TypeDefinedOrder", ()),
    _FooterStructSpec("StringType", ()),
    _FooterStructSpec("MapType", ()),
    _FooterStructSpec("ListType", ()),
    _FooterStructSpec("EnumType", ()),
    _FooterStructSpec("DateType", ()),
    _FooterStructSpec("NullType", ()),
    _FooterStructSpec("JsonType", ()),
    _FooterStructSpec("BsonType", ()),
    _FooterStructSpec("UUIDType", ()),
    _FooterStructSpec("Float16Type", ()),
    _FooterStructSpec("MilliSeconds", ()),
    _FooterStructSpec("MicroSeconds", ()),
    _FooterStructSpec("NanoSeconds", ()),
    _FooterStructSpec("EncryptionWithFooterKey", ()),
)
_FOOTER_BY_KIND = {spec.kind: spec for spec in _FOOTER_SPECS}
_FOOTER_FIELD_BY_ID = {
    spec.kind: {item.field_id: item for item in spec.fields} for spec in _FOOTER_SPECS
}
_FOOTER_ALLOWED_FIELDS: dict[_FooterKind, dict[int, str]] = {
    spec.kind: {item.field_id: item.wire_type for item in spec.fields} for spec in _FOOTER_SPECS
}
_FOOTER_REQUIRED = {
    spec.kind: sum(1 << item.field_id for item in spec.fields if item.required)
    for spec in _FOOTER_SPECS
}


@dataclass(slots=True, repr=False)
class _FooterStructFrame:
    """Mutable structural bookkeeping and bounded retained fields."""

    spec: _FooterStructSpec
    field_id: int
    opaque: bool
    retained: list[_FooterField] | None
    seen: int = 0
    members: int = 0


@dataclass(slots=True, repr=False)
class _FooterListFrame:
    """Mutable count and bounded retained items for one declared list."""

    spec: _FooterFieldSpec
    field_id: int
    opaque: bool
    remaining: int
    element_type: _FooterWireType
    retained: list[_FooterValue] | None


_FooterFrame = _FooterStructFrame | _FooterListFrame


def _footer_push_struct(
    stack: list[_FooterFrame],
    cursor: _CompactWireCursor,
    budget: _FooterBudget,
    kind: _FooterKind | None,
    field_id: int,
    opaque: bool,
) -> None:
    if kind is None:
        raise _CompactWireMalformed("schema_mismatch")
    budget.reserve(
        _FooterCharge(
            total_retained_records=2 if opaque else 3,
            semantic_work_units=1,
        )
    )
    cursor.begin_struct()
    stack.append(
        _FooterStructFrame(_FOOTER_BY_KIND[kind], field_id, opaque, None if opaque else [])
    )


def _footer_push_list(
    stack: list[_FooterFrame],
    cursor: _CompactWireCursor,
    budget: _FooterBudget,
    parent_kind: _FooterKind,
    spec: _FooterFieldSpec,
    opaque: bool,
) -> None:
    element_type = spec.element_type
    if element_type is None:
        raise _CompactWireMalformed("schema_mismatch")
    budget.reserve(_FooterCharge(semantic_work_units=1))
    _, count = cursor.read_list_header(element_type)
    budget.reserve(
        _FooterCharge(
            schema_nodes=count if parent_kind == "FileMetaData" and spec.field_id == 2 else 0,
            row_groups=count if parent_kind == "FileMetaData" and spec.field_id == 4 else 0,
            total_column_chunks=count if parent_kind == "RowGroup" and spec.field_id == 1 else 0,
            total_metadata_entries=count
            if (
                parent_kind == "FileMetaData"
                and spec.field_id == 5
                or parent_kind == "ColumnMetaData"
                and spec.field_id == 8
            )
            else 0,
            total_containers=1,
            total_retained_records=2 if opaque else 3,
            semantic_work_units=count + 1,
        )
    )
    stack.append(
        _FooterListFrame(spec, spec.field_id, opaque, count, element_type, None if opaque else [])
    )


def _footer_scalar(
    cursor: _CompactWireCursor,
    budget: _FooterBudget,
    wire_type: _FooterWireType,
    enum_domain: tuple[int, ...] | None,
    opaque: bool,
) -> _FooterValue:
    value: bool | int
    match wire_type:
        case "bool":
            value = cursor.read_bool()
        case "byte":
            value = cursor.read_byte()
        case "i16":
            value = cursor.read_i16()
        case "i32":
            value = cursor.read_i32()
        case "i64":
            value = cursor.read_i64()
        case "binary":
            start, end = cursor.read_binary_span()
            if opaque:
                return "opaque"
            budget.reserve(_FooterCharge(total_retained_records=1, semantic_work_units=1))
            return _FooterSpan(start, end)
        case "double":
            cursor._consume(8)
            return "opaque"
        case "struct" | "list":
            raise _CompactWireMalformed("invalid_wire")
    if enum_domain is not None and value not in enum_domain:
        raise _CompactWireMalformed("schema_mismatch")
    return "opaque" if opaque else value


def _footer_append(
    frame: _FooterFrame,
    field_id: int,
    value: _FooterValue,
    budget: _FooterBudget,
) -> None:
    match frame:
        case _FooterStructFrame():
            if frame.retained is not None:
                budget.reserve(_FooterCharge(total_retained_records=2, semantic_work_units=1))
                frame.retained.append(_FooterField(field_id, value))
        case _FooterListFrame():
            if frame.retained is not None:
                budget.reserve(_FooterCharge(total_retained_records=1))
                frame.retained.append(value)
            frame.remaining -= 1


def _footer_finish_struct(
    frame: _FooterStructFrame,
    cursor: _CompactWireCursor,
    budget: _FooterBudget,
) -> _FooterValue:
    required = _FOOTER_REQUIRED[frame.spec.kind]
    if frame.seen & required != required or frame.spec.union and frame.members != 1:
        raise _CompactWireMalformed("schema_mismatch")
    budget.reserve(_FooterCharge(semantic_work_units=1))
    cursor.end_struct()
    leaf_count = int(frame.spec.kind == "SchemaElement" and bool(frame.seen & 2))
    if frame.retained is None:
        budget.reserve(_FooterCharge(leaf_columns=leaf_count))
        return "opaque"
    budget.reserve(
        _FooterCharge(
            leaf_columns=leaf_count,
            total_retained_records=len(frame.retained) + 2,
            semantic_work_units=2,
        )
    )
    return _FooterNode(frame.spec.kind, tuple(frame.retained))


def _footer_finish_list(frame: _FooterListFrame, budget: _FooterBudget) -> _FooterValue:
    budget.reserve(_FooterCharge(semantic_work_units=1))
    if frame.retained is None:
        return "opaque"
    budget.reserve(
        _FooterCharge(
            total_retained_records=len(frame.retained) + 2,
            semantic_work_units=2,
        )
    )
    return _FooterList(frame.element_type, tuple(frame.retained))


def _parse_footer(
    cursor: _CompactWireCursor, extent: _FooterExtent, budget: _FooterBudget
) -> _FooterDocument:
    if (
        type(cursor) is not _CompactWireCursor
        or type(extent) is not _FooterExtent
        or type(budget) is not _FooterBudget
    ):
        raise PinnedReadRefused("invalid_contract")
    extent.__post_init__()
    budget.limits.__post_init__()
    if (
        cursor.kind != "footer"
        or cursor.position != cursor.start
        or cursor.stop - cursor.start != extent.footer_end - extent.footer_start
        or cursor.budget is not budget.wire_budget
        or cursor._structs
        or cursor._pending_bool is not None
    ):
        raise PinnedReadRefused("invalid_contract")
    budget.reserve(_FooterCharge(total_retained_records=4, semantic_work_units=1))
    cursor.begin_struct()
    stack: list[_FooterFrame] = [_FooterStructFrame(_FOOTER_BY_KIND["FileMetaData"], 0, False, [])]
    while stack:
        frame = stack[-1]
        match frame:
            case _FooterStructFrame():
                budget.reserve(_FooterCharge(semantic_work_units=1))
                header = cursor.read_field(_FOOTER_ALLOWED_FIELDS[frame.spec.kind])
                if header is None:
                    completed = _footer_finish_struct(frame, cursor, budget)
                    stack.pop()
                    if not stack:
                        cursor.finish(require_exhausted=True)
                        if not isinstance(completed, _FooterNode):
                            raise _CompactWireMalformed("schema_mismatch")
                        budget.reserve(
                            _FooterCharge(total_retained_records=1, semantic_work_units=1)
                        )
                        return _FooterDocument(completed, cursor.payload, extent, budget)
                    _footer_append(stack[-1], frame.field_id, completed, budget)
                    continue
                spec = _FOOTER_FIELD_BY_ID[frame.spec.kind][header.field_id]
                frame.seen |= 1 << header.field_id
                frame.members += 1
                opaque = frame.opaque or spec.opaque
                match spec.wire_type:
                    case "struct":
                        _footer_push_struct(
                            stack, cursor, budget, spec.child_kind, spec.field_id, opaque
                        )
                    case "list":
                        _footer_push_list(stack, cursor, budget, frame.spec.kind, spec, opaque)
                    case "bool" | "byte" | "i16" | "i32" | "i64" | "binary" | "double":
                        value = _footer_scalar(
                            cursor, budget, spec.wire_type, spec.enum_domain, opaque
                        )
                        _footer_append(frame, spec.field_id, value, budget)
            case _FooterListFrame():
                if frame.remaining == 0:
                    completed = _footer_finish_list(frame, budget)
                    stack.pop()
                    _footer_append(stack[-1], frame.field_id, completed, budget)
                elif frame.element_type == "struct":
                    _footer_push_struct(
                        stack, cursor, budget, frame.spec.child_kind, 0, frame.opaque
                    )
                else:
                    value = _footer_scalar(
                        cursor,
                        budget,
                        frame.element_type,
                        frame.spec.enum_domain,
                        frame.opaque,
                    )
                    _footer_append(frame, 0, value, budget)
    raise _CompactWireMalformed("invalid_wire")


_FooterCapability = Literal[
    "root_repetition_present",
    "primitive_child_count_present",
    "row_group_ordinal_interpretation_unresolved",
    "unresolved_schema_tree",
    "ambiguous_projection_name",
    "unresolved_logical_compatibility",
    "external_column_reference",
    "plaintext_column_metadata_absent",
    "encrypted_column_metadata_present",
    "column_encryption_present",
    "footer_encryption_present",
    "footer_signing_metadata_present",
    "arrow_schema_metadata_present",
    "arrow_extension_metadata_present",
    "opaque_metadata_present",
    "unbounded_bloom_extent",
    "legacy_index_page_reference",
    "unrecognized_file_version",
    "unresolved_extent_interpretation",
    "page_profile_unverified",
    "native_conversion_unproved",
    "source_binding_unproved",
    "utc_value_gate_unproved",
]


@dataclass(frozen=True, slots=True, repr=False)
class _FooterLeafProfile:
    schema_index: int
    leaf_ordinal: int | None
    ancestor_indices: tuple[int, ...] | None
    path: tuple[_FooterSpan, ...] | None
    physical_type: int
    repetition: int | None
    type_length: int | None
    converted_type: int | None
    scale: int | None
    precision: int | None
    field_id: int | None
    logical_type: _FooterNode | None
    definition_depth: int | None
    repetition_depth: int | None
    flags: frozenset[_FooterCapability]


@dataclass(frozen=True, slots=True, repr=False)
class _FooterChunkProfile:
    row_group_index: int
    column_index: int
    schema_index: int | None
    physical_type: int | None
    codec: int | None
    encodings: tuple[int, ...] | None
    num_values: int | None
    row_group_rows: int
    compressed_size: int | None
    uncompressed_size: int | None
    data_page_offset: int | None
    dictionary_page_offset: int | None
    index_page_offset: int | None
    legacy_file_offset: int
    candidate_extent: tuple[int, int] | None
    flags: frozenset[_FooterCapability]


@dataclass(frozen=True, slots=True, repr=False)
class _FooterProfile:
    document: _FooterDocument
    leaves: tuple[_FooterLeafProfile, ...]
    chunks: tuple[_FooterChunkProfile, ...]
    flags: frozenset[_FooterCapability]
    read_eligible: Literal[False]


def _footer_fixed_decimal_capacity(
    *, byte_width: int, precision: int, budget: _FooterBudget
) -> bool:
    if (
        type(byte_width) is not int
        or type(precision) is not int
        or not 0 < byte_width <= 2147483647
        or not 0 < precision <= 2147483647
        or type(budget) is not _FooterBudget
        or type(budget.limits) is not _FooterLimits
        or type(budget.wire_budget) is not _CompactWireBudget
    ):
        raise PinnedReadRefused("invalid_contract")
    budget.limits.__post_init__()
    budget.reserve(_FooterCharge(semantic_work_units=2))
    bits = 8 * byte_width - 1
    if 4 * precision <= bits:
        return True
    if 3 * precision >= bits:
        return False
    work = precision * (precision + 1) // 2 + 2 * precision + 2
    budget.reserve(_FooterCharge(semantic_work_units=work))
    value = 1
    for _ in range(precision):
        value *= 10
    return value.bit_length() <= bits


def _complete_footer_profile(document: _FooterDocument) -> _FooterProfile:
    if (
        type(document) is not _FooterDocument
        or type(document.root) is not _FooterNode
        or document.root.kind != "FileMetaData"
        or type(document.payload) is not bytes
        or type(document.extent) is not _FooterExtent
        or type(document.budget) is not _FooterBudget
        or type(document.budget.limits) is not _FooterLimits
        or type(document.budget.wire_budget) is not _CompactWireBudget
        or document.read_eligible is not False
        or document.budget.active_schema_depth != 0
    ):
        raise PinnedReadRefused("invalid_contract")
    document.extent.__post_init__()
    document.budget.limits.__post_init__()
    budget = document.budget
    flags = _footer_profile_new_flags(budget)
    for flag in _FOOTER_PROFILE_MANDATORY:
        _footer_profile_work(budget)
        _footer_profile_flag(flags, flag, budget)
    version = _footer_profile_required_integer(document.root, 1, budget)
    _footer_profile_work(budget)
    if version not in (1, 2):
        _footer_profile_flag(flags, "unrecognized_file_version", budget)
    encryption = _footer_profile_value(document.root, 8, budget)
    _footer_profile_work(budget)
    if encryption is not None:
        _footer_profile_flag(flags, "footer_encryption_present", budget)
    signing = _footer_profile_value(document.root, 9, budget)
    _footer_profile_work(budget)
    if signing is not None:
        _footer_profile_flag(flags, "footer_signing_metadata_present", budget)
    if _footer_profile_has_opaque(document.root, budget):
        _footer_profile_flag(flags, "opaque_metadata_present", budget)
    _footer_profile_metadata(document, document.root, 5, flags)
    leaves, resolved = _footer_profile_schema(document, flags)
    chunks = _footer_profile_groups(document, leaves, resolved, flags)
    _footer_profile_allocate(budget, len(leaves) + 1)
    frozen_leaves = tuple(leaves)
    _footer_profile_allocate(budget, len(chunks) + 1)
    frozen_chunks = tuple(chunks)
    frozen_flags = _footer_profile_freeze_flags(flags, budget)
    if budget.active_schema_depth != 0:
        raise PinnedReadRefused("invalid_contract")
    _footer_profile_allocate(budget, 1)
    return _FooterProfile(document, frozen_leaves, frozen_chunks, frozen_flags, False)


_FOOTER_PROFILE_MANDATORY: tuple[_FooterCapability, ...] = (
    "page_profile_unverified",
    "native_conversion_unproved",
    "source_binding_unproved",
    "utc_value_gate_unproved",
)


def _footer_profile_work(budget: _FooterBudget, units: int = 1) -> None:
    budget.reserve(_FooterCharge(semantic_work_units=units))


def _footer_profile_allocate(budget: _FooterBudget, records: int, work: int = 1) -> None:
    budget.reserve(_FooterCharge(total_retained_records=records, semantic_work_units=work))


def _footer_profile_value(
    node: _FooterNode, field_id: int, budget: _FooterBudget
) -> _FooterValue | None:
    for item in node.fields:
        _footer_profile_work(budget)
        if item.field_id == field_id:
            return item.value
    return None


def _footer_profile_integer(node: _FooterNode, field_id: int, budget: _FooterBudget) -> int | None:
    value = _footer_profile_value(node, field_id, budget)
    return value if type(value) is int else None


def _footer_profile_required_integer(
    node: _FooterNode, field_id: int, budget: _FooterBudget
) -> int:
    value = _footer_profile_integer(node, field_id, budget)
    if value is None:
        raise PinnedReadRefused("invalid_contract")
    return value


def _footer_profile_node(
    node: _FooterNode, field_id: int, budget: _FooterBudget
) -> _FooterNode | None:
    value = _footer_profile_value(node, field_id, budget)
    return value if isinstance(value, _FooterNode) else None


def _footer_profile_values(
    node: _FooterNode, field_id: int, budget: _FooterBudget
) -> tuple[_FooterValue, ...]:
    value = _footer_profile_value(node, field_id, budget)
    return value.values if isinstance(value, _FooterList) else ()


def _footer_profile_span(node: _FooterNode, field_id: int, budget: _FooterBudget) -> _FooterSpan:
    value = _footer_profile_value(node, field_id, budget)
    if not isinstance(value, _FooterSpan):
        raise PinnedReadRefused("invalid_contract")
    return value


def _footer_profile_as_node(value: _FooterValue | str) -> _FooterNode:
    if not isinstance(value, _FooterNode):
        raise PinnedReadRefused("invalid_contract")
    return value


def _footer_profile_new_flags(budget: _FooterBudget) -> set[_FooterCapability]:
    _footer_profile_allocate(budget, 1)
    return set()


def _footer_profile_flag(
    flags: set[_FooterCapability], flag: _FooterCapability, budget: _FooterBudget
) -> None:
    _footer_profile_work(budget)
    if flag not in flags:
        _footer_profile_allocate(budget, 1)
        flags.add(flag)


def _footer_profile_merge(
    flags: set[_FooterCapability],
    other: set[_FooterCapability] | frozenset[_FooterCapability],
    budget: _FooterBudget,
) -> None:
    for flag in other:
        _footer_profile_work(budget)
        _footer_profile_flag(flags, flag, budget)


def _footer_profile_freeze_flags(
    flags: set[_FooterCapability], budget: _FooterBudget
) -> frozenset[_FooterCapability]:
    _footer_profile_allocate(budget, len(flags) + 1)
    return frozenset(flags)


def _footer_profile_equal(
    document: _FooterDocument, left: _FooterSpan, right: _FooterSpan | bytes
) -> bool:
    left_size = left.end - left.start
    right_size = right.end - right.start if isinstance(right, _FooterSpan) else len(right)
    document.budget.reserve(
        _FooterCharge(semantic_work_units=1, span_comparison_bytes=left_size + right_size)
    )
    if left_size != right_size:
        return False
    for index in range(left_size):
        other = (
            document.payload[right.start + index]
            if isinstance(right, _FooterSpan)
            else right[index]
        )
        if document.payload[left.start + index] != other:
            return False
    return True


def _footer_profile_has_opaque(node: _FooterNode, budget: _FooterBudget) -> bool:
    _footer_profile_allocate(budget, 2)
    pending: list[_FooterNode | _FooterList] = [node]
    while pending:
        _footer_profile_work(budget)
        current = pending.pop()
        length = len(current.fields) if isinstance(current, _FooterNode) else len(current.values)
        for index in range(length):
            _footer_profile_work(budget)
            value = (
                current.fields[index].value
                if isinstance(current, _FooterNode)
                else current.values[index]
            )
            if type(value) is str:
                return True
            if isinstance(value, (_FooterNode, _FooterList)):
                _footer_profile_allocate(budget, 1)
                pending.append(value)
    return False


@dataclass(slots=True, repr=False)
class _FooterSchemaState:
    index: int
    node: _FooterNode
    name: _FooterSpan
    physical: int | None
    repetition: int | None
    children: int | None
    width: int | None
    converted: int | None
    scale: int | None
    precision: int | None
    field_id: int | None
    logical: _FooterNode | None
    flags: set[_FooterCapability]
    parent: int | None = None
    definition_depth: int = 0
    repetition_depth: int = 0


@dataclass(slots=True, repr=False)
class _FooterSchemaFrame:
    index: int
    remaining: int
    parent: _FooterSchemaFrame | None


_FOOTER_CONVERTED_PHYSICAL = (
    (6,),
    (),
    (),
    (),
    (6,),
    (1, 2, 6, 7),
    (1,),
    (1,),
    (2,),
    (2,),
    (2,),
    (1,),
    (1,),
    (1,),
    (2,),
    (1,),
    (1,),
    (1,),
    (2,),
    (6,),
    (6,),
    (7,),
)


def _footer_profile_decimal(
    state: _FooterSchemaState, precision: int, scale: int, budget: _FooterBudget
) -> None:
    if precision <= 0 or scale < 0 or scale > precision:
        raise _CompactWireMalformed("schema_mismatch")
    if state.physical == 1 and precision > 9 or state.physical == 2 and precision > 18:
        raise _CompactWireMalformed("schema_mismatch")
    if state.physical == 7 and (
        state.width is None
        or not _footer_fixed_decimal_capacity(
            byte_width=state.width, precision=precision, budget=budget
        )
    ):
        raise _CompactWireMalformed("schema_mismatch")


def _footer_profile_annotations(state: _FooterSchemaState, budget: _FooterBudget) -> None:
    _footer_profile_work(budget)
    physical = state.physical
    converted = state.converted
    logical = state.logical
    if physical == 7 and (state.width is None or state.width <= 0):
        raise _CompactWireMalformed("schema_mismatch")
    _footer_profile_work(budget)
    if converted is not None:
        permitted = _FOOTER_CONVERTED_PHYSICAL[converted]
        paired_integer = False
        if logical is not None:
            _footer_profile_work(budget)
            paired_integer = (
                logical.fields[0].field_id == 10 and 11 <= converted <= 18 and physical in (1, 2)
            )
        if converted in (1, 2, 3):
            if physical is not None:
                raise _CompactWireMalformed("schema_mismatch")
            _footer_profile_flag(state.flags, "unresolved_logical_compatibility", budget)
        elif physical not in permitted and not paired_integer:
            raise _CompactWireMalformed("schema_mismatch")
        if converted == 21 and state.width != 12:
            raise _CompactWireMalformed("schema_mismatch")
    mapped: int | None = None
    tag: int | None = None
    _footer_profile_work(budget)
    if logical is not None:
        _footer_profile_work(budget)
        tag = logical.fields[0].field_id
        body = _footer_profile_as_node(logical.fields[0].value)
        if tag in (2, 3):
            if physical is not None:
                raise _CompactWireMalformed("schema_mismatch")
            mapped = 1 if tag == 2 else 3
            _footer_profile_flag(state.flags, "unresolved_logical_compatibility", budget)
        elif tag in (1, 4, 12, 13, 17, 18):
            if physical != 6:
                raise _CompactWireMalformed("schema_mismatch")
            if tag in (1, 4, 12, 13):
                mapped = 0 if tag == 1 else 4 if tag == 4 else 19 if tag == 12 else 20
        elif tag == 5:
            if physical not in (1, 2, 6, 7):
                raise _CompactWireMalformed("schema_mismatch")
            precision = _footer_profile_required_integer(body, 2, budget)
            scale = _footer_profile_required_integer(body, 1, budget)
            _footer_profile_decimal(state, precision, scale, budget)
            if (
                state.precision is not None
                and state.precision != precision
                or state.scale is not None
                and state.scale != scale
            ):
                raise _CompactWireMalformed("schema_mismatch")
            if converted == 5 and (state.precision is None or state.scale is None):
                _footer_profile_flag(state.flags, "unresolved_logical_compatibility", budget)
            mapped = 5
        elif tag == 6:
            if physical != 1:
                raise _CompactWireMalformed("schema_mismatch")
            mapped = 6
        elif tag in (7, 8):
            unit_node = _footer_profile_node(body, 2, budget)
            if unit_node is None:
                raise PinnedReadRefused("invalid_contract")
            _footer_profile_work(budget)
            unit = unit_node.fields[0].field_id
            if physical != (1 if tag == 7 and unit == 1 else 2):
                raise _CompactWireMalformed("schema_mismatch")
            if unit != 3:
                mapped = (6 if tag == 7 else 8) + unit
            _footer_profile_flag(state.flags, "unresolved_logical_compatibility", budget)
        elif tag == 10:
            width = _footer_profile_required_integer(body, 1, budget)
            signed = _footer_profile_value(body, 2, budget)
            if width not in (8, 16, 32, 64) or physical not in (1, 2):
                raise _CompactWireMalformed("schema_mismatch")
            ordinal = 0 if width == 8 else 1 if width == 16 else 2 if width == 32 else 3
            mapped = 11 + ordinal + (4 if signed is True else 0)
            if physical != (2 if width == 64 else 1):
                _footer_profile_flag(state.flags, "unresolved_logical_compatibility", budget)
        elif tag in (14, 15):
            if physical != 7 or state.width != (16 if tag == 14 else 2):
                raise _CompactWireMalformed("schema_mismatch")
        elif tag == 16:
            _footer_profile_flag(state.flags, "unresolved_logical_compatibility", budget)
        elif physical is None:
            raise _CompactWireMalformed("schema_mismatch")
        _footer_profile_work(budget)
        if mapped is not None:
            if converted is None:
                _footer_profile_flag(state.flags, "unresolved_logical_compatibility", budget)
            elif converted != mapped:
                raise _CompactWireMalformed("schema_mismatch")
        elif converted is not None:
            _footer_profile_flag(state.flags, "unresolved_logical_compatibility", budget)
    _footer_profile_work(budget)
    if converted == 5 and tag != 5:
        if state.precision is None or state.scale is None:
            raise _CompactWireMalformed("schema_mismatch")
        _footer_profile_decimal(state, state.precision, state.scale, budget)
    elif tag != 5 and converted != 5 and (state.precision is not None or state.scale is not None):
        _footer_profile_flag(state.flags, "unresolved_logical_compatibility", budget)
    if _footer_profile_has_opaque(state.node, budget):
        _footer_profile_flag(state.flags, "opaque_metadata_present", budget)


def _footer_profile_schema(
    document: _FooterDocument, flags: set[_FooterCapability]
) -> tuple[list[_FooterLeafProfile], bool]:
    budget = document.budget
    declarations = _footer_profile_values(document.root, 2, budget)
    if not declarations:
        raise _CompactWireMalformed("schema_mismatch")
    _footer_profile_allocate(budget, 1)
    states: list[_FooterSchemaState] = []
    ambiguous = False
    for index, declaration in enumerate(declarations):
        _footer_profile_work(budget)
        node = _footer_profile_as_node(declaration)
        name = _footer_profile_span(node, 4, budget)
        physical = _footer_profile_integer(node, 1, budget)
        repetition = _footer_profile_integer(node, 3, budget)
        children = _footer_profile_integer(node, 5, budget)
        width = _footer_profile_integer(node, 2, budget)
        converted = _footer_profile_integer(node, 6, budget)
        scale = _footer_profile_integer(node, 7, budget)
        precision = _footer_profile_integer(node, 8, budget)
        field_id = _footer_profile_integer(node, 9, budget)
        logical = _footer_profile_node(node, 10, budget)
        own_flags = _footer_profile_new_flags(budget)
        if children is not None and children < 0:
            raise _CompactWireMalformed("schema_mismatch")
        _footer_profile_work(budget)
        if index == 0:
            if physical is not None or children is None:
                raise _CompactWireMalformed("schema_mismatch")
            if repetition is not None:
                _footer_profile_flag(flags, "root_repetition_present", budget)
        else:
            if repetition is None or physical is None and children is None:
                raise _CompactWireMalformed("schema_mismatch")
            if physical is not None and children is not None:
                ambiguous = True
                _footer_profile_flag(own_flags, "primitive_child_count_present", budget)
        _footer_profile_allocate(budget, 2)
        state = _FooterSchemaState(
            index,
            node,
            name,
            physical,
            repetition,
            children,
            width,
            converted,
            scale,
            precision,
            field_id,
            logical,
            own_flags,
        )
        states.append(state)
        _footer_profile_annotations(state, budget)
    if ambiguous:
        _footer_profile_flag(flags, "unresolved_schema_tree", budget)
    else:
        root_children = states[0].children
        if root_children is None:
            raise PinnedReadRefused("invalid_contract")
        budget.push_schema()
        frame: _FooterSchemaFrame | None = _FooterSchemaFrame(0, root_children, None)
        for index in range(1, len(states)):
            _footer_profile_work(budget)
            while frame is not None and frame.remaining == 0:
                budget.pop_schema()
                frame = frame.parent
            if frame is None:
                raise _CompactWireMalformed("schema_mismatch")
            state = states[index]
            _footer_profile_allocate(budget, 1, 0)
            budget.push_schema()
            child = _FooterSchemaFrame(index, state.children or 0, frame)
            frame.remaining -= 1
            state.parent = frame.index
            parent = states[frame.index]
            state.definition_depth = parent.definition_depth + (state.repetition in (1, 2))
            state.repetition_depth = parent.repetition_depth + (state.repetition == 2)
            frame = child
        while frame is not None:
            if frame.remaining:
                raise _CompactWireMalformed("schema_mismatch")
            budget.pop_schema()
            frame = frame.parent
        for index in range(1, len(states)):
            _footer_profile_work(budget)
            current = states[index]
            for other_index in range(index + 1, len(states)):
                _footer_profile_work(budget)
                other = states[other_index]
                if current.parent == other.parent and _footer_profile_equal(
                    document, current.name, other.name
                ):
                    _footer_profile_flag(current.flags, "ambiguous_projection_name", budget)
                    _footer_profile_flag(other.flags, "ambiguous_projection_name", budget)
    _footer_profile_allocate(budget, 1)
    leaves: list[_FooterLeafProfile] = []
    for state in states:
        _footer_profile_work(budget)
        _footer_profile_merge(flags, state.flags, budget)
        if state.physical is None:
            continue
        ancestor_indices: tuple[int, ...] | None = None
        path: tuple[_FooterSpan, ...] | None = None
        ordinal: int | None = None
        definition_depth: int | None = None
        repetition_depth: int | None = None
        if ambiguous:
            _footer_profile_flag(state.flags, "unresolved_schema_tree", budget)
        else:
            _footer_profile_allocate(budget, 2, 2)
            ancestors: list[int] = []
            components: list[_FooterSpan] = []
            _footer_profile_allocate(budget, 1, 0)
            components.append(state.name)
            parent_index = state.parent
            while parent_index is not None:
                _footer_profile_work(budget)
                parent = states[parent_index]
                _footer_profile_allocate(budget, 1, 0)
                ancestors.append(parent_index)
                if parent_index != 0:
                    _footer_profile_allocate(budget, 1, 0)
                    components.append(parent.name)
                _footer_profile_merge(state.flags, parent.flags, budget)
                parent_index = parent.parent
            _footer_profile_work(budget, len(ancestors) + len(components))
            ancestors.reverse()
            components.reverse()
            _footer_profile_allocate(budget, len(ancestors) + 1)
            ancestor_indices = tuple(ancestors)
            _footer_profile_allocate(budget, len(components) + 1)
            path = tuple(components)
            ordinal = len(leaves)
            definition_depth = state.definition_depth
            repetition_depth = state.repetition_depth
        frozen_flags = _footer_profile_freeze_flags(state.flags, budget)
        _footer_profile_allocate(budget, 2)
        leaves.append(
            _FooterLeafProfile(
                state.index,
                ordinal,
                ancestor_indices,
                path,
                state.physical,
                state.repetition,
                state.width,
                state.converted,
                state.scale,
                state.precision,
                state.field_id,
                state.logical,
                definition_depth,
                repetition_depth,
                frozen_flags,
            )
        )
        _footer_profile_merge(flags, state.flags, budget)
    _footer_profile_allocate(budget, 3)
    return leaves, not ambiguous


@dataclass(slots=True, repr=False)
class _FooterChunkState:
    row_group_index: int
    column_index: int
    schema_index: int | None
    physical_type: int | None
    codec: int | None
    encodings: tuple[int, ...] | None
    num_values: int | None
    row_group_rows: int
    compressed_size: int | None
    uncompressed_size: int | None
    data_page_offset: int | None
    dictionary_page_offset: int | None
    index_page_offset: int | None
    legacy_file_offset: int
    candidate_extent: tuple[int, int] | None
    flags: set[_FooterCapability]


@dataclass(frozen=True, slots=True, repr=False)
class _FooterProfileInterval:
    start: int
    end: int
    chunk_index: int
    page: bool


def _footer_profile_nonnegative(value: int | None) -> None:
    if value is not None and value < 0:
        raise _CompactWireMalformed("schema_mismatch")


def _footer_profile_metadata(
    document: _FooterDocument, node: _FooterNode, field_id: int, flags: set[_FooterCapability]
) -> None:
    budget = document.budget
    entries = _footer_profile_values(node, field_id, budget)
    _footer_profile_work(budget)
    if entries:
        _footer_profile_flag(flags, "opaque_metadata_present", budget)
    for index, value in enumerate(entries):
        _footer_profile_work(budget)
        entry = _footer_profile_as_node(value)
        key = _footer_profile_span(entry, 1, budget)
        for previous_index in range(index):
            _footer_profile_work(budget)
            previous = _footer_profile_as_node(entries[previous_index])
            previous_key = _footer_profile_span(previous, 1, budget)
            if _footer_profile_equal(document, key, previous_key):
                raise _CompactWireMalformed("schema_mismatch")
        if _footer_profile_equal(document, key, b"ARROW:schema"):
            _footer_profile_flag(flags, "arrow_schema_metadata_present", budget)
        if _footer_profile_equal(document, key, b"ARROW:extension:name"):
            _footer_profile_flag(flags, "arrow_extension_metadata_present", budget)
        if _footer_profile_equal(document, key, b"ARROW:extension:metadata"):
            _footer_profile_flag(flags, "arrow_extension_metadata_present", budget)


def _footer_profile_encodings(metadata: _FooterNode, budget: _FooterBudget) -> tuple[int, ...]:
    values = _footer_profile_values(metadata, 2, budget)
    _footer_profile_allocate(budget, 1)
    encodings: list[int] = []
    for value in values:
        _footer_profile_work(budget)
        if type(value) is not int:
            raise PinnedReadRefused("invalid_contract")
        _footer_profile_allocate(budget, 1, 0)
        encodings.append(value)
    _footer_profile_allocate(budget, len(encodings) + 1)
    return tuple(encodings)


def _footer_profile_encoding_stats(
    metadata: _FooterNode, encodings: tuple[int, ...], budget: _FooterBudget
) -> None:
    for value in _footer_profile_values(metadata, 13, budget):
        _footer_profile_work(budget)
        node = _footer_profile_as_node(value)
        count = _footer_profile_required_integer(node, 3, budget)
        _footer_profile_nonnegative(count)
        if count > 0:
            declared = _footer_profile_required_integer(node, 2, budget)
            found = False
            for encoding in encodings:
                _footer_profile_work(budget)
                if encoding == declared:
                    found = True
                    break
            if not found:
                raise _CompactWireMalformed("schema_mismatch")


def _footer_profile_interval(
    intervals: list[_FooterProfileInterval],
    start: int,
    end: int,
    chunk_index: int,
    page: bool,
    budget: _FooterBudget,
) -> None:
    if start < end:
        _footer_profile_allocate(budget, 2)
        intervals.append(_FooterProfileInterval(start, end, chunk_index, page))


def _footer_profile_sidecar(
    document: _FooterDocument,
    intervals: list[_FooterProfileInterval],
    offset: int | None,
    length: int | None,
    chunk_index: int,
    external: bool,
) -> None:
    _footer_profile_nonnegative(offset)
    _footer_profile_nonnegative(length)
    if (offset is None) != (length is None):
        raise _CompactWireMalformed("schema_mismatch")
    if offset is not None and length is not None and not external:
        if offset > document.extent.footer_start or offset + length > document.extent.footer_start:
            raise _CompactWireMalformed("schema_mismatch")
        _footer_profile_interval(
            intervals, offset, offset + length, chunk_index, False, document.budget
        )


def _footer_profile_candidate(
    document: _FooterDocument,
    data: int,
    dictionary: int | None,
    compressed: int,
    values: int,
    external: bool,
    flags: set[_FooterCapability],
) -> tuple[int, int] | None:
    budget = document.budget
    _footer_profile_nonnegative(data)
    _footer_profile_nonnegative(dictionary)
    if external:
        return None
    footer = document.extent.footer_start
    if data > footer or dictionary is not None and dictionary > footer:
        raise _CompactWireMalformed("schema_mismatch")
    if values > 0 and compressed == 0:
        raise _CompactWireMalformed("schema_mismatch")
    _footer_profile_work(budget)
    if dictionary is not None:
        if dictionary > 0 and dictionary >= data:
            raise _CompactWireMalformed("schema_mismatch")
        if dictionary == 0 or compressed == 0:
            _footer_profile_flag(flags, "unresolved_extent_interpretation", budget)
            return None
        start = dictionary
        end = start + compressed
        if data >= end:
            raise _CompactWireMalformed("schema_mismatch")
    else:
        start = data
        end = start + compressed
    if end > footer:
        raise _CompactWireMalformed("schema_mismatch")
    _footer_profile_allocate(budget, 3)
    return start, end


def _footer_profile_chunk(
    document: _FooterDocument,
    chunk: _FooterNode,
    row_group_index: int,
    column_index: int,
    group_rows: int,
    leaf: _FooterLeafProfile | None,
    intervals: list[_FooterProfileInterval],
    chunk_index: int,
    ordinal_present: bool,
) -> _FooterChunkState:
    budget = document.budget
    flags = _footer_profile_new_flags(budget)
    _footer_profile_work(budget)
    if leaf is not None:
        _footer_profile_merge(flags, leaf.flags, budget)
    schema_index = leaf.schema_index if leaf is not None else None
    legacy = _footer_profile_required_integer(chunk, 2, budget)
    _footer_profile_nonnegative(legacy)
    external_value = _footer_profile_value(chunk, 1, budget)
    _footer_profile_work(budget)
    external = external_value is not None
    if external:
        _footer_profile_flag(flags, "external_column_reference", budget)
    crypto = _footer_profile_value(chunk, 8, budget)
    _footer_profile_work(budget)
    if crypto is not None:
        _footer_profile_flag(flags, "column_encryption_present", budget)
    encrypted = _footer_profile_value(chunk, 9, budget)
    _footer_profile_work(budget)
    if encrypted is not None:
        _footer_profile_flag(flags, "encrypted_column_metadata_present", budget)
    _footer_profile_work(budget)
    if ordinal_present:
        _footer_profile_flag(flags, "row_group_ordinal_interpretation_unresolved", budget)
    if _footer_profile_has_opaque(chunk, budget):
        _footer_profile_flag(flags, "opaque_metadata_present", budget)
    for offset_field in (4, 6):
        _footer_profile_work(budget)
        offset = _footer_profile_integer(chunk, offset_field, budget)
        length = _footer_profile_integer(chunk, offset_field + 1, budget)
        _footer_profile_sidecar(document, intervals, offset, length, chunk_index, external)
    metadata = _footer_profile_node(chunk, 3, budget)
    physical: int | None = None
    codec: int | None = None
    encodings: tuple[int, ...] | None = None
    values: int | None = None
    compressed: int | None = None
    uncompressed: int | None = None
    data: int | None = None
    dictionary: int | None = None
    index: int | None = None
    candidate: tuple[int, int] | None = None
    _footer_profile_work(budget)
    if metadata is None:
        _footer_profile_flag(flags, "plaintext_column_metadata_absent", budget)
    else:
        physical = _footer_profile_required_integer(metadata, 1, budget)
        codec = _footer_profile_required_integer(metadata, 4, budget)
        values = _footer_profile_required_integer(metadata, 5, budget)
        uncompressed = _footer_profile_required_integer(metadata, 6, budget)
        compressed = _footer_profile_required_integer(metadata, 7, budget)
        data = _footer_profile_required_integer(metadata, 9, budget)
        index = _footer_profile_integer(metadata, 10, budget)
        dictionary = _footer_profile_integer(metadata, 11, budget)
        _footer_profile_nonnegative(values)
        _footer_profile_nonnegative(uncompressed)
        _footer_profile_nonnegative(compressed)
        encodings = _footer_profile_encodings(metadata, budget)
        _footer_profile_encoding_stats(metadata, encodings, budget)
        if leaf is not None:
            if physical != leaf.physical_type:
                raise _CompactWireMalformed("schema_mismatch")
            path = _footer_profile_values(metadata, 3, budget)
            if leaf.path is None or len(path) != len(leaf.path):
                raise _CompactWireMalformed("schema_mismatch")
            for component_index in range(len(path)):
                _footer_profile_work(budget)
                component = path[component_index]
                if not isinstance(component, _FooterSpan):
                    raise PinnedReadRefused("invalid_contract")
                if not _footer_profile_equal(document, component, leaf.path[component_index]):
                    raise _CompactWireMalformed("schema_mismatch")
            if leaf.repetition_depth == 0 and values != group_rows:
                raise _CompactWireMalformed("schema_mismatch")
        candidate = _footer_profile_candidate(
            document, data, dictionary, compressed, values, external, flags
        )
        if candidate is not None:
            _footer_profile_interval(
                intervals, candidate[0], candidate[1], chunk_index, True, budget
            )
        _footer_profile_nonnegative(index)
        _footer_profile_work(budget)
        if index is not None:
            if not external and index >= document.extent.footer_start:
                raise _CompactWireMalformed("schema_mismatch")
            _footer_profile_flag(flags, "legacy_index_page_reference", budget)
        bloom_offset = _footer_profile_integer(metadata, 14, budget)
        bloom_length = _footer_profile_integer(metadata, 15, budget)
        _footer_profile_nonnegative(bloom_offset)
        _footer_profile_nonnegative(bloom_length)
        if bloom_offset is None and bloom_length is not None:
            raise _CompactWireMalformed("schema_mismatch")
        _footer_profile_work(budget)
        if bloom_offset is not None:
            if external or bloom_length is None:
                if not external and bloom_offset >= document.extent.footer_start:
                    raise _CompactWireMalformed("schema_mismatch")
                _footer_profile_flag(flags, "unbounded_bloom_extent", budget)
            else:
                _footer_profile_sidecar(
                    document, intervals, bloom_offset, bloom_length, chunk_index, False
                )
        _footer_profile_metadata(document, metadata, 8, flags)
    _footer_profile_allocate(budget, 2)
    return _FooterChunkState(
        row_group_index,
        column_index,
        schema_index,
        physical,
        codec,
        encodings,
        values,
        group_rows,
        compressed,
        uncompressed,
        data,
        dictionary,
        index,
        legacy,
        candidate,
        flags,
    )


def _footer_profile_groups(
    document: _FooterDocument,
    leaves: list[_FooterLeafProfile],
    resolved: bool,
    flags: set[_FooterCapability],
) -> list[_FooterChunkProfile]:
    budget = document.budget
    file_rows = _footer_profile_required_integer(document.root, 3, budget)
    _footer_profile_nonnegative(file_rows)
    column_orders = _footer_profile_value(document.root, 7, budget)
    if (
        resolved
        and isinstance(column_orders, _FooterList)
        and len(column_orders.values) != len(leaves)
    ):
        raise _CompactWireMalformed("schema_mismatch")
    _footer_profile_allocate(budget, 2, 2)
    states: list[_FooterChunkState] = []
    intervals: list[_FooterProfileInterval] = []
    total_rows = 0
    for group_index, value in enumerate(_footer_profile_values(document.root, 4, budget)):
        _footer_profile_work(budget)
        group = _footer_profile_as_node(value)
        rows = _footer_profile_required_integer(group, 3, budget)
        uncompressed = _footer_profile_required_integer(group, 2, budget)
        compressed = _footer_profile_integer(group, 6, budget)
        hint = _footer_profile_integer(group, 5, budget)
        ordinal = _footer_profile_integer(group, 7, budget)
        _footer_profile_nonnegative(rows)
        _footer_profile_nonnegative(uncompressed)
        _footer_profile_nonnegative(compressed)
        _footer_profile_nonnegative(hint)
        total_rows += rows
        _footer_profile_work(budget)
        if ordinal is not None:
            _footer_profile_flag(flags, "row_group_ordinal_interpretation_unresolved", budget)
        declarations = _footer_profile_values(group, 1, budget)
        if resolved and len(declarations) != len(leaves):
            raise _CompactWireMalformed("schema_mismatch")
        for sorting_value in _footer_profile_values(group, 4, budget):
            _footer_profile_work(budget)
            sorting = _footer_profile_as_node(sorting_value)
            sorting_index = _footer_profile_required_integer(sorting, 1, budget)
            if not 0 <= sorting_index < len(declarations):
                raise _CompactWireMalformed("schema_mismatch")
        first_chunk = len(states)
        complete_sizes = True
        compressed_sum = 0
        uncompressed_sum = 0
        for column_index, declaration in enumerate(declarations):
            _footer_profile_work(budget)
            chunk = _footer_profile_as_node(declaration)
            leaf = leaves[column_index] if resolved else None
            state = _footer_profile_chunk(
                document,
                chunk,
                group_index,
                column_index,
                rows,
                leaf,
                intervals,
                len(states),
                ordinal is not None,
            )
            states.append(state)
            _footer_profile_work(budget)
            if not resolved:
                _footer_profile_flag(state.flags, "unresolved_schema_tree", budget)
            if state.compressed_size is None or state.uncompressed_size is None:
                complete_sizes = False
            else:
                compressed_sum += state.compressed_size
                uncompressed_sum += state.uncompressed_size
        if complete_sizes and (
            uncompressed_sum != uncompressed
            or compressed is not None
            and compressed_sum != compressed
        ):
            raise _CompactWireMalformed("schema_mismatch")
        _footer_profile_work(budget)
        if hint is not None:
            minimum: int | None = None
            complete_candidates = first_chunk < len(states)
            for chunk_index in range(first_chunk, len(states)):
                _footer_profile_work(budget)
                candidate = states[chunk_index].candidate_extent
                if candidate is None or candidate[0] == candidate[1]:
                    complete_candidates = False
                elif minimum is None or candidate[0] < minimum:
                    minimum = candidate[0]
            if complete_candidates:
                if hint != minimum:
                    raise _CompactWireMalformed("schema_mismatch")
            else:
                _footer_profile_flag(flags, "unresolved_extent_interpretation", budget)
                for chunk_index in range(first_chunk, len(states)):
                    _footer_profile_work(budget)
                    _footer_profile_flag(
                        states[chunk_index].flags, "unresolved_extent_interpretation", budget
                    )
    if total_rows != file_rows:
        raise _CompactWireMalformed("schema_mismatch")
    for index, interval in enumerate(intervals):
        _footer_profile_work(budget)
        for other_index in range(index + 1, len(intervals)):
            _footer_profile_work(budget)
            other = intervals[other_index]
            if interval.start < other.end and other.start < interval.end:
                if (
                    interval.page
                    or other.page
                    or interval.start != other.start
                    or interval.end != other.end
                ):
                    raise _CompactWireMalformed("schema_mismatch")
                _footer_profile_flag(
                    states[interval.chunk_index].flags, "unresolved_extent_interpretation", budget
                )
                _footer_profile_flag(
                    states[other.chunk_index].flags, "unresolved_extent_interpretation", budget
                )
    _footer_profile_allocate(budget, 1)
    chunks: list[_FooterChunkProfile] = []
    for state in states:
        _footer_profile_work(budget)
        _footer_profile_merge(flags, state.flags, budget)
        frozen_flags = _footer_profile_freeze_flags(state.flags, budget)
        _footer_profile_allocate(budget, 2)
        chunks.append(
            _FooterChunkProfile(
                state.row_group_index,
                state.column_index,
                state.schema_index,
                state.physical_type,
                state.codec,
                state.encodings,
                state.num_values,
                state.row_group_rows,
                state.compressed_size,
                state.uncompressed_size,
                state.data_page_offset,
                state.dictionary_page_offset,
                state.index_page_offset,
                state.legacy_file_offset,
                state.candidate_extent,
                frozen_flags,
            )
        )
    return chunks


@dataclass(frozen=True, slots=True, repr=False)
class _FooterEnvelopeReceipt:
    source_opens_before: int
    source_opens_after: int
    total_read_bytes_before: int
    total_read_bytes_after: int
    ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    whole_source_digest_verified: Literal[False]


@dataclass(frozen=True, slots=True, repr=False)
class _FooterEnvelope:
    payload: bytes
    extent: _FooterExtent
    source_identity: tuple[int, int, int, int, int]
    receipt: _FooterEnvelopeReceipt
    read_budget: _PinnedReadBudget = _footer_record_field(repr=False, compare=False)


def _read_footer_envelope(
    root: Path,
    identity: PinnedParquetFile,
    *,
    maximum: int,
    max_footer_bytes: int,
    max_retained_payload_bytes: int,
    budget: _PinnedReadBudget,
) -> _FooterEnvelope:
    from liq.store.timestamp_metadata import (
        TimestampMetadataRefused,
        _pinned_source,
        _read_pinned_range,
    )

    if (
        type(maximum) is not int
        or maximum <= 0
        or type(max_footer_bytes) is not int
        or max_footer_bytes <= 0
        or type(max_retained_payload_bytes) is not int
        or max_retained_payload_bytes <= 0
        or type(budget) is not _PinnedReadBudget
    ):
        raise PinnedReadRefused("invalid_contract")
    opens_before = budget.source_opens
    bytes_before = budget.total_read_bytes
    with _pinned_source(root, identity, maximum, budget=budget) as source:
        size = source.size
        if size < 12:
            raise TimestampMetadataRefused("footer envelope length")
        if (
            max_retained_payload_bytes < 12
            or budget.limits.max_total_read_bytes - budget.total_read_bytes < 12
        ):
            raise PinnedReadRefused("resource_limit")
        leading = _read_pinned_range(source, offset=0, length=4)
        if leading != b"PAR1":
            if leading == b"PARE":
                raise PinnedReadRefused("unsupported_mode")
            raise TimestampMetadataRefused("footer envelope magic")
        trailer_start = size - 8
        trailing = _read_pinned_range(source, offset=trailer_start, length=8)
        if not (
            trailing[4] == 80 and trailing[5] == 65 and trailing[6] == 82 and trailing[7] == 49
        ):
            if trailing[4] == 80 and trailing[5] == 65 and trailing[6] == 82 and trailing[7] == 69:
                raise PinnedReadRefused("unsupported_mode")
            raise TimestampMetadataRefused("footer envelope magic")
        footer_length = trailing[0] | (trailing[1] << 8) | (trailing[2] << 16) | (trailing[3] << 24)
        footer_start = trailer_start - footer_length
        if footer_length == 0 or footer_start < 4:
            raise TimestampMetadataRefused("footer envelope length")
        if (
            footer_length > max_footer_bytes
            or footer_length + 12 > max_retained_payload_bytes
            or footer_length > budget.limits.max_total_read_bytes - budget.total_read_bytes
        ):
            raise PinnedReadRefused("resource_limit")
        payload = _read_pinned_range(source, offset=footer_start, length=footer_length)
        source_identity = source.identity
    extent = _FooterExtent(size, footer_start, trailer_start)
    receipt = _FooterEnvelopeReceipt(
        opens_before,
        budget.source_opens,
        bytes_before,
        budget.total_read_bytes,
        ((0, 4), (trailer_start, size), (footer_start, trailer_start)),
        False,
    )
    return _FooterEnvelope(payload, extent, source_identity, receipt, budget)


def _profile_footer_envelope(envelope: _FooterEnvelope, *, budget: _FooterBudget) -> _FooterProfile:
    if (
        type(envelope) is not _FooterEnvelope
        or type(envelope.payload) is not bytes
        or type(envelope.extent) is not _FooterExtent
        or type(envelope.receipt) is not _FooterEnvelopeReceipt
        or envelope.receipt.whole_source_digest_verified is not False
        or type(envelope.read_budget) is not _PinnedReadBudget
        or type(budget) is not _FooterBudget
        or type(budget.wire_budget) is not _CompactWireBudget
    ):
        raise PinnedReadRefused("invalid_contract")
    envelope.extent.__post_init__()
    if len(envelope.payload) != envelope.extent.footer_end - envelope.extent.footer_start:
        raise PinnedReadRefused("invalid_contract")
    budget.limits.__post_init__()
    budget.wire_budget.limits.__post_init__()
    envelope.read_budget.check_deadline()
    cursor = _CompactWireCursor(
        envelope.payload, 0, len(envelope.payload), budget.wire_budget, kind="footer"
    )
    document = _parse_footer(cursor, envelope.extent, budget)
    envelope.read_budget.check_deadline()
    profile = _complete_footer_profile(document)
    envelope.read_budget.check_deadline()
    return profile
