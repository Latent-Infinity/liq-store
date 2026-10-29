"""Source-free contract tests for bounded pinned Parquet reads."""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import stat
import subprocess
import sys
import unittest
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from liq.store import bounded_parquet
from liq.store.bounded_parquet import (
    ParquetProjection,
    PinnedParquetRead,
    PinnedReadLimits,
    PinnedReadReceipt,
    PinnedReadRefused,
    read_pinned_parquet,
)
from liq.store.exceptions import DataCorruptionError, StorageError


class ParquetProjectionTests(unittest.TestCase):
    def test_projection_is_frozen_and_uses_half_open_utc_bounds(self) -> None:
        projection = ParquetProjection(
            columns=("timestamp", "close"),
            start_inclusive=datetime(2025, 1, 2, tzinfo=UTC),
            end_exclusive=datetime(2025, 1, 3, tzinfo=UTC),
        )

        self.assertEqual(projection.columns, ("timestamp", "close"))
        with self.assertRaises(FrozenInstanceError):
            projection.columns = ("timestamp",)  # type: ignore[misc]

    def test_projection_rejects_empty_or_duplicate_columns(self) -> None:
        start = datetime(2025, 1, 2, tzinfo=UTC)
        end = datetime(2025, 1, 3, tzinfo=UTC)

        for columns in ((), ("timestamp", "timestamp")):
            with self.subTest(columns=columns), self.assertRaises(PinnedReadRefused):
                ParquetProjection(columns, start, end)

    def test_projection_requires_timestamp_and_scalar_column_names(self) -> None:
        start = datetime(2025, 1, 2, tzinfo=UTC)
        end = datetime(2025, 1, 3, tzinfo=UTC)

        for columns in (("close",), ("timestamp", 1)):
            with self.subTest(columns=columns), self.assertRaises(PinnedReadRefused):
                ParquetProjection(columns, start, end)  # type: ignore[arg-type]

    def test_projection_rejects_missing_non_utc_and_reversed_bounds(self) -> None:
        utc = datetime(2025, 1, 2, tzinfo=UTC)
        local = datetime(2025, 1, 2, tzinfo=timezone(timedelta(hours=1)))

        invalid_bounds = (
            (None, utc),
            (datetime(2025, 1, 2), utc),
            (datetime(2025, 1, 2, tzinfo=local.tzinfo), utc),
            (utc, utc),
            (utc + timedelta(days=1), utc),
        )
        for start, end in invalid_bounds:
            with self.subTest(start=start, end=end), self.assertRaises(PinnedReadRefused):
                ParquetProjection(("timestamp",), start, end)  # type: ignore[arg-type]


class PinnedReadLimitsTests(unittest.TestCase):
    def test_limits_are_frozen_and_each_positive_integer_is_required(self) -> None:
        field_names = (
            "max_source_size_bytes",
            "max_total_read_bytes",
            "max_source_opens",
            "max_decoded_rows",
            "max_output_rows",
            "max_row_groups",
            "max_row_group_uncompressed_bytes",
            "max_total_uncompressed_bytes",
            "batch_rows",
            "deadline_ns",
        )
        limits = PinnedReadLimits(**dict.fromkeys(field_names, 1))

        with self.assertRaises(FrozenInstanceError):
            limits.max_source_size_bytes = 2  # type: ignore[misc]

        for field_name in field_names:
            for invalid in (0, -1, True, 1.5, math.inf):
                values = dict.fromkeys(field_names, 1)
                values[field_name] = invalid
                with (
                    self.subTest(field=field_name, value=invalid),
                    self.assertRaises(PinnedReadRefused),
                ):
                    PinnedReadLimits(**values)  # type: ignore[arg-type]


class ReadRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.projection = ParquetProjection(
            columns=("timestamp", "close"),
            start_inclusive=datetime(2025, 1, 2, tzinfo=UTC),
            end_exclusive=datetime(2025, 1, 3, tzinfo=UTC),
        )
        self.limits = PinnedReadLimits(
            max_source_size_bytes=1,
            max_total_read_bytes=1,
            max_source_opens=1,
            max_decoded_rows=1,
            max_output_rows=1,
            max_row_groups=1,
            max_row_group_uncompressed_bytes=1,
            max_total_uncompressed_bytes=1,
            batch_rows=1,
            deadline_ns=1,
        )
        self.expected = SimpleNamespace(
            identity=SimpleNamespace(path="neutral.parquet", sha256="a" * 64),
            row_count=1,
            row_groups=(),
        )

    def test_receipt_and_result_contract_records_are_frozen(self) -> None:
        receipt = PinnedReadReceipt(
            source_size_bytes=1,
            total_read_bytes=1,
            source_opens=1,
            decoded_rows=1,
            output_rows=1,
            row_groups=1,
            total_uncompressed_bytes=1,
            completed_ns=1,
        )
        result = PinnedParquetRead(
            frame=None,  # type: ignore[arg-type]
            identity=self.expected.identity,
            receipt=receipt,
            projection=self.projection,
            complete=True,
        )

        with self.assertRaises(FrozenInstanceError):
            receipt.decoded_rows = 2  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.complete = False  # type: ignore[misc]

    def test_refusal_is_storage_error_but_not_data_corruption(self) -> None:
        error = PinnedReadRefused("unsupported_mode")

        self.assertIsInstance(error, StorageError)
        self.assertNotIsInstance(error, DataCorruptionError)
        self.assertEqual(error.reason_code, "unsupported_mode")

    def test_unsupported_mode_refuses_before_clock_or_filesystem_callbacks(self) -> None:
        def unexpected_clock_call() -> int:
            self.fail("clock callback must not run for the unsupported mode")

        with (
            patch.object(Path, "open", side_effect=AssertionError("open called")) as open_mock,
            patch.object(Path, "stat", side_effect=AssertionError("stat called")) as stat_mock,
            patch.object(
                Path, "read_bytes", side_effect=AssertionError("read called")
            ) as read_mock,
            self.assertRaises(PinnedReadRefused) as raised,
        ):
            read_pinned_parquet(
                Path("unused-root"),
                self.expected,
                self.projection,
                self.limits,
                clock_ns=unexpected_clock_call,
            )

        self.assertEqual(raised.exception.reason_code, "unsupported_mode")
        open_mock.assert_not_called()
        stat_mock.assert_not_called()
        read_mock.assert_not_called()

    def test_module_import_adds_no_backend_or_decoder_imports(self) -> None:
        code = """
import importlib
import sys
import liq.store
baseline = set(sys.modules)
importlib.import_module('liq.store.bounded_parquet')
added = set(sys.modules) - baseline
forbidden = (
    'pyarrow', 'duckdb', 'fsspec', 'timestamp_metadata',
    'timestamp_projection', 'liq.store.parquet',
)
assert not [name for name in added if any(part in name for part in forbidden)], added
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


class _LoggedPinnedIO:
    """Neutral syscall doubles; no file is created, opened, or decoded."""

    def __init__(self) -> None:
        self.payload = b"neutral storage bytes"
        self.root = Path("/sandbox")
        self.relative_path = "nested/neutral.parquet"
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.events: list[str] = []
        self.open_paths: list[str] = []
        self.acquired: list[int] = []
        self.closed: list[int] = []
        self.requests: list[int] = []
        self.hashed: list[bytes] = []
        self.short_reads: list[int] = []
        self.offset = 0
        self.fail_open = 0
        self.expire_after = ""
        self.expired = False
        self.fstat_calls = 0
        self.before = self.metadata()
        self.after = self.metadata()
        self.located = self.metadata()
        self.stack = ExitStack()

    def metadata(self, **changes: int) -> SimpleNamespace:
        values = {
            "st_dev": 1,
            "st_ino": 2,
            "st_size": len(self.payload),
            "st_mtime_ns": 3,
            "st_ctime_ns": 4,
            "st_mode": stat.S_IFREG | 0o600,
            "st_nlink": 1,
        }
        values.update(changes)
        return SimpleNamespace(**values)

    def event(self, name: str) -> None:
        self.events.append(name)
        if name == self.expire_after:
            self.expired = True

    def clock(self) -> int:
        self.events.append("clock")
        return 100 if self.expired else 1

    def open(self, path: str, flags: int, *, dir_fd: int | None = None) -> int:
        self.open_paths.append(path)
        self.event(f"open:{len(self.open_paths)}")
        if not flags & os.O_NOFOLLOW:
            raise AssertionError("every acquisition must use no-follow")
        if path != "/" and dir_fd not in self.acquired:
            raise AssertionError("relative acquisitions require an acquired parent")
        if len(self.open_paths) == self.fail_open:
            raise OSError("neutral acquisition failure")
        fd = 10 + len(self.open_paths)
        self.acquired.append(fd)
        return fd

    def close(self, fd: int) -> None:
        if fd not in self.acquired or fd in self.closed:
            raise AssertionError("unknown or already closed descriptor")
        self.closed.append(fd)
        self.event("close")

    def fstat(self, fd: int) -> SimpleNamespace:
        if fd not in self.acquired:
            raise AssertionError("metadata requires acquired descriptor")
        self.fstat_calls += 1
        self.event(f"fstat:{self.fstat_calls}")
        return self.before if self.fstat_calls == 1 else self.after

    def stat(self, path: str, *, dir_fd: int, follow_symlinks: bool) -> SimpleNamespace:
        if path != "neutral.parquet" or dir_fd not in self.acquired or follow_symlinks:
            raise AssertionError("path identity must be anchored and no-follow")
        self.event("stat")
        return self.located

    def read(self, fd: int, size: int) -> bytes:
        if fd not in self.acquired or fd in self.closed:
            raise AssertionError("read requires a live descriptor")
        self.requests.append(size)
        self.event(f"read:{len(self.requests)}")
        width = min(size, self.short_reads.pop(0)) if self.short_reads else size
        result = self.payload[self.offset : self.offset + width]
        self.offset += len(result)
        return result

    def __enter__(self) -> _LoggedPinnedIO:
        self.module = importlib.import_module("liq.store.timestamp_metadata")
        self.identity = self.module.PinnedParquetFile(self.relative_path, self.digest)
        real_sha256 = hashlib.sha256

        def observed_hash(payload: bytes):
            self.hashed.append(payload)
            self.event("hash")
            return real_sha256(payload)

        for name in ("open", "close", "fstat", "stat", "read"):
            self.stack.enter_context(
                patch.object(self.module.os, name, side_effect=getattr(self, name))
            )
        self.stack.enter_context(
            patch.object(self.module.hashlib, "sha256", side_effect=observed_hash)
        )
        return self

    def __exit__(self, *exc_info) -> None:
        self.stack.__exit__(*exc_info)


class PinnedByteBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = PinnedReadLimits(
            max_source_size_bytes=100,
            max_total_read_bytes=100,
            max_source_opens=10,
            max_decoded_rows=1,
            max_output_rows=1,
            max_row_groups=1,
            max_row_group_uncompressed_bytes=100,
            max_total_uncompressed_bytes=100,
            batch_rows=1,
            deadline_ns=100,
        )

    def budget(self, io: _LoggedPinnedIO, **changes: int):
        budget_type = getattr(bounded_parquet, "_PinnedReadBudget", None)
        self.assertIsNotNone(budget_type, "bounded reads require _PinnedReadBudget")
        return budget_type(replace(self.limits, **changes), clock_ns=io.clock)

    def assert_closed(self, io: _LoggedPinnedIO) -> None:
        self.assertCountEqual(io.closed, io.acquired)

    def test_entry_contracts_reject_malformed_records_before_callbacks(self) -> None:
        io = _LoggedPinnedIO()
        budget_type = type(self.budget(io))
        projection = ParquetProjection(
            ("timestamp",),
            datetime(2025, 1, 2, tzinfo=UTC),
            datetime(2025, 1, 3, tzinfo=UTC),
        )
        with (
            patch.object(os, "open", side_effect=AssertionError("unexpected open")) as opened,
            patch.object(
                hashlib, "sha256", side_effect=AssertionError("unexpected hash")
            ) as hashed,
        ):
            for limits, clock in ((SimpleNamespace(), io.clock), (self.limits, 1)):
                with self.subTest(limits=limits, clock=clock):
                    with self.assertRaises(PinnedReadRefused) as raised:
                        budget_type(limits, clock_ns=clock)
                    self.assertEqual(raised.exception.reason_code, "invalid_contract")
            for supplied_projection, supplied_limits in (
                (SimpleNamespace(), self.limits),
                (projection, SimpleNamespace()),
            ):
                with self.subTest(projection=supplied_projection, limits=supplied_limits):
                    with self.assertRaises(PinnedReadRefused) as raised:
                        read_pinned_parquet(
                            io.root,
                            SimpleNamespace(),
                            supplied_projection,
                            supplied_limits,
                            clock_ns=io.clock,
                        )
                    self.assertEqual(raised.exception.reason_code, "invalid_contract")
            error = PinnedReadRefused("unrecognized-neutral-reason")
            self.assertEqual(error.reason_code, "invalid_contract")
            self.assertEqual(str(error), "invalid_contract")
            opened.assert_not_called()
            hashed.assert_not_called()
        self.assertFalse(io.events)

    def test_budget_rejects_malformed_byte_counts_and_records_resource_overrun(self) -> None:
        for method_name, values in (
            ("before_read", (True, 0, -1, 1.5, None)),
            ("record_read", (True, -1, 1.5, None)),
        ):
            for value in values:
                with self.subTest(method=method_name, value=value):
                    io = _LoggedPinnedIO()
                    budget = self.budget(io)
                    with self.assertRaises(PinnedReadRefused) as raised:
                        getattr(budget, method_name)(value)
                    self.assertEqual(raised.exception.reason_code, "invalid_contract")
                    self.assertEqual((budget.source_opens, budget.total_read_bytes), (0, 0))
                    self.assertFalse(io.events)
        io = _LoggedPinnedIO()
        budget = self.budget(io, max_total_read_bytes=2)
        with self.assertRaises(PinnedReadRefused) as raised:
            budget.record_read(3)
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(budget.total_read_bytes, 3)
        self.assertFalse(io.open_paths)

    def test_budget_invalid_clock_results_refuse_before_acquisition_accounting(self) -> None:
        for value in (True, -1, 1.5, math.inf, None):
            with self.subTest(clock_value=value):
                io = _LoggedPinnedIO()
                budget_type = type(self.budget(io))

                def invalid_clock(result=value, events=io.events):
                    events.append("clock")
                    return result

                budget = budget_type(self.limits, clock_ns=invalid_clock)
                with self.assertRaises(PinnedReadRefused) as raised:
                    budget.before_open()
                self.assertEqual(raised.exception.reason_code, "invalid_contract")
                self.assertEqual(io.events, ["clock"])
                self.assertEqual((budget.source_opens, budget.total_read_bytes), (0, 0))
                self.assertFalse(io.open_paths)

    def test_budget_reuses_immutable_limits_and_counts_actual_bytes(self) -> None:
        io = _LoggedPinnedIO()
        budget = self.budget(io, max_total_read_bytes=7)
        with self.assertRaises(FrozenInstanceError):
            budget.limits.max_total_read_bytes = 8
        self.assertEqual((budget.source_opens, budget.total_read_bytes), (0, 0))
        budget.before_open()
        self.assertEqual(budget.source_opens, 1)
        self.assertEqual(budget.before_read(9), 7)
        budget.record_read(2)
        self.assertEqual(budget.total_read_bytes, 2)
        self.assertEqual(budget.before_read(9), 5)
        budget.record_read(5)
        with self.assertRaises(PinnedReadRefused) as raised:
            budget.before_read(1)
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(budget.total_read_bytes, 7)

    def test_budget_expired_deadline_refuses_each_operation_without_counters(self) -> None:
        for operation in ("before_open", "before_read", "check_deadline"):
            with self.subTest(operation=operation):
                io = _LoggedPinnedIO()
                budget = self.budget(io)
                io.expired = True
                with self.assertRaises(PinnedReadRefused) as raised:
                    method = getattr(budget, operation)
                    method(1) if operation == "before_read" else method()
                self.assertEqual(raised.exception.reason_code, "deadline_exceeded")
                self.assertEqual((budget.source_opens, budget.total_read_bytes), (0, 0))
                self.assertFalse(io.open_paths)

    def test_pinned_bytes_invalid_declarations_refuse_before_acquisition(self) -> None:
        cases = (
            (Path("relative"), "nested/neutral.parquet", "a" * 64, 100),
            (Path("/sandbox/../escape"), "nested/neutral.parquet", "a" * 64, 100),
            (Path("/sandbox"), "../neutral.parquet", "a" * 64, 100),
            (Path("/sandbox"), "/neutral.parquet", "a" * 64, 100),
            (Path("/sandbox"), "nested//neutral.parquet", "a" * 64, 100),
            (Path("/sandbox"), "neutral.parquet", "A" * 64, 100),
            (Path("/sandbox"), "neutral.parquet", "a" * 63, 100),
            (Path("/sandbox"), "neutral.parquet", "a" * 64, 0),
            (Path("/sandbox"), "neutral.parquet", "a" * 64, True),
        )
        for root, path, digest, maximum in cases:
            with self.subTest(root=root, path=path, digest=digest, maximum=maximum):
                io = _LoggedPinnedIO()
                budget = self.budget(io)
                with io, self.assertRaises(PinnedReadRefused) as raised:
                    identity = io.module.PinnedParquetFile(path, digest)
                    io.module._pinned_bytes(root, identity, maximum, budget=budget)
                self.assertEqual(raised.exception.reason_code, "invalid_contract")
                self.assertFalse(io.open_paths)
                self.assertFalse(io.requests)

    def test_pinned_bytes_counts_every_attempt_and_refuses_before_open_limit(self) -> None:
        for allowance in (1, 2, 3):
            with self.subTest(allowance=allowance):
                io = _LoggedPinnedIO()
                budget = self.budget(io, max_source_opens=allowance)
                with io, self.assertRaises(PinnedReadRefused) as raised:
                    io.module._pinned_bytes(io.root, io.identity, 100, budget=budget)
                self.assertEqual(raised.exception.reason_code, "resource_limit")
                self.assertEqual(budget.source_opens, allowance)
                self.assertEqual(len(io.open_paths), allowance)
                self.assertFalse(io.requests)
                self.assert_closed(io)

    def test_pinned_bytes_failed_acquisitions_are_counted_and_cleanup_is_complete(self) -> None:
        for failed_attempt in (1, 2, 3, 4):
            with self.subTest(failed_attempt=failed_attempt):
                io = _LoggedPinnedIO()
                io.fail_open = failed_attempt
                budget = self.budget(io)
                with io, self.assertRaisesRegex(OSError, "neutral acquisition failure"):
                    io.module._pinned_bytes(io.root, io.identity, 100, budget=budget)
                self.assertEqual(budget.source_opens, failed_attempt)
                self.assertFalse(io.requests)
                self.assert_closed(io)

    def test_pinned_bytes_size_and_cumulative_caps_refuse_before_payload_read(self) -> None:
        for bound in ("maximum", "max_source_size_bytes", "max_total_read_bytes", "remaining"):
            with self.subTest(bound=bound):
                io = _LoggedPinnedIO()
                size = len(io.payload)
                changes = {bound: size - 1} if bound.startswith("max_") else {}
                budget = self.budget(io, **changes)
                maximum = size - 1 if bound == "maximum" else 100
                if bound == "remaining":
                    budget.record_read(100 - size + 1)
                with io, self.assertRaises(PinnedReadRefused) as raised:
                    io.module._pinned_bytes(io.root, io.identity, maximum, budget=budget)
                self.assertEqual(raised.exception.reason_code, "resource_limit")
                self.assertEqual(io.fstat_calls, 1)
                self.assertFalse(io.requests)
                self.assertFalse(io.hashed)
                self.assert_closed(io)

    def test_pinned_bytes_short_reads_share_exact_allowance_without_growth_probe(self) -> None:
        io = _LoggedPinnedIO()
        io.short_reads = [2, 3]
        size = len(io.payload)
        budget = self.budget(io, max_total_read_bytes=size, max_source_opens=4)
        with io:
            result = io.module._pinned_bytes(io.root, io.identity, 100, budget=budget)
        self.assertEqual(result, io.payload)
        self.assertEqual(io.requests, [size, size - 2, size - 5])
        self.assertEqual(budget.total_read_bytes, size)
        self.assertEqual(budget.source_opens, 4)
        self.assertEqual(io.open_paths, ["/", "sandbox", "nested", "neutral.parquet"])
        self.assertEqual(len(io.hashed), 1)
        self.assertIs(result, io.hashed[0])
        self.assert_closed(io)

    def test_pinned_bytes_deadline_expiry_prevents_next_work_and_closes_descriptors(self) -> None:
        for event, forbidden in (
            ("initial", "open:1"),
            ("open:1", "open:2"),
            ("open:4", "fstat:1"),
            ("fstat:1", "read:1"),
            ("read:1", "read:2"),
            ("fstat:2", "stat"),
            ("stat", "hash"),
            ("hash", None),
        ):
            with self.subTest(expiry=event):
                io = _LoggedPinnedIO()
                io.expire_after = event
                io.short_reads = [2]
                budget = self.budget(io)
                io.expired = event == "initial"
                with io, self.assertRaises(PinnedReadRefused) as raised:
                    io.module._pinned_bytes(io.root, io.identity, 100, budget=budget)
                self.assertEqual(raised.exception.reason_code, "deadline_exceeded")
                if forbidden is not None:
                    self.assertNotIn(forbidden, io.events)
                else:
                    self.assertEqual(len(io.hashed), 1)
                self.assert_closed(io)

    def test_pinned_bytes_integrity_failures_preserve_corruption_taxonomy(self) -> None:
        for anomaly in ("nonregular", "linked", "empty", "truncated", "grew", "fd", "path"):
            with self.subTest(anomaly=anomaly):
                io = _LoggedPinnedIO()
                budget = self.budget(io)
                if anomaly == "nonregular":
                    io.before = io.metadata(st_mode=stat.S_IFDIR | 0o700)
                elif anomaly == "linked":
                    io.before = io.metadata(st_nlink=2)
                elif anomaly == "empty":
                    io.before = io.metadata(st_size=0)
                elif anomaly == "truncated":
                    io.payload = io.payload[:-1]
                elif anomaly == "grew":
                    io.after = io.metadata(st_size=len(io.payload) + 1)
                elif anomaly == "fd":
                    io.after = io.metadata(st_ino=999)
                else:
                    io.located = io.metadata(st_ino=999)
                with io, self.assertRaises(io.module.TimestampMetadataRefused) as raised:
                    io.module._pinned_bytes(io.root, io.identity, 100, budget=budget)
                self.assertIsInstance(raised.exception, DataCorruptionError)
                self.assertNotIsInstance(raised.exception, PinnedReadRefused)
                self.assertFalse(io.hashed)
                if anomaly in ("nonregular", "linked", "empty"):
                    self.assertFalse(io.requests)
                elif anomaly in ("grew", "fd", "path"):
                    self.assertEqual(io.requests, [len(io.payload)])
                self.assert_closed(io)

    def test_metadata_consumer_receives_same_hashed_payload_without_reacquisition(self) -> None:
        io = _LoggedPinnedIO()
        budget = self.budget(io, max_total_read_bytes=len(io.payload))
        with io:
            original = io.module._pinned_bytes
            expected = io.module.TimestampFileMetadata(io.identity, 1, ())
            with (
                patch.object(
                    io.module,
                    "_pinned_bytes",
                    side_effect=lambda root, identity, maximum: original(
                        root, identity, maximum, budget=budget
                    ),
                ) as helper,
                patch.object(io.module, "_inspect_file", return_value=expected) as parser,
            ):
                result = io.module.inspect_timestamp_metadata(
                    io.root, (io.identity,), max_files=1, max_file_bytes=100, max_row_groups=1
                )
            helper.assert_called_once_with(io.root, io.identity, 100)
            parser.assert_called_once()
            self.assertIs(parser.call_args.args[0], io.hashed[0])
            self.assertEqual(result, (expected,))
        self.assertEqual(len(io.hashed), 1)
        self.assertEqual(len(io.open_paths), 4)
        self.assertEqual(io.requests, [len(io.payload)])
        self.assert_closed(io)

    def test_metadata_consumer_digest_mismatch_never_reaches_parser(self) -> None:
        io = _LoggedPinnedIO()
        io.digest = "0" * 64
        budget = self.budget(io)
        with io:
            original = io.module._pinned_bytes
            with (
                patch.object(
                    io.module,
                    "_pinned_bytes",
                    side_effect=lambda root, identity, maximum: original(
                        root, identity, maximum, budget=budget
                    ),
                ),
                patch.object(io.module, "_inspect_file") as parser,
                self.assertRaises(io.module.TimestampMetadataRefused),
            ):
                io.module.inspect_timestamp_metadata(
                    io.root, (io.identity,), max_files=1, max_file_bytes=100, max_row_groups=1
                )
            parser.assert_not_called()
        self.assertEqual(len(io.hashed), 1)
        self.assertEqual(len(io.open_paths), 4)
        self.assert_closed(io)

    def test_legacy_pinned_bytes_preserves_growth_probe_and_integrity_refusal(self) -> None:
        for growth in (False, True):
            with self.subTest(growth=growth):
                io = _LoggedPinnedIO()
                original = io.payload
                if growth:
                    io.payload += b"!"
                with io:
                    if growth:
                        with self.assertRaises(io.module.TimestampMetadataRefused):
                            io.module._pinned_bytes(io.root, io.identity, 100)
                    else:
                        result = io.module._pinned_bytes(io.root, io.identity, 100)
                        self.assertEqual(result, original)
                self.assertEqual(io.requests, [len(original), 1])
                self.assert_closed(io)


class CompactWireCursorTests(unittest.TestCase):
    """Neutral wire declarations exercise bounds without a file or native parser."""

    def api(self):
        names = (
            "_CompactWireLimits",
            "_CompactWireBudget",
            "_CompactWireCursor",
            "_CompactWireMalformed",
        )
        resolved = {}
        for name in names:
            value = getattr(bounded_parquet, name, None)
            self.assertIsNotNone(value, f"canonical wire contract requires {name}")
            resolved[name] = value
        return SimpleNamespace(**resolved)

    def budget(self, **changes: int):
        api = self.api()
        values = {
            "max_input_bytes": 256,
            "max_object_bytes": 128,
            "max_header_bytes": 64,
            "max_depth": 4,
            "max_fields_per_struct": 8,
            "max_total_fields": 16,
            "max_items_per_container": 32,
            "max_total_items": 48,
            "max_string_bytes": 32,
            "max_total_string_bytes": 48,
            "max_varint_bytes": 10,
            "max_total_wire_bytes": 256,
            "max_work_units": 1024,
            "max_copied_payload_bytes": 48,
        }
        values.update(changes)
        return api._CompactWireBudget(api._CompactWireLimits(**values))

    def cursor(self, payload: bytes, *, budget=None, kind: str = "page_header"):
        api = self.api()
        if budget is None:
            budget = self.budget()
        return api._CompactWireCursor(payload, 0, len(payload), budget, kind=kind), budget

    def assert_refused(self, error_type, reason: str, operation) -> None:
        with self.assertRaises(error_type) as raised:
            operation()
        self.assertEqual(raised.exception.reason_code, reason)
        self.assertEqual(str(raised.exception), reason)

    def assert_field(self, field, field_id: int, wire_type: str, boolean=None) -> None:
        self.assertEqual(
            (field.field_id, field.wire_type, field.boolean), (field_id, wire_type, boolean)
        )

    def test_wire_declarations_reject_invalid_limits_before_boundary_calls(self) -> None:
        api = self.api()
        limits = self.budget().limits
        with (
            patch.object(os, "open", side_effect=AssertionError("unexpected open")) as opened,
            patch.object(
                hashlib, "sha256", side_effect=AssertionError("unexpected hash")
            ) as hashed,
        ):
            for name in limits.__dataclass_fields__:
                for invalid in (True, False, 0, -1, 1.5, math.inf, "2", None):
                    with self.subTest(limit=name, invalid=invalid):
                        self.assert_refused(
                            PinnedReadRefused,
                            "invalid_contract",
                            lambda name=name, invalid=invalid: replace(limits, **{name: invalid}),
                        )
            self.assert_refused(
                PinnedReadRefused, "invalid_contract", lambda: replace(limits, max_varint_bytes=11)
            )
            self.assert_refused(
                PinnedReadRefused,
                "invalid_contract",
                lambda: api._CompactWireBudget(SimpleNamespace()),
            )
            opened.assert_not_called()
            hashed.assert_not_called()
        with self.assertRaises(FrozenInstanceError):
            limits.max_depth = 5

    def test_wire_cursor_validates_spans_types_and_field_declarations(self) -> None:
        api = self.api()
        budget = self.budget()
        for payload, start, stop, supplied_budget, kind in (
            (bytearray(b"x"), 0, 1, budget, "footer"),
            ("x", 0, 1, budget, "footer"),
            (b"x", True, 1, budget, "footer"),
            (b"x", 0, False, budget, "footer"),
            (b"x", 0.0, 1, budget, "footer"),
            (b"x", -1, 1, budget, "footer"),
            (b"x", 1, 0, budget, "footer"),
            (b"x", 0, 2, budget, "footer"),
            (b"x", 0, 1, SimpleNamespace(), "footer"),
            (b"x", 0, 1, budget, "other"),
        ):
            with self.subTest(payload=payload, start=start, stop=stop, kind=kind):
                self.assert_refused(
                    PinnedReadRefused,
                    "invalid_contract",
                    lambda payload=payload, start=start, stop=stop, supplied_budget=supplied_budget, kind=kind: (
                        api._CompactWireCursor(payload, start, stop, supplied_budget, kind=kind)
                    ),
                )
        self.assertEqual(budget.total_wire_bytes, 0)
        cursor = api._CompactWireCursor(b"!\x02?", 1, 2, budget, kind="footer")
        self.assertEqual(cursor.position, 1)
        self.assertEqual(cursor.read_i32(), 1)
        self.assertEqual(cursor.position, 2)
        cursor.finish(require_exhausted=True)
        for declaration in (
            {True: "i32"},
            {0: "i32"},
            {-1: "i32"},
            {32768: "i32"},
            {1: "unknown"},
            {1: 5},
            [(1, "i32")],
        ):
            with self.subTest(declaration=declaration):
                cursor, _ = self.cursor(b"\x15\x02\x00")
                cursor.begin_struct()
                self.assert_refused(
                    PinnedReadRefused,
                    "invalid_contract",
                    lambda cursor=cursor, declaration=declaration: cursor.read_field(declaration),
                )
                self.assertEqual(cursor.position, 0)
        empty, _ = self.cursor(b"")
        self.assert_refused(api._CompactWireMalformed, "truncated_wire", empty.read_i32)

    def test_wire_signed_integers_decode_canonical_width_boundaries(self) -> None:
        cases = (
            ("read_i16", b"\x00", 0),
            ("read_i16", b"\x01", -1),
            ("read_i16", b"\x7e", 63),
            ("read_i16", b"\x80\x01", 64),
            ("read_i16", b"\xfe\xff\x03", 32767),
            ("read_i16", b"\xff\xff\x03", -32768),
            ("read_i32", b"\xfe\xff\xff\xff\x0f", 2147483647),
            ("read_i32", b"\xff\xff\xff\xff\x0f", -2147483648),
            ("read_i64", b"\xfe" + b"\xff" * 8 + b"\x01", 9223372036854775807),
            ("read_i64", b"\xff" * 9 + b"\x01", -9223372036854775808),
        )
        for method, payload, expected in cases:
            with self.subTest(method=method, payload=payload):
                cursor, budget = self.cursor(payload)
                self.assertEqual(getattr(cursor, method)(), expected)
                self.assertEqual(cursor.position, len(payload))
                self.assertEqual(budget.total_wire_bytes, len(payload))
                self.assertEqual(budget.work_units, len(payload))

    def test_wire_varints_refuse_nonminimal_truncated_and_overflow_forms(self) -> None:
        api = self.api()
        for method, payload, reason in (
            ("read_i16", b"\x80\x00", "noncanonical_wire"),
            ("read_i32", b"\x81\x00", "noncanonical_wire"),
            ("read_i64", b"\x80", "truncated_wire"),
            ("read_i16", b"\xff\xff\x04", "invalid_wire"),
            ("read_i32", b"\xff\xff\xff\xff\x10", "invalid_wire"),
            ("read_i64", b"\xff" * 9 + b"\x02", "invalid_wire"),
            ("read_i16", b"\x80\x80\x80\x01", "invalid_wire"),
            ("read_i64", b"\x80" * 10 + b"\x01", "invalid_wire"),
        ):
            with self.subTest(method=method, payload=payload):
                cursor, _ = self.cursor(payload)
                self.assert_refused(api._CompactWireMalformed, reason, getattr(cursor, method))
        cursor, budget = self.cursor(b"\x80\x01", budget=self.budget(max_varint_bytes=1))
        self.assert_refused(PinnedReadRefused, "resource_limit", cursor.read_i32)
        self.assertEqual(cursor.position, 1)
        self.assertEqual(budget.total_wire_bytes, 1)
        self.assertEqual(cursor.read_byte(), 1)
        self.assertEqual(budget.total_wire_bytes, 2)

    def test_wire_fields_restore_parent_state_across_nested_structs(self) -> None:
        cursor, budget = self.cursor(b"\x15\x02\x0c\x28\x15\x04\x00\x15\x06\x00")
        parent = {1: "i32", 20: "struct", 21: "i32"}
        child = {1: "i32"}
        cursor.begin_struct()
        self.assert_field(cursor.read_field(parent), 1, "i32")
        self.assertEqual(cursor.read_i32(), 1)
        self.assert_field(cursor.read_field(parent), 20, "struct")
        cursor.begin_struct()
        self.assert_field(cursor.read_field(child), 1, "i32")
        self.assertEqual(cursor.read_i32(), 2)
        self.assertIsNone(cursor.read_field(child))
        cursor.end_struct()
        self.assert_field(cursor.read_field(parent), 21, "i32")
        self.assertEqual(cursor.read_i32(), 3)
        self.assertIsNone(cursor.read_field(parent))
        cursor.end_struct()
        cursor.finish(require_exhausted=True)
        self.assertEqual(cursor.position, 10)
        self.assertEqual(budget.total_fields, 4)
        descending, _ = self.cursor(b"\x05\x28\x00\x05\x02\x00\x00")
        descending.begin_struct()
        allowed = {1: "i32", 20: "i32"}
        self.assert_field(descending.read_field(allowed), 20, "i32")
        self.assertEqual(descending.read_i32(), 0)
        self.assert_field(descending.read_field(allowed), 1, "i32")
        self.assertEqual(descending.read_i32(), 0)
        self.assertIsNone(descending.read_field(allowed))
        descending.end_struct()

    def test_wire_field_headers_refuse_aliases_duplicates_and_type_disagreement(self) -> None:
        api = self.api()
        for payload, fields, reason, consumed in (
            (b"\x10x", {1: "i32"}, "noncanonical_wire", 1),
            (b"\x05\x02x", {1: "i32"}, "noncanonical_wire", 2),
            (b"\x15x", {2: "i32"}, "schema_mismatch", 1),
            (b"\x14x", {1: "i32"}, "schema_mismatch", 1),
            (b"\x1fx", {1: "i32"}, "invalid_wire", 1),
        ):
            with self.subTest(payload=payload):
                cursor, _ = self.cursor(payload)
                cursor.begin_struct()
                self.assert_refused(
                    api._CompactWireMalformed,
                    reason,
                    lambda cursor=cursor, fields=fields: cursor.read_field(fields),
                )
                self.assertEqual(cursor.position, consumed)
        cursor, _ = self.cursor(b"\x15\x00\x05\x02x")
        cursor.begin_struct()
        cursor.read_field({1: "i32"})
        cursor.read_i32()
        self.assert_refused(
            api._CompactWireMalformed, "schema_mismatch", lambda: cursor.read_field({1: "i32"})
        )
        self.assertEqual(cursor.position, 4)
        cursor, _ = self.cursor(b"\x05\xfe\xff\x03\x00\x15x")
        cursor.begin_struct()
        cursor.read_field({32767: "i32"})
        cursor.read_i32()
        self.assert_refused(
            api._CompactWireMalformed, "invalid_wire", lambda: cursor.read_field({32767: "i32"})
        )
        self.assertEqual(cursor.position, 6)

    def test_wire_booleans_distinguish_embedded_values_and_standalone_bytes(self) -> None:
        api = self.api()
        cursor, budget = self.cursor(b"\x11\x12\x00")
        fields = {1: "bool", 2: "bool"}
        cursor.begin_struct()
        for field_id, expected in ((1, True), (2, False)):
            self.assert_field(cursor.read_field(fields), field_id, "bool", expected)
            position = cursor.position
            charged = budget.total_wire_bytes
            self.assertIs(cursor.read_bool(), expected)
            self.assertEqual(cursor.position, position)
            self.assertEqual(budget.total_wire_bytes, charged)
        self.assertIsNone(cursor.read_field(fields))
        cursor.end_struct()
        cursor.finish(require_exhausted=True)
        for value, expected in ((b"\x01", True), (b"\x02", False)):
            cursor, _ = self.cursor(value)
            self.assertIs(cursor.read_bool(), expected)
            self.assertEqual(cursor.position, 1)
            self.assert_refused(api._CompactWireMalformed, "truncated_wire", cursor.read_bool)
        for value in (b"\x00", b"\x03", b"\xff"):
            with self.subTest(value=value):
                cursor, _ = self.cursor(value)
                self.assert_refused(api._CompactWireMalformed, "invalid_wire", cursor.read_bool)
        cursor, _ = self.cursor(b"\x11\x00")
        cursor.begin_struct()
        cursor.read_field({1: "bool"})
        self.assert_refused(
            api._CompactWireMalformed, "invalid_wire", lambda: cursor.read_field({1: "bool"})
        )
        self.assertEqual(cursor.position, 1)

    def test_wire_list_headers_validate_canonical_counts_and_element_types(self) -> None:
        api = self.api()
        for payload, count in ((b"\x05", 0), (b"\xe5", 14), (b"\xf5\x0f", 15)):
            with self.subTest(count=count):
                cursor, budget = self.cursor(payload)
                self.assertEqual(cursor.read_list_header("i32"), ("i32", count))
                self.assertEqual(budget.total_items, count)
                self.assertEqual(budget.work_units, len(payload) + count)
        for payload, reason in (
            (b"\xf5\x0e", "noncanonical_wire"),
            (b"\xf5\x8f\x00", "noncanonical_wire"),
            (b"\xf5\xff\xff\xff\xff\x08", "invalid_wire"),
            (b"\xf5\x80", "truncated_wire"),
            (b"\x04", "schema_mismatch"),
            (b"\x0a", "invalid_wire"),
            (b"\x0b", "invalid_wire"),
            (b"\x0d", "invalid_wire"),
            (b"\x0f", "invalid_wire"),
        ):
            with self.subTest(payload=payload):
                cursor, _ = self.cursor(payload)
                self.assert_refused(
                    api._CompactWireMalformed,
                    reason,
                    lambda cursor=cursor: cursor.read_list_header("i32"),
                )
        cursor, _ = self.cursor(b"\x05")
        self.assert_refused(
            PinnedReadRefused, "invalid_contract", lambda: cursor.read_list_header("unknown")
        )
        self.assertEqual(cursor.position, 0)

    def test_wire_container_limits_reserve_counts_before_body_consumption(self) -> None:
        cursor, budget = self.cursor(b"\x35abc", budget=self.budget(max_items_per_container=2))
        self.assert_refused(
            PinnedReadRefused, "resource_limit", lambda: cursor.read_list_header("i32")
        )
        self.assertEqual(cursor.position, 1)
        self.assertEqual(budget.total_items, 0)
        budget = self.budget(max_total_items=3)
        first, _ = self.cursor(b"\x25", budget=budget)
        second, _ = self.cursor(b"\x25ab", budget=budget)
        self.assertEqual(first.read_list_header("i32"), ("i32", 2))
        self.assert_refused(
            PinnedReadRefused, "resource_limit", lambda: second.read_list_header("i32")
        )
        self.assertEqual(second.position, 1)
        self.assertEqual(budget.total_items, 2)
        self.assertEqual(budget.work_units, 4)

    def test_wire_binary_values_account_exact_copies_and_precheck_payload_limits(self) -> None:
        api = self.api()
        cursor, budget = self.cursor(b"\x00\x03abc")
        self.assertEqual(cursor.read_binary(), b"")
        self.assertEqual(cursor.read_binary(), b"abc")
        self.assertEqual(cursor.position, 5)
        self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (3, 3))
        self.assertEqual((budget.total_wire_bytes, budget.work_units), (5, 5))
        for limit in ("max_string_bytes", "max_total_string_bytes", "max_copied_payload_bytes"):
            with self.subTest(limit=limit):
                exact, _ = self.cursor(b"\x02ab", budget=self.budget(**{limit: 2}))
                self.assertEqual(exact.read_binary(), b"ab")
                cursor, budget = self.cursor(b"\x03abc", budget=self.budget(**{limit: 2}))
                self.assert_refused(PinnedReadRefused, "resource_limit", cursor.read_binary)
                self.assertEqual(cursor.position, 1)
                self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (0, 0))
                self.assertEqual(budget.total_wire_bytes, 1)
        for payload, reason in (
            (b"\x03ab", "truncated_wire"),
            (b"\x80\x00", "noncanonical_wire"),
            (b"\xff\xff\xff\xff\x08", "invalid_wire"),
        ):
            with self.subTest(payload=payload):
                cursor, budget = self.cursor(payload)
                self.assert_refused(api._CompactWireMalformed, reason, cursor.read_binary)
                self.assertEqual(budget.copied_payload_bytes, 0)
        for limit in ("max_total_string_bytes", "max_copied_payload_bytes"):
            with self.subTest(shared_limit=limit):
                budget = self.budget(**{limit: 3})
                first, _ = self.cursor(b"\x02ab", budget=budget)
                second, _ = self.cursor(b"\x02cd", budget=budget)
                self.assertEqual(first.read_binary(), b"ab")
                self.assert_refused(PinnedReadRefused, "resource_limit", second.read_binary)
                self.assertEqual(second.position, 1)
                self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (2, 2))

    def test_wire_signed_bytes_and_little_endian_doubles_have_fixed_widths(self) -> None:
        api = self.api()
        cursor, budget = self.cursor(b"\x00\x7f\x80\xff")
        self.assertEqual([cursor.read_byte() for _ in range(4)], [0, 127, -128, -1])
        self.assertEqual((cursor.position, budget.total_wire_bytes), (4, 4))
        cursor, _ = self.cursor(b"\x00\x00\x00\x00\x00\x00\xf0\x3f")
        self.assertEqual(cursor.read_double(), 1.0)
        self.assertEqual(cursor.position, 8)
        short, budget = self.cursor(b"\x00" * 7)
        self.assert_refused(api._CompactWireMalformed, "truncated_wire", short.read_double)
        self.assertEqual((short.position, budget.total_wire_bytes), (0, 0))

    def test_wire_byte_and_work_limits_refuse_before_the_next_read(self) -> None:
        for limit in (
            "max_object_bytes",
            "max_header_bytes",
            "max_total_wire_bytes",
            "max_work_units",
        ):
            with self.subTest(limit=limit):
                cursor, budget = self.cursor(b"\x00\x01\x02", budget=self.budget(**{limit: 2}))
                self.assertEqual(cursor.read_byte(), 0)
                self.assertEqual(cursor.read_byte(), 1)
                self.assert_refused(PinnedReadRefused, "resource_limit", cursor.read_byte)
                self.assertEqual(cursor.position, 2)
                self.assertEqual((budget.total_wire_bytes, budget.work_units), (2, 2))
        api = self.api()
        budget = self.budget(max_input_bytes=2)
        self.assert_refused(
            PinnedReadRefused,
            "resource_limit",
            lambda: api._CompactWireCursor(b"abc", 0, 1, budget, kind="footer"),
        )
        self.assertEqual(budget.total_wire_bytes, 0)
        budget = self.budget(max_total_wire_bytes=2)
        first, _ = self.cursor(b"\x00\x01", budget=budget)
        second, _ = self.cursor(b"\x02", budget=budget)
        first.read_byte()
        first.read_byte()
        self.assert_refused(PinnedReadRefused, "resource_limit", second.read_byte)
        self.assertEqual(second.position, 0)
        self.assertEqual(budget.total_wire_bytes, 2)

    def test_wire_depth_field_and_declaration_work_limits_bound_structural_state(self) -> None:
        cursor, budget = self.cursor(b"\x00\x00", budget=self.budget(max_depth=2))
        cursor.begin_struct()
        cursor.begin_struct()
        self.assert_refused(PinnedReadRefused, "resource_limit", cursor.begin_struct)
        self.assertEqual(cursor.position, 0)
        self.assertIsNone(cursor.read_field({}))
        cursor.end_struct()
        self.assertIsNone(cursor.read_field({}))
        cursor.end_struct()
        cursor.finish(require_exhausted=True)
        cursor, budget = self.cursor(b"\x15\x00\x15x", budget=self.budget(max_fields_per_struct=1))
        cursor.begin_struct()
        cursor.read_field({1: "i32"})
        cursor.read_i32()
        self.assert_refused(
            PinnedReadRefused, "resource_limit", lambda: cursor.read_field({2: "i32"})
        )
        self.assertLessEqual(cursor.position, 3)
        self.assertEqual(budget.total_fields, 1)
        budget = self.budget(max_total_fields=1)
        first, _ = self.cursor(b"\x15\x00\x00", budget=budget)
        first.begin_struct()
        first.read_field({1: "i32"})
        first.read_i32()
        first.read_field({1: "i32"})
        first.end_struct()
        second, _ = self.cursor(b"\x15x", budget=budget)
        second.begin_struct()
        self.assert_refused(
            PinnedReadRefused, "resource_limit", lambda: second.read_field({1: "i32"})
        )
        self.assertLessEqual(second.position, 1)
        self.assertEqual(budget.total_fields, 1)
        cursor, budget = self.cursor(b"\x15x", budget=self.budget(max_work_units=1))
        cursor.begin_struct()
        self.assert_refused(
            PinnedReadRefused, "resource_limit", lambda: cursor.read_field({1: "i32"})
        )
        self.assertEqual((cursor.position, budget.work_units), (0, 1))

    def test_wire_finish_distinguishes_footer_exhaustion_and_header_boundary(self) -> None:
        api = self.api()
        header, _ = self.cursor(b"\x00tail")
        header.begin_struct()
        self.assertIsNone(header.read_field({}))
        header.end_struct()
        header.finish(require_exhausted=False)
        self.assertEqual(header.position, 1)
        footer, _ = self.cursor(b"\x00tail", kind="footer")
        footer.begin_struct()
        footer.read_field({})
        footer.end_struct()
        self.assert_refused(
            api._CompactWireMalformed, "invalid_wire", lambda: footer.finish(require_exhausted=True)
        )
        self.assertEqual(footer.position, 1)
        for operation in ("end_struct", "finish"):
            with self.subTest(operation=operation):
                cursor, _ = self.cursor(b"\x00")
                cursor.begin_struct()
                callback = (
                    cursor.end_struct
                    if operation == "end_struct"
                    else lambda cursor=cursor: cursor.finish(require_exhausted=False)
                )
                self.assert_refused(api._CompactWireMalformed, "invalid_wire", callback)
                self.assertEqual(cursor.position, 0)
        cursor, _ = self.cursor(b"\x11\x00")
        cursor.begin_struct()
        cursor.read_field({1: "bool"})
        self.assert_refused(
            api._CompactWireMalformed,
            "invalid_wire",
            lambda: cursor.finish(require_exhausted=False),
        )
        self.assertEqual(cursor.position, 1)

    def test_wire_operations_do_not_invoke_filesystem_or_hash_boundaries(self) -> None:
        api = self.api()
        with (
            patch.object(os, "open", side_effect=AssertionError("unexpected open")) as opened,
            patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected read")) as read,
            patch.object(
                hashlib, "sha256", side_effect=AssertionError("unexpected hash")
            ) as hashed,
        ):
            cursor, _ = self.cursor(b"\x15\x02\x00")
            cursor.begin_struct()
            cursor.read_field({1: "i32"})
            self.assertEqual(cursor.read_i32(), 1)
            cursor.read_field({1: "i32"})
            cursor.end_struct()
            cursor.finish(require_exhausted=True)
            malformed, _ = self.cursor(b"\x80")
            with self.assertRaises(api._CompactWireMalformed) as raised:
                malformed.read_i32()
            self.assertIsInstance(raised.exception, DataCorruptionError)
            self.assertNotIsInstance(raised.exception, PinnedReadRefused)
            self.assertEqual(str(raised.exception), "truncated_wire")
            opened.assert_not_called()
            read.assert_not_called()
            hashed.assert_not_called()

    def test_wire_boolean_list_tags_use_canonical_writer_form(self) -> None:
        api = self.api()
        cursor, budget = self.cursor(b"\x11\x02")
        self.assertEqual(cursor.read_list_header("bool"), ("bool", 1))
        self.assertFalse(cursor.read_bool())
        self.assertEqual(budget.total_items, 1)
        cursor.finish(require_exhausted=True)
        for payload in (b"\x02", b"\x12\x01"):
            with self.subTest(payload=payload):
                cursor, _ = self.cursor(payload)
                self.assert_refused(
                    api._CompactWireMalformed,
                    "noncanonical_wire",
                    lambda cursor=cursor: cursor.read_list_header("bool"),
                )
                self.assertEqual(cursor.position, 1)

    def binary_span_reader(self, cursor):
        reader = getattr(cursor, "read_binary_span", None)
        self.assertTrue(callable(reader), "binary spans require callable read_binary_span")
        return reader

    def test_binary_span_preserves_payload_identity_and_absolute_body_offsets(self) -> None:
        api = self.api()
        payload = b"xx\x03abczz"
        budget = self.budget(max_copied_payload_bytes=1)
        cursor = api._CompactWireCursor(payload, 2, 6, budget, kind="footer")
        reader = self.binary_span_reader(cursor)
        with patch.object(type(cursor), "read_binary", side_effect=AssertionError("copy called")):
            span = reader()
        self.assertIs(type(span), tuple)
        self.assertEqual(span, (3, 6))
        for offset in span:
            self.assertIs(type(offset), int)
        self.assertIs(cursor.payload, payload)
        self.assertEqual(cursor.position, 6)
        self.assertEqual((budget.total_wire_bytes, budget.work_units), (4, 4))
        self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (3, 0))

    def test_binary_span_empty_and_sequential_bodies_have_exact_offsets(self) -> None:
        cursor, budget = self.cursor(b"\x02ab\x00\x01z")
        reader = self.binary_span_reader(cursor)
        self.assertEqual(reader(), (1, 3))
        self.assertEqual(reader(), (4, 4))
        self.assertEqual(reader(), (5, 6))
        self.assertEqual(cursor.position, 6)
        self.assertEqual((budget.total_wire_bytes, budget.work_units), (6, 6))
        self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (3, 0))

    def test_binary_span_string_limits_are_shared_and_checked_before_advancement(self) -> None:
        for limit in ("max_string_bytes", "max_total_string_bytes"):
            with self.subTest(limit=limit):
                exact, exact_budget = self.cursor(b"\x02ab", budget=self.budget(**{limit: 2}))
                self.assertEqual(self.binary_span_reader(exact)(), (1, 3))
                self.assertEqual(exact_budget.total_string_bytes, 2)
                self.assertEqual(exact_budget.copied_payload_bytes, 0)
                cursor, budget = self.cursor(b"\x03abc", budget=self.budget(**{limit: 2}))
                reader = self.binary_span_reader(cursor)
                self.assert_refused(PinnedReadRefused, "resource_limit", reader)
                self.assertEqual(cursor.position, 1)
                self.assertEqual((budget.total_wire_bytes, budget.work_units), (1, 1))
                self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (0, 0))
        budget = self.budget(max_total_string_bytes=3)
        first, _ = self.cursor(b"\x02ab", budget=budget)
        second, _ = self.cursor(b"\x02cd", budget=budget)
        first_reader = self.binary_span_reader(first)
        second_reader = self.binary_span_reader(second)
        self.assertEqual(first_reader(), (1, 3))
        self.assert_refused(PinnedReadRefused, "resource_limit", second_reader)
        self.assertEqual(second.position, 1)
        self.assertEqual((budget.total_wire_bytes, budget.work_units), (4, 4))
        self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (2, 0))

    def test_binary_span_wire_and_work_limits_reserve_whole_bodies(self) -> None:
        for limit in ("max_total_wire_bytes", "max_work_units"):
            with self.subTest(limit=limit):
                exact, exact_budget = self.cursor(b"\x02ab", budget=self.budget(**{limit: 3}))
                self.assertEqual(self.binary_span_reader(exact)(), (1, 3))
                self.assertEqual((exact_budget.total_wire_bytes, exact_budget.work_units), (3, 3))
                cursor, budget = self.cursor(b"\x03abc", budget=self.budget(**{limit: 3}))
                reader = self.binary_span_reader(cursor)
                self.assert_refused(PinnedReadRefused, "resource_limit", reader)
                self.assertEqual(cursor.position, 1)
                self.assertEqual((budget.total_wire_bytes, budget.work_units), (1, 1))
                self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (0, 0))

    def test_binary_span_object_and_header_limits_remain_distinct(self) -> None:
        for limit in ("max_object_bytes", "max_header_bytes"):
            with self.subTest(limit=limit):
                exact, exact_budget = self.cursor(b"\x02ab", budget=self.budget(**{limit: 3}))
                self.assertEqual(self.binary_span_reader(exact)(), (1, 3))
                self.assertEqual((exact_budget.total_wire_bytes, exact_budget.work_units), (3, 3))
                cursor, budget = self.cursor(b"\x03abc", budget=self.budget(**{limit: 3}))
                reader = self.binary_span_reader(cursor)
                self.assert_refused(PinnedReadRefused, "resource_limit", reader)
                self.assertEqual(cursor.position, 1)
                self.assertEqual((budget.total_wire_bytes, budget.work_units), (1, 1))
                self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (0, 0))
        footer, budget = self.cursor(
            b"\x03abc", budget=self.budget(max_header_bytes=3), kind="footer"
        )
        self.assertEqual(self.binary_span_reader(footer)(), (1, 4))
        self.assertEqual(footer.position, 4)
        self.assertEqual((budget.total_wire_bytes, budget.work_units), (4, 4))
        self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (3, 0))

    def test_binary_span_malformed_lengths_preserve_only_consumed_prefix_charges(self) -> None:
        api = self.api()
        for payload, reason, consumed in (
            (b"\x03ab", "truncated_wire", 1),
            (b"\x80\x00", "noncanonical_wire", 2),
            (b"\xff\xff\xff\xff\x08", "invalid_wire", 5),
        ):
            with self.subTest(reason=reason):
                cursor, budget = self.cursor(payload)
                reader = self.binary_span_reader(cursor)
                self.assert_refused(api._CompactWireMalformed, reason, reader)
                self.assertEqual(cursor.position, consumed)
                self.assertEqual((budget.total_wire_bytes, budget.work_units), (consumed, consumed))
                self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (0, 0))

    def test_binary_spans_share_string_accounting_without_spending_copy_allowance(self) -> None:
        budget = self.budget(max_total_string_bytes=5, max_copied_payload_bytes=2)
        first, _ = self.cursor(b"\x02ab", budget=budget)
        copying, _ = self.cursor(b"\x02cd", budget=budget)
        third, _ = self.cursor(b"\x01z", budget=budget)
        first_reader = self.binary_span_reader(first)
        third_reader = self.binary_span_reader(third)
        self.assertEqual(first_reader(), (1, 3))
        self.assertEqual(copying.read_binary(), b"cd")
        self.assertEqual(third_reader(), (1, 2))
        self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (5, 2))
        self.assertEqual((budget.total_wire_bytes, budget.work_units), (8, 8))
        refused, _ = self.cursor(b"\x01z", budget=budget)
        self.assert_refused(PinnedReadRefused, "resource_limit", refused.read_binary)
        self.assertEqual(refused.position, 1)
        self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (5, 2))
        self.assertEqual((budget.total_wire_bytes, budget.work_units), (9, 9))
        independent, budget = self.cursor(
            b"\x03abc", budget=self.budget(max_copied_payload_bytes=1)
        )
        self.assert_refused(PinnedReadRefused, "resource_limit", independent.read_binary)
        self.assertEqual(independent.position, 1)
        self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (0, 0))
        self.assertEqual((budget.total_wire_bytes, budget.work_units), (1, 1))

    def test_binary_span_exhaustion_has_no_extra_charges_or_boundary_calls(self) -> None:
        api = self.api()
        cursor, budget = self.cursor(b"\x02ab")
        reader = self.binary_span_reader(cursor)
        with (
            patch.object(os, "open", side_effect=AssertionError("unexpected open")) as opened,
            patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected read")) as read,
            patch.object(
                hashlib, "sha256", side_effect=AssertionError("unexpected hash")
            ) as hashed,
        ):
            modules_before = set(sys.modules)
            self.assertEqual(reader(), (1, 3))
            cursor.finish(require_exhausted=True)
            self.assert_refused(api._CompactWireMalformed, "truncated_wire", cursor.read_binary)
            self.assertEqual(cursor.position, 3)
            self.assertEqual((budget.total_wire_bytes, budget.work_units), (3, 3))
            self.assertEqual((budget.total_string_bytes, budget.copied_payload_bytes), (2, 0))
            self.assertEqual(set(sys.modules), modules_before)
            opened.assert_not_called()
            read.assert_not_called()
            hashed.assert_not_called()


_GrammarField = tuple[int, str, bool, str | None, tuple[int, ...] | None]
_GRAMMAR_TYPES = tuple(range(8))
_GRAMMAR_ENCODINGS = (0, 2, 3, 4, 5, 6, 7, 8, 9)
_GRAMMAR_FIELDS: dict[str, tuple[_GrammarField, ...]] = {
    "FileMetaData": (
        (1, "i32", True, None, None),
        (2, "list:struct", True, "SchemaElement", None),
        (3, "i64", True, None, None),
        (4, "list:struct", True, "RowGroup", None),
        (5, "list:struct", False, "KeyValue", None),
        (6, "binary", False, None, None),
        (7, "list:struct", False, "ColumnOrder", None),
        (8, "struct", False, "EncryptionAlgorithm", None),
        (9, "binary", False, None, None),
    ),
    "SchemaElement": (
        (1, "i32", False, None, _GRAMMAR_TYPES),
        (2, "i32", False, None, None),
        (3, "i32", False, None, (0, 1, 2)),
        (4, "binary", True, None, None),
        (5, "i32", False, None, None),
        (6, "i32", False, None, tuple(range(22))),
        (7, "i32", False, None, None),
        (8, "i32", False, None, None),
        (9, "i32", False, None, None),
        (10, "struct", False, "LogicalType", None),
    ),
    "RowGroup": (
        (1, "list:struct", True, "ColumnChunk", None),
        (2, "i64", True, None, None),
        (3, "i64", True, None, None),
        (4, "list:struct", False, "SortingColumn", None),
        (5, "i64", False, None, None),
        (6, "i64", False, None, None),
        (7, "i16", False, None, None),
    ),
    "ColumnChunk": (
        (1, "binary", False, None, None),
        (2, "i64", True, None, None),
        (3, "struct", False, "ColumnMetaData", None),
        (4, "i64", False, None, None),
        (5, "i32", False, None, None),
        (6, "i64", False, None, None),
        (7, "i32", False, None, None),
        (8, "struct", False, "ColumnCryptoMetaData", None),
        (9, "binary", False, None, None),
    ),
    "ColumnMetaData": (
        (1, "i32", True, None, _GRAMMAR_TYPES),
        (2, "list:i32", True, None, _GRAMMAR_ENCODINGS),
        (3, "list:binary", True, None, None),
        (4, "i32", True, None, tuple(range(8))),
        (5, "i64", True, None, None),
        (6, "i64", True, None, None),
        (7, "i64", True, None, None),
        (8, "list:struct", False, "KeyValue", None),
        (9, "i64", True, None, None),
        (10, "i64", False, None, None),
        (11, "i64", False, None, None),
        (12, "struct", False, "Statistics", None),
        (13, "list:struct", False, "PageEncodingStats", None),
        (14, "i64", False, None, None),
        (15, "i32", False, None, None),
        (16, "struct", False, "SizeStatistics", None),
        (17, "struct", False, "GeospatialStatistics", None),
    ),
    "KeyValue": ((1, "binary", True, None, None), (2, "binary", False, None, None)),
    "SortingColumn": (
        (1, "i32", True, None, None),
        (2, "bool", True, None, None),
        (3, "bool", True, None, None),
    ),
    "PageEncodingStats": (
        (1, "i32", True, None, (0, 1, 2, 3)),
        (2, "i32", True, None, _GRAMMAR_ENCODINGS),
        (3, "i32", True, None, None),
    ),
    "ColumnOrder": ((1, "struct", False, "TypeDefinedOrder", None),),
    "LogicalType": (
        (1, "struct", False, "StringType", None),
        (2, "struct", False, "MapType", None),
        (3, "struct", False, "ListType", None),
        (4, "struct", False, "EnumType", None),
        (5, "struct", False, "DecimalType", None),
        (6, "struct", False, "DateType", None),
        (7, "struct", False, "TimeType", None),
        (8, "struct", False, "TimestampType", None),
        (10, "struct", False, "IntType", None),
        (11, "struct", False, "NullType", None),
        (12, "struct", False, "JsonType", None),
        (13, "struct", False, "BsonType", None),
        (14, "struct", False, "UUIDType", None),
        (15, "struct", False, "Float16Type", None),
        (16, "struct", False, "VariantType", None),
        (17, "struct", False, "GeometryType", None),
        (18, "struct", False, "GeographyType", None),
    ),
    "DecimalType": ((1, "i32", True, None, None), (2, "i32", True, None, None)),
    "TimeType": ((1, "bool", True, None, None), (2, "struct", True, "TimeUnit", None)),
    "TimestampType": ((1, "bool", True, None, None), (2, "struct", True, "TimeUnit", None)),
    "TimeUnit": (
        (1, "struct", False, "MilliSeconds", None),
        (2, "struct", False, "MicroSeconds", None),
        (3, "struct", False, "NanoSeconds", None),
    ),
    "IntType": ((1, "byte", True, None, None), (2, "bool", True, None, None)),
    "VariantType": ((1, "byte", False, None, None),),
    "GeometryType": ((1, "binary", False, None, None),),
    "GeographyType": ((1, "binary", False, None, None), (2, "i32", False, None, (0, 1, 2, 3, 4))),
    "ColumnCryptoMetaData": (
        (1, "struct", False, "EncryptionWithFooterKey", None),
        (2, "struct", False, "EncryptionWithColumnKey", None),
    ),
    "EncryptionWithColumnKey": (
        (1, "list:binary", True, None, None),
        (2, "binary", False, None, None),
    ),
    "EncryptionAlgorithm": (
        (1, "struct", False, "AesGcmV1", None),
        (2, "struct", False, "AesGcmCtrV1", None),
    ),
    "AesGcmV1": (
        (1, "binary", False, None, None),
        (2, "binary", False, None, None),
        (3, "bool", False, None, None),
    ),
    "AesGcmCtrV1": (
        (1, "binary", False, None, None),
        (2, "binary", False, None, None),
        (3, "bool", False, None, None),
    ),
    "Statistics": (
        (1, "binary", False, None, None),
        (2, "binary", False, None, None),
        (3, "i64", False, None, None),
        (4, "i64", False, None, None),
        (5, "binary", False, None, None),
        (6, "binary", False, None, None),
        (7, "bool", False, None, None),
        (8, "bool", False, None, None),
    ),
    "SizeStatistics": (
        (1, "i64", False, None, None),
        (2, "list:i64", False, None, None),
        (3, "list:i64", False, None, None),
    ),
    "GeospatialStatistics": (
        (1, "struct", False, "BoundingBox", None),
        (2, "list:i32", False, None, None),
    ),
    "BoundingBox": (
        (1, "double", True, None, None),
        (2, "double", True, None, None),
        (3, "double", True, None, None),
        (4, "double", True, None, None),
        (5, "double", False, None, None),
        (6, "double", False, None, None),
        (7, "double", False, None, None),
        (8, "double", False, None, None),
    ),
    "TypeDefinedOrder": (),
    "StringType": (),
    "MapType": (),
    "ListType": (),
    "EnumType": (),
    "DateType": (),
    "NullType": (),
    "JsonType": (),
    "BsonType": (),
    "UUIDType": (),
    "Float16Type": (),
    "MilliSeconds": (),
    "MicroSeconds": (),
    "NanoSeconds": (),
    "EncryptionWithFooterKey": (),
}
_GRAMMAR_UNIONS = frozenset(
    ("ColumnOrder", "LogicalType", "TimeUnit", "ColumnCryptoMetaData", "EncryptionAlgorithm")
)
_GRAMMAR_TAGS = {
    "bool": 1,
    "byte": 3,
    "i16": 4,
    "i32": 5,
    "i64": 6,
    "double": 7,
    "binary": 8,
    "list": 9,
    "struct": 12,
}
_GRAMMAR_ANCHOR = bytes.fromhex("15 02 19 1c 48 01 6e 15 00 00 16 00 19 0c 00")
_GRAMMAR_LIMITS = {
    "max_schema_nodes": 32,
    "max_schema_depth": 8,
    "max_leaf_columns": 16,
    "max_row_groups": 4,
    "max_total_column_chunks": 64,
    "max_total_containers": 256,
    "max_total_metadata_entries": 32,
    "max_total_retained_records": 4096,
    "max_semantic_work_units": 65536,
    "max_span_comparison_bytes": 8192,
}
_GRAMMAR_COUNTERS = (
    "schema_nodes",
    "leaf_columns",
    "row_groups",
    "total_column_chunks",
    "total_containers",
    "total_metadata_entries",
    "total_retained_records",
    "semantic_work_units",
    "span_comparison_bytes",
)


def _grammar_unsigned(value: int) -> bytes:
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _grammar_integer(value: int) -> bytes:
    return _grammar_unsigned(value * 2 if value >= 0 else -value * 2 - 1)


def _grammar_list(element: str, bodies: tuple[bytes, ...]) -> bytes:
    tag = _GRAMMAR_TAGS[element]
    count = len(bodies)
    header = bytes(((min(count, 15) << 4) | tag,))
    if count >= 15:
        header += _grammar_unsigned(count)
    return header + b"".join(bodies)


def _grammar_structure(fields: tuple[tuple[int, str, bytes], ...]) -> bytes:
    parts: list[bytes] = []
    previous = 0
    for field_id, kind, body in fields:
        tag = body[0] if kind == "bool" else _GRAMMAR_TAGS[kind.split(":")[0]]
        delta = field_id - previous
        header = (
            bytes(((delta << 4) | tag,))
            if 1 <= delta <= 15
            else bytes((tag,)) + _grammar_integer(field_id)
        )
        parts.append(header if kind == "bool" else header + body)
        previous = field_id
    return b"".join(parts) + b"\x00"


def _grammar_route(target: str) -> dict[str, int]:
    choices = {
        name: fields[0][0] for name, fields in _GRAMMAR_FIELDS.items() if name in _GRAMMAR_UNIONS
    }
    for field_id, _, _, child, _ in _GRAMMAR_FIELDS["LogicalType"]:
        if child == target:
            choices["LogicalType"] = field_id
    if target in ("TimeUnit", "MilliSeconds", "MicroSeconds", "NanoSeconds"):
        choices["LogicalType"] = 7
    for field_id, _, _, child, _ in _GRAMMAR_FIELDS["TimeUnit"]:
        if child == target:
            choices["TimeUnit"] = field_id
    if target == "EncryptionWithColumnKey":
        choices["ColumnCryptoMetaData"] = 2
    if target == "AesGcmCtrV1":
        choices["EncryptionAlgorithm"] = 2
    return choices


def _grammar_object(
    kind: str,
    *,
    choices: dict[str, int],
    target: str | None = None,
    replacement: bytes | None = None,
    omit: int | None = None,
    field_change: tuple[int, str, bytes] | None = None,
    extra: tuple[int, str, bytes] | None = None,
    union_members: tuple[int, ...] | None = None,
    duplicate: bool = False,
) -> bytes:
    if kind == target and replacement is not None:
        return replacement
    fields: list[tuple[int, str, bytes]] = []
    for field_id, wire, _, child, domain in _GRAMMAR_FIELDS[kind]:
        if kind in _GRAMMAR_UNIONS:
            members = (
                union_members if kind == target and union_members is not None else (choices[kind],)
            )
            if field_id not in members:
                continue
        if kind == target and field_id == omit:
            continue
        if kind == target and field_change is not None and field_id == field_change[0]:
            fields.append(field_change)
            continue
        primitive = wire.split(":")[-1]
        if child is not None:
            body = _grammar_object(
                child,
                choices=choices,
                target=target,
                replacement=replacement,
                omit=omit,
                field_change=field_change,
                extra=extra,
                union_members=union_members,
                duplicate=duplicate,
            )
        elif primitive == "binary":
            body = b"\x04item"
        elif primitive == "bool":
            body = b"\x02"
        elif primitive == "double":
            body = b"\x00" * 8
        elif primitive == "byte":
            body = b"\x08"
        else:
            body = _grammar_integer(domain[0] if domain else 0)
        if wire.startswith("list:"):
            body = _grammar_list(primitive, (body,))
        fields.append((field_id, wire, body))
    if kind == target and duplicate:
        fields.append(fields[0])
    if kind == target and extra is not None:
        fields.append(extra)
    return _grammar_structure(tuple(fields))


class FooterGrammarTests(unittest.TestCase):
    """Hand-specified protocol objects exercise metadata grammar without source data."""

    def api(self):
        names = (
            "_FooterLimits",
            "_FooterCharge",
            "_FooterBudget",
            "_FooterExtent",
            "_FooterDocument",
            "_FooterNode",
            "_FooterField",
            "_FooterList",
            "_FooterSpan",
            "_parse_footer",
        )
        found = {}
        for name in names:
            value = getattr(bounded_parquet, name, None)
            self.assertTrue(callable(value), f"missing callable {name}")
            found[name] = value
        return SimpleNamespace(**found)

    def wire_budget(self, **changes: int):
        values = {
            "max_input_bytes": 65536,
            "max_object_bytes": 65536,
            "max_header_bytes": 64,
            "max_depth": 32,
            "max_fields_per_struct": 32,
            "max_total_fields": 4096,
            "max_items_per_container": 256,
            "max_total_items": 4096,
            "max_string_bytes": 256,
            "max_total_string_bytes": 8192,
            "max_varint_bytes": 10,
            "max_total_wire_bytes": 65536,
            "max_work_units": 131072,
            "max_copied_payload_bytes": 1,
        }
        values.update(changes)
        return CompactWireCursorTests().budget(**values)

    def budget(self, **changes: int):
        api = self.api()
        values = dict(_GRAMMAR_LIMITS)
        values.update(changes)
        return api._FooterBudget(wire_budget=self.wire_budget(), limits=api._FooterLimits(**values))

    def prepare(self, payload: bytes = _GRAMMAR_ANCHOR, *, budget=None):
        api = self.api()
        if budget is None:
            budget = self.budget()
        cursor = bounded_parquet._CompactWireCursor(
            payload, 0, len(payload), budget.wire_budget, kind="footer"
        )
        extent = api._FooterExtent(
            source_size=len(payload) + 17, footer_start=9, footer_end=9 + len(payload)
        )
        return api, cursor, extent, budget

    def parse(self, payload: bytes = _GRAMMAR_ANCHOR, *, budget=None):
        api, cursor, extent, budget = self.prepare(payload, budget=budget)
        return api._parse_footer(cursor, extent, budget), cursor, budget

    def fragment(self, target: str, **changes) -> bytes:
        return _grammar_object(
            "FileMetaData", choices=_grammar_route(target), target=target, **changes
        )

    def malformed(self, payload: bytes, reason: str = "schema_mismatch"):
        api, cursor, extent, budget = self.prepare(payload)
        with self.assertRaises(bounded_parquet._CompactWireMalformed) as raised:
            api._parse_footer(cursor, extent, budget)
        self.assertEqual(raised.exception.reason_code, reason)
        self.assertEqual(str(raised.exception), reason)
        return cursor, budget

    def field(self, node, field_id: int):
        matches = [field.value for field in node.fields if field.field_id == field_id]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def nodes(self, node):
        yield node
        for field in node.fields:
            value = field.value
            if isinstance(value, bounded_parquet._FooterNode):
                yield from self.nodes(value)
            elif isinstance(value, bounded_parquet._FooterList):
                for item in value.values:
                    if isinstance(item, bounded_parquet._FooterNode):
                        yield from self.nodes(item)

    def state(self, budget) -> tuple[int, ...]:
        return tuple(getattr(budget, name) for name in _GRAMMAR_COUNTERS) + (
            budget.active_schema_depth,
            budget.schema_depth_peak,
        )

    def test_footer_anchor_has_exact_spans_counters_and_retained_identity(self) -> None:
        api = self.api()
        document, cursor, budget = self.parse()
        self.assertIsInstance(document, api._FooterDocument)
        self.assertIs(document.payload, _GRAMMAR_ANCHOR)
        self.assertIs(document.budget, budget)
        self.assertIs(document.read_eligible, False)
        self.assertEqual(
            document.extent, api._FooterExtent(source_size=32, footer_start=9, footer_end=24)
        )
        self.assertEqual(document.root.kind, "FileMetaData")
        self.assertEqual(tuple(field.field_id for field in document.root.fields), (1, 2, 3, 4))
        self.assertEqual(self.field(document.root, 1), 1)
        self.assertEqual(self.field(document.root, 3), 0)
        schema = self.field(document.root, 2)
        self.assertEqual(schema.element_type, "struct")
        self.assertEqual(len(schema.values), 1)
        name = self.field(schema.values[0], 4)
        self.assertEqual((name.start, name.end), (6, 7))
        self.assertEqual(tuple(field.field_id for field in schema.values[0].fields), (4, 5))
        self.assertEqual(self.field(document.root, 4).values, ())
        self.assertEqual(cursor.position, 15)
        self.assertEqual(
            (
                budget.wire_budget.total_wire_bytes,
                budget.wire_budget.total_fields,
                budget.wire_budget.total_items,
                budget.wire_budget.total_string_bytes,
                budget.wire_budget.copied_payload_bytes,
            ),
            (15, 6, 1, 1, 0),
        )
        self.assertEqual(
            (
                budget.total_containers,
                budget.schema_nodes,
                budget.row_groups,
                budget.total_column_chunks,
                budget.total_metadata_entries,
                budget.leaf_columns,
            ),
            (2, 1, 0, 0, 0, 0),
        )
        with self.assertRaises(FrozenInstanceError):
            document.read_eligible = True

    def test_footer_all_reachable_structures_accept_complete_neutral_fields(self) -> None:
        self.api()
        self.assertEqual(len(_GRAMMAR_FIELDS), 42)
        self.assertNotIn("FileCryptoMetaData", _GRAMMAR_FIELDS)
        for kind in _GRAMMAR_FIELDS:
            with self.subTest(kind=kind):
                payload = self.fragment(kind)
                document, cursor, budget = self.parse(payload)
                self.assertEqual(cursor.position, len(payload))
                self.assertIs(document.payload, payload)
                self.assertIs(document.read_eligible, False)
                self.assertEqual(budget.wire_budget.copied_payload_bytes, 0)

    def test_footer_every_required_field_is_checked_at_its_struct_stop(self) -> None:
        self.api()
        for kind, fields in _GRAMMAR_FIELDS.items():
            for field_id, _, required, _, _ in fields:
                if required:
                    with self.subTest(kind=kind, field=field_id):
                        self.malformed(self.fragment(kind, omit=field_id))

    def test_footer_known_fields_reject_wrong_wire_types(self) -> None:
        self.api()
        for kind, fields in _GRAMMAR_FIELDS.items():
            for field_id, wire, _, _, _ in fields:
                wrong = "binary" if wire != "binary" else "i32"
                body = b"\x01n" if wrong == "binary" else b"\x00"
                choices = _grammar_route(kind)
                if kind in _GRAMMAR_UNIONS:
                    choices[kind] = field_id
                with self.subTest(kind=kind, field=field_id):
                    payload = _grammar_object(
                        "FileMetaData",
                        choices=choices,
                        target=kind,
                        field_change=(field_id, wrong, body),
                    )
                    self.malformed(payload)

    def test_footer_unknown_duplicate_and_reserved_fields_are_rejected(self) -> None:
        self.api()
        for kind, fields in _GRAMMAR_FIELDS.items():
            with self.subTest(kind=kind, defect="unknown"):
                self.malformed(self.fragment(kind, extra=(100, "i32", b"\x00")))
            if fields:
                with self.subTest(kind=kind, defect="duplicate"):
                    self.malformed(self.fragment(kind, duplicate=True))
        self.malformed(self.fragment("LogicalType", extra=(9, "struct", b"\x00")))

    def test_footer_unions_accept_each_branch_and_reject_zero_or_many(self) -> None:
        self.api()
        for kind in sorted(_GRAMMAR_UNIONS):
            members = tuple(field[0] for field in _GRAMMAR_FIELDS[kind])
            for field_id in members:
                with self.subTest(kind=kind, member=field_id):
                    payload = self.fragment(kind, union_members=(field_id,))
                    _, cursor, _ = self.parse(payload)
                    self.assertEqual(cursor.position, len(payload))
            with self.subTest(kind=kind, defect="empty"):
                self.malformed(self.fragment(kind, union_members=()))
            if len(members) > 1:
                with self.subTest(kind=kind, defect="multiple"):
                    self.malformed(self.fragment(kind, union_members=members[:2]))

    def test_footer_enum_domains_apply_to_scalars_and_each_encoding_element(self) -> None:
        self.api()
        for kind, fields in _GRAMMAR_FIELDS.items():
            for field_id, wire, _, _, domain in fields:
                if domain is None:
                    continue
                for value in (
                    *domain,
                    -1,
                    max(domain) + 1,
                    *((1,) if domain == _GRAMMAR_ENCODINGS else ()),
                ):
                    body = _grammar_integer(value)
                    if wire.startswith("list:"):
                        body = _grammar_list("i32", (_grammar_integer(domain[0]), body))
                    with self.subTest(kind=kind, field=field_id, value=value):
                        payload = self.fragment(kind, field_change=(field_id, wire, body))
                        if value in domain:
                            _, cursor, _ = self.parse(payload)
                            self.assertEqual(cursor.position, len(payload))
                        else:
                            self.malformed(payload)

    def test_footer_list_element_tags_and_non_enum_integer_lists_are_distinct(self) -> None:
        self.api()
        for kind, fields in _GRAMMAR_FIELDS.items():
            for field_id, wire, _, _, _ in fields:
                if wire.startswith("list:"):
                    with self.subTest(kind=kind, field=field_id):
                        self.malformed(
                            self.fragment(
                                kind,
                                field_change=(field_id, wire, _grammar_list("byte", (b"\x01",))),
                            )
                        )
        payload = self.fragment(
            "GeospatialStatistics",
            field_change=(2, "list:i32", _grammar_list("i32", (_grammar_integer(99),))),
        )
        _, cursor, _ = self.parse(payload)
        self.assertEqual(cursor.position, len(payload))

    def test_footer_optional_absence_zero_and_false_remain_distinct(self) -> None:
        self.api()
        absent, _, _ = self.parse(self.fragment("SchemaElement", omit=2))
        present, _, _ = self.parse(self.fragment("SchemaElement"))
        a = next(node for node in self.nodes(absent.root) if node.kind == "SchemaElement")
        p = next(node for node in self.nodes(present.root) if node.kind == "SchemaElement")
        self.assertNotIn(2, tuple(field.field_id for field in a.fields))
        self.assertEqual(self.field(p, 2), 0)
        sorting = next(node for node in self.nodes(present.root) if node.kind == "SortingColumn")
        self.assertIs(self.field(sorting, 2), False)
        self.assertIs(self.field(sorting, 3), False)

    def test_footer_opaque_statistics_never_unpack_copy_or_return_values(self) -> None:
        self.api()
        payload = self.fragment("BoundingBox")
        cursor_type = bounded_parquet._CompactWireCursor
        with (
            patch.object(
                cursor_type, "read_double", side_effect=AssertionError("opaque float decoded")
            ),
            patch.object(
                bounded_parquet, "unpack_from", side_effect=AssertionError("opaque float unpacked")
            ),
            patch.object(
                cursor_type, "read_binary", side_effect=AssertionError("opaque body copied")
            ),
            patch.object(hashlib, "sha256", side_effect=AssertionError("opaque body hashed")),
        ):
            document, cursor, budget = self.parse(payload)
        self.assertEqual(cursor.position, len(payload))
        self.assertEqual(budget.wire_budget.copied_payload_bytes, 0)
        column = next(node for node in self.nodes(document.root) if node.kind == "ColumnMetaData")
        for field_id in (12, 16, 17):
            self.assertEqual(self.field(column, field_id), "opaque")
        self.assertNotIn("item", repr(document))
        self.assertNotIn("item", repr(column))
        self.assertEqual(self.field(document.root, 6), "opaque")
        self.assertEqual(self.field(document.root, 9), "opaque")
        key_value = next(node for node in self.nodes(document.root) if node.kind == "KeyValue")
        self.assertEqual(self.field(key_value, 2), "opaque")

    def test_footer_late_truncation_and_trailing_bytes_return_no_document(self) -> None:
        self.api()
        for stop in range(1, len(_GRAMMAR_ANCHOR)):
            with self.subTest(stop=stop):
                self.malformed(_GRAMMAR_ANCHOR[:stop], "truncated_wire")
        self.malformed(_GRAMMAR_ANCHOR + b"\x00", "invalid_wire")
        self.malformed(self.fragment("BoundingBox", omit=4))

    def test_footer_every_opaque_field_is_discarded_while_identity_fields_keep_spans(self) -> None:
        api = self.api()
        cases = (
            ("FileMetaData", (6, 9)),
            ("ColumnMetaData", (12, 16, 17)),
            ("KeyValue", (2,)),
            ("GeometryType", (1,)),
            ("GeographyType", (1,)),
            ("AesGcmV1", (1, 2)),
            ("AesGcmCtrV1", (1, 2)),
            ("EncryptionWithColumnKey", (2,)),
            ("ColumnChunk", (9,)),
        )
        for kind, opaque_fields in cases:
            with self.subTest(kind=kind):
                document, _, budget = self.parse(self.fragment(kind))
                nodes = tuple(self.nodes(document.root))
                targets = tuple(node for node in nodes if node.kind == kind)
                self.assertGreater(len(targets), 0)
                for node in targets:
                    for field_id in opaque_fields:
                        self.assertEqual(self.field(node, field_id), "opaque")
                    self.assertNotIn("item", repr(node))
                for node in nodes:
                    if node.kind == "SchemaElement":
                        self.assertIsInstance(self.field(node, 4), api._FooterSpan)
                    elif node.kind == "KeyValue":
                        self.assertIsInstance(self.field(node, 1), api._FooterSpan)
                    elif node.kind in ("ColumnMetaData", "EncryptionWithColumnKey"):
                        field_id = 3 if node.kind == "ColumnMetaData" else 1
                        path = self.field(node, field_id)
                        self.assertEqual(path.element_type, "binary")
                        self.assertEqual(len(path.values), 1)
                        self.assertIsInstance(path.values[0], api._FooterSpan)
                self.assertEqual(budget.wire_budget.copied_payload_bytes, 0)

    def test_footer_declarations_refuse_before_wire_or_boundary_actions(self) -> None:
        api = self.api()
        for name in _GRAMMAR_LIMITS:
            for invalid in (True, 0, -1, 1.5, "1", None):
                values = dict(_GRAMMAR_LIMITS)
                values[name] = invalid
                with self.subTest(limit=name, invalid=invalid):
                    with (
                        self.assertRaises(PinnedReadRefused) as raised,
                    ):
                        api._FooterLimits(**values)
                    self.assertEqual(raised.exception.reason_code, "invalid_contract")
        for extent in ((True, 0, 1), (0, 0, 1), (8, -1, 1), (8, 2, 2), (8, 2, 9), (8, 0, 1.5)):
            with self.subTest(extent=extent):
                with self.assertRaises(PinnedReadRefused) as raised:
                    api._FooterExtent(*extent)
                self.assertEqual(raised.exception.reason_code, "invalid_contract")
        wire = self.wire_budget()
        limits = api._FooterLimits(**_GRAMMAR_LIMITS)
        with (
            patch.object(os, "open", side_effect=AssertionError("unexpected file open")),
            patch.object(hashlib, "sha256", side_effect=AssertionError("unexpected hash")),
        ):
            for supplied_wire, supplied_limits in ((None, limits), (wire, None)):
                with self.subTest(wire=supplied_wire is None):
                    with (
                        self.assertRaises(PinnedReadRefused) as raised,
                    ):
                        api._FooterBudget(wire_budget=supplied_wire, limits=supplied_limits)
                    self.assertEqual(raised.exception.reason_code, "invalid_contract")
        for defect in (
            "cursor_type",
            "extent_type",
            "budget_type",
            "kind",
            "position",
            "identity",
            "length",
        ):
            _, cursor, extent, budget = self.prepare()
            supplied_cursor, supplied_extent, supplied_budget = cursor, extent, budget
            if defect == "cursor_type":
                supplied_cursor = None
            elif defect == "extent_type":
                supplied_extent = None
            elif defect == "budget_type":
                supplied_budget = None
            elif defect == "kind":
                supplied_cursor = bounded_parquet._CompactWireCursor(
                    _GRAMMAR_ANCHOR, 0, 15, budget.wire_budget, kind="page_header"
                )
            elif defect == "position":
                cursor.read_byte()
            elif defect == "identity":
                supplied_budget = self.budget()
            else:
                supplied_extent = api._FooterExtent(33, 9, 25)
            before = (cursor.position, budget.wire_budget.total_wire_bytes, self.state(budget))
            with self.subTest(defect=defect):
                with (
                    patch.object(
                        bounded_parquet._CompactWireCursor,
                        "read_field",
                        side_effect=AssertionError("unexpected traversal"),
                    ),
                    self.assertRaises(PinnedReadRefused) as raised,
                ):
                    api._parse_footer(supplied_cursor, supplied_extent, supplied_budget)
                self.assertEqual(raised.exception.reason_code, "invalid_contract")
                self.assertEqual(
                    (cursor.position, budget.wire_budget.total_wire_bytes, self.state(budget)),
                    before,
                )

    def test_footer_charges_reject_wrong_types_and_negative_axes_atomically(self) -> None:
        api = self.api()
        budget = self.budget()
        budget.reserve(api._FooterCharge(schema_nodes=1))
        before = self.state(budget)
        for axis in _GRAMMAR_COUNTERS:
            for invalid in (True, -1, 1.5, "1", None):
                with self.subTest(axis=axis, invalid=invalid):
                    with (
                        self.assertRaises(PinnedReadRefused) as raised,
                    ):
                        api._FooterCharge(**{axis: invalid})
                    self.assertEqual(raised.exception.reason_code, "invalid_contract")
                    self.assertEqual(self.state(budget), before)
        for invalid in (None, True, 1, {"schema_nodes": 1}):
            with self.subTest(invalid=invalid):
                with self.assertRaises(PinnedReadRefused) as raised:
                    budget.reserve(invalid)
                self.assertEqual(raised.exception.reason_code, "invalid_contract")
                self.assertEqual(self.state(budget), before)
        budget.reserve(api._FooterCharge())
        self.assertEqual(self.state(budget), before)
        self.assertEqual(budget.wire_budget.total_wire_bytes, 0)

    def test_footer_each_additive_limit_allows_exact_boundary_without_refund(self) -> None:
        api = self.api()
        for axis in _GRAMMAR_COUNTERS:
            with self.subTest(axis=axis):
                budget = self.budget(**{"max_" + axis: 2})
                budget.reserve(api._FooterCharge(**{axis: 2}))
                self.assertEqual(getattr(budget, axis), 2)
                before = self.state(budget)
                with self.assertRaises(PinnedReadRefused) as raised:
                    budget.reserve(api._FooterCharge(**{axis: 1}))
                self.assertEqual(raised.exception.reason_code, "resource_limit")
                self.assertEqual(self.state(budget), before)
                self.assertEqual(budget.wire_budget.total_wire_bytes, 0)

    def test_footer_semantic_batches_do_not_commit_earlier_axes_on_later_refusal(self) -> None:
        api = self.api()
        budget = self.budget(max_schema_nodes=2, max_row_groups=2)
        budget.reserve(api._FooterCharge(schema_nodes=1, row_groups=1))
        before = self.state(budget)
        with self.assertRaises(PinnedReadRefused) as raised:
            budget.reserve(api._FooterCharge(schema_nodes=1, row_groups=2))
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(self.state(budget), before)
        budget.reserve(api._FooterCharge(schema_nodes=1, row_groups=1))
        self.assertEqual((budget.schema_nodes, budget.row_groups), (2, 2))
        for exhausted in ("max_semantic_work_units", "max_span_comparison_bytes"):
            comparison = self.budget(**{exhausted: 1})
            comparison.reserve(
                api._FooterCharge(
                    **{
                        "semantic_work_units"
                        if exhausted.endswith("work_units")
                        else "span_comparison_bytes": 1
                    }
                )
            )
            before = self.state(comparison)
            with self.subTest(exhausted=exhausted):
                with self.assertRaises(PinnedReadRefused) as raised:
                    comparison.reserve(
                        api._FooterCharge(semantic_work_units=1, span_comparison_bytes=2)
                    )
                self.assertEqual(raised.exception.reason_code, "resource_limit")
                self.assertEqual(self.state(comparison), before)

    def test_footer_schema_depth_tracks_active_peak_and_nonrefundable_work(self) -> None:
        self.api()
        budget = self.budget(max_schema_depth=2)
        budget.push_schema()
        budget.push_schema()
        self.assertEqual(
            (
                budget.active_schema_depth,
                budget.schema_depth_peak,
                budget.total_retained_records,
                budget.semantic_work_units,
            ),
            (2, 2, 2, 2),
        )
        before = self.state(budget)
        with self.assertRaises(PinnedReadRefused) as raised:
            budget.push_schema()
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(self.state(budget), before)
        budget.pop_schema()
        self.assertEqual(
            (
                budget.active_schema_depth,
                budget.schema_depth_peak,
                budget.total_retained_records,
                budget.semantic_work_units,
            ),
            (1, 2, 2, 3),
        )
        budget.pop_schema()
        before = self.state(budget)
        self.assertEqual(budget.active_schema_depth, 0)
        with self.assertRaises(PinnedReadRefused) as raised:
            budget.pop_schema()
        self.assertEqual(raised.exception.reason_code, "invalid_contract")
        self.assertEqual(self.state(budget), before)

    def test_footer_schema_push_and_pop_reserve_all_axes_before_mutation(self) -> None:
        api = self.api()
        for exhausted in ("max_total_retained_records", "max_semantic_work_units"):
            budget = self.budget(**{exhausted: 1})
            axis = exhausted.removeprefix("max_")
            budget.reserve(api._FooterCharge(**{axis: 1}))
            before = self.state(budget)
            with self.subTest(exhausted=exhausted):
                with self.assertRaises(PinnedReadRefused) as raised:
                    budget.push_schema()
                self.assertEqual(raised.exception.reason_code, "resource_limit")
                self.assertEqual(self.state(budget), before)
        budget = self.budget(max_semantic_work_units=1)
        budget.push_schema()
        before = self.state(budget)
        with self.assertRaises(PinnedReadRefused) as raised:
            budget.pop_schema()
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(self.state(budget), before)

    def test_footer_cardinality_refusal_follows_header_but_precedes_element_body(self) -> None:
        api = self.api()
        budget = self.budget(max_schema_nodes=1)
        payload = bytes.fromhex("15 02 19 2c")
        _, cursor, extent, _ = self.prepare(payload, budget=budget)
        header_states: list[tuple[int, ...]] = []
        original_read_list = bounded_parquet._CompactWireCursor.read_list_header

        def observe_header(reader, expected_element_type: str):
            result = original_read_list(reader, expected_element_type)
            header_states.append(self.state(budget))
            return result

        with (
            patch.object(bounded_parquet._CompactWireCursor, "read_list_header", observe_header),
            patch.object(
                bounded_parquet._CompactWireCursor,
                "read_binary_span",
                side_effect=AssertionError("unexpected body traversal"),
            ),
            self.assertRaises(PinnedReadRefused) as raised,
        ):
            api._parse_footer(cursor, extent, budget)
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual((cursor.position, budget.wire_budget.total_wire_bytes), (4, 4))
        self.assertEqual((budget.schema_nodes, budget.total_containers), (0, 0))
        self.assertGreater(budget.semantic_work_units, 0)
        self.assertEqual(len(header_states), 1)
        self.assertEqual(self.state(budget), header_states[0])
        admitted = self.budget(max_schema_nodes=2)
        _, cursor, extent, _ = self.prepare(payload, budget=admitted)
        with self.assertRaises(bounded_parquet._CompactWireMalformed) as raised:
            api._parse_footer(cursor, extent, admitted)
        self.assertEqual(raised.exception.reason_code, "truncated_wire")
        self.assertEqual(
            (cursor.position, admitted.schema_nodes, admitted.total_containers), (4, 2, 1)
        )

    def test_footer_grammar_role_limits_are_shared_across_nested_lists(self) -> None:
        api = self.api()
        for kind, field_id, limit, counter in (
            ("FileMetaData", 2, "max_schema_nodes", "schema_nodes"),
            ("FileMetaData", 4, "max_row_groups", "row_groups"),
            ("RowGroup", 1, "max_total_column_chunks", "total_column_chunks"),
            ("FileMetaData", 5, "max_total_metadata_entries", "total_metadata_entries"),
        ):
            with self.subTest(limit=limit):
                budget = self.budget(**{limit: 2})
                payload = self.fragment(kind, field_change=(field_id, "list:struct", b"\x3c"))
                _, cursor, extent, _ = self.prepare(payload, budget=budget)
                with self.assertRaises(PinnedReadRefused) as raised:
                    api._parse_footer(cursor, extent, budget)
                self.assertEqual(raised.exception.reason_code, "resource_limit")
                self.assertLessEqual(getattr(budget, counter), 2)
        payload = self.fragment("FileMetaData")
        exact = self.budget(max_total_metadata_entries=2)
        _, _, exact = self.parse(payload, budget=exact)
        self.assertEqual(exact.total_metadata_entries, 2)
        refused = self.budget(max_total_metadata_entries=1)
        _, cursor, extent, _ = self.prepare(payload, budget=refused)
        with self.assertRaises(PinnedReadRefused) as raised:
            api._parse_footer(cursor, extent, refused)
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(refused.total_metadata_entries, 1)

    def test_footer_empty_containers_and_shared_wire_budget_are_not_reset(self) -> None:
        api = self.api()
        exact = self.budget(max_total_containers=2)
        self.parse(budget=exact)
        self.assertEqual(exact.total_containers, 2)
        refused = self.budget(max_total_containers=1)
        _, cursor, extent, _ = self.prepare(budget=refused)
        with self.assertRaises(PinnedReadRefused) as raised:
            api._parse_footer(cursor, extent, refused)
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(refused.total_containers, 1)
        shared = self.budget()
        prefix = bounded_parquet._CompactWireCursor(
            b"\x01n", 0, 2, shared.wire_budget, kind="footer"
        )
        self.assertEqual(prefix.read_binary_span(), (1, 2))
        self.parse(budget=shared)
        self.parse(budget=shared)
        self.assertEqual(
            (
                shared.wire_budget.total_wire_bytes,
                shared.wire_budget.total_string_bytes,
                shared.wire_budget.copied_payload_bytes,
            ),
            (32, 3, 0),
        )
        self.assertEqual((shared.schema_nodes, shared.total_containers), (2, 4))

    def test_footer_record_and_work_exhaustion_precede_later_traversal(self) -> None:
        api = self.api()
        for limit in ("max_total_retained_records", "max_semantic_work_units"):
            with self.subTest(limit=limit):
                budget = self.budget(**{limit: 1})
                budget.reserve(api._FooterCharge(**{limit.removeprefix("max_"): 1}))
                before = self.state(budget)
                _, cursor, extent, _ = self.prepare(budget=budget)
                with (
                    patch.object(
                        bounded_parquet._CompactWireCursor,
                        "read_field",
                        side_effect=AssertionError("unexpected wire action"),
                    ),
                    self.assertRaises(PinnedReadRefused) as raised,
                ):
                    api._parse_footer(cursor, extent, budget)
                self.assertEqual(raised.exception.reason_code, "resource_limit")
                self.assertEqual(self.state(budget), before)
                self.assertEqual((cursor.position, budget.wire_budget.total_wire_bytes), (0, 0))

    def test_footer_physical_leaf_quota_counts_unresolved_declarations(self) -> None:
        api = self.api()
        declaration = _grammar_structure(((1, "i32", b"\x00"), (4, "binary", b"\x01n")))
        payload = _grammar_structure(
            (
                (1, "i32", b"\x02"),
                (2, "list:struct", _grammar_list("struct", (declaration, declaration))),
                (3, "i64", b"\x00"),
                (4, "list:struct", _grammar_list("struct", ())),
            )
        )
        exact = self.budget(max_leaf_columns=2)
        document, cursor, exact = self.parse(payload, budget=exact)
        self.assertEqual((exact.leaf_columns, cursor.position), (2, len(payload)))
        self.assertIs(document.read_eligible, False)
        refused = self.budget(max_leaf_columns=1)
        _, cursor, extent, _ = self.prepare(payload, budget=refused)
        with self.assertRaises(PinnedReadRefused) as raised:
            api._parse_footer(cursor, extent, refused)
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(refused.leaf_columns, 1)

    def test_footer_post_stop_reservation_refusal_precedes_document_construction(self) -> None:
        """Injected fault order is not numeric-cap or final-constructor exhaustion proof."""
        api = self.api()
        original_reserve = api._FooterBudget.reserve
        for axis in ("total_retained_records", "semantic_work_units"):
            _, cursor, extent, budget = self.prepare()
            rejected_states: list[tuple[int, ...]] = []

            def refuse_late(
                owner,
                charge,
                *,
                selected=axis,
                reader=cursor,
                expected_budget=budget,
                states=rejected_states,
            ):
                if reader.position == reader.stop and getattr(charge, selected) > 0:
                    self.assertIs(owner, expected_budget)
                    states.append(self.state(owner))
                    raise PinnedReadRefused("resource_limit")
                return original_reserve(owner, charge)

            with self.subTest(axis=axis):
                with (
                    patch.object(api._FooterBudget, "reserve", refuse_late),
                    patch.object(
                        bounded_parquet,
                        "_FooterDocument",
                        side_effect=AssertionError("document constructed before refusal"),
                    ) as constructed,
                    self.assertRaises(PinnedReadRefused) as raised,
                ):
                    api._parse_footer(cursor, extent, budget)
                self.assertEqual(raised.exception.reason_code, "resource_limit")
                self.assertEqual(cursor.position, 15)
                self.assertEqual(len(rejected_states), 1)
                self.assertEqual(self.state(budget), rejected_states[0])
                constructed.assert_not_called()


class FooterProfileTests(unittest.TestCase):
    """Neutral declarations distinguish semantic consistency from reader admission."""

    MANDATORY = frozenset(
        (
            "page_profile_unverified",
            "native_conversion_unproved",
            "source_binding_unproved",
            "utc_value_gate_unproved",
        )
    )
    FLAGS = frozenset(
        (
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
        )
    )
    # Union ID, physical type, fixed width, converted tag, expected uncertainty.
    LOGICAL_CASES = (
        (1, 6, None, 0, False),
        (2, None, None, 1, True),
        (3, None, None, 3, True),
        (4, 6, None, 4, False),
        (5, 2, None, 5, False),
        (6, 1, None, 6, False),
        (7, 1, None, 7, True),
        (8, 2, None, 9, True),
        (10, 1, None, 15, False),
        (11, 2, None, None, False),
        (12, 6, None, 19, False),
        (13, 6, None, 20, False),
        (14, 7, 16, None, False),
        (15, 7, 2, None, False),
        (16, 2, None, None, True),
        (17, 6, None, None, False),
        (18, 6, None, None, False),
    )
    CONVERTED_CASES = (
        (0, 6, None),
        (1, None, None),
        (2, None, None),
        (3, None, None),
        (4, 6, None),
        (5, 2, None),
        (6, 1, None),
        (7, 1, None),
        (8, 2, None),
        (9, 2, None),
        (10, 2, None),
        (11, 1, None),
        (12, 1, None),
        (13, 1, None),
        (14, 2, None),
        (15, 1, None),
        (16, 1, None),
        (17, 1, None),
        (18, 2, None),
        (19, 6, None),
        (20, 6, None),
        (21, 7, 12),
    )

    def api(self):
        found = {}
        for name in (
            "_FooterLeafProfile",
            "_FooterChunkProfile",
            "_FooterProfile",
            "_complete_footer_profile",
        ):
            value = getattr(bounded_parquet, name, None)
            self.assertTrue(callable(value), f"missing callable {name}")
            found[name] = value
        return SimpleNamespace(**found)

    @staticmethod
    def binary(value: bytes) -> bytes:
        return _grammar_unsigned(len(value)) + value

    @staticmethod
    def structure(fields: tuple[tuple[int, str, bytes], ...]) -> bytes:
        return _grammar_structure(tuple(sorted(fields)))

    def node(
        self,
        name: bytes = b"n",
        *,
        physical: int | None = 2,
        repetition: int | None = 0,
        children: int | None = None,
        logical: bytes | None = None,
        converted: int | None = None,
        length: int | None = None,
        extra: tuple[tuple[int, str, bytes], ...] = (),
    ) -> bytes:
        fields = [(4, "binary", self.binary(name))]
        for field_id, value in (
            (1, physical),
            (2, length),
            (3, repetition),
            (5, children),
            (6, converted),
        ):
            if value is not None:
                fields.append((field_id, "i32", _grammar_integer(value)))
        if logical is not None:
            fields.append((10, "struct", logical))
        return self.structure(tuple(fields) + extra)

    def root(self, children: int = 1, **changes) -> bytes:
        return self.node(b"root", physical=None, repetition=None, children=children, **changes)

    def logical(
        self,
        tag: int,
        *,
        unit: int = 1,
        utc: bool = False,
        width: int = 8,
        signed: bool = True,
        precision: int = 2,
        scale: int = 0,
    ) -> bytes:
        fields = ()
        if tag == 5:
            fields = ((1, "i32", _grammar_integer(scale)), (2, "i32", _grammar_integer(precision)))
        elif tag in (7, 8):
            fields = (
                (1, "bool", bytes((1 if utc else 2,))),
                (2, "struct", self.structure(((unit, "struct", b"\x00"),))),
            )
        elif tag == 10:
            fields = ((1, "byte", bytes((width & 255,))), (2, "bool", bytes((1 if signed else 2,))))
        return self.structure(((tag, "struct", self.structure(fields)),))

    def meta(
        self,
        *,
        physical: int = 2,
        path: tuple[bytes, ...] = (b"n",),
        values: int = 0,
        compressed: int = 0,
        uncompressed: int = 0,
        data: int = 100,
        encodings: tuple[int, ...] = (0,),
        codec: int = 0,
        extra: tuple[tuple[int, str, bytes], ...] = (),
    ) -> bytes:
        return self.structure(
            (
                (1, "i32", _grammar_integer(physical)),
                (2, "list:i32", _grammar_list("i32", tuple(map(_grammar_integer, encodings)))),
                (3, "list:binary", _grammar_list("binary", tuple(map(self.binary, path)))),
                (4, "i32", _grammar_integer(codec)),
                (5, "i64", _grammar_integer(values)),
                (6, "i64", _grammar_integer(uncompressed)),
                (7, "i64", _grammar_integer(compressed)),
                (9, "i64", _grammar_integer(data)),
            )
            + extra
        )

    def chunk(
        self,
        metadata: bytes | None,
        *,
        legacy: int = 0,
        external: bytes | None = None,
        extra: tuple[tuple[int, str, bytes], ...] = (),
    ) -> bytes:
        fields = [(2, "i64", _grammar_integer(legacy))]
        if external is not None:
            fields.append((1, "binary", self.binary(external)))
        if metadata is not None:
            fields.append((3, "struct", metadata))
        return self.structure(tuple(fields) + extra)

    def group(
        self,
        chunks: tuple[bytes, ...] = (),
        *,
        rows: int = 0,
        size: int = 0,
        compressed: int | None = None,
        extra: tuple[tuple[int, str, bytes], ...] = (),
    ) -> bytes:
        fields = [
            (1, "list:struct", _grammar_list("struct", chunks)),
            (2, "i64", _grammar_integer(size)),
            (3, "i64", _grammar_integer(rows)),
        ]
        if compressed is not None:
            fields.append((6, "i64", _grammar_integer(compressed)))
        return self.structure(tuple(fields) + extra)

    def payload(
        self,
        schema: tuple[bytes, ...] | None = None,
        groups: tuple[bytes, ...] = (),
        *,
        rows: int = 0,
        version: int = 1,
        extra: tuple[tuple[int, str, bytes], ...] = (),
    ) -> bytes:
        if schema is None:
            schema = (self.root(), self.node())
        return self.structure(
            (
                (1, "i32", _grammar_integer(version)),
                (2, "list:struct", _grammar_list("struct", schema)),
                (3, "i64", _grammar_integer(rows)),
                (4, "list:struct", _grammar_list("struct", groups)),
            )
            + extra
        )

    def single(
        self,
        metadata: bytes | None = None,
        *,
        rows: int = 0,
        size: int = 0,
        group_extra=(),
        chunk_extra=(),
        external=None,
        schema=None,
        root_extra=(),
    ) -> bytes:
        if metadata is None:
            metadata = self.meta(values=rows)
        return self.payload(
            schema,
            (
                self.group(
                    (self.chunk(metadata, extra=chunk_extra, external=external),),
                    rows=rows,
                    size=size,
                    extra=group_extra,
                ),
            ),
            rows=rows,
            extra=root_extra,
        )

    def document(self, payload: bytes | None = None, **limit_changes):
        if payload is None:
            payload = self.payload()
        grammar = FooterGrammarTests()
        budget = grammar.budget(**limit_changes)
        cursor = bounded_parquet._CompactWireCursor(
            payload, 0, len(payload), budget.wire_budget, kind="footer"
        )
        extent = bounded_parquet._FooterExtent(
            source_size=1032 + len(payload),
            footer_start=1024,
            footer_end=1024 + len(payload),
        )
        return bounded_parquet._parse_footer(cursor, extent, budget)

    def complete(self, payload: bytes | None = None, **limit_changes):
        api = self.api()
        document = self.document(payload, **limit_changes)
        result = api._complete_footer_profile(document)
        self.assertIsInstance(result, api._FooterProfile)
        self.assertIs(result.document, document)
        self.assertIs(result.read_eligible, False)
        self.assertIsInstance(result.flags, frozenset)
        self.assertTrue(result.flags >= self.MANDATORY)
        self.assertTrue(result.flags <= self.FLAGS)
        return document, result

    def refused(self, payload: bytes, reason: str = "schema_mismatch", **limits) -> None:
        api = self.api()
        document = self.document(payload, **limits)
        error = (
            PinnedReadRefused
            if reason == "resource_limit"
            else bounded_parquet._CompactWireMalformed
        )
        with self.assertRaises(error) as raised:
            api._complete_footer_profile(document)
        self.assertEqual(raised.exception.reason_code, reason)

    @staticmethod
    def state(budget) -> tuple[int, ...]:
        return tuple(getattr(budget, name) for name in _GRAMMAR_COUNTERS) + (
            budget.active_schema_depth,
            budget.schema_depth_peak,
        )

    @staticmethod
    def span_bytes(document, spans) -> tuple[bytes, ...]:
        return tuple(document.payload[span.start : span.end] for span in spans)

    def sole_leaf(self, result):
        self.assertEqual(len(result.leaves), 1)
        return result.leaves[0]

    def sole_chunk(self, result):
        self.assertEqual(len(result.chunks), 1)
        return result.chunks[0]

    def kv(self, key: bytes, value: bytes | None = None) -> bytes:
        fields = [(1, "binary", self.binary(key))]
        if value is not None:
            fields.append((2, "binary", self.binary(value)))
        return self.structure(tuple(fields))

    def test_profile_empty_and_scalar_records_are_complete_frozen_and_ineligible(self) -> None:
        _, empty = self.complete(self.payload((self.root(0),)))
        self.assertEqual((empty.leaves, empty.chunks, empty.flags), ((), (), self.MANDATORY))
        document, result = self.complete(self.single())
        leaf = self.sole_leaf(result)
        chunk = self.sole_chunk(result)
        self.assertEqual(
            (leaf.schema_index, leaf.leaf_ordinal, leaf.ancestor_indices), (1, 0, (0,))
        )
        self.assertEqual(self.span_bytes(document, leaf.path), (b"n",))
        self.assertEqual(
            (
                leaf.physical_type,
                leaf.repetition,
                leaf.type_length,
                leaf.converted_type,
                leaf.scale,
                leaf.precision,
                leaf.field_id,
                leaf.logical_type,
                leaf.definition_depth,
                leaf.repetition_depth,
                leaf.flags,
            ),
            (2, 0, None, None, None, None, None, None, 0, 0, frozenset()),
        )
        self.assertEqual(
            (
                chunk.row_group_index,
                chunk.column_index,
                chunk.schema_index,
                chunk.physical_type,
                chunk.codec,
                chunk.encodings,
                chunk.num_values,
                chunk.row_group_rows,
                chunk.compressed_size,
                chunk.uncompressed_size,
                chunk.data_page_offset,
                chunk.dictionary_page_offset,
                chunk.index_page_offset,
                chunk.legacy_file_offset,
                chunk.candidate_extent,
                chunk.flags,
            ),
            (0, 0, 1, 2, 0, (0,), 0, 0, 0, 0, 100, None, None, 0, (100, 100), frozenset()),
        )
        for record, name, value in (
            (result, "read_eligible", True),
            (leaf, "physical_type", 0),
            (chunk, "codec", 1),
        ):
            with self.subTest(record=type(record).__name__), self.assertRaises(FrozenInstanceError):
                setattr(record, name, value)

    def test_profile_root_roles_and_independent_nonroot_rules(self) -> None:
        invalid = (
            (),
            (self.node(repetition=None),),
            (self.node(physical=None, repetition=None),),
            (self.root(-1),),
            (self.root(), self.node(repetition=None)),
            (self.root(), self.node(physical=None, children=None)),
            (self.root(), self.node(children=-1)),
            (self.root(2), self.node()),
            (self.root(0), self.node()),
        )
        for schema in invalid:
            with self.subTest(schema=schema):
                self.refused(self.payload(schema))
        for repetition in (None, 0, 1, 2):
            with self.subTest(repetition=repetition):
                root = self.node(b"root", physical=None, children=1, repetition=repetition)
                _, result = self.complete(self.payload((root, self.node())))
                leaf = self.sole_leaf(result)
                self.assertEqual((leaf.definition_depth, leaf.repetition_depth), (0, 0))
                self.assertEqual("root_repetition_present" in result.flags, repetition is not None)

    def test_profile_ambiguous_child_presence_preserves_every_declaration(self) -> None:
        for count in (0, 1):
            with self.subTest(count=count):
                schema = (self.root(2), self.node(children=count), self.node(b"item"))
                group = self.group((self.chunk(self.meta(values=7, compressed=1)),))
                document, result = self.complete(self.payload(schema, (group,)))
                self.assertEqual(len(result.leaves), 2)
                for index, leaf in enumerate(result.leaves, 1):
                    self.assertEqual(leaf.schema_index, index)
                    self.assertEqual(
                        (
                            leaf.leaf_ordinal,
                            leaf.ancestor_indices,
                            leaf.path,
                            leaf.definition_depth,
                            leaf.repetition_depth,
                        ),
                        (None, None, None, None, None),
                    )
                    self.assertIn("unresolved_schema_tree", leaf.flags)
                self.assertIn("primitive_child_count_present", result.leaves[0].flags)
                self.assertIsNone(self.sole_chunk(result).schema_index)
                self.assertIn("unresolved_schema_tree", self.sole_chunk(result).flags)
                self.assertIs(result.document, document)
        with self.subTest(independent_missing_repetition=True):
            self.refused(self.payload((self.root(), self.node(children=0, repetition=None))))

    def test_profile_nested_levels_paths_and_empty_groups_use_source_order(self) -> None:
        schema = (
            self.root(2),
            self.node(b"item", physical=None, children=2, repetition=1),
            self.node(b"n", repetition=2),
            self.node(b"empty", physical=None, children=0),
            self.node(b"last"),
        )
        document, result = self.complete(self.payload(schema))
        self.assertEqual(len(result.leaves), 2)
        self.assertEqual(
            [
                (
                    x.schema_index,
                    x.leaf_ordinal,
                    x.ancestor_indices,
                    x.definition_depth,
                    x.repetition_depth,
                )
                for x in result.leaves
            ],
            [(2, 0, (0, 1), 2, 1), (4, 1, (0,), 0, 0)],
        )
        self.assertEqual(
            [self.span_bytes(document, x.path) for x in result.leaves],
            [(b"item", b"n"), (b"last",)],
        )
        self.assertEqual(document.budget.active_schema_depth, 0)
        self.assertEqual(document.budget.schema_depth_peak, 3)
        self.refused(self.payload(schema), "resource_limit", max_schema_depth=2)
        self.complete(self.payload(schema), max_schema_depth=3)

    def test_profile_duplicate_siblings_mark_every_descendant_without_joining_paths(self) -> None:
        schema = (
            self.root(1),
            self.node(b"item", physical=None, children=3),
            self.node(),
            self.node(),
            self.node(),
        )
        metadata = self.meta(path=(b"item", b"n"))
        group = self.group(tuple(self.chunk(metadata) for _ in range(3)))
        _, result = self.complete(self.payload(schema, (group,)))
        self.assertEqual(len(result.leaves), 3)
        self.assertEqual(len(result.chunks), 3)
        for record in (*result.leaves, *result.chunks, result):
            self.assertIn("ambiguous_projection_name", record.flags)
        empty = (
            self.root(2),
            self.node(b"", physical=None, children=0),
            self.node(b"", physical=None, children=0),
        )
        _, result = self.complete(self.payload(empty))
        self.assertEqual(result.leaves, ())
        self.assertIn("ambiguous_projection_name", result.flags)
        distinct = (
            self.root(2),
            self.node(b"n.item"),
            self.node(b"n", physical=None, children=1),
            self.node(b"item"),
        )
        document, result = self.complete(self.payload(distinct))
        self.assertEqual(
            [self.span_bytes(document, x.path) for x in result.leaves],
            [(b"n.item",), (b"n", b"item")],
        )
        self.assertNotIn("ambiguous_projection_name", result.flags)

    def test_profile_all_logical_families_preserve_annotations_and_uncertainty(self) -> None:
        for tag, physical, length, converted, uncertain in self.LOGICAL_CASES:
            with self.subTest(tag=tag):
                extra = (
                    ((7, "i32", _grammar_integer(0)), (8, "i32", _grammar_integer(2)))
                    if tag == 5
                    else ()
                )
                node = self.node(
                    physical=physical,
                    length=length,
                    logical=self.logical(tag),
                    converted=converted,
                    children=0 if physical is None else None,
                    extra=extra,
                )
                document, result = self.complete(self.payload((self.root(), node)))
                self.assertEqual("unresolved_logical_compatibility" in result.flags, uncertain)
                if physical is None:
                    self.assertEqual(result.leaves, ())
                else:
                    leaf = self.sole_leaf(result)
                    self.assertEqual(
                        (leaf.physical_type, leaf.type_length, leaf.converted_type),
                        (physical, length, converted),
                    )
                    schema = FooterGrammarTests().field(document.root, 2)
                    original = FooterGrammarTests().field(schema.values[1], 10)
                    self.assertIs(leaf.logical_type, original)
                if tag not in (11, 16):
                    wrong = self.node(
                        physical=0,
                        logical=self.logical(tag),
                        length=length,
                        converted=converted,
                        extra=extra,
                    )
                    self.refused(self.payload((self.root(), wrong)))

    def test_profile_all_converted_families_are_independently_checked(self) -> None:
        for converted, physical, length in self.CONVERTED_CASES:
            with self.subTest(converted=converted):
                extra = (
                    ((7, "i32", _grammar_integer(0)), (8, "i32", _grammar_integer(2)))
                    if converted == 5
                    else ()
                )
                node = self.node(
                    physical=physical,
                    converted=converted,
                    length=length,
                    children=0 if physical is None else None,
                    extra=extra,
                )
                _, result = self.complete(self.payload((self.root(), node)))
                if physical is not None:
                    leaf = self.sole_leaf(result)
                    self.assertIsNone(leaf.logical_type)
                    self.assertEqual(leaf.converted_type, converted)
                else:
                    self.assertIn("unresolved_logical_compatibility", result.flags)
                self.refused(
                    self.payload(
                        (
                            self.root(),
                            self.node(physical=0, converted=converted, length=length, extra=extra),
                        )
                    )
                )

    def test_profile_temporal_integer_and_optional_annotation_domains(self) -> None:
        for tag in (7, 8):
            for unit in (1, 2, 3):
                for utc in (False, True):
                    with self.subTest(tag=tag, unit=unit, utc=utc):
                        physical = 1 if tag == 7 and unit == 1 else 2
                        _, result = self.complete(
                            self.payload(
                                (
                                    self.root(),
                                    self.node(
                                        physical=physical,
                                        logical=self.logical(tag, unit=unit, utc=utc),
                                    ),
                                )
                            )
                        )
                        leaf = self.sole_leaf(result)
                        temporal = leaf.logical_type.fields[0].value
                        self.assertIs(FooterGrammarTests().field(temporal, 1), utc)
                        self.assertIn("unresolved_logical_compatibility", result.flags)
        for width, signed, converted in (
            (8, False, 11),
            (16, True, 16),
            (32, False, 13),
            (64, True, 18),
        ):
            for physical in (1, 2):
                with self.subTest(width=width, signed=signed, physical=physical):
                    _, result = self.complete(
                        self.payload(
                            (
                                self.root(),
                                self.node(
                                    physical=physical,
                                    logical=self.logical(10, width=width, signed=signed),
                                    converted=converted,
                                ),
                            )
                        )
                    )
                    self.assertEqual(
                        "unresolved_logical_compatibility" in result.flags,
                        physical != (2 if width == 64 else 1),
                    )
        for width in (-128, 0, 7, 127):
            with self.subTest(width=width):
                self.refused(
                    self.payload((self.root(), self.node(logical=self.logical(10, width=width))))
                )
        _, result = self.complete(
            self.payload(
                (self.root(), self.node(length=3, extra=((9, "i32", _grammar_integer(-7)),)))
            )
        )
        leaf = self.sole_leaf(result)
        self.assertEqual((leaf.type_length, leaf.field_id, leaf.flags), (3, -7, frozenset()))

    def test_profile_paired_annotations_distinguish_conflict_from_missing_companions(self) -> None:
        for logical, converted, physical in (
            (self.logical(1), 19, 6),
            (self.logical(7, unit=1), 8, 1),
            (self.logical(10, width=16), 15, 1),
            (self.logical(10, width=8, signed=False), 15, 1),
        ):
            with self.subTest(converted=converted, logical=logical):
                self.refused(
                    self.payload(
                        (
                            self.root(),
                            self.node(physical=physical, logical=logical, converted=converted),
                        )
                    )
                )
        for extra in ((), ((7, "i32", _grammar_integer(0)),)):
            with self.subTest(missing_decimal_companion=extra):
                _, result = self.complete(
                    self.payload(
                        (self.root(), self.node(logical=self.logical(5), converted=5, extra=extra))
                    )
                )
                self.assertIn("unresolved_logical_compatibility", result.flags)
        for field_id, value in ((7, 1), (8, 3)):
            with self.subTest(conflicting_decimal=field_id):
                self.refused(
                    self.payload(
                        (
                            self.root(),
                            self.node(
                                logical=self.logical(5),
                                extra=((field_id, "i32", _grammar_integer(value)),),
                            ),
                        )
                    )
                )
        _, result = self.complete(
            self.payload((self.root(), self.node(logical=self.logical(11), converted=9)))
        )
        self.assertIn("unresolved_logical_compatibility", result.flags)
        self.refused(
            self.payload(
                (
                    self.root(),
                    self.node(physical=7, length=16, logical=self.logical(14), converted=21),
                )
            )
        )

    def test_profile_decimal_domains_capacity_and_large_declarations_are_bounded(self) -> None:
        for precision, scale in ((0, 0), (-1, 0), (2, -1), (2, 3)):
            with self.subTest(precision=precision, scale=scale):
                self.refused(
                    self.payload(
                        (
                            self.root(),
                            self.node(logical=self.logical(5, precision=precision, scale=scale)),
                        )
                    )
                )
        for physical, length, precision, valid in (
            (1, None, 9, True),
            (1, None, 10, False),
            (2, None, 18, True),
            (2, None, 19, False),
            (7, 1, 2, True),
            (7, 1, 3, False),
            (7, 16, 38, True),
            (7, 16, 39, False),
            (6, None, 2147483647, True),
            (7, 2147483647, 2147483647, True),
        ):
            for legacy in (False, True):
                with self.subTest(
                    physical=physical, length=length, precision=precision, legacy=legacy
                ):
                    node = self.node(
                        physical=physical,
                        length=length,
                        logical=None if legacy else self.logical(5, precision=precision),
                        converted=5 if legacy else None,
                        extra=(
                            (7, "i32", _grammar_integer(0)),
                            (8, "i32", _grammar_integer(precision)),
                        )
                        if legacy
                        else (),
                    )
                    payload = self.payload((self.root(), node))
                    if valid:
                        _, result = self.complete(payload)
                        self.assertEqual(self.sole_leaf(result).physical_type, physical)
                    else:
                        self.refused(payload)
        for length in (None, 0, -1):
            with self.subTest(length=length):
                self.refused(self.payload((self.root(), self.node(physical=7, length=length))))
        for extra in ((), ((7, "i32", _grammar_integer(0)),), ((8, "i32", _grammar_integer(2)),)):
            with self.subTest(legacy_missing=extra):
                self.refused(self.payload((self.root(), self.node(converted=5, extra=extra))))
        huge = self.node(
            physical=7, length=536870912, logical=self.logical(5, precision=1073741824)
        )
        self.refused(self.payload((self.root(), huge)), "resource_limit")

    def test_profile_all_physical_codec_encoding_declarations_remain_enumerated(self) -> None:
        for physical in range(8):
            with self.subTest(physical=physical):
                _, result = self.complete(
                    self.single(
                        self.meta(physical=physical),
                        schema=(
                            self.root(),
                            self.node(physical=physical, length=3 if physical == 7 else None),
                        ),
                    )
                )
                self.assertEqual(self.sole_leaf(result).physical_type, physical)
                self.assertEqual(self.sole_chunk(result).physical_type, physical)
        for codec in range(8):
            for encoding in (0, 2, 3, 4, 5, 6, 7, 8, 9):
                with self.subTest(codec=codec, encoding=encoding):
                    _, result = self.complete(
                        self.single(self.meta(codec=codec, encodings=(encoding, encoding)))
                    )
                    chunk = self.sole_chunk(result)
                    self.assertEqual((chunk.codec, chunk.encodings), (codec, (encoding, encoding)))

    def test_profile_row_and_size_sums_do_not_infer_missing_plaintext_or_ratios(self) -> None:
        valid = self.single(
            self.meta(values=2, compressed=20, uncompressed=7),
            rows=2,
            size=7,
            group_extra=((6, "i64", _grammar_integer(20)),),
        )
        _, result = self.complete(valid)
        self.assertEqual(self.sole_chunk(result).compressed_size, 20)
        for rows, group_rows, size, compressed, values, chunk_size, chunk_compressed in (
            (-1, 0, 0, None, 0, 0, 0),
            (0, -1, 0, None, 0, 0, 0),
            (0, 0, -1, None, 0, 0, 0),
            (0, 0, 0, -1, 0, 0, 0),
            (0, 0, 0, None, -1, 0, 0),
            (0, 0, 0, None, 0, -1, 0),
            (0, 0, 0, None, 0, 0, -1),
            (1, 0, 0, None, 0, 0, 0),
            (0, 0, 1, None, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0),
        ):
            with self.subTest(
                case=(rows, group_rows, size, compressed, values, chunk_size, chunk_compressed)
            ):
                meta = self.meta(
                    values=values, uncompressed=chunk_size, compressed=chunk_compressed
                )
                group = self.group(
                    (self.chunk(meta),), rows=group_rows, size=size, compressed=compressed
                )
                self.refused(self.payload(groups=(group,), rows=rows))
        group = self.group((self.chunk(None),), rows=3, size=99, compressed=101)
        _, result = self.complete(self.payload(groups=(group,), rows=3))
        chunk = self.sole_chunk(result)
        self.assertEqual(
            (
                chunk.physical_type,
                chunk.codec,
                chunk.encodings,
                chunk.num_values,
                chunk.compressed_size,
                chunk.uncompressed_size,
                chunk.data_page_offset,
                chunk.dictionary_page_offset,
                chunk.index_page_offset,
                chunk.candidate_extent,
            ),
            (None,) * 10,
        )
        self.assertEqual(chunk.schema_index, 1)
        self.assertIn("plaintext_column_metadata_absent", chunk.flags)

    def test_profile_positional_identity_paths_orders_and_repeated_values(self) -> None:
        for metadata in (
            self.meta(physical=1),
            self.meta(path=()),
            self.meta(path=(b"n", b"item")),
            self.meta(path=(b"item",)),
        ):
            with self.subTest(metadata=metadata):
                self.refused(self.single(metadata))
        self.refused(self.payload(groups=(self.group(),)))
        for orders, valid in (
            ((), False),
            ((b"\x1c\x00\x00",), True),
            ((b"\x1c\x00\x00", b"\x1c\x00\x00"), False),
        ):
            with self.subTest(orders=len(orders)):
                payload = self.single(
                    root_extra=((7, "list:struct", _grammar_list("struct", orders)),)
                )
                if valid:
                    self.complete(payload)
                else:
                    self.refused(payload)
        for repetition, valid in ((0, False), (1, False), (2, True)):
            with self.subTest(repetition=repetition):
                schema = (
                    self.root(),
                    self.node(b"item", physical=None, children=1, repetition=repetition),
                    self.node(),
                )
                payload = self.single(
                    self.meta(path=(b"item", b"n"), values=5, compressed=3), rows=2, schema=schema
                )
                if valid:
                    _, result = self.complete(payload)
                    self.assertEqual(
                        (
                            self.sole_leaf(result).repetition_depth,
                            self.sole_chunk(result).num_values,
                        ),
                        (1, 5),
                    )
                else:
                    self.refused(payload)

    def test_profile_sorting_ordinals_and_encoding_stats_are_declarations(self) -> None:
        for ordinal in (-1, 0, 7):
            with self.subTest(ordinal=ordinal):
                sorting = self.structure(
                    ((1, "i32", _grammar_integer(0)), (2, "bool", b"\x02"), (3, "bool", b"\x01"))
                )
                extra = (
                    (4, "list:struct", _grammar_list("struct", (sorting, sorting))),
                    (7, "i16", _grammar_integer(ordinal)),
                )
                document, result = self.complete(self.single(group_extra=extra))
                self.assertIn(
                    "row_group_ordinal_interpretation_unresolved", self.sole_chunk(result).flags
                )
                group = FooterGrammarTests().field(document.root, 4).values[0]
                self.assertEqual(FooterGrammarTests().field(group, 7), ordinal)
                self.assertEqual(len(FooterGrammarTests().field(group, 4).values), 2)
        for index in (-1, 1):
            with self.subTest(index=index):
                sorting = self.structure(
                    (
                        (1, "i32", _grammar_integer(index)),
                        (2, "bool", b"\x01"),
                        (3, "bool", b"\x02"),
                    )
                )
                self.refused(
                    self.single(
                        schema=(self.root(), self.node(children=0)),
                        group_extra=((4, "list:struct", _grammar_list("struct", (sorting,))),),
                    )
                )
        for count, encoding, valid in ((-1, 0, False), (0, 9, True), (1, 0, True), (1, 9, False)):
            with self.subTest(count=count, encoding=encoding):
                stats = self.structure(
                    (
                        (1, "i32", _grammar_integer(2)),
                        (2, "i32", _grammar_integer(encoding)),
                        (3, "i32", _grammar_integer(count)),
                    )
                )
                payload = self.single(
                    self.meta(extra=((13, "list:struct", _grammar_list("struct", (stats, stats))),))
                )
                if valid:
                    document, _ = self.complete(payload)
                    metadata = next(
                        node
                        for node in FooterGrammarTests().nodes(document.root)
                        if node.kind == "ColumnMetaData"
                    )
                    self.assertEqual(len(FooterGrammarTests().field(metadata, 13).values), 2)
                else:
                    self.refused(payload)

    def test_profile_candidate_intervals_preserve_dictionary_zero_and_empty_cases(self) -> None:
        cases = (
            (100, 10, None, 0, (100, 110), False),
            (100, 20, 90, 0, (90, 110), False),
            (100, 10, 0, 0, None, True),
            (100, 0, None, 0, (100, 100), False),
            (1024, 0, None, 0, (1024, 1024), False),
            (100, 0, 90, 0, None, True),
        )
        for data, size, dictionary, values, expected, uncertain in cases:
            with self.subTest(case=(data, size, dictionary)):
                extra = () if dictionary is None else ((11, "i64", _grammar_integer(dictionary)),)
                _, result = self.complete(
                    self.single(
                        self.meta(data=data, compressed=size, values=values, extra=extra),
                        rows=values,
                    )
                )
                chunk = self.sole_chunk(result)
                self.assertEqual(
                    (chunk.data_page_offset, chunk.dictionary_page_offset, chunk.candidate_extent),
                    (data, dictionary, expected),
                )
                self.assertEqual("unresolved_extent_interpretation" in chunk.flags, uncertain)
        for data, size, dictionary, values in (
            (-1, 0, None, 0),
            (1025, 0, None, 0),
            (1024, 1, None, 0),
            (1020, 5, None, 0),
            (100, 10, -1, 0),
            (100, 10, 100, 0),
            (100, 10, 101, 0),
            (100, 10, 90, 0),
            (100, 0, None, 1),
            (100, 0, 0, 1),
        ):
            with self.subTest(invalid=(data, size, dictionary, values)):
                extra = () if dictionary is None else ((11, "i64", _grammar_integer(dictionary)),)
                self.refused(
                    self.single(
                        self.meta(data=data, compressed=size, values=values, extra=extra),
                        rows=values,
                    )
                )

    def test_profile_external_and_deprecated_references_do_not_bind_local_extent(self) -> None:
        for external in (b"", b"item"):
            with self.subTest(external=external):
                chunk = self.chunk(
                    self.meta(
                        data=2000,
                        compressed=5,
                        extra=(
                            (14, "i64", _grammar_integer(3000)),
                            (15, "i32", _grammar_integer(5)),
                        ),
                    ),
                    external=external,
                    legacy=9223372036854775807,
                    extra=((4, "i64", _grammar_integer(4000)), (5, "i32", _grammar_integer(5))),
                )
                _, result = self.complete(self.payload(groups=(self.group((chunk,)),)))
                actual = self.sole_chunk(result)
                self.assertIsNone(actual.candidate_extent)
                self.assertEqual(actual.legacy_file_offset, 9223372036854775807)
                self.assertTrue(
                    {"external_column_reference", "unbounded_bloom_extent"} <= actual.flags
                )
        _, result = self.complete(
            self.payload(groups=(self.group((self.chunk(self.meta(), legacy=5000),)),))
        )
        self.assertEqual(self.sole_chunk(result).legacy_file_offset, 5000)
        self.refused(self.payload(groups=(self.group((self.chunk(self.meta(), legacy=-1),)),)))
        self.refused(self.single(self.meta(data=-1), external=b""))
        self.refused(self.single(external=b"", chunk_extra=((4, "i64", _grammar_integer(2000)),)))

    def test_profile_index_bloom_pair_presence_and_local_endpoints(self) -> None:
        for offset_id, length_id in ((4, 5), (6, 7)):
            for offset, length, valid in (
                (0, 0, True),
                (1024, 0, True),
                (1023, 1, True),
                (-1, 0, False),
                (0, -1, False),
                (1025, 0, False),
                (1024, 1, False),
            ):
                with self.subTest(pair=(offset_id, length_id), offset=offset, length=length):
                    payload = self.single(
                        chunk_extra=(
                            (offset_id, "i64", _grammar_integer(offset)),
                            (length_id, "i32", _grammar_integer(length)),
                        )
                    )
                    if valid:
                        self.complete(payload)
                    else:
                        self.refused(payload)
            for field_id, wire in ((offset_id, "i64"), (length_id, "i32")):
                with self.subTest(unpaired=field_id):
                    self.refused(self.single(chunk_extra=((field_id, wire, _grammar_integer(0)),)))
        for offset, length, valid, unbounded in (
            (200, None, True, True),
            (1024, None, False, False),
            (1024, 0, True, False),
            (1023, 1, True, False),
            (1024, 1, False, False),
            (-1, 0, False, False),
            (100, -1, False, False),
        ):
            with self.subTest(bloom=(offset, length)):
                extra = ((14, "i64", _grammar_integer(offset)),)
                if length is not None:
                    extra += ((15, "i32", _grammar_integer(length)),)
                payload = self.single(self.meta(extra=extra))
                if valid:
                    _, result = self.complete(payload)
                    self.assertEqual("unbounded_bloom_extent" in result.flags, unbounded)
                else:
                    self.refused(payload)
        self.refused(self.single(self.meta(extra=((15, "i32", _grammar_integer(0)),))))
        for offset, valid in ((0, True), (1023, True), (1024, False), (-1, False)):
            with self.subTest(legacy_index=offset):
                payload = self.single(self.meta(extra=((10, "i64", _grammar_integer(offset)),)))
                if valid:
                    _, result = self.complete(payload)
                    self.assertIn("legacy_index_page_reference", self.sole_chunk(result).flags)
                else:
                    self.refused(payload)

    def test_profile_cross_group_interval_overlap_alias_and_touching_rules(self) -> None:
        first = self.group((self.chunk(self.meta(data=100, compressed=10)),))
        for start, valid in ((105, False), (100, False), (110, True), (90, True)):
            with self.subTest(page_start=start):
                second = self.group((self.chunk(self.meta(data=start, compressed=10)),))
                payload = self.payload(groups=(first, second))
                if valid:
                    self.complete(payload)
                else:
                    self.refused(payload)
        for start, size, valid in ((105, 1, False), (110, 1, True), (105, 0, True)):
            with self.subTest(page_sidecar=(start, size)):
                second = self.group(
                    (
                        self.chunk(
                            self.meta(data=500),
                            extra=(
                                (4, "i64", _grammar_integer(start)),
                                (5, "i32", _grammar_integer(size)),
                            ),
                        ),
                    )
                )
                payload = self.payload(groups=(first, second))
                if valid:
                    self.complete(payload)
                else:
                    self.refused(payload)
        for start, size, valid, alias in (
            (205, 10, False, False),
            (200, 10, True, True),
            (210, 10, True, False),
            (205, 0, True, False),
        ):
            with self.subTest(sidecars=(start, size)):
                groups = tuple(
                    self.group(
                        (
                            self.chunk(
                                self.meta(data=500),
                                extra=(
                                    (4, "i64", _grammar_integer(offset)),
                                    (5, "i32", _grammar_integer(length)),
                                ),
                            ),
                        )
                    )
                    for offset, length in ((200, 10), (start, size))
                )
                if valid:
                    _, result = self.complete(self.payload(groups=groups))
                    self.assertEqual(len(result.chunks), 2)
                    for chunk in result.chunks:
                        self.assertEqual("unresolved_extent_interpretation" in chunk.flags, alias)
                else:
                    self.refused(self.payload(groups=groups))

    def test_profile_optional_group_hint_is_checked_only_when_present(self) -> None:
        for hint, valid in ((None, True), (100, True), (101, False), (-1, False)):
            with self.subTest(hint=hint):
                extra = () if hint is None else ((5, "i64", _grammar_integer(hint)),)
                payload = self.single(self.meta(compressed=10), group_extra=extra)
                if valid:
                    _, result = self.complete(payload)
                    self.assertNotIn("unresolved_extent_interpretation", result.flags)
                else:
                    self.refused(payload)
        for hint in (None, 0, 2000):
            with self.subTest(empty_hint=hint):
                extra = () if hint is None else ((5, "i64", _grammar_integer(hint)),)
                _, result = self.complete(self.payload((self.root(0),), (self.group(extra=extra),)))
                self.assertEqual(
                    "unresolved_extent_interpretation" in result.flags, hint is not None
                )
            with self.subTest(external_hint=hint):
                extra = () if hint is None else ((5, "i64", _grammar_integer(hint)),)
                _, result = self.complete(self.single(external=b"", group_extra=extra))
                self.assertEqual(
                    "unresolved_extent_interpretation" in result.flags, hint is not None
                )

    def test_profile_metadata_duplicates_are_scoped_and_values_never_select_winners(self) -> None:
        for key in (b"", b"n"):
            entries = (self.kv(key), self.kv(key, b"marker"))
            for scope in ("file", "column"):
                with self.subTest(key=key, scope=scope):
                    body = _grammar_list("struct", entries)
                    payload = (
                        self.single(root_extra=((5, "list:struct", body),))
                        if scope == "file"
                        else self.single(self.meta(extra=((8, "list:struct", body),)))
                    )
                    self.refused(payload)
        key_list = _grammar_list("struct", (self.kv(b"n"),))
        groups = tuple(
            self.group((self.chunk(self.meta(extra=((8, "list:struct", key_list),))),))
            for _ in range(2)
        )
        _, result = self.complete(
            self.payload(groups=groups, extra=((5, "list:struct", key_list),))
        )
        self.assertEqual(len(result.chunks), 2)
        self.assertEqual(result.flags, self.MANDATORY | {"opaque_metadata_present"})
        for chunk in result.chunks:
            self.assertEqual(chunk.flags, frozenset(("opaque_metadata_present",)))

    def test_profile_metadata_key_matching_is_exact_and_absent_values_still_flag(self) -> None:
        cases = (
            (b"ARROW:schema", "arrow_schema_metadata_present"),
            (b"ARROW:extension:name", "arrow_extension_metadata_present"),
            (b"ARROW:extension:metadata", "arrow_extension_metadata_present"),
            (b"arrow:schema", None),
            (b"ARROW:extension:other", None),
            (b"ARROW:schema\x00", None),
            (b"", None),
        )
        for key, flag in cases:
            for value in (None, b"opaque-marker"):
                with self.subTest(key=key, value_present=value is not None):
                    entries = _grammar_list("struct", (self.kv(key, value),))
                    _, result = self.complete(
                        self.single(self.meta(extra=((8, "list:struct", entries),)))
                    )
                    expected = {"opaque_metadata_present"}
                    if flag is not None:
                        expected.add(flag)
                    self.assertEqual(self.sole_chunk(result).flags, frozenset(expected))
                    self.assertEqual(result.flags, self.MANDATORY | expected)
        _, result = self.complete(
            self.payload(
                (self.root(0),),
                extra=((5, "list:struct", _grammar_list("struct", (self.kv(b"ARROW:schema"),))),),
            )
        )
        self.assertNotIn("arrow_extension_metadata_present", result.flags)

    def test_profile_metadata_comparison_reservations_charge_complete_operands(self) -> None:
        api = self.api()
        for key, expected in (
            (b"", (12, 20, 24)),
            (b"n", (13, 21, 25)),
            (b"ARROW:schema", (24, 32, 36)),
        ):
            with self.subTest(key=key):
                document = self.document(
                    self.payload(
                        (self.root(0),),
                        extra=((5, "list:struct", _grammar_list("struct", (self.kv(key),))),),
                    )
                )
                observed = []
                reserve = bounded_parquet._FooterBudget.reserve

                def observe(current, charge, observed=observed, reserve=reserve):
                    if charge.span_comparison_bytes:
                        observed.append((charge.semantic_work_units, charge.span_comparison_bytes))
                    return reserve(current, charge)

                before = document.budget.span_comparison_bytes
                with patch.object(bounded_parquet._FooterBudget, "reserve", observe):
                    api._complete_footer_profile(document)
                self.assertEqual(observed, [(1, value) for value in expected])
                self.assertEqual(document.budget.span_comparison_bytes - before, sum(expected))

    def test_profile_comparison_quota_refusal_is_atomic_without_claiming_body_observation(
        self,
    ) -> None:
        """Observe the reservation boundary; source review establishes byte-access ordering."""
        api = self.api()
        payload = self.payload(
            (self.root(0),), extra=((5, "list:struct", _grammar_list("struct", (self.kv(b"n"),))),)
        )
        for allowance, valid in ((58, False), (59, True)):
            with self.subTest(allowance=allowance):
                document = self.document(payload, max_span_comparison_bytes=allowance)
                if valid:
                    result = api._complete_footer_profile(document)
                    self.assertIs(result.document, document)
                    self.assertEqual(document.budget.span_comparison_bytes, 59)
                    continue
                denied = []
                reserve = bounded_parquet._FooterBudget.reserve

                def observe(current, charge, denied=denied, reserve=reserve):
                    before = self.state(current)
                    try:
                        return reserve(current, charge)
                    except PinnedReadRefused:
                        denied.append((charge, before, self.state(current)))
                        raise

                with (
                    patch.object(bounded_parquet._FooterBudget, "reserve", observe),
                    patch.object(bounded_parquet, "_FooterProfile") as publication,
                ):
                    with self.assertRaises(PinnedReadRefused) as raised:
                        api._complete_footer_profile(document)
                    self.assertEqual(raised.exception.reason_code, "resource_limit")
                    publication.assert_not_called()
                self.assertEqual(len(denied), 1)
                charge, before, after = denied[0]
                self.assertEqual(
                    (charge.semantic_work_units, charge.span_comparison_bytes), (1, 25)
                )
                self.assertEqual(before, after)
                self.assertEqual(document.budget.span_comparison_bytes, 34)

    def test_profile_opaque_metadata_has_no_decode_copy_hash_or_float_path(self) -> None:
        api = self.api()
        marker = b"opaque-marker"
        stats = self.structure(((1, "binary", self.binary(marker)),))
        metadata = self.meta(
            extra=(
                (8, "list:struct", _grammar_list("struct", (self.kv(b"n", marker),))),
                (12, "struct", stats),
            )
        )
        document = self.document(
            self.single(metadata, root_extra=((6, "binary", self.binary(marker)),))
        )
        payload = document.payload
        with (
            patch.object(
                bounded_parquet._CompactWireCursor,
                "read_binary",
                side_effect=AssertionError("copied body"),
            ),
            patch.object(
                bounded_parquet._CompactWireCursor,
                "read_double",
                side_effect=AssertionError("decoded float"),
            ),
            patch.object(
                bounded_parquet, "unpack_from", side_effect=AssertionError("unpacked statistic")
            ),
            patch.object(hashlib, "sha256", side_effect=AssertionError("hashed body")),
        ):
            result = api._complete_footer_profile(document)
        self.assertIs(result.document.payload, payload)
        self.assertEqual(document.budget.wire_budget.copied_payload_bytes, 0)
        self.assertIn("opaque_metadata_present", result.flags)
        self.assertNotIn(marker.decode("ascii"), repr((result, result.leaves, result.chunks)))

    def test_profile_all_twenty_three_flags_have_declaration_based_witnesses(self) -> None:
        crypto = self.structure(((1, "struct", b"\x00"),))
        encryption = self.structure(((1, "struct", b"\x00"),))
        entries = _grammar_list(
            "struct", (self.kv(b"ARROW:schema"), self.kv(b"ARROW:extension:name"))
        )
        schema = (
            self.node(
                b"root",
                physical=None,
                repetition=0,
                children=2,
                logical=self.logical(2),
                converted=1,
            ),
            self.node(children=0),
            self.node(b"item"),
        )
        metadata = self.meta(
            extra=(
                (10, "i64", _grammar_integer(200)),
                (11, "i64", _grammar_integer(0)),
                (14, "i64", _grammar_integer(300)),
            )
        )
        chunks = (
            self.chunk(
                metadata, extra=((8, "struct", crypto), (9, "binary", self.binary(b"marker")))
            ),
            self.chunk(None, external=b""),
        )
        group = self.group(
            chunks, extra=((5, "i64", _grammar_integer(0)), (7, "i16", _grammar_integer(-1)))
        )
        _, result = self.complete(
            self.payload(
                schema,
                (group,),
                version=3,
                extra=(
                    (5, "list:struct", entries),
                    (8, "struct", encryption),
                    (9, "binary", self.binary(b"marker")),
                ),
            )
        )
        self.assertEqual(result.flags, self.FLAGS - {"ambiguous_projection_name"})
        self.assertEqual(len(result.leaves), 2)
        self.assertEqual(len(result.chunks), 2)
        self.assertIn("primitive_child_count_present", result.leaves[0].flags)
        for leaf in result.leaves:
            self.assertIn("unresolved_schema_tree", leaf.flags)
        self.assertTrue(
            {"column_encryption_present", "encrypted_column_metadata_present"}
            <= result.chunks[0].flags
        )
        self.assertTrue(
            {"external_column_reference", "plaintext_column_metadata_absent"}
            <= result.chunks[1].flags
        )
        _, duplicate = self.complete(self.payload((self.root(2), self.node(), self.node())))
        self.assertEqual(result.flags | duplicate.flags, self.FLAGS)
        for member in (1, 2):
            with self.subTest(column_crypto_member=member):
                body = (
                    b"\x00"
                    if member == 1
                    else self.structure(
                        ((1, "list:binary", _grammar_list("binary", (self.binary(b"n"),))),)
                    )
                )
                _, encrypted = self.complete(
                    self.single(
                        chunk_extra=((8, "struct", self.structure(((member, "struct", body),))),)
                    )
                )
                self.assertIn("column_encryption_present", self.sole_chunk(encrypted).flags)
        for member in (1, 2):
            with self.subTest(footer_crypto_member=member):
                _, encrypted = self.complete(
                    self.payload(
                        (self.root(0),),
                        extra=((8, "struct", self.structure(((member, "struct", b"\x00"),))),),
                    )
                )
                self.assertIn("footer_encryption_present", encrypted.flags)

    def test_profile_reuses_budgets_without_recounting_roles_or_resetting_work(self) -> None:
        api = self.api()
        document = self.document(self.single())
        budget = document.budget
        budget.reserve(
            bounded_parquet._FooterCharge(semantic_work_units=7, total_retained_records=3)
        )
        cardinality = self.state(budget)[:6]
        wire_names = (
            "total_wire_bytes",
            "work_units",
            "total_fields",
            "total_items",
            "total_string_bytes",
            "copied_payload_bytes",
        )
        wire_before = tuple(getattr(budget.wire_budget, name) for name in wire_names)
        before = self.state(budget)
        first = api._complete_footer_profile(document)
        after_first = self.state(budget)
        second = api._complete_footer_profile(document)
        self.assertIs(first.document, document)
        self.assertIs(second.document.budget, budget)
        self.assertEqual(first.leaves, second.leaves)
        self.assertEqual(first.chunks, second.chunks)
        self.assertEqual(first.flags, second.flags)
        self.assertEqual(self.state(budget)[:6], cardinality)
        self.assertEqual(
            tuple(getattr(budget.wire_budget, name) for name in wire_names), wire_before
        )
        for index in (6, 7, 8):
            self.assertGreater(after_first[index], before[index])
            self.assertGreater(self.state(budget)[index], after_first[index])
        self.assertEqual(budget.active_schema_depth, 0)

    def test_profile_invalid_entry_and_residual_depth_refuse_without_charge(self) -> None:
        api = self.api()
        document = self.document()
        invalid = (
            None,
            True,
            SimpleNamespace(),
            replace(document, read_eligible=True),
            replace(document, payload=bytearray(document.payload)),
            replace(document, root=bounded_parquet._FooterNode("SchemaElement", ())),
        )
        before = self.state(document.budget)
        for value in invalid:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(PinnedReadRefused) as raised:
                    api._complete_footer_profile(value)
                self.assertEqual(raised.exception.reason_code, "invalid_contract")
                self.assertEqual(self.state(document.budget), before)
        document.budget.push_schema()
        before = self.state(document.budget)
        with self.assertRaises(PinnedReadRefused) as raised:
            api._complete_footer_profile(document)
        self.assertEqual(raised.exception.reason_code, "invalid_contract")
        self.assertEqual(self.state(document.budget), before)

    def test_profile_depth_refusal_preserves_prior_edge_charge_and_failed_push_state(self) -> None:
        api = self.api()
        document = self.document(max_schema_depth=1)
        budget = document.budget
        push = bounded_parquet._FooterBudget.push_schema
        reserve = bounded_parquet._FooterBudget.reserve
        events = []
        denied = []

        def observe_reserve(current, charge):
            result = reserve(current, charge)
            events.append((charge.total_retained_records, charge.semantic_work_units))
            return result

        def observe_push(current):
            before = self.state(current)
            try:
                return push(current)
            except PinnedReadRefused:
                denied.append((before, self.state(current), tuple(events)))
                raise

        with (
            patch.object(bounded_parquet._FooterBudget, "reserve", observe_reserve),
            patch.object(bounded_parquet._FooterBudget, "push_schema", observe_push),
            patch.object(bounded_parquet, "_FooterProfile") as publication,
        ):
            with self.assertRaises(PinnedReadRefused) as raised:
                api._complete_footer_profile(document)
            self.assertEqual(raised.exception.reason_code, "resource_limit")
            publication.assert_not_called()
        self.assertEqual(len(denied), 1)
        before, after, successful = denied[0]
        self.assertEqual(before, after)
        self.assertEqual(successful[-1], (1, 0))
        self.assertEqual((budget.active_schema_depth, budget.schema_depth_peak), (1, 1))

    def test_profile_record_constructors_follow_their_reserved_allocation_batches(self) -> None:
        """Observe reservation-to-constructor ordering, without claiming allocator-byte bounds."""
        api = self.api()
        document = self.document(self.single())
        reserve = bounded_parquet._FooterBudget.reserve
        last = []
        constructed = []

        def observe_reserve(current, charge):
            result = reserve(current, charge)
            last[:] = [(charge.total_retained_records, charge.semantic_work_units)]
            return result

        def observe_leaf(*args, **kwargs):
            self.assertEqual(last, [(2, 1)])
            constructed.append("leaf")
            return api._FooterLeafProfile(*args, **kwargs)

        def observe_chunk(*args, **kwargs):
            self.assertEqual(last, [(2, 1)])
            constructed.append("chunk")
            return api._FooterChunkProfile(*args, **kwargs)

        def observe_profile(*args, **kwargs):
            self.assertEqual(last, [(1, 1)])
            self.assertEqual(document.budget.active_schema_depth, 0)
            constructed.append("profile")
            return api._FooterProfile(*args, **kwargs)

        with (
            patch.object(bounded_parquet._FooterBudget, "reserve", observe_reserve),
            patch.object(bounded_parquet, "_FooterLeafProfile", side_effect=observe_leaf),
            patch.object(bounded_parquet, "_FooterChunkProfile", side_effect=observe_chunk),
            patch.object(bounded_parquet, "_FooterProfile", side_effect=observe_profile),
        ):
            result = api._complete_footer_profile(document)
        self.assertEqual(constructed, ["leaf", "chunk", "profile"])
        self.assertIs(result.document, document)

    def test_profile_exhausted_work_or_records_never_publish_or_reset_counters(self) -> None:
        api = self.api()
        for counter, limit in (
            ("semantic_work_units", "max_semantic_work_units"),
            ("total_retained_records", "max_total_retained_records"),
        ):
            with self.subTest(counter=counter):
                document = self.document()
                budget = document.budget
                amount = getattr(budget.limits, limit) - getattr(budget, counter)
                budget.reserve(bounded_parquet._FooterCharge(**{counter: amount}))
                before = self.state(budget)
                with patch.object(bounded_parquet, "_FooterProfile") as publication:
                    with self.assertRaises(PinnedReadRefused) as raised:
                        api._complete_footer_profile(document)
                    self.assertEqual(raised.exception.reason_code, "resource_limit")
                    publication.assert_not_called()
                self.assertEqual(getattr(budget, counter), before[_GRAMMAR_COUNTERS.index(counter)])
                self.assertEqual(self.state(budget)[:6], before[:6])

    def test_profile_late_group_and_metadata_contradictions_publish_nothing(self) -> None:
        api = self.api()
        first = self.group((self.chunk(self.meta()),))
        late = self.group((self.chunk(self.meta()),), size=1)
        duplicate = _grammar_list("struct", (self.kv(b"n"), self.kv(b"n")))
        for payload in (
            self.payload(groups=(first, late)),
            self.payload(groups=(first,), extra=((5, "list:struct", duplicate),)),
        ):
            with self.subTest(payload=payload):
                document = self.document(payload)
                before = document.budget.semantic_work_units
                with patch.object(bounded_parquet, "_FooterProfile") as publication:
                    with self.assertRaises(bounded_parquet._CompactWireMalformed) as raised:
                        api._complete_footer_profile(document)
                    self.assertEqual(raised.exception.reason_code, "schema_mismatch")
                    publication.assert_not_called()
                self.assertGreater(document.budget.semantic_work_units, before)

    def test_profile_duplicate_nonempty_groups_mark_distinct_descendant_paths(self) -> None:
        schema = (
            self.root(2),
            self.node(b"item", physical=None, children=1),
            self.node(b"left"),
            self.node(b"item", physical=None, children=1),
            self.node(b"right"),
        )
        chunks = (
            self.chunk(self.meta(path=(b"item", b"left"))),
            self.chunk(self.meta(path=(b"item", b"right"))),
        )
        document, result = self.complete(self.payload(schema, groups=(self.group(chunks),)))
        self.assertEqual(len(result.leaves), 2)
        self.assertEqual(len(result.chunks), 2)
        self.assertEqual(
            tuple(self.span_bytes(document, leaf.path) for leaf in result.leaves),
            ((b"item", b"left"), (b"item", b"right")),
        )
        self.assertIn("ambiguous_projection_name", result.flags)
        for leaf, chunk in zip(result.leaves, result.chunks, strict=True):
            self.assertIn("ambiguous_projection_name", leaf.flags)
            self.assertIn("ambiguous_projection_name", chunk.flags)
            self.assertEqual(chunk.schema_index, leaf.schema_index)

    def test_profile_negative_sizes_refuse_when_plaintext_sums_are_unavailable(self) -> None:
        schema = (self.root(2), self.node(b"n"), self.node(b"item"))
        absent = self.chunk(None)
        for group_size, group_compressed, metadata in (
            (-1, None, self.meta()),
            (0, -1, self.meta()),
            (0, None, self.meta(uncompressed=-1)),
            (0, None, self.meta(compressed=-1)),
            (0, None, self.meta(values=-1)),
        ):
            with self.subTest(
                group_size=group_size, group_compressed=group_compressed, metadata=metadata
            ):
                chunks = (self.chunk(metadata), absent)
                self.refused(
                    self.payload(
                        schema,
                        groups=(
                            self.group(
                                chunks,
                                size=group_size,
                                compressed=group_compressed,
                            ),
                        ),
                    )
                )
        # Equal negative aggregate counts still violate each declaration's domain.
        self.refused(
            self.payload(
                (self.root(1), self.node(repetition=2)),
                groups=(self.group((absent,), rows=-1),),
                rows=-1,
            )
        )

    def test_profile_balanced_negative_group_rows_refuse_with_missing_plaintext(self) -> None:
        groups = (
            self.group((self.chunk(None),), rows=-1),
            self.group((self.chunk(None),), rows=1),
        )
        self.refused(self.payload(groups=groups, rows=0))

    def test_profile_record_caps_replay_every_allocation_boundary_atomically(self) -> None:
        """Use real caps and trace-prefix arithmetic, not inferred operation identities."""
        api = self.api()
        reserve = bounded_parquet._FooterBudget.reserve

        def trace(document):
            events = []

            def observe(current, charge):
                before = self.state(current)
                increments = tuple(getattr(charge, name) for name in _GRAMMAR_COUNTERS)
                try:
                    reserve(current, charge)
                except PinnedReadRefused:
                    events.append((before, increments, self.state(current), False))
                    raise
                events.append((before, increments, self.state(current), True))

            result = None
            refusal = None
            with (
                patch.object(bounded_parquet._FooterBudget, "reserve", observe),
                patch.object(
                    bounded_parquet,
                    "_FooterProfile",
                    side_effect=api._FooterProfile,
                ) as publication,
            ):
                try:
                    result = api._complete_footer_profile(document)
                except PinnedReadRefused as error:
                    refusal = error.reason_code
            return tuple(events), result, refusal, publication.call_count

        nested_schema = (
            self.node(b"root", physical=None, repetition=0, children=1),
            self.node(b"item", physical=None, repetition=1, children=1),
            self.node(b"n", logical=self.logical(8), converted=9),
        )
        payloads = (
            self.payload((self.root(0),)),
            self.payload(
                nested_schema, groups=(self.group((self.chunk(self.meta(path=(b"item", b"n"))),)),)
            ),
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                document = self.document(payload)
                grammar_records = document.budget.total_retained_records
                expected, result, refusal, allocations = trace(document)
                self.assertIsNone(refusal)
                self.assertIsInstance(result, api._FooterProfile)
                self.assertEqual(allocations, 1)
                self.assertTrue(expected)
                self.assertTrue(result.flags >= self.MANDATORY)
                for before, increments, after, succeeded in expected:
                    self.assertTrue(succeeded)
                    self.assertEqual(
                        after,
                        tuple(
                            before[index] + increments[index] if index < 9 else before[index]
                            for index in range(11)
                        ),
                    )
                boundaries = tuple(index for index, event in enumerate(expected) if event[1][6] > 0)
                self.assertTrue(boundaries)
                self.assertEqual(expected[boundaries[-1]][1][6:8], (1, 1))
                for index in boundaries:
                    cumulative_end = expected[index][2][6]
                    for exact in (False, True):
                        with self.subTest(boundary=index, exact=exact):
                            cap = cumulative_end if exact else cumulative_end - 1
                            replay_document = self.document(payload, max_total_retained_records=cap)
                            self.assertEqual(
                                replay_document.budget.total_retained_records, grammar_records
                            )
                            actual, replay_result, replay_refusal, replay_allocations = trace(
                                replay_document
                            )
                            successful_count = (
                                len(actual) if replay_refusal is None else len(actual) - 1
                            )
                            self.assertEqual(actual[:successful_count], expected[:successful_count])
                            if not exact:
                                self.assertEqual(replay_refusal, "resource_limit")
                                self.assertEqual(len(actual), index + 1)
                                self.assertEqual(actual[index][:2], expected[index][:2])
                                self.assertFalse(actual[index][3])
                                self.assertEqual(actual[index][0], actual[index][2])
                            else:
                                self.assertGreaterEqual(successful_count, index + 1)
                            if exact and index == boundaries[-1]:
                                self.assertIsNone(replay_refusal)
                                self.assertIsInstance(replay_result, api._FooterProfile)
                                self.assertEqual(replay_allocations, 1)
                                self.assertEqual(actual, expected)
                            else:
                                self.assertEqual(replay_refusal, "resource_limit")
                                self.assertIsNone(replay_result)
                                self.assertEqual(replay_allocations, 0)
                                self.assertFalse(actual[-1][3])
                                self.assertEqual(actual[-1][:2], expected[successful_count][:2])
                                self.assertEqual(actual[-1][0], actual[-1][2])

    def decimal_reader(self):
        function = getattr(bounded_parquet, "_footer_fixed_decimal_capacity", None)
        self.assertTrue(callable(function), "missing callable _footer_fixed_decimal_capacity")
        return function

    def test_decimal_capacity_exact_results_and_coarse_extremes(self) -> None:
        function = self.decimal_reader()
        for width, precision, expected, work in (
            (1, 1, True, 2),
            (1, 2, True, 11),
            (1, 3, False, 2),
            (16, 38, True, 821),
            (16, 39, False, 862),
            (2147483647, 2147483647, True, 2),
            (1, 2147483647, False, 2),
        ):
            with self.subTest(width=width, precision=precision):
                budget = FooterGrammarTests().budget()
                before = self.state(budget)
                actual = function(byte_width=width, precision=precision, budget=budget)
                self.assertIs(actual, expected)
                self.assertEqual(budget.semantic_work_units, work)
                after = self.state(budget)
                self.assertEqual(after[:7] + after[8:], before[:7] + before[8:])

    def test_decimal_capacity_work_boundary_is_atomic_and_cumulative(self) -> None:
        function = self.decimal_reader()
        for remaining, valid in ((10, False), (11, True)):
            with self.subTest(remaining=remaining):
                budget = FooterGrammarTests().budget(max_semantic_work_units=20)
                budget.reserve(bounded_parquet._FooterCharge(semantic_work_units=20 - remaining))
                before = self.state(budget)
                if valid:
                    self.assertIs(function(byte_width=1, precision=2, budget=budget), True)
                    self.assertEqual(budget.semantic_work_units, 20)
                else:
                    with self.assertRaises(PinnedReadRefused) as raised:
                        function(byte_width=1, precision=2, budget=budget)
                    self.assertEqual(raised.exception.reason_code, "resource_limit")
                    self.assertEqual(budget.semantic_work_units, before[7] + 2)
                after = self.state(budget)
                self.assertEqual(after[:7] + after[8:], before[:7] + before[8:])
        budget = FooterGrammarTests().budget()
        budget.push_schema()
        depth = (budget.active_schema_depth, budget.schema_depth_peak)
        self.assertIs(function(byte_width=1, precision=2, budget=budget), True)
        self.assertEqual((budget.active_schema_depth, budget.schema_depth_peak), depth)
        self.assertEqual(budget.semantic_work_units, 12)

    def test_decimal_capacity_rejects_invalid_contract_before_any_charge(self) -> None:
        function = self.decimal_reader()

        class IntegerSubclass(int):
            pass

        for name in ("byte_width", "precision"):
            for invalid in (True, False, 0, -1, 2147483648, 1.0, "1", None, IntegerSubclass(1)):
                with self.subTest(name=name, invalid_type=type(invalid).__name__):
                    budget = FooterGrammarTests().budget()
                    before = self.state(budget)
                    kwargs = {"byte_width": 1, "precision": 2, "budget": budget}
                    kwargs[name] = invalid
                    with self.assertRaises(PinnedReadRefused) as raised:
                        function(**kwargs)
                    self.assertEqual(raised.exception.reason_code, "invalid_contract")
                    self.assertEqual(self.state(budget), before)
        for invalid in (None, True, SimpleNamespace()):
            with self.subTest(budget_type=type(invalid).__name__):
                with self.assertRaises(PinnedReadRefused) as raised:
                    function(byte_width=1, precision=2, budget=invalid)
                self.assertEqual(raised.exception.reason_code, "invalid_contract")

    def test_decimal_capacity_large_ambiguous_work_refuses_without_spending_batch(self) -> None:
        function = self.decimal_reader()
        budget = FooterGrammarTests().budget()
        before = self.state(budget)
        with self.assertRaises(PinnedReadRefused) as raised:
            function(byte_width=536870912, precision=1073741824, budget=budget)
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(budget.semantic_work_units, 2)
        after = self.state(budget)
        self.assertEqual(after[:7] + after[8:], before[:7] + before[8:])


class _EnvelopeLoggedIO(_LoggedPinnedIO):
    """Independent positional responses; no complete file or data is assembled."""

    def __init__(self) -> None:
        super().__init__()
        self.before = self.metadata(st_size=128)
        self.after = self.metadata(st_size=128)
        self.located = self.metadata(st_size=128)
        self.responses = {
            (0, 4): b"PAR1",
            (120, 8): bytes.fromhex("0f 00 00 00 50 41 52 31"),
            (105, 15): _GRAMMAR_ANCHOR,
        }
        self.positional_requests: list[tuple[int, int, int]] = []
        self.close_attempts: list[int] = []
        self.close_errors: dict[int, OSError] = {}
        self.pread_error: OSError | None = None
        self.fail_pread = 0
        self.fstat_error = 0
        self.stat_error = False

    def open(self, path: str, flags: int, *, dir_fd: int | None = None) -> int:
        required = os.O_NONBLOCK if path == "neutral.parquet" else os.O_DIRECTORY
        if not flags & required:
            raise AssertionError("acquisition requires directory or nonblocking leaf flags")
        return super().open(path, flags, dir_fd=dir_fd)

    def close(self, fd: int) -> None:
        self.close_attempts.append(fd)
        self.event(f"close:{fd}")
        if fd in self.close_errors:
            raise self.close_errors[fd]
        self.closed.append(fd)

    def fstat(self, fd: int) -> SimpleNamespace:
        result = super().fstat(fd)
        if self.fstat_calls == self.fstat_error:
            raise OSError("neutral fstat failure")
        return result

    def stat(self, path: str, *, dir_fd: int, follow_symlinks: bool) -> SimpleNamespace:
        result = super().stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if self.stat_error:
            raise OSError("neutral stat failure")
        return result

    def pread(self, fd: int, length: int, offset: int) -> bytes:
        if fd not in self.acquired or fd in self.close_attempts:
            raise AssertionError("positional read requires live acquired leaf")
        self.positional_requests.append((fd, length, offset))
        self.event(f"pread:{len(self.positional_requests)}")
        if len(self.positional_requests) == self.fail_pread:
            raise self.pread_error or OSError("neutral positional failure")
        return self.responses[(offset, length)]

    def __enter__(self):
        self.module = importlib.import_module("liq.store.timestamp_metadata")
        self.identity = self.module.PinnedParquetFile(self.relative_path, self.digest)
        source_os = SimpleNamespace(
            O_RDONLY=os.O_RDONLY,
            O_DIRECTORY=os.O_DIRECTORY,
            O_NOFOLLOW=os.O_NOFOLLOW,
            O_NONBLOCK=os.O_NONBLOCK,
            open=self.open,
            close=self.close,
            fstat=self.fstat,
            stat=self.stat,
            read=self.read,
            pread=self.pread,
        )
        self.stack.enter_context(patch.object(self.module, "os", source_os))
        self.stack.enter_context(
            patch.object(
                self.module.os,
                "read",
                side_effect=AssertionError("sequential payload read forbidden"),
            )
        )
        self.stack.enter_context(
            patch.object(
                self.module.hashlib,
                "sha256",
                side_effect=AssertionError("partial source hash forbidden"),
            )
        )
        self.stack.enter_context(
            patch.object(
                self.module.pq, "ParquetFile", side_effect=AssertionError("native parser forbidden")
            )
        )
        return self


class FooterEnvelopeTests(unittest.TestCase):
    """Neutral ownership and range contracts; no source admission or serializer."""

    class IntegerSubclass(int):
        pass

    def api(self, module, *names: str):
        found = {}
        for name in names:
            value = getattr(module, name, None)
            self.assertTrue(callable(value), f"missing callable {name}")
            found[name] = value
        return SimpleNamespace(**found)

    def budget(self, io: _EnvelopeLoggedIO, **changes: int):
        limits = PinnedReadLimits(
            max_source_size_bytes=2 * 1024 * 1024,
            max_total_read_bytes=2 * 1024 * 1024,
            max_source_opens=20,
            max_decoded_rows=1,
            max_output_rows=1,
            max_row_groups=1,
            max_row_group_uncompressed_bytes=1,
            max_total_uncompressed_bytes=1,
            batch_rows=1,
            deadline_ns=100,
        )
        return bounded_parquet._PinnedReadBudget(replace(limits, **changes), clock_ns=io.clock)

    def owner(self, io: _EnvelopeLoggedIO, budget):
        api = self.api(io.module, "_pinned_source")
        return api._pinned_source(io.root, io.identity, 2 * 1024 * 1024, budget=budget)

    def envelope(self, io: _EnvelopeLoggedIO, budget=None, **changes: int):
        api = self.api(bounded_parquet, "_read_footer_envelope")
        values = {
            "maximum": 2 * 1024 * 1024,
            "max_footer_bytes": 1024,
            "max_retained_payload_bytes": 2048,
        }
        values.update(changes)
        return api._read_footer_envelope(
            io.root, io.identity, budget=self.budget(io) if budget is None else budget, **values
        )

    def continuation(self):
        return self.api(bounded_parquet, "_profile_footer_envelope")._profile_footer_envelope

    def assert_cleanup(self, io: _EnvelopeLoggedIO) -> None:
        self.assertCountEqual(io.close_attempts, io.acquired)
        self.assertEqual(len(io.close_attempts), len(set(io.close_attempts)))

    def assert_policy(self, reason: str, action, *args, **kwargs) -> None:
        with self.assertRaises(PinnedReadRefused) as raised:
            action(*args, **kwargs)
        self.assertEqual(raised.exception.reason_code, reason)

    def assert_integrity(self, io: _EnvelopeLoggedIO, reason: str, action, *args, **kwargs) -> None:
        with self.assertRaises(io.module.TimestampMetadataRefused) as raised:
            action(*args, **kwargs)
        self.assertEqual(str(raised.exception), reason)

    def test_owner_validates_before_acquisition(self) -> None:
        with _EnvelopeLoggedIO() as io:
            owner = self.api(io.module, "_pinned_source")._pinned_source
            budget = self.budget(io)
            cases = (
                (Path("relative"), io.identity, 128, budget),
                (Path("/sandbox/../other"), io.identity, 128, budget),
                (io.root, io.module.PinnedParquetFile("../neutral.parquet", "0" * 64), 128, budget),
                (io.root, io.module.PinnedParquetFile("neutral.parquet", "bad"), 128, budget),
                (io.root, io.identity, True, budget),
                (io.root, io.identity, 0, budget),
                (io.root, io.identity, self.IntegerSubclass(128), budget),
                (io.root, io.identity, 128, "invalid"),
            )
            for root, identity, maximum, selected in cases:
                with self.subTest(root=root, maximum=maximum):
                    with (
                        self.assertRaises(PinnedReadRefused) as raised,
                        owner(root, identity, maximum, budget=selected),
                    ):
                        self.fail("invalid owner yielded")
                    self.assertEqual(raised.exception.reason_code, "invalid_contract")
                    self.assertEqual(io.events, [])

    def test_owner_counts_successful_and_failed_open_attempts(self) -> None:
        for failure in range(1, 5):
            with self.subTest(failed_open=failure), _EnvelopeLoggedIO() as io:
                self.api(io.module, "_pinned_source")
                budget = self.budget(io)
                io.fail_open = failure
                with self.assertRaises(OSError), self.owner(io, budget):
                    self.fail("failed acquisition yielded")
                self.assertEqual(budget.source_opens, failure)
                self.assertEqual(len(io.open_paths), failure)
                self.assert_cleanup(io)
        for cap in (1, 2, 3):
            with self.subTest(open_cap=cap), _EnvelopeLoggedIO() as io:
                self.api(io.module, "_pinned_source")
                budget = self.budget(io, max_source_opens=cap)
                with self.assertRaises(PinnedReadRefused) as raised, self.owner(io, budget):
                    self.fail("exhausted acquisition yielded")
                self.assertEqual(raised.exception.reason_code, "resource_limit")
                self.assertEqual((budget.source_opens, len(io.open_paths)), (cap, cap))
                self.assert_cleanup(io)
        for stage in ("open:1", "open:2", "open:3", "open:4", "fstat:1"):
            with self.subTest(deadline=stage), _EnvelopeLoggedIO() as io:
                self.api(io.module, "_pinned_source")
                io.expire_after = stage
                with (
                    self.assertRaises(PinnedReadRefused) as raised,
                    self.owner(io, self.budget(io)),
                ):
                    self.fail("expired acquisition yielded")
                self.assertEqual(raised.exception.reason_code, "deadline_exceeded")
                self.assertEqual(io.positional_requests, [])
                self.assert_cleanup(io)

    def test_owner_lifetime_and_borrowing(self) -> None:
        with _EnvelopeLoggedIO() as io:
            self.api(io.module, "_pinned_source")
            budget = self.budget(io)
            with self.owner(io, budget) as source:
                self.assertEqual(source.borrow_fd(), 14)
                self.assertEqual(source.size, 128)
                self.assertEqual(source.identity, (1, 2, 128, 3, 4))
                self.assertIs(source.budget, budget)
                self.assertEqual(io.close_attempts, [11, 12])
                for private in (
                    str(io.root),
                    io.relative_path,
                    "neutral.parquet",
                    str(source.identity),
                ):
                    self.assertNotIn(private, repr(source))
            self.assertEqual(io.open_paths, ["/", "sandbox", "nested", "neutral.parquet"])
            self.assertEqual(budget.source_opens, 4)
            self.assertEqual(io.close_attempts, [11, 12, 14, 13])
            self.assertLess(io.events.index("fstat:2"), io.events.index("stat"))
            self.assertLess(io.events.index("stat"), io.events.index("close:14"))
            self.assert_policy("invalid_contract", source.borrow_fd)
            self.assertEqual(io.positional_requests, [])

    def test_owner_distinguishes_source_and_read_limits(self) -> None:
        with _EnvelopeLoggedIO() as io:
            self.api(io.module, "_pinned_source")
            budget = self.budget(io, max_total_read_bytes=1)
            with self.owner(io, budget) as source:
                self.assertEqual(source.size, 128)
            self.assertEqual(budget.total_read_bytes, 0)
        for maximum, cap in ((127, 128), (128, 127)):
            with self.subTest(maximum=maximum, cap=cap), _EnvelopeLoggedIO() as io:
                owner = self.api(io.module, "_pinned_source")._pinned_source
                budget = self.budget(io, max_source_size_bytes=cap)
                with (
                    self.assertRaises(PinnedReadRefused) as raised,
                    owner(io.root, io.identity, maximum, budget=budget),
                ):
                    self.fail("oversized source yielded")
                self.assertEqual(raised.exception.reason_code, "resource_limit")
                self.assertEqual(io.positional_requests, [])
                self.assert_cleanup(io)
        with _EnvelopeLoggedIO() as io:
            self.api(io.module, "_pinned_source")
            budget = self.budget(io, max_total_read_bytes=127)
            self.assert_policy(
                "resource_limit", io.module._pinned_bytes, io.root, io.identity, 128, budget=budget
            )
            self.assertEqual(budget.total_read_bytes, 0)
            self.assert_cleanup(io)

    def test_owner_preserves_consumer_errors(self) -> None:
        with _EnvelopeLoggedIO() as io:
            self.api(io.module, "_pinned_source")
            failure = RuntimeError("neutral consumer failure")
            with (
                self.assertRaises(RuntimeError) as raised,
                self.owner(io, self.budget(io)) as source,
            ):
                raise failure
            self.assertIs(raised.exception, failure)
            self.assertEqual(io.fstat_calls, 1)
            self.assertNotIn("stat", io.events)
            self.assert_cleanup(io)
            self.assert_policy("invalid_contract", source.borrow_fd)

    def test_owner_checks_final_identity_before_return(self) -> None:
        for stage in ("after", "located"):
            for field_name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
                with self.subTest(stage=stage, field=field_name), _EnvelopeLoggedIO() as io:
                    self.api(io.module, "_pinned_source")
                    metadata = getattr(io, stage)
                    setattr(metadata, field_name, getattr(metadata, field_name) + 1)
                    with (
                        self.assertRaises(io.module.TimestampMetadataRefused),
                        self.owner(io, self.budget(io)),
                    ):
                        pass
                    self.assert_cleanup(io)
        for stage in ("fstat", "stat"):
            with self.subTest(failure=stage), _EnvelopeLoggedIO() as io:
                self.api(io.module, "_pinned_source")
                io.fstat_error = 2 if stage == "fstat" else 0
                io.stat_error = stage == "stat"
                with self.assertRaises(OSError), self.owner(io, self.budget(io)):
                    pass
                self.assert_cleanup(io)

    def test_owner_tracks_child_before_parent_close_failure(self) -> None:
        for child_fails in (False, True):
            with self.subTest(child_fails=child_fails), _EnvelopeLoggedIO() as io:
                self.api(io.module, "_pinned_source")
                failure = OSError("neutral parent close failure")
                child_failure = OSError("neutral child cleanup failure")
                io.close_errors[11] = failure
                if child_fails:
                    io.close_errors[12] = child_failure
                with self.assertRaises(OSError) as raised, self.owner(io, self.budget(io)):
                    self.fail("failed parent close yielded")
                self.assertIs(raised.exception, failure)
                self.assertIs(failure.__cause__, child_failure if child_fails else None)
                self.assertEqual(io.acquired, [11, 12])
                self.assertEqual(io.close_attempts, [11, 12])
                self.assertEqual(io.closed, [] if child_fails else [12])
                self.assertEqual(io.positional_requests, [])

    def test_owner_retains_primary_and_cleanup_failures(self) -> None:
        for consumer_fails in (False, True):
            with self.subTest(consumer_fails=consumer_fails), _EnvelopeLoggedIO() as io:
                self.api(io.module, "_pinned_source")
                consumer = OSError("neutral operation failure")
                leaf = OSError("neutral leaf close failure")
                parent = OSError("neutral final parent failure")
                io.close_errors = {14: leaf, 13: parent}
                with (
                    self.assertRaises(OSError) as raised,
                    self.owner(io, self.budget(io)) as source,
                ):
                    if consumer_fails:
                        raise consumer
                primary = consumer if consumer_fails else leaf
                self.assertIs(raised.exception, primary)
                chain = []
                current = primary.__cause__
                for _ in range(3):
                    if current is None:
                        break
                    chain.append(current)
                    current = current.__cause__
                self.assertEqual(chain, [] if consumer_fails else [parent])
                state = BaseException.__dict__["__dict__"].__get__(primary, BaseException)
                retained = state[io.module._PINNED_CLEANUP_ERRORS]
                self.assertIs(type(retained), tuple)
                self.assertEqual(len(retained), 2 if consumer_fails else 1)
                self.assertIs(retained[0], leaf if consumer_fails else parent)
                if consumer_fails:
                    self.assertIs(retained[1], parent)
                    self.assertIsNone(leaf.__cause__)
                    self.assertIsNone(parent.__cause__)
                    self.assertIs(leaf.__context__, consumer)
                    self.assertIs(parent.__context__, consumer)
                self.assertIsNone(current)
                self.assert_cleanup(io)
                self.assert_policy("invalid_contract", source.borrow_fd)

    def test_owner_parent_close_failure_remains_primary(self) -> None:
        with _EnvelopeLoggedIO() as io:
            self.api(io.module, "_pinned_source")
            parent_failure = OSError("neutral final parent close failure")
            io.close_errors[13] = parent_failure

            with self.assertRaises(OSError) as raised, self.owner(io, self.budget(io)) as source:
                pass

            self.assertIs(raised.exception, parent_failure)
            self.assertIs(type(raised.exception), OSError)
            self.assertIsNone(parent_failure.__cause__)
            self.assertIsNone(parent_failure.__context__)
            state = BaseException.__dict__["__dict__"].__get__(parent_failure, BaseException)
            self.assertNotIn(io.module._PINNED_CLEANUP_ERRORS, state)
            self.assertEqual(io.acquired, [11, 12, 13, 14])
            self.assertEqual(io.close_attempts, [11, 12, 14, 13])
            self.assertEqual(io.closed, [11, 12, 14])
            self.assertLess(io.events.index("stat"), io.events.index("close:14"))
            self.assertEqual(io.positional_requests, [])
            self.assert_cleanup(io)
            self.assert_policy("invalid_contract", source.borrow_fd)

    def test_range_rejects_invalid_handle_and_bounds(self) -> None:
        with _EnvelopeLoggedIO() as io:
            reader = self.api(io.module, "_pinned_source", "_read_pinned_range")._read_pinned_range
            budget = self.budget(io)
            with self.owner(io, budget) as source:
                for offset, length in (
                    (-1, 1),
                    (0, 0),
                    (128, 1),
                    (127, 2),
                    (True, 1),
                    (0, True),
                    (0.0, 1),
                    (0, 1.0),
                    (self.IntegerSubclass(0), 1),
                    (0, self.IntegerSubclass(1)),
                ):
                    with self.subTest(offset=offset, length=length):
                        before = list(io.events)
                        self.assert_policy(
                            "invalid_contract", reader, source, offset=offset, length=length
                        )
                        self.assertEqual(io.events, before)
                self.assert_policy("invalid_contract", reader, None, offset=0, length=1)
            self.assert_policy("invalid_contract", reader, source, offset=0, length=1)
            self.assertEqual(budget.total_read_bytes, 0)
            self.assertEqual(io.positional_requests, [])
        with _EnvelopeLoggedIO() as io:
            reader = self.api(io.module, "_pinned_source", "_read_pinned_range")._read_pinned_range
            with self.owner(io, None) as source:
                self.assert_policy("invalid_contract", reader, source, offset=0, length=1)
            self.assertEqual(io.positional_requests, [])

    def test_range_single_request_preserves_bytes_identity(self) -> None:
        with _EnvelopeLoggedIO() as io:
            reader = self.api(io.module, "_pinned_source", "_read_pinned_range")._read_pinned_range
            budget = self.budget(io)
            expected = io.responses[(105, 15)]
            with self.owner(io, budget) as source:
                result = reader(source, offset=105, length=15)
                self.assertIs(result, expected)
            self.assertEqual(io.positional_requests, [(14, 15, 105)])
            self.assertEqual(budget.total_read_bytes, 15)
            self.assert_cleanup(io)

    def test_range_prechecks_cumulative_and_request_caps(self) -> None:
        for available in (14, 15):
            with self.subTest(available=available), _EnvelopeLoggedIO() as io:
                reader = self.api(
                    io.module, "_pinned_source", "_read_pinned_range"
                )._read_pinned_range
                budget = self.budget(io, max_total_read_bytes=available + 3)
                budget.record_read(3)
                with self.owner(io, budget) as source:
                    if available == 14:
                        self.assert_policy("resource_limit", reader, source, offset=105, length=15)
                        self.assertEqual(io.positional_requests, [])
                        self.assertEqual(budget.total_read_bytes, 3)
                    else:
                        self.assertIs(reader(source, offset=105, length=15), _GRAMMAR_ANCHOR)
                        self.assertEqual(budget.total_read_bytes, 18)
        for requested in (1024 * 1024, 1024 * 1024 + 1):
            with self.subTest(requested=requested), _EnvelopeLoggedIO() as io:
                reader = self.api(
                    io.module, "_pinned_source", "_read_pinned_range"
                )._read_pinned_range
                for metadata in (io.before, io.after, io.located):
                    metadata.st_size = 2 * 1024 * 1024
                io.responses[(0, requested)] = b"n"
                budget = self.budget(io)
                with self.owner(io, budget) as source:
                    if requested == 1024 * 1024:
                        self.assert_integrity(
                            io,
                            "incomplete bounded range",
                            reader,
                            source,
                            offset=0,
                            length=requested,
                        )
                        self.assertEqual(io.positional_requests, [(14, requested, 0)])
                        self.assertEqual(budget.total_read_bytes, 1)
                    else:
                        self.assert_policy(
                            "resource_limit", reader, source, offset=0, length=requested
                        )
                        self.assertEqual(io.positional_requests, [])
                        self.assertEqual(budget.total_read_bytes, 0)

    def test_range_short_or_zero_result_refuses_without_retry(self) -> None:
        for payload in (b"", b"n"):
            with self.subTest(size=len(payload)), _EnvelopeLoggedIO() as io:
                reader = self.api(
                    io.module, "_pinned_source", "_read_pinned_range"
                )._read_pinned_range
                io.responses[(105, 15)] = payload
                budget = self.budget(io)
                with self.owner(io, budget) as source:
                    self.assert_integrity(
                        io, "incomplete bounded range", reader, source, offset=105, length=15
                    )
                self.assertEqual(io.positional_requests, [(14, 15, 105)])
                self.assertEqual(budget.total_read_bytes, len(payload))

    def test_range_deadlines_and_os_failures_preserve_accounting(self) -> None:
        for stage in ("before", "after", "os"):
            with self.subTest(stage=stage), _EnvelopeLoggedIO() as io:
                reader = self.api(
                    io.module, "_pinned_source", "_read_pinned_range"
                )._read_pinned_range
                budget = self.budget(io)
                failure = OSError("neutral pread failure")
                with (
                    self.assertRaises(OSError if stage == "os" else PinnedReadRefused) as raised,
                    self.owner(io, budget) as source,
                ):
                    io.expired = stage == "before"
                    io.expire_after = "pread:1" if stage == "after" else ""
                    io.fail_pread = 1 if stage == "os" else 0
                    io.pread_error = failure
                    reader(source, offset=105, length=15)
                if stage == "os":
                    self.assertIs(raised.exception, failure)
                else:
                    self.assertEqual(raised.exception.reason_code, "deadline_exceeded")
                self.assertEqual(len(io.positional_requests), 0 if stage == "before" else 1)
                self.assertEqual(budget.total_read_bytes, 15 if stage == "after" else 0)
                self.assert_cleanup(io)

    def test_envelope_validates_new_contract_before_callbacks(self) -> None:
        with _EnvelopeLoggedIO() as io:
            reader = self.api(bounded_parquet, "_read_footer_envelope")._read_footer_envelope
            budget = self.budget(io)
            for name in ("maximum", "max_footer_bytes", "max_retained_payload_bytes"):
                for invalid in (0, -1, True, 1.0, self.IntegerSubclass(1)):
                    with self.subTest(name=name, value=invalid):
                        self.assert_policy(
                            "invalid_contract", self.envelope, io, budget, **{name: invalid}
                        )
                        self.assertEqual(io.events, [])
            for root, identity, selected in (
                (Path("relative"), io.identity, budget),
                (io.root, io.module.PinnedParquetFile("neutral.parquet", "bad"), budget),
                (io.root, io.identity, None),
            ):
                with self.subTest(root=root, valid_budget=selected is budget):
                    self.assert_policy(
                        "invalid_contract",
                        reader,
                        root,
                        identity,
                        maximum=128,
                        max_footer_bytes=15,
                        max_retained_payload_bytes=27,
                        budget=selected,
                    )
                    self.assertEqual(io.events, [])

    def test_envelope_exact_ranges_and_cumulative_receipt(self) -> None:
        with _EnvelopeLoggedIO() as io:
            self.api(
                bounded_parquet,
                "_read_footer_envelope",
                "_FooterEnvelope",
                "_FooterEnvelopeReceipt",
            )
            budget = self.budget(io)
            budget.before_open()
            budget.record_read(3)
            envelope = self.envelope(io, budget, max_footer_bytes=15, max_retained_payload_bytes=27)
            self.assertIs(envelope.payload, _GRAMMAR_ANCHOR)
            self.assertEqual(
                (
                    envelope.extent.source_size,
                    envelope.extent.footer_start,
                    envelope.extent.footer_end,
                ),
                (128, 105, 120),
            )
            self.assertEqual(envelope.source_identity, (1, 2, 128, 3, 4))
            self.assertIs(envelope.read_budget, budget)
            receipt = envelope.receipt
            self.assertEqual(
                (
                    receipt.source_opens_before,
                    receipt.source_opens_after,
                    receipt.total_read_bytes_before,
                    receipt.total_read_bytes_after,
                ),
                (1, 5, 3, 30),
            )
            self.assertEqual(receipt.ranges, ((0, 4), (120, 128), (105, 120)))
            self.assertIs(receipt.whole_source_digest_verified, False)
            self.assertEqual(io.positional_requests, [(14, 4, 0), (14, 8, 120), (14, 15, 105)])
            with self.assertRaises(FrozenInstanceError):
                envelope.payload = b"n"
            with self.assertRaises(FrozenInstanceError):
                receipt.whole_source_digest_verified = True
            self.assert_cleanup(io)

    def test_envelope_prechecks_framing_and_footer_allowances(self) -> None:
        cases = (
            (11, 100, 100, 0),
            (100, 11, 100, 0),
            (26, 100, 100, 2),
            (100, 26, 100, 2),
            (100, 100, 14, 2),
        )
        for read_cap, retained, footer_cap, expected_calls in cases:
            with self.subTest(caps=(read_cap, retained, footer_cap)), _EnvelopeLoggedIO() as io:
                self.api(bounded_parquet, "_read_footer_envelope")
                budget = self.budget(io, max_total_read_bytes=read_cap)
                self.assert_policy(
                    "resource_limit",
                    self.envelope,
                    io,
                    budget,
                    max_footer_bytes=footer_cap,
                    max_retained_payload_bytes=retained,
                )
                self.assertEqual(len(io.positional_requests), expected_calls)
                self.assertEqual(budget.total_read_bytes, 12 if expected_calls else 0)
                self.assert_cleanup(io)

    def test_envelope_rejects_invalid_length_extents(self) -> None:
        for size in (1, 11):
            with self.subTest(source_size=size), _EnvelopeLoggedIO() as io:
                self.api(bounded_parquet, "_read_footer_envelope")
                for metadata in (io.before, io.after, io.located):
                    metadata.st_size = size
                self.assert_integrity(io, "footer envelope length", self.envelope, io)
                self.assertEqual(io.positional_requests, [])
                self.assert_cleanup(io)
        for length in (0, 117, 2**32 - 1):
            with self.subTest(footer_length=length), _EnvelopeLoggedIO() as io:
                self.api(bounded_parquet, "_read_footer_envelope")
                io.responses[(120, 8)] = length.to_bytes(4, "little") + b"PAR1"
                self.assert_integrity(io, "footer envelope length", self.envelope, io)
                self.assertEqual(len(io.positional_requests), 2)
                self.assert_cleanup(io)

    def test_envelope_read_failures_prevent_construction(self) -> None:
        for stage in (1, 2, 3):
            for failure_kind in ("os", "positive_short", "zero_short", "deadline"):
                with self.subTest(stage=stage, failure=failure_kind), _EnvelopeLoggedIO() as io:
                    self.api(
                        bounded_parquet,
                        "_read_footer_envelope",
                        "_FooterEnvelope",
                        "_FooterEnvelopeReceipt",
                    )
                    failure = OSError("neutral envelope read failure")
                    ranges = ((0, 4), (120, 8), (105, 15))
                    completed_bytes = (0, 4, 12)[stage - 1]
                    returned_bytes = ranges[stage - 1][1]
                    error_type = io.module.TimestampMetadataRefused
                    if failure_kind == "os":
                        io.fail_pread = stage
                        io.pread_error = failure
                        returned_bytes = 0
                        error_type = OSError
                    elif failure_kind == "deadline":
                        io.expire_after = f"pread:{stage}"
                        error_type = PinnedReadRefused
                    else:
                        payload = b"n" if failure_kind == "positive_short" else b""
                        io.responses[ranges[stage - 1]] = payload
                        returned_bytes = len(payload)
                    budget = self.budget(io)
                    with (
                        patch.object(
                            bounded_parquet,
                            "_FooterEnvelope",
                            side_effect=AssertionError("failed read published envelope"),
                        ) as constructor,
                        patch.object(
                            bounded_parquet,
                            "_FooterEnvelopeReceipt",
                            side_effect=AssertionError("failed read published receipt"),
                        ) as receipt,
                        self.assertRaises(error_type) as raised,
                    ):
                        self.envelope(io, budget)
                    if failure_kind == "os":
                        self.assertIs(raised.exception, failure)
                    elif failure_kind == "deadline":
                        self.assertEqual(raised.exception.reason_code, "deadline_exceeded")
                    else:
                        self.assertEqual(str(raised.exception), "incomplete bounded range")
                    constructor.assert_not_called()
                    receipt.assert_not_called()
                    self.assertEqual(
                        io.positional_requests,
                        [(14, length, offset) for offset, length in ranges[:stage]],
                    )
                    self.assertEqual(budget.total_read_bytes, completed_bytes + returned_bytes)
                    self.assert_cleanup(io)

    def test_envelope_identity_close_and_unknown_digest(self) -> None:
        for outcome in ("success", "identity", "close"):
            with self.subTest(outcome=outcome), _EnvelopeLoggedIO() as io:
                api = self.api(
                    bounded_parquet,
                    "_read_footer_envelope",
                    "_FooterEnvelope",
                    "_FooterEnvelopeReceipt",
                )
                private_payload = b"neutral-opaque-footer-sentinel"
                footer_length = len(private_payload)
                io.responses[(120, 8)] = footer_length.to_bytes(4, "little") + b"PAR1"
                io.responses[(120 - footer_length, footer_length)] = private_payload
                if outcome == "identity":
                    io.after.st_ino += 1
                if outcome == "close":
                    io.close_errors[14] = OSError("neutral close refusal")
                constructions = []

                def envelope_record(
                    *args, io=io, constructions=constructions, record=api._FooterEnvelope, **kwargs
                ):
                    self.assert_cleanup(io)
                    self.assertIn("stat", io.events)
                    constructions.append("envelope")
                    return record(*args, **kwargs)

                def receipt_record(
                    *args,
                    io=io,
                    constructions=constructions,
                    record=api._FooterEnvelopeReceipt,
                    **kwargs,
                ):
                    self.assert_cleanup(io)
                    constructions.append("receipt")
                    return record(*args, **kwargs)

                with (
                    patch.object(bounded_parquet, "_FooterEnvelope", side_effect=envelope_record),
                    patch.object(
                        bounded_parquet, "_FooterEnvelopeReceipt", side_effect=receipt_record
                    ),
                ):
                    if outcome == "success":
                        envelope = self.envelope(io)
                        self.assertIs(envelope.receipt.whole_source_digest_verified, False)
                        self.assertIs(envelope.payload, private_payload)
                        self.assertEqual(constructions, ["receipt", "envelope"])
                        for name in ("fd", "path", "header", "trailer"):
                            self.assertFalse(hasattr(envelope, name))
                        for record in (envelope, envelope.receipt):
                            for private in (
                                private_payload.decode("ascii"),
                                str(io.root),
                                io.relative_path,
                                "neutral.parquet",
                                str(envelope.source_identity),
                            ):
                                self.assertNotIn(private, repr(record))
                    else:
                        with self.assertRaises(
                            OSError if outcome == "close" else io.module.TimestampMetadataRefused
                        ):
                            self.envelope(io)
                        self.assertEqual(constructions, [])
                self.assert_cleanup(io)

    def test_envelope_magic_controls_refusal_stage(self) -> None:
        for position, value, expected_calls in (
            ("head", b"PARE", 1),
            ("tail", b"PARE", 2),
            ("head", b"nope", 1),
            ("tail", b"nope", 2),
        ):
            with self.subTest(position=position, value=value), _EnvelopeLoggedIO() as io:
                self.api(bounded_parquet, "_read_footer_envelope")
                if position == "head":
                    io.responses[(0, 4)] = value
                else:
                    io.responses[(120, 8)] = bytes.fromhex("0f 00 00 00") + value
                if value == b"PARE":
                    self.assert_policy("unsupported_mode", self.envelope, io)
                else:
                    self.assert_integrity(io, "footer envelope magic", self.envelope, io)
                self.assertEqual(len(io.positional_requests), expected_calls)
                self.assert_cleanup(io)

    def test_envelope_preserves_bounded_refusal_types(self) -> None:
        for metadata in ({"st_mode": stat.S_IFDIR | 0o700}, {"st_nlink": 2}, {"st_size": 0}):
            with self.subTest(metadata=metadata), _EnvelopeLoggedIO() as io:
                self.api(bounded_parquet, "_read_footer_envelope")
                io.before = io.metadata(**metadata)
                with self.assertRaises(io.module.TimestampMetadataRefused):
                    self.envelope(io)
                self.assertEqual(io.positional_requests, [])
                self.assert_cleanup(io)
        with _EnvelopeLoggedIO() as io:
            self.api(bounded_parquet, "_read_footer_envelope")
            io.responses[(120, 8)] = b"n"
            budget = self.budget(io)
            self.assert_integrity(io, "incomplete bounded range", self.envelope, io, budget)
            self.assertEqual(budget.total_read_bytes, 5)
            self.assert_cleanup(io)

    def test_footer_extra_tail_refuses_without_retry_or_signature_claim(self) -> None:
        with _EnvelopeLoggedIO() as io:
            function = self.continuation()
            self.api(bounded_parquet, "_read_footer_envelope")
            io.responses[(120, 8)] = bytes.fromhex("10 00 00 00 50 41 52 31")
            io.responses[(104, 16)] = _GRAMMAR_ANCHOR + b"\x00"
            envelope = self.envelope(io)
            before = list(io.positional_requests)
            with self.assertRaises(bounded_parquet._CompactWireMalformed) as raised:
                function(envelope, budget=FooterGrammarTests().budget())
            self.assertEqual(raised.exception.reason_code, "invalid_wire")
            self.assertEqual(io.positional_requests, before)
            self.assert_cleanup(io)

    def test_footer_continuation_preserves_payload_and_budget_identity(self) -> None:
        with _EnvelopeLoggedIO() as io:
            function = self.continuation()
            envelope = self.envelope(io)
            budget = FooterGrammarTests().budget()
            budget.reserve(bounded_parquet._FooterCharge(semantic_work_units=3))
            budget.wire_budget._reserve(work_units=2)
            profile = function(envelope, budget=budget)
            self.assertIs(profile.document.payload, envelope.payload)
            self.assertIs(profile.document.budget, budget)
            self.assertIs(profile.document.budget.wire_budget, budget.wire_budget)
            self.assertGreater(budget.semantic_work_units, 3)
            self.assertGreater(budget.wire_budget.work_units, 2)
            self.assertIs(profile.read_eligible, False)
            self.assertGreaterEqual(profile.flags, FooterProfileTests.MANDATORY)
            self.assertEqual(len(io.positional_requests), 3)
            self.assert_cleanup(io)

    def test_footer_continuation_starts_after_source_closure(self) -> None:
        with _EnvelopeLoggedIO() as io:
            function = self.continuation()
            envelope = self.envelope(io)
            parser = bounded_parquet._parse_footer
            calls = []

            def observed(cursor, extent, budget):
                self.assert_cleanup(io)
                self.assertIn("stat", io.events)
                self.assertIs(cursor.payload, envelope.payload)
                self.assertEqual((cursor.start, cursor.stop), (0, 15))
                self.assertIs(extent, envelope.extent)
                self.assertIs(cursor.budget, budget.wire_budget)
                calls.append("parse")
                return parser(cursor, extent, budget)

            with patch.object(bounded_parquet, "_parse_footer", side_effect=observed):
                result = function(envelope, budget=FooterGrammarTests().budget())
            self.assertEqual(calls, ["parse"])
            self.assertIs(result.document.payload, envelope.payload)

    def test_footer_continuation_rejects_invalid_entry_contracts(self) -> None:
        with _EnvelopeLoggedIO() as io:
            function = self.continuation()
            envelope = self.envelope(io)
            budget = FooterGrammarTests().budget()
            invalids = (
                None,
                replace(envelope, payload="n"),
                replace(envelope, payload=b"n"),
                replace(envelope, read_budget=None),
                replace(
                    envelope, receipt=replace(envelope.receipt, whole_source_digest_verified=True)
                ),
            )
            for invalid in invalids:
                with (
                    self.subTest(invalid_type=type(invalid).__name__),
                    patch.object(
                        bounded_parquet,
                        "_parse_footer",
                        side_effect=AssertionError("invalid continuation parsed"),
                    ) as parser,
                ):
                    self.assert_policy("invalid_contract", function, invalid, budget=budget)
                    parser.assert_not_called()
            with patch.object(
                bounded_parquet,
                "_parse_footer",
                side_effect=AssertionError("invalid budget parsed"),
            ):
                self.assert_policy("invalid_contract", function, envelope, budget=None)

    def test_footer_continuation_preserves_failures_without_output(self) -> None:
        for stage in ("parse", "profile", "before", "between", "after"):
            with self.subTest(stage=stage), _EnvelopeLoggedIO() as io:
                function = self.continuation()
                envelope = self.envelope(io)
                parse = bounded_parquet._parse_footer
                complete = bounded_parquet._complete_footer_profile
                failure = bounded_parquet._CompactWireMalformed("schema_mismatch")
                calls = []

                def observed_parse(
                    *args, calls=calls, stage=stage, failure=failure, parse=parse, io=io, **kwargs
                ):
                    calls.append("parse")
                    if stage == "parse":
                        raise failure
                    result = parse(*args, **kwargs)
                    if stage == "between":
                        io.expired = True
                    return result

                def observed_profile(
                    *args,
                    calls=calls,
                    stage=stage,
                    failure=failure,
                    complete=complete,
                    io=io,
                    **kwargs,
                ):
                    calls.append("profile")
                    if stage == "profile":
                        raise failure
                    result = complete(*args, **kwargs)
                    if stage == "after":
                        io.expired = True
                    return result

                io.expired = stage == "before"
                with (
                    patch.object(bounded_parquet, "_parse_footer", side_effect=observed_parse),
                    patch.object(
                        bounded_parquet, "_complete_footer_profile", side_effect=observed_profile
                    ),
                ):
                    with self.assertRaises(
                        bounded_parquet._CompactWireMalformed
                        if stage in ("parse", "profile")
                        else PinnedReadRefused
                    ) as raised:
                        function(envelope, budget=FooterGrammarTests().budget())
                    if stage in ("parse", "profile"):
                        self.assertIs(raised.exception, failure)
                    else:
                        self.assertEqual(raised.exception.reason_code, "deadline_exceeded")
                self.assertEqual(
                    calls,
                    []
                    if stage == "before"
                    else ["parse"]
                    if stage in ("parse", "between")
                    else ["parse", "profile"],
                )
                self.assertEqual(len(io.positional_requests), 3)

    def test_footer_continuation_consumes_remaining_budgets(self) -> None:
        for axis in ("wire", "semantic"):
            with self.subTest(axis=axis), _EnvelopeLoggedIO() as io:
                function = self.continuation()
                envelope = self.envelope(io)
                budget = FooterGrammarTests().budget()
                if axis == "semantic":
                    budget.reserve(
                        bounded_parquet._FooterCharge(
                            semantic_work_units=budget.limits.max_semantic_work_units
                        )
                    )
                else:
                    budget.wire_budget._reserve(
                        wire_bytes=budget.wire_budget.limits.max_total_wire_bytes
                    )
                with patch.object(
                    bounded_parquet,
                    "_complete_footer_profile",
                    side_effect=AssertionError("exhausted parse completed Profile"),
                ) as profile:
                    self.assert_policy("resource_limit", function, envelope, budget=budget)
                    profile.assert_not_called()
                if axis == "semantic":
                    self.assertEqual(
                        budget.semantic_work_units, budget.limits.max_semantic_work_units
                    )
                else:
                    self.assertEqual(
                        budget.wire_budget.total_wire_bytes,
                        budget.wire_budget.limits.max_total_wire_bytes,
                    )


class PageBudgetTests(unittest.TestCase):
    axes = (
        "pages",
        "compressed_bytes",
        "uncompressed_bytes",
        "values",
        "levels",
        "dictionary_entries",
        "dictionary_payload_bytes",
        "codec_scratch_bytes",
        "output_rows",
        "output_bytes",
        "allocation_bytes",
        "work_units",
    )

    class IntegerSubclass(int):
        pass

    def api(self) -> SimpleNamespace:
        names = ("_PageLimits", "_PageCharge", "_PageBudget")
        found = {}
        for name in names:
            value = getattr(bounded_parquet, name, None)
            self.assertTrue(callable(value), f"missing callable {name}")
            found[name] = value
        return SimpleNamespace(**found)

    def budget(self, **changes: int) -> bounded_parquet._PageBudget:
        api = self.api()
        limits = dict.fromkeys(("max_" + axis for axis in self.axes), 10)
        limits.update(changes)
        read_budget = bounded_parquet._PinnedReadBudget(
            PinnedReadLimits(
                max_source_size_bytes=100,
                max_total_read_bytes=100,
                max_source_opens=10,
                max_decoded_rows=100,
                max_output_rows=100,
                max_row_groups=10,
                max_row_group_uncompressed_bytes=100,
                max_total_uncompressed_bytes=100,
                batch_rows=10,
                deadline_ns=10,
            ),
            clock_ns=lambda: 0,
        )
        return api._PageBudget(
            read_budget=read_budget,
            wire_budget=CompactWireCursorTests().budget(),
            limits=api._PageLimits(**limits),
        )

    def state(self, budget: bounded_parquet._PageBudget) -> tuple[int, ...]:
        return tuple(getattr(budget, axis) for axis in self.axes)

    def test_page_limits_require_exact_positive_integers_on_every_axis(self) -> None:
        api = self.api()
        for axis in self.axes:
            for invalid in (True, False, 0, -1, 1.0, "1", None, self.IntegerSubclass(1)):
                with self.subTest(axis=axis, invalid=invalid):
                    limits: dict[str, int | float | str | None] = dict.fromkeys(
                        ("max_" + name for name in self.axes), 1
                    )
                    limits["max_" + axis] = invalid
                    with self.assertRaises(PinnedReadRefused) as raised:
                        api._PageLimits(**limits)
                    self.assertEqual(raised.exception.reason_code, "invalid_contract")

    def test_page_charges_require_exact_nonnegative_integers_on_every_axis(self) -> None:
        api = self.api()
        for axis in self.axes:
            for invalid in (True, False, -1, 0.0, "0", None, self.IntegerSubclass(0)):
                with self.subTest(axis=axis, invalid=invalid):
                    with self.assertRaises(PinnedReadRefused) as raised:
                        api._PageCharge(**{axis: invalid})
                    self.assertEqual(raised.exception.reason_code, "invalid_contract")

    def test_page_declarations_are_frozen_on_every_axis(self) -> None:
        api = self.api()
        limits = api._PageLimits(**dict.fromkeys(("max_" + axis for axis in self.axes), 1))
        charge = api._PageCharge()
        for axis in self.axes:
            for record, field in ((limits, "max_" + axis), (charge, axis)):
                with (
                    self.subTest(axis=axis, field=field),
                    self.assertRaises(FrozenInstanceError),
                ):
                    setattr(record, field, 2)

    def test_page_budget_retains_supplied_objects_and_starts_at_zero(self) -> None:
        api = self.api()
        supplied = self.budget()
        budget = api._PageBudget(
            read_budget=supplied.read_budget,
            wire_budget=supplied.wire_budget,
            limits=supplied.limits,
        )
        self.assertIs(budget.read_budget, supplied.read_budget)
        self.assertIs(budget.wire_budget, supplied.wire_budget)
        self.assertIs(budget.limits, supplied.limits)
        self.assertEqual(self.state(budget), (0,) * 12)

    def test_page_budget_requires_exact_budget_and_limit_types(self) -> None:
        api = self.api()
        budget = self.budget()

        class ReadSubclass(bounded_parquet._PinnedReadBudget):
            pass

        class WireSubclass(bounded_parquet._CompactWireBudget):
            pass

        class LimitsSubclass(api._PageLimits):
            pass

        subclasses = {
            "read_budget": ReadSubclass(budget.read_budget.limits, clock_ns=lambda: 0),
            "wire_budget": WireSubclass(budget.wire_budget.limits),
            "limits": LimitsSubclass(**dict.fromkeys(("max_" + axis for axis in self.axes), 10)),
        }
        supplied = {
            "read_budget": budget.read_budget,
            "wire_budget": budget.wire_budget,
            "limits": budget.limits,
        }
        for name in supplied:
            for invalid in (None, True, SimpleNamespace(), subclasses[name]):
                with self.subTest(argument=name, invalid=type(invalid).__name__):
                    with self.assertRaises(PinnedReadRefused) as raised:
                        api._PageBudget(**(supplied | {name: invalid}))
                    self.assertEqual(raised.exception.reason_code, "invalid_contract")

    def test_page_budget_revalidates_supplied_limits(self) -> None:
        api = self.api()
        for axis in self.axes:
            with self.subTest(axis=axis):
                budget = self.budget()
                object.__setattr__(budget.limits, "max_" + axis, True)
                with self.assertRaises(PinnedReadRefused) as raised:
                    api._PageBudget(
                        read_budget=budget.read_budget,
                        wire_budget=budget.wire_budget,
                        limits=budget.limits,
                    )
                self.assertEqual(raised.exception.reason_code, "invalid_contract")

    def test_page_reservation_requires_exact_charge_type_before_clock(self) -> None:
        api = self.api()
        budget = self.budget()
        budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 1)))

        class ChargeSubclass(api._PageCharge):
            pass

        for invalid in (None, True, 1, {}, SimpleNamespace(), ChargeSubclass()):
            with self.subTest(invalid=type(invalid).__name__):
                with (
                    patch.object(budget.read_budget, "clock_ns") as clock,
                    self.assertRaises(PinnedReadRefused) as raised,
                ):
                    budget.reserve(invalid)
                self.assertEqual(raised.exception.reason_code, "invalid_contract")
                clock.assert_not_called()
                self.assertEqual(self.state(budget), (1,) * 12)

    def test_page_reservation_revalidates_charge_before_clock_without_mutation(self) -> None:
        api = self.api()
        budget = self.budget()
        budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 1)))
        for axis in self.axes:
            for invalid in (True, False, -1, 0.0, "0", None, self.IntegerSubclass(0)):
                with self.subTest(axis=axis, invalid=invalid):
                    charge = api._PageCharge(**dict.fromkeys(self.axes, 1))
                    object.__setattr__(charge, axis, invalid)
                    with (
                        patch.object(budget.read_budget, "clock_ns") as clock,
                        self.assertRaises(PinnedReadRefused) as raised,
                    ):
                        budget.reserve(charge)
                    self.assertEqual(raised.exception.reason_code, "invalid_contract")
                    clock.assert_not_called()
                    self.assertEqual(self.state(budget), (1,) * 12)

    def test_page_default_and_explicit_zero_charges_do_not_consume_capacity(self) -> None:
        api = self.api()
        for charge in (api._PageCharge(), api._PageCharge(**dict.fromkeys(self.axes, 0))):
            with self.subTest(charge=charge):
                budget = self.budget()
                self.assertIsNone(budget.reserve(charge))
                self.assertEqual(self.state(budget), (0,) * 12)

    def test_page_exact_capacity_and_zero_after_exhaustion_are_valid(self) -> None:
        api = self.api()
        budget = self.budget(**dict.fromkeys(("max_" + axis for axis in self.axes), 1))
        self.assertIsNone(budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 1))))
        self.assertIsNone(budget.reserve(api._PageCharge()))
        self.assertEqual(self.state(budget), (1,) * 12)

    def test_page_axes_accumulate_independently_across_calls(self) -> None:
        api = self.api()
        for index, axis in enumerate(self.axes):
            with self.subTest(axis=axis):
                budget = self.budget()
                budget.reserve(api._PageCharge(**{axis: 2}))
                budget.reserve(api._PageCharge(**{axis: 3}))
                self.assertEqual(self.state(budget), (0,) * index + (5,) + (0,) * (11 - index))

    def test_page_any_axis_over_remaining_refuses_all_changes_atomically(self) -> None:
        api = self.api()
        for axis in self.axes:
            with self.subTest(axis=axis):
                budget = self.budget()
                budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 2)))
                increments = dict.fromkeys(self.axes, 1) | {axis: 9}
                with self.assertRaises(PinnedReadRefused) as raised:
                    budget.reserve(api._PageCharge(**increments))
                self.assertEqual(raised.exception.reason_code, "resource_limit")
                self.assertEqual(self.state(budget), (2,) * 12)

    def test_page_exact_remaining_capacity_is_available_after_refusal(self) -> None:
        api = self.api()
        budget = self.budget()
        budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 3)))
        with self.assertRaises(PinnedReadRefused) as raised:
            budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 8)))
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertIsNone(budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 7))))
        self.assertEqual(self.state(budget), (10,) * 12)

    def test_page_shared_deadline_refuses_before_any_counter_mutation(self) -> None:
        api = self.api()
        for now in (10, 11):
            for increment in (0, 1, 10):
                with self.subTest(now=now, increment=increment):
                    budget = self.budget()
                    budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 1)))
                    charge = api._PageCharge(**dict.fromkeys(self.axes, increment))
                    with (
                        patch.object(budget.read_budget, "clock_ns", return_value=now) as clock,
                        self.assertRaises(PinnedReadRefused) as raised,
                    ):
                        budget.reserve(charge)
                    self.assertEqual(raised.exception.reason_code, "deadline_exceeded")
                    clock.assert_called()
                    self.assertEqual(self.state(budget), (1,) * 12)

    def test_page_live_shared_clock_allows_reservation_before_deadline(self) -> None:
        api = self.api()
        budget = self.budget()
        with patch.object(budget.read_budget, "clock_ns", return_value=9) as clock:
            self.assertIsNone(budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 1))))
        clock.assert_called()
        self.assertEqual(self.state(budget), (1,) * 12)

    def test_page_shared_clock_contract_refusal_preserves_counters(self) -> None:
        api = self.api()
        for invalid in (True, -1, 0.0, "0", None):
            with self.subTest(clock=invalid):
                budget = self.budget()
                budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 1)))
                with (
                    patch.object(budget.read_budget, "clock_ns", return_value=invalid),
                    self.assertRaises(PinnedReadRefused) as raised,
                ):
                    budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 1)))
                self.assertEqual(raised.exception.reason_code, "invalid_contract")
                self.assertEqual(self.state(budget), (1,) * 12)

    def test_page_accounting_preserves_preexisting_io_and_wire_counters(self) -> None:
        api = self.api()
        supplied = self.budget()
        read = supplied.read_budget
        wire = supplied.wire_budget
        read.before_open()
        read.record_read(3)
        wire._reserve(
            wire_bytes=2, work_units=3, field_count=4, items=5, string_bytes=6, copied_bytes=7
        )
        wire_axes = (
            "total_wire_bytes",
            "work_units",
            "total_fields",
            "total_items",
            "total_string_bytes",
            "copied_payload_bytes",
        )
        budget = api._PageBudget(read_budget=read, wire_budget=wire, limits=supplied.limits)
        self.assertEqual((read.source_opens, read.total_read_bytes), (1, 3))
        self.assertEqual(tuple(getattr(wire, axis) for axis in wire_axes), (2, 3, 4, 5, 6, 7))
        budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 1)))
        with self.assertRaises(PinnedReadRefused) as raised:
            budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 10)))
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual((read.source_opens, read.total_read_bytes), (1, 3))
        self.assertEqual(tuple(getattr(wire, axis) for axis in wire_axes), (2, 3, 4, 5, 6, 7))
        self.assertEqual(self.state(budget), (1,) * 12)

    def test_page_reserved_capacity_is_retained_after_caller_failure(self) -> None:
        api = self.api()
        budget = self.budget()
        with self.assertRaises(RuntimeError):
            budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 4)))
            raise RuntimeError("neutral downstream failure")
        self.assertEqual(self.state(budget), (4,) * 12)
        with self.assertRaises(PinnedReadRefused) as raised:
            budget.reserve(api._PageCharge(**dict.fromkeys(self.axes, 7)))
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(self.state(budget), (4,) * 12)


class PageHeaderAdmissionTests(unittest.TestCase):
    body_axes = (
        "compressed_bytes",
        "uncompressed_bytes",
        "values",
        "dictionary_entries",
        "dictionary_payload_bytes",
    )

    def api(self) -> SimpleNamespace:
        names = (
            "_admit_page_header",
            "_PageHeaderAdmission",
            "_PageDataV1Header",
            "_PageIndexHeader",
            "_PageDictionaryHeader",
            "_PageDataV2Header",
            "_PageStatistics",
        )
        found = {}
        for name in names:
            value = getattr(bounded_parquet, name, None)
            self.assertTrue(callable(value), f"missing callable {name}")
            found[name] = value
        return SimpleNamespace(**found)

    def detail_fields(self, kind: int) -> tuple[tuple[int, str, bytes], ...]:
        declarations = {
            0: ((1, 3), (2, 8), (3, 3), (4, 4)),
            1: (),
            2: ((1, 2), (2, 2)),
            3: ((1, 3), (2, 1), (3, 2), (4, 9), (5, 2), (6, 3)),
        }
        return tuple((field, "i32", _grammar_integer(value)) for field, value in declarations[kind])

    def root_fields(
        self,
        kind: int,
        *,
        detail=None,
        compressed: int = 7,
        uncompressed: int = 11,
        crc: int | None = None,
    ) -> tuple[tuple[int, str, bytes], ...]:
        fields = (
            (1, "i32", _grammar_integer(kind)),
            (2, "i32", _grammar_integer(uncompressed)),
            (3, "i32", _grammar_integer(compressed)),
        )
        if crc is not None:
            fields += ((4, "i32", _grammar_integer(crc)),)
        body = self.detail_fields(kind) if detail is None else detail
        return fields + (((5, 6, 7, 8)[kind], "struct", _grammar_structure(body)),)

    def wire(self, kind: int, **changes) -> bytes:
        return _grammar_structure(self.root_fields(kind, **changes))

    def context(
        self, payload: bytes, *, start: int = 0, stop: int | None = None, page=None, wire=None
    ) -> tuple[bounded_parquet._CompactWireCursor, bounded_parquet._PageBudget]:
        page_limits = dict.fromkeys(("max_" + axis for axis in PageBudgetTests.axes), 1 << 33)
        page_limits.update(page or {})
        budget = PageBudgetTests().budget(**page_limits)
        wire_limits = {
            "max_input_bytes": 1024,
            "max_object_bytes": 1024,
            "max_header_bytes": 512,
            "max_total_fields": 64,
            "max_total_wire_bytes": 2048,
            "max_work_units": 16384,
        }
        wire_limits.update(wire or {})
        budget.wire_budget = CompactWireCursorTests().budget(**wire_limits)
        cursor = bounded_parquet._CompactWireCursor(
            payload,
            start,
            len(payload) if stop is None else stop,
            budget.wire_budget,
            kind="page_header",
        )
        return cursor, budget

    def parse(
        self, payload: bytes, *, remaining: int | None = None, **changes
    ) -> tuple[
        bounded_parquet._PageHeaderAdmission,
        bounded_parquet._CompactWireCursor,
        bounded_parquet._PageBudget,
    ]:
        cursor, budget = self.context(payload, **changes)
        api = self.api()
        result = api._admit_page_header(
            cursor,
            chunk_remaining_bytes=len(payload) + 7 if remaining is None else remaining,
            budget=budget,
        )
        self.assertIsInstance(result, api._PageHeaderAdmission)
        return result, cursor, budget

    def refused(
        self, payload: bytes, reason: str, *, remaining: int | None = None
    ) -> tuple[bounded_parquet._CompactWireCursor, bounded_parquet._PageBudget]:
        cursor, budget = self.context(payload)
        with self.assertRaises(
            (PinnedReadRefused, bounded_parquet._CompactWireMalformed)
        ) as raised:
            self.api()._admit_page_header(
                cursor,
                chunk_remaining_bytes=len(payload) + 32 if remaining is None else remaining,
                budget=budget,
            )
        self.assertEqual(raised.exception.reason_code, reason)
        self.assertEqual(budget.pages, 1)
        self.assertEqual(tuple(getattr(budget, axis) for axis in self.body_axes), (0,) * 5)
        return cursor, budget

    def test_page_header_private_interface_is_explicit(self) -> None:
        self.api()

    def test_all_variants_preserve_fields_and_atomic_body_charge_without_decode(self) -> None:
        api = self.api()
        cases = (
            (
                0,
                api._PageDataV1Header,
                {
                    "num_values": 3,
                    "encoding": 8,
                    "definition_level_encoding": 3,
                    "repetition_level_encoding": 4,
                    "statistics": None,
                },
                (7, 11, 3, 0, 0),
            ),
            (1, api._PageIndexHeader, {}, (7, 11, 0, 0, 0)),
            (
                2,
                api._PageDictionaryHeader,
                {"num_values": 2, "encoding": 2, "is_sorted": None},
                (7, 11, 0, 2, 11),
            ),
            (
                3,
                api._PageDataV2Header,
                {
                    "num_values": 3,
                    "num_nulls": 1,
                    "num_rows": 2,
                    "encoding": 9,
                    "definition_levels_byte_length": 2,
                    "repetition_levels_byte_length": 3,
                    "is_compressed": None,
                    "effective_is_compressed": True,
                    "statistics": None,
                },
                (7, 11, 3, 0, 0),
            ),
        )
        for kind, record, expected, charge in cases:
            with self.subTest(kind=kind):
                payload = self.wire(kind)
                cursor, budget = self.context(payload)
                reserve = budget.reserve
                body_charges = []

                def observe(value, *, budget=budget, body_charges=body_charges, reserve=reserve):
                    if any(getattr(value, axis) for axis in self.body_axes):
                        body_charges.append(value)
                    counters = (
                        budget.read_budget.source_opens,
                        budget.read_budget.total_read_bytes,
                        budget.wire_budget.total_wire_bytes,
                        budget.wire_budget.work_units,
                        budget.wire_budget.total_fields,
                        budget.wire_budget.total_items,
                        budget.wire_budget.total_string_bytes,
                        budget.wire_budget.copied_payload_bytes,
                    )
                    result = reserve(value)
                    self.assertEqual(
                        (
                            budget.read_budget.source_opens,
                            budget.read_budget.total_read_bytes,
                            budget.wire_budget.total_wire_bytes,
                            budget.wire_budget.work_units,
                            budget.wire_budget.total_fields,
                            budget.wire_budget.total_items,
                            budget.wire_budget.total_string_bytes,
                            budget.wire_budget.copied_payload_bytes,
                        ),
                        counters,
                    )
                    return result

                with patch.object(budget, "reserve", observe):
                    result = api._admit_page_header(
                        cursor, chunk_remaining_bytes=len(payload) + 7, budget=budget
                    )
                self.assertIsInstance(result, api._PageHeaderAdmission)
                self.assertIs(type(result), api._PageHeaderAdmission)
                self.assertIs(type(result.detail), record)
                self.assertEqual(
                    (result.header_start, result.header_end, result.body_start, result.body_end),
                    (0, len(payload), len(payload), len(payload) + 7),
                )
                self.assertEqual(
                    (result.page_type, result.compressed_page_size, result.uncompressed_page_size),
                    (kind, 7, 11),
                )
                self.assertIsNone(result.crc)
                self.assertIsNone(result.payload)
                self.assertIs(result.decode_eligible, False)
                self.assertEqual(cursor.position, len(payload))
                self.assertEqual(len(body_charges), 1)
                self.assertEqual(
                    tuple(getattr(body_charges[0], axis) for axis in self.body_axes), charge
                )
                self.assertEqual(tuple(getattr(budget, axis) for axis in self.body_axes), charge)
                self.assertEqual(budget.pages, 1)
                self.assertEqual(
                    (
                        budget.levels,
                        budget.codec_scratch_bytes,
                        budget.output_rows,
                        budget.output_bytes,
                        budget.allocation_bytes,
                    ),
                    (0,) * 5,
                )
                self.assertEqual(
                    (budget.read_budget.source_opens, budget.read_budget.total_read_bytes), (0, 0)
                )
                self.assertEqual(budget.wire_budget.copied_payload_bytes, 0)
                self.assertEqual(budget.wire_budget.total_wire_bytes, len(payload))
                for name, value in expected.items():
                    self.assertEqual(getattr(result.detail, name), value)
                with self.assertRaises(FrozenInstanceError):
                    result.page_type = 99
                if kind != 1:
                    with self.assertRaises(FrozenInstanceError):
                        result.detail.num_values = 99
                if kind == 0:
                    self.assertFalse(hasattr(result.detail, "num_rows"))

    def test_signed_crc_and_optional_flags_preserve_presence(self) -> None:
        for crc in (None, -(1 << 31), -1, 0, (1 << 31) - 1):
            result, _, _ = self.parse(self.wire(1, crc=crc))
            self.assertEqual(result.crc, crc)
        for flag in (None, False, True):
            for kind, field, name in ((2, 3, "is_sorted"), (3, 7, "is_compressed")):
                with self.subTest(kind=kind, flag=flag):
                    detail = self.detail_fields(kind)
                    if flag is not None:
                        detail += ((field, "bool", b"\x01" if flag else b"\x02"),)
                    result, _, _ = self.parse(self.wire(kind, detail=detail, uncompressed=7))
                    self.assertIs(getattr(result.detail, name), flag)
                    if kind == 3:
                        assert isinstance(result.detail, bounded_parquet._PageDataV2Header)
                        self.assertIs(result.detail.effective_is_compressed, flag is not False)

    def test_statistics_are_frozen_zero_copy_spans_with_absent_counts_distinct_from_zero(
        self,
    ) -> None:
        statistics = (
            (1, "binary", b"\x01\xaa"),
            (2, "binary", b"\x01\xbb"),
            (3, "i64", _grammar_integer(0)),
            (4, "i64", _grammar_integer(2)),
            (5, "binary", b"\x01\xcc"),
            (6, "binary", b"\x01\xdd"),
            (7, "bool", b"\x02"),
            (8, "bool", b"\x01"),
        )
        for kind, field in ((0, 5), (3, 8)):
            with self.subTest(kind=kind):
                detail = self.detail_fields(kind) + (
                    (field, "struct", _grammar_structure(statistics)),
                )
                payload = self.wire(kind, detail=detail)
                with patch.object(
                    bounded_parquet._CompactWireCursor,
                    "read_binary",
                    side_effect=AssertionError("statistics must remain spans"),
                ):
                    result, _, budget = self.parse(payload)
                self.assertIs(result.payload, payload)
                assert isinstance(
                    result.detail,
                    (
                        bounded_parquet._PageDataV1Header,
                        bounded_parquet._PageDataV2Header,
                    ),
                )
                stats = result.detail.statistics
                assert stats is not None
                self.assertIs(type(stats), self.api()._PageStatistics)
                for name, marker in (
                    ("max", b"\xaa"),
                    ("min", b"\xbb"),
                    ("max_value", b"\xcc"),
                    ("min_value", b"\xdd"),
                ):
                    span = getattr(stats, name)
                    self.assertEqual(span, (payload.index(marker), payload.index(marker) + 1))
                self.assertEqual((stats.null_count, stats.distinct_count), (0, 2))
                self.assertIs(stats.is_max_value_exact, False)
                self.assertIs(stats.is_min_value_exact, True)
                self.assertEqual(budget.wire_budget.total_string_bytes, 4)
                self.assertEqual(budget.wire_budget.copied_payload_bytes, 0)
                frozen_field = "null_count"
                with self.assertRaises(FrozenInstanceError):
                    setattr(stats, frozen_field, 1)
                empty = self.detail_fields(kind) + ((field, "struct", b"\x00"),)
                result, _, _ = self.parse(self.wire(kind, detail=empty))
                assert isinstance(
                    result.detail,
                    (
                        bounded_parquet._PageDataV1Header,
                        bounded_parquet._PageDataV2Header,
                    ),
                )
                assert result.detail.statistics is not None
                for name in (
                    "max",
                    "min",
                    "max_value",
                    "min_value",
                    "null_count",
                    "distinct_count",
                    "is_max_value_exact",
                    "is_min_value_exact",
                ):
                    self.assertIsNone(getattr(result.detail.statistics, name))
                self.assertIsNone(result.payload)
                zero_span = self.detail_fields(kind) + (
                    (field, "struct", _grammar_structure(((1, "binary", b"\x00"),))),
                )
                payload = self.wire(kind, detail=zero_span)
                result, _, budget = self.parse(payload)
                self.assertIs(result.payload, payload)
                assert isinstance(
                    result.detail,
                    (
                        bounded_parquet._PageDataV1Header,
                        bounded_parquet._PageDataV2Header,
                    ),
                )
                assert result.detail.statistics is not None
                assert result.detail.statistics.max is not None
                self.assertEqual(result.detail.statistics.max[0], result.detail.statistics.max[1])
                self.assertEqual(budget.wire_budget.total_string_bytes, 0)

    def test_required_root_and_detail_fields_cannot_be_omitted(self) -> None:
        for kind in range(4):
            root = self.root_fields(kind)
            for index in range(3):
                with self.subTest(kind=kind, root=index):
                    self.refused(
                        _grammar_structure(root[:index] + root[index + 1 :]), "schema_mismatch"
                    )
            detail = self.detail_fields(kind)
            for index in range(len(detail)):
                with self.subTest(kind=kind, detail=index):
                    self.refused(
                        self.wire(kind, detail=detail[:index] + detail[index + 1 :]),
                        "schema_mismatch",
                    )

    def test_reordered_root_and_nested_fields_are_admitted(self) -> None:
        for kind in range(4):
            with self.subTest(kind=kind):
                root = self.root_fields(
                    kind, detail=tuple(reversed(self.detail_fields(kind))), crc=-1
                )
                payload = _grammar_structure(tuple(reversed(root)))
                result, cursor, _ = self.parse(payload)
                self.assertEqual((result.page_type, result.crc), (kind, -1))
                self.assertEqual(cursor.position, len(payload))

    def test_union_and_field_schema_contradictions_refuse(self) -> None:
        root = self.root_fields(0)
        bad_roots = (
            root[:-1],
            root + ((6, "struct", b"\x00"),),
            root[:-1] + ((6, "struct", b"\x00"),),
            root + (root[0],),
            root + ((9, "i32", b"\x00"),),
            ((1, "i64", b"\x00"),) + root[1:],
        )
        for fields in bad_roots:
            self.refused(_grammar_structure(fields), "schema_mismatch")
        for kind in range(4):
            fields = self.detail_fields(kind)
            self.refused(self.wire(kind, detail=fields + ((9, "i32", b"\x00"),)), "schema_mismatch")
            if fields:
                self.refused(self.wire(kind, detail=fields + (fields[0],)), "schema_mismatch")
                self.refused(
                    self.wire(kind, detail=((1, "i64", b"\x00"),) + fields[1:]), "schema_mismatch"
                )
        for statistics in (
            ((9, "i64", b"\x00"),),
            ((3, "i32", b"\x00"),),
            ((3, "i64", b"\x00"), (3, "i64", b"\x00")),
        ):
            fields = self.detail_fields(0) + ((5, "struct", _grammar_structure(statistics)),)
            self.refused(self.wire(0, detail=fields), "schema_mismatch")

    def test_every_header_prefix_truncates_without_body_charge(self) -> None:
        for kind in range(4):
            payload = self.wire(kind)
            for stop in range(len(payload)):
                with self.subTest(kind=kind, stop=stop):
                    self.refused(payload[:stop], "truncated_wire")
        self.refused(b"\x15\x80", "truncated_wire")

    def test_known_encodings_are_preserved_in_every_encoding_slot(self) -> None:
        for kind, field, name in (
            (0, 2, "encoding"),
            (0, 3, "definition_level_encoding"),
            (0, 4, "repetition_level_encoding"),
            (2, 2, "encoding"),
            (3, 4, "encoding"),
        ):
            for encoding in (0, 2, 3, 4, 5, 6, 7, 8, 9):
                with self.subTest(kind=kind, field=field, encoding=encoding):
                    detail = tuple(
                        (number, wire, _grammar_integer(encoding) if number == field else body)
                        for number, wire, body in self.detail_fields(kind)
                    )
                    result, _, _ = self.parse(self.wire(kind, detail=detail))
                    self.assertEqual(getattr(result.detail, name), encoding)
                    self.assertIs(result.decode_eligible, False)

    def test_unknown_page_and_encoding_enums_are_unsupported(self) -> None:
        for kind in (-1, 4, (1 << 31) - 1):
            root = self.root_fields(1)
            self.refused(
                _grammar_structure(((1, "i32", _grammar_integer(kind)),) + root[1:]),
                "unsupported_mode",
            )
        for kind, field in ((0, 2), (0, 3), (0, 4), (2, 2), (3, 4)):
            for encoding in (-1, 1, 10):
                detail = tuple(
                    (number, wire, _grammar_integer(encoding) if number == field else body)
                    for number, wire, body in self.detail_fields(kind)
                )
                self.refused(self.wire(kind, detail=detail), "unsupported_mode")

    def test_negative_sizes_counts_and_level_lengths_are_schema_refusals(self) -> None:
        for name in ("compressed", "uncompressed"):
            self.refused(self.wire(1, **{name: -1}), "schema_mismatch")
        for kind, fields in ((0, (1,)), (2, (1,)), (3, (1, 2, 3, 5, 6))):
            for field in fields:
                detail = tuple(
                    (number, wire, _grammar_integer(-1) if number == field else body)
                    for number, wire, body in self.detail_fields(kind)
                )
                self.refused(self.wire(kind, detail=detail), "schema_mismatch")
        for field in (3, 4):
            stats = _grammar_structure(((field, "i64", _grammar_integer(-1)),))
            self.refused(
                self.wire(0, detail=self.detail_fields(0) + ((5, "struct", stats),)),
                "schema_mismatch",
            )

    def test_zero_and_maximum_i32_declarations_do_not_materialize_counts(self) -> None:
        for value in (0, (1 << 31) - 1):
            for kind in range(4):
                detail = self.detail_fields(kind)
                count_fields = {0: (1,), 1: (), 2: (1,), 3: (1, 2, 3)}[kind]
                detail = tuple(
                    (
                        number,
                        wire,
                        _grammar_integer(value)
                        if number in count_fields
                        else b"\x00"
                        if kind == 3 and number in (5, 6)
                        else body,
                    )
                    for number, wire, body in detail
                )
                payload = self.wire(kind, detail=detail, compressed=value, uncompressed=value)
                result, _, budget = self.parse(payload, remaining=len(payload) + value)
                self.assertEqual(
                    (budget.compressed_bytes, budget.uncompressed_bytes), (value, value)
                )
                self.assertEqual(budget.values, value if kind in (0, 3) else 0)
                self.assertEqual(budget.dictionary_entries, value if kind == 2 else 0)
                self.assertEqual(
                    (budget.output_rows, budget.output_bytes, budget.allocation_bytes), (0, 0, 0)
                )
                self.assertIs(result.decode_eligible, False)

    def test_v2_count_level_subtraction_and_uncompressed_size_relations(self) -> None:
        cases = (
            (7, 11, {2: 4}),
            (7, 11, {3: 4}),
            (7, 11, {5: 8, 6: 0}),
            (7, 11, {5: 0, 6: 8}),
            (11, 7, {5: 8, 6: 0}),
            (11, 7, {5: 0, 6: 8}),
            (7, 11, {5: 4, 6: 4}),
            (11, 7, {5: 4, 6: 4}),
        )
        for compressed, uncompressed, changed in cases:
            detail = tuple(
                (number, wire, _grammar_integer(changed[number]) if number in changed else body)
                for number, wire, body in self.detail_fields(3)
            )
            self.refused(
                self.wire(3, detail=detail, compressed=compressed, uncompressed=uncompressed),
                "schema_mismatch",
            )
        detail = self.detail_fields(3) + ((7, "bool", b"\x02"),)
        self.refused(self.wire(3, detail=detail), "schema_mismatch")
        result, _, _ = self.parse(
            self.wire(3, detail=detail, compressed=5, uncompressed=5), remaining=128
        )
        assert isinstance(result.detail, bounded_parquet._PageDataV2Header)
        self.assertIs(result.detail.effective_is_compressed, False)

    def test_nonzero_start_header_window_and_chunk_extent_are_distinct(self) -> None:
        header = self.wire(0)
        payload = b"\xff\xff" + header + b"\xff\xfe"
        result, cursor, _ = self.parse(payload, start=2, remaining=len(header) + 7)
        self.assertEqual(
            (result.header_start, result.header_end, result.body_end),
            (2, 2 + len(header), 2 + len(header) + 7),
        )
        self.assertEqual(cursor.position, 2 + len(header))
        self.assertGreater(result.body_end, cursor.stop)
        result, _, _ = self.parse(header, remaining=len(header) + 7)
        self.assertEqual(result.body_end, len(header) + 7)
        self.refused(header, "truncated_wire", remaining=len(header) + 6)

    def test_caller_types_extent_freshness_and_shared_budget_precede_clock_or_consumption(
        self,
    ) -> None:
        class IntegerSubclass(int):
            pass

        class CursorSubclass(bounded_parquet._CompactWireCursor):
            pass

        class BudgetSubclass(bounded_parquet._PageBudget):
            pass

        payload = self.wire(1)
        for invalid in (True, -1, 0.0, IntegerSubclass(100), None, len(payload) - 1):
            cursor, budget = self.context(payload)
            before = PageBudgetTests().state(budget)
            with (
                patch.object(
                    budget.read_budget,
                    "clock_ns",
                    side_effect=AssertionError("invalid caller reached clock"),
                ),
                self.assertRaises(PinnedReadRefused) as raised,
            ):
                self.api()._admit_page_header(cursor, chunk_remaining_bytes=invalid, budget=budget)
            self.assertEqual(raised.exception.reason_code, "invalid_contract")
            self.assertEqual(cursor.position, 0)
            self.assertEqual(PageBudgetTests().state(budget), before)
        for invalid in (
            "cursor",
            "cursor_subclass",
            "budget",
            "budget_subclass",
            "kind",
            "advanced",
            "struct",
            "bool",
            "shared",
        ):
            cursor, budget = self.context(payload)
            supplied_cursor, supplied_budget = cursor, budget
            if invalid == "cursor":
                supplied_cursor = SimpleNamespace()
            if invalid == "cursor_subclass":
                supplied_cursor = CursorSubclass(
                    payload, 0, len(payload), budget.wire_budget, kind="page_header"
                )
            if invalid == "budget":
                supplied_budget = SimpleNamespace()
            if invalid == "budget_subclass":
                supplied_budget = BudgetSubclass(
                    read_budget=budget.read_budget,
                    wire_budget=budget.wire_budget,
                    limits=budget.limits,
                )
            if invalid == "kind":
                cursor.kind = "footer"
            if invalid == "advanced":
                cursor.position = 1
            if invalid == "struct":
                cursor.begin_struct()
            if invalid == "bool":
                cursor._pending_bool = False
            if invalid == "shared":
                cursor.budget = CompactWireCursorTests().budget()
            position = cursor.position
            counters = (
                PageBudgetTests().state(budget),
                budget.wire_budget.total_wire_bytes,
                budget.wire_budget.work_units,
                budget.wire_budget.total_fields,
            )
            with (
                patch.object(
                    budget.read_budget,
                    "clock_ns",
                    side_effect=AssertionError("invalid caller reached clock"),
                ),
                self.assertRaises(PinnedReadRefused) as raised,
            ):
                self.api()._admit_page_header(
                    supplied_cursor, chunk_remaining_bytes=100, budget=supplied_budget
                )
            self.assertEqual(raised.exception.reason_code, "invalid_contract")
            self.assertEqual(cursor.position, position)
            self.assertEqual(
                (
                    PageBudgetTests().state(budget),
                    budget.wire_budget.total_wire_bytes,
                    budget.wire_budget.work_units,
                    budget.wire_budget.total_fields,
                ),
                counters,
            )

    def test_attempt_charge_survives_malformed_header_and_repeated_cursor_is_invalid(self) -> None:
        payload = _grammar_structure(self.root_fields(1) + ((9, "i32", b"\x00"),))
        cursor, budget = self.refused(payload, "schema_mismatch")
        self.assertGreater(cursor.position, 0)
        before = (
            cursor.position,
            PageBudgetTests().state(budget),
            budget.wire_budget.total_wire_bytes,
        )
        with (
            patch.object(
                budget.read_budget,
                "clock_ns",
                side_effect=AssertionError("advanced cursor retried"),
            ),
            self.assertRaises(PinnedReadRefused) as raised,
        ):
            self.api()._admit_page_header(cursor, chunk_remaining_bytes=100, budget=budget)
        self.assertEqual(raised.exception.reason_code, "invalid_contract")
        self.assertEqual(
            (cursor.position, PageBudgetTests().state(budget), budget.wire_budget.total_wire_bytes),
            before,
        )
        next_payload = self.wire(1)
        fresh = bounded_parquet._CompactWireCursor(
            next_payload, 0, len(next_payload), budget.wire_budget, kind="page_header"
        )
        api = self.api()
        result = api._admit_page_header(fresh, chunk_remaining_bytes=100, budget=budget)
        self.assertIsInstance(result, api._PageHeaderAdmission)
        self.assertEqual((budget.pages, budget.compressed_bytes), (2, 7))

    def test_final_body_charge_refuses_atomically_on_each_late_axis(self) -> None:
        for kind, axis, declared in (
            (0, "compressed_bytes", 7),
            (0, "uncompressed_bytes", 11),
            (0, "values", 3),
            (2, "dictionary_entries", 2),
            (2, "dictionary_payload_bytes", 11),
        ):
            payload = self.wire(kind)
            cursor, budget = self.context(payload, page={"max_" + axis: declared})
            budget.reserve(bounded_parquet._PageCharge(**dict.fromkeys(self.body_axes, 1)))
            with self.assertRaises(PinnedReadRefused) as raised:
                self.api()._admit_page_header(cursor, chunk_remaining_bytes=100, budget=budget)
            self.assertEqual(raised.exception.reason_code, "resource_limit")
            self.assertEqual(tuple(getattr(budget, name) for name in self.body_axes), (1,) * 5)
            self.assertEqual(cursor.position, len(payload))
            self.assertEqual(budget.pages, 1)
            self.assertGreater(budget.work_units, 0)

    def test_exact_cumulative_body_and_page_limits_do_not_reset_between_headers(self) -> None:
        payload = self.wire(0)
        cursor, budget = self.context(
            payload,
            page={
                "max_pages": 2,
                "max_compressed_bytes": 14,
                "max_uncompressed_bytes": 22,
                "max_values": 6,
            },
        )
        api = self.api()
        for _ in range(2):
            cursor = bounded_parquet._CompactWireCursor(
                payload, 0, len(payload), budget.wire_budget, kind="page_header"
            )
            result = api._admit_page_header(cursor, chunk_remaining_bytes=100, budget=budget)
            self.assertIsInstance(result, api._PageHeaderAdmission)
        self.assertEqual(
            (budget.pages, budget.compressed_bytes, budget.uncompressed_bytes, budget.values),
            (2, 14, 22, 6),
        )
        cursor = bounded_parquet._CompactWireCursor(
            payload, 0, len(payload), budget.wire_budget, kind="page_header"
        )
        before = (PageBudgetTests().state(budget), budget.wire_budget.total_wire_bytes)
        with self.assertRaises(PinnedReadRefused) as raised:
            self.api()._admit_page_header(cursor, chunk_remaining_bytes=100, budget=budget)
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(cursor.position, 0)
        self.assertEqual(
            (PageBudgetTests().state(budget), budget.wire_budget.total_wire_bytes), before
        )

    def test_entry_deadline_refuses_before_attempt_or_wire_mutation(self) -> None:
        cursor, budget = self.context(self.wire(0))
        with (
            patch.object(budget.read_budget, "clock_ns", return_value=10),
            self.assertRaises(PinnedReadRefused) as raised,
        ):
            self.api()._admit_page_header(cursor, chunk_remaining_bytes=100, budget=budget)
        self.assertEqual(raised.exception.reason_code, "deadline_exceeded")
        self.assertEqual(PageBudgetTests().state(budget), (0,) * len(PageBudgetTests.axes))
        self.assertEqual(
            (cursor.position, budget.wire_budget.total_wire_bytes, budget.wire_budget.work_units),
            (0, 0, 0),
        )

    def test_deadline_after_field_dispatch_prevents_scalar_consumption(self) -> None:
        cursor, budget = self.context(self.wire(0))
        now = [0]
        read_field = cursor.read_field

        def expire_after_field(allowed):
            value = read_field(allowed)
            now[0] = 10
            return value

        with (
            patch.object(budget.read_budget, "clock_ns", side_effect=lambda: now[0]),
            patch.object(cursor, "read_field", expire_after_field),
            self.assertRaises(PinnedReadRefused) as raised,
        ):
            self.api()._admit_page_header(cursor, chunk_remaining_bytes=100, budget=budget)
        self.assertEqual(raised.exception.reason_code, "deadline_exceeded")
        self.assertEqual(
            (cursor.position, budget.wire_budget.total_wire_bytes, budget.pages), (1, 1, 1)
        )
        self.assertEqual(tuple(getattr(budget, axis) for axis in self.body_axes), (0,) * 5)

    def test_logical_work_reservation_precedes_struct_fields_scalars_and_retained_records(
        self,
    ) -> None:
        api = self.api()
        stats = _grammar_structure(
            ((1, "binary", b"\x01\xaa"), (3, "i64", b"\x00"), (7, "bool", b"\x02"))
        )
        detail = self.detail_fields(3) + ((8, "struct", stats),)
        cursor, budget = self.context(self.wire(3, detail=detail))
        seen = []
        previous = [0]

        def observe(name, operation):
            def call(*args, **kwargs):
                self.assertGreater(budget.work_units, previous[0], name)
                previous[0] = budget.work_units
                seen.append(name)
                return operation(*args, **kwargs)

            return call

        with ExitStack() as stack:
            for name in (
                "begin_struct",
                "read_field",
                "read_i32",
                "read_i64",
                "read_bool",
                "read_binary_span",
            ):
                stack.enter_context(
                    patch.object(cursor, name, observe(name, getattr(cursor, name)))
                )
            for name in ("_PageStatistics", "_PageDataV2Header", "_PageHeaderAdmission"):
                stack.enter_context(
                    patch.object(bounded_parquet, name, observe(name, getattr(api, name)))
                )
            result = api._admit_page_header(cursor, chunk_remaining_bytes=200, budget=budget)
        self.assertIsInstance(result, api._PageHeaderAdmission)
        self.assertIs(result.decode_eligible, False)
        for name in (
            "begin_struct",
            "read_field",
            "read_i32",
            "read_i64",
            "read_bool",
            "read_binary_span",
            "_PageStatistics",
            "_PageDataV2Header",
            "_PageHeaderAdmission",
        ):
            self.assertIn(name, seen)

    def test_deadline_after_root_completion_prevents_final_body_charge(self) -> None:
        payload = self.wire(0)
        cursor, budget = self.context(payload)
        now = [0]
        finish = cursor.finish

        def expire_after_finish(*, require_exhausted):
            finish(require_exhausted=require_exhausted)
            now[0] = 10

        with (
            patch.object(budget.read_budget, "clock_ns", side_effect=lambda: now[0]),
            patch.object(cursor, "finish", expire_after_finish),
            self.assertRaises(PinnedReadRefused) as raised,
        ):
            self.api()._admit_page_header(cursor, chunk_remaining_bytes=100, budget=budget)
        self.assertEqual(raised.exception.reason_code, "deadline_exceeded")
        self.assertEqual((cursor.position, budget.pages), (len(payload), 1))
        self.assertEqual(tuple(getattr(budget, axis) for axis in self.body_axes), (0,) * 5)

    def test_work_exhaustion_precedes_first_field_byte(self) -> None:
        cursor, budget = self.context(self.wire(1), page={"max_work_units": 1})
        with self.assertRaises(PinnedReadRefused) as raised:
            self.api()._admit_page_header(cursor, chunk_remaining_bytes=100, budget=budget)
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(
            (budget.pages, budget.work_units, cursor.position, budget.wire_budget.total_wire_bytes),
            (1, 1, 0, 0),
        )

    def test_wire_and_header_byte_limits_precede_advance(self) -> None:
        payload = self.wire(0)
        result, cursor, _ = self.parse(
            payload + b"\xff", remaining=len(payload) + 7, wire={"max_header_bytes": len(payload)}
        )
        self.assertEqual((result.header_end, cursor.position), (len(payload), len(payload)))
        for limits, position in (
            ({"max_total_wire_bytes": 1}, 1),
            ({"max_header_bytes": len(payload) - 1}, len(payload) - 1),
        ):
            cursor, budget = self.context(payload, wire=limits)
            with self.assertRaises(PinnedReadRefused) as raised:
                self.api()._admit_page_header(cursor, chunk_remaining_bytes=100, budget=budget)
            self.assertEqual(raised.exception.reason_code, "resource_limit")
            self.assertEqual(
                (cursor.position, budget.wire_budget.total_wire_bytes), (position, position)
            )
            self.assertEqual(tuple(getattr(budget, axis) for axis in self.body_axes), (0,) * 5)

    def test_statistics_byte_quota_refuses_before_span_body_or_copy(self) -> None:
        stats = _grammar_structure(((1, "binary", b"\x03abc"),))
        payload = self.wire(0, detail=self.detail_fields(0) + ((5, "struct", stats),))
        cursor, budget = self.context(payload, wire={"max_string_bytes": 2})
        with (
            patch.object(
                cursor, "read_binary", side_effect=AssertionError("binary copy forbidden")
            ),
            self.assertRaises(PinnedReadRefused) as raised,
        ):
            self.api()._admit_page_header(cursor, chunk_remaining_bytes=100, budget=budget)
        self.assertEqual(raised.exception.reason_code, "resource_limit")
        self.assertEqual(cursor.position, payload.index(b"abc"))
        self.assertEqual(
            (budget.wire_budget.total_string_bytes, budget.wire_budget.copied_payload_bytes), (0, 0)
        )
        self.assertEqual(tuple(getattr(budget, axis) for axis in self.body_axes), (0,) * 5)


class PageLevelScanTests(unittest.TestCase):
    def api(self) -> SimpleNamespace:
        found = {}
        for name in ("_PageLevelInput", "_PageLevelSummary", "_scan_page_levels"):
            value = getattr(bounded_parquet, name, None)
            self.assertTrue(callable(value), f"missing callable {name}")
            found[name] = value
        return SimpleNamespace(**found)

    def budget(self, **changes: int) -> bounded_parquet._PageBudget:
        limits = dict.fromkeys(("max_" + axis for axis in PageBudgetTests.axes), 1 << 40)
        limits.update(changes)
        return PageBudgetTests().budget(**limits)

    def section(self, version: str = "v1", **changes):
        api = self.api()
        n = changes.pop("n", 8)
        max_def = changes.pop("max_def", 7)
        max_rep = changes.pop("max_rep", 0)
        repetition = changes.pop("repetition", b"")
        definition = changes.pop("definition", b"\x03\x88\xc6\xfa")
        prefix = changes.pop("prefix", b"")
        suffix = changes.pop("suffix", b"\xa5")
        outside = changes.pop("outside", b"")
        def_encoding = changes.pop("def_encoding", 3)
        rep_encoding = changes.pop("rep_encoding", 3)
        nulls = changes.pop("nulls", 0 if max_def == 0 else max(n - 1, 0))
        rows = changes.pop("rows", n)
        compression = changes.pop("compression", None)
        self.assertFalse(changes, f"unknown section options {tuple(changes)}")
        if version == "v1":
            if max_rep > 0 and rep_encoding == 3:
                repetition = len(repetition).to_bytes(4, "little") + repetition
            if max_def > 0 and def_encoding == 3:
                definition = len(definition).to_bytes(4, "little") + definition
            header = bounded_parquet._PageDataV1Header(n, 0, def_encoding, rep_encoding, None)
        else:
            self.assertEqual(version, "v2")
            header = bounded_parquet._PageDataV2Header(
                num_values=n,
                num_nulls=nulls,
                num_rows=rows,
                encoding=0,
                definition_levels_byte_length=len(definition),
                repetition_levels_byte_length=len(repetition),
                is_compressed=compression,
                effective_is_compressed=compression is not False,
                statistics=None,
            )
        body = repetition + definition + suffix
        return api._PageLevelInput(
            payload=prefix + body + outside,
            start=len(prefix),
            stop=len(prefix) + len(body),
            header=header,
            max_definition_level=max_def,
            max_repetition_level=max_rep,
        )

    def forged(self, record, **changes):
        from dataclasses import replace

        copy = replace(record)
        for name, value in changes.items():
            object.__setattr__(copy, name, value)
        return copy

    def scan(self, section, budget=None):
        if budget is None:
            budget = self.budget()
        api = self.api()
        result = api._scan_page_levels(section, budget=budget)
        self.assertIsInstance(result, api._PageLevelSummary)
        return result, budget

    def refused(self, section, reason: str, budget=None):
        if budget is None:
            budget = self.budget()
        with self.assertRaises(
            (PinnedReadRefused, bounded_parquet._CompactWireMalformed)
        ) as raised:
            self.api()._scan_page_levels(section, budget=budget)
        self.assertEqual(raised.exception.reason_code, reason)
        return budget

    def counters(self, budget, **expected: int) -> None:
        self.assertEqual(
            PageBudgetTests().state(budget),
            tuple(expected.get(axis, 0) for axis in PageBudgetTests.axes),
        )
        self.assertEqual(budget.wire_budget.total_wire_bytes, 0)
        self.assertEqual(budget.wire_budget.work_units, 0)
        self.assertEqual(budget.wire_budget.total_fields, 0)
        self.assertEqual(budget.wire_budget.total_items, 0)
        self.assertEqual(budget.wire_budget.total_string_bytes, 0)
        self.assertEqual(budget.wire_budget.copied_payload_bytes, 0)
        self.assertEqual(budget.read_budget.source_opens, 0)
        self.assertEqual(budget.read_budget.total_read_bytes, 0)

    def counts(self, result, *, positions=8, non_nulls=1, rows=8, first=0) -> None:
        self.assertEqual(
            (
                result.positions,
                result.non_nulls,
                result.row_starts,
                result.first_repetition_level,
            ),
            (positions, non_nulls, rows, first),
        )

    def positive_cases(self):
        shared = (
            ("hybrid", {}, 36, 32, (8, 1, 8, 0)),
            (
                "asymmetric",
                {
                    "max_rep": 3,
                    "max_def": 3,
                    "repetition": b"\x03\xc0\x00",
                    "definition": b"\x10\x03",
                    "nulls": 0,
                    "rows": 7,
                },
                43,
                35,
                (8, 8, 7, 0),
            ),
            (
                "multibyte",
                {"n": 3, "max_def": 0x1234, "definition": b"\x06\x34\x12", "nulls": 0},
                20,
                16,
                (3, 3, 3, 0),
            ),
            (
                "repeat",
                {"n": 3, "max_def": 2, "definition": b"\x06\x02", "nulls": 0},
                19,
                15,
                (3, 3, 3, 0),
            ),
            (
                "repeat-zero",
                {"n": 3, "max_def": 2, "definition": b"\x06\x00", "nulls": 3},
                19,
                15,
                (3, 0, 3, 0),
            ),
            ("literal-tail", {"n": 5, "max_def": 5, "nulls": 5}, 27, 23, (5, 0, 5, 0)),
            (
                "repeat-to-literal",
                {
                    "max_rep": 3,
                    "max_def": 3,
                    "repetition": b"\x04\x00\x03\x84\x0c",
                    "definition": b"\x06\x03\x0a\x02",
                    "nulls": 5,
                    "rows": 5,
                },
                49,
                41,
                (8, 3, 5, 0),
            ),
            (
                "literal-to-repeat",
                {
                    "n": 11,
                    "max_rep": 3,
                    "max_def": 3,
                    "repetition": b"\x03\xc0\x00\x06\x00",
                    "definition": b"\x16\x03",
                    "nulls": 0,
                    "rows": 10,
                },
                55,
                47,
                (11, 11, 10, 0),
            ),
            (
                "nonminimal-varint",
                {"definition": b"\x83\x00\x88\xc6\xfa"},
                37,
                33,
                (8, 1, 8, 0),
            ),
            (
                "maximum-width",
                {"n": 1, "max_def": (1 << 64) - 1, "definition": b"\x02" + b"\xff" * 8, "nulls": 0},
                20,
                16,
                (1, 1, 1, 0),
            ),
            ("zero-active", {"n": 0, "definition": b"", "suffix": b""}, 7, 3, (0, 0, 0, None)),
            (
                "nonzero-offset",
                {"prefix": b"\xff\xee", "outside": b"\x80\xff"},
                36,
                32,
                (8, 1, 8, 0),
            ),
        )
        cases = []
        for name, options, v1_work, v2_work, counts in shared:
            for version, work in (("v1", v1_work), ("v2", v2_work)):
                cases.append((name, version, options, work, counts))
        for n in (0, 1, (1 << 31) - 1):
            for version in ("v1", "v2"):
                cases.append(
                    (
                        f"implicit-{n}",
                        version,
                        {"n": n, "max_def": 0, "definition": b"", "nulls": 0},
                        3,
                        (n, n, n, 0 if n else None),
                    )
                )
        cases.append(
            (
                "legacy",
                "v1",
                {"def_encoding": 4, "definition": b"\x05\x39\x77"},
                30,
                (8, 1, 8, 0),
            )
        )
        cases.append(
            (
                "legacy-zero",
                "v1",
                {"n": 0, "def_encoding": 4, "definition": b"", "suffix": b""},
                3,
                (0, 0, 0, None),
            )
        )
        for raw in (b"\x00", b"\x40", b"\x80", b"\xc0"):
            cases.append(
                (
                    f"legacy-last-{raw.hex()}",
                    "v1",
                    {"n": 1, "max_def": 3, "def_encoding": 4, "definition": raw, "suffix": b""},
                    7,
                    (1, 1 if raw == b"\xc0" else 0, 1, 0),
                )
            )
        for rep_encoding, def_encoding, repetition, definition, work, first in (
            (4, 3, b"\xc0\x00", b"\x10\x03", 37, 3),
            (3, 4, b"\x03\xc0\x00", b"\xff\xff", 38, 0),
        ):
            cases.append(
                (
                    f"mixed-{rep_encoding}-{def_encoding}",
                    "v1",
                    {
                        "max_rep": 3,
                        "max_def": 3,
                        "rep_encoding": rep_encoding,
                        "def_encoding": def_encoding,
                        "repetition": repetition,
                        "definition": definition,
                    },
                    work,
                    (8, 8, 7, first),
                )
            )
        for flag in (None, False, True):
            cases.append(
                (
                    f"opaque-{flag}",
                    "v2",
                    {"compression": flag, "suffix": b"\xff\x80\x00\xff"},
                    32,
                    (8, 1, 8, 0),
                )
            )
        return cases

    def test_private_input_and_summary_have_only_the_adopted_frozen_fields(self) -> None:
        from dataclasses import FrozenInstanceError, fields

        api = self.api()
        self.assertEqual(
            tuple(field.name for field in fields(api._PageLevelInput)),
            (
                "payload",
                "start",
                "stop",
                "header",
                "max_definition_level",
                "max_repetition_level",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(api._PageLevelSummary)),
            (
                "repetition_span",
                "definition_span",
                "values_span",
                "positions",
                "non_nulls",
                "row_starts",
                "first_repetition_level",
            ),
        )
        section = self.section()
        result, _ = self.scan(section)
        self.assertFalse(hasattr(section, "__dict__"))
        self.assertFalse(hasattr(result, "__dict__"))
        for record, name in ((section, "start"), (result, "positions")):
            with self.assertRaises(FrozenInstanceError):
                setattr(record, name, 1)

    def test_normative_hybrid_vectors_have_exact_spans_counts_and_work(self) -> None:
        for version, end, work in (("v1", 8, 36), ("v2", 4, 32)):
            for prefix in (b"", b"\xff\xee"):
                with self.subTest(version=version, offset=len(prefix)):
                    section = self.section(version, prefix=prefix, outside=b"\x80\xff")
                    result, budget = self.scan(section)
                    start = len(prefix)
                    self.assertEqual(result.repetition_span, (start, start))
                    self.assertEqual(result.definition_span, (start, start + end))
                    self.assertEqual(result.values_span, (start + end, start + end + 1))
                    self.counts(result)
                    self.counters(budget, levels=16, work_units=work)

    def test_asymmetric_repetition_vector_distinguishes_hybrid_bit_order(self) -> None:
        for version, rep_end, def_end, work in (
            ("v1", 7, 13, 43),
            ("v2", 3, 5, 35),
        ):
            with self.subTest(version=version):
                section = self.section(
                    version,
                    max_rep=3,
                    max_def=3,
                    repetition=b"\x03\xc0\x00",
                    definition=b"\x10\x03",
                    nulls=0,
                    rows=7,
                )
                result, budget = self.scan(section)
                self.assertEqual(result.repetition_span, (0, rep_end))
                self.assertEqual(result.definition_span, (rep_end, def_end))
                self.assertEqual(result.values_span, (def_end, def_end + 1))
                self.counts(result, non_nulls=8, rows=7, first=0)
                self.counters(budget, levels=16, work_units=work)

    def test_multibyte_repeated_scalar_is_little_endian(self) -> None:
        for version, end, work in (("v1", 7, 20), ("v2", 3, 16)):
            with self.subTest(version=version):
                section = self.section(
                    version,
                    n=3,
                    max_def=0x1234,
                    definition=b"\x06\x34\x12",
                    nulls=0,
                )
                result, budget = self.scan(section)
                self.assertEqual(result.definition_span, (0, end))
                self.counts(result, positions=3, non_nulls=3, rows=3)
                self.counters(budget, levels=6, work_units=work)

    def test_multiple_runs_with_different_stream_boundaries_reconcile(self) -> None:
        cases = (
            (8, b"\x04\x00\x03\x84\x0c", b"\x06\x03\x0a\x02", 3, 5, 49, 41),
            (11, b"\x03\xc0\x00\x06\x00", b"\x16\x03", 11, 10, 55, 47),
        )
        for n, repetition, definition, non_nulls, rows, v1_work, v2_work in cases:
            for version, rep_end, end, work in (
                ("v1", 9, 17 if n == 8 else 15, v1_work),
                ("v2", 5, 9 if n == 8 else 7, v2_work),
            ):
                with self.subTest(version=version, n=n):
                    result, budget = self.scan(
                        self.section(
                            version,
                            n=n,
                            max_rep=3,
                            max_def=3,
                            repetition=repetition,
                            definition=definition,
                            nulls=n - non_nulls,
                            rows=rows,
                        )
                    )
                    self.assertEqual(result.repetition_span, (0, rep_end))
                    self.assertEqual(result.definition_span, (rep_end, end))
                    self.assertEqual(result.values_span, (end, end + 1))
                    self.counts(result, positions=n, non_nulls=non_nulls, rows=rows)
                    self.counters(budget, levels=2 * n, work_units=work)

    def test_single_byte_repeated_scalar_has_independent_exact_accounting(self) -> None:
        for version, work in (("v1", 19), ("v2", 15)):
            with self.subTest(version=version):
                section = self.section(version, n=3, max_def=2, definition=b"\x06\x02", nulls=0)
                result, budget = self.scan(section)
                self.counts(result, positions=3, non_nulls=3, rows=3)
                self.counters(budget, levels=6, work_units=work)

    def test_repeated_zero_level_is_distinct_from_a_missing_scalar(self) -> None:
        for version, work in (("v1", 19), ("v2", 15)):
            with self.subTest(version=version):
                result, budget = self.scan(
                    self.section(version, n=3, max_def=2, definition=b"\x06\x00", nulls=3)
                )
                self.counts(result, positions=3, non_nulls=0, rows=3)
                self.counters(budget, levels=6, work_units=work)
                self.refused(
                    self.section(version, n=3, max_def=2, definition=b"\x06", nulls=3),
                    "truncated_wire",
                )

    def test_v1_level_streams_use_their_own_encoding_declarations(self) -> None:
        for rep_encoding, def_encoding, repetition, definition, rep_end, end, work, first in (
            (4, 3, b"\xc0\x00", b"\x10\x03", 2, 8, 37, 3),
            (3, 4, b"\x03\xc0\x00", b"\xff\xff", 7, 9, 38, 0),
        ):
            with self.subTest(rep_encoding=rep_encoding, def_encoding=def_encoding):
                result, budget = self.scan(
                    self.section(
                        max_rep=3,
                        max_def=3,
                        rep_encoding=rep_encoding,
                        def_encoding=def_encoding,
                        repetition=repetition,
                        definition=definition,
                    )
                )
                self.assertEqual(result.repetition_span, (0, rep_end))
                self.assertEqual(result.definition_span, (rep_end, end))
                self.assertEqual(result.values_span, (end, end + 1))
                self.counts(result, non_nulls=8, rows=7, first=first)
                self.counters(budget, levels=16, work_units=work)

    def test_legacy_vector_uses_msb_order_without_a_length_prefix(self) -> None:
        section = self.section(def_encoding=4, definition=b"\x05\x39\x77", prefix=b"\xee")
        result, budget = self.scan(section)
        self.assertEqual(result.repetition_span, (1, 1))
        self.assertEqual(result.definition_span, (1, 4))
        self.assertEqual(result.values_span, (4, 5))
        self.counts(result)
        self.counters(budget, levels=16, work_units=30)

    def test_legacy_last_byte_requires_zero_unused_bits(self) -> None:
        for raw in (b"\x00", b"\x40", b"\x80", b"\xc0"):
            with self.subTest(raw=raw):
                result, budget = self.scan(
                    self.section(n=1, max_def=3, def_encoding=4, definition=raw, suffix=b"")
                )
                self.counts(
                    result,
                    positions=1,
                    non_nulls=1 if raw == b"\xc0" else 0,
                    rows=1,
                )
                self.counters(budget, levels=2, work_units=7)
        self.refused(
            self.section(n=1, max_def=3, def_encoding=4, definition=b"\x01"),
            "schema_mismatch",
        )

    def test_terminal_literal_padding_has_no_logical_range_or_null_observations(self) -> None:
        for version, work in (("v1", 27), ("v2", 23)):
            with self.subTest(version=version):
                result, budget = self.scan(self.section(version, n=5, max_def=5, nulls=5))
                self.counts(result, positions=5, non_nulls=0, rows=5)
                self.counters(budget, levels=10, work_units=work)

    def test_repeated_surplus_and_extra_complete_literal_groups_are_unsupported(self) -> None:
        for version in ("v1", "v2"):
            for raw in (
                b"\x12\x07",
                b"\xfe\xff\xff\xff\x0f\x07",
                b"\x05" + b"\x00" * 6,
            ):
                with self.subTest(version=version, raw=raw):
                    self.refused(self.section(version, definition=raw), "unsupported_mode")

    def test_trailing_section_content_is_not_parsed_to_salvage_a_result(self) -> None:
        for version in ("v1", "v2"):
            for tail in (b"\x02\x00", b"\x00", b"\x80" * 6):
                with self.subTest(version=version, tail=tail):
                    self.refused(
                        self.section(version, definition=b"\x03\x88\xc6\xfa" + tail),
                        "unsupported_mode",
                    )

    def test_each_truncated_literal_prefix_refuses_even_if_some_values_fit(self) -> None:
        for version in ("v1", "v2"):
            for end in range(4):
                with self.subTest(version=version, end=end):
                    self.refused(
                        self.section(version, definition=b"\x03\x88\xc6\xfa"[:end]),
                        "truncated_wire",
                    )
        self.refused(
            self.section("v2", n=1, definition=b"\x03\x00", nulls=1),
            "truncated_wire",
        )

    def test_v1_prefix_and_v2_declared_sections_cannot_read_beyond_the_span(self) -> None:
        for version in ("v1", "v2"):
            section = self.section(version)
            for end in range(4):
                with self.subTest(version=version, end=end):
                    self.refused(self.forged(section, stop=end), "truncated_wire")
        section = self.section()
        changed = bytearray(section.payload)
        changed[:4] = b"\xff\xff\xff\xff"
        self.refused(self.forged(section, payload=bytes(changed)), "truncated_wire")

    def test_zero_overflowing_and_unterminated_run_headers_are_malformed(self) -> None:
        cases = (
            b"\x00",
            b"\x01",
            b"\x80" * 5,
            b"\x80" * 5 + b"\x00",
            b"\xff\xff\xff\xff\x10",
            b"\xff\xff\xff\xff\x0f",
        )
        for version in ("v1", "v2"):
            for raw in cases:
                with self.subTest(version=version, raw=raw):
                    self.refused(self.section(version, definition=raw), "schema_mismatch")
        for version in ("v1", "v2"):
            self.refused(self.section(version, definition=b"\x80"), "truncated_wire")

    def test_nonminimal_but_bounded_run_varint_is_not_rejected_as_noncanonical(self) -> None:
        for version, work in (("v1", 37), ("v2", 33)):
            with self.subTest(version=version):
                result, budget = self.scan(
                    self.section(version, definition=b"\x83\x00\x88\xc6\xfa")
                )
                self.counts(result)
                self.counters(budget, levels=16, work_units=work)

    def test_logical_values_outside_the_schema_maximum_refuse(self) -> None:
        for version in ("v1", "v2"):
            for raw in (b"\x10\x07", b"\x03\xff\xff"):
                with self.subTest(version=version, raw=raw):
                    self.refused(
                        self.section(version, max_def=2, definition=raw),
                        "schema_mismatch",
                    )

    def test_v2_null_and_row_counts_must_match_the_scanned_levels(self) -> None:
        for changes in (
            {"nulls": 6},
            {"rows": 7},
            {"nulls": 8},
        ):
            with self.subTest(changes=changes):
                self.refused(self.section("v2", **changes), "schema_mismatch")

    def test_v1_can_continue_a_row_but_v2_must_begin_at_repetition_zero(self) -> None:
        changes = {
            "max_def": 3,
            "max_rep": 3,
            "repetition": b"\x03\x03\x00",
            "definition": b"\x10\x03",
            "nulls": 0,
            "rows": 7,
        }
        result, budget = self.scan(self.section("v1", **changes))
        self.counts(result, non_nulls=8, rows=7, first=3)
        self.counters(budget, levels=16, work_units=43)
        self.refused(self.section("v2", **changes), "schema_mismatch")

    def test_v2_opaque_value_remainder_is_untouched_for_each_compression_flag(self) -> None:
        for flag in (None, False, True):
            with self.subTest(flag=flag):
                section = self.section("v2", compression=flag, suffix=b"\xff\x80\x00\xff")
                result, budget = self.scan(section)
                self.assertEqual(result.values_span, (4, 8))
                self.assertEqual(section.payload[4:8], b"\xff\x80\x00\xff")
                self.counts(result)
                self.counters(budget, levels=16, work_units=32)

    def test_implicit_streams_are_constant_space_and_charge_logical_positions(self) -> None:
        for version in ("v1", "v2"):
            for n in (0, 1, (1 << 31) - 1):
                with self.subTest(version=version, n=n):
                    section = self.section(version, n=n, max_def=0, definition=b"", nulls=0)
                    result, budget = self.scan(
                        section,
                        self.budget(max_levels=max(1, 2 * n), max_work_units=3),
                    )
                    self.assertEqual(result.repetition_span, (0, 0))
                    self.assertEqual(result.definition_span, (0, 0))
                    self.assertEqual(result.values_span, (0, 1))
                    self.counts(
                        result,
                        positions=n,
                        non_nulls=n,
                        rows=n,
                        first=0 if n else None,
                    )
                    self.counters(budget, levels=2 * n, work_units=3)

    def test_unused_v1_encoding_declarations_do_not_create_implicit_bytes(self) -> None:
        for encoding in (0, 2, 3, 4, 5, 6, 7, 8, 9):
            with self.subTest(encoding=encoding):
                result, budget = self.scan(
                    self.section(
                        n=2,
                        max_def=0,
                        definition=b"",
                        def_encoding=encoding,
                        rep_encoding=encoding,
                    )
                )
                self.counts(result, positions=2, non_nulls=2, rows=2)
                self.assertEqual(result.values_span, (0, 1))
                self.counters(budget, levels=4, work_units=3)

    def test_zero_positions_preserve_each_active_streams_framing(self) -> None:
        for version, encoding, end, work in (
            ("v1", 3, 4, 7),
            ("v1", 4, 0, 3),
            ("v2", 3, 0, 3),
        ):
            with self.subTest(version=version, encoding=encoding):
                result, budget = self.scan(
                    self.section(version, n=0, definition=b"", def_encoding=encoding, suffix=b"")
                )
                self.assertEqual(result.definition_span, (0, end))
                self.assertEqual(result.values_span, (end, end))
                self.counts(result, positions=0, non_nulls=0, rows=0, first=None)
                self.counters(budget, levels=0, work_units=work)

    def test_v2_nonempty_zero_width_sections_are_explicitly_unsupported(self) -> None:
        for changes in (
            {"max_def": 0, "definition": b"\x02"},
            {"max_rep": 0, "repetition": b"\x02"},
        ):
            with self.subTest(changes=changes):
                self.refused(self.section("v2", **changes), "unsupported_mode")

    def test_exact_and_one_below_level_and_work_limits(self) -> None:
        for name, version, options, work, counts in self.positive_cases():
            with self.subTest(name=name, version=version):
                n, non_nulls, rows, first = counts
                section = self.section(version, **options)
                result, budget = self.scan(
                    section,
                    self.budget(max_levels=max(1, 2 * n), max_work_units=work),
                )
                self.counts(result, positions=n, non_nulls=non_nulls, rows=rows, first=first)
                self.counters(budget, levels=2 * n, work_units=work)
                if n:
                    budget = self.budget(max_levels=2 * n - 1)
                    self.refused(section, "resource_limit", budget)
                    self.counters(budget)
                budget = self.budget(max_work_units=work - 1)
                self.refused(section, "resource_limit", budget)
                self.counters(budget, levels=2 * n, work_units=work - 1)

    def test_active_reservation_boundaries_refuse_atomically_and_check_deadline(self) -> None:
        section = self.section()
        for ceiling, spent in (
            (1, 1),
            (2, 2),
            (6, 6),
            (7, 7),
            (8, 8),
            (10, 8),
            (11, 11),
            (12, 12),
            (13, 13),
            (35, 35),
        ):
            with self.subTest(ceiling=ceiling):
                budget = self.budget(max_work_units=ceiling)
                self.refused(section, "resource_limit", budget)
                self.counters(budget, levels=16, work_units=spent)
        for prefix in (1, 2, 6, 7, 8, 11, 12, 13, 35):
            with self.subTest(deadline_prefix=prefix):
                budget = self.budget()
                reserve = budget.reserve
                now = [0]
                reached = [False]

                def expire_at_prefix(
                    charge,
                    *,
                    reserve=reserve,
                    budget=budget,
                    now=now,
                    reached=reached,
                    prefix=prefix,
                ):
                    reserve(charge)
                    if budget.work_units == prefix:
                        reached[0] = True
                        now[0] = 10

                with (
                    patch.object(
                        budget.read_budget, "clock_ns", side_effect=lambda now=now: now[0]
                    ),
                    patch.object(budget, "reserve", expire_at_prefix),
                ):
                    self.refused(section, "deadline_exceeded", budget)
                self.assertTrue(reached[0])
                self.counters(budget, levels=16, work_units=prefix)

    def test_repeated_scans_accumulate_without_reset_or_refund(self) -> None:
        section = self.section()
        budget = self.budget(max_levels=32, max_work_units=72)
        for _ in range(2):
            result, _ = self.scan(section, budget)
            self.counts(result)
        self.counters(budget, levels=32, work_units=72)
        self.refused(section, "resource_limit", budget)
        self.counters(budget, levels=32, work_units=72)

    def test_malformed_scan_retains_entry_and_completed_work_reservations(self) -> None:
        for version, spent, total in (("v1", 8, 44), ("v2", 4, 36)):
            with self.subTest(version=version):
                budget = self.budget()
                self.refused(
                    self.section(version, definition=b"\x03\x88"),
                    "truncated_wire",
                    budget,
                )
                self.counters(budget, levels=16, work_units=spent)
                result, _ = self.scan(self.section(version), budget)
                self.counts(result)
                self.counters(budget, levels=32, work_units=total)

    def test_scanner_does_not_recharge_header_values_or_touch_other_axes(self) -> None:
        budget = self.budget()
        budget.reserve(
            bounded_parquet._PageCharge(values=8, compressed_bytes=16, uncompressed_bytes=20)
        )
        result, _ = self.scan(self.section(), budget)
        self.counts(result)
        self.counters(
            budget,
            values=8,
            compressed_bytes=16,
            uncompressed_bytes=20,
            levels=16,
            work_units=36,
        )

    def test_invalid_input_fields_refuse_before_clock_or_counter_mutation(self) -> None:
        class IntegerSubclass(int):
            pass

        class PayloadSubclass(bytes):
            pass

        base = self.section()
        cases = [
            {"payload": value}
            for value in (
                None,
                "bytes",
                bytearray(base.payload),
                memoryview(base.payload),
                PayloadSubclass(base.payload),
            )
        ]
        cases += [
            {name: value}
            for name in ("start", "stop")
            for value in (True, 0.0, IntegerSubclass(0), -1, len(base.payload) + 1)
        ]
        cases += [
            {"start": 2, "stop": 1},
            {"header": SimpleNamespace()},
            {"header": bounded_parquet._PageDictionaryHeader(1, 0, None)},
            {"max_definition_level": -1},
            {"max_definition_level": True},
            {"max_definition_level": 1.0},
            {"max_definition_level": IntegerSubclass(7)},
            {"max_repetition_level": -1},
            {"max_repetition_level": True},
            {"max_repetition_level": 1.0},
            {"max_repetition_level": IntegerSubclass(0)},
            {"max_repetition_level": 8},
        ]
        for changes in cases:
            with self.subTest(changes=tuple(changes)):
                section = self.forged(base, **changes)
                budget = self.budget()
                with patch.object(
                    budget.read_budget,
                    "clock_ns",
                    side_effect=AssertionError("invalid input reached clock"),
                ):
                    self.refused(section, "invalid_contract", budget)
                self.counters(budget)

    def test_wrong_input_and_budget_types_are_not_duck_typed(self) -> None:
        from dataclasses import fields

        class BudgetSubclass(bounded_parquet._PageBudget):
            pass

        class HeaderSubclass(bounded_parquet._PageDataV1Header):
            pass

        section = self.section()
        input_subclass = type("InputSubclass", (type(section),), {})
        derived = input_subclass(
            **{field.name: getattr(section, field.name) for field in fields(section)}
        )
        for bad in (None, SimpleNamespace(), section.payload, derived):
            budget = self.budget()
            with (
                self.subTest(input_type=type(bad).__name__),
                patch.object(
                    budget.read_budget,
                    "clock_ns",
                    side_effect=AssertionError("wrong input type reached clock"),
                ),
            ):
                self.refused(bad, "invalid_contract", budget)
            self.counters(budget)
        budget = self.budget()
        with patch.object(
            budget.read_budget,
            "clock_ns",
            side_effect=AssertionError("header subclass reached clock"),
        ):
            self.refused(
                self.forged(section, header=HeaderSubclass(8, 0, 3, 3, None)),
                "invalid_contract",
                budget,
            )
        self.counters(budget)
        valid = self.budget()
        derived_budget = BudgetSubclass(
            read_budget=valid.read_budget,
            wire_budget=valid.wire_budget,
            limits=valid.limits,
        )
        for bad in (None, SimpleNamespace(), derived_budget):
            with (
                self.subTest(budget_type=type(bad).__name__),
                self.assertRaises(PinnedReadRefused) as raised,
            ):
                self.api()._scan_page_levels(section, budget=bad)
            self.assertEqual(raised.exception.reason_code, "invalid_contract")

    def test_invalid_header_scalars_are_revalidated_before_clock(self) -> None:
        cases = [
            (version, "num_values", value)
            for version in ("v1", "v2")
            for value in (-1, True, 1.0, 1 << 31)
        ]
        cases += [
            ("v2", name, value)
            for name in (
                "num_nulls",
                "num_rows",
                "definition_levels_byte_length",
                "repetition_levels_byte_length",
            )
            for value in (-1, True, 1.0, 1 << 31)
        ]
        cases += [
            ("v1", "definition_level_encoding", True),
            ("v1", "repetition_level_encoding", 1.0),
            ("v2", "num_nulls", 9),
            ("v2", "num_rows", 9),
            ("v2", "is_compressed", 1),
            ("v2", "effective_is_compressed", None),
            ("v2", "effective_is_compressed", False),
        ]
        for version, name, value in cases:
            with self.subTest(version=version, field=name, value=value):
                section = self.section(version)
                section = self.forged(section, header=self.forged(section.header, **{name: value}))
                budget = self.budget()
                with patch.object(
                    budget.read_budget,
                    "clock_ns",
                    side_effect=AssertionError("invalid header reached clock"),
                ):
                    self.refused(section, "invalid_contract", budget)
                self.counters(budget)

    def test_unsupported_active_encodings_and_widths_do_not_become_profiles(self) -> None:
        for encoding in (0, 2, 5, 6, 7, 8, 9):
            with self.subTest(encoding=encoding):
                self.refused(self.section(def_encoding=encoding), "unsupported_mode")
        self.refused(self.section(max_def=1 << 64), "unsupported_mode")

    def test_maximum_supported_level_width_has_bounded_scalar_state(self) -> None:
        for version, work in (("v1", 20), ("v2", 16)):
            with self.subTest(version=version):
                result, budget = self.scan(
                    self.section(
                        version,
                        n=1,
                        max_def=(1 << 64) - 1,
                        definition=b"\x02" + b"\xff" * 8,
                        nulls=0,
                    )
                )
                self.counts(result, positions=1, non_nulls=1, rows=1)
                self.counters(budget, levels=2, work_units=work)

    def test_entry_deadline_prevents_all_level_and_work_charges(self) -> None:
        section = self.section()
        budget = self.budget()
        with patch.object(budget.read_budget, "clock_ns", return_value=10):
            self.refused(section, "deadline_exceeded", budget)
        self.counters(budget)

    def test_deadline_after_entry_preserves_only_the_atomic_entry_charge(self) -> None:
        section = self.section()
        budget = self.budget()
        reserve = budget.reserve
        now = [0]
        calls = []

        def expire_after_entry(charge):
            reserve(charge)
            calls.append(charge)
            now[0] = 10

        with (
            patch.object(budget.read_budget, "clock_ns", side_effect=lambda: now[0]),
            patch.object(budget, "reserve", expire_after_entry),
        ):
            self.refused(section, "deadline_exceeded", budget)
        self.assertEqual(len(calls), 1)
        self.counters(budget, levels=16, work_units=1)

    def test_late_deadline_prevents_summary_after_implicit_counts(self) -> None:
        section = self.section(n=3, max_def=0, definition=b"")
        budget = self.budget()
        reserve = budget.reserve
        now = [0]
        calls = []

        def expire_before_summary(charge):
            reserve(charge)
            calls.append(charge)
            if len(calls) == 2:
                now[0] = 10

        with (
            patch.object(budget.read_budget, "clock_ns", side_effect=lambda: now[0]),
            patch.object(budget, "reserve", expire_before_summary),
        ):
            self.refused(section, "deadline_exceeded", budget)
        self.assertEqual(len(calls), 2)
        self.counters(budget, levels=6, work_units=2)


if __name__ == "__main__":
    unittest.main()


class PageChecksumTests(unittest.TestCase):
    vectors = (
        (b"", 0x00000000),
        (b"123456789", 0xCBF43926 - (1 << 32)),
        (b"abc", 0x352441C2),
        (b"\x00", 0xD202EF8D - (1 << 32)),
        (b"\xff", 0xFF000000 - (1 << 32)),
    )

    def api(self) -> SimpleNamespace:
        value = getattr(bounded_parquet, "_verify_page_crc", None)
        self.assertTrue(callable(value), "missing callable _verify_page_crc")
        return SimpleNamespace(verify=value)

    def budget(self, **changes: int) -> bounded_parquet._PageBudget:
        limits = dict.fromkeys(("max_" + axis for axis in PageBudgetTests.axes), 1 << 40)
        limits.update(changes)
        return PageBudgetTests().budget(**limits)

    def state(self, budget: bounded_parquet._PageBudget) -> dict[str, int]:
        return {axis: getattr(budget, axis) for axis in PageBudgetTests.axes}

    def assert_work(
        self,
        budget: bounded_parquet._PageBudget,
        before: dict[str, int],
        amount: int,
    ) -> None:
        expected = before.copy()
        expected["work_units"] += amount
        self.assertEqual(self.state(budget), expected)

    def test_private_callable_has_exact_keyword_only_boundary(self) -> None:
        import inspect

        parameters = inspect.signature(self.api().verify).parameters
        self.assertEqual(tuple(parameters), ("payload", "start", "stop", "crc", "budget"))
        self.assertEqual(parameters["payload"].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in ("start", "stop", "crc", "budget"):
            self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIs(parameters[name].default, inspect.Parameter.empty)

    def test_standard_anchors_preserve_other_page_wire_and_io_counters(self) -> None:
        api = self.api()
        for payload, crc in self.vectors:
            with self.subTest(payload=payload):
                budget = self.budget()
                before = self.state(budget)
                read_before = vars(budget.read_budget).copy()
                wire_before = vars(budget.wire_budget).copy()
                self.assertIsNone(
                    api.verify(payload, start=0, stop=len(payload), crc=crc, budget=budget)
                )
                self.assert_work(budget, before, 2 + 9 * len(payload))
                self.assertEqual(vars(budget.read_budget), read_before)
                self.assertEqual(vars(budget.wire_budget), wire_before)

    def test_nonzero_spans_ignore_changed_header_and_trailing_bytes(self) -> None:
        api = self.api()
        for payload, crc in self.vectors:
            for prefix, suffix in ((b"header", b"tail"), (b"\xff\x00", b"\x00\xffmore")):
                with self.subTest(payload=payload, prefix=prefix):
                    budget = self.budget()
                    before = self.state(budget)
                    self.assertIsNone(
                        api.verify(
                            prefix + payload + suffix,
                            start=len(prefix),
                            stop=len(prefix) + len(payload),
                            crc=crc,
                            budget=budget,
                        )
                    )
                    self.assert_work(budget, before, 2 + 9 * len(payload))

    def test_absent_checksum_has_constant_work_and_present_zero_is_checked(self) -> None:
        api = self.api()
        for payload in (b"", b"abc", b"123456789"):
            with self.subTest(absent_payload=payload):
                budget = self.budget()
                before = self.state(budget)
                self.assertIsNone(
                    api.verify(payload, start=0, stop=len(payload), crc=None, budget=budget)
                )
                self.assert_work(budget, before, 2)
        budget = self.budget()
        with self.assertRaises(bounded_parquet._CompactWireMalformed) as caught:
            api.verify(b"\x00", start=0, stop=1, crc=0, budget=budget)
        self.assertEqual(caught.exception.reason_code, "invalid_wire")
        self.assertEqual(str(caught.exception), "invalid_wire")
        self.assertEqual(budget.work_units, 11)

    def test_each_mismatch_is_corruption_and_retains_full_work(self) -> None:
        api = self.api()
        for payload, crc in self.vectors:
            with self.subTest(payload=payload):
                budget = self.budget()
                before = self.state(budget)
                with self.assertRaises(bounded_parquet._CompactWireMalformed) as caught:
                    api.verify(payload, start=0, stop=len(payload), crc=crc ^ 1, budget=budget)
                self.assertEqual(caught.exception.reason_code, "invalid_wire")
                self.assertEqual(str(caught.exception), "invalid_wire")
                self.assert_work(budget, before, 2 + 9 * len(payload))

    def test_every_positive_vector_has_exact_and_one_below_work_boundary(self) -> None:
        api = self.api()
        cases = self.vectors + ((b"123456789", None),)
        for payload, crc in cases:
            required = 2 if crc is None else 2 + 9 * len(payload)
            with self.subTest(payload=payload, crc=crc, allowance=required):
                budget = self.budget(max_work_units=required)
                before = self.state(budget)
                self.assertIsNone(
                    api.verify(payload, start=0, stop=len(payload), crc=crc, budget=budget)
                )
                self.assert_work(budget, before, required)
            with self.subTest(payload=payload, crc=crc, allowance=required - 1):
                budget = self.budget(max_work_units=required - 1)
                before = self.state(budget)
                with self.assertRaises(PinnedReadRefused) as caught:
                    api.verify(payload, start=0, stop=len(payload), crc=crc, budget=budget)
                self.assertEqual(caught.exception.reason_code, "resource_limit")
                self.assert_work(budget, before, required - 1)

    def test_cumulative_calls_do_not_recharge_declared_body_dimensions(self) -> None:
        api = self.api()
        budget = self.budget(max_work_units=24)
        budget.reserve(
            bounded_parquet._PageCharge(
                work_units=2, compressed_bytes=1, uncompressed_bytes=1, values=1
            )
        )
        before = self.state(budget)
        for _ in range(2):
            self.assertIsNone(
                api.verify(b"\x00", start=0, stop=1, crc=0xD202EF8D - (1 << 32), budget=budget)
            )
        self.assert_work(budget, before, 22)
        full = self.state(budget)
        with self.assertRaises(PinnedReadRefused) as caught:
            api.verify(b"", start=0, stop=0, crc=0, budget=budget)
        self.assertEqual(caught.exception.reason_code, "resource_limit")
        self.assertEqual(self.state(budget), full)

    def test_final_resource_refusal_precedes_checksum_mismatch(self) -> None:
        api = self.api()
        for payload in (b"", b"\x00"):
            required = 2 + 9 * len(payload)
            with self.subTest(payload=payload):
                budget = self.budget(max_work_units=required - 1)
                before = self.state(budget)
                with self.assertRaises(PinnedReadRefused) as caught:
                    api.verify(payload, start=0, stop=len(payload), crc=1, budget=budget)
                self.assertEqual(caught.exception.reason_code, "resource_limit")
                self.assert_work(budget, before, required - 1)

    def test_deadline_at_each_checkpoint_retains_only_admitted_work(self) -> None:
        api = self.api()
        for payload, crc, required in (
            (b"\x00", 0xD202EF8D - (1 << 32), 11),
            (b"abc", None, 2),
            (b"", 1, 2),
        ):
            for fail_at in range(1, required + 1):
                with self.subTest(payload=payload, crc=crc, fail_at=fail_at):
                    budget = self.budget()
                    before = self.state(budget)
                    calls = [0]

                    def clock(failure: int = fail_at, count: list[int] = calls) -> int:
                        count[0] += 1
                        return 10 if count[0] == failure else 0

                    budget.read_budget.clock_ns = clock
                    with self.assertRaises(PinnedReadRefused) as caught:
                        api.verify(payload, start=0, stop=len(payload), crc=crc, budget=budget)
                    self.assertEqual(caught.exception.reason_code, "deadline_exceeded")
                    self.assertEqual(calls[0], fail_at)
                    self.assert_work(budget, before, fail_at - 1)

    def test_invalid_arguments_precede_clock_and_leave_all_counters_unchanged(self) -> None:
        api = self.api()

        class DerivedBytes(bytes):
            pass

        class DerivedInt(int):
            pass

        class DerivedBudget(bounded_parquet._PageBudget):
            pass

        cases = [
            {"payload": value}
            for value in (None, bytearray(b"x"), memoryview(b"x"), "x", DerivedBytes(b"x"))
        ]
        cases += [{"start": value} for value in (None, False, 0.0, DerivedInt(0), -1, 2)]
        cases += [{"stop": value} for value in (None, True, 1.0, DerivedInt(1), -1, 2)]
        cases += [
            {"crc": value}
            for value in (False, 0.0, "0", DerivedInt(0), -(1 << 31) - 1, 1 << 31, 0xFFFFFFFF)
        ]
        cases += [
            {"budget": value} for value in (None, SimpleNamespace(), object.__new__(DerivedBudget))
        ]
        cases.append({"start": 1, "stop": 0})

        def forbidden_clock() -> int:
            raise AssertionError("invalid caller reached clock")

        for changes in cases:
            with self.subTest(changes=changes):
                budget = self.budget()
                budget.read_budget.clock_ns = forbidden_clock
                before = self.state(budget)
                read_before = vars(budget.read_budget).copy()
                wire_before = vars(budget.wire_budget).copy()
                arguments = {"payload": b"x", "start": 0, "stop": 1, "crc": 0, "budget": budget}
                arguments.update(changes)
                with self.assertRaises(PinnedReadRefused) as caught:
                    api.verify(**arguments)
                self.assertEqual(caught.exception.reason_code, "invalid_contract")
                self.assertEqual(self.state(budget), before)
                self.assertEqual(vars(budget.read_budget), read_before)
                self.assertEqual(vars(budget.wire_budget), wire_before)


class PageHeaderDemandInputTests(unittest.TestCase):
    wire_axes = (
        "total_wire_bytes",
        "work_units",
        "total_fields",
        "total_items",
        "total_string_bytes",
        "copied_payload_bytes",
    )

    def api(self) -> SimpleNamespace:
        names = (
            "_CompactWireCursor",
            "_PageBudget",
            "_PageCharge",
            "_PinnedReadBudget",
            "_admit_page_header",
        )
        found = {}
        for name in names:
            value = getattr(bounded_parquet, name, None)
            self.assertTrue(callable(value), f"missing canonical callable {name}")
            found[name] = value
        return SimpleNamespace(**found)

    def environment(
        self,
        data: bytes,
        *,
        wire: dict[str, int] | None = None,
        page: dict[str, int] | None = None,
        read: dict[str, int] | None = None,
        clock: object | None = None,
    ) -> SimpleNamespace:
        from dataclasses import replace

        api = self.api()
        wire_values = {
            "max_input_bytes": 1 << 30,
            "max_object_bytes": 1 << 30,
            "max_header_bytes": 1 << 30,
            "max_depth": 100,
            "max_fields_per_struct": 1000,
            "max_total_fields": 10000,
            "max_items_per_container": 1 << 30,
            "max_total_items": 1 << 30,
            "max_string_bytes": 1 << 30,
            "max_total_string_bytes": 1 << 30,
            "max_varint_bytes": 10,
            "max_total_wire_bytes": 1 << 30,
            "max_work_units": 1 << 40,
            "max_copied_payload_bytes": 1 << 40,
        }
        wire_values.update(wire or {})
        wire_budget = CompactWireCursorTests().budget(**wire_values)
        page_values = dict.fromkeys(("max_" + axis for axis in PageBudgetTests.axes), 1 << 40)
        page_values.update(page or {})
        prototype = PageBudgetTests().budget(**page_values)
        read_values = {"max_source_size_bytes": 1 << 30, "max_total_read_bytes": 1 << 30}
        read_values.update(read or {})
        read_budget = api._PinnedReadBudget(
            replace(prototype.read_budget.limits, **read_values),
            clock_ns=(lambda: 0) if clock is None else clock,
        )
        page_budget = api._PageBudget(
            read_budget=read_budget, wire_budget=wire_budget, limits=prototype.limits
        )
        env = SimpleNamespace(
            api=api, data=data, wire=wire_budget, page=page_budget, read=read_budget, calls=[]
        )

        def counted_fetch(offset: int, length: int) -> bytes:
            env.calls.append((offset, length))
            self.assertEqual(read_budget.before_read(length), length)
            segment = data[offset : offset + length]
            read_budget.record_read(len(segment))
            return segment

        env.fetch = counted_fetch
        return env

    def cursor(
        self,
        env: SimpleNamespace,
        *,
        stop: object = None,
        **changes: object,
    ) -> bounded_parquet._CompactWireCursor:
        arguments = {
            "payload": b"",
            "start": 0,
            "stop": len(env.data) if stop is None else stop,
            "budget": env.wire,
            "kind": "page_header",
            "fetch": env.fetch,
            "page_budget": env.page,
        }
        arguments.update(changes)
        return env.api._CompactWireCursor(**arguments)

    def assert_metrics(
        self,
        env: SimpleNamespace,
        cursor: bounded_parquet._CompactWireCursor,
        *,
        page_work: int = 0,
        allocation: int = 0,
        wire_bytes: int = 0,
        wire_work: int = 0,
        copied: int = 0,
        strings: int = 0,
        read_bytes: int = 0,
        position: int = 0,
        payload: bytes = b"",
    ) -> None:
        expected_page = dict.fromkeys(PageBudgetTests.axes, 0)
        expected_page.update(work_units=page_work, allocation_bytes=allocation)
        self.assertEqual(
            {axis: getattr(env.page, axis) for axis in PageBudgetTests.axes}, expected_page
        )
        self.assertEqual(
            {axis: getattr(env.wire, axis) for axis in self.wire_axes},
            {
                "total_wire_bytes": wire_bytes,
                "work_units": wire_work,
                "total_fields": 0,
                "total_items": 0,
                "total_string_bytes": strings,
                "copied_payload_bytes": copied,
            },
        )
        self.assertEqual(env.read.source_opens, 0)
        self.assertEqual(env.read.total_read_bytes, read_bytes)
        self.assertEqual(cursor.position, position)
        self.assertEqual(cursor.payload, payload)

    def test_constructor_exposes_optional_keyword_only_pair(self) -> None:
        import inspect

        parameters = inspect.signature(self.api()._CompactWireCursor).parameters
        for name in ("fetch", "page_budget"):
            self.assertIn(name, parameters)
            self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNone(parameters[name].default)

    def test_invalid_input_mode_contract_precedes_clocks_callbacks_and_counters(self) -> None:
        class DerivedBytes(bytes):
            pass

        class DerivedInt(int):
            pass

        class DerivedPageBudget(bounded_parquet._PageBudget):
            pass

        class DerivedKind(str):
            pass

        class DerivedWireBudget(bounded_parquet._CompactWireBudget):
            pass

        def forbidden_clock() -> int:
            raise AssertionError("invalid constructor reached clock")

        cases = (
            {"fetch": None},
            {"page_budget": None},
            {"fetch": 1},
            {"page_budget": object()},
            {"page_budget": object.__new__(DerivedPageBudget)},
            {"kind": "footer"},
            {"kind": DerivedKind("page_header")},
            {"budget": None},
            {"budget": object.__new__(DerivedWireBudget)},
            {"payload": b"x"},
            {"payload": bytearray()},
            {"payload": DerivedBytes()},
            {"start": 1},
            {"start": False},
            {"start": 0.0},
            {"start": DerivedInt(0)},
            {"stop": -1},
            {"stop": True},
            {"stop": 1.0},
            {"stop": DerivedInt(1)},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                env = self.environment(b"x", clock=forbidden_clock)
                with self.assertRaises(PinnedReadRefused) as caught:
                    self.cursor(env, **changes)
                self.assertEqual(caught.exception.reason_code, "invalid_contract")
                self.assertEqual(env.calls, [])
                self.assertEqual(
                    PageBudgetTests().state(env.page), (0,) * len(PageBudgetTests.axes)
                )
                self.assertEqual(
                    tuple(getattr(env.wire, axis) for axis in self.wire_axes), (0,) * 6
                )
                self.assertEqual(env.read.total_read_bytes, 0)

    def test_mismatched_wire_or_invalid_read_budget_precedes_any_clock(self) -> None:
        def forbidden_clock() -> int:
            raise AssertionError("invalid budget binding reached clock")

        class DerivedReadBudget(bounded_parquet._PinnedReadBudget):
            pass

        for invalid in ("wire", "read", "read_subclass"):
            with self.subTest(invalid=invalid):
                env = self.environment(b"x", clock=forbidden_clock)
                if invalid == "wire":
                    changes = {"budget": CompactWireCursorTests().budget()}
                else:
                    env.page.read_budget = (
                        None if invalid == "read" else object.__new__(DerivedReadBudget)
                    )
                    changes = {}
                with self.assertRaises(PinnedReadRefused) as caught:
                    self.cursor(env, **changes)
                self.assertEqual(caught.exception.reason_code, "invalid_contract")
                self.assertEqual(env.calls, [])
                self.assertEqual(
                    PageBudgetTests().state(env.page), (0,) * len(PageBudgetTests.axes)
                )
                self.assertEqual(env.read.total_read_bytes, 0)

    def test_omitted_and_explicit_absent_options_preserve_resident_path_without_clock(self) -> None:
        def forbidden_clock() -> int:
            raise AssertionError("resident cursor reached page clock")

        for options in ({}, {"fetch": None, "page_budget": None}):
            with self.subTest(options=options):
                env = self.environment(b"\xff", clock=forbidden_clock)
                cursor = env.api._CompactWireCursor(
                    b"\xff", 0, 1, env.wire, kind="page_header", **options
                )
                self.assertEqual(cursor.read_byte(), -1)
                self.assertEqual(env.calls, [])
                self.assert_metrics(
                    env, cursor, wire_bytes=1, wire_work=1, position=1, payload=b"\xff"
                )

    def test_first_incremental_byte_uses_the_refreshed_payload(self) -> None:
        env = self.environment(b"\xff")
        cursor = self.cursor(env)
        self.assertEqual(cursor.read_byte(), -1)
        self.assertEqual(env.calls, [(0, 1)])
        self.assert_metrics(
            env,
            cursor,
            page_work=3,
            allocation=2,
            wire_bytes=1,
            wire_work=2,
            copied=1,
            read_bytes=1,
            position=1,
            payload=b"\xff",
        )

    def test_double_uses_the_refreshed_eight_byte_payload(self) -> None:
        double = b"\x00\x00\x00\x00\x00\x00\xf0\x3f"
        for prefix in (b"", b"\x7f"):
            with self.subTest(prefix=prefix):
                data = prefix + double
                env = self.environment(data)
                cursor = self.cursor(env)
                if prefix:
                    self.assertEqual(cursor.read_byte(), 127)
                self.assertEqual(cursor.read_double(), 1.0)
                if prefix:
                    self.assertEqual(env.calls, [(0, 1), (1, 8)])
                    self.assert_metrics(
                        env,
                        cursor,
                        page_work=6,
                        allocation=19,
                        wire_bytes=9,
                        wire_work=19,
                        copied=10,
                        read_bytes=9,
                        position=9,
                        payload=data,
                    )
                else:
                    self.assertEqual(env.calls, [(0, 8)])
                    self.assert_metrics(
                        env,
                        cursor,
                        page_work=3,
                        allocation=16,
                        wire_bytes=8,
                        wire_work=16,
                        copied=8,
                        read_bytes=8,
                        position=8,
                        payload=data,
                    )

    def test_binary_span_offsets_and_prior_prefix_survive_later_refill(self) -> None:
        data = b"\x08ABCDEFGH\x7f"
        env = self.environment(data)
        cursor = self.cursor(env)
        span = cursor.read_binary_span()
        self.assertEqual(span, (1, 9))
        self.assertEqual(cursor.payload[span[0] : span[1]], b"ABCDEFGH")
        self.assert_metrics(
            env,
            cursor,
            page_work=6,
            allocation=19,
            wire_bytes=9,
            wire_work=19,
            copied=10,
            strings=8,
            read_bytes=9,
            position=9,
            payload=data[:9],
        )
        self.assertEqual(cursor.read_byte(), 127)
        self.assertEqual(cursor.payload[span[0] : span[1]], b"ABCDEFGH")
        self.assertEqual(env.calls, [(0, 1), (1, 8), (9, 1)])
        self.assert_metrics(
            env,
            cursor,
            page_work=9,
            allocation=30,
            wire_bytes=10,
            wire_work=30,
            copied=20,
            strings=8,
            read_bytes=10,
            position=10,
            payload=data,
        )

    def test_binary_copy_remains_stable_and_is_charged_separately(self) -> None:
        data = b"\x08ABCDEFGH\x7f"
        env = self.environment(data)
        cursor = self.cursor(env)
        copied_value = cursor.read_binary()
        self.assertEqual(copied_value, b"ABCDEFGH")
        self.assert_metrics(
            env,
            cursor,
            page_work=6,
            allocation=27,
            wire_bytes=9,
            wire_work=19,
            copied=18,
            strings=8,
            read_bytes=9,
            position=9,
            payload=data[:9],
        )
        self.assertEqual(cursor.read_byte(), 127)
        self.assertEqual(copied_value, b"ABCDEFGH")
        self.assert_metrics(
            env,
            cursor,
            page_work=9,
            allocation=38,
            wire_bytes=10,
            wire_work=30,
            copied=28,
            strings=8,
            read_bytes=10,
            position=10,
            payload=data,
        )

    def test_zero_length_binary_consume_has_entry_work_but_no_empty_callback(self) -> None:
        for copying in (False, True):
            with self.subTest(copying=copying):
                env = self.environment(b"\x00")
                cursor = self.cursor(env)
                value = cursor.read_binary() if copying else cursor.read_binary_span()
                self.assertEqual(value, b"" if copying else (1, 1))
                self.assertEqual(env.calls, [(0, 1)])
                self.assert_metrics(
                    env,
                    cursor,
                    page_work=4,
                    allocation=2,
                    wire_bytes=1,
                    wire_work=2,
                    copied=1,
                    read_bytes=1,
                    position=1,
                    payload=b"\x00",
                )

    def test_large_binary_span_uses_existing_read_segments_and_literal_accounting(self) -> None:
        data = b"\x81\x80\x40" + b"Z" * 1048577
        env = self.environment(data)
        cursor = self.cursor(env)
        self.assertEqual(cursor.read_binary_span(), (3, 1048580))
        self.assertEqual(env.calls, [(0, 1), (1, 1), (2, 1), (3, 1048576), (1048579, 1)])
        self.assertEqual(len(cursor.payload), 1048580)
        self.assertEqual(cursor.payload[:3], b"\x81\x80\x40")
        self.assertEqual(cursor.payload[-1:], b"Z")
        self.assertEqual(cursor.position, 1048580)
        self.assertEqual(env.page.work_units, 14)
        self.assertEqual(env.page.allocation_bytes, 3145745)
        self.assertEqual(env.wire.total_wire_bytes, 1048580)
        self.assertEqual(env.wire.work_units, 3145745)
        self.assertEqual(env.wire.copied_payload_bytes, 2097165)
        self.assertEqual(env.wire.total_string_bytes, 1048577)
        self.assertEqual(env.read.total_read_bytes, 1048580)
        self.assertEqual(
            {
                axis: getattr(env.page, axis)
                for axis in PageBudgetTests.axes
                if axis not in ("work_units", "allocation_bytes")
            },
            dict.fromkeys(
                (
                    axis
                    for axis in PageBudgetTests.axes
                    if axis not in ("work_units", "allocation_bytes")
                ),
                0,
            ),
        )

    def test_root_stop_does_not_acquire_following_serialized_body(self) -> None:
        header = b"\x15\x00\x15\x02\x15\x02\x2c\x15\x00\x15\x00\x15\x06\x15\x06\x00\x00"
        env = self.environment(header + b"\xff")
        cursor = self.cursor(env)
        admission = env.api._admit_page_header(cursor, chunk_remaining_bytes=18, budget=env.page)
        self.assertEqual(env.calls, [(offset, 1) for offset in range(17)])
        self.assertEqual(cursor.payload, header)
        self.assertEqual(cursor.position, 17)
        self.assertEqual(
            (
                admission.header_start,
                admission.header_end,
                admission.body_start,
                admission.body_end,
            ),
            (0, 17, 17, 18),
        )
        self.assertEqual((admission.compressed_page_size, admission.uncompressed_page_size), (1, 1))
        self.assertEqual(env.read.total_read_bytes, 17)
        self.assertEqual(
            (env.page.pages, env.page.compressed_bytes, env.page.uncompressed_bytes), (1, 1, 1)
        )
        self.assertFalse(admission.decode_eligible)

    def test_logical_chunk_bound_precedes_every_reservation_and_callback(self) -> None:
        for data, stop, operation in ((b"", 0, "read_byte"), (b"12345678", 7, "read_double")):
            with self.subTest(stop=stop):
                env = self.environment(data)
                cursor = self.cursor(env, stop=stop)
                with self.assertRaises(bounded_parquet._CompactWireMalformed) as caught:
                    getattr(cursor, operation)()
                self.assertEqual(caught.exception.reason_code, "truncated_wire")
                self.assertEqual(env.calls, [])
                self.assert_metrics(env, cursor)

    def test_logical_extent_can_exceed_small_resident_header_input_allowance(self) -> None:
        env = self.environment(b"\x7f" + b"ignored", wire={"max_input_bytes": 1})
        cursor = self.cursor(env)
        self.assertEqual(cursor.read_byte(), 127)
        self.assertEqual(env.calls, [(0, 1)])
        self.assert_metrics(
            env,
            cursor,
            page_work=3,
            allocation=2,
            wire_bytes=1,
            wire_work=2,
            copied=1,
            read_bytes=1,
            position=1,
            payload=b"\x7f",
        )
        with self.assertRaises(PinnedReadRefused) as caught:
            cursor.read_byte()
        self.assertEqual(caught.exception.reason_code, "resource_limit")
        self.assertEqual(env.calls, [(0, 1)])
        self.assert_metrics(
            env,
            cursor,
            page_work=3,
            allocation=2,
            wire_bytes=1,
            wire_work=2,
            copied=1,
            read_bytes=1,
            position=1,
            payload=b"\x7f",
        )

    def test_input_object_and_header_limits_refuse_before_wire_and_page_charges(self) -> None:
        for axis in ("max_input_bytes", "max_object_bytes", "max_header_bytes"):
            with self.subTest(axis=axis):
                env = self.environment(b"12345678", wire={axis: 7})
                cursor = self.cursor(env)
                with self.assertRaises(PinnedReadRefused) as caught:
                    cursor.read_double()
                self.assertEqual(caught.exception.reason_code, "resource_limit")
                self.assertEqual(env.calls, [])
                self.assert_metrics(env, cursor)

    def test_grammar_wire_reservation_failure_leaves_page_and_callback_untouched(self) -> None:
        for axis in ("max_total_wire_bytes", "max_work_units"):
            with self.subTest(axis=axis):
                env = self.environment(b"12345678", wire={axis: 7})
                cursor = self.cursor(env)
                with self.assertRaises(PinnedReadRefused) as caught:
                    cursor.read_double()
                self.assertEqual(caught.exception.reason_code, "resource_limit")
                self.assertEqual(env.calls, [])
                self.assert_metrics(env, cursor)

    def test_prefix_copy_wire_refusal_retains_only_grammar_and_page_entry(self) -> None:
        for changes in ({"max_copied_payload_bytes": 7}, {"max_work_units": 15}):
            with self.subTest(changes=changes):
                env = self.environment(b"12345678", wire=changes)
                cursor = self.cursor(env)
                with self.assertRaises(PinnedReadRefused) as caught:
                    cursor.read_double()
                self.assertEqual(caught.exception.reason_code, "resource_limit")
                self.assertEqual(env.calls, [])
                self.assert_metrics(env, cursor, page_work=1, wire_bytes=8, wire_work=8)

    def test_page_allocation_refusal_keeps_prepaid_wire_copy_without_fetch(self) -> None:
        env = self.environment(b"12345678", page={"max_allocation_bytes": 15})
        cursor = self.cursor(env)
        with self.assertRaises(PinnedReadRefused) as caught:
            cursor.read_double()
        self.assertEqual(caught.exception.reason_code, "resource_limit")
        self.assertEqual(env.calls, [])
        self.assert_metrics(env, cursor, page_work=1, wire_bytes=8, wire_work=16, copied=8)

    def test_page_entry_refusal_retains_grammar_without_refill_reservations(self) -> None:
        env = self.environment(b"12345678", page={"max_work_units": 1})
        env.page.reserve(env.api._PageCharge(work_units=1))
        cursor = self.cursor(env)
        with self.assertRaises(PinnedReadRefused) as caught:
            cursor.read_double()
        self.assertEqual(caught.exception.reason_code, "resource_limit")
        self.assertEqual(env.calls, [])
        self.assert_metrics(env, cursor, page_work=1, wire_bytes=8, wire_work=8)

    def test_prefetch_work_refusal_is_atomic_with_its_allocation(self) -> None:
        env = self.environment(b"12345678", page={"max_work_units": 1})
        cursor = self.cursor(env)
        with self.assertRaises(PinnedReadRefused) as caught:
            cursor.read_double()
        self.assertEqual(caught.exception.reason_code, "resource_limit")
        self.assertEqual(env.calls, [])
        self.assert_metrics(env, cursor, page_work=1, wire_bytes=8, wire_work=16, copied=8)

    def test_precopy_work_refusal_retains_read_but_not_new_prefix_or_position(self) -> None:
        env = self.environment(b"12345678", page={"max_work_units": 2})
        cursor = self.cursor(env)
        with self.assertRaises(PinnedReadRefused) as caught:
            cursor.read_double()
        self.assertEqual(caught.exception.reason_code, "resource_limit")
        self.assertEqual(env.calls, [(0, 8)])
        self.assert_metrics(
            env,
            cursor,
            page_work=2,
            allocation=16,
            wire_bytes=8,
            wire_work=16,
            copied=8,
            read_bytes=8,
        )

    def test_whole_missing_read_allowance_is_checked_before_any_partial_fetch(self) -> None:
        env = self.environment(b"12345678", read={"max_total_read_bytes": 7})
        cursor = self.cursor(env)
        with self.assertRaises(PinnedReadRefused) as caught:
            cursor.read_double()
        self.assertEqual(caught.exception.reason_code, "resource_limit")
        self.assertEqual(env.calls, [])
        self.assert_metrics(env, cursor, page_work=1, wire_bytes=8, wire_work=8)

    def test_binary_string_limits_keep_only_the_admitted_length_prefix(self) -> None:
        for axis in ("max_string_bytes", "max_total_string_bytes"):
            with self.subTest(axis=axis):
                env = self.environment(b"\x08ABCDEFGH", wire={axis: 7})
                cursor = self.cursor(env)
                with self.assertRaises(PinnedReadRefused) as caught:
                    cursor.read_binary_span()
                self.assertEqual(caught.exception.reason_code, "resource_limit")
                self.assertEqual(env.calls, [(0, 1)])
                self.assert_metrics(
                    env,
                    cursor,
                    page_work=3,
                    allocation=2,
                    wire_bytes=1,
                    wire_work=2,
                    copied=1,
                    read_bytes=1,
                    position=1,
                    payload=b"\x08",
                )

    def test_binary_copy_reservations_fail_before_binary_refill(self) -> None:
        cases = (
            ({"max_copied_payload_bytes": 8}, {}, 3, 2, 2, 1),
            ({}, {"max_allocation_bytes": 9}, 3, 2, 10, 9),
        )
        for wire, page, page_work, allocation, wire_work, copied in cases:
            with self.subTest(wire=wire, page=page):
                env = self.environment(b"\x08ABCDEFGH", wire=wire, page=page)
                cursor = self.cursor(env)
                with self.assertRaises(PinnedReadRefused) as caught:
                    cursor.read_binary()
                self.assertEqual(caught.exception.reason_code, "resource_limit")
                self.assertEqual(env.calls, [(0, 1)])
                self.assert_metrics(
                    env,
                    cursor,
                    page_work=page_work,
                    allocation=allocation,
                    wire_bytes=1 if wire else 9,
                    wire_work=wire_work,
                    copied=copied,
                    strings=0 if wire else 8,
                    read_bytes=1,
                    position=1,
                    payload=b"\x08",
                )

    def test_later_binary_failures_preserve_exact_cross_ledger_prefixes(self) -> None:
        cases = (
            ({"max_copied_payload_bytes": 17}, {}, {}, 4, 10, 10, 9, 1, [(0, 1)]),
            ({}, {"max_allocation_bytes": 26}, {}, 4, 10, 19, 18, 1, [(0, 1)]),
            ({}, {"max_work_units": 5}, {}, 5, 27, 19, 18, 9, [(0, 1), (1, 8)]),
            ({}, {}, {"max_total_read_bytes": 8}, 4, 10, 10, 9, 1, [(0, 1)]),
        )
        for wire, page, read, page_work, allocation, wire_work, copied, read_bytes, calls in cases:
            with self.subTest(wire=wire, page=page, read=read):
                env = self.environment(b"\x08ABCDEFGH", wire=wire, page=page, read=read)
                cursor = self.cursor(env)
                with self.assertRaises(PinnedReadRefused) as caught:
                    cursor.read_binary()
                self.assertEqual(caught.exception.reason_code, "resource_limit")
                self.assertEqual(env.calls, calls)
                self.assert_metrics(
                    env,
                    cursor,
                    page_work=page_work,
                    allocation=allocation,
                    wire_bytes=9,
                    wire_work=wire_work,
                    copied=copied,
                    strings=8,
                    read_bytes=read_bytes,
                    position=1,
                    payload=b"\x08",
                )

    def test_each_single_byte_deadline_checkpoint_has_a_literal_retained_prefix(self) -> None:
        expected = (
            (0, 0, 0, 1, 0, [], b""),
            (1, 0, 0, 1, 0, [], b""),
            (1, 0, 0, 2, 1, [], b""),
            (2, 2, 0, 2, 1, [(0, 1)], b""),
            (2, 2, 1, 2, 1, [(0, 1)], b""),
            (2, 2, 1, 2, 1, [(0, 1)], b""),
            (3, 2, 1, 2, 1, [(0, 1)], b"x"),
        )
        for fail_at, (
            work,
            allocation,
            read_bytes,
            wire_work,
            copied,
            requests,
            payload,
        ) in enumerate(expected, 1):
            with self.subTest(fail_at=fail_at):
                calls = [0]

                def clock(failure: int = fail_at, count: list[int] = calls) -> int:
                    count[0] += 1
                    return 10 if count[0] == failure else 0

                env = self.environment(b"x", clock=clock)
                cursor = self.cursor(env)
                with self.assertRaises(PinnedReadRefused) as caught:
                    cursor.read_byte()
                self.assertEqual(caught.exception.reason_code, "deadline_exceeded")
                self.assertEqual(calls[0], fail_at)
                self.assertEqual(env.calls, requests)
                self.assert_metrics(
                    env,
                    cursor,
                    page_work=work,
                    allocation=allocation,
                    wire_bytes=1,
                    wire_work=wire_work,
                    copied=copied,
                    read_bytes=read_bytes,
                    payload=payload,
                )

    def test_callback_exceptions_propagate_unchanged_without_retry_or_refund(self) -> None:
        for partial in (0, 2):
            with self.subTest(partial=partial):
                env = self.environment(b"12345678")
                sentinel = RuntimeError("neutral supplier exception")

                def raising_fetch(
                    offset: int,
                    length: int,
                    env: SimpleNamespace = env,
                    partial: int = partial,
                    sentinel: RuntimeError = sentinel,
                ) -> bytes:
                    env.calls.append((offset, length))
                    self.assertEqual(env.read.before_read(length), length)
                    if partial:
                        env.read.record_read(partial)
                    raise sentinel

                env.fetch = raising_fetch
                cursor = self.cursor(env)
                with self.assertRaises(RuntimeError) as caught:
                    cursor.read_double()
                self.assertIs(caught.exception, sentinel)
                self.assertEqual(env.calls, [(0, 8)])
                self.assert_metrics(
                    env,
                    cursor,
                    page_work=2,
                    allocation=16,
                    wire_bytes=8,
                    wire_work=16,
                    copied=8,
                    read_bytes=partial,
                )

    def test_wrong_callback_type_or_length_is_invalid_contract_without_retry(self) -> None:
        for returned in (None, bytearray(b"12345678"), b"1234567", b"123456789"):
            with self.subTest(
                return_type=type(returned).__name__,
                returned_length=None if returned is None else len(returned),
            ):
                env = self.environment(b"12345678")

                def broken_fetch(
                    offset: int,
                    length: int,
                    env: SimpleNamespace = env,
                    returned: object = returned,
                ) -> object:
                    env.calls.append((offset, length))
                    self.assertEqual(env.read.before_read(length), length)
                    if isinstance(returned, (bytes, bytearray)):
                        env.read.record_read(len(returned))
                    return returned

                env.fetch = broken_fetch
                cursor = self.cursor(env)
                with self.assertRaises(PinnedReadRefused) as caught:
                    cursor.read_double()
                self.assertEqual(caught.exception.reason_code, "invalid_contract")
                self.assertEqual(env.calls, [(0, 8)])
                self.assert_metrics(
                    env,
                    cursor,
                    page_work=2,
                    allocation=16,
                    wire_bytes=8,
                    wire_work=16,
                    copied=8,
                    read_bytes=0 if returned is None else len(returned),
                )

    def test_all_exact_double_allowances_are_sufficient_without_hidden_charge(self) -> None:
        data = b"\x00\x00\x00\x00\x00\x00\xf0\x3f"
        env = self.environment(
            data,
            wire={
                "max_input_bytes": 8,
                "max_object_bytes": 8,
                "max_header_bytes": 8,
                "max_total_wire_bytes": 8,
                "max_work_units": 16,
                "max_copied_payload_bytes": 8,
            },
            page={"max_work_units": 3, "max_allocation_bytes": 16},
            read={"max_total_read_bytes": 8},
        )
        cursor = self.cursor(env)
        self.assertEqual(cursor.read_double(), 1.0)
        self.assert_metrics(
            env,
            cursor,
            page_work=3,
            allocation=16,
            wire_bytes=8,
            wire_work=16,
            copied=8,
            read_bytes=8,
            position=8,
            payload=data,
        )

    def test_binary_copy_exact_string_copy_allocation_and_read_allowances_are_sufficient(
        self,
    ) -> None:
        data = b"\x08ABCDEFGH"
        env = self.environment(
            data,
            wire={
                "max_input_bytes": 9,
                "max_object_bytes": 9,
                "max_header_bytes": 9,
                "max_string_bytes": 8,
                "max_total_string_bytes": 8,
                "max_total_wire_bytes": 9,
                "max_work_units": 19,
                "max_copied_payload_bytes": 18,
            },
            page={"max_work_units": 6, "max_allocation_bytes": 27},
            read={"max_total_read_bytes": 9},
        )
        cursor = self.cursor(env)
        self.assertEqual(cursor.read_binary(), b"ABCDEFGH")
        self.assert_metrics(
            env,
            cursor,
            page_work=6,
            allocation=27,
            wire_bytes=9,
            wire_work=19,
            copied=18,
            strings=8,
            read_bytes=9,
            position=9,
            payload=data,
        )

    def test_repeated_consumes_use_cumulative_prefix_copy_budget(self) -> None:
        env = self.environment(
            b"xy",
            wire={"max_total_wire_bytes": 2, "max_work_units": 5, "max_copied_payload_bytes": 3},
            page={"max_work_units": 6, "max_allocation_bytes": 5},
            read={"max_total_read_bytes": 2},
        )
        cursor = self.cursor(env)
        self.assertEqual((cursor.read_byte(), cursor.read_byte()), (120, 121))
        self.assertEqual(env.calls, [(0, 1), (1, 1)])
        self.assert_metrics(
            env,
            cursor,
            page_work=6,
            allocation=5,
            wire_bytes=2,
            wire_work=5,
            copied=3,
            read_bytes=2,
            position=2,
            payload=b"xy",
        )
        with self.assertRaises(bounded_parquet._CompactWireMalformed) as caught:
            cursor.read_byte()
        self.assertEqual(caught.exception.reason_code, "truncated_wire")
        self.assertEqual(env.calls, [(0, 1), (1, 1)])
        self.assert_metrics(
            env,
            cursor,
            page_work=6,
            allocation=5,
            wire_bytes=2,
            wire_work=5,
            copied=3,
            read_bytes=2,
            position=2,
            payload=b"xy",
        )

    def test_later_segment_allocation_refusal_retains_loaded_prefix_without_advancing_demand(
        self,
    ) -> None:
        data = b"\x81\x80\x40" + b"Z" * 1048577
        env = self.environment(data, page={"max_allocation_bytes": 3145744})
        cursor = self.cursor(env)
        with self.assertRaises(PinnedReadRefused) as caught:
            cursor.read_binary_span()
        self.assertEqual(caught.exception.reason_code, "resource_limit")
        self.assertEqual(env.calls, [(0, 1), (1, 1), (2, 1), (3, 1048576)])
        self.assertEqual(cursor.position, 3)
        self.assertEqual(len(cursor.payload), 1048579)
        self.assertEqual(cursor.payload[:3], b"\x81\x80\x40")
        self.assertEqual(cursor.payload[-1:], b"Z")
        expected_page = dict.fromkeys(PageBudgetTests.axes, 0)
        expected_page.update(work_units=12, allocation_bytes=2097164)
        self.assertEqual(
            {axis: getattr(env.page, axis) for axis in PageBudgetTests.axes}, expected_page
        )
        self.assertEqual(
            {axis: getattr(env.wire, axis) for axis in self.wire_axes},
            {
                "total_wire_bytes": 1048580,
                "work_units": 3145745,
                "total_fields": 0,
                "total_items": 0,
                "total_string_bytes": 1048577,
                "copied_payload_bytes": 2097165,
            },
        )
        self.assertEqual(env.read.source_opens, 0)
        self.assertEqual(env.read.total_read_bytes, 1048579)


class LiveSourceHeaderTests(unittest.TestCase):
    metadata_flags = (
        "root_repetition_present",
        "row_group_ordinal_interpretation_unresolved",
        "unresolved_logical_compatibility",
        "arrow_schema_metadata_present",
        "arrow_extension_metadata_present",
        "opaque_metadata_present",
        "page_profile_unverified",
        "native_conversion_unproved",
        "source_binding_unproved",
        "utc_value_gate_unproved",
    )
    blocked_flags = (
        "primitive_child_count_present",
        "unresolved_schema_tree",
        "ambiguous_projection_name",
        "external_column_reference",
        "plaintext_column_metadata_absent",
        "encrypted_column_metadata_present",
        "column_encryption_present",
        "footer_encryption_present",
        "footer_signing_metadata_present",
        "unbounded_bloom_extent",
        "legacy_index_page_reference",
        "unrecognized_file_version",
        "unresolved_extent_interpretation",
    )

    class Ranges(dict[tuple[int, int], bytes]):
        def __init__(self, payload: bytes) -> None:
            super().__init__()
            self.payload = payload

        def __missing__(self, key: tuple[int, int]) -> bytes:
            offset, length = key
            return self.payload[offset : offset + length]

    def api(self):
        names = (
            "_LiveFooterAcquisition",
            "_OwnedPageHeaderAdmission",
            "_read_footer_from_source",
            "_footer_envelope_from_acquisition",
            "_admit_owned_initial_page_header",
        )
        values = {name: getattr(bounded_parquet, name, None) for name in names}
        for name, value in values.items():
            self.assertTrue(callable(value), f"missing owner interface {name}")
        return SimpleNamespace(**values)

    def io_for(self, payload: bytes) -> _EnvelopeLoggedIO:
        io = _EnvelopeLoggedIO()
        io.payload = payload
        io.digest = hashlib.sha256(payload).hexdigest()
        io.before = io.metadata()
        io.after = io.metadata()
        io.located = io.metadata()
        io.responses = self.Ranges(payload)
        return io

    def budgets(self, io, *, read=None, page=None):
        read_budget = FooterEnvelopeTests().budget(io, **(read or {}))
        wire = FooterGrammarTests().wire_budget(
            max_copied_payload_bytes=65536, max_header_bytes=512
        )
        footer = bounded_parquet._FooterBudget(
            wire_budget=wire, limits=bounded_parquet._FooterLimits(**_GRAMMAR_LIMITS)
        )
        limits = dict.fromkeys(("max_" + axis for axis in PageBudgetTests.axes), 1 << 24)
        limits.update(page or {})
        pages = bounded_parquet._PageBudget(
            read_budget=read_budget,
            wire_budget=wire,
            limits=bounded_parquet._PageLimits(**limits),
        )
        return SimpleNamespace(read=read_budget, wire=wire, footer=footer, page=pages)

    def fixture(self, flags=(), *, other_flags=(), dictionary=False, kind=None):
        f = FooterProfileTests()
        header_api = PageHeaderAdmissionTests()
        actual_kind = 2 if dictionary else 0
        if kind is not None:
            actual_kind = kind
        detail = (
            ((1, "i32", _grammar_integer(2)), (2, "i32", _grammar_integer(0)))
            if actual_kind == 2
            else tuple((n, "i32", _grammar_integer(v)) for n, v in ((1, 3), (2, 0), (3, 3), (4, 3)))
        )
        header = header_api.wire(actual_kind, detail=detail, compressed=7, uncompressed=7)
        data_header = header_api.wire(
            0,
            detail=tuple(
                (n, "i32", _grammar_integer(v)) for n, v in ((1, 3), (2, 0), (3, 3), (4, 3))
            ),
            compressed=7,
            uncompressed=7,
        )
        segment = header + b"\x00" * 7
        if dictionary:
            segment += data_header + b"\x00" * 7
        root_extra = []
        group_extra = []
        version = 1
        columns = [(b"n", tuple(flags))]
        if other_flags:
            columns.append((b"m", tuple(other_flags)))
        if "ambiguous_projection_name" in flags:
            columns.append((b"n", ()))
        leaves = []
        chunks = []
        body = b"PAR1" + b"\x00" * 60
        for name, column_flags in columns:
            origin = len(body)
            node_changes = {}
            meta_extra = []
            chunk_extra = []
            external = None
            data_origin = origin + len(header) + 7 if dictionary else origin
            missing_metadata = False
            for flag in column_flags:
                if flag == "root_repetition_present":
                    pass
                elif flag in ("primitive_child_count_present", "unresolved_schema_tree"):
                    node_changes["children"] = 0
                elif flag == "row_group_ordinal_interpretation_unresolved":
                    group_extra.append((7, "i16", _grammar_integer(-1)))
                elif flag == "unresolved_logical_compatibility":
                    node_changes.update(logical=f.logical(11), converted=9)
                elif flag == "external_column_reference":
                    external = b"neutral-external.parquet"
                elif flag == "plaintext_column_metadata_absent":
                    missing_metadata = True
                elif flag == "encrypted_column_metadata_present":
                    chunk_extra.append((9, "binary", f.binary(b"opaque-marker")))
                elif flag == "column_encryption_present":
                    chunk_extra.append((8, "struct", f.structure(((1, "struct", b"\x00"),))))
                elif flag == "footer_encryption_present":
                    root_extra.append((8, "struct", f.structure(((1, "struct", b"\x00"),))))
                elif flag == "footer_signing_metadata_present":
                    root_extra.append((9, "binary", f.binary(b"opaque-marker")))
                elif flag in (
                    "arrow_schema_metadata_present",
                    "arrow_extension_metadata_present",
                    "opaque_metadata_present",
                ):
                    key = {
                        "arrow_schema_metadata_present": b"ARROW:schema",
                        "arrow_extension_metadata_present": b"ARROW:extension:name",
                        "opaque_metadata_present": b"neutral-key",
                    }[flag]
                    entry = f.kv(key, b"opaque-marker")
                    root_extra.append((5, "list:struct", _grammar_list("struct", (entry,))))
                elif flag == "unbounded_bloom_extent":
                    meta_extra.append((14, "i64", _grammar_integer(origin)))
                elif flag == "legacy_index_page_reference":
                    meta_extra.append((10, "i64", _grammar_integer(origin)))
                elif flag == "unrecognized_file_version":
                    version = 3
                elif flag == "unresolved_extent_interpretation":
                    meta_extra.append((11, "i64", _grammar_integer(0)))
                elif flag not in (
                    "unresolved_schema_tree",
                    "ambiguous_projection_name",
                    *self.metadata_flags,
                ):
                    raise AssertionError(f"undeclared neutral flag witness {flag}")
            if dictionary:
                meta_extra.append((11, "i64", _grammar_integer(origin)))
            leaves.append(f.node(name, **node_changes))
            metadata = (
                None
                if missing_metadata
                else f.meta(
                    path=(name,),
                    values=3,
                    data=data_origin,
                    compressed=len(segment),
                    uncompressed=len(segment),
                    encodings=(0, 3),
                    extra=tuple(meta_extra),
                )
            )
            chunks.append(f.chunk(metadata, external=external, extra=tuple(chunk_extra)))
            body += segment
        count = len(leaves)
        root = (
            f.node(b"root", physical=None, repetition=0, children=count)
            if "root_repetition_present" in flags
            else f.root(count)
        )
        footer = f.payload(
            (root, *leaves),
            (
                f.group(
                    tuple(chunks), rows=3, size=len(segment) * len(chunks), extra=tuple(group_extra)
                ),
            ),
            rows=3,
            version=version,
            extra=tuple(root_extra),
        )
        payload = body + footer + len(footer).to_bytes(4, "little") + b"PAR1"
        return SimpleNamespace(
            payload=payload,
            footer=footer,
            header=header,
            origin=64,
            body_size=7,
            footer_start=len(body),
            dictionary=dictionary,
            expected_flags=tuple(flags) + tuple(other_flags),
        )

    def acquire(self, io, source, ledgers, before, *, integrated=True):
        api = self.api()
        result = api._read_footer_from_source(
            source,
            max_footer_bytes=2 * 1024 * 1024,
            max_retained_payload_bytes=2 * 1024 * 1024,
            budget=ledgers.read,
            receipt_before=before,
            page_budget=ledgers.page if integrated else None,
        )
        self.assertIsInstance(result, api._LiveFooterAcquisition)
        return result

    def composition(self, fixture=None):
        from contextlib import contextmanager

        @contextmanager
        def opened():
            case = self.fixture() if fixture is None else fixture
            _, witness_profile = FooterProfileTests().complete(case.footer)
            for flag in case.expected_flags:
                self.assertIn(
                    flag, witness_profile.flags, f"neutral declaration witness missing: {flag}"
                )
            with self.io_for(case.payload) as io:
                ledgers = self.budgets(io)
                before = (ledgers.read.source_opens, ledgers.read.total_read_bytes)
                with FooterEnvelopeTests().owner(io, ledgers.read) as source:
                    acquisition = self.acquire(io, source, ledgers, before)
                    envelope = self.api()._footer_envelope_from_acquisition(acquisition)
                    self.assertIsInstance(envelope, bounded_parquet._FooterEnvelope)
                    profile = bounded_parquet._profile_footer_envelope(
                        envelope, budget=ledgers.footer
                    )
                    yield SimpleNamespace(
                        io=io,
                        case=case,
                        ledgers=ledgers,
                        source=source,
                        acquisition=acquisition,
                        envelope=envelope,
                        profile=profile,
                    )
            self.assertCountEqual(io.close_attempts, io.acquired)

        return opened()

    def admit(self, env, **changes):
        values = {"row_group_index": 0, "column_index": 0, "budget": env.ledgers.page}
        values.update(changes)
        return self.api()._admit_owned_initial_page_header(
            env.acquisition, env.envelope, env.profile, **values
        )

    def policy(self, code, action, *args, **kwargs):
        with self.assertRaises(PinnedReadRefused) as caught:
            action(*args, **kwargs)
        self.assertEqual(caught.exception.reason_code, code)

    def test_actual_owner_footer_and_header_share_one_source_and_exact_ranges(self):
        with self.composition() as env:
            owned = self.admit(env)
            self.assertIsInstance(owned, self.api()._OwnedPageHeaderAdmission)
            self.assertIs(owned.acquisition, env.acquisition)
            self.assertIs(owned.profile, env.profile)
            self.assertIs(owned.chunk, env.profile.chunks[0])
            self.assertIs(owned.leaf, env.profile.leaves[0])
            self.assertEqual(owned.page_origin, 64)
            h = len(env.case.header)
            self.assertEqual(owned.absolute_header_span, (64, 64 + h))
            self.assertEqual(owned.absolute_body_span, (64 + h, 64 + h + 7))
            requests = [(offset, length) for _, length, offset in env.io.positional_requests]
            size = len(env.case.payload)
            self.assertEqual(
                requests[:3], [(0, 4), (size - 8, 8), (env.case.footer_start, len(env.case.footer))]
            )
            self.assertEqual(requests[3:], [(64 + index, 1) for index in range(h)])
            self.assertEqual(env.ledgers.read.total_read_bytes, 12 + len(env.case.footer) + h)
            self.assertEqual(
                env.ledgers.page.allocation_bytes, 12 + len(env.case.footer) + h + h * (h + 1) // 2
            )
            self.assertEqual(env.ledgers.read.source_opens, len(env.io.open_paths))
            self.assertFalse(env.envelope.receipt.whole_source_digest_verified)
            self.assertFalse(env.profile.read_eligible)
            self.assertFalse(owned.admission.decode_eligible)

    def test_compatibility_wrapper_delegates_without_early_result_construction(self):
        api = self.api()
        case = self.fixture()
        with self.io_for(case.payload) as io:
            ledgers = self.budgets(io)
            ledgers.read.source_opens = 2
            ledgers.read.total_read_bytes = 7
            events = []
            original_extent = bounded_parquet._FooterExtent
            original_receipt = bounded_parquet._FooterEnvelopeReceipt
            original_envelope = bounded_parquet._FooterEnvelope

            def construct(factory, *args, **kwargs):
                self.assertCountEqual(io.close_attempts, io.acquired)
                events.append(factory.__name__)
                return factory(*args, **kwargs)

            with (
                patch.object(
                    bounded_parquet, "_read_footer_from_source", wraps=api._read_footer_from_source
                ) as acquire,
                patch.object(
                    bounded_parquet,
                    "_FooterExtent",
                    side_effect=lambda *a, **k: construct(original_extent, *a, **k),
                ),
                patch.object(
                    bounded_parquet,
                    "_FooterEnvelopeReceipt",
                    side_effect=lambda *a, **k: construct(original_receipt, *a, **k),
                ),
                patch.object(
                    bounded_parquet,
                    "_FooterEnvelope",
                    side_effect=lambda *a, **k: construct(original_envelope, *a, **k),
                ),
            ):
                envelope = FooterEnvelopeTests().envelope(io, budget=ledgers.read)
            acquire.assert_called_once()
            self.assertEqual(acquire.call_args.kwargs["receipt_before"], (2, 7))
            self.assertIsNone(acquire.call_args.kwargs.get("page_budget"))
            self.assertEqual(events, ["_FooterExtent", "_FooterEnvelopeReceipt", "_FooterEnvelope"])
            self.assertEqual(envelope.receipt.total_read_bytes_before, 7)
            self.assertEqual(envelope.receipt.total_read_bytes_after, 19 + len(case.footer))
            self.assertEqual(ledgers.page.allocation_bytes, 0)

    def test_outer_ownership_and_scalar_mismatches_precede_clock_or_header_read(self):
        with self.composition() as env:
            api = self.api()
            other = self.budgets(env.io)
            bad_tokens = (
                replace(env.acquisition, read_budget=other.read),
                replace(env.acquisition, page_budget=other.page),
                replace(env.acquisition, wire_budget=other.wire),
                replace(env.acquisition, source_size=env.acquisition.source_size + 1),
                replace(env.acquisition, source_identity=(9, 9, 9, 9, 9)),
            )
            for token in bad_tokens:
                with self.subTest(token=repr(type(token))):
                    before = (
                        len(env.io.events),
                        len(env.io.positional_requests),
                        env.ledgers.page.work_units,
                    )
                    self.policy(
                        "invalid_contract",
                        api._admit_owned_initial_page_header,
                        token,
                        env.envelope,
                        env.profile,
                        row_group_index=0,
                        column_index=0,
                        budget=env.ledgers.page,
                    )
                    self.assertEqual(
                        before,
                        (
                            len(env.io.events),
                            len(env.io.positional_requests),
                            env.ledgers.page.work_units,
                        ),
                    )
            for value in (True, -1, "0", None):
                with self.subTest(index=value):
                    self.policy("invalid_contract", self.admit, env, row_group_index=value)

    def test_detached_payload_extent_document_and_cloned_budget_are_not_binding(self):
        with self.composition() as env:
            api = self.api()
            other_payload = bytes(bytearray(env.envelope.payload))
            self.assertIsNot(other_payload, env.envelope.payload)
            invalid = (
                (replace(env.envelope, payload=other_payload), env.profile, env.ledgers.page),
                (
                    replace(env.envelope, extent=replace(env.envelope.extent)),
                    env.profile,
                    env.ledgers.page,
                ),
                (
                    env.envelope,
                    replace(
                        env.profile, document=replace(env.profile.document, payload=other_payload)
                    ),
                    env.ledgers.page,
                ),
                (env.envelope, env.profile, self.budgets(env.io).page),
            )
            for envelope, profile, pages in invalid:
                with self.subTest(case=len(env.io.events)):
                    before = (len(env.io.events), env.ledgers.page.work_units)
                    self.policy(
                        "invalid_contract",
                        api._admit_owned_initial_page_header,
                        env.acquisition,
                        envelope,
                        profile,
                        row_group_index=0,
                        column_index=0,
                        budget=pages,
                    )
                    self.assertEqual(before, (len(env.io.events), env.ledgers.page.work_units))

    def test_literal_footer_reservations_and_receipt_prefix_are_exact(self):
        payload = b"PAR1" + b"\x00" * 16 + b"\x00" * 4 + (4).to_bytes(4, "little") + b"PAR1"
        with self.io_for(payload) as io:
            ledgers = self.budgets(
                io,
                read={"max_total_read_bytes": 23},
                page={"max_allocation_bytes": 21, "max_work_units": 12},
            )
            ledgers.read.source_opens = 2
            ledgers.read.total_read_bytes = 7
            ledgers.page.allocation_bytes = 5
            ledgers.page.work_units = 9
            with FooterEnvelopeTests().owner(io, ledgers.read) as source:
                clocks = io.events.count("clock")
                token = self.acquire(io, source, ledgers, (2, 7))
                self.assertEqual(io.events.count("clock") - clocks, 9)
                self.assertEqual(token.receipt_before, (2, 7))
                self.assertEqual(token.receipt_after, (6, 23))
                envelope = self.api()._footer_envelope_from_acquisition(token)
                self.assertIsInstance(envelope, bounded_parquet._FooterEnvelope)
                self.assertEqual(envelope.receipt.ranges, ((0, 4), (24, 32), (20, 24)))
                self.assertEqual(
                    (
                        ledgers.read.total_read_bytes,
                        ledgers.page.allocation_bytes,
                        ledgers.page.work_units,
                    ),
                    (23, 21, 12),
                )

    def test_literal_one_below_limits_keep_only_the_admitted_prefix(self):
        payload = b"PAR1" + b"\x00" * 20 + (4).to_bytes(4, "little") + b"PAR1"
        for axis in ("allocation", "work", "read"):
            with self.subTest(axis=axis), self.io_for(payload) as io:
                read = {"max_total_read_bytes": 22 if axis == "read" else 23}
                page = {
                    "max_allocation_bytes": 20 if axis == "allocation" else 21,
                    "max_work_units": 11 if axis == "work" else 12,
                }
                ledgers = self.budgets(io, read=read, page=page)
                ledgers.read.total_read_bytes = 7
                ledgers.page.allocation_bytes = 5
                ledgers.page.work_units = 9
                before = (0, 7)
                with FooterEnvelopeTests().owner(io, ledgers.read) as source:
                    self.policy(
                        "resource_limit",
                        self.api()._read_footer_from_source,
                        source,
                        max_footer_bytes=32,
                        max_retained_payload_bytes=32,
                        budget=ledgers.read,
                        receipt_before=before,
                        page_budget=ledgers.page,
                    )
                self.assertEqual(
                    (
                        ledgers.read.total_read_bytes,
                        ledgers.page.allocation_bytes,
                        ledgers.page.work_units,
                    ),
                    (19, 17, 11),
                )
                self.assertEqual(len(io.positional_requests), 2)

    def test_short_read_and_post_read_deadline_retain_actual_bytes_and_reservations(self):
        payload = b"PAR1" + b"\x00" * 20 + (4).to_bytes(4, "little") + b"PAR1"
        for fault, expected_read in (("short", 21), ("deadline", 23)):
            with self.subTest(fault=fault), self.io_for(payload) as io:
                ledgers = self.budgets(io)
                ledgers.read.total_read_bytes = 7
                ledgers.page.allocation_bytes = 5
                ledgers.page.work_units = 9
                if fault == "short":
                    io.responses[20, 4] = b"\x00\x00"
                else:
                    io.expire_after = "pread:3"
                error = (
                    io.module.TimestampMetadataRefused if fault == "short" else PinnedReadRefused
                )
                with (
                    self.assertRaises(error),
                    FooterEnvelopeTests().owner(io, ledgers.read) as source,
                ):
                    self.api()._read_footer_from_source(
                        source,
                        max_footer_bytes=32,
                        max_retained_payload_bytes=32,
                        budget=ledgers.read,
                        receipt_before=(0, 7),
                        page_budget=ledgers.page,
                    )
                self.assertEqual(
                    (
                        ledgers.read.total_read_bytes,
                        ledgers.page.allocation_bytes,
                        ledgers.page.work_units,
                    ),
                    (expected_read, 21, 12),
                )
                self.assertEqual(len(io.positional_requests), 3)

    def test_range_cap_refusal_retains_full_requested_footer_reservation(self):
        n = 1048577
        payload = b"PAR1" + b"\x00" * n + n.to_bytes(4, "little") + b"PAR1"
        with self.io_for(payload) as io:
            ledgers = self.budgets(io)
            with FooterEnvelopeTests().owner(io, ledgers.read) as source:
                self.policy(
                    "resource_limit",
                    self.api()._read_footer_from_source,
                    source,
                    max_footer_bytes=n,
                    max_retained_payload_bytes=n + 12,
                    budget=ledgers.read,
                    receipt_before=(0, 0),
                    page_budget=ledgers.page,
                )
            self.assertEqual(ledgers.read.total_read_bytes, 12)
            self.assertEqual((ledgers.page.allocation_bytes, ledgers.page.work_units), (n + 12, 3))
            self.assertEqual(len(io.positional_requests), 2)

    def test_final_pre_read_deadline_keeps_request_reservation_without_read(self):
        payload = b"PAR1" + b"\x00" * 20 + (4).to_bytes(4, "little") + b"PAR1"
        with self.io_for(payload) as io:
            ledgers = self.budgets(io)
            ledgers.read.total_read_bytes = 7
            ledgers.page.allocation_bytes = 5
            ledgers.page.work_units = 9
            calls = []

            def clock():
                calls.append(1)
                return 100 if len(calls) >= 8 else 1

            with (
                self.assertRaises(PinnedReadRefused) as caught,
                FooterEnvelopeTests().owner(io, ledgers.read) as source,
            ):
                ledgers.read.clock_ns = clock
                self.api()._read_footer_from_source(
                    source,
                    max_footer_bytes=32,
                    max_retained_payload_bytes=32,
                    budget=ledgers.read,
                    receipt_before=(0, 7),
                    page_budget=ledgers.page,
                )
            self.assertEqual(caught.exception.reason_code, "deadline_exceeded")
            self.assertEqual(
                (
                    ledgers.read.total_read_bytes,
                    ledgers.page.allocation_bytes,
                    ledgers.page.work_units,
                ),
                (19, 21, 12),
            )
            self.assertEqual(len(io.positional_requests), 2)

    def test_footer_os_failure_retains_request_charge_and_primary_error(self):
        payload = b"PAR1" + b"\x00" * 20 + (4).to_bytes(4, "little") + b"PAR1"
        with self.io_for(payload) as io:
            ledgers = self.budgets(io)
            ledgers.read.total_read_bytes = 7
            ledgers.page.allocation_bytes = 5
            ledgers.page.work_units = 9
            marker = OSError("neutral positional witness")
            io.fail_pread = 3
            io.pread_error = marker
            with (
                self.assertRaises(OSError) as caught,
                FooterEnvelopeTests().owner(io, ledgers.read) as source,
            ):
                self.api()._read_footer_from_source(
                    source,
                    max_footer_bytes=32,
                    max_retained_payload_bytes=32,
                    budget=ledgers.read,
                    receipt_before=(0, 7),
                    page_budget=ledgers.page,
                )
            self.assertIs(caught.exception, marker)
            self.assertEqual(
                (
                    ledgers.read.total_read_bytes,
                    ledgers.page.allocation_bytes,
                    ledgers.page.work_units,
                ),
                (19, 21, 12),
            )
            self.assertEqual(len(io.positional_requests), 3)

    def test_borrowed_helper_rejects_bad_snapshots_and_foreign_budget_before_clock(self):
        payload = b"PAR1" + b"\x00" * 20 + (4).to_bytes(4, "little") + b"PAR1"
        with self.io_for(payload) as io:
            ledgers = self.budgets(io)
            with FooterEnvelopeTests().owner(io, ledgers.read) as source:
                cases = (
                    {"receipt_before": (True, 0)},
                    {"receipt_before": (0,)},
                    {"receipt_before": (ledgers.read.source_opens, 0)},
                    {"receipt_before": (0, 1)},
                    {"budget": self.budgets(io).read},
                    {"page_budget": self.budgets(io).page},
                )
                for changes in cases:
                    with self.subTest(keys=tuple(changes)):
                        values: dict[str, object] = {
                            "max_footer_bytes": 32,
                            "max_retained_payload_bytes": 32,
                            "budget": ledgers.read,
                            "receipt_before": (0, 0),
                            "page_budget": ledgers.page,
                        }
                        values.update(changes)
                        before = (
                            len(io.events),
                            len(io.positional_requests),
                            ledgers.page.allocation_bytes,
                            ledgers.page.work_units,
                        )
                        self.policy(
                            "invalid_contract",
                            self.api()._read_footer_from_source,
                            source,
                            **values,
                        )
                        self.assertEqual(
                            before,
                            (
                                len(io.events),
                                len(io.positional_requests),
                                ledgers.page.allocation_bytes,
                                ledgers.page.work_units,
                            ),
                        )

    def test_binding_lookup_work_is_separate_from_actual_header_parser_work(self):
        with self.composition() as env:
            original = bounded_parquet._admit_page_header
            observed = []
            start = env.ledgers.page.work_units

            def actual(cursor, *, chunk_remaining_bytes, budget):
                observed.append(budget.work_units)
                result = original(
                    cursor, chunk_remaining_bytes=chunk_remaining_bytes, budget=budget
                )
                observed.append(budget.work_units)
                return result

            with patch.object(bounded_parquet, "_admit_page_header", side_effect=actual):
                owned = self.admit(env)
            self.assertIsInstance(owned, self.api()._OwnedPageHeaderAdmission)
            self.assertEqual(len(observed), 2)
            self.assertEqual(observed[0] - start, 4)
            self.assertEqual(env.ledgers.page.work_units - observed[1], 1)
            self.assertIs(env.ledgers.footer.wire_budget, env.ledgers.page.wire_budget)
            self.assertIs(env.ledgers.page.read_budget, env.ledgers.read)

    def test_all_twenty_three_capabilities_have_declaration_backed_outcomes(self):
        self.assertEqual(
            set(self.metadata_flags) | set(self.blocked_flags), FooterProfileTests.FLAGS
        )
        for flag in (*self.metadata_flags, *self.blocked_flags):
            with self.subTest(flag=flag), self.composition(self.fixture((flag,))) as env:
                self.assertIn(flag, env.profile.flags)
                before = len(env.io.positional_requests)
                if flag in self.blocked_flags:
                    self.policy("unsupported_mode", self.admit, env)
                    self.assertEqual(len(env.io.positional_requests), before)
                else:
                    owned = self.admit(env)
                    self.assertIsInstance(owned, self.api()._OwnedPageHeaderAdmission)
                    self.assertIn(flag, owned.profile.flags)
                    self.assertFalse(owned.profile.read_eligible)

    def test_unselected_column_markers_do_not_become_selected_column_policy(self):
        for flag in (
            "external_column_reference",
            "column_encryption_present",
            "encrypted_column_metadata_present",
        ):
            with (
                self.subTest(flag=flag),
                self.composition(self.fixture(other_flags=(flag,))) as env,
            ):
                self.assertIn(flag, env.profile.flags)
                owned = self.admit(env)
                self.assertIsInstance(owned, self.api()._OwnedPageHeaderAdmission)
                self.assertEqual(owned.chunk.column_index, 0)
                self.assertNotIn(flag, owned.chunk.flags | owned.leaf.flags)

    def test_refuse_wins_and_global_blockers_are_not_overridden(self):
        cases = (
            (("opaque_metadata_present", "external_column_reference"), (), False),
            (("root_repetition_present", "arrow_schema_metadata_present"), (), True),
            (("footer_signing_metadata_present",), ("column_encryption_present",), False),
            ((), ("primitive_child_count_present",), False),
        )
        for flags, other, allowed in cases:
            with (
                self.subTest(flags=flags, other=other),
                self.composition(self.fixture(flags, other_flags=other)) as env,
            ):
                if "primitive_child_count_present" in other:
                    self.assertIn("unresolved_schema_tree", env.profile.flags)
                    self.assertNotIn(
                        "primitive_child_count_present",
                        env.profile.chunks[0].flags | env.profile.leaves[0].flags,
                    )
                before = len(env.io.positional_requests)
                if allowed:
                    self.assertIsInstance(self.admit(env), self.api()._OwnedPageHeaderAdmission)
                else:
                    self.policy("unsupported_mode", self.admit, env)
                    self.assertEqual(len(env.io.positional_requests), before)

    def test_unknown_flag_and_invalid_private_flag_container_refuse(self):
        with self.composition() as env:
            api = self.api()
            for flags in (frozenset({"not-a-declared-capability"}), set(), frozenset({1})):
                with self.subTest(flags=repr(flags)):
                    profile = replace(env.profile, flags=flags)
                    self.policy(
                        "invalid_contract",
                        api._admit_owned_initial_page_header,
                        env.acquisition,
                        env.envelope,
                        profile,
                        row_group_index=0,
                        column_index=0,
                        budget=env.ledgers.page,
                    )

    def test_initial_dictionary_origin_and_actual_header_type_must_agree(self):
        with self.composition(self.fixture(dictionary=True)) as env:
            owned = self.admit(env)
            self.assertIsInstance(owned, self.api()._OwnedPageHeaderAdmission)
            self.assertEqual(owned.page_origin, 64)
            self.assertEqual(owned.admission.page_type, 2)
            self.assertTrue(
                all(
                    offset < 64 + len(env.case.header)
                    for _, _, offset in env.io.positional_requests[3:]
                )
            )
        with self.composition(self.fixture(dictionary=True, kind=0)) as env:
            with self.assertRaises(bounded_parquet._CompactWireMalformed) as caught:
                self.admit(env)
            self.assertEqual(caught.exception.reason_code, "schema_mismatch")

    def test_missing_duplicate_and_empty_chunk_requests_do_not_acquire_headers(self):
        with self.composition() as env:
            api = self.api()
            before = len(env.io.positional_requests)
            self.policy("invalid_contract", self.admit, env, column_index=4)
            duplicate = replace(env.profile, chunks=(env.profile.chunks[0], env.profile.chunks[0]))
            self.policy(
                "invalid_contract",
                api._admit_owned_initial_page_header,
                env.acquisition,
                env.envelope,
                duplicate,
                row_group_index=0,
                column_index=0,
                budget=env.ledgers.page,
            )
            empty = replace(
                env.profile.chunks[0],
                candidate_extent=(64, 64),
                compressed_size=0,
                uncompressed_size=0,
                num_values=0,
            )
            profile = replace(env.profile, chunks=(empty,))
            self.policy(
                "invalid_contract",
                api._admit_owned_initial_page_header,
                env.acquisition,
                env.envelope,
                profile,
                row_group_index=0,
                column_index=0,
                budget=env.ledgers.page,
            )
            self.assertEqual(len(env.io.positional_requests), before)

    def test_closed_acquisition_precedes_clock_and_cannot_reopen(self):
        with self.composition() as env:
            owned = self.admit(env)
            self.assertIsInstance(owned, self.api()._OwnedPageHeaderAdmission)
        before = (len(env.io.events), len(env.io.open_paths), len(env.io.positional_requests))
        self.policy("invalid_contract", self.admit, env)
        self.assertEqual(
            before, (len(env.io.events), len(env.io.open_paths), len(env.io.positional_requests))
        )

    def test_source_identity_and_cleanup_failure_prevent_envelope_construction(self):
        for fault in ("identity", "close"):
            with self.subTest(fault=fault), self.io_for(self.fixture().payload) as io:
                ledgers = self.budgets(io)
                if fault == "identity":
                    io.after = io.metadata(st_ino=99)
                else:
                    io.close_errors[14] = OSError("neutral close witness")
                original = bounded_parquet._FooterEnvelope
                with (
                    patch.object(bounded_parquet, "_FooterEnvelope", wraps=original) as constructor,
                    self.assertRaises((io.module.TimestampMetadataRefused, OSError)),
                ):
                    FooterEnvelopeTests().envelope(io, budget=ledgers.read)
                constructor.assert_not_called()
                self.assertCountEqual(io.close_attempts, io.acquired)

    def test_primary_consumer_error_retains_cleanup_error_without_retry(self):
        with self.io_for(self.fixture().payload) as io:
            ledgers = self.budgets(io)
            marker = RuntimeError("neutral primary witness")
            close_error = OSError("neutral cleanup witness")
            io.close_errors[14] = close_error
            with (
                self.assertRaises(RuntimeError) as caught,
                FooterEnvelopeTests().owner(io, ledgers.read) as source,
            ):
                self.acquire(io, source, ledgers, (0, 0))
                raise marker
            self.assertIs(caught.exception, marker)
            self.assertIsNone(marker.__cause__)
            state = BaseException.__dict__["__dict__"].__get__(marker, BaseException)
            self.assertEqual(state[io.module._PINNED_CLEANUP_ERRORS], (close_error,))
            self.assertEqual(len(io.positional_requests), 3)
            self.assertCountEqual(io.close_attempts, io.acquired)

    def manual_consumer_token(self, io, source, ledgers, before, *, integrated):
        read_range = io.module._read_pinned_range
        size = source.size

        def read(offset, length):
            if integrated:
                ledgers.page.reserve(
                    bounded_parquet._PageCharge(allocation_bytes=length, work_units=1)
                )
            return read_range(source, offset=offset, length=length)

        self.assertEqual(read(0, 4), b"PAR1")
        trailer = read(size - 8, 8)
        self.assertEqual(trailer[4:], b"PAR1")
        length = int.from_bytes(trailer[:4], "little")
        start = size - 8 - length
        payload = read(start, length)
        return self.api()._LiveFooterAcquisition(
            source=source,
            read_budget=ledgers.read,
            page_budget=ledgers.page if integrated else None,
            wire_budget=ledgers.wire if integrated else None,
            source_identity=source.identity,
            source_size=size,
            footer_start=start,
            footer_end=size - 8,
            payload=payload,
            receipt_before=before,
            receipt_after=(ledgers.read.source_opens, ledgers.read.total_read_bytes),
        )

    def manual_consumer_environment(self, fixture=None):
        from contextlib import contextmanager

        @contextmanager
        def opened():
            case = self.fixture() if fixture is None else fixture
            with self.io_for(case.payload) as io:
                ledgers = self.budgets(io)
                before = (ledgers.read.source_opens, ledgers.read.total_read_bytes)
                with FooterEnvelopeTests().owner(io, ledgers.read) as source:
                    token = self.manual_consumer_token(io, source, ledgers, before, integrated=True)
                    extent = bounded_parquet._FooterExtent(
                        token.source_size, token.footer_start, token.footer_end
                    )
                    receipt = bounded_parquet._FooterEnvelopeReceipt(
                        token.receipt_before[0],
                        token.receipt_after[0],
                        token.receipt_before[1],
                        token.receipt_after[1],
                        (
                            (0, 4),
                            (token.footer_end, token.source_size),
                            (token.footer_start, token.footer_end),
                        ),
                        False,
                    )
                    envelope = bounded_parquet._FooterEnvelope(
                        token.payload, extent, token.source_identity, receipt, ledgers.read
                    )
                    profile = bounded_parquet._profile_footer_envelope(
                        envelope, budget=ledgers.footer
                    )
                    yield SimpleNamespace(
                        io=io,
                        case=case,
                        ledgers=ledgers,
                        source=source,
                        acquisition=token,
                        envelope=envelope,
                        profile=profile,
                    )
            self.assertCountEqual(io.close_attempts, io.acquired)

        return opened()

    def test_pure_factory_receipt_tuples_cached_geometry_and_payload_refuse(self):
        case = self.fixture()
        with self.io_for(case.payload) as io:
            ledgers = self.budgets(io)
            before = (0, 0)
            with FooterEnvelopeTests().owner(io, ledgers.read) as source:
                token = self.manual_consumer_token(io, source, ledgers, before, integrated=False)
                cases = (
                    {"receipt_before": ()},
                    {"receipt_before": (True, 0)},
                    {"receipt_before": (-1, 0)},
                    {
                        "receipt_after": (
                            ledgers.read.source_opens + 1,
                            ledgers.read.total_read_bytes,
                        )
                    },
                    {
                        "receipt_after": (
                            ledgers.read.source_opens,
                            ledgers.read.total_read_bytes + 1,
                        )
                    },
                    {"receipt_after": (ledgers.read.source_opens,)},
                    {"source_size": token.source_size + 1},
                    {"source_identity": (9, 9, 9, 9, 9)},
                    {"footer_start": -1},
                    {"footer_end": token.footer_end + 1},
                    {"payload": token.payload[:-1]},
                )
                for changes in cases:
                    with self.subTest(fields=tuple(changes)):
                        bad = replace(token, **changes)
                        checkpoint = (
                            len(io.events),
                            len(io.positional_requests),
                            ledgers.read.source_opens,
                            ledgers.read.total_read_bytes,
                            ledgers.page.work_units,
                            ledgers.page.allocation_bytes,
                        )
                        self.policy(
                            "invalid_contract", self.api()._footer_envelope_from_acquisition, bad
                        )
                        self.assertEqual(
                            checkpoint,
                            (
                                len(io.events),
                                len(io.positional_requests),
                                ledgers.read.source_opens,
                                ledgers.read.total_read_bytes,
                                ledgers.page.work_units,
                                ledgers.page.allocation_bytes,
                            ),
                        )

    def test_pure_factory_current_counter_drift_is_not_a_cached_receipt(self):
        for axis in ("source_opens", "total_read_bytes"):
            with self.subTest(axis=axis), self.io_for(self.fixture().payload) as io:
                ledgers = self.budgets(io)
                with FooterEnvelopeTests().owner(io, ledgers.read) as source:
                    token = self.manual_consumer_token(
                        io, source, ledgers, (0, 0), integrated=False
                    )
                    setattr(ledgers.read, axis, getattr(ledgers.read, axis) + 1)
                    checkpoint = (
                        len(io.events),
                        len(io.positional_requests),
                        ledgers.read.source_opens,
                        ledgers.read.total_read_bytes,
                    )
                    self.policy(
                        "invalid_contract", self.api()._footer_envelope_from_acquisition, token
                    )
                    self.assertEqual(
                        checkpoint,
                        (
                            len(io.events),
                            len(io.positional_requests),
                            ledgers.read.source_opens,
                            ledgers.read.total_read_bytes,
                        ),
                    )

    def test_pure_factory_valid_post_cleanup_token_performs_no_source_operation(self):
        with self.io_for(self.fixture().payload) as io:
            ledgers = self.budgets(io)
            with FooterEnvelopeTests().owner(io, ledgers.read) as source:
                token = self.manual_consumer_token(io, source, ledgers, (0, 0), integrated=False)
            self.assertTrue(source._closed)
            checkpoint = (
                len(io.events),
                len(io.positional_requests),
                ledgers.read.source_opens,
                ledgers.read.total_read_bytes,
            )
            envelope = self.api()._footer_envelope_from_acquisition(token)
            self.assertIsInstance(envelope, bounded_parquet._FooterEnvelope)
            self.assertIs(envelope.payload, token.payload)
            self.assertIs(envelope.read_budget, ledgers.read)
            self.assertEqual(envelope.source_identity, token.source_identity)
            self.assertEqual(
                (envelope.receipt.source_opens_before, envelope.receipt.total_read_bytes_before),
                token.receipt_before,
            )
            self.assertEqual(
                (envelope.receipt.source_opens_after, envelope.receipt.total_read_bytes_after),
                token.receipt_after,
            )
            self.assertFalse(envelope.receipt.whole_source_digest_verified)
            self.assertEqual(
                checkpoint,
                (
                    len(io.events),
                    len(io.positional_requests),
                    ledgers.read.source_opens,
                    ledgers.read.total_read_bytes,
                ),
            )

    def test_initial_header_rejects_current_counter_drift_before_clock(self):
        for axis in ("source_opens", "total_read_bytes"):
            with self.subTest(axis=axis), self.manual_consumer_environment() as env:
                setattr(env.ledgers.read, axis, getattr(env.ledgers.read, axis) + 1)
                checkpoint = (
                    len(env.io.events),
                    len(env.io.positional_requests),
                    env.ledgers.page.work_units,
                    env.ledgers.page.allocation_bytes,
                )
                self.policy("invalid_contract", self.admit, env)
                self.assertEqual(
                    checkpoint,
                    (
                        len(env.io.events),
                        len(env.io.positional_requests),
                        env.ledgers.page.work_units,
                        env.ledgers.page.allocation_bytes,
                    ),
                )

    def test_multi_entry_owner_work_prefix_refusals_precede_header_acquisition(self):
        for admitted in (0, 1, 2, 3, 4, 5):
            with (
                self.subTest(admitted=admitted),
                self.manual_consumer_environment(
                    self.fixture(other_flags=("opaque_metadata_present",))
                ) as env,
            ):
                self.assertEqual((len(env.profile.chunks), len(env.profile.leaves)), (2, 2))
                pages = env.ledgers.page
                start = pages.work_units
                pages.limits = replace(pages.limits, max_work_units=start + admitted)
                checkpoint = (
                    len(env.io.positional_requests),
                    env.ledgers.read.total_read_bytes,
                    pages.allocation_bytes,
                    pages.pages,
                    pages.compressed_bytes,
                    pages.uncompressed_bytes,
                    pages.values,
                )
                with patch.object(
                    bounded_parquet,
                    "_admit_page_header",
                    side_effect=AssertionError("header parser before owner quota"),
                ):
                    self.policy("resource_limit", self.admit, env)
                self.assertEqual(pages.work_units, start + admitted)
                self.assertEqual(
                    checkpoint,
                    (
                        len(env.io.positional_requests),
                        env.ledgers.read.total_read_bytes,
                        pages.allocation_bytes,
                        pages.pages,
                        pages.compressed_bytes,
                        pages.uncompressed_bytes,
                        pages.values,
                    ),
                )

    def test_multi_entry_exact_and_one_below_result_work_preserve_real_parser_charges(self):
        case = self.fixture(other_flags=("opaque_metadata_present",))
        real_parser = bounded_parquet._admit_page_header
        resident_cursor, resident_budget = PageHeaderAdmissionTests().context(case.header)
        reference = real_parser(
            resident_cursor, chunk_remaining_bytes=len(case.header) + 7, budget=resident_budget
        )
        self.assertEqual(reference.header_end, len(case.header))
        parser_work = resident_budget.work_units + 3 * len(case.header)
        for final_units in (0, 1):
            with (
                self.subTest(final_units=final_units),
                self.manual_consumer_environment(case) as env,
            ):
                self.assertEqual((len(env.profile.chunks), len(env.profile.leaves)), (2, 2))
                pages = env.ledgers.page
                start = pages.work_units
                pages.limits = replace(
                    pages.limits, max_work_units=start + 6 + parser_work + final_units
                )
                requests = len(env.io.positional_requests)
                observed = []

                def actual(cursor, *, chunk_remaining_bytes, budget, observed=observed):
                    observed.append(budget.work_units)
                    result = real_parser(
                        cursor, chunk_remaining_bytes=chunk_remaining_bytes, budget=budget
                    )
                    observed.append(budget.work_units)
                    return result

                with patch.object(bounded_parquet, "_admit_page_header", side_effect=actual):
                    if final_units:
                        owned = self.admit(env)
                        self.assertIsInstance(owned, self.api()._OwnedPageHeaderAdmission)
                    else:
                        self.policy("resource_limit", self.admit, env)
                self.assertEqual(observed, [start + 6, start + 6 + parser_work])
                self.assertEqual(pages.work_units, start + 6 + parser_work + final_units)
                self.assertEqual(len(env.io.positional_requests) - requests, len(case.header))
                self.assertEqual(
                    (pages.pages, pages.compressed_bytes, pages.uncompressed_bytes, pages.values),
                    (1, 7, 7, 3),
                )

    def test_pure_factory_rejects_foreign_tokens_and_unpaired_compatibility_wire(self):
        with self.manual_consumer_environment() as env:
            api = self.api()
            before = (
                len(env.io.events),
                len(env.io.positional_requests),
                env.ledgers.read.total_read_bytes,
                env.ledgers.page.work_units,
                env.ledgers.page.allocation_bytes,
            )
            tokens = (None, replace(env.acquisition, page_budget=None))
            with patch.object(
                env.ledgers.read, "clock_ns", side_effect=AssertionError("pure factory clock")
            ):
                for token in tokens:
                    with self.subTest(token=type(token).__name__):
                        self.policy(
                            "invalid_contract", api._footer_envelope_from_acquisition, token
                        )
                        self.assertEqual(
                            before,
                            (
                                len(env.io.events),
                                len(env.io.positional_requests),
                                env.ledgers.read.total_read_bytes,
                                env.ledgers.page.work_units,
                                env.ledgers.page.allocation_bytes,
                            ),
                        )

    def test_footer_range_preflight_refuses_before_clock_reservation_or_io(self):
        case = self.fixture()
        read_range = bounded_parquet._live_footer_read
        with self.io_for(case.payload) as io:
            ledgers = self.budgets(io, read={"max_total_read_bytes": 1})
            with FooterEnvelopeTests().owner(io, ledgers.read) as source:
                before = (
                    len(io.events),
                    len(io.positional_requests),
                    ledgers.read.total_read_bytes,
                )
                axes = {axis: getattr(ledgers.page, axis) for axis in PageBudgetTests.axes}
                cases = (
                    (source, 0, 0, "invalid_contract"),
                    (source, source.size - 1, 2, "invalid_contract"),
                    (None, 0, 1, "invalid_contract"),
                    (source, 0, 2, "resource_limit"),
                )
                with patch.object(
                    ledgers.read, "clock_ns", side_effect=AssertionError("range preflight clock")
                ):
                    for selected, offset, length, reason in cases:
                        with self.subTest(offset=offset, length=length, reason=reason):
                            self.policy(reason, read_range, selected, offset, length, ledgers.page)
                            self.assertEqual(
                                before,
                                (
                                    len(io.events),
                                    len(io.positional_requests),
                                    ledgers.read.total_read_bytes,
                                ),
                            )
                            self.assertEqual(
                                axes,
                                {
                                    axis: getattr(ledgers.page, axis)
                                    for axis in PageBudgetTests.axes
                                },
                            )
            self.assertCountEqual(io.close_attempts, io.acquired)

    def test_metered_lookup_rejects_malformed_duplicate_and_missing_members(self):
        with self.manual_consumer_environment() as env:
            profile = env.profile
            chunk, leaf = profile.chunks[0], profile.leaves[0]
            cases = (
                ("malformed chunk", replace(profile, chunks=(None,)), 2),
                (
                    "boolean schema index",
                    replace(profile, chunks=(replace(chunk, schema_index=True),)),
                    2,
                ),
                (
                    "negative schema index",
                    replace(profile, chunks=(replace(chunk, schema_index=-1),)),
                    2,
                ),
                ("malformed leaf", replace(profile, leaves=(None,)), 3),
                (
                    "boolean leaf index",
                    replace(profile, leaves=(replace(leaf, schema_index=True),)),
                    3,
                ),
                ("duplicate leaf", replace(profile, leaves=(leaf, leaf)), 4),
                ("missing leaf", replace(profile, leaves=()), 2),
            )
            for label, candidate, work in cases:
                with self.subTest(case=label):
                    axes = {axis: getattr(env.ledgers.page, axis) for axis in PageBudgetTests.axes}
                    before = (
                        len(env.io.positional_requests),
                        env.ledgers.read.source_opens,
                        env.ledgers.read.total_read_bytes,
                    )
                    clocks = env.io.events.count("clock")
                    self.policy(
                        "invalid_contract",
                        self.api()._admit_owned_initial_page_header,
                        env.acquisition,
                        env.envelope,
                        candidate,
                        row_group_index=0,
                        column_index=0,
                        budget=env.ledgers.page,
                    )
                    self.assertEqual(
                        before,
                        (
                            len(env.io.positional_requests),
                            env.ledgers.read.source_opens,
                            env.ledgers.read.total_read_bytes,
                        ),
                    )
                    self.assertEqual(env.io.events.count("clock") - clocks, work)
                    self.assertEqual(
                        {axis: getattr(env.ledgers.page, axis) for axis in PageBudgetTests.axes},
                        axes | {"work_units": axes["work_units"] + work},
                    )

    def test_initial_extent_and_origin_refusals_preserve_the_metadata_prefix(self):
        with self.manual_consumer_environment() as env:
            profile = env.profile
            chunk = profile.chunks[0]
            origin, stop = chunk.candidate_extent
            cases = (
                ("unresolved extent", replace(chunk, candidate_extent=None), "unsupported_mode"),
                (
                    "mutable extent",
                    replace(chunk, candidate_extent=[origin, stop]),
                    "invalid_contract",
                ),
                (
                    "boolean extent",
                    replace(chunk, candidate_extent=(True, stop)),
                    "invalid_contract",
                ),
                (
                    "leading magic overlap",
                    replace(chunk, candidate_extent=(3, stop)),
                    "unsupported_mode",
                ),
                (
                    "footer overlap",
                    replace(chunk, candidate_extent=(origin, env.acquisition.footer_start + 1)),
                    "unsupported_mode",
                ),
                (
                    "dictionary origin",
                    replace(chunk, dictionary_page_offset=origin + 1),
                    "unsupported_mode",
                ),
                ("data origin", replace(chunk, data_page_offset=origin + 1), "unsupported_mode"),
            )
            for label, candidate, reason in cases:
                with self.subTest(case=label):
                    axes = {axis: getattr(env.ledgers.page, axis) for axis in PageBudgetTests.axes}
                    before = (len(env.io.positional_requests), env.ledgers.read.total_read_bytes)
                    self.policy(
                        reason,
                        self.api()._admit_owned_initial_page_header,
                        env.acquisition,
                        env.envelope,
                        replace(profile, chunks=(candidate,)),
                        row_group_index=0,
                        column_index=0,
                        budget=env.ledgers.page,
                    )
                    self.assertEqual(
                        before, (len(env.io.positional_requests), env.ledgers.read.total_read_bytes)
                    )
                    self.assertEqual(
                        {axis: getattr(env.ledgers.page, axis) for axis in PageBudgetTests.axes},
                        axes | {"work_units": axes["work_units"] + 4},
                    )

    def test_selected_declaration_refusals_precede_header_acquisition(self):
        with self.manual_consumer_environment() as env:
            profile = env.profile
            chunk, leaf = profile.chunks[0], profile.leaves[0]
            limit = env.ledgers.wire.limits.max_items_per_container
            size = len(env.envelope.payload)
            cases = (
                (
                    "missing physical type",
                    replace(chunk, physical_type=None),
                    leaf,
                    "unsupported_mode",
                ),
                ("missing codec", replace(chunk, codec=None), leaf, "unsupported_mode"),
                (
                    "encoding cardinality",
                    replace(chunk, encodings=(0,) * (limit + 1)),
                    leaf,
                    "invalid_contract",
                ),
                ("boolean encoding", replace(chunk, encodings=(False,)), leaf, "unsupported_mode"),
                ("negative encoding", replace(chunk, encodings=(-1,)), leaf, "unsupported_mode"),
                ("foreign path span", chunk, replace(leaf, path=(None,)), "unsupported_mode"),
                (
                    "out of payload span",
                    chunk,
                    replace(leaf, path=(bounded_parquet._FooterSpan(size, size + 1),)),
                    "unsupported_mode",
                ),
            )
            for label, selected_chunk, selected_leaf, reason in cases:
                with self.subTest(case=label):
                    axes = {axis: getattr(env.ledgers.page, axis) for axis in PageBudgetTests.axes}
                    before = (len(env.io.positional_requests), env.ledgers.read.total_read_bytes)
                    self.policy(
                        reason,
                        self.api()._admit_owned_initial_page_header,
                        env.acquisition,
                        env.envelope,
                        replace(profile, chunks=(selected_chunk,), leaves=(selected_leaf,)),
                        row_group_index=0,
                        column_index=0,
                        budget=env.ledgers.page,
                    )
                    self.assertEqual(
                        before, (len(env.io.positional_requests), env.ledgers.read.total_read_bytes)
                    )
                    self.assertEqual(
                        {axis: getattr(env.ledgers.page, axis) for axis in PageBudgetTests.axes},
                        axes | {"work_units": axes["work_units"] + 4},
                    )


class OwnedPageBodyTests(unittest.TestCase):
    """Opaque byte-protocol ownership; these bytes do not represent market observations."""

    def fixture(self, payload=b"\x00" * 7, *, crc=None, kind=0, codec=0):
        footer_api = FooterProfileTests()
        header_api = PageHeaderAdmissionTests()
        count = 3 if payload else 0
        if kind == 2:
            detail = ((1, "i32", _grammar_integer(2)), (2, "i32", _grammar_integer(0)))
        elif kind == 3:
            detail = tuple(
                (field, "i32", _grammar_integer(value))
                for field, value in ((1, count), (2, 0), (3, count), (4, 0), (5, 0), (6, 0))
            )
        else:
            detail = tuple(
                (field, "i32", _grammar_integer(value))
                for field, value in ((1, count), (2, 0), (3, 3), (4, 3))
            )
        header = header_api.wire(
            kind, detail=detail, compressed=len(payload), uncompressed=len(payload), crc=crc
        )
        segment = header + payload
        data_origin = 64
        extra = ()
        if kind == 2:
            data_origin += len(segment)
            extra = ((11, "i64", _grammar_integer(64)),)
            segment += header_api.wire(0, compressed=7, uncompressed=7) + b"\x00" * 7
        metadata = footer_api.meta(
            path=(b"n",),
            values=count,
            data=data_origin,
            compressed=len(segment),
            uncompressed=len(segment),
            encodings=(0, 3),
            codec=codec,
            extra=extra,
        )
        footer = footer_api.payload(
            (footer_api.root(1), footer_api.node(b"n")),
            (footer_api.group((footer_api.chunk(metadata),), rows=count, size=len(segment)),),
            rows=count,
        )
        prefix = b"PAR1" + b"\x00" * 60 + segment
        return SimpleNamespace(
            payload=prefix + footer + len(footer).to_bytes(4, "little") + b"PAR1",
            footer=footer,
            header=header,
            origin=64,
            body_size=len(payload),
            body=payload,
            footer_start=len(prefix),
            dictionary=kind == 2,
            expected_flags=(),
        )

    def environment(self: OwnedPageBodyTests | SequentialOwnedPageTests, *args, **kwargs):
        from contextlib import contextmanager

        @contextmanager
        def opened():
            case = self.fixture(*args, **kwargs)
            helper = LiveSourceHeaderTests()
            with helper.io_for(case.payload) as io:
                ledgers = helper.budgets(io)
                before = (ledgers.read.source_opens, ledgers.read.total_read_bytes)
                with FooterEnvelopeTests().owner(io, ledgers.read) as source:
                    acquisition = helper.acquire(io, source, ledgers, before)
                    envelope = helper.api()._footer_envelope_from_acquisition(acquisition)
                    profile = bounded_parquet._profile_footer_envelope(
                        envelope, budget=ledgers.footer
                    )
                    yield SimpleNamespace(
                        io=io,
                        case=case,
                        ledgers=ledgers,
                        source=source,
                        acquisition=acquisition,
                        envelope=envelope,
                        profile=profile,
                    )
            self.assertCountEqual(io.close_attempts, io.acquired)

        return opened()

    def admit(self, env):
        return LiveSourceHeaderTests().admit(env)

    def body(self, env, header, *, budget=None):
        return bounded_parquet._read_owned_initial_page_body(
            header, budget=env.ledgers.page if budget is None else budget
        )

    def counters(self: OwnedPageBodyTests | SequentialOwnedPageTests, env):
        return SimpleNamespace(
            read=(env.ledgers.read.source_opens, env.ledgers.read.total_read_bytes),
            closes=tuple(env.io.close_attempts),
            page={axis: getattr(env.ledgers.page, axis) for axis in PageBudgetTests.axes},
            wire=tuple(
                getattr(env.ledgers.wire, axis)
                for axis in (
                    "total_wire_bytes",
                    "work_units",
                    "total_fields",
                    "total_items",
                    "total_string_bytes",
                    "copied_payload_bytes",
                )
            ),
            requests=tuple(env.io.positional_requests),
            events=tuple(env.io.events),
        )

    def assert_quiet_refusal(self, env, header, *, budget=None):
        before = self.counters(env)
        with self.assertRaises(PinnedReadRefused) as caught:
            self.body(env, header, budget=budget)
        self.assertEqual(caught.exception.reason_code, "invalid_contract")
        self.assertEqual(self.counters(env).__dict__, before.__dict__)

    def assert_body_delta(self, env, before, *, size, work):
        after = self.counters(env)
        self.assertEqual(after.read, (before.read[0], before.read[1] + size))
        self.assertEqual(after.wire, before.wire)
        self.assertEqual(
            after.page,
            before.page
            | {
                "allocation_bytes": before.page["allocation_bytes"] + 2 * env.case.body_size,
                "work_units": before.page["work_units"] + work,
            },
        )

    def test_exact_body_ranges_and_ledgers_for_initial_header_variants(self):
        for kind, codec in ((0, 0), (2, 0), (3, 0), (0, 1)):
            with (
                self.subTest(kind=kind, codec=codec),
                self.environment(kind=kind, codec=codec) as env,
            ):
                header = self.admit(env)
                self.assertFalse(header.admission.decode_eligible)
                self.assertEqual(header.chunk.codec, codec)
                before = self.counters(env)
                self.assertEqual(
                    before.read,
                    (len(env.io.acquired), len(env.case.footer) + 12 + len(env.case.header)),
                )
                self.assertEqual(header.body_entry_work_units, before.page["work_units"])
                result = self.body(env, header)
                self.assertIsInstance(result, bounded_parquet._OwnedPageBody)
                self.assertIs(result.header, header)
                self.assertEqual(result.payload, b"\x00" * 7)
                self.assertFalse(result.crc_verified)
                self.assertEqual(result.receipt_after, (before.read[0], before.read[1] + 7))
                start = 64 + len(env.case.header)
                self.assertEqual(header.absolute_body_span, (start, start + 7))
                self.assertEqual(
                    [request[1:] for request in env.io.positional_requests[len(before.requests) :]],
                    [(7, start)],
                )
                self.assertEqual(tuple(env.io.close_attempts), before.closes)
                self.assertNotIn(env.source.borrow_fd(), env.io.close_attempts)
                self.assertFalse(env.source._closed)
                self.assertFalse(env.profile.read_eligible)
                self.assertFalse(header.admission.decode_eligible)
                self.assertFalse(env.envelope.receipt.whole_source_digest_verified)
                self.assert_body_delta(env, before, size=7, work=6)
                with self.assertRaises(FrozenInstanceError):
                    result.payload = b"replacement"
                self.assert_quiet_refusal(env, header)

    def test_crc_absence_empty_zero_and_signed_known_vector(self):
        vectors = ((b"", None, False, 5), (b"", 0, True, 5), (b"123456789", -873187034, True, 87))
        for payload, crc, verified, work in vectors:
            with self.subTest(payload=payload, crc=crc), self.environment(payload, crc=crc) as env:
                header = self.admit(env)
                before = self.counters(env)
                result = self.body(env, header)
                self.assertEqual(result.payload, payload)
                self.assertIs(result.crc_verified, verified)
                self.assert_body_delta(env, before, size=len(payload), work=work)
                if not payload:
                    self.assertEqual(tuple(env.io.positional_requests), before.requests)
                self.assert_quiet_refusal(env, header)

    def test_crc_mismatch_retains_acquisition_and_scalar_work_without_result(self):
        with self.environment(b"123456789", crc=0) as env:
            header = self.admit(env)
            before = self.counters(env)
            with self.assertRaises(bounded_parquet._CompactWireMalformed) as caught:
                self.body(env, header)
            self.assertEqual(caught.exception.reason_code, "invalid_wire")
            self.assert_body_delta(env, before, size=9, work=86)
            self.assert_quiet_refusal(env, header)

    def test_stage_and_geometry_drift_refuse_before_clock_or_io(self):
        faults = (
            "read_bytes",
            "opens",
            "work",
            "wire",
            "closed",
            "origin",
            "span",
            "stamp",
            "source",
            "identity",
            "read_budget",
            "profile",
            "leaf",
        )
        for fault in faults:
            with self.subTest(fault=fault), self.environment() as env:
                header = self.admit(env)
                undo = None
                if fault == "read_bytes":
                    env.io.module._read_pinned_range(env.source, offset=0, length=1)
                elif fault == "opens":
                    env.ledgers.read.source_opens += 1
                elif fault == "work":
                    env.ledgers.page.reserve(bounded_parquet._PageCharge(work_units=1))
                elif fault == "wire":
                    original = env.ledgers.page.wire_budget
                    env.ledgers.page.wire_budget = FooterGrammarTests().wire_budget()

                    def undo(env=env, original=original):
                        env.ledgers.page.wire_budget = original
                elif fault == "closed":
                    env.source._closed = True

                    def undo(env=env):
                        env.source._closed = False
                elif fault == "origin":
                    header = replace(header, page_origin=header.page_origin + 1)
                elif fault == "span":
                    header = replace(
                        header,
                        admission=replace(header.admission, body_end=header.admission.body_end + 1),
                    )
                elif fault == "stamp":
                    header = replace(header, body_entry_work_units=True)
                elif fault == "source":
                    header = replace(
                        header, acquisition=replace(header.acquisition, source=object())
                    )
                elif fault == "identity":
                    identity = header.acquisition.source_identity
                    header = replace(
                        header,
                        acquisition=replace(
                            header.acquisition, source_identity=(99, *identity[1:])
                        ),
                    )
                elif fault == "read_budget":
                    header = replace(
                        header,
                        acquisition=replace(
                            header.acquisition, read_budget=PageBudgetTests().budget().read_budget
                        ),
                    )
                elif fault == "profile":
                    header = replace(header, profile=replace(header.profile, chunks=()))
                else:
                    header = replace(header, leaf=replace(header.leaf, path=(b"other",)))
                try:
                    self.assert_quiet_refusal(env, header)
                finally:
                    if undo is not None:
                        undo()

    def test_foreign_budget_and_closed_source_cannot_acquire_or_reopen(self):
        with self.environment() as env:
            header = self.admit(env)
            self.assert_quiet_refusal(env, header, budget=PageBudgetTests().budget())
            self.assert_quiet_refusal(env, None)
        self.assert_quiet_refusal(env, header)

    def test_whole_read_allowance_is_checked_without_consuming_header_stage(self):
        with self.environment() as env:
            header = self.admit(env)
            before = self.counters(env)
            env.ledgers.read.limits = replace(
                env.ledgers.read.limits, max_total_read_bytes=before.read[1] + 6
            )
            with self.assertRaises(PinnedReadRefused) as caught:
                self.body(env, header)
            self.assertEqual(caught.exception.reason_code, "resource_limit")
            self.assertEqual(self.counters(env).__dict__, before.__dict__)

    def test_allocation_reservation_is_atomic_and_visible_at_actual_range_boundary(self):
        for allowance in (13, 14):
            with self.subTest(allowance=allowance), self.environment() as env:
                header = self.admit(env)
                before = self.counters(env)
                env.ledgers.page.limits = replace(
                    env.ledgers.page.limits,
                    max_allocation_bytes=before.page["allocation_bytes"] + allowance,
                )
                actual = env.io.module._read_pinned_range
                observations = []

                def observed(
                    source, *, offset, length, env=env, actual=actual, observations=observations
                ):
                    self.assertIs(source, env.source)
                    observations.append(
                        (env.ledgers.page.allocation_bytes, env.ledgers.page.work_units)
                    )
                    return actual(source, offset=offset, length=length)

                with patch.object(env.io.module, "_read_pinned_range", side_effect=observed):
                    if allowance == 13:
                        with self.assertRaises(PinnedReadRefused) as caught:
                            self.body(env, header)
                        self.assertEqual(caught.exception.reason_code, "resource_limit")
                        self.assertEqual(observations, [])
                        self.assertEqual(self.counters(env).page, before.page)
                        self.assertEqual(tuple(env.io.positional_requests), before.requests)
                    else:
                        self.body(env, header)
                        self.assertEqual(
                            observations,
                            [
                                (
                                    before.page["allocation_bytes"] + 14,
                                    before.page["work_units"] + 2,
                                )
                            ],
                        )

    def test_segment_ceiling_and_clock_count_use_one_actual_range_check(self):
        cap = 1024 * 1024
        cases = ((cap, ((cap, 0),), 6, 8), (cap + 1, ((cap, 0), (1, cap)), 7, 11))
        for size, relative_ranges, work, clocks in cases:
            with self.subTest(size=size), self.environment(b"n" * size) as env:
                header = self.admit(env)
                before = self.counters(env)
                result = self.body(env, header)
                self.assertEqual(result.payload, env.case.body)
                start = 64 + len(env.case.header)
                self.assertEqual(
                    [request[1:] for request in env.io.positional_requests[len(before.requests) :]],
                    [(length, start + offset) for length, offset in relative_ranges],
                )
                self.assert_body_delta(env, before, size=size, work=work)
                self.assertEqual(env.io.events[len(before.events) :].count("clock"), clocks)

    def test_existing_read_sizing_retains_its_cap_and_one_deadline_check(self):
        with self.environment() as env:
            for requested, expected in (
                (1, 1),
                (1024 * 1024, 1024 * 1024),
                (1024 * 1024 + 1, 1024 * 1024),
            ):
                with self.subTest(requested=requested):
                    before = self.counters(env)
                    self.assertEqual(env.ledgers.read.before_read(requested), expected)
                    after = self.counters(env)
                    self.assertEqual(after.read, before.read)
                    self.assertEqual(after.page, before.page)
                    self.assertEqual(after.wire, before.wire)
                    self.assertEqual(after.requests, before.requests)
                    self.assertEqual(after.events, before.events + ("clock",))

    def test_work_refusals_keep_the_exact_completed_prefix(self):
        for allowance in range(6):
            with self.subTest(allowance=allowance), self.environment() as env:
                header = self.admit(env)
                before = self.counters(env)
                env.ledgers.page.limits = replace(
                    env.ledgers.page.limits,
                    max_work_units=before.page["work_units"] + allowance,
                )
                with self.assertRaises(PinnedReadRefused) as caught:
                    self.body(env, header)
                self.assertEqual(caught.exception.reason_code, "resource_limit")
                if allowance == 0:
                    self.assertEqual(self.counters(env).page, before.page)
                    self.assertEqual(tuple(env.io.positional_requests), before.requests)
                else:
                    self.assert_body_delta(
                        env, before, size=7 if allowance >= 2 else 0, work=allowance
                    )
                    self.assert_quiet_refusal(env, header)

    def test_short_segment_and_os_failure_retain_charges_without_retry(self):
        for fault in ("short_first", "short_last", "os"):
            with self.subTest(fault=fault), self.environment(b"n" * (1024 * 1024 + 1)) as env:
                header = self.admit(env)
                before = self.counters(env)
                start = 64 + len(env.case.header)
                if fault == "short_first":
                    env.io.responses[(start, 1024 * 1024)] = b"n"
                elif fault == "short_last":
                    env.io.responses[(start + 1024 * 1024, 1)] = b""
                else:
                    env.io.fail_pread = len(before.requests) + 1
                with self.assertRaises((env.io.module.TimestampMetadataRefused, OSError)):
                    self.body(env, header)
                read_bytes = {"short_first": 1, "short_last": 1024 * 1024, "os": 0}[fault]
                work = 3 if fault == "short_last" else 2
                self.assert_body_delta(env, before, size=read_bytes, work=work)
                self.assertEqual(len(env.io.positional_requests) - len(before.requests), work - 1)
                self.assert_quiet_refusal(env, header)

    def test_entry_and_postread_deadlines_do_not_publish_or_refund(self):
        for fault in ("entry", "postread"):
            with self.subTest(fault=fault), self.environment() as env:
                header = self.admit(env)
                before = self.counters(env)
                if fault == "entry":
                    env.io.expired = True
                else:
                    env.io.expire_after = f"pread:{len(before.requests) + 1}"
                try:
                    with self.assertRaises(PinnedReadRefused) as caught:
                        self.body(env, header)
                    self.assertEqual(caught.exception.reason_code, "deadline_exceeded")
                    if fault == "entry":
                        self.assertEqual(self.counters(env).page, before.page)
                        self.assertEqual(tuple(env.io.positional_requests), before.requests)
                    else:
                        self.assert_body_delta(env, before, size=7, work=2)
                finally:
                    env.io.expired = False

    def test_owner_cleanup_and_final_identity_prevent_outer_publication(self):
        for fault in ("identity", "close"):
            published = []
            with self.subTest(fault=fault):
                with self.assertRaises((OSError, StorageError)):
                    with self.environment() as env:
                        header = self.admit(env)
                        result = self.body(env, header)
                        if fault == "identity":
                            env.io.after = env.io.metadata(st_ino=99)
                        else:
                            env.io.close_errors[env.io.acquired[-1]] = OSError("neutral cleanup")
                    published.append(result)
                self.assertEqual(published, [])
                self.assertCountEqual(env.io.close_attempts, env.io.acquired)

    def test_primary_crc_refusal_retains_cleanup_error_without_publication(self):
        close_error = OSError("neutral cleanup witness")
        with (
            self.assertRaises(bounded_parquet._CompactWireMalformed) as caught,
            self.environment(b"123456789", crc=0) as env,
        ):
            header = self.admit(env)
            env.io.close_errors[env.io.acquired[-1]] = close_error
            self.body(env, header)
        self.assertEqual(caught.exception.reason_code, "invalid_wire")
        state = BaseException.__dict__["__dict__"].__get__(caught.exception, BaseException)
        self.assertEqual(state[env.io.module._PINNED_CLEANUP_ERRORS], (close_error,))
        self.assertCountEqual(env.io.close_attempts, env.io.acquired)

    def test_preread_and_preassembly_deadlines_preserve_the_admitted_prefix(self):
        for fault in ("preread", "preassembly"):
            with self.subTest(fault=fault), self.environment() as env:
                header = self.admit(env)
                before = self.counters(env)
                actual = env.io.module._read_pinned_range

                def observed(source, *, offset, length, env=env, actual=actual, fault=fault):
                    if fault == "preread":
                        env.io.expired = True
                    payload = actual(source, offset=offset, length=length)
                    env.io.expired = True
                    return payload

                try:
                    with (
                        patch.object(env.io.module, "_read_pinned_range", side_effect=observed),
                        self.assertRaises(PinnedReadRefused) as caught,
                    ):
                        self.body(env, header)
                    self.assertEqual(caught.exception.reason_code, "deadline_exceeded")
                    self.assert_body_delta(
                        env, before, size=7 if fault == "preassembly" else 0, work=2
                    )
                finally:
                    env.io.expired = False

    def test_read_budget_invalid_and_exhausted_requests_preserve_state(self):
        for requested, exhausted in ((0, False), (-1, False), (True, False), (1, True)):
            with self.subTest(requested=requested, exhausted=exhausted), self.environment() as env:
                if exhausted:
                    env.ledgers.read.limits = replace(
                        env.ledgers.read.limits,
                        max_total_read_bytes=env.ledgers.read.total_read_bytes,
                    )
                before = self.counters(env)
                with self.assertRaises(PinnedReadRefused) as caught:
                    env.ledgers.read.before_read(requested)
                self.assertEqual(
                    caught.exception.reason_code,
                    "resource_limit" if exhausted else "invalid_contract",
                )
                expected = before.__dict__ | {
                    "events": before.events + (("clock",) if exhausted else ()),
                }
                self.assertEqual(self.counters(env).__dict__, expected)

    def test_actual_foreign_read_prevents_a_successful_body_receipt(self):
        with self.environment() as env:
            header = self.admit(env)
            before = self.counters(env)
            actual = env.io.module._read_pinned_range

            def observed(source, *, offset, length):
                self.assertIs(source, env.source)
                payload = actual(source, offset=offset, length=length)
                self.assertEqual(actual(source, offset=0, length=1), b"P")
                return payload

            with (
                patch.object(env.io.module, "_read_pinned_range", side_effect=observed),
                self.assertRaises(PinnedReadRefused) as caught,
            ):
                self.body(env, header)
            self.assertEqual(caught.exception.reason_code, "invalid_contract")
            self.assert_body_delta(env, before, size=8, work=6)
            start = 64 + len(env.case.header)
            self.assertEqual(
                [request[1:] for request in env.io.positional_requests[len(before.requests) :]],
                [(7, start), (1, 0)],
            )
            self.assertEqual(tuple(env.io.close_attempts), before.closes)
            self.assertFalse(env.source._closed)
            self.assert_quiet_refusal(env, header)


class SequentialOwnedPageTests(unittest.TestCase):
    """Neutral wire witnesses for one source-owned opaque column chunk."""

    plain_three = b"\x15\x00\x15\x06\x15\x06\x2c\x15\x02\x15\x00\x15\x06\x15\x06\x00\x00"
    plain_two = b"\x15\x00\x15\x04\x15\x04\x2c\x15\x02\x15\x00\x15\x06\x15\x06\x00\x00"
    dictionary_two = b"\x15\x04\x15\x04\x15\x04\x4c\x15\x04\x15\x00\x00\x00"
    encoded_three = b"\x15\x00\x15\x06\x15\x06\x2c\x15\x02\x15\x10\x15\x06\x15\x06\x00\x00"
    ordinary_pages = ((plain_three, b"\x01\x02\x03"), (plain_two, b"\x04\x05"))
    mixed_pages = (
        (dictionary_two, b"ab"),
        (encoded_three, b"\x01\x02\x03"),
        (plain_two, b"\x04\x05"),
    )
    environment = OwnedPageBodyTests.environment
    counters = OwnedPageBodyTests.counters

    def fixture(self, case=None):
        return self.case(self.ordinary_pages) if case is None else case

    def case(self, pages, metadata=None, *, rows=2, declared_size=None):
        """Keep column, group-size and row declarations independently testable on the wire."""
        api = FooterProfileTests()
        segment = b"".join(header + body for header, body in pages)
        if metadata is None:
            metadata = api.meta(
                values=2,
                data=64,
                compressed=len(segment),
                uncompressed=len(segment),
                encodings=(0, 3),
            )
        footer = api.payload(
            (api.root(1), api.node(b"n")),
            (
                api.group(
                    (api.chunk(metadata),),
                    rows=rows,
                    size=len(segment) if declared_size is None else declared_size,
                ),
            ),
            rows=rows,
        )
        prefix = b"PAR1" + b"\x00" * 60 + segment
        return SimpleNamespace(
            payload=prefix + footer + len(footer).to_bytes(4, "little") + b"PAR1",
            footer=footer,
            pages=pages,
            footer_start=len(prefix),
        )

    def mixed_case(self):
        metadata = FooterProfileTests().meta(
            values=2,
            data=79,
            compressed=54,
            uncompressed=54,
            encodings=(0, 3, 8),
            extra=((11, "i64", _grammar_integer(64)),),
        )
        return self.case(self.mixed_pages, metadata)

    def api(self):
        start = getattr(bounded_parquet, "_start_owned_column_chunk", None)
        self.assertTrue(callable(start), "missing source-owned sequential chunk start")
        return start

    def start(self, env):
        owner = self.api()(
            env.acquisition,
            env.envelope,
            env.profile,
            row_group_index=0,
            column_index=0,
            budget=env.ledgers.page,
        )
        self.assertIsNotNone(owner, "missing sequential owner result")
        return owner

    def assert_prefix(self, env, before, expected):
        after = self.counters(env)
        self.assertEqual(
            (
                after.read[1] - before.read[1],
                after.page["work_units"] - before.page["work_units"],
                after.page["allocation_bytes"] - before.page["allocation_bytes"],
                after.events.count("clock") - before.events.count("clock"),
            ),
            expected,
        )
        self.assertEqual(after.read[0], before.read[0])

    def assert_invalid(self, env, operation):
        before = self.counters(env)
        with self.assertRaises(PinnedReadRefused) as caught:
            operation()
        self.assertEqual(caught.exception.reason_code, "invalid_contract")
        self.assertEqual(self.counters(env), before)

    def assert_poisoned(self, env, owner, body):
        self.assert_invalid(env, lambda: owner.advance(body))
        self.assert_invalid(env, lambda: owner.finish(body))
        self.assert_invalid(env, lambda: owner.current_body)

    def test_two_literal_data_pages_have_exact_ranges_receipts_and_ledgers(self):
        with self.environment() as env:
            before = self.counters(env)
            actual = env.io.module._read_pinned_range
            observed = []

            def boundary(source, *, offset, length):
                if offset in (81, 101):
                    observed.append(
                        (
                            offset,
                            length,
                            env.ledgers.page.work_units - before.page["work_units"],
                            env.ledgers.page.allocation_bytes - before.page["allocation_bytes"],
                        )
                    )
                return actual(source, offset=offset, length=length)

            with patch.object(env.io.module, "_read_pinned_range", side_effect=boundary):
                owner = self.start(env)
                first = owner.current_body
                self.assertEqual(first.payload, b"\x01\x02\x03")
                self.assertEqual(first.header.absolute_body_span, (81, 84))
                self.assert_prefix(env, before, (20, 95, 176, 167))
                self.assertEqual(
                    [request[1:] for request in env.io.positional_requests[len(before.requests) :]],
                    [(1, offset) for offset in range(64, 81)] + [(3, 81)],
                )
                second = owner.advance(first)
                self.assertIs(owner.current_body, second)
                self.assertEqual(second.payload, b"\x04\x05")
                self.assertEqual(second.header.absolute_body_span, (101, 103))
                self.assert_prefix(env, before, (39, 186, 350, 330))
                receipt = owner.finish(second)
            self.assertEqual(observed, [(81, 3, 90, 176), (101, 2, 181, 350)])
            self.assert_prefix(env, before, (39, 188, 350, 332))
            self.assertEqual(
                [request[1:] for request in env.io.positional_requests[len(before.requests) :]],
                [(1, offset) for offset in range(64, 81)]
                + [(3, 81)]
                + [(1, offset) for offset in range(84, 101)]
                + [(2, 101)],
            )
            self.assertEqual((receipt.row_group_index, receipt.column_index), (0, 0))
            self.assertEqual(receipt.byte_extent, (64, 103))
            self.assertEqual((receipt.page_count, receipt.data_page_count), (2, 2))
            self.assertEqual(receipt.dictionary_page_count, 0)
            self.assertIsNone(receipt.first_dictionary_offset)
            self.assertEqual(receipt.first_data_offset, 64)
            self.assertEqual(
                (receipt.compressed_total, receipt.declared_uncompressed_total), (39, 39)
            )
            self.assertEqual(receipt.data_positions, 2)
            self.assertEqual(receipt.receipt_after, self.counters(env).read)
            self.assertEqual(receipt.status, "opaque_framing_provisional")
            self.assertIs(receipt.whole_source_digest_verified, False)
            self.assertFalse(env.profile.read_eligible)
            self.assertFalse(first.header.admission.decode_eligible)
            after = self.counters(env)
            self.assertEqual(
                (after.wire[0] - before.wire[0], after.wire[5] - before.wire[5]), (34, 306)
            )
            self.assertEqual(after.page["compressed_bytes"] - before.page["compressed_bytes"], 5)
            self.assertEqual(
                after.page["uncompressed_bytes"] - before.page["uncompressed_bytes"], 5
            )
            self.assertEqual(after.page["values"] - before.page["values"], 2)
            with self.assertRaises(FrozenInstanceError):
                receipt.data_positions = 99

    def test_dictionary_and_mixed_data_encodings_reconcile_without_counting_entries(self):
        with self.environment(case=self.mixed_case()) as env:
            before = self.counters(env)
            owner = self.start(env)
            dictionary = owner.current_body
            self.assertEqual(dictionary.payload, b"ab")
            self.assert_prefix(env, before, (15, 75, 108, 131))
            encoded = owner.advance(dictionary)
            self.assertEqual(encoded.payload, b"\x01\x02\x03")
            self.assertEqual(encoded.header.page_origin, 79)
            self.assert_prefix(env, before, (35, 168, 284, 296))
            plain = owner.advance(encoded)
            self.assertEqual(plain.header.page_origin, 99)
            self.assert_prefix(env, before, (54, 259, 458, 459))
            receipt = owner.finish(plain)
            self.assert_prefix(env, before, (54, 261, 458, 461))
            self.assertEqual((receipt.page_count, receipt.data_page_count), (3, 2))
            self.assertEqual(receipt.dictionary_page_count, 1)
            self.assertEqual((receipt.first_dictionary_offset, receipt.first_data_offset), (64, 79))
            self.assertEqual(receipt.data_positions, 2)
            after = self.counters(env)
            self.assertEqual(
                after.page["dictionary_entries"] - before.page["dictionary_entries"], 2
            )
            self.assertEqual(
                after.page["dictionary_payload_bytes"] - before.page["dictionary_payload_bytes"], 2
            )
            self.assertEqual(
                (after.wire[0] - before.wire[0], after.wire[5] - before.wire[5]), (47, 397)
            )

    def test_legacy_dictionary_value_encoding_can_follow_the_owned_dictionary(self):
        encoded = self.encoded_three[:10] + b"\x04" + self.encoded_three[11:]
        pages = (self.mixed_pages[0], (encoded, b"\x01\x02\x03"), self.mixed_pages[2])
        metadata = FooterProfileTests().meta(
            values=2,
            data=79,
            compressed=54,
            uncompressed=54,
            encodings=(0, 3, 8, 2),
            extra=((11, "i64", _grammar_integer(64)),),
        )
        with self.environment(case=self.case(pages, metadata)) as env:
            before = self.counters(env)
            owner = self.start(env)
            encoded_body = owner.advance(owner.current_body)
            self.assertEqual(encoded_body.header.admission.detail.encoding, 2)
            last = owner.advance(encoded_body)
            receipt = owner.finish(last)
            self.assertEqual(receipt.data_positions, 2)
            self.assert_prefix(env, before, (54, 262, 458, 462))

    def test_wrong_acknowledgment_and_early_finish_are_pure_and_recoverable(self):
        with self.environment() as env:
            owner = self.start(env)
            first = owner.current_body
            self.assert_invalid(env, lambda: owner.advance(object()))
            self.assert_invalid(env, lambda: owner.advance(replace(first)))
            self.assert_invalid(env, lambda: owner.finish(first))
            self.assertIs(owner.current_body, first)
            second = owner.advance(first)
            self.assert_invalid(env, lambda: owner.advance(first))
            self.assert_invalid(env, lambda: owner.advance(second))
            self.assertIs(owner.current_body, second)
            owner.finish(second)
            self.assert_invalid(env, lambda: owner.finish(second))
            self.assert_invalid(env, lambda: owner.current_body)

        with self.environment() as env:
            owner = self.start(env)
            first = owner.current_body
            for slot in ("_body", "_chunk", "_leaf"):
                for invalid in (None, object()):
                    with self.subTest(slot=slot, invalid_type=type(invalid).__name__):
                        saved = getattr(owner, slot)
                        setattr(owner, slot, invalid)
                        try:
                            if slot == "_body":
                                self.assert_invalid(env, lambda: owner.current_body)
                            self.assert_invalid(env, lambda: owner.advance(first))
                            self.assertEqual(owner._phase, "ready")
                        finally:
                            setattr(owner, slot, saved)
                        self.assertIs(owner.current_body, first)
            second = owner.advance(first)
            self.assertEqual(owner.finish(second).page_count, 2)

    def test_dictionary_payload_encoding_is_not_a_prior_dictionary_requirement(self):
        dictionary = self.dictionary_two[:10] + b"\x04" + self.dictionary_two[11:]
        pages = ((dictionary, b"ab"), self.mixed_pages[1], self.mixed_pages[2])
        metadata = FooterProfileTests().meta(
            values=2,
            data=79,
            compressed=54,
            uncompressed=54,
            encodings=(0, 3, 8, 2),
            extra=((11, "i64", _grammar_integer(64)),),
        )
        with self.environment(case=self.case(pages, metadata)) as env:
            before = self.counters(env)
            owner = self.start(env)
            self.assertEqual(owner.current_body.header.admission.detail.encoding, 2)
            encoded = owner.advance(owner.current_body)
            last = owner.advance(encoded)
            self.assertEqual(owner.finish(last).data_positions, 2)
            self.assert_prefix(env, before, (54, 264, 458, 464))

    def test_intervening_actual_read_and_work_or_closed_source_are_pure_refusals(self):
        for operation, fault in (
            (operation, fault)
            for operation in ("advance", "finish")
            for fault in ("read", "work", "closed")
        ):
            with self.subTest(operation=operation, fault=fault), self.environment() as env:
                owner = self.start(env)
                body = owner.current_body
                if operation == "finish":
                    body = owner.advance(body)
                if fault == "read":
                    self.assertEqual(
                        env.io.module._read_pinned_range(env.source, offset=0, length=1), b"P"
                    )
                elif fault == "work":
                    env.ledgers.page.reserve(bounded_parquet._PageCharge(work_units=1))
                else:
                    env.source._closed = True
                try:
                    action = owner.advance if operation == "advance" else owner.finish
                    self.assert_invalid(env, lambda action=action, body=body: action(body))
                finally:
                    if fault == "closed":
                        env.source._closed = False

        for operation, final_work, expected in (
            ("advance", 186, (19, 91, 174, 163)),
            ("finish", 188, (0, 2, 0, 2)),
        ):
            with self.subTest(late_closure=operation), self.environment() as env:
                work_origin = env.ledgers.page.work_units
                owner = self.start(env)
                body = owner.current_body
                if operation == "finish":
                    body = owner.advance(body)
                before = self.counters(env)
                original_clock = env.ledgers.read.clock_ns
                fired = []

                def closing_clock(
                    clock=original_clock,
                    page=env.ledgers.page,
                    source=env.source,
                    target_work=work_origin + final_work,
                    events=fired,
                ) -> int:
                    stamp = clock()
                    if not events and page.work_units == target_work - 1:
                        events.append("closed")
                        source._closed = True
                    return stamp

                env.ledgers.read.clock_ns = closing_clock
                try:
                    with self.assertRaises(PinnedReadRefused) as caught:
                        getattr(owner, operation)(body)
                    self.assertEqual(caught.exception.reason_code, "invalid_contract")
                    self.assertEqual(fired, ["closed"])
                    self.assertEqual(env.ledgers.page.work_units, work_origin + final_work)
                    self.assert_prefix(env, before, expected)
                    self.assertEqual(owner._phase, "poisoned")
                    self.assert_poisoned(env, owner, body)
                finally:
                    env.ledgers.read.clock_ns = original_clock
                    env.source._closed = False

    def test_successor_work_limits_keep_literal_prefix_and_poison_admitted_transition(self):
        cases = (
            (0, 0, 0),
            (77, 17, 170),
            (78, 17, 170),
            (79, 17, 170),
            (82, 17, 170),
            (83, 17, 170),
            (84, 17, 170),
            (85, 17, 174),
            (86, 19, 174),
            (89, 19, 174),
            (90, 19, 174),
        )
        for allowance, read_bytes, allocation in cases:
            with self.subTest(allowance=allowance), self.environment() as env:
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                env.ledgers.page.limits = replace(
                    env.ledgers.page.limits,
                    max_work_units=before.page["work_units"] + allowance,
                )
                with self.assertRaises(PinnedReadRefused) as caught:
                    owner.advance(body)
                self.assertEqual(caught.exception.reason_code, "resource_limit")
                after = self.counters(env)
                self.assertEqual(after.read[1] - before.read[1], read_bytes)
                self.assertEqual(after.page["work_units"] - before.page["work_units"], allowance)
                self.assertEqual(
                    after.page["allocation_bytes"] - before.page["allocation_bytes"], allocation
                )
                self.assert_poisoned(env, owner, body)

    def test_finish_work_limits_poison_without_reading_or_publishing(self):
        for allowance in (0, 1):
            with self.subTest(allowance=allowance), self.environment() as env:
                owner = self.start(env)
                last = owner.advance(owner.current_body)
                before = self.counters(env)
                env.ledgers.page.limits = replace(
                    env.ledgers.page.limits,
                    max_work_units=before.page["work_units"] + allowance,
                )
                with self.assertRaises(PinnedReadRefused) as caught:
                    owner.finish(last)
                self.assertEqual(caught.exception.reason_code, "resource_limit")
                self.assert_prefix(env, before, (0, allowance, 0, allowance + 1))
                self.assert_poisoned(env, owner, last)

    def test_initial_policy_refusals_precede_stamp_and_body_acquisition(self):
        api = FooterProfileTests()
        encoded = self.plain_three[:10] + b"\x10" + self.plain_three[11:]
        legacy_encoded = self.plain_three[:10] + b"\x04" + self.plain_three[11:]
        cases = (
            ((encoded, b"\x01\x02\x03"), (0, 3, 8), 82, 152),
            ((legacy_encoded, b"\x01\x02\x03"), (0, 2, 3), 82, 152),
            ((self.plain_three, b"\x01\x02\x03"), (3,), 83, 153),
        )
        for page, encodings, work, clocks in cases:
            pages = (page, self.ordinary_pages[1])
            metadata = api.meta(
                values=2, data=64, compressed=39, uncompressed=39, encodings=encodings
            )
            with (
                self.subTest(encodings=encodings),
                self.environment(case=self.case(pages, metadata)) as env,
            ):
                before = self.counters(env)
                with self.assertRaises(bounded_parquet._CompactWireMalformed) as caught:
                    self.start(env)
                self.assertEqual(caught.exception.reason_code, "schema_mismatch")
                self.assert_prefix(env, before, (17, work, 170, clocks))
                self.assertEqual(
                    [request[1:] for request in env.io.positional_requests[len(before.requests) :]],
                    [(1, offset) for offset in range(64, 81)],
                )

        # These are typed-profile misuse cases, not malformed provider observations.
        for encodings, work in (
            (None, 1),
            ([0, 3], 1),
            ((True, 3), 2),
            (("not-an-encoding", 3), 2),
        ):
            with self.subTest(policy_encodings=encodings), self.environment() as env:
                owner = self.start(env)
                body = owner.current_body
                original_chunk = owner._chunk
                owner._chunk = replace(original_chunk, encodings=encodings)
                before = self.counters(env)
                try:
                    with self.assertRaises(PinnedReadRefused) as caught:
                        owner._check_header(body.header.admission, body.header.page_origin)
                    self.assertEqual(caught.exception.reason_code, "invalid_contract")
                    self.assert_prefix(env, before, (0, work, 0, work))
                    self.assertEqual(self.counters(env).requests, before.requests)
                    self.assertEqual(self.counters(env).wire, before.wire)
                    self.assertEqual(owner._phase, "ready")
                    self.assertIs(owner.current_body, body)
                finally:
                    owner._chunk = original_chunk

    def test_dictionary_order_and_first_data_offset_refuse_before_later_body(self):
        api = FooterProfileTests()
        cases = (
            (
                (self.mixed_pages[0], self.mixed_pages[0], *self.ordinary_pages),
                api.meta(
                    values=2,
                    data=79,
                    compressed=69,
                    uncompressed=69,
                    encodings=(0, 3),
                    extra=((11, "i64", _grammar_integer(64)),),
                ),
                1,
                92,
            ),
            (
                (*self.ordinary_pages, self.mixed_pages[0]),
                api.meta(values=2, data=64, compressed=54, uncompressed=54, encodings=(0, 3)),
                2,
                116,
            ),
            (
                self.mixed_pages,
                api.meta(
                    values=2,
                    data=80,
                    compressed=54,
                    uncompressed=54,
                    encodings=(0, 3, 8),
                    extra=((11, "i64", _grammar_integer(64)),),
                ),
                1,
                96,
            ),
        )
        for pages, metadata, accepted_pages, refused_body_start in cases:
            with (
                self.subTest(accepted_pages=accepted_pages, refused_body_start=refused_body_start),
                self.environment(case=self.case(pages, metadata)) as env,
            ):
                owner = self.start(env)
                body = owner.current_body
                for _ in range(accepted_pages - 1):
                    body = owner.advance(body)
                before = self.counters(env)
                with self.assertRaises(bounded_parquet._CompactWireMalformed) as caught:
                    owner.advance(body)
                self.assertEqual(caught.exception.reason_code, "schema_mismatch")
                requests = env.io.positional_requests[len(before.requests) :]
                self.assertTrue(requests)
                self.assertTrue(all(offset < refused_body_start for _, _, offset in requests))
                self.assert_poisoned(env, owner, body)

    def test_index_and_unknown_page_types_refuse_without_body_reads(self):
        index = b"\x15\x02\x15\x04\x15\x04\x3c\x00\x00"
        unknown = b"\x15\x08" + self.plain_three[2:]
        api = FooterProfileTests()
        for header, body in ((index, b"ab"), (unknown, b"\x01\x02\x03")):
            length = len(header) + len(body)
            metadata = api.meta(
                values=0, data=64, compressed=length, uncompressed=length, encodings=(0, 3)
            )
            case = self.case(((header, body),), metadata, rows=0)
            with self.subTest(header=header), self.environment(case=case) as env:
                before = self.counters(env)
                with self.assertRaises(PinnedReadRefused) as caught:
                    self.start(env)
                self.assertEqual(caught.exception.reason_code, "unsupported_mode")
                self.assertEqual(self.counters(env).read[1] - before.read[1], len(header))
                self.assertTrue(
                    all(
                        offset < 64 + len(header)
                        for _, _, offset in env.io.positional_requests[len(before.requests) :]
                    )
                )
        metadata = api.meta(values=1, data=64, compressed=31, uncompressed=31, encodings=(0, 3))
        case = self.case((self.ordinary_pages[0], (index, b"ab")), metadata, rows=1)
        with self.environment(case=case) as env:
            owner = self.start(env)
            body = owner.current_body
            before = self.counters(env)
            with self.assertRaises(PinnedReadRefused) as caught:
                owner.advance(body)
            self.assertEqual(caught.exception.reason_code, "unsupported_mode")
            self.assertEqual(self.counters(env).read[1] - before.read[1], 9)
            self.assertTrue(
                all(
                    offset < 93
                    for _, _, offset in env.io.positional_requests[len(before.requests) :]
                )
            )
            self.assert_poisoned(env, owner, body)

        for page_type, detail, error, reason in (
            (0, object(), bounded_parquet._CompactWireMalformed, "schema_mismatch"),
            (4, None, PinnedReadRefused, "unsupported_mode"),
        ):
            with self.subTest(policy_page_type=page_type), self.environment() as env:
                owner = self.start(env)
                body = owner.current_body
                admission = replace(body.header.admission, page_type=page_type, detail=detail)
                before = self.counters(env)
                with self.assertRaises(error) as caught:
                    owner._check_header(admission, body.header.page_origin)
                self.assertEqual(caught.exception.reason_code, reason)
                self.assert_prefix(env, before, (0, 1, 0, 1))
                self.assertEqual(self.counters(env).requests, before.requests)
                self.assertEqual(self.counters(env).wire, before.wire)
                self.assertEqual(owner._phase, "ready")
                self.assertIs(owner.current_body, body)

    def test_declared_uncompressed_and_data_position_mismatches_poison_finish(self):
        api = FooterProfileTests()
        for values, uncompressed in ((3, 39), (2, 40)):
            metadata = api.meta(
                values=values,
                data=64,
                compressed=39,
                uncompressed=uncompressed,
                encodings=(0, 3),
            )
            case = self.case(self.ordinary_pages, metadata, rows=values, declared_size=uncompressed)
            with (
                self.subTest(values=values, uncompressed=uncompressed),
                self.environment(case=case) as env,
            ):
                owner = self.start(env)
                last = owner.advance(owner.current_body)
                before = self.counters(env)
                with self.assertRaises(bounded_parquet._CompactWireMalformed) as caught:
                    owner.finish(last)
                self.assertEqual(caught.exception.reason_code, "schema_mismatch")
                self.assert_prefix(env, before, (0, 1, 0, 1))
                self.assert_poisoned(env, owner, last)

    def test_truncated_successor_and_unframed_tail_prevent_completion(self):
        cases = (
            (self.ordinary_pages[0], (self.plain_two[:-1], b"")),
            (self.ordinary_pages[0], (self.plain_two, b"\x04")),
            (*self.ordinary_pages, (b"\x00", b"")),
        )
        for pages in cases:
            with self.subTest(last=pages[-1]), self.environment(case=self.case(pages)) as env:
                owner = self.start(env)
                body = owner.current_body
                if len(pages) == 3:
                    body = owner.advance(body)
                with self.assertRaises((PinnedReadRefused, bounded_parquet._CompactWireMalformed)):
                    owner.advance(body)
                self.assert_poisoned(env, owner, body)

    def test_later_crc_mismatch_retains_actual_acquisition_without_completion(self):
        bad_crc = b"\x15\x00\x15\x12\x15\x12\x15\x00\x1c\x15\x02\x15\x00\x15\x06\x15\x06\x00\x00"
        pages = (self.ordinary_pages[0], (bad_crc, b"123456789"))
        with self.environment(case=self.case(pages)) as env:
            owner = self.start(env)
            body = owner.current_body
            before = self.counters(env)
            with self.assertRaises(bounded_parquet._CompactWireMalformed) as caught:
                owner.advance(body)
            self.assertEqual(caught.exception.reason_code, "invalid_wire")
            self.assertEqual(self.counters(env).read[1] - before.read[1], 28)
            self.assertEqual(
                [request[1:] for request in env.io.positional_requests[len(before.requests) :]][-1],
                (9, 103),
            )
            self.assert_poisoned(env, owner, body)

    def test_actual_short_body_and_os_failure_preserve_exact_prefix(self):
        for fault, expected in (("short", (18, 86, 174, 158)), ("os", (17, 86, 174, 157))):
            with self.subTest(fault=fault), self.environment() as env:
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                if fault == "short":
                    env.io.responses[(101, 2)] = b"\x04"
                else:
                    env.io.fail_pread = len(before.requests) + 18
                with self.assertRaises((env.io.module.TimestampMetadataRefused, OSError)):
                    owner.advance(body)
                self.assert_prefix(env, before, expected)
                self.assert_poisoned(env, owner, body)

    def test_entry_and_postread_deadline_failures_poison_without_refund(self):
        for fault, expected in (("entry", (0, 0, 0, 1)), ("postread", (19, 86, 174, 158))):
            with self.subTest(fault=fault), self.environment() as env:
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                if fault == "entry":
                    env.io.expired = True
                else:
                    env.io.expire_after = f"pread:{len(before.requests) + 18}"
                try:
                    with self.assertRaises(PinnedReadRefused) as caught:
                        owner.advance(body)
                    self.assertEqual(caught.exception.reason_code, "deadline_exceeded")
                    self.assert_prefix(env, before, expected)
                finally:
                    env.io.expired = False
                self.assert_poisoned(env, owner, body)

    def test_successor_allocation_is_reserved_atomically_before_body(self):
        for allowance in (173, 174):
            with self.subTest(allowance=allowance), self.environment() as env:
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                env.ledgers.page.limits = replace(
                    env.ledgers.page.limits,
                    max_allocation_bytes=before.page["allocation_bytes"] + allowance,
                )
                if allowance == 173:
                    with self.assertRaises(PinnedReadRefused) as caught:
                        owner.advance(body)
                    self.assertEqual(caught.exception.reason_code, "resource_limit")
                    self.assert_prefix(env, before, (17, 84, 170, 155))
                    self.assert_poisoned(env, owner, body)
                else:
                    last = owner.advance(body)
                    self.assert_prefix(env, before, (19, 91, 174, 163))
                    owner.finish(last)

    def test_empty_body_and_present_zero_crc_are_distinct_without_zero_length_reads(self):
        empty = b"\x15\x00\x15\x00\x15\x00\x2c\x15\x00\x15\x00\x15\x06\x15\x06\x00\x00"
        zero_crc = b"\x15\x00\x15\x00\x15\x00\x15\x00\x1c\x15\x00\x15\x00\x15\x06\x15\x06\x00\x00"
        for header, verified in ((empty, False), (zero_crc, True)):
            metadata = FooterProfileTests().meta(
                values=0,
                data=64,
                compressed=len(header),
                uncompressed=len(header),
                encodings=(0, 3),
            )
            case = self.case(((header, b""),), metadata, rows=0)
            with self.subTest(verified=verified), self.environment(case=case) as env:
                before = self.counters(env)
                owner = self.start(env)
                body = owner.current_body
                self.assertEqual(body.payload, b"")
                self.assertIs(body.crc_verified, verified)
                self.assertEqual(self.counters(env).read[1] - before.read[1], len(header))
                self.assertTrue(
                    all(
                        length == 1
                        for _, length, _ in env.io.positional_requests[len(before.requests) :]
                    )
                )
                receipt = owner.finish(body)
                self.assertEqual(receipt.data_positions, 0)
                self.assertEqual(receipt.byte_extent, (64, 64 + len(header)))

    def test_v2_policy_does_not_invent_explicit_level_encoding_fields(self):
        header = (
            b"\x15\x06\x15\x06\x15\x06\x5c\x15\x02\x15\x00\x15\x02\x15\x00\x15\x00\x15\x00\x00\x00"
        )
        metadata = FooterProfileTests().meta(
            values=1, data=64, compressed=24, uncompressed=24, encodings=(0,)
        )
        case = self.case(((header, b"abc"),), metadata, rows=1)
        with self.environment(case=case) as env:
            before = self.counters(env)
            owner = self.start(env)
            body = owner.current_body
            self.assertEqual(body.header.admission.page_type, 3)
            self.assertEqual(body.payload, b"abc")
            receipt = owner.finish(body)
            self.assertEqual(receipt.data_positions, 1)
            self.assertEqual(self.counters(env).read[1] - before.read[1], 24)

    def test_successor_body_uses_the_existing_segment_cap(self):
        header = (
            b"\x15\x00\x15\x82\x80\x80\x01\x15\x82\x80\x80\x01\x2c"
            b"\x15\x02\x15\x00\x15\x06\x15\x06\x00\x00"
        )
        pages = (self.ordinary_pages[0], (header, b"n" * (1024 * 1024 + 1)))
        with self.environment(case=self.case(pages)) as env:
            owner = self.start(env)
            before = self.counters(env)
            last = owner.advance(owner.current_body)
            self.assertEqual(len(last.payload), 1024 * 1024 + 1)
            self.assertEqual(
                [request[1:] for request in env.io.positional_requests[len(before.requests) :]],
                [(1, offset) for offset in range(84, 107)]
                + [(1024 * 1024, 107), (1, 107 + 1024 * 1024)],
            )
            receipt = owner.finish(last)
            self.assertEqual(receipt.compressed_total, 1048620)

    def test_final_identity_and_cleanup_failure_prevent_outer_publication(self):
        for fault in ("identity", "close"):
            published = []
            with self.subTest(fault=fault):
                with self.assertRaises((OSError, StorageError)):
                    with self.environment() as env:
                        owner = self.start(env)
                        last = owner.advance(owner.current_body)
                        receipt = owner.finish(last)
                        if fault == "identity":
                            env.io.after = env.io.metadata(st_ino=99)
                        else:
                            env.io.close_errors[env.io.acquired[-1]] = OSError("neutral cleanup")
                    published.append(receipt)
                self.assertEqual(published, [])
                self.assertCountEqual(env.io.close_attempts, env.io.acquired)

    def test_legacy_initial_wrappers_keep_unmodified_ledger_costs(self):
        with self.environment() as env:
            before = self.counters(env)
            header = bounded_parquet._admit_owned_initial_page_header(
                env.acquisition,
                env.envelope,
                env.profile,
                row_group_index=0,
                column_index=0,
                budget=env.ledgers.page,
            )
            self.assert_prefix(env, before, (17, 81, 170, 151))
            result = bounded_parquet._read_owned_initial_page_body(header, budget=env.ledgers.page)
            self.assertEqual(result.payload, b"\x01\x02\x03")
            self.assert_prefix(env, before, (20, 87, 176, 159))

    def test_start_rejects_foreign_outer_bindings_without_clock_or_acquisition(self):
        with self.environment() as env:
            start = self.api()
            copied_payload = bytes(bytearray(env.envelope.payload))
            foreign_envelope = replace(env.envelope, payload=copied_payload)
            foreign_profile = replace(
                env.profile, document=replace(env.profile.document, payload=copied_payload)
            )
            foreign_budget = bounded_parquet._PageBudget(
                read_budget=env.ledgers.read,
                wire_budget=env.ledgers.wire,
                limits=env.ledgers.page.limits,
            )
            self.assert_invalid(
                env,
                lambda: start(
                    env.acquisition,
                    foreign_envelope,
                    env.profile,
                    row_group_index=0,
                    column_index=0,
                    budget=env.ledgers.page,
                ),
            )
            self.assert_invalid(
                env,
                lambda: start(
                    env.acquisition,
                    env.envelope,
                    foreign_profile,
                    row_group_index=0,
                    column_index=0,
                    budget=env.ledgers.page,
                ),
            )
            self.assert_invalid(
                env,
                lambda: start(
                    env.acquisition,
                    env.envelope,
                    env.profile,
                    row_group_index=0,
                    column_index=0,
                    budget=foreign_budget,
                ),
            )
            self.assert_invalid(
                env,
                lambda: start(
                    env.acquisition,
                    env.envelope,
                    env.profile,
                    row_group_index=-1,
                    column_index=0,
                    budget=env.ledgers.page,
                ),
            )

        with self.environment() as env:
            owner = self.start(env)
            body = owner.current_body
            admit_header = getattr(bounded_parquet, "_admit_owned_page_header", None)
            read_body = getattr(bounded_parquet, "_read_owned_page_body", None)
            if not callable(admit_header) or not callable(read_body):
                self.fail("missing owned-page header/body helper")
            for supplied_owner in (object(), owner):
                with self.subTest(helper_owner_type=type(supplied_owner).__name__):
                    self.assert_invalid(
                        env,
                        lambda supplied_owner=supplied_owner: admit_header(
                            env.acquisition,
                            env.envelope,
                            env.profile,
                            row_group_index=0,
                            column_index=0,
                            budget=env.ledgers.page,
                            owner=supplied_owner,
                        ),
                    )
                    self.assert_invalid(
                        env,
                        lambda supplied_owner=supplied_owner: read_body(
                            body.header, budget=env.ledgers.page, owner=supplied_owner
                        ),
                    )
                    self.assertEqual(owner._phase, "ready")
                    self.assertIs(owner.current_body, body)

            foreign_profile = replace(env.profile)
            foreign_header = replace(body.header, profile=foreign_profile)
            owner._phase, owner._body = "attempted", None
            try:
                self.assert_invalid(
                    env,
                    lambda: admit_header(
                        env.acquisition,
                        env.envelope,
                        foreign_profile,
                        row_group_index=0,
                        column_index=0,
                        budget=env.ledgers.page,
                        owner=owner,
                    ),
                )
                self.assert_invalid(
                    env,
                    lambda: read_body(foreign_header, budget=env.ledgers.page, owner=owner),
                )
                self.assertEqual(owner._phase, "attempted")
                self.assertIsNone(owner._body)
            finally:
                owner._phase, owner._body = "ready", body
            owner_type = type(owner)
            self.assert_invalid(env, lambda: owner_type.advance(object(), body))
            self.assert_invalid(env, lambda: owner_type.finish(object(), body))
            self.assertIs(owner.current_body, body)
            second = owner.advance(body)
            self.assertEqual(owner.finish(second).page_count, 2)

    def level_case(
        self, version="v1", *, repetition=0, n=8, malformed=False, codec=0, uncompressed_extra=0
    ):
        """Compose admitted neutral level wires through the existing footer/page helpers."""
        levels = PageLevelScanTests().section(
            version,
            n=n,
            max_def=int(repetition > 0),
            max_rep=int(repetition == 2),
            repetition=b"\x03\xaa" if repetition == 2 and n else b"",
            definition=b"\x03\x55" if repetition and n else b"",
            nulls=n // 2 if repetition else 0,
            rows=n // 2 if repetition == 2 else n,
            suffix=b"\xa5" if n else b"",
        )
        detail = levels.header
        if version == "v1":
            fields = ((1, n), (2, 0), (3, 3), (4, 3))
            kind = 0
        else:
            fields = (
                (1, n),
                (2, detail.num_nulls + int(malformed)),
                (3, detail.num_rows),
                (4, 0),
                (5, detail.definition_levels_byte_length),
                (6, detail.repetition_levels_byte_length),
            )
            kind = 3
        payload = levels.payload
        if malformed and version == "v1":
            payload = b"\x02\x00\x00\x00\x00\x01\xa5"
        header = PageHeaderAdmissionTests().wire(
            kind,
            detail=tuple((key, "i32", _grammar_integer(value)) for key, value in fields),
            compressed=len(payload),
            uncompressed=len(payload) + uncompressed_extra,
        )
        footer_api = FooterProfileTests()
        segment = header + payload
        rows = n // 2 if repetition == 2 else n
        metadata = footer_api.meta(
            values=n,
            data=64,
            compressed=len(segment),
            uncompressed=len(segment) + uncompressed_extra,
            encodings=(0, 3),
            codec=codec,
        )
        footer = footer_api.payload(
            (footer_api.root(1), footer_api.node(b"n", repetition=repetition)),
            (
                footer_api.group(
                    (footer_api.chunk(metadata),), rows=rows, size=len(segment) + uncompressed_extra
                ),
            ),
            rows=rows,
        )
        prefix = b"PAR1" + b"\x00" * 60 + segment
        return SimpleNamespace(
            payload=prefix + footer + len(footer).to_bytes(4, "little") + b"PAR1",
            footer=footer,
            footer_start=len(prefix),
        )

    def consume_owned_levels(self, owner, body):
        try:
            result = owner.consume_levels(body)
        except PinnedReadRefused as error:
            self.fail(f"admitted codec-0 data page refused level consumption: {error.reason_code}")
        self.assertIsInstance(result, bounded_parquet._OwnedPageLevels)
        return result

    def assert_level_only_delta(self, env, before, *, work, levels):
        after = self.counters(env)
        self.assertEqual(after.read, before.read)
        self.assertEqual(after.requests, before.requests)
        self.assertEqual(after.closes, before.closes)
        self.assertEqual(after.wire, before.wire)
        self.assertEqual(
            after.page,
            before.page
            | {
                "work_units": before.page["work_units"] + work,
                "levels": before.page["levels"] + levels,
            },
        )

    def test_owned_level_consumption_binds_flat_pages_and_preserves_framing(self):
        for version in ("v1", "v2"):
            with (
                self.subTest(version=version),
                self.environment(case=self.level_case(version)) as env,
            ):
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                result = self.consume_owned_levels(owner, body)
                self.assertIs(result.body, body)
                self.assertIs(owner.current_body, body)
                self.assertEqual(result.receipt_after, body.receipt_after)
                self.assertEqual(result.work_before, before.page["work_units"])
                self.assertEqual(result.work_after, before.page["work_units"] + 5)
                PageLevelScanTests().counts(result.summary, positions=8, non_nulls=8, rows=8)
                self.assertEqual(result.summary.repetition_span, (0, 0))
                self.assertEqual(result.summary.definition_span, (0, 0))
                self.assertEqual(result.summary.values_span, (0, len(body.payload)))
                self.assert_level_only_delta(env, before, work=5, levels=16)
                self.assertFalse(hasattr(result, "__dict__"))
                with self.assertRaises(FrozenInstanceError):
                    result.work_after = 0
                receipt = owner.finish(body)
                self.assertEqual((receipt.page_count, receipt.data_positions), (1, 8))
                self.assertEqual(receipt.status, "opaque_framing_provisional")
                self.assertIs(receipt.whole_source_digest_verified, False)
                self.assertFalse(env.profile.read_eligible)
                self.assertFalse(body.header.admission.decode_eligible)

    def test_owned_levels_allow_nonterminal_scan_and_successor_scan(self):
        with self.environment() as env:
            owner = self.start(env)
            first = owner.current_body
            first_result = self.consume_owned_levels(owner, first)
            self.assert_invalid(env, lambda: owner.consume_levels(first))
            second = owner.advance(first)
            self.assert_invalid(env, lambda: owner.consume_levels(first))
            before = self.counters(env)
            second_result = self.consume_owned_levels(owner, second)
            self.assertIs(second_result.body, second)
            self.assertIsNot(first_result, second_result)
            self.assert_level_only_delta(env, before, work=5, levels=2)
            receipt = owner.finish(second)
            self.assertEqual((receipt.page_count, receipt.data_positions), (2, 2))
            self.assert_invalid(env, lambda: owner.consume_levels(second))

    def test_owned_levels_scan_actual_nullable_and_repeated_level_streams(self):
        for version, repetition, end, work, rows in (
            ("v1", 1, 6, 36, 8),
            ("v2", 1, 2, 32, 8),
            ("v1", 2, 12, 44, 4),
            ("v2", 2, 4, 36, 4),
        ):
            with (
                self.subTest(version=version, repetition=repetition),
                self.environment(case=self.level_case(version, repetition=repetition)) as env,
            ):
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                result = self.consume_owned_levels(owner, body)
                rep_end = end // 2 if repetition == 2 else 0
                self.assertEqual(result.summary.repetition_span, (0, rep_end))
                self.assertEqual(result.summary.definition_span, (rep_end, end))
                self.assertEqual(result.summary.values_span, (end, end + 1))
                PageLevelScanTests().counts(result.summary, non_nulls=4, rows=rows)
                self.assert_level_only_delta(env, before, work=work, levels=16)
                self.assertEqual(owner.finish(body).data_positions, 8)

    def test_owned_levels_preserve_zero_position_stream_boundaries(self):
        for version, repetition, end, work in (
            ("v1", 0, 0, 5),
            ("v2", 0, 0, 5),
            ("v1", 1, 4, 9),
            ("v2", 1, 0, 5),
        ):
            with (
                self.subTest(version=version, repetition=repetition),
                self.environment(case=self.level_case(version, repetition=repetition, n=0)) as env,
            ):
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                result = self.consume_owned_levels(owner, body)
                PageLevelScanTests().counts(
                    result.summary, positions=0, non_nulls=0, rows=0, first=None
                )
                self.assertEqual(result.summary.definition_span, (0, end))
                self.assertEqual(result.summary.values_span, (end, end))
                self.assert_level_only_delta(env, before, work=work, levels=0)
                self.assertEqual(owner.finish(body).data_positions, 0)

    def test_owned_levels_reject_foreign_cloned_and_repeated_bodies_without_charge(self):
        with self.environment() as env, self.environment() as foreign:
            owner = self.start(env)
            body = owner.current_body
            foreign_body = self.start(foreign).current_body
            for supplied in (object(), replace(body), foreign_body):
                with self.subTest(body_type=type(supplied).__name__):
                    self.assert_invalid(
                        env, lambda supplied=supplied: owner.consume_levels(supplied)
                    )
            self.assert_invalid(env, lambda: type(owner).consume_levels(object(), body))
            self.consume_owned_levels(owner, body)
            self.assert_invalid(env, lambda: owner.consume_levels(body))
            self.assertIs(owner.current_body, body)
            second = owner.advance(body)
            self.assertEqual(owner.finish(second).page_count, 2)

    def test_owned_levels_unsupported_dictionary_and_codec_keep_framing_usable(self):
        cases = (self.mixed_case(), self.level_case(codec=1))
        for index, case in enumerate(cases):
            with self.subTest(case=index), self.environment(case=case) as env:
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                with self.assertRaises(PinnedReadRefused) as caught:
                    owner.consume_levels(body)
                self.assertEqual(caught.exception.reason_code, "unsupported_mode")
                self.assertEqual(self.counters(env), before)
                self.assertIs(owner.current_body, body)
                if index == 0:
                    body = owner.advance(body)
                    body = owner.advance(body)
                receipt = owner.finish(body)
                self.assertEqual(receipt.status, "opaque_framing_provisional")

    def test_owned_levels_reject_external_work_read_or_closed_source_before_admission(self):
        for fault in ("work", "read", "closed", "scanner"):
            with self.subTest(fault=fault), self.environment() as env:
                owner = self.start(env)
                body = owner.current_body
                if fault == "work":
                    env.ledgers.page.reserve(bounded_parquet._PageCharge(work_units=1))
                elif fault == "read":
                    env.io.module._read_pinned_range(env.source, offset=0, length=1)
                elif fault == "scanner":
                    section = bounded_parquet._PageLevelInput(
                        body.payload, 0, len(body.payload), body.header.admission.detail, 0, 0
                    )
                    bounded_parquet._scan_page_levels(section, budget=env.ledgers.page)
                else:
                    env.source._closed = True
                try:
                    self.assert_invalid(
                        env, lambda owner=owner, body=body: owner.consume_levels(body)
                    )
                    self.assertIs(owner.current_body, body)
                finally:
                    if fault == "closed":
                        env.source._closed = False

    def test_owned_levels_external_work_after_success_still_invalidates_transition(self):
        with self.environment() as env:
            owner = self.start(env)
            body = owner.current_body
            self.consume_owned_levels(owner, body)
            env.ledgers.page.reserve(bounded_parquet._PageCharge(work_units=1))
            self.assert_invalid(env, lambda: owner.advance(body))
            self.assert_invalid(env, lambda: owner.consume_levels(body))

    def test_owned_levels_malformed_wire_and_v2_counts_poison_without_refund(self):
        for version, work, levels, reason in (
            ("v1", 9, 16, "schema_mismatch"),
            ("v2", 31, 16, "schema_mismatch"),
        ):
            with (
                self.subTest(version=version),
                self.environment(
                    case=self.level_case(version, repetition=1, malformed=True)
                ) as env,
            ):
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                with self.assertRaises(bounded_parquet._CompactWireMalformed) as caught:
                    owner.consume_levels(body)
                self.assertEqual(caught.exception.reason_code, reason)
                self.assert_level_only_delta(env, before, work=work, levels=levels)
                self.assert_poisoned(env, owner, body)
                self.assert_invalid(env, lambda owner=owner, body=body: owner.consume_levels(body))

    def test_owned_levels_budget_failures_preserve_charged_prefix_and_poison(self):
        for allowance, spent, levels in ((0, 0, 0), (1, 1, 0), (2, 2, 2), (4, 4, 2)):
            with self.subTest(allowance=allowance), self.environment() as env:
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                env.ledgers.page.limits = replace(
                    env.ledgers.page.limits,
                    max_work_units=before.page["work_units"] + allowance,
                )
                with self.assertRaises(PinnedReadRefused) as caught:
                    owner.consume_levels(body)
                self.assertEqual(caught.exception.reason_code, "resource_limit")
                self.assert_level_only_delta(env, before, work=spent, levels=levels)
                self.assert_poisoned(env, owner, body)

    def test_owned_levels_exact_budget_succeeds_and_one_below_levels_poison(self):
        for available in (1, 2):
            with self.subTest(levels=available), self.environment() as env:
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                env.ledgers.page.limits = replace(
                    env.ledgers.page.limits,
                    max_work_units=before.page["work_units"] + 5,
                    max_levels=before.page["levels"] + available,
                )
                if available == 1:
                    with self.assertRaises(PinnedReadRefused) as caught:
                        owner.consume_levels(body)
                    self.assertEqual(caught.exception.reason_code, "resource_limit")
                    self.assert_level_only_delta(env, before, work=1, levels=0)
                    self.assert_poisoned(env, owner, body)
                else:
                    self.consume_owned_levels(owner, body)
                    self.assert_level_only_delta(env, before, work=5, levels=2)

    def test_owned_levels_deadline_failure_poison_preserves_actual_work_origin(self):
        for spent in (0, 1, 4):
            with self.subTest(spent=spent), self.environment() as env:
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                original_clock = env.ledgers.read.clock_ns
                target = before.page["work_units"] + spent

                def deadline(clock=original_clock, budget=env.ledgers.page, target=target):
                    stamp = clock()
                    if budget.work_units == target:
                        return budget.read_budget.limits.deadline_ns
                    return stamp

                env.ledgers.read.clock_ns = deadline
                try:
                    with self.assertRaises(PinnedReadRefused) as caught:
                        owner.consume_levels(body)
                    self.assertEqual(caught.exception.reason_code, "deadline_exceeded")
                    self.assert_level_only_delta(
                        env, before, work=spent, levels=2 if spent >= 2 else 0
                    )
                    self.assert_poisoned(env, owner, body)
                finally:
                    env.ledgers.read.clock_ns = original_clock

    def test_owned_levels_final_liveness_and_receipt_check_prevent_publication(self):
        for fault in ("closed", "read"):
            with self.subTest(fault=fault), self.environment() as env:
                owner = self.start(env)
                body = owner.current_body
                before = self.counters(env)
                original_clock = env.ledgers.read.clock_ns
                fired = []
                target = before.page["work_units"] + 4

                def invalidate(
                    clock=original_clock, target=target, env=env, fired=fired, fault=fault
                ):
                    stamp = clock()
                    if not fired and env.ledgers.page.work_units == target:
                        fired.append(fault)
                        if fault == "closed":
                            env.source._closed = True
                        else:
                            env.io.module._read_pinned_range(env.source, offset=0, length=1)
                    return stamp

                env.ledgers.read.clock_ns = invalidate
                try:
                    with self.assertRaises(PinnedReadRefused) as caught:
                        owner.consume_levels(body)
                    self.assertEqual(caught.exception.reason_code, "invalid_contract")
                    self.assertEqual(fired, [fault])
                    self.assertEqual(env.ledgers.page.work_units, before.page["work_units"] + 5)
                    self.assert_poisoned(env, owner, body)
                finally:
                    env.ledgers.read.clock_ns = original_clock
                    env.source._closed = False

    def test_owned_levels_uncompressed_size_mismatch_poison_before_scanner(self):
        with self.environment(case=self.level_case(uncompressed_extra=1)) as env:
            owner = self.start(env)
            body = owner.current_body
            before = self.counters(env)
            with self.assertRaises(bounded_parquet._CompactWireMalformed) as caught:
                owner.consume_levels(body)
            self.assertEqual(caught.exception.reason_code, "schema_mismatch")
            self.assert_level_only_delta(env, before, work=1, levels=0)
            self.assert_poisoned(env, owner, body)

    def test_owned_levels_reuse_scanner_with_exact_admitted_body_and_leaf(self):
        with self.environment(case=self.level_case("v2", repetition=2)) as env:
            owner = self.start(env)
            body = owner.current_body
            before = self.counters(env)
            actual_scan = bounded_parquet._scan_page_levels
            observed = []

            def observe(section, *, budget):
                summary = actual_scan(section, budget=budget)
                observed.append((section, budget, summary))
                return summary

            with patch.object(bounded_parquet, "_scan_page_levels", side_effect=observe):
                result = self.consume_owned_levels(owner, body)
            self.assertEqual(len(observed), 1)
            section, budget, summary = observed[0]
            self.assertIs(section.payload, body.payload)
            self.assertEqual((section.start, section.stop), (0, len(body.payload)))
            self.assertIs(section.header, body.header.admission.detail)
            self.assertEqual(section.max_definition_level, body.header.leaf.definition_depth)
            self.assertEqual(section.max_repetition_level, body.header.leaf.repetition_depth)
            self.assertIs(budget, env.ledgers.page)
            self.assertIs(result.summary, summary)
            self.assert_level_only_delta(env, before, work=36, levels=16)
            self.assertEqual(owner.finish(body).data_positions, 8)
