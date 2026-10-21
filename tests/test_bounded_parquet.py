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


if __name__ == "__main__":
    unittest.main()
