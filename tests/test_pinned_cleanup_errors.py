"""Neutral regression cases for retaining pinned-source cleanup failures."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from liq.store import timestamp_metadata


def _exception_state(error: BaseException) -> dict[object, object]:
    return BaseException.__dict__["__dict__"].__get__(error, BaseException)


def _retained_errors(error: BaseException) -> tuple[BaseException, ...]:
    key = getattr(timestamp_metadata, "_PINNED_CLEANUP_ERRORS", None)
    if key is None:
        raise AssertionError("ordered cleanup retention is missing")
    retained = _exception_state(error).get(key)
    if type(retained) is not tuple:
        raise AssertionError("ordered cleanup retention is missing")
    return retained


def _assert_same_sequence(
    actual: tuple[BaseException, ...], expected: tuple[BaseException, ...]
) -> None:
    if len(actual) != len(expected) or any(
        item is not reference for item, reference in zip(actual, expected, strict=True)
    ):
        raise AssertionError("cleanup failure identities or order changed")


def _exercise_overlap(case: str) -> None:
    primary = RuntimeError("neutral primary")
    cleanup = OSError("neutral cleanup")
    context = RuntimeError("neutral context")

    if case == "cleanup_causes_primary":
        cleanup.__cause__ = primary
        events = [cleanup]
    elif case == "primary_causes_cleanup":
        primary.__cause__ = cleanup
        events = [cleanup]
    elif case == "duplicate_cleanup":
        events = [cleanup, cleanup]
    elif case == "self_alias":
        events = [primary]
    elif case == "context_backedge":
        cleanup.__context__ = primary
        events = [cleanup]
    elif case == "preexisting_cycle":
        primary.__cause__ = context
        context.__context__ = primary
        events = [cleanup]
    else:
        raise AssertionError("unknown overlap case")

    cause_before = primary.__cause__
    context_before = primary.__context__
    cleanup_cause_before = cleanup.__cause__
    cleanup_context_before = cleanup.__context__
    timestamp_metadata._retain_pinned_cleanup_error(primary, events)

    if primary.__cause__ is not cause_before or primary.__context__ is not context_before:
        raise AssertionError("primary's incoming exception graph changed")
    if cleanup.__cause__ is not cleanup_cause_before:
        raise AssertionError("cleanup's pre-existing cause changed")
    if cleanup.__context__ is not cleanup_context_before:
        raise AssertionError("cleanup's pre-existing context changed")
    if case == "preexisting_cycle" and context.__context__ is not primary:
        raise AssertionError("pre-existing cycle changed")
    _assert_same_sequence(_retained_errors(primary), tuple(events))


def _base_link(error: BaseException, name: str) -> BaseException | None:
    return BaseException.__dict__[name].__get__(error, BaseException)


def _exercise_hostile_graph(case: str) -> None:
    class HostileGraphError(RuntimeError):
        def __getattribute__(self, name: str):
            if name in {"__cause__", "__context__", "__dict__", "_pinned_cleanup_errors"}:
                raise AttributeError(name)
            return super().__getattribute__(name)

        def __setattr__(self, name: str, value: object) -> None:
            if name in {"__cause__", "__context__", "__dict__", "_pinned_cleanup_errors"}:
                raise AttributeError(name)
            super().__setattr__(name, value)

    primary = HostileGraphError("neutral primary")
    first = HostileGraphError("neutral first cleanup")
    second = HostileGraphError("neutral second cleanup")
    original_primary = primary
    if case == "hostile_fresh":
        timestamp_metadata._retain_pinned_cleanup_error(primary, [first, second])
        if (
            _base_link(primary, "__cause__") is not first
            or _base_link(first, "__cause__") is not second
            or _base_link(second, "__cause__") is not None
            or any(
                _base_link(error, "__context__") is not None for error in (primary, first, second)
            )
        ):
            raise AssertionError("fresh cause chain did not use the base exception slots")
    elif case == "hostile_context":
        context = HostileGraphError("neutral existing context")
        context_slot = BaseException.__dict__["__context__"]
        context_slot.__set__(primary, context)
        context_slot.__set__(context, first)
        context_slot.__set__(first, second)
        participants = (primary, context, first, second)
        before = tuple(
            (_base_link(error, "__cause__"), _base_link(error, "__context__"))
            for error in participants
        )
        timestamp_metadata._retain_pinned_cleanup_error(primary, [first, second])
        after = tuple(
            (_base_link(error, "__cause__"), _base_link(error, "__context__"))
            for error in participants
        )
        if any(
            actual_link is not original_link
            for actual, original in zip(after, before, strict=True)
            for actual_link, original_link in zip(actual, original, strict=True)
        ):
            raise AssertionError("pre-existing context graph changed")
    else:
        raise AssertionError("unknown hostile graph case")
    if primary is not original_primary or type(primary) is not HostileGraphError:
        raise AssertionError("primary exception identity or type changed")
    _assert_same_sequence(_retained_errors(primary), (first, second))


class PinnedCleanupErrorTests(unittest.TestCase):
    def _bounded_overlap(self, case: str) -> None:
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), case],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self.fail(f"cleanup retention did not terminate for {case}")
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_fresh_distinct_cleanup_errors_keep_ordered_cause_chain(self) -> None:
        primary = RuntimeError("neutral primary")
        first = OSError("neutral first close")
        second = OSError("neutral second close")

        timestamp_metadata._retain_pinned_cleanup_error(primary, [first, second])

        self.assertIs(primary.__cause__, first)
        self.assertIs(first.__cause__, second)
        self.assertIsNone(second.__cause__)
        _assert_same_sequence(_retained_errors(primary), (first, second))

    def test_no_cleanup_errors_leave_exception_unchanged(self) -> None:
        primary = RuntimeError("neutral primary")
        cause = OSError("neutral original cause")
        context = RuntimeError("neutral original context")
        primary.__cause__ = cause
        primary.__context__ = context
        state_before = _exception_state(primary).copy()

        timestamp_metadata._retain_pinned_cleanup_error(primary, [])

        self.assertIs(primary.__cause__, cause)
        self.assertIs(primary.__context__, context)
        self.assertEqual(_exception_state(primary), state_before)

    def test_existing_cause_and_context_links_remain_unchanged(self) -> None:
        primary = RuntimeError("neutral primary")
        original_cause = OSError("neutral original cause")
        original_context = RuntimeError("neutral original context")
        cleanup = OSError("neutral cleanup")
        cleanup_cause = OSError("neutral cleanup cause")
        cleanup_context = RuntimeError("neutral cleanup context")
        primary.__cause__ = original_cause
        primary.__context__ = original_context
        cleanup.__cause__ = cleanup_cause
        cleanup.__context__ = cleanup_context

        timestamp_metadata._retain_pinned_cleanup_error(primary, [cleanup])

        self.assertIs(primary.__cause__, original_cause)
        self.assertIs(cleanup.__cause__, cleanup_cause)
        self.assertIsNone(cleanup_cause.__cause__)
        self.assertIsNone(original_cause.__cause__)
        self.assertIs(primary.__context__, original_context)
        self.assertIs(cleanup.__context__, cleanup_context)
        _assert_same_sequence(_retained_errors(primary), (cleanup,))

    def test_repeated_invocation_appends_events_without_rewriting_cause_chain(self) -> None:
        primary = RuntimeError("neutral primary")
        first = OSError("neutral first cleanup")
        second = OSError("neutral later cleanup")

        timestamp_metadata._retain_pinned_cleanup_error(primary, [first])
        timestamp_metadata._retain_pinned_cleanup_error(primary, [second])

        self.assertIs(primary.__cause__, first)
        self.assertIsNone(first.__cause__)
        self.assertIsNone(second.__cause__)
        _assert_same_sequence(_retained_errors(primary), (first, second))

    def test_cleanup_cause_backedge_terminates_without_rewriting_links(self) -> None:
        self._bounded_overlap("cleanup_causes_primary")

    def test_existing_primary_cause_alias_terminates_without_rewriting_links(self) -> None:
        self._bounded_overlap("primary_causes_cleanup")

    def test_duplicate_cleanup_event_terminates_and_keeps_both_events(self) -> None:
        self._bounded_overlap("duplicate_cleanup")

    def test_primary_self_alias_terminates_without_creating_cause_cycle(self) -> None:
        self._bounded_overlap("self_alias")

    def test_context_backedge_terminates_without_rewriting_context(self) -> None:
        self._bounded_overlap("context_backedge")

    def test_preexisting_cycle_terminates_without_rewriting_links(self) -> None:
        self._bounded_overlap("preexisting_cycle")

    def test_hostile_subclass_cannot_block_private_event_retention(self) -> None:
        class HostileError(RuntimeError):
            def __getattribute__(self, name: str):
                if name in {"__dict__", "_pinned_cleanup_errors"}:
                    raise AttributeError(name)
                return super().__getattribute__(name)

            def __setattr__(self, name: str, value: object) -> None:
                if name in {"__dict__", "_pinned_cleanup_errors"}:
                    raise AttributeError(name)
                super().__setattr__(name, value)

        primary = HostileError("neutral primary")
        cleanup = OSError("neutral cleanup")

        timestamp_metadata._retain_pinned_cleanup_error(primary, [cleanup])

        self.assertIs(primary.__cause__, cleanup)
        _assert_same_sequence(_retained_errors(primary), (cleanup,))

    def test_foreign_named_private_attribute_is_preserved(self) -> None:
        primary = RuntimeError("neutral primary")
        primary._pinned_cleanup_errors = "foreign value"
        cleanup = OSError("neutral cleanup")

        timestamp_metadata._retain_pinned_cleanup_error(primary, [cleanup])

        self.assertEqual(primary._pinned_cleanup_errors, "foreign value")
        _assert_same_sequence(_retained_errors(primary), (cleanup,))

    def test_hostile_graph_hooks_allow_fresh_ordered_cause_chain(self) -> None:
        self._bounded_overlap("hostile_fresh")

    def test_hostile_graph_hooks_preserve_existing_context_graph(self) -> None:
        self._bounded_overlap("hostile_context")


if __name__ == "__main__":
    case = sys.argv[1]
    if case.startswith("hostile_"):
        _exercise_hostile_graph(case)
    else:
        _exercise_overlap(case)
