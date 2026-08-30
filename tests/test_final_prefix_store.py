"""The final-prefix store: one owner for the checkpoint and its counters.

The status transitions were written out by hand at eight sites before the
extraction -- "ready" twice, "not ready" three times, "unused" four -- which is
how the counters drift away from the checkpoint they describe.

These execute the real store. Native capture and restore stay in the client, so
nothing here touches `ctx_tgt` or `lib`: the client calls the primitive and
hands the result over.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.native_llama.final_prefix_store import (
    FinalPrefixExperimentStatus,
    FinalPrefixStore,
)
from orbit.native_llama.prefix_anchor import PrefixAnchorState


class ReadyTransitionTests(unittest.TestCase):
    def test_mark_ready_records_residency_and_clears_the_reason(self) -> None:
        store = FinalPrefixStore()
        store.record_fallback("earlier failure")

        store.mark_ready(64, captured=True)

        self.assertTrue(store.status.initialized)
        self.assertEqual(store.status.prefix_tokens, 64)
        self.assertTrue(store.status.last_used)
        self.assertIsNone(
            store.status.failure_reason,
            "a successful capture must clear the previous failure reason",
        )

    def test_capture_and_restore_increment_separate_counters(self) -> None:
        """A restore must not inflate the capture total, or vice versa.

        These are reported separately: conflating them would misreport how the
        prefix was obtained.
        """
        store = FinalPrefixStore()

        store.mark_ready(10, captured=True)
        store.mark_ready(10, restored=True)
        store.mark_ready(10, restored=True)

        self.assertEqual(store.status.capture_count, 1)
        self.assertEqual(store.status.restore_count, 2)

    def test_mark_ready_without_a_flag_counts_neither(self) -> None:
        store = FinalPrefixStore()
        store.mark_ready(8)
        self.assertEqual(store.status.capture_count, 0)
        self.assertEqual(store.status.restore_count, 0)
        self.assertTrue(store.status.initialized)


class NotReadyTransitionTests(unittest.TestCase):
    def test_mark_not_ready_clears_residency_only(self) -> None:
        """It must NOT count a fallback: the caller records that separately.

        Counting here would double-count every failure, since both paths that
        use it follow immediately with `record_fallback`.
        """
        store = FinalPrefixStore()
        store.mark_ready(32, captured=True)

        store.mark_not_ready()

        self.assertFalse(store.status.initialized)
        self.assertEqual(store.status.prefix_tokens, 0)
        self.assertEqual(
            store.status.fallback_count, 0,
            "mark_not_ready must not count a fallback; its callers do",
        )

    def test_record_fallback_counts_once_and_marks_unused(self) -> None:
        store = FinalPrefixStore()
        store.mark_ready(16, captured=True)

        store.record_fallback("restore_failed")

        self.assertEqual(store.status.fallback_count, 1)
        self.assertEqual(store.status.failure_reason, "restore_failed")
        self.assertFalse(store.status.last_used)

    def test_the_failure_pair_counts_exactly_one_fallback(self) -> None:
        """The shipped sequence: mark_not_ready then record_fallback."""
        store = FinalPrefixStore()
        store.mark_ready(16, restored=True)

        store.mark_not_ready()
        store.record_fallback("capture_failed")

        self.assertEqual(store.status.fallback_count, 1)
        self.assertFalse(store.status.initialized)


class UnusedTests(unittest.TestCase):
    def test_mark_unused_changes_nothing_else(self) -> None:
        """A turn that did not use the prefix leaves the checkpoint valid.

        The counters describe history; the checkpoint may still be perfectly
        restorable next turn.
        """
        store = FinalPrefixStore()
        store.mark_ready(24, captured=True)
        anchor_before = store.anchor

        store.mark_unused()

        self.assertFalse(store.status.last_used)
        self.assertTrue(
            store.status.initialized,
            "an unused turn must not un-initialize a valid prefix",
        )
        self.assertEqual(store.status.prefix_tokens, 24)
        self.assertEqual(store.status.capture_count, 1)
        self.assertIs(store.anchor, anchor_before)

    def test_mark_unused_does_not_clear_a_failure_reason(self) -> None:
        """A turn that skipped the prefix must not erase why it failed before.

        `final_prefix_experiment_status()` reports `failure_reason`, so
        clearing it here would make a real fallback look like it never
        happened -- the diagnostic disappears while the counter still says one
        fallback occurred.
        """
        store = FinalPrefixStore()
        store.record_fallback("restore_failed")

        store.mark_unused()

        self.assertEqual(
            store.status.failure_reason, "restore_failed",
            "mark_unused must leave the recorded failure intact",
        )
        self.assertEqual(store.status.fallback_count, 1)


class InvalidationTests(unittest.TestCase):
    def test_invalidate_drops_the_checkpoint_entirely(self) -> None:
        """A flagged-but-present checkpoint could still be restored later.

        Starts from a genuinely VALID anchor: an earlier version of this test
        began from an empty one, which is already invalid, so it could not tell
        a real drop from doing nothing.
        """
        store = FinalPrefixStore()
        store.anchor = PrefixAnchorState(
            prefix_hash="h", token_count=48, valid=True, checkpoint_data=b"kv"
        )
        store.mark_ready(48, captured=True)
        self.assertTrue(store.anchor.valid, "precondition: a real checkpoint")

        store.invalidate("session_reset")

        self.assertFalse(
            store.anchor.valid,
            "the checkpoint must be replaced, not merely flagged; a surviving "
            "valid anchor could still be restored on a later turn",
        )
        self.assertIsNone(store.anchor.checkpoint_data)
        self.assertFalse(store.status.initialized)
        self.assertEqual(store.status.prefix_tokens, 0)
        self.assertEqual(store.status.failure_reason, "session_reset")
        self.assertFalse(store.status.last_used)

    def test_invalidate_does_not_count_a_fallback(self) -> None:
        """Invalidation is not a fallback; conflating them skews the totals."""
        store = FinalPrefixStore()
        store.mark_ready(8, captured=True)

        store.invalidate("route_change")

        self.assertEqual(store.status.fallback_count, 0)

    def test_repeated_invalidation_is_safe(self) -> None:
        store = FinalPrefixStore()
        store.invalidate("first")
        store.invalidate("second")
        self.assertFalse(store.anchor.valid)
        self.assertEqual(store.status.failure_reason, "second")

    def test_counters_survive_invalidation(self) -> None:
        """History is not erased by dropping the current checkpoint."""
        store = FinalPrefixStore()
        store.mark_ready(8, captured=True)
        store.mark_ready(8, restored=True)

        store.invalidate("gone")

        self.assertEqual(store.status.capture_count, 1)
        self.assertEqual(store.status.restore_count, 1)


class FailureAtomicityTests(unittest.TestCase):
    """A failed native operation must not leave a reusable checkpoint."""

    def test_a_failed_capture_leaves_no_valid_anchor(self) -> None:
        store = FinalPrefixStore()
        store.mark_not_ready()
        store.record_fallback("capture_exception")

        self.assertFalse(store.anchor.valid)
        self.assertFalse(store.status.initialized)

    def test_a_failed_restore_after_a_valid_state_drops_residency(self) -> None:
        """The prior checkpoint must not stay marked resident."""
        store = FinalPrefixStore()
        store.mark_ready(64, captured=True)

        store.mark_not_ready()
        store.record_fallback("restore_failed")

        self.assertFalse(store.status.initialized)
        self.assertEqual(store.status.prefix_tokens, 0)
        self.assertEqual(store.status.failure_reason, "restore_failed")

    def test_recovery_after_failure_clears_the_reason(self) -> None:
        store = FinalPrefixStore()
        store.record_fallback("transient")

        store.mark_ready(12, captured=True)

        self.assertIsNone(store.status.failure_reason)
        self.assertTrue(store.status.initialized)


class ClientOwnershipTests(unittest.TestCase):
    """The client projects the store; it holds no second copy."""

    def _client(self):
        from orbit.native_llama.client import NativeLlamaClient

        return object.__new__(NativeLlamaClient)

    def test_a_bare_client_reads_as_no_prefix(self) -> None:
        """`object.__new__` never runs `__init__`; absence must be empty."""
        client = self._client()
        self.assertFalse(client._final_prefix_anchor_state.valid)
        self.assertFalse(client._final_prefix_status.initialized)

    def test_client_transitions_reach_the_store(self) -> None:
        client = self._client()
        client._record_final_prefix_fallback("boom")

        self.assertEqual(client._final_prefix_status.fallback_count, 1)
        self.assertEqual(client._final_prefix_store().status.fallback_count, 1)
        self.assertIs(
            client._final_prefix_status, client._final_prefix_store().status,
            "the client must project the store, not copy it",
        )

    def test_the_restore_site_counts_a_restore_not_a_capture(self) -> None:
        """The call sites must pick the right counter, not just `mark_ready`.

        `mark_ready` splits capture from restore, but nothing here pinned that
        the RESTORE path passes `restored=True`: a site delegating with
        `captured=True` was caught only by a pre-existing probe test. This
        drives the shipped restore branch and checks which total moved.
        """
        import ast
        import inspect

        from orbit.native_llama.client import NativeLlamaClient

        source = inspect.getsource(
            NativeLlamaClient._prepare_memory_with_final_prefix
        )
        tree = ast.parse(source.lstrip()).body[0]
        calls = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and "mark_ready" in ast.unparse(node)
        ]
        self.assertEqual(len(calls), 2, "expected one capture and one restore site")
        self.assertEqual(
            sum("restored=True" in c for c in calls), 1,
            "exactly one site must count a restore",
        )
        self.assertEqual(
            sum("captured=True" in c for c in calls), 1,
            "exactly one site must count a capture; a restore counted as a "
            "capture misreports how the prefix was obtained",
        )

    def test_client_invalidation_reaches_the_store(self) -> None:
        client = self._client()
        client._final_prefix_store().mark_ready(20, captured=True)

        client._invalidate_final_prefix("session_reset")

        self.assertFalse(client._final_prefix_anchor_state.valid)
        self.assertFalse(client._final_prefix_status.initialized)
        self.assertEqual(client._final_prefix_status.failure_reason, "session_reset")

    def test_assigning_the_status_replaces_the_owners_copy(self) -> None:
        """Whole-object assignment worked before the extraction."""
        client = self._client()
        replacement = FinalPrefixExperimentStatus(capture_count=7)

        client._final_prefix_status = replacement

        self.assertIs(client._final_prefix_status, replacement)
        self.assertIs(client._final_prefix_store().status, replacement)

    def test_there_is_exactly_one_status_class(self) -> None:
        """`client.FinalPrefixExperimentStatus` must BE the store's class.

        The extraction briefly left two same-named dataclasses -- one still
        defined in client.py, one in the store -- so the status the client
        actually holds failed `isinstance` against the name importers use.
        Assignment-only tests could not see it; a comparison would break.
        """
        from orbit.native_llama import client as client_module
        from orbit.native_llama import final_prefix_store

        self.assertIs(
            client_module.FinalPrefixExperimentStatus,
            final_prefix_store.FinalPrefixExperimentStatus,
            "two same-named status classes make isinstance fail for importers",
        )
        client = self._client()
        self.assertIsInstance(
            client._final_prefix_status,
            client_module.FinalPrefixExperimentStatus,
        )

    def test_the_store_never_touches_native_state(self) -> None:
        """No ctx_tgt, no lib: capture and restore stay client-side.

        Checked over parsed code rather than raw text: the module docstring
        names these very things to say it does NOT use them.
        """
        import ast

        tree = ast.parse((SRC / "orbit/native_llama/final_prefix_store.py").read_text())
        names = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        for forbidden in ("ctx_tgt", "lib", "ctypes", "capture_prefix_anchor",
                          "restore_prefix_anchor"):
            self.assertNotIn(
                forbidden, names,
                f"the store must not reach native state ({forbidden})",
            )


if __name__ == "__main__":
    unittest.main()
