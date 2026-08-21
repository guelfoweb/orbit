"""Rebuilt final/retry windows must remain parseable conversations.

A compacted window that opens with the assistant is not a conversation the
admission validator can parse: the turn fails before inference and the user
gets nothing. This was reached on the second turn of an ordinary chat, where
the retry rebuild placed the retained assistant excerpt ahead of the user
message it was answering.

These drive the real builders and validate their output with the production
parser rather than comparing against a hand-written list, so the tests fail for
the same reason the runtime did.
"""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from orbit.backend.base import TokenCount
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.context_manager import _parse_turns
from orbit.runtime.evidence import EvidenceStore


class _CountingBackend:
    context_tokens = 16384
    thinking = False

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        digest = hashlib.sha256(b"stable").hexdigest()
        return TokenCount(
            tokens=50, context_tokens=16384, rendered_hash=digest, token_hash=digest
        )


TWO_TURN_HISTORY = [
    {"role": "system", "content": "You are Orbit."},
    {"role": "user", "content": "hi, how are you?"},
    {"role": "assistant", "content": "Hi! I'm doing well, thanks for asking!"},
    {"role": "user", "content": "who designed you?"},
]


def _runtime(store=None, history=None):
    return ChatRuntime(
        backend=_CountingBackend(),
        system_prompt="You are Orbit.",
        messages=[dict(m) for m in (history or TWO_TURN_HISTORY)],
        context_tokens=16384,
        evidence_store=store,
    )


def _seeded_store(tmp: str) -> EvidenceStore:
    store = EvidenceStore(Path(tmp) / "evidence")
    store.add(
        "exec_shell_full_command",
        "recorded output " + "p" * 120,
        metadata={
            "tool_call_id": "call-1",
            "user_turn_id": "turn-1",
            "produced_by_phase": "tool_call",
        },
    )
    return store


class RebuiltWindowsParseTests(unittest.TestCase):
    """Every rebuilt window must survive the admission parser."""

    def _assert_parses(self, messages, label: str) -> None:
        try:
            _parse_turns(messages)
        except ValueError as exc:
            roles = [m.get("role") for m in messages]
            self.fail(f"{label} produced an unparseable window {roles}: {exc}")

    def test_retry_window_parses_without_evidence(self) -> None:
        built = _runtime()._chat_final_retry_messages()
        self._assert_parses(built, "_chat_final_retry_messages")

    def test_retry_window_parses_with_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = _runtime(_seeded_store(tmp))._chat_final_retry_messages()
            self._assert_parses(built, "_chat_final_retry_messages")

    def test_final_window_parses_with_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = _runtime(_seeded_store(tmp))._chat_final_messages()
            self._assert_parses(built, "_chat_final_messages")

    def test_completion_repair_window_parses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = _runtime(_seeded_store(tmp)).chat_final_completion_repair_messages(
                "Answer the user directly now."
            )
            self.assertIsNotNone(built, "repair window unexpectedly unavailable")
            assert built is not None
            self._assert_parses(built, "chat_final_completion_repair_messages")


class OrderingTests(unittest.TestCase):
    """The user turn must precede the assistant reply it prompted."""

    def _first_roles(self, messages):
        return [m.get("role") for m in messages if m.get("role") in {"user", "assistant"}]

    def test_retry_puts_user_before_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = _runtime(_seeded_store(tmp))._chat_final_retry_messages()
            roles = self._first_roles(built)
            self.assertEqual(roles[:2], ["user", "assistant"], f"got {roles}")

    def test_final_puts_user_before_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = _runtime(_seeded_store(tmp))._chat_final_messages()
            roles = self._first_roles(built)
            self.assertEqual(roles[:2], ["user", "assistant"], f"got {roles}")

    def test_repair_puts_user_before_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = _runtime(_seeded_store(tmp)).chat_final_completion_repair_messages(
                "Answer now."
            )
            assert built is not None
            roles = self._first_roles(built)
            self.assertEqual(roles[:2], ["user", "assistant"], f"got {roles}")


class SelectionUnchangedTests(unittest.TestCase):
    """Reordering must not change which messages are selected or their content."""

    def test_same_messages_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = _runtime(_seeded_store(tmp))._chat_final_retry_messages()
            contents = {str(m.get("content")) for m in built}
            self.assertIn("who designed you?", contents, "latest user turn dropped")
            self.assertTrue(
                any("doing well" in c for c in contents), "assistant reply dropped"
            )

    def test_no_message_is_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            built = _runtime(_seeded_store(tmp))._chat_final_retry_messages()
            user_turns = [m for m in built if m.get("role") == "user"]
            self.assertEqual(len(user_turns), 1, "user turn duplicated")

    def test_input_history_is_not_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _runtime(_seeded_store(tmp))
            before = copy.deepcopy(runtime.messages)
            runtime._chat_final_retry_messages()
            runtime._chat_final_messages()
            runtime.chat_final_completion_repair_messages("Answer now.")
            self.assertEqual(runtime.messages, before, "builder mutated history")

    def test_first_turn_history_still_parses(self) -> None:
        """A single-turn history must remain valid after the reorder."""
        history = [
            {"role": "system", "content": "You are Orbit."},
            {"role": "user", "content": "hi, how are you?"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            built = _runtime(_seeded_store(tmp), history)._chat_final_retry_messages()
            _parse_turns(built)


if __name__ == "__main__":
    unittest.main()
