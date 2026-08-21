"""A stored session must be resumable, or it must not be offered.

Message-shape validation accepted histories the runtime cannot parse: an
assistant reply before any user turn, a tool result with no matching call.
Loading one succeeded, and the failure surfaced later as a context-admission
error on the next prompt -- a conversation the user had watched work.

Structure is now checked at load against the same grammar admission uses, so
an unusable session is reported when it is read rather than when it is next
used. The file itself is never touched: it is the user's data, and a history
rewritten to make it parse would hide the corruption instead of reporting it.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from orbit.runtime.context_manager import conversation_structure_error
from orbit.runtime.sessions import SessionStore

CALL = {"id": "call-1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
CALL2 = {"id": "call-2", "type": "function", "function": {"name": "g", "arguments": "{}"}}


def _msg(role, content="x", **extra):
    return {"role": role, "content": content, **extra}


VALID_CORPUS = {
    "system,user": [_msg("system"), _msg("user")],
    "system,user,assistant": [_msg("system"), _msg("user"), _msg("assistant")],
    "system,user,assistant,user": [
        _msg("system"), _msg("user"), _msg("assistant"), _msg("user"),
    ],
    "multi-turn": [
        _msg("system"), _msg("user"), _msg("assistant"),
        _msg("user"), _msg("assistant"), _msg("user"), _msg("assistant"),
    ],
    "tool call and result": [
        _msg("system"), _msg("user"),
        _msg("assistant", "", tool_calls=[CALL]),
        _msg("tool", "result", tool_call_id="call-1", name="f"),
        _msg("assistant"),
    ],
}

INVALID_CORPUS = {
    "assistant only": [_msg("assistant")],
    "assistant,user": [_msg("assistant"), _msg("user")],
    "system,assistant,user": [_msg("system"), _msg("assistant"), _msg("user")],
    "tool before user": [
        _msg("system"), _msg("tool", "r", tool_call_id="call-1", name="f"), _msg("user"),
    ],
    "message after terminal assistant": [
        _msg("system"), _msg("user"), _msg("assistant"),
        _msg("tool", "r", tool_call_id="call-1", name="f"),
    ],
    "assistant before tool results": [
        _msg("system"), _msg("user"),
        _msg("assistant", "", tool_calls=[CALL]),
        _msg("assistant"),
    ],
    "orphan tool result": [
        _msg("system"), _msg("user"), _msg("tool", "r", tool_call_id="call-9", name="f"),
    ],
    "tool result id mismatch": [
        _msg("system"), _msg("user"),
        _msg("assistant", "", tool_calls=[CALL]),
        _msg("tool", "r", tool_call_id="call-wrong", name="f"),
    ],
    "missing tool result": [
        _msg("system"), _msg("user"),
        _msg("assistant", "", tool_calls=[CALL, CALL2]),
        _msg("tool", "r", tool_call_id="call-1", name="f"),
        _msg("user"),
    ],
    "valid messages, invalid order": [
        _msg("system"), _msg("assistant"), _msg("user"), _msg("assistant"),
    ],
}


def _write(path: Path, messages) -> str:
    payload = {
        "version": 1,
        "updated_at": "2026-08-21T00:00:00+00:00",
        "workdir": "/tmp/wd",
        "model": "m",
        "base_url": "http://localhost",
        "messages": messages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidSessionsTests(unittest.TestCase):
    def test_every_valid_history_loads_unchanged(self) -> None:
        for name, messages in VALID_CORPUS.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "s.json"
                digest = _write(path, messages)
                loaded, warning = SessionStore(path).load_resumable()
                self.assertIsNone(warning, f"{name} warned: {warning}")
                self.assertEqual(loaded, messages, f"{name} altered on load")
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), digest
                )

    def test_valid_session_is_offered_by_the_picker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            _write(path, VALID_CORPUS["system,user,assistant"])
            self.assertIsNotNone(SessionStore(path)._summary())


class InvalidSessionsTests(unittest.TestCase):
    def test_every_invalid_history_is_rejected_with_a_reason(self) -> None:
        for name, messages in INVALID_CORPUS.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "s.json"
                digest = _write(path, messages)
                loaded, warning = SessionStore(path).load_resumable()

                self.assertIsNone(loaded, f"{name} returned unusable history")
                self.assertIsNotNone(warning, f"{name} loaded without warning")
                assert warning is not None
                self.assertIn("cannot be resumed", warning)
                # The parser's own reason must survive into the warning, or the
                # user is told something is wrong without being told what.
                reason = conversation_structure_error(messages)
                self.assertIsNotNone(reason)
                assert reason is not None
                self.assertIn(reason, warning, f"{name} lost the parser reason")
                # The file is the user's data and must be left alone.
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), digest,
                    f"{name} modified the stored session",
                )

    def test_invalid_session_is_not_offered_by_the_picker(self) -> None:
        for name, messages in INVALID_CORPUS.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "s.json"
                _write(path, messages)
                self.assertIsNone(
                    SessionStore(path)._summary(), f"{name} offered as resumable"
                )
                self.assertTrue(path.exists(), f"{name} file was removed")

    def test_repeated_loads_do_not_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            digest = _write(path, INVALID_CORPUS["assistant,user"])
            store = SessionStore(path)
            first = store.load_resumable()
            second = store.load_resumable()
            self.assertEqual(first, second)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


class ShapeLayerTests(unittest.TestCase):
    """Some corruption is caught before the grammar is ever consulted."""

    def test_unsupported_role_is_rejected_by_shape_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            digest = _write(path, [_msg("system"), _msg("user"), _msg("wizard")])
            loaded, warning = SessionStore(path).load_resumable()
            self.assertIsNone(loaded)
            self.assertIsNotNone(warning)
            assert warning is not None
            self.assertIn("malformed message", warning)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_tool_message_without_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            _write(path, [
                _msg("system"), _msg("user"),
                _msg("assistant", "", tool_calls=[CALL]),
                {"role": "tool", "content": "r", "tool_call_id": "call-1"},
            ])
            loaded, warning = SessionStore(path).load_resumable()
            self.assertIsNone(loaded)
            self.assertIsNotNone(warning)


class FallbackTests(unittest.TestCase):
    def test_fresh_fallback_is_structurally_valid(self) -> None:
        """Rejection must leave the caller able to start a usable session."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            _write(path, INVALID_CORPUS["system,assistant,user"])
            loaded, _warning = SessionStore(path).load_resumable()
            fresh = loaded or []
            self.assertEqual(fresh, [])
            self.assertIsNone(conversation_structure_error(fresh))
            self.assertIsNone(
                conversation_structure_error(
                    [*fresh, {"role": "system", "content": "s"},
                     {"role": "user", "content": "hi"}]
                )
            )


class PersistenceVersusResumabilityTests(unittest.TestCase):
    """Persistable and resumable are different questions.

    Orbit stores state that is not a conversation -- a lone attested tool
    result kept so an evidence card stays promptable is legitimate persisted
    data, and `test_evidence.py::test_session_save_load_keeps_card_promptable`
    depends on it loading. Applying the conversation grammar in the generic
    loader broke that. These pin the boundary so it is not moved back.
    """

    EVIDENCE_ONLY = [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "exec_shell_full_command",
            "content": "evidence:ev_000000000000_0000000000000000",
            "evidence_id": "ev_000000000000_0000000000000000",
        }
    ]

    def test_evidence_only_state_still_loads_through_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            digest = _write(path, self.EVIDENCE_ONLY)
            loaded, warning = SessionStore(path).load_with_warning()
            self.assertIsNone(warning, "generic persistence rejected valid state")
            self.assertEqual(loaded, self.EVIDENCE_ONLY)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_evidence_only_state_is_not_a_resumable_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            _write(path, self.EVIDENCE_ONLY)
            loaded, warning = SessionStore(path).load_resumable()
            self.assertIsNone(loaded)
            self.assertIsNotNone(warning)

    def test_evidence_only_state_is_not_offered_as_a_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            _write(path, self.EVIDENCE_ONLY)
            self.assertIsNone(SessionStore(path)._summary())
            self.assertTrue(path.exists())

    def test_generic_loader_does_not_apply_conversation_grammar(self) -> None:
        """The guard against moving _parse_turns back into load_with_warning."""
        for name, messages in INVALID_CORPUS.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "s.json"
                _write(path, messages)
                loaded, warning = SessionStore(path).load_with_warning()
                self.assertIsNone(
                    warning,
                    f"{name}: conversation grammar leaked into generic load",
                )
                self.assertEqual(loaded, messages)


class SharedGrammarTests(unittest.TestCase):
    def test_persistence_uses_the_runtime_grammar(self) -> None:
        """One grammar: what admission rejects, load must also reject."""
        for name, messages in INVALID_CORPUS.items():
            with self.subTest(name):
                self.assertIsNotNone(conversation_structure_error(messages))
        for name, messages in VALID_CORPUS.items():
            with self.subTest(name):
                self.assertIsNone(conversation_structure_error(messages))


if __name__ == "__main__":
    unittest.main()
