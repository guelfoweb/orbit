"""Final retries preserve committed conversation, not transient route output."""
from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from orbit.backend.base import ChatResult, TokenCount
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.context_manager import _parse_turns
from orbit.runtime.evidence import EvidenceStore, tool_evidence_ref
from orbit.runtime.messages import CHAT_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT


class RetryBackend:
    context_tokens = 16384
    thinking = False

    def __init__(self):
        self.calls = []

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        digest = hashlib.sha256(repr(messages).encode()).hexdigest()
        return TokenCount(tokens=100, context_tokens=16384,
                          rendered_hash=digest, token_hash=digest)

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                    on_delta=None, on_progress=None):
        self.calls.append(copy.deepcopy(messages))
        if len(self.calls) == 1:
            on_delta("TRANSIENT route prose")  # Real route filter aborts this.
            raise AssertionError("route prose should have been aborted")
        on_delta("CEDAR")
        return ChatResult(content="CEDAR", model="fake", finish_reason="stop",
                          tool_calls=[], prompt_tokens=100, completion_tokens=2,
                          cached_tokens=0, prompt_tokens_per_second=None,
                          generation_tokens_per_second=None)


class FinalRetryHistoryTests(unittest.TestCase):
    def assert_projection(self, history):
        runtime = ChatRuntime(backend=RetryBackend(), messages=copy.deepcopy(history))
        before = copy.deepcopy(runtime.messages)
        projected = runtime._chat_final_retry_messages()
        self.assertEqual(projected, [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, *history])
        self.assertEqual(runtime.messages, before)
        _parse_turns(projected)
        return runtime, projected

    def test_multiple_turns_preserve_every_message_once(self):
        history = [
            {"role": "user", "content": "Remember the destination Kyoto."},
            {"role": "assistant", "content": "Destination recorded."},
            {"role": "user", "content": "Remember the departure Tuesday."},
            {"role": "assistant", "content": "Departure recorded."},
            {"role": "user", "content": "Where and when?"},
        ]
        self.assert_projection(history)

    def test_short_and_first_turn_histories(self):
        for history in ([], [{"role": "user", "content": "hello"}],
                        [{"role": "user", "content": "hello"},
                         {"role": "assistant", "content": "hello"},
                         {"role": "user", "content": "hello"}]):
            with self.subTest(history=history):
                self.assert_projection(history)

    def test_long_committed_assistant_is_not_truncated(self):
        self.assert_projection([
            {"role": "user", "content": "Explain the sequence."},
            {"role": "assistant", "content": "detail " * 1000 + "last detail"},
            {"role": "user", "content": "What was the last detail?"},
        ])

    def test_tool_envelopes_and_raw_results_stay_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "evidence")
            record = store.add("exec_shell_full_command", "exit_code: 0\nSTDOUT:\nLinux",
                               metadata={"command": "uname"})
            visible = [{"role": "user", "content": "Check the system."},
                       {"role": "assistant", "content": "The system is Linux."},
                       {"role": "user", "content": "Which system was it?"}]
            history = [visible[0],
                       {"role": "assistant", "content": "controller tool request",
                        "tool_calls": [{"id": "call-1", "function": {"name": "exec_shell_full_command",
                                       "arguments": '{"command":"uname"}'}}]},
                       {"role": "tool", "tool_call_id": "call-1",
                        "content": tool_evidence_ref(record) + " RAW_SENTINEL " * 500},
                       *visible[1:]]
            runtime = ChatRuntime(backend=RetryBackend(), messages=copy.deepcopy(history),
                                  evidence_store=store)
            projected = runtime._chat_final_retry_messages()
            self.assertEqual(projected[1:-1], visible)
            self.assertIn("evidence_context:", projected[-1]["content"])
            self.assertIn(record.raw_ref, projected[-1]["content"])
            self.assertNotIn("RAW_SENTINEL", repr(projected))
            self.assertNotIn("controller tool request", repr(projected))
            self.assertEqual(runtime.messages, history)
            _parse_turns(projected)

    def test_tool_free_history_with_evidence_keeps_all_user_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "evidence")
            store.add("system_info", "OS: Linux")
            history = [{"role": "user", "content": "Remember the destination Kyoto."},
                       {"role": "assistant", "content": "Recorded."},
                       {"role": "user", "content": "Where?"}]
            runtime = ChatRuntime(backend=RetryBackend(), messages=history, evidence_store=store)
            self.assertEqual(runtime._chat_final_retry_messages()[1:-1], history)

    def test_session_memory_is_preserved(self):
        memory = {"role": "system", "content": "[Session memory]\nEarlier destination: Kyoto"}
        question = {"role": "user", "content": "Where?"}
        runtime = ChatRuntime(backend=RetryBackend(), messages=[
            {"role": "system", "content": ROUTE_SYSTEM_PROMPT}, memory, question])
        self.assertEqual(runtime._chat_final_retry_messages(), [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT}, memory, question])

    def test_reset_and_new_runtime_exclude_previous_history(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                runtime = ChatRuntime(backend=RetryBackend(), system_prompt=ROUTE_SYSTEM_PROMPT)
                runtime.messages.extend([{"role": "user", "content": "private declaration"},
                                         {"role": "assistant", "content": "acknowledged"}])
                if reset:
                    runtime.reset()
                else:
                    runtime = ChatRuntime(backend=RetryBackend(), system_prompt=ROUTE_SYSTEM_PROMPT)
                question = {"role": "user", "content": "hello"}
                runtime.messages.append(question)
                self.assertEqual(runtime._chat_final_retry_messages(), [
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT}, question])

    def test_streamed_retry_preserves_declaration_acknowledgement_question(self):
        backend = RetryBackend()
        committed = [
            {"role": "user", "content": "Remember the codeword CEDAR."},
            {"role": "assistant", "content": "Acknowledged."},
        ]
        question = {"role": "user", "content": "What codeword did I give you?"}
        runtime = ChatRuntime(backend=backend, system_prompt=ROUTE_SYSTEM_PROMPT,
                              messages=[{"role": "system", "content": ROUTE_SYSTEM_PROMPT},
                                        *copy.deepcopy(committed)])
        phases = []
        with tempfile.TemporaryDirectory() as tmp:
            runtime.ask_auto(question["content"], temperature=0, max_tokens=32,
                             workdir=Path(tmp), on_final_delta=lambda _: None,
                             on_progress=lambda _: None,
                             on_phase_start=lambda p: phases.append(p.phase))
        self.assertIn("chat_final_retry", phases)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[1],
                         [{"role": "system", "content": CHAT_SYSTEM_PROMPT},
                          *committed, question])
        self.assertNotIn("TRANSIENT", repr(backend.calls[1]))


if __name__ == "__main__":
    unittest.main()
