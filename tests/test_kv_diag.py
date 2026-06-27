from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orbit.backend.base import ChatResult, Message
from orbit.runtime import ChatRuntime
from orbit.runtime.kv_diag import fingerprint_prompt, reset_diagnostics_for_tests


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[Message] = []
        self.tools = None

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.calls += 1
        self.messages = messages
        self.tools = tools
        return ChatResult(
            content="ok",
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=10,
            completion_tokens=2,
            cached_tokens=4,
            prompt_tokens_per_second=12.5,
            generation_tokens_per_second=3.5,
        )


class KVDiagTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_diagnostics_for_tests()

    def test_diag_default_off_does_not_write_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "0", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None)
                runtime.ask_chat("secret prompt text", temperature=0, max_tokens=32)

            self.assertFalse(log_path.exists())

    def test_diag_on_writes_hashes_not_raw_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None)
                runtime.ask_chat("secret prompt text", temperature=0, max_tokens=32)

            payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(payload["event"], "kv_diag_model_call")
        self.assertEqual(payload["phase"], "chat_final")
        self.assertIn("stable_prefix_hash", payload)
        self.assertIn("full_prompt_hash", payload)
        self.assertEqual(payload["prompt_tokens"], 10)
        self.assertEqual(payload["cached_tokens"], 4)
        self.assertEqual(payload["reused_tokens"], 4)
        self.assertEqual(payload["evaluated_tokens"], 6)
        self.assertNotIn("secret prompt text", json.dumps(payload))

    def test_stable_prefix_hash_is_stable_for_identical_inputs(self) -> None:
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "hello"},
        ]
        first = fingerprint_prompt(messages, tools=[])
        second = fingerprint_prompt(messages, tools=[])

        self.assertEqual(first.stable_prefix_hash, second.stable_prefix_hash)
        self.assertEqual(first.tool_schema_hash, second.tool_schema_hash)
        self.assertEqual(first.full_prompt_hash, second.full_prompt_hash)

    def test_capability_summary_hash_changes_when_summary_changes(self) -> None:
        base = [{"role": "system", "content": "policy"}, {"role": "user", "content": "hello"}]
        with_python = [
            *base,
            {"role": "system", "content": "Local tools available: python3.\nUnavailable: pandoc."},
        ]
        with_pandoc = [
            *base,
            {"role": "system", "content": "Local tools available: python3, pandoc.\nUnavailable: none."},
        ]

        first = fingerprint_prompt(with_python, tools=[])
        second = fingerprint_prompt(with_pandoc, tools=[])

        self.assertNotEqual(first.capability_summary_hash, second.capability_summary_hash)
        self.assertNotEqual(first.stable_prefix_hash, second.stable_prefix_hash)

    def test_tool_schema_hash_changes_with_tools_on_off(self) -> None:
        messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "hello"}]
        off = fingerprint_prompt(messages, tools=[])
        on = fingerprint_prompt(
            messages,
            tools=[{"type": "function", "function": {"name": "system_info", "parameters": {}}}],
        )

        self.assertNotEqual(off.tool_schema_hash, on.tool_schema_hash)
        self.assertNotEqual(off.stable_prefix_hash, on.stable_prefix_hash)

    def test_consecutive_same_prompt_reports_no_component_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None)
                runtime.ask_chat("same prompt", temperature=0, max_tokens=32)
                runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None)
                runtime.ask_chat("same prompt", temperature=0, max_tokens=32)

            lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(lines[0]["changed_components"], [])
        self.assertEqual(lines[1]["changed_components"], [])


if __name__ == "__main__":
    unittest.main()
