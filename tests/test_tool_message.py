from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.evidence import EvidenceStore
from orbit.runtime.tool_message import assistant_tool_call_message, tool_result_message
from orbit.runtime.tools import ToolResult


class ToolMessageTests(unittest.TestCase):
    def test_assistant_tool_call_message_includes_tool_calls_when_present(self) -> None:
        tool_calls = [{"id": "call-1", "function": {"name": "read_file", "arguments": {"path": "a.txt"}}}]

        message = assistant_tool_call_message("", tool_calls)

        self.assertEqual(
            message,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'}}],
            },
        )

    def test_assistant_tool_call_message_sanitizes_malformed_arguments(self) -> None:
        tool_calls = [{"id": "call-1", "function": {"name": "exec_shell_full_command", "arguments": '{"command":"unterminated'}}]

        message = assistant_tool_call_message("", tool_calls)

        arguments = message["tool_calls"][0]["function"]["arguments"]  # type: ignore[index]
        self.assertIn("invalid_arguments", arguments)
        self.assertTrue(arguments.startswith("{"))

    def test_assistant_tool_call_message_omits_empty_tool_calls(self) -> None:
        message = assistant_tool_call_message("done", [])

        self.assertEqual(message, {"role": "assistant", "content": "done"})

    def test_tool_result_message_preserves_tool_call_id_name_and_content(self) -> None:
        tool_call = {"id": "call-1", "function": {"name": "read_file", "arguments": {"path": "a.txt"}}}
        result = ToolResult(name="read_file", content="hello")

        message = tool_result_message(tool_call, result)

        self.assertEqual(
            message,
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "read_file",
                "content": "hello",
            },
        )

    def test_tool_result_message_uses_evidence_card_when_store_is_available(self) -> None:
        tool_call = {
            "id": "call-1",
            "function": {"name": "exec_shell_full_command", "arguments": {"command": "cat big.txt"}},
        }
        raw = "start\n" + ("secret middle " * 200) + "\nend"
        result = ToolResult(name="exec_shell_full_command", content=raw)

        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            message = tool_result_message(tool_call, result, evidence_store=store)

            self.assertEqual(message["role"], "tool")
            self.assertEqual(message["tool_call_id"], "call-1")
            self.assertEqual(message["name"], "exec_shell_full_command")
            self.assertIn("tool_evidence_ref: true", message["content"])
            self.assertIn("raw_ref:", message["content"])
            self.assertNotIn("evidence_excerpt:", message["content"])
            self.assertNotIn("tool_evidence_card: true", message["content"])
            self.assertLess(len(str(message["content"])), len(raw))
            self.assertNotIn(raw, message["content"])
            self.assertEqual(store.load_raw(message["evidence_id"]), raw)


if __name__ == "__main__":
    unittest.main()
