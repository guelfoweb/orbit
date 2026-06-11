from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.tool_message import assistant_tool_call_message, tool_result_message
from orbit.runtime.tools import ToolResult


class ToolMessageTests(unittest.TestCase):
    def test_assistant_tool_call_message_includes_tool_calls_when_present(self) -> None:
        tool_calls = [{"id": "call-1", "function": {"name": "read_file", "arguments": {"path": "a.txt"}}}]

        message = assistant_tool_call_message("", tool_calls)

        self.assertEqual(message, {"role": "assistant", "content": "", "tool_calls": tool_calls})

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


if __name__ == "__main__":
    unittest.main()
