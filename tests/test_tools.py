from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.tools import default_tool_names, execute_tool, tool_definitions, tool_names


class ToolTests(unittest.TestCase):
    def test_only_shell_full_is_exposed(self) -> None:
        self.assertEqual(tool_names(), ("exec_shell_full_command",))
        self.assertEqual(default_tool_names(), ("exec_shell_full_command",))
        definitions = tool_definitions()
        self.assertEqual([item["function"]["name"] for item in definitions], ["exec_shell_full_command"])

    def test_tool_definitions_respect_allowed_names(self) -> None:
        self.assertEqual(tool_definitions(("read_file",)), [])
        self.assertEqual(len(tool_definitions(("exec_shell_full_command",))), 1)

    def test_unknown_tool_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_tool("read_file", {"path": "note.txt"}, workdir=Path(tmp))

        self.assertEqual(result.content, "error: unknown tool: read_file")

    def test_exec_shell_full_runs_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_tool("exec_shell_full_command", {"command": "printf hello"}, workdir=Path(tmp))

        self.assertEqual(result.content, "hello")

    def test_exec_shell_full_invalid_arguments_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_tool("exec_shell_full_command", "{", workdir=Path(tmp))

        self.assertIn("invalid JSON", result.content)


if __name__ == "__main__":
    unittest.main()
