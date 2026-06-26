from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.tools import default_tool_names, execute_tool, tool_definitions, tool_names


class ToolTests(unittest.TestCase):
    def test_shell_and_fetch_url_are_exposed(self) -> None:
        self.assertEqual(tool_names(), ("exec_shell_full_command", "fetch_url", "list_directory"))
        self.assertEqual(default_tool_names(), ("exec_shell_full_command", "fetch_url", "list_directory"))
        definitions = tool_definitions()
        self.assertEqual([item["function"]["name"] for item in definitions], ["exec_shell_full_command", "fetch_url", "list_directory"])

    def test_tool_definitions_respect_allowed_names(self) -> None:
        self.assertEqual(tool_definitions(("read_file",)), [])
        self.assertEqual(len(tool_definitions(("exec_shell_full_command",))), 1)
        self.assertEqual(len(tool_definitions(("fetch_url",))), 1)
        self.assertEqual(len(tool_definitions(("list_directory",))), 1)

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

    def test_fetch_url_runs_with_structured_result(self) -> None:
        class FakeHeaders:
            def get_content_type(self) -> str:
                return "text/html"

            def get_content_charset(self) -> str:
                return "utf-8"

        class FakeResponse:
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def geturl(self) -> str:
                return "https://example.com/final"

            def getcode(self) -> int:
                return 200

            def read(self, amount: int = -1) -> bytes:
                del amount
                return b"<html><title>Example</title><body>Hello web</body></html>"

        with tempfile.TemporaryDirectory() as tmp:
            with patch("orbit.runtime.web.urlopen", return_value=FakeResponse()):
                result = execute_tool("fetch_url", {"url": "https://example.com"}, workdir=Path(tmp))

        self.assertIn("url_fetch: true", result.content)
        self.assertIn("status: ok", result.content)
        self.assertIn("title: Example", result.content)

    def test_list_directory_runs_with_compact_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello", encoding="utf-8")

            result = execute_tool("list_directory", {"path": "."}, workdir=root)

        self.assertIn("directory_listing:", result.content)
        self.assertIn("[file] README.md", result.content)


if __name__ == "__main__":
    unittest.main()
