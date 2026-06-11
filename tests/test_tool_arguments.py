from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.tool_arguments import parse_tool_arguments, parse_tool_arguments_or_empty


class ToolArgumentsTests(unittest.TestCase):
    def test_parse_tool_arguments_accepts_dict(self) -> None:
        self.assertEqual(parse_tool_arguments({"path": "."}), {"path": "."})

    def test_parse_tool_arguments_accepts_empty_string(self) -> None:
        self.assertEqual(parse_tool_arguments(""), {})

    def test_parse_tool_arguments_reports_invalid_json(self) -> None:
        result = parse_tool_arguments("{bad")

        self.assertIsInstance(result, str)
        self.assertIn("invalid JSON tool arguments", result)

    def test_parse_tool_arguments_requires_object(self) -> None:
        self.assertEqual(parse_tool_arguments("[]"), "error: tool arguments must be a JSON object")

    def test_parse_tool_arguments_or_empty_is_permissive(self) -> None:
        self.assertEqual(parse_tool_arguments_or_empty("{bad"), {})
        self.assertEqual(parse_tool_arguments_or_empty("[]"), {})


if __name__ == "__main__":
    unittest.main()
