from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.tool_loop_state import ToolLoopState


class ToolLoopStateTests(unittest.TestCase):
    def test_default_round_limit_is_one(self) -> None:
        state = ToolLoopState(("read_file",))

        self.assertEqual(state.round_limit, 1)
        self.assertFalse(state.round_limit_reached())
        state.increment_round()
        self.assertTrue(state.round_limit_reached())

    def test_removed_edit_tools_use_default_round_limit(self) -> None:
        state = ToolLoopState(("edit_file",))

        self.assertEqual(state.round_limit, 1)
        state.increment_round()
        self.assertTrue(state.round_limit_reached())

    def test_shell_full_allows_four_rounds(self) -> None:
        state = ToolLoopState(("exec_shell_full_command",))

        self.assertEqual(state.round_limit, 4)
        for _ in range(3):
            state.increment_round()
            self.assertFalse(state.round_limit_reached())
        state.increment_round()
        self.assertTrue(state.round_limit_reached())

    def test_tracks_repeated_tool_calls(self) -> None:
        state = ToolLoopState(("read_file",))
        tool_call = {"id": "call-1", "function": {"name": "read_file", "arguments": {"path": "a.txt"}}}

        self.assertFalse(state.has_seen_tool_call(tool_call))
        signature = state.mark_tool_call(tool_call)

        self.assertEqual(signature[0], "read_file")
        self.assertTrue(state.has_seen_tool_call(tool_call))


if __name__ == "__main__":
    unittest.main()
