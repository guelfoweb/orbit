from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.tool_loop_state import (
    EVIDENCE_CANDIDATE_PATHS_FOUND,
    EVIDENCE_DIRECT_CONTENT_READ,
    EVIDENCE_DIRECT_READ_FAILED,
    RECONSIDER_FILE_RECOVERY,
    ToolLoopState,
    ToolTurnState,
)
from orbit.runtime import tool_loop


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

    def test_shell_full_allows_eight_rounds(self) -> None:
        state = ToolLoopState(("exec_shell_full_command",))

        self.assertEqual(state.round_limit, 8)
        for _ in range(7):
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

    def test_mutative_tool_call_budget_is_separate_and_bounded(self) -> None:
        original_mutative = tool_loop.MUTATIVE_TOOL_CALL_MAX_TOKENS
        original_file_recovery = tool_loop.FILE_RECOVERY_TOOL_CALL_MAX_TOKENS
        try:
            tool_loop.MUTATIVE_TOOL_CALL_MAX_TOKENS = 192
            tool_loop.FILE_RECOVERY_TOOL_CALL_MAX_TOKENS = 64

            self.assertEqual(tool_loop._tool_call_max_tokens(512, mutative=False), 96)
            self.assertEqual(tool_loop._tool_call_max_tokens(512, mutative=True), 192)
            self.assertEqual(tool_loop._tool_call_max_tokens(160, mutative=True), 160)
            self.assertEqual(tool_loop._tool_call_max_tokens(512, mutative=False, file_recovery=True), 64)
            self.assertEqual(tool_loop._tool_call_max_tokens(48, mutative=False, file_recovery=True), 48)
        finally:
            tool_loop.MUTATIVE_TOOL_CALL_MAX_TOKENS = original_mutative
            tool_loop.FILE_RECOVERY_TOOL_CALL_MAX_TOKENS = original_file_recovery

    def test_tool_turn_state_tracks_file_recovery_transitions(self) -> None:
        state = ToolTurnState(requested_user_path="vulnerable_service.py")

        state.mark_direct_read_failed("No such file")
        self.assertEqual(state.evidence_state, EVIDENCE_DIRECT_READ_FAILED)
        self.assertTrue(state.direct_read_failed)
        self.assertFalse(state.finalizable)

        state.mark_candidate_paths_found(["./samples/vulnerable_service.py"])
        self.assertEqual(state.evidence_state, EVIDENCE_CANDIDATE_PATHS_FOUND)
        self.assertEqual(state.candidate_paths, ["./samples/vulnerable_service.py"])

        state.mark_direct_content_read()
        self.assertEqual(state.evidence_state, EVIDENCE_DIRECT_CONTENT_READ)
        self.assertTrue(state.content_evidence_satisfied)
        self.assertTrue(state.finalizable)

    def test_tool_turn_state_limits_reconsider_once_per_kind(self) -> None:
        state = ToolTurnState()

        self.assertTrue(state.can_reconsider(RECONSIDER_FILE_RECOVERY))
        state.mark_reconsider(RECONSIDER_FILE_RECOVERY)
        self.assertFalse(state.can_reconsider(RECONSIDER_FILE_RECOVERY))


if __name__ == "__main__":
    unittest.main()
