from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.completion_budget import resolve_max_tokens


class CompletionBudgetPolicyTests(unittest.TestCase):
    def test_route_budget(self) -> None:
        self.assertEqual(resolve_max_tokens("route"), 64)
        self.assertEqual(resolve_max_tokens("route", 32), 32)

    def test_tool_call_budgets(self) -> None:
        self.assertEqual(resolve_max_tokens("tool_call"), 96)
        self.assertEqual(resolve_max_tokens("tool_call", 512), 96)
        self.assertEqual(resolve_max_tokens("tool_call_file_recovery", 512), 64)

    def test_chat_budget_respects_requested_with_cap(self) -> None:
        self.assertEqual(resolve_max_tokens("chat"), 192)
        self.assertEqual(resolve_max_tokens("chat", 32), 64)
        self.assertEqual(resolve_max_tokens("chat", 512), 256)

    def test_final_from_tool_structural_evidence_budgets(self) -> None:
        self.assertEqual(resolve_max_tokens("final_from_tool", 32, evidence_kind="shell", evidence_chars=80), 96)
        self.assertEqual(resolve_max_tokens("final_from_tool", 512, evidence_kind="unknown", evidence_chars=80), 96)
        self.assertEqual(resolve_max_tokens("final_from_tool", 32, evidence_kind="shell_error", evidence_chars=120), 128)
        self.assertEqual(resolve_max_tokens("final_from_tool", 32, evidence_kind="shell", evidence_chars=1200), 192)
        self.assertEqual(resolve_max_tokens("final_from_tool", 512, evidence_kind="web_search", evidence_chars=1200), 192)
        self.assertEqual(resolve_max_tokens("final_from_tool", 32, evidence_kind="read", evidence_chars=8000), 256)

    def test_retry_and_repair_budgets(self) -> None:
        self.assertEqual(resolve_max_tokens("chat_final_retry", 32), 128)
        self.assertEqual(resolve_max_tokens("chat_final_retry", 32, previous_finish_reason="length"), 192)
        self.assertEqual(resolve_max_tokens("final_from_tool_retry", 512, previous_finish_reason="length"), 192)
        self.assertEqual(resolve_max_tokens("repair", 32), 128)
        self.assertEqual(resolve_max_tokens("repair", 512), 160)
        self.assertEqual(resolve_max_tokens("repair", 512, previous_finish_reason="length"), 192)

    def test_no_user_text_required(self) -> None:
        self.assertEqual(
            resolve_max_tokens(
                "final_from_tool",
                requested_max_tokens=64,
                evidence_kind="grep_search",
                evidence_chars=1000,
                previous_finish_reason="stop",
            ),
            192,
        )


if __name__ == "__main__":
    unittest.main()
