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

    def test_chat_budget_respects_explicit_user_limit(self) -> None:
        self.assertEqual(resolve_max_tokens("chat"), 192)
        self.assertEqual(resolve_max_tokens("chat", 32), 64)
        self.assertEqual(resolve_max_tokens("chat", 512), 512)
        self.assertEqual(resolve_max_tokens("chat", 2048), 2048)
        self.assertEqual(resolve_max_tokens("chat", 8192), 4096)

    def test_final_from_tool_structural_evidence_budgets(self) -> None:
        self.assertEqual(resolve_max_tokens("final_from_tool", 32, evidence_kind="shell", evidence_chars=80), 96)
        self.assertEqual(resolve_max_tokens("final_from_tool", 512, evidence_kind="unknown", evidence_chars=80), 96)
        self.assertEqual(resolve_max_tokens("final_from_tool", 32, evidence_kind="shell_error", evidence_chars=120), 128)
        self.assertEqual(resolve_max_tokens("final_from_tool", 512, evidence_kind="system_info", evidence_chars=300), 160)
        self.assertEqual(resolve_max_tokens("final_from_tool", 32, evidence_kind="shell", evidence_chars=1200), 192)
        self.assertEqual(resolve_max_tokens("final_from_tool", 512, evidence_kind="directory_listing", evidence_chars=500), 96)
        self.assertEqual(resolve_max_tokens("final_from_tool", 512, evidence_kind="directory_listing", evidence_chars=501), 192)
        self.assertEqual(resolve_max_tokens("final_from_tool", 512, evidence_kind="web_search", evidence_chars=1200), 192)
        self.assertEqual(resolve_max_tokens("final_from_tool", 32, evidence_kind="read", evidence_chars=8000), 256)
        self.assertEqual(resolve_max_tokens("final_from_tool", 1024, evidence_kind="fetch", evidence_chars=8000), 256)

    def test_full_document_budget_defaults_to_dedicated_cap_and_honors_explicit_limit(self) -> None:
        self.assertEqual(resolve_max_tokens("full_document"), 1024)
        self.assertEqual(resolve_max_tokens("full_document", 256), 256)
        self.assertEqual(resolve_max_tokens("full_document", 512), 512)
        self.assertEqual(resolve_max_tokens("full_document", 1024), 1024)
        # An explicit budget above the default is honored verbatim. Exact
        # full-document admission, not this budget, decides whether it fits.
        self.assertEqual(resolve_max_tokens("full_document", 2048), 2048)
        self.assertEqual(resolve_max_tokens("full_document", 8192), 8192)

    def test_finalization_budget_is_independent_of_investigation_limits(self) -> None:
        # Finalization answers from frozen evidence in its own session, so the
        # investigation's per-call limit must not decide how long a report can be.
        self.assertEqual(resolve_max_tokens("finalization"), 4096)
        self.assertNotEqual(resolve_max_tokens("finalization"), resolve_max_tokens("chat"))
        # An explicit caller limit stays authoritative, as for full_document.
        self.assertEqual(resolve_max_tokens("finalization", 512), 512)
        self.assertEqual(resolve_max_tokens("finalization", 8192), 8192)

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
