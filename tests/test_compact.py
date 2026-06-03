from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.core.compaction import SUMMARY_MARKER, build_hybrid_refinement_messages, compact_messages, plan_compaction


class CompactTests(unittest.TestCase):
    def test_compact_keeps_recent_messages_and_adds_summary(self) -> None:
        messages = [{"role": "system", "content": "base"}]
        for idx in range(20):
            messages.append({"role": "user", "content": f"user {idx}"})
        compacted, changed = compact_messages(messages)
        self.assertTrue(changed)
        self.assertEqual(compacted[0]["role"], "system")
        self.assertEqual(compacted[1]["role"], "system")
        self.assertIn(SUMMARY_MARKER, compacted[1]["content"])
        self.assertIn("Working memory:", compacted[1]["content"])
        self.assertIn("Durable memory:", compacted[1]["content"])
        self.assertEqual(len(compacted), 12)
        self.assertEqual(compacted[-1]["content"], "user 19")
        self.assertEqual([item["content"] for item in compacted[-4:]], ["user 16", "user 17", "user 18", "user 19"])

    def test_compact_noop_for_short_session(self) -> None:
        messages = [{"role": "system", "content": "base"}, {"role": "user", "content": "hello"}]
        compacted, changed = compact_messages(messages)
        self.assertFalse(changed)
        self.assertEqual(compacted, messages)

    def test_compact_limits_recent_window_by_budget(self) -> None:
        messages = [{"role": "system", "content": "base"}]
        for idx in range(6):
            messages.append({"role": "user", "content": f"older {idx}"})
        for idx in range(6):
            messages.append({"role": "assistant", "content": "x" * 5000 + str(idx)})
        compacted, changed = compact_messages(messages)
        self.assertTrue(changed)
        recent = compacted[2:]
        self.assertLess(len(recent), 12)
        self.assertGreaterEqual(len(recent), 4)

    def test_plan_compaction_builds_refinement_messages(self) -> None:
        messages = [{"role": "system", "content": "base"}]
        for idx in range(20):
            messages.append({"role": "user", "content": f"user {idx}"})
        plan = plan_compaction(messages)
        self.assertIsNotNone(plan)
        refinement_messages = build_hybrid_refinement_messages(plan)
        self.assertEqual(refinement_messages[0]["role"], "system")
        self.assertIn("Rewrite session memory", refinement_messages[0]["content"])
        self.assertIn(SUMMARY_MARKER, refinement_messages[1]["content"])

    def test_plan_compaction_shrinks_recent_window_when_overflowing(self) -> None:
        messages = [{"role": "system", "content": "base"}]
        for idx in range(20):
            messages.append({"role": "assistant", "content": ("x" * 1200) + str(idx)})
        normal = plan_compaction(messages)
        overflowed = plan_compaction(messages, overflow_tokens=3000)
        self.assertIsNotNone(normal)
        self.assertIsNotNone(overflowed)
        self.assertLess(len(overflowed.recent_messages), len(normal.recent_messages))
        self.assertEqual(overflowed.recent_messages[-1]["content"], ("x" * 1200) + "19")
