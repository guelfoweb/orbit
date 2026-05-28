from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.core.tool_session_state import ToolDedupCache, ToolSessionState, ToolTrustDecay


class ToolSessionStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "demo.txt").write_text("hello", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_read_before_write_guard_warns_then_allows(self) -> None:
        state = ToolSessionState(self.root)
        ok, reason = state.read_guard.check_write("demo.txt")
        self.assertFalse(ok)
        self.assertIn("has not been read", reason)
        ok, reason = state.read_guard.check_write("demo.txt")
        self.assertTrue(ok)
        self.assertIn("prior warning", reason)

    def test_read_before_write_guard_lifts_after_read(self) -> None:
        state = ToolSessionState(self.root)
        ok, _ = state.read_guard.check_write("demo.txt")
        self.assertFalse(ok)
        state.read_guard.record_read("demo.txt")
        ok, reason = state.read_guard.check_write("demo.txt")
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_dedup_returns_cached_result_for_pure_tools(self) -> None:
        dedup = ToolDedupCache()
        dedup.record("read_file", {"path": "demo.txt"}, {"ok": True, "content": "hello"})
        result = dedup.lookup("read_file", {"path": "demo.txt"})
        self.assertIsNotNone(result)
        self.assertTrue(result["_dedup_cached"])
        self.assertEqual(result["content"], "hello")

    def test_dedup_ignores_side_effect_tools(self) -> None:
        dedup = ToolDedupCache()
        dedup.record("write_file", {"path": "demo.txt"}, {"ok": True})
        result = dedup.lookup("write_file", {"path": "demo.txt"})
        self.assertIsNone(result)

    def test_trust_decay_warns_then_drops(self) -> None:
        trust = ToolTrustDecay()
        for _ in range(3):
            trust.record("search_web", False)
        self.assertEqual(trust.level("search_web"), "warn")
        for _ in range(2):
            trust.record("search_web", False)
        self.assertEqual(trust.level("search_web"), "drop")

    def test_trust_decay_resets_on_success(self) -> None:
        trust = ToolTrustDecay()
        for _ in range(3):
            trust.record("search_web", False)
        trust.record("search_web", True)
        self.assertEqual(trust.level("search_web"), "ok")
