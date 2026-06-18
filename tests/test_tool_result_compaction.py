from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult
from orbit.runtime.tool_result_compaction import (
    ORIGINAL_TOOL_CONTENT_KEY,
    compact_tool_results,
    find_tool_result_candidates,
    persistent_messages,
)


class ToolResultCompactionTests(unittest.TestCase):
    def test_finds_old_large_tool_results(self) -> None:
        messages = [
            {"role": "system", "content": "s"},
            {"role": "tool", "name": "read_file", "tool_call_id": "1", "content": "x" * 1000},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u"},
        ]

        candidates = find_tool_result_candidates(messages)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].tool, "read_file")
        self.assertGreater(candidates[0].estimated_tokens, 200)
        self.assertEqual(candidates[0].age_messages, 2)

    def test_compacts_with_model_generated_summary_and_preserves_original(self) -> None:
        messages = [
            {"role": "system", "content": "s"},
            {"role": "tool", "name": "read_file", "tool_call_id": "1", "content": "important " * 200},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u"},
        ]

        report = compact_tool_results(messages, backend=FakeCompactionBackend(), temperature=0)

        self.assertTrue(report.changed)
        self.assertGreater(report.saved_tokens, 0)
        self.assertIn("[Compacted tool result]", messages[1]["content"])
        self.assertIn("durable facts", messages[1]["content"])
        self.assertIn(ORIGINAL_TOOL_CONTENT_KEY, messages[1])

        saved = persistent_messages(messages)
        self.assertEqual(saved[1]["content"], "important " * 200)
        self.assertNotIn(ORIGINAL_TOOL_CONTENT_KEY, saved[1])

    def test_skips_when_summary_is_not_smaller(self) -> None:
        messages = [
            {"role": "tool", "name": "read_file", "tool_call_id": "1", "content": "x" * 1000},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u"},
        ]

        report = compact_tool_results(messages, backend=VerboseCompactionBackend(), temperature=0)

        self.assertFalse(report.changed)
        self.assertEqual(messages[0]["content"], "x" * 1000)


class FakeCompactionBackend:
    def chat(self, messages, *, temperature, max_tokens, tools=None):
        return ChatResult(
            content="durable facts with numbers and paths",
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None, on_delta=None, on_progress=None):
        raise AssertionError("streaming should not be used")


class VerboseCompactionBackend(FakeCompactionBackend):
    def chat(self, messages, *, temperature, max_tokens, tools=None):
        return ChatResult(
            content="y" * 2000,
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )


if __name__ == "__main__":
    unittest.main()
