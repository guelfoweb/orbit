from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult
from orbit.runtime.session_memory import MemoryRefresh
from orbit.terminal.status import format_memory_refresh, format_turn_status
from orbit.terminal.theme import dim


class StatusTests(unittest.TestCase):
    def test_format_turn_status_includes_stop_tokens_cache_and_speed(self) -> None:
        status = format_turn_status(
            ChatResult(
                content="hello",
                model="gemma4",
                finish_reason="stop",
                tool_calls=[],
                prompt_tokens=10,
                completion_tokens=3,
                cached_tokens=8,
                prompt_tokens_per_second=12.5,
                generation_tokens_per_second=3.4,
            )
        )

        self.assertIn("model: gemma4", status)
        self.assertIn("stop: stop", status)
        self.assertIn("tks: 10->3, cached 8", status)
        self.assertIn("pf 12.5/s", status)
        self.assertIn("gen 3.4/s", status)

    def test_format_turn_status_includes_context_window_and_usage_percent(self) -> None:
        status = format_turn_status(
            ChatResult(
                content="hello",
                model="gemma4",
                finish_reason="stop",
                tool_calls=[],
                prompt_tokens=None,
                completion_tokens=None,
                cached_tokens=None,
                prompt_tokens_per_second=None,
                generation_tokens_per_second=None,
            ),
            estimated_context_tokens=2212,
            context_tokens=8192,
        )

        self.assertIn("model: gemma4 | ctx: 8192 (27%) | stop: stop", status)

    def test_format_turn_status_includes_elapsed_time(self) -> None:
        result = ChatResult(
            content="hello",
            model="gemma4",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

        self.assertIn("time: 34s", format_turn_status(result, elapsed_seconds=34))
        self.assertIn("time: 1m 19s", format_turn_status(result, elapsed_seconds=79))
        self.assertTrue(format_turn_status(result, elapsed_seconds=34).endswith("stop: stop | time: 34s"))

    def test_dim_wraps_text_in_ansi_escape(self) -> None:
        self.assertEqual(dim("model: gemma4"), "\033[2mmodel: gemma4\033[0m")

    def test_format_memory_refresh_includes_savings_timing_and_threshold(self) -> None:
        status = format_memory_refresh(
            MemoryRefresh(
                changed=True,
                reason="memory-refreshed",
                estimated_tokens_before=1000,
                estimated_tokens_after=250,
                elapsed_seconds=12.34,
                context_tokens=1600,
                threshold_tokens=1200,
            )
        )

        self.assertIn("memory: 1000->250 est. tokens", status)
        self.assertIn("saved 750 (75%)", status)
        self.assertIn("12.3s", status)
        self.assertIn("threshold 1200/1600", status)


if __name__ == "__main__":
    unittest.main()
