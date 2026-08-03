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
from orbit.runtime.turn_trace import ModelStepMetrics
from orbit.terminal.status import (
    TokenUsageAccumulator,
    format_memory_refresh,
    format_session_token_usage,
    format_turn_status,
    summarize_turn_token_usage,
)
from orbit.terminal.theme import dim, yellow_dim


class StatusTests(unittest.TestCase):
    def test_session_token_usage_reports_real_processed_total(self) -> None:
        usage = TokenUsageAccumulator()
        usage.add(ModelStepMetrics(1, "route", "stop", 100, 5, 20, 10.0, 2.0, 0))
        usage.add(ModelStepMetrics(1, "chat_final", "stop", 300, 15, 0, 10.0, 2.0, 0))

        self.assertEqual(
            format_session_token_usage(usage.snapshot()),
            "session tks: 420 total (400 in + 20 out) | work: 400 (380 prefill + 20 decode) | cache: 20 (5%) | calls: 2",
        )

    def test_failed_call_makes_token_total_unavailable_without_hiding_attempt(self) -> None:
        usage = TokenUsageAccumulator()
        usage.add(ModelStepMetrics(1, "route", "stop", 100, 5, 20, 10.0, 2.0, 0))
        usage.add_failed_call()

        rendered = format_session_token_usage(usage.snapshot())

        self.assertIn("session tks: unavailable", rendered)
        self.assertIn("calls: 2", rendered)
        self.assertIn("failed: 1 (token usage unavailable)", rendered)

    def test_turn_token_usage_sums_every_model_call_and_real_evaluated_tokens(self) -> None:
        steps = [
            ModelStepMetrics(1, "route", "stop", 100, 5, 20, 10.0, 2.0, 0),
            ModelStepMetrics(1, "tool_call", "stop", 200, 10, 50, 10.0, 2.0, 1),
            ModelStepMetrics(2, "post_tool_route", "stop", 300, 15, 0, 10.0, 2.0, 0),
        ]

        usage = summarize_turn_token_usage(steps)

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.model_calls, 3)
        self.assertEqual(usage.prompt_tokens, 600)
        self.assertEqual(usage.evaluated_tokens, 530)
        self.assertEqual(usage.cached_tokens, 70)
        self.assertEqual(usage.completion_tokens, 30)

    def test_format_turn_status_prefers_complete_turn_totals(self) -> None:
        result = ChatResult(
            content="hello",
            model="gemma4",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=300,
            completion_tokens=15,
            cached_tokens=0,
            prompt_tokens_per_second=12.5,
            generation_tokens_per_second=3.4,
        )
        usage = summarize_turn_token_usage(
            [
                ModelStepMetrics(1, "route", "stop", 100, 5, 20, 10.0, 2.0, 0),
                ModelStepMetrics(1, "tool_call", "stop", 200, 10, 50, 10.0, 2.0, 1),
                ModelStepMetrics(2, "post_tool_route", "stop", 300, 15, 0, 10.0, 2.0, 0),
            ]
        )

        status = format_turn_status(result, turn_token_usage=usage)

        self.assertIn("tks: 630 total (600 in + 30 out)", status)
        self.assertIn("work: 560 (530 prefill + 30 decode)", status)
        self.assertIn("cache: 70 (12%)", status)
        self.assertIn("calls: 3", status)

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

        self.assertNotIn("model:", status)
        self.assertIn("stop: stop", status)
        self.assertIn("tks: 10->3, cached 8", status)
        self.assertIn("cache: 80%", status)
        self.assertIn("__ pf 12.5/s | gen 3.4/s", status)

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

        self.assertIn("ctx: 2212/8192 (27%) | stop: stop", status)

    def test_format_turn_status_includes_context_pressure(self) -> None:
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

        moderate = format_turn_status(result, estimated_context_tokens=4280, context_tokens=8192)
        high = format_turn_status(result, estimated_context_tokens=5900, context_tokens=8192)
        refresh = format_turn_status(result, estimated_context_tokens=7000, context_tokens=8192)

        self.assertIn("ctx: 4280/8192 (52%)", moderate)
        self.assertIn("pressure: moderate", moderate)
        self.assertIn("ctx: 5900/8192 (72%)", high)
        self.assertIn("pressure: high | consider /compact tools", high)
        self.assertIn("pressure: memory refresh", refresh)

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

        short = format_turn_status(result, elapsed_seconds=34, terminal_columns=72)
        long = format_turn_status(result, elapsed_seconds=79, terminal_columns=72)

        self.assertIn("time elapsed: 34s", short)
        self.assertIn("time elapsed: 1m 19s", long)
        self.assertEqual(len(short.splitlines()[0]), 72)
        self.assertTrue(short.endswith("stop: stop"))

    def test_dim_wraps_text_in_ansi_escape(self) -> None:
        self.assertEqual(dim("model: gemma4"), "\033[2mmodel: gemma4\033[0m")
        self.assertEqual(yellow_dim("[text 10 chars #12345678]"), "\033[2m\033[33m[text 10 chars #12345678]\033[0m")

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
