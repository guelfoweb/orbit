"""Telemetry must survive a backend attempt that failed and was recovered.

A malformed tool-call JSON makes the backend raise after inference has already
run. The runtime retries and the user gets a correct answer, but the accounting
used to treat that attempt as a reason to discard every figure it had: one
recovered failure erased the whole session's token usage, including tokens that
had been measured correctly before it.

The attempt is still counted -- it really happened. What changes is that a
number already known stays known, and the total says plainly that it is
missing something rather than pretending to be exact.
"""

from __future__ import annotations

import unittest

from orbit.backend.base import ChatResult
from orbit.terminal.status import TokenUsageAccumulator, TurnTokenUsage


def _result(prompt: int, completion: int, cached: int = 0) -> ChatResult:
    return ChatResult(
        content="ok",
        model="m",
        finish_reason="stop",
        tool_calls=[],
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        prompt_tokens_per_second=None,
        generation_tokens_per_second=None,
    )


class CounterSemanticsTests(unittest.TestCase):
    def test_success_only(self) -> None:
        usage = TokenUsageAccumulator()
        usage.add_result(_result(881, 42))
        self.assertEqual(usage.prompt_tokens, 881)
        self.assertEqual(usage.completion_tokens, 42)
        self.assertEqual(usage.model_calls, 1)
        self.assertEqual(usage.failed_calls, 0)
        self.assertFalse(usage.usage_incomplete)

    def test_failure_before_any_metrics(self) -> None:
        usage = TokenUsageAccumulator()
        usage.add_failed_call()
        self.assertEqual(usage.model_calls, 1)
        self.assertEqual(usage.failed_calls, 1)
        self.assertTrue(usage.usage_incomplete)
        # Nothing was measured, so nothing is claimed -- but zero is a real
        # answer here, not a fabricated one.
        self.assertEqual(usage.prompt_tokens, 0)

    def test_known_metrics_survive_a_later_failure(self) -> None:
        usage = TokenUsageAccumulator()
        usage.add_result(_result(881, 42))
        usage.add_failed_call()
        self.assertEqual(usage.prompt_tokens, 881, "measured tokens were discarded")
        self.assertEqual(usage.completion_tokens, 42)
        self.assertTrue(usage.usage_incomplete)

    def test_retry_after_failure_keeps_accumulating(self) -> None:
        """The real Ornith path: success, recovered failure, success."""
        usage = TokenUsageAccumulator()
        usage.add_result(_result(881, 42))
        usage.add_failed_call()
        usage.add_result(_result(900, 50))
        self.assertEqual(usage.prompt_tokens, 1781)
        self.assertEqual(usage.completion_tokens, 92)
        self.assertEqual(usage.model_calls, 3)
        self.assertEqual(usage.failed_calls, 1)
        self.assertTrue(usage.usage_incomplete)

    def test_multiple_failures_then_success(self) -> None:
        usage = TokenUsageAccumulator()
        usage.add_result(_result(100, 10))
        usage.add_failed_call()
        usage.add_failed_call()
        usage.add_result(_result(200, 20))
        self.assertEqual(usage.prompt_tokens, 300)
        self.assertEqual(usage.model_calls, 4)
        self.assertEqual(usage.failed_calls, 2)

    def test_failure_counted_exactly_once(self) -> None:
        usage = TokenUsageAccumulator()
        before_calls, before_failed = usage.model_calls, usage.failed_calls
        usage.add_failed_call()
        self.assertEqual(usage.model_calls, before_calls + 1)
        self.assertEqual(usage.failed_calls, before_failed + 1)

    def test_cache_metrics_are_not_poisoned_by_a_failure(self) -> None:
        """Unknownness must stay with the unknown quantity."""
        usage = TokenUsageAccumulator()
        usage.add_result(_result(1000, 10, cached=400))
        usage.add_failed_call()
        self.assertEqual(usage.cached_tokens, 400)
        self.assertEqual(usage.evaluated_tokens, 600)

    def test_snapshot_carries_completeness(self) -> None:
        usage = TokenUsageAccumulator()
        usage.add_result(_result(881, 42))
        usage.add_failed_call()
        snap = usage.snapshot()
        self.assertIsInstance(snap, TurnTokenUsage)
        self.assertEqual(snap.prompt_tokens, 881)
        self.assertEqual(snap.failed_calls, 1)
        self.assertTrue(snap.usage_incomplete)


class RenderingTests(unittest.TestCase):
    """Incomplete totals must not be presented as exact."""

    def _turn_line(self, usage: TurnTokenUsage) -> str:
        from orbit.terminal.status import _token_metric_lines

        return "\n".join(_token_metric_lines(usage, columns=200))

    def test_partial_total_is_marked(self) -> None:
        usage = TurnTokenUsage(
            model_calls=3, prompt_tokens=1781, evaluated_tokens=1781,
            cached_tokens=0, completion_tokens=92, failed_calls=1,
            usage_incomplete=True,
        )
        line = self._turn_line(usage)
        self.assertIn("1,781 in", line, "known figures must still be shown")
        self.assertIn("(partial)", line, "incomplete total presented as exact")

    def test_complete_total_is_not_marked(self) -> None:
        usage = TurnTokenUsage(
            model_calls=1, prompt_tokens=881, evaluated_tokens=881,
            cached_tokens=0, completion_tokens=42, failed_calls=0,
        )
        line = self._turn_line(usage)
        self.assertIn("881 in", line)
        self.assertNotIn("(partial)", line)

    def test_session_wording_says_attempts(self) -> None:
        from orbit.terminal.status import format_session_token_usage

        usage = TurnTokenUsage(
            model_calls=3, prompt_tokens=1781, evaluated_tokens=1781,
            cached_tokens=0, completion_tokens=92, failed_calls=2,
            usage_incomplete=True,
        )
        text = format_session_token_usage(usage)
        self.assertIn("failed attempts: 2", text)
        self.assertNotIn("token usage unavailable", text)


if __name__ == "__main__":
    unittest.main()
