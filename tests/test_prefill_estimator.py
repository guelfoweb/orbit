from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal.prefill_estimator import (
    FALLBACK_PREFILL_TOKENS_PER_SECOND,
    MAX_PREFILL_TOKENS_PER_SECOND,
    MIN_PREFILL_TOKENS_PER_SECOND,
    PrefillEstimator,
)


class PrefillEstimatorTests(unittest.TestCase):
    def test_uses_fallback_initial_rate(self) -> None:
        estimator = PrefillEstimator()

        self.assertEqual(estimator.rate, FALLBACK_PREFILL_TOKENS_PER_SECOND)
        self.assertEqual(estimator.estimate_seconds(120), 10)

    def test_updates_with_exponential_moving_average(self) -> None:
        estimator = PrefillEstimator()

        estimator.update(prompt_tokens=100, prompt_tokens_per_second=22.0)

        self.assertAlmostEqual(estimator.rate, 14.0)

    def test_ignores_missing_or_invalid_metrics(self) -> None:
        estimator = PrefillEstimator()

        estimator.update(prompt_tokens=100, prompt_tokens_per_second=None)
        estimator.update(prompt_tokens=100, prompt_tokens_per_second=0)

        self.assertEqual(estimator.rate, FALLBACK_PREFILL_TOKENS_PER_SECOND)

    def test_clamps_observed_rates(self) -> None:
        low = PrefillEstimator()
        low.update(prompt_tokens=100, prompt_tokens_per_second=1.0)

        high = PrefillEstimator()
        high.update(prompt_tokens=100, prompt_tokens_per_second=200.0)

        self.assertGreaterEqual(low.rate, MIN_PREFILL_TOKENS_PER_SECOND)
        self.assertLessEqual(high.rate, MAX_PREFILL_TOKENS_PER_SECOND)

    def test_refuses_non_positive_token_estimates(self) -> None:
        estimator = PrefillEstimator()

        self.assertIsNone(estimator.estimate_seconds(0))
        self.assertIsNone(estimator.estimate_seconds(-1))


if __name__ == "__main__":
    unittest.main()
