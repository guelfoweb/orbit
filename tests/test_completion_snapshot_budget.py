"""Token-budgeted lossless snapshots: fit exactly, or do not ask at all."""

from __future__ import annotations

import unittest

from orbit.runtime.completion_shadow import (
    COMPLETION_SNAPSHOT_TOKEN_BUDGET,
    VERIFIER_A_INSTRUCTION,
    VERIFIER_B_INSTRUCTION,
    VERIFIER_MAX_TOKENS,
    CompletionSnapshot,
    build_lossless_snapshot,
    evaluate_completion_shadow,
    snapshot_fits_budget,
)


class _Record:
    def __init__(self, evidence_id: str, artifacts=()) -> None:
        self.evidence_id = evidence_id
        self.metadata = {"artifacts": [{"handle": h} for h in artifacts]}


def _words(text: str) -> int:
    return len(text.split())


class LosslessConstructionTests(unittest.TestCase):
    def test_no_record_content_is_truncated(self) -> None:
        huge = "x" * 100_000
        s = build_lossless_snapshot(
            request="r", records=[_Record("ev_a")], load_raw=lambda _i: huge
        )
        self.assertEqual(s.evidence[0][1], huge)
        self.assertTrue(s.fidelity.lossless)
        self.assertEqual(s.fidelity.truncated_records, ())

    def test_no_record_is_omitted_to_fit(self) -> None:
        records = [_Record(f"ev_{i}") for i in range(60)]
        s = build_lossless_snapshot(request="r", records=records, load_raw=lambda _i: "t")
        self.assertEqual(len(s.evidence), 60)
        self.assertEqual(s.fidelity.omitted_record_count, 0)
        self.assertTrue(s.fidelity.lossless)

    def test_no_artifact_is_omitted_to_fit(self) -> None:
        handles = [f"/w/{i}" for i in range(50)]
        s = build_lossless_snapshot(
            request="r", records=[_Record("ev_a", handles)], load_raw=lambda _i: "t"
        )
        self.assertEqual(len(s.artifacts), 50)
        self.assertEqual(s.fidelity.omitted_artifact_count, 0)

    def test_request_is_included_whole(self) -> None:
        request = "q" * 50_000
        s = build_lossless_snapshot(
            request=request, records=[_Record("ev_a")], load_raw=lambda _i: "t"
        )
        self.assertEqual(s.request, request)
        self.assertFalse(s.fidelity.request_truncated)

    def test_an_unusable_record_is_still_reported(self) -> None:
        s = build_lossless_snapshot(
            request="r", records=[_Record(""), _Record("ev_a")], load_raw=lambda _i: "t"
        )
        self.assertFalse(s.fidelity.lossless)
        self.assertIn("record_unusable", s.fidelity.reasons)


class BudgetBoundaryTests(unittest.TestCase):
    """Off-by-one at the budget is the failure that would matter."""

    def _sized(self, total_tokens: int):
        """Build a snapshot whose counted prompt is exactly `total_tokens`.

        The template wrapper is part of that total, so it is subtracted here
        too -- otherwise these boundary tests would be measuring a different
        quantity from the one the budget actually bounds.
        """
        from orbit.runtime.completion_shadow import CHAT_TEMPLATE_OVERHEAD_TOKENS

        snapshot = CompletionSnapshot("r", (("ev_a", "t"),), (), "d" * 64)
        overhead = max(_words(VERIFIER_A_INSTRUCTION), _words(VERIFIER_B_INSTRUCTION))
        body = total_tokens - overhead - CHAT_TEMPLATE_OVERHEAD_TOKENS

        def count(text: str) -> int:
            return body if text == snapshot.render() else _words(text)

        return snapshot, count

    def test_exactly_at_budget_fits(self) -> None:
        snapshot, count = self._sized(COMPLETION_SNAPSHOT_TOKEN_BUDGET)
        fits, total = snapshot_fits_budget(snapshot, count)
        self.assertTrue(fits)
        self.assertEqual(total, COMPLETION_SNAPSHOT_TOKEN_BUDGET)

    def test_one_token_over_budget_does_not_fit(self) -> None:
        snapshot, count = self._sized(COMPLETION_SNAPSHOT_TOKEN_BUDGET + 1)
        fits, total = snapshot_fits_budget(snapshot, count)
        self.assertFalse(fits)
        self.assertEqual(total, COMPLETION_SNAPSHOT_TOKEN_BUDGET + 1)

    def test_one_token_under_budget_fits(self) -> None:
        snapshot, count = self._sized(COMPLETION_SNAPSHOT_TOKEN_BUDGET - 1)
        self.assertTrue(snapshot_fits_budget(snapshot, count)[0])

    def test_instruction_is_counted_not_only_the_snapshot(self) -> None:
        # Budgeting the snapshot alone would let the real prompt exceed it.
        from orbit.runtime.completion_shadow import CHAT_TEMPLATE_OVERHEAD_TOKENS

        snapshot = CompletionSnapshot("r", (("ev_a", "t"),), (), "d" * 64)
        _fits, total = snapshot_fits_budget(snapshot, _words)
        self.assertEqual(
            total,
            _words(snapshot.render())
            + max(_words(VERIFIER_A_INSTRUCTION), _words(VERIFIER_B_INSTRUCTION))
            + CHAT_TEMPLATE_OVERHEAD_TOKENS,
            "the counted prompt must be snapshot + instruction + template wrapper",
        )


class SkipInvariantTests(unittest.TestCase):
    """VERIFIER CALLED => SNAPSHOT LOSSLESS AND WITHIN BUDGET."""

    def _observe(self, *, fits: bool):
        calls: list[str] = []

        class _R:
            content = "COMPLETE evidence: ev_000000000000_0000000000000000"
            prompt_tokens = 10
            completion_tokens = 2

        def ask(instruction, _rendered):
            calls.append(instruction)
            return _R()

        obs = evaluate_completion_shadow(
            action=4,
            snapshot=build_lossless_snapshot(
                request="r", records=[_Record("ev_a")], load_raw=lambda _i: "t"
            ),
            ask=ask,
            active_evidence_ids={"ev_000000000000_0000000000000000"},
            reattest=lambda _i: "raw",
            fits_budget=fits,
            snapshot_tokens=999,
        )
        return obs, calls

    def test_oversized_snapshot_calls_no_verifier_at_all(self) -> None:
        obs, calls = self._observe(fits=False)
        self.assertEqual(calls, [], "neither A nor B may be asked")
        self.assertEqual(obs.blocked_by, "snapshot_too_large")
        self.assertTrue(obs.verification_skipped)
        self.assertEqual(obs.calls, 0)

    def test_skipped_verification_cannot_would_stop(self) -> None:
        obs, _ = self._observe(fits=False)
        self.assertFalse(obs.would_stop)

    def test_b_is_not_called_when_a_is_skipped(self) -> None:
        obs, calls = self._observe(fits=False)
        self.assertIsNone(obs.verifier_a)
        self.assertIsNone(obs.verifier_b)
        self.assertEqual(len(calls), 0)

    def test_within_budget_runs_the_verifier(self) -> None:
        obs, calls = self._observe(fits=True)
        self.assertGreaterEqual(len(calls), 1)
        self.assertFalse(obs.verification_skipped)
        self.assertNotEqual(obs.blocked_by, "snapshot_too_large")

    def test_snapshot_tokens_are_recorded_either_way(self) -> None:
        for fits in (True, False):
            with self.subTest(fits=fits):
                obs, _ = self._observe(fits=fits)
                self.assertEqual(obs.snapshot_tokens, 999)


class RuntimeWiringTests(unittest.TestCase):
    """The call site must build losslessly and must not estimate tokens.

    Both were mutants that survived a behavioural-only suite: the fixture
    tokenizer is generous enough that a truncating builder still fits, and an
    estimating fallback still produces a number. Pinning the call site is what
    catches them.
    """

    def _source(self) -> str:
        import inspect

        from orbit.runtime import analysis_runtime

        return inspect.getsource(analysis_runtime.AnalysisRuntime._observe_completion_shadow)

    def test_observation_builds_the_lossless_snapshot(self) -> None:
        source = self._source()
        self.assertIn("build_lossless_snapshot(", source)
        self.assertNotIn("snapshot = build_snapshot(", source)

    def test_token_count_failure_is_never_estimated(self) -> None:
        # Comments are stripped first: the word "estimate" appears in the
        # prose explaining why estimating is forbidden, and matching that
        # would be checking the documentation rather than the code.
        import ast

        source = self._source()
        self.assertIn("_TokenCountUnavailable", source)
        self.assertIn("count_text_tokens", source)
        import textwrap

        code = ast.unparse(ast.parse(textwrap.dedent(source)))
        for estimate in ("len(text) // 4", "len(text)//4", "// 3.5", "* 0.25"):
            self.assertNotIn(estimate, code, "the budget must never be estimated")

    def test_unavailable_tokenizer_skips_rather_than_guesses(self) -> None:
        from orbit.runtime.completion_shadow import evaluate_completion_shadow as run

        calls: list[str] = []
        obs = run(
            action=4,
            snapshot=build_lossless_snapshot(
                request="r", records=[_Record("ev_a")], load_raw=lambda _i: "t"
            ),
            ask=lambda i, _r: calls.append(i),
            active_evidence_ids=set(),
            reattest=lambda _i: "raw",
            fits_budget=False,
            snapshot_tokens=None,
        )
        self.assertEqual(calls, [])
        self.assertEqual(obs.blocked_by, "snapshot_too_large")
        self.assertIsNone(obs.snapshot_tokens)


class ContextSafetyTests(unittest.TestCase):
    def test_worst_case_prompt_plus_generation_fits_the_qualified_context(self) -> None:
        # The budget bounds the whole prompt, so the worst case is fixed rather
        # than dependent on any particular analysis.
        worst = COMPLETION_SNAPSHOT_TOKEN_BUDGET + VERIFIER_MAX_TOKENS
        self.assertLess(worst, 16384)
        self.assertLess(worst, 16384 * 0.5, "leave the analysis most of its context")


if __name__ == "__main__":
    unittest.main()
