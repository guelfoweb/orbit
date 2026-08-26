"""Shadow completion gate: it must observe, and never decide."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from orbit.runtime.completion_shadow import (
    SHADOW_ENV,
    CompletionSnapshot,
    ShadowLedger,
    build_snapshot,
    evaluate_completion_shadow,
    parse_verifier_a,
    parse_verifier_b,
    scheduled_actions,
    shadow_enabled,
)


@dataclass
class _Response:
    content: str
    prompt_tokens: int = 100
    completion_tokens: int = 8


@dataclass
class _Record:
    evidence_id: str
    metadata: dict


def _snapshot(digest: str = "d" * 64) -> CompletionSnapshot:
    return CompletionSnapshot(
        request="extract the C2", evidence=(("ev_a", "seen"),), artifacts=(), digest=digest
    )


def _run(answers, *, active=("ev_000000000000_0000000000000000",), reattest_ok=True):
    """Drive the gate with scripted verifier answers."""
    seen: list[str] = []

    def ask(instruction: str, rendered: str):
        seen.append(instruction)
        return _Response(answers[len(seen) - 1])

    observation = evaluate_completion_shadow(
        action=4,
        snapshot=_snapshot(),
        ask=ask,
        active_evidence_ids=set(active),
        reattest=(lambda _id: "raw" if reattest_ok else None),
    )
    return observation, seen


class ScheduleTests(unittest.TestCase):
    def test_default_is_off(self) -> None:
        self.assertFalse(shadow_enabled({}))
        self.assertFalse(shadow_enabled({SHADOW_ENV: "0"}))
        self.assertFalse(shadow_enabled({SHADOW_ENV: "true"}))

    def test_explicit_opt_in(self) -> None:
        self.assertTrue(shadow_enabled({SHADOW_ENV: "1"}))

    def test_schedules_are_deterministic(self) -> None:
        due = scheduled_actions("after4every2")
        self.assertEqual([a for a in range(1, 13) if due(a)], [4, 6, 8, 10, 12])

    def test_unknown_schedule_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scheduled_actions("adaptive")


class ParsingTests(unittest.TestCase):
    def test_complete_parses_and_collects_ids(self) -> None:
        verdict = parse_verifier_a(
            "COMPLETE evidence: ev_271d2d4de64d_752e09e7c1a2673a"
        )
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.decision, "COMPLETE")
        self.assertEqual(verdict.evidence_ids, ("ev_271d2d4de64d_752e09e7c1a2673a",))

    def test_continue_parses(self) -> None:
        verdict = parse_verifier_a("CONTINUE missing: the decoded URL")
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.decision, "CONTINUE")

    def test_ambiguous_answer_is_continue(self) -> None:
        # Naming both words is not a completion, however it is phrased.
        for text in ("COMPLETE but CONTINUE checking", "maybe complete", "", "   "):
            with self.subTest(text=text):
                self.assertEqual(parse_verifier_a(text).decision, "CONTINUE")

    def test_b_requires_clean_no_gap(self) -> None:
        self.assertEqual(parse_verifier_b("NO_GAP").decision, "NO_GAP")
        self.assertTrue(parse_verifier_b("NO_GAP").ok)
        for text in ("GAP missing: the C2 host", "", "unsure", "NO_GAP missing: x"):
            with self.subTest(text=text):
                verdict = parse_verifier_b(text)
                self.assertNotEqual((verdict.ok, verdict.decision), (True, "NO_GAP"))


class ParserInvariantTests(unittest.TestCase):
    """The gate's redundancy rests on this: an unparsed verdict is never
    affirmative. If a parser ever returned `ok=False` alongside COMPLETE or
    NO_GAP, the `ok` checks in the gate would be the only thing standing
    between a malformed answer and a WOULD_STOP."""

    CASES = (
        "COMPLETE", "CONTINUE", "NO_GAP", "GAP", "banana", "", "   ",
        "complete", "Complete", "COMPLETE CONTINUE", "COMPLETE\nCONTINUE",
        "xx COMPLETE", "NO_GAP missing: x", "COMPLETE evidence: ev_a",
    )

    def test_unparsed_a_is_never_complete(self) -> None:
        for text in self.CASES:
            with self.subTest(text=text):
                verdict = parse_verifier_a(text)
                if not verdict.ok:
                    self.assertNotEqual(verdict.decision, "COMPLETE")

    def test_unparsed_b_is_never_no_gap(self) -> None:
        for text in self.CASES:
            with self.subTest(text=text):
                verdict = parse_verifier_b(text)
                if not verdict.ok:
                    self.assertNotEqual(verdict.decision, "NO_GAP")


class GateTests(unittest.TestCase):
    def test_complete_citing_nothing_is_refused(self) -> None:
        # An uncheckable completion is not a completion: with no cited id the
        # re-attestation gate has nothing to verify.
        observation, seen = _run(["COMPLETE", "NO_GAP"])
        self.assertFalse(observation.would_stop)
        self.assertEqual(observation.blocked_by, "verifier_a_cited_no_evidence")
        self.assertEqual(len(seen), 1, "B must not be paid for an uncited claim")

    def test_complete_and_no_gap_would_stop(self) -> None:
        observation, seen = _run(["COMPLETE evidence: ev_000000000000_0000000000000000", "NO_GAP"])
        self.assertTrue(observation.would_stop)
        self.assertIsNone(observation.blocked_by)
        self.assertEqual(len(seen), 2)

    def test_a_continue_never_calls_b(self) -> None:
        observation, seen = _run(["CONTINUE missing: the payload"])
        self.assertFalse(observation.would_stop)
        self.assertEqual(len(seen), 1)
        self.assertIsNone(observation.verifier_b)
        self.assertEqual(observation.blocked_by, "verifier_a_continue")

    def test_b_gap_blocks(self) -> None:
        observation, _ = _run(["COMPLETE evidence: ev_000000000000_0000000000000000", "GAP missing: the C2"])
        self.assertFalse(observation.would_stop)
        self.assertEqual(observation.blocked_by, "verifier_b_gap")

    def test_malformed_a_blocks(self) -> None:
        observation, seen = _run(["banana"])
        self.assertFalse(observation.would_stop)
        self.assertEqual(len(seen), 1)

    def test_malformed_b_blocks(self) -> None:
        observation, _ = _run(["COMPLETE evidence: ev_000000000000_0000000000000000", "banana"])
        self.assertFalse(observation.would_stop)
        self.assertEqual(observation.blocked_by, "verifier_b_unparsed")

    def test_b_cannot_see_a_verdict(self) -> None:
        seen: list[str] = []

        def ask(instruction: str, rendered: str):
            seen.append(instruction + "||" + rendered)
            return _Response("COMPLETE evidence: ev_000000000000_0000000000000000" if len(seen) == 1 else "NO_GAP")

        evaluate_completion_shadow(
            action=4,
            snapshot=_snapshot(),
            ask=ask,
            active_evidence_ids={"ev_000000000000_0000000000000000"},
            reattest=lambda _id: "raw",
        )
        self.assertNotIn("COMPLETE", seen[1])
        self.assertNotIn("verifier_a", seen[1].lower())

    def test_both_verifiers_see_the_same_snapshot(self) -> None:
        rendered: list[str] = []

        def ask(instruction: str, text: str):
            rendered.append(text)
            return _Response("COMPLETE evidence: ev_000000000000_0000000000000000" if len(rendered) == 1 else "NO_GAP")

        evaluate_completion_shadow(
            action=4,
            snapshot=_snapshot(),
            ask=ask,
            active_evidence_ids={"ev_000000000000_0000000000000000"},
            reattest=lambda _id: "raw",
        )
        self.assertEqual(rendered[0], rendered[1])

    def test_stale_evidence_reference_blocks(self) -> None:
        observation, seen = _run(
            ["COMPLETE evidence: ev_111111111111_2222222222222222", "NO_GAP"],
            active=("ev_000000000000_0000000000000000",),
        )
        self.assertFalse(observation.would_stop)
        self.assertEqual(observation.blocked_by, "referenced_evidence_not_active")
        self.assertEqual(len(seen), 1, "B must not be paid for a dead reference")

    def test_failed_reattest_blocks(self) -> None:
        observation, _ = _run(
            ["COMPLETE evidence: ev_000000000000_0000000000000000", "NO_GAP"],
            reattest_ok=False,
        )
        self.assertFalse(observation.would_stop)
        self.assertEqual(observation.blocked_by, "referenced_evidence_reattest_failed")

    def test_keyboard_interrupt_is_contained(self) -> None:
        """Ctrl-C during a verifier must not escape into the loop.

        The loop guards KeyboardInterrupt around its own step only, so an
        interrupt raised here would unwind past `run_autonomous` to a caller
        holding a pre-run checkpoint and delete the history and provenance of
        every completed step. Enabling a diagnostic must not create a window
        in which Ctrl-C is destructive.
        """

        def ask(instruction: str, rendered: str):
            raise KeyboardInterrupt()

        observation = evaluate_completion_shadow(
            action=4,
            snapshot=_snapshot(),
            ask=ask,
            active_evidence_ids=set(),
            reattest=lambda _id: "raw",
        )
        self.assertFalse(observation.would_stop)
        self.assertEqual(observation.blocked_by, "verifier_error: KeyboardInterrupt")

    def test_system_exit_is_contained(self) -> None:
        def ask(instruction: str, rendered: str):
            raise SystemExit(1)

        observation = evaluate_completion_shadow(
            action=4,
            snapshot=_snapshot(),
            ask=ask,
            active_evidence_ids=set(),
            reattest=lambda _id: "raw",
        )
        self.assertFalse(observation.would_stop)

    def test_verifier_exception_blocks(self) -> None:
        def ask(instruction: str, rendered: str):
            raise RuntimeError("backend down")

        observation = evaluate_completion_shadow(
            action=4,
            snapshot=_snapshot(),
            ask=ask,
            active_evidence_ids={"ev_000000000000_0000000000000000"},
            reattest=lambda _id: "raw",
        )
        self.assertFalse(observation.would_stop)
        self.assertTrue(observation.blocked_by.startswith("verifier_error"))


class SnapshotTests(unittest.TestCase):
    def test_snapshot_is_content_addressed(self) -> None:
        records = [_Record("ev_a", {"artifacts": [{"handle": "/w/x.txt"}]})]
        one = build_snapshot(request="r", records=records, load_raw=lambda _i: "text")
        two = build_snapshot(request="r", records=records, load_raw=lambda _i: "text")
        three = build_snapshot(request="r", records=records, load_raw=lambda _i: "other")
        self.assertEqual(one.digest, two.digest)
        self.assertNotEqual(one.digest, three.digest)
        self.assertIn("/w/x.txt", one.render())

    def test_snapshot_is_bounded(self) -> None:
        records = [_Record(f"ev_{i}", {}) for i in range(50)]
        snapshot = build_snapshot(
            request="r", records=records, load_raw=lambda _i: "x" * 5000
        )
        self.assertLessEqual(len(snapshot.evidence), 12)
        self.assertTrue(all(len(text) <= 600 for _id, text in snapshot.evidence))


class AccountingTests(unittest.TestCase):
    def test_ledger_sums_separately_from_the_loop(self) -> None:
        observation, _ = _run(["COMPLETE evidence: ev_000000000000_0000000000000000", "NO_GAP"])
        ledger = ShadowLedger(observations=[observation])
        self.assertEqual(ledger.calls, 2)
        self.assertEqual(ledger.prompt_tokens, 200)
        self.assertEqual(ledger.output_tokens, 16)
        self.assertEqual(ledger.would_stop_actions, [4])


if __name__ == "__main__":
    unittest.main()
