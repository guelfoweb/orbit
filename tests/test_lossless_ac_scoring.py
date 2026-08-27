"""Deterministic scoring of the fresh lossless A+C corpus.

Evaluation only. These tests pin the conclusions drawn from that corpus so a
later change to the scorer cannot silently rewrite them, and they assert the
scorer reaches no model.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(ROOT / "src"))

from score_completion_shadow import (  # noqa: E402
    classify_a,
    classify_b,
    evaluate_finding,
    score_text,
)

CORPUS = Path(
    "/home/guelfoweb/LAB/orbit-checkpoints/completion-shadow-lossless-ac-20260827-175528"
)
ORACLES = json.loads(
    (ROOT / "scripts/evaluation/completion_shadow_oracles.json").read_text()
)["samples"]


def _checkpoint(label: str, action: int) -> dict:
    from orbit.runtime.completion_shadow_ledger import read_ledger

    led = read_ledger(CORPUS / f"runs/{label}/completion-shadow.jsonl")
    return next(c for c in led.checkpoints if c["action"] == action)


def _text(checkpoint: dict) -> str:
    return "\n".join(e["text"] for e in checkpoint["snapshot_evidence"])


def _findings_added_after_action_4(label: str) -> list[str]:
    """Required findings the FULL EvidenceStore has that action 4 did not.

    The authoritative comparison: everything the analysis ever established,
    at full length, against what the verifiers were shown at action 4.
    """
    from orbit.runtime.evidence import EvidenceStore

    store = EvidenceStore(root=CORPUS / f"runs/{label}/evidence")
    store.load_index()
    full = "\n".join(
        (store.load_raw(r.evidence_id) or "") for r in store.records.values()
    )
    at4 = _text(_checkpoint(label, 4))
    return [
        spec["id"]
        for spec in ORACLES[label]["required_findings"]
        if evaluate_finding(spec, full).satisfied
        and not evaluate_finding(spec, at4).satisfied
    ]


class CorpusPresent(unittest.TestCase):
    def setUp(self) -> None:
        if not CORPUS.is_dir():
            self.skipTest("fresh lossless corpus not present")


class RunAScoringTests(CorpusPresent):
    def test_a4_is_oracle_complete(self) -> None:
        score = score_text(ORACLES["A"], _text(_checkpoint("A", 4)), 4)
        self.assertEqual(score.state, "COMPLETE")
        self.assertEqual(score.missing, [])

    def test_a4_verifiers_are_both_correct(self) -> None:
        cp = _checkpoint("A", 4)
        self.assertEqual(classify_a(cp["verifier_a"], True, False), "A_TRUE_COMPLETE")
        self.assertEqual(classify_b(cp["verifier_b"], True, False), "B_TRUE_NO_GAP")
        self.assertTrue(cp["would_stop"])

    def test_no_material_finding_appears_after_action_4(self) -> None:
        """The counterfactual: stopping at 4 would have lost nothing required.

        Compared against the FULL EvidenceStore, not the final snapshot. The
        final snapshot is itself a bounded projection and is strictly poorer
        than action 4 -- it satisfies fewer findings -- so testing against it
        would be the weaker claim, and could pass while the run had in fact
        discovered something new.
        """
        self.assertEqual(_findings_added_after_action_4("A"), [])


class RunCScoringTests(CorpusPresent):
    def test_c4_has_every_required_finding(self) -> None:
        score = score_text(ORACLES["C"], _text(_checkpoint("C", 4)), 4)
        self.assertEqual(score.state, "COMPLETE")
        self.assertEqual(score.missing, [], "all 8 material findings present")

    def test_c4_carries_the_correct_domain_and_not_the_stale_one(self) -> None:
        text = _text(_checkpoint("C", 4))
        self.assertIn("gibuzuy37v2v.top", text)
        self.assertNotIn("gibuyuy37v2v.top", text)

    def test_stale_domain_could_not_satisfy_the_finding(self) -> None:
        spec = next(
            f for f in ORACLES["C"]["required_findings"] if f["id"] == "c2_domain"
        )
        self.assertFalse(evaluate_finding(spec, "C2 is gibuyuy37v2v.top").satisfied)

    def test_c4_b_gap_is_a_false_gap(self) -> None:
        # B's stated omission -- the second-stage payload body -- is real prose
        # but is not an oracle requirement, and the sandbox has no network, so
        # it is unsatisfiable by construction rather than merely unmet.
        cp = _checkpoint("C", 4)
        self.assertEqual(cp["verifier_b"], "GAP")
        self.assertEqual(classify_b(cp["verifier_b"], True, False), "B_FALSE_GAP")

    def test_no_material_finding_appears_after_action_4(self) -> None:
        self.assertEqual(_findings_added_after_action_4("C"), [])

    def test_the_full_store_is_richer_than_the_final_snapshot(self) -> None:
        """Guards the guard: if the full store were no larger than the bounded
        final snapshot, the stronger comparison above would prove nothing."""
        from orbit.runtime.completion_shadow_ledger import read_ledger
        from orbit.runtime.evidence import EvidenceStore

        led = read_ledger(CORPUS / "runs/C/completion-shadow.jsonl")
        bounded = "\n".join(
            e["text"] for e in (led.run_final.get("final_snapshot_evidence") or [])
        )
        store = EvidenceStore(root=CORPUS / "runs/C/evidence")
        store.load_index()
        full = "\n".join(
            (store.load_raw(r.evidence_id) or "") for r in store.records.values()
        )
        self.assertGreater(len(full), len(bounded) * 2)


class IncompleteStateIsReachableTests(CorpusPresent):
    """Every state in this corpus is COMPLETE, so the INCOMPLETE path needs
    pinning directly -- otherwise a scorer that always answered COMPLETE would
    reproduce these results exactly."""

    def test_removing_a_required_finding_yields_incomplete(self) -> None:
        text = _text(_checkpoint("C", 4))
        without = text.replace("gibuzuy37v2v.top", "REDACTED")
        score = score_text(ORACLES["C"], without, 4)
        self.assertEqual(score.state, "INCOMPLETE")
        self.assertIn("c2_domain", score.missing)

    def test_a_stale_only_snapshot_is_incomplete(self) -> None:
        # The trap in its natural setting: an analysis that established only
        # the superseded spelling has not established the C2.
        text = _text(_checkpoint("C", 4)).replace(
            "gibuzuy37v2v.top", "gibuyuy37v2v.top"
        )
        score = score_text(ORACLES["C"], text, 4)
        self.assertEqual(score.state, "INCOMPLETE")
        self.assertIn("c2_domain", score.missing)

    def test_an_empty_snapshot_is_incomplete_not_complete(self) -> None:
        score = score_text(ORACLES["C"], "", 4)
        self.assertEqual(score.state, "INCOMPLETE")
        self.assertEqual(len(score.missing), 8)


class SkippedCheckpointTests(CorpusPresent):
    def test_skipped_checkpoints_are_never_graded_as_observed(self) -> None:
        """A budget skip is an absent observation, not a verifier mistake."""
        from orbit.runtime.completion_shadow_ledger import read_ledger

        led = read_ledger(CORPUS / "runs/C/completion-shadow.jsonl")
        skipped = [c for c in led.checkpoints if c.get("verification_skipped")]
        self.assertEqual([c["action"] for c in skipped], [6, 8, 10, 12])
        for cp in skipped:
            self.assertIsNone(cp["verifier_a"])
            self.assertIsNone(cp["verifier_b"])
            self.assertEqual(cp["blocked_by"], "snapshot_too_large")
            self.assertEqual(classify_a(cp["verifier_a"], True, False), "A_NOT_CALLED")
            self.assertEqual(classify_b(cp["verifier_b"], True, False), "B_NOT_CALLED")

    def test_skipped_checkpoints_cannot_would_stop(self) -> None:
        from orbit.runtime.completion_shadow_ledger import read_ledger

        led = read_ledger(CORPUS / "runs/C/completion-shadow.jsonl")
        for cp in led.checkpoints:
            if cp.get("verification_skipped"):
                self.assertFalse(cp["would_stop"])


class CounterfactualTests(CorpusPresent):
    def test_a_only_would_have_stopped_both_runs_at_action_4(self) -> None:
        for label in ("A", "C"):
            with self.subTest(run=label):
                self.assertEqual(_checkpoint(label, 4)["verifier_a"], "COMPLETE")

    def test_a_only_makes_no_false_stop_on_this_corpus(self) -> None:
        # Both action-4 states are oracle COMPLETE, so an A-only stop there
        # loses nothing required. This is a fact about these two runs, not a
        # general safety claim about A.
        for label in ("A", "C"):
            with self.subTest(run=label):
                score = score_text(
                    ORACLES[label], _text(_checkpoint(label, 4)), 4
                )
                self.assertEqual(score.state, "COMPLETE")

    def test_a_plus_b_never_stops_run_c(self) -> None:
        self.assertEqual(_checkpoint("C", 4)["verifier_b"], "GAP")


class ScorerIsOfflineTests(unittest.TestCase):
    def test_scoring_makes_no_network_connection(self) -> None:
        import socket
        import urllib.request
        from unittest import mock

        if not CORPUS.is_dir():
            self.skipTest("fresh lossless corpus not present")
        from score_completion_shadow import score_corpus

        def refuse(*_a, **_k):
            raise AssertionError("scorer attempted a network connection")

        with mock.patch.object(socket.socket, "connect", refuse), \
             mock.patch.object(socket, "create_connection", refuse), \
             mock.patch.object(socket, "getaddrinfo", refuse), \
             mock.patch.object(urllib.request, "urlopen", refuse):
            report = score_corpus(
                CORPUS,
                {"samples": ORACLES},
            )
        self.assertEqual(sorted(report["runs"]), ["A", "C"])


if __name__ == "__main__":
    unittest.main()
