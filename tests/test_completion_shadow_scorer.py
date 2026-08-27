"""Deterministic scorer for the completion-shadow corpus.

Evaluation-only code. These tests pin the predicates and the conservative
rules, and assert the scorer cannot reach a model.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

from score_completion_shadow import (  # noqa: E402
    CheckpointScore,
    classify_a,
    classify_b,
    classify_gap,
    evaluate_finding,
    score_text,
)

ORACLES = json.loads(
    (Path(__file__).resolve().parents[1]
     / "scripts/evaluation/completion_shadow_oracles.json").read_text()
)


class PredicateTests(unittest.TestCase):
    def test_literal_present_and_absent(self) -> None:
        spec = {"id": "x", "kind": "literal_all", "values": ["curl"]}
        self.assertTrue(evaluate_finding(spec, "uses curl -useb").satisfied)
        self.assertFalse(evaluate_finding(spec, "uses wget").satisfied)

    def test_literal_all_needs_every_value(self) -> None:
        # Pins `all` rather than `any`: a multi-value requirement is not met by
        # one of its parts, which is what makes a conjunctive finding meaningful.
        spec = {"id": "x", "kind": "literal_all", "values": ["alpha", "beta"]}
        self.assertTrue(evaluate_finding(spec, "alpha and beta").satisfied)
        self.assertFalse(evaluate_finding(spec, "alpha only").satisfied)
        self.assertFalse(evaluate_finding(spec, "beta only").satisfied)

    def test_literal_any_needs_only_one(self) -> None:
        spec = {"id": "x", "kind": "literal_any", "values": ["alpha", "beta"]}
        self.assertTrue(evaluate_finding(spec, "beta only").satisfied)
        self.assertFalse(evaluate_finding(spec, "gamma").satisfied)

    def test_absent_all_is_inverted(self) -> None:
        """Inverted, and scored against source bytes rather than narrative.

        Absence is a property of the artifact. Deciding it from cumulative
        narrative let an analysis step that *explains* the artifact makes no
        such call introduce the token and unsatisfy the finding.
        """
        spec = {"id": "x", "kind": "absent_all", "values": ["eval("]}
        self.assertTrue(
            evaluate_finding(spec, "narrative", source_text="safe code").satisfied
        )
        self.assertFalse(
            evaluate_finding(spec, "narrative", source_text="eval(payload)").satisfied
        )

    def test_absent_all_without_source_is_unscorable(self) -> None:
        spec = {"id": "x", "kind": "absent_all", "values": ["eval("]}
        self.assertTrue(evaluate_finding(spec, "safe code").unscorable)

    def test_regex_is_bounded_not_semantic(self) -> None:
        spec = {"id": "x", "kind": "regex", "pattern": r"(?i)Get-Date"}
        self.assertTrue(evaluate_finding(spec, "$t = Get-Date").satisfied)
        self.assertFalse(evaluate_finding(spec, "the current time").satisfied)

    def test_unknown_kind_is_unscorable_not_guessed(self) -> None:
        result = evaluate_finding({"id": "x", "kind": "vibes"}, "anything")
        self.assertTrue(result.unscorable)
        self.assertFalse(result.satisfied)


class StaleEvidenceTrapTests(unittest.TestCase):
    """The superseded domain must never satisfy the authoritative finding."""

    SPEC = next(
        f for f in ORACLES["samples"]["C"]["required_findings"] if f["id"] == "c2_domain"
    )

    def test_wrong_domain_alone_does_not_satisfy(self) -> None:
        self.assertFalse(evaluate_finding(self.SPEC, "C2 is gibuyuy37v2v.top").satisfied)

    def test_correct_domain_satisfies(self) -> None:
        self.assertTrue(evaluate_finding(self.SPEC, "C2 is gibuzuy37v2v.top").satisfied)

    def test_correction_record_still_satisfies(self) -> None:
        # A record that quotes the error in order to correct it is authoritative.
        text = "report said gibuyuy37v2v.top but the correct value is gibuzuy37v2v.top"
        result = evaluate_finding(self.SPEC, text)
        self.assertTrue(result.satisfied)
        self.assertIn("forbidden present", result.detail)

    def test_neither_domain_does_not_satisfy(self) -> None:
        self.assertFalse(evaluate_finding(self.SPEC, "no domain here").satisfied)


class CompletenessTests(unittest.TestCase):
    ORACLE = {"required_findings": [
        {"id": "one", "kind": "literal_all", "values": ["alpha"]},
        {"id": "two", "kind": "literal_all", "values": ["beta"]},
    ]}

    def test_all_present_is_complete(self) -> None:
        s = score_text(self.ORACLE, "alpha and beta", 4)
        self.assertTrue(s.oracle_complete)
        self.assertEqual(s.state, "COMPLETE")

    def test_one_missing_is_incomplete(self) -> None:
        s = score_text(self.ORACLE, "alpha only", 4)
        self.assertFalse(s.oracle_complete)
        self.assertEqual(s.missing, ["two"])

    def test_unscorable_required_finding_forces_indeterminate(self) -> None:
        oracle = {"required_findings": [
            {"id": "one", "kind": "literal_all", "values": ["alpha"]},
            {"id": "two", "kind": "unscorable"},
        ]}
        s = score_text(oracle, "alpha", 4)
        self.assertEqual(s.state, "INDETERMINATE")
        self.assertFalse(s.oracle_complete, "an undecided requirement is never complete")


class VerifierClassificationTests(unittest.TestCase):
    def test_a_false_complete_detected(self) -> None:
        self.assertEqual(classify_a("COMPLETE", False, False), "A_FALSE_COMPLETE")

    def test_a_true_complete_and_continue(self) -> None:
        self.assertEqual(classify_a("COMPLETE", True, False), "A_TRUE_COMPLETE")
        self.assertEqual(classify_a("CONTINUE", False, False), "A_TRUE_CONTINUE")
        self.assertEqual(classify_a("CONTINUE", True, False), "A_FALSE_CONTINUE")

    def test_b_false_gap_detected(self) -> None:
        self.assertEqual(classify_b("GAP", True, False), "B_FALSE_GAP")

    def test_b_false_no_gap_detected(self) -> None:
        self.assertEqual(classify_b("NO_GAP", False, False), "B_FALSE_NO_GAP")

    def test_b_not_called_when_a_continued(self) -> None:
        """B's absence is read from `blocked_by`, not from a bare missing verdict.

        A_CONTINUE means B was deliberately never asked -- the gate's own cost
        control. Without that reason, a missing verdict is a failed verifier.
        """
        self.assertEqual(
            classify_b(None, False, False, blocked_by="verifier_a_continue"),
            "B_NOT_CALLED",
        )
        self.assertEqual(classify_b(None, False, False), "B_ERRORED")

    def test_indeterminate_dominates(self) -> None:
        self.assertEqual(classify_a("COMPLETE", False, True), "A_INDETERMINATE")
        self.assertEqual(classify_b("GAP", False, True), "B_INDETERMINATE")


class GapMaterialityTests(unittest.TestCase):
    def test_real_gap_when_findings_missing(self) -> None:
        s = CheckpointScore(action=4, required_total=2, missing=["two"])
        self.assertEqual(classify_gap("missing: the C2", s), "MATERIAL_AND_REAL")

    def test_gap_against_complete_oracle_is_not_material(self) -> None:
        s = CheckpointScore(action=4, required_total=2, satisfied=["one", "two"])
        self.assertEqual(
            classify_gap("missing: something", s), "NON_MATERIAL_OR_ALREADY_SATISFIED"
        )

    def test_unscorable_gap_is_not_guessed(self) -> None:
        s = CheckpointScore(action=4, required_total=1, unscorable=["one"])
        self.assertEqual(classify_gap("missing: x", s), "UNSCORABLE")
        self.assertEqual(classify_gap("", CheckpointScore(4, 1)), "UNSCORABLE")


class ScorerIsOfflineTests(unittest.TestCase):
    def test_scorer_makes_no_network_connection(self) -> None:
        """Behavioural, not a substring scan.

        Importing the EvidenceStore transitively pulls in the backend module,
        so a source scan cannot establish this. Blocking the socket layer and
        scoring the real corpus can.
        """
        import socket
        from unittest import mock

        corpus = Path("/home/guelfoweb/LAB/orbit-checkpoints"
                      "/completion-shadow-corpus-20260827-103458")
        if not corpus.is_dir():
            self.skipTest("corpus not present")
        from score_completion_shadow import score_corpus

        def refuse(*_a, **_k):
            raise AssertionError("scorer attempted a network connection")

        import urllib.request

        with mock.patch.object(socket.socket, "connect", refuse), \
             mock.patch.object(socket.socket, "connect_ex", refuse), \
             mock.patch.object(socket, "create_connection", refuse), \
             mock.patch.object(socket, "getaddrinfo", refuse), \
             mock.patch.object(urllib.request, "urlopen", refuse):
            report = score_corpus(corpus, ORACLES)
        self.assertEqual(sorted(report["runs"]), ["A", "B", "C"])

    def test_scorer_never_imports_a_backend_or_network(self) -> None:
        source = (Path(__file__).resolve().parents[1]
                  / "scripts/evaluation/score_completion_shadow.py").read_text()
        for forbidden in ("requests", "urllib", "httpx", "socket",
                          "LlamaServerBackend", "complete_chat", "chat_stream",
                          "NativeLlamaClient", "openai"):
            self.assertNotIn(forbidden, source, f"scorer must not reach {forbidden}")

    def test_oracles_are_evaluator_only(self) -> None:
        # Nothing under src/ may load the oracle manifest.
        root = Path(__file__).resolve().parents[1] / "src"
        hits = [p for p in root.rglob("*.py")
                if "completion_shadow_oracles" in p.read_text(errors="ignore")]
        self.assertEqual(hits, [], "oracle must never enter runtime")


if __name__ == "__main__":
    unittest.main()
