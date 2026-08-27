"""Deterministic scoring of the fresh lossless A+C corpus.

Evaluation only. These tests pin the conclusions drawn from that corpus so a
later change to the scorer cannot silently rewrite them, and they assert the
scorer reaches no model.
"""
from __future__ import annotations

import hashlib
import json
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(ROOT / "src"))

from score_completion_shadow import (  # noqa: E402
    classify_a,
    classify_b,
    evaluate_finding,
    score_corpus,
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


def _findings_changed_after_action_4(label: str) -> tuple[list[str], list[str]]:
    """How the required findings differ between action 4 and the FULL store.

    Returns (gained, lost). Both directions matter, and for opposite reasons.

    `gained` is the obvious question: did stopping at action 4 miss something
    the analysis went on to establish?

    `lost` exists because not every predicate is monotonic. `absent_all` -- "no
    dangerous capability appears" -- can only *lose* satisfaction as text
    grows, so a capability discovered after action 4 shows up here and could
    never appear in `gained`. Reporting only `gained` would leave the one class
    of late discovery that most undermines an early stop unable to register.
    """
    from orbit.runtime.evidence import EvidenceStore

    store = EvidenceStore(root=CORPUS / f"runs/{label}/evidence")
    store.load_index()
    full = "\n".join(
        (store.load_raw(r.evidence_id) or "") for r in store.records.values()
    )
    at4 = _text(_checkpoint(label, 4))
    gained, lost = [], []
    for spec in ORACLES[label]["required_findings"]:
        if spec["kind"] == "unscorable":
            continue
        before = evaluate_finding(spec, at4).satisfied
        after = evaluate_finding(spec, full).satisfied
        if after and not before:
            gained.append(spec["id"])
        elif before and not after:
            lost.append(spec["id"])
    return gained, lost


def _findings_added_after_action_4(label: str) -> list[str]:
    """Back-compat view: only the findings gained. See the function above."""
    return _findings_changed_after_action_4(label)[0]


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
        gained, lost = _findings_changed_after_action_4("A")
        self.assertEqual(gained, [], "a finding appeared only after action 4")
        self.assertEqual(lost, [], "a finding stopped holding after action 4")


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
        gained, lost = _findings_changed_after_action_4("C")
        self.assertEqual(gained, [], "a finding appeared only after action 4")
        self.assertEqual(lost, [], "a finding stopped holding after action 4")

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


class SyntheticCorpusTests(unittest.TestCase):
    """Tests for the scorer's own structure, built rather than borrowed.

    The preserved corpus cannot exercise these: it has no truncated run, no
    crashed verifier, and no absent workload class. Without a corpus it
    constructs itself, the label-discovery and skip-vs-error paths ship
    untested -- and a scorer with those changes reverted passes anyway.
    """

    def _corpus(self, runs: dict[str, dict]) -> Path:
        root = Path(tempfile.mkdtemp(prefix="scorer-corpus-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for label, spec in runs.items():
            run_dir = root / "runs" / label
            (run_dir / "evidence").mkdir(parents=True)
            lines = [json.dumps({
                "record": "run_start", "run_id": "r1", "schema_version": 1,
                "request": "analyze x", "schedule": [4], "soft_max_actions": 8,
                "max_actions": 12, "max_model_calls": 15,
            })]
            for cp in spec.get("checkpoints", ()):
                evidence = [["ev_" + "0" * 12 + "_" + "0" * 16, cp.get("text", "")]]
                # The scorer verifies snapshot digests before it will score, so
                # a fixture has to carry real ones -- which is also what proves
                # that check is live rather than decorative.
                digest = hashlib.sha256(json.dumps(
                    {"request": "analyze x", "evidence": evidence, "artifacts": []},
                    ensure_ascii=False, sort_keys=True,
                ).encode("utf-8")).hexdigest()
                lines.append(json.dumps({
                    "record": "checkpoint", "run_id": "r1", "schema_version": 1,
                    "request": "analyze x", "action": cp["action"],
                    "snapshot_evidence": [{"evidence_id": evidence[0][0],
                                           "text": evidence[0][1]}],
                    "snapshot_artifacts": [], "snapshot_sha256": digest,
                    "snapshot_tokens": 10, "token_budget": 6144,
                    "verification_skipped": cp.get("skipped", False),
                    "snapshot_fidelity": {"lossless": True, "reasons": []},
                    "verifier_a": cp.get("a"), "verifier_a_detail": "",
                    "verifier_a_evidence_ids": [], "verifier_a_raw": "",
                    "verifier_b": cp.get("b"), "verifier_b_detail": cp.get("b_detail", ""),
                    "verifier_b_raw": "", "verifier_calls": 0,
                    "verifier_output_tokens": 0, "verifier_prompt_tokens": 0,
                    "verifier_wall_seconds": 0.0, "would_stop": False,
                    "blocked_by": cp.get("blocked_by"),
                }))
            (run_dir / "completion-shadow.jsonl").write_text("\n".join(lines) + "\n")
            if spec.get("run_json", True):
                (run_dir / "run.json").write_text(json.dumps({
                    "label": label, "artifact": "x", "artifact_sha256_before": "0" * 64,
                    "artifact_sha256_after": "0" * 64, "stop_reason": "done",
                    "actions_executed": 4, "model_calls": 4, "wall_seconds": 1.0,
                    "shadow": True, "steps": 4, "replans": 0, "cancelled": False,
                    "final_report": "", "prompt": "analyze x", "step_detail": [],
                }))
        return root

    @staticmethod
    def _oracles(labels) -> dict:
        return {"schema_version": 1, "samples": {
            label: {"artifact": "x", "artifact_sha256": "0" * 64, "conclusion": "c",
                    "required_findings": [
                        {"id": "marker", "kind": "literal_all", "values": ["MARKER"]}]}
            for label in labels}}

    def test_every_run_with_a_run_json_is_scored(self) -> None:
        """Pins label discovery: a hardcoded A/B/C loop cannot pass this."""
        corpus = self._corpus({
            "Q": {"checkpoints": [{"action": 4, "a": "COMPLETE", "b": "NO_GAP", "text": "MARKER"}]},
            "Z": {"checkpoints": [{"action": 4, "a": "COMPLETE", "b": "NO_GAP", "text": "MARKER"}]},
        })
        report = score_corpus(corpus, self._oracles(("Q", "Z")))
        self.assertEqual(sorted(report["runs"]), ["Q", "Z"])

    def test_a_run_directory_without_run_json_is_reported_not_dropped(self) -> None:
        """A run killed before its final write must not pass for an absent one."""
        corpus = self._corpus({
            "Q": {"checkpoints": [{"action": 4, "a": "COMPLETE", "b": "NO_GAP", "text": "MARKER"}]},
            "P": {"run_json": False,
                  "checkpoints": [{"action": 4, "a": "COMPLETE", "b": "NO_GAP", "text": "MARKER"}]},
        })
        report = score_corpus(corpus, self._oracles(("Q", "P")))
        self.assertEqual(sorted(report["runs"]), ["Q"])
        self.assertEqual(report["incomplete_runs"], ["P"])

    def test_a_crashed_verifier_is_not_scored_as_a_budget_skip(self) -> None:
        """The distinction the whole A_NOT_CALLED label depends on."""
        corpus = self._corpus({"Q": {"checkpoints": [
            {"action": 4, "a": None, "b": None, "text": "MARKER",
             "skipped": True, "blocked_by": "snapshot_too_large"},
            {"action": 6, "a": None, "b": None, "text": "MARKER",
             "skipped": False, "blocked_by": "verifier_error: RuntimeError"},
        ]}})
        rows = score_corpus(corpus, self._oracles(("Q",)))["runs"]["Q"]["checkpoints"]
        self.assertEqual((rows[0]["a_class"], rows[0]["b_class"]),
                         ("A_NOT_CALLED", "B_NOT_CALLED"))
        self.assertEqual((rows[1]["a_class"], rows[1]["b_class"]),
                         ("A_ERRORED", "B_ERRORED"))

    def test_observed_schedule_reflects_what_was_checkpointed(self) -> None:
        corpus = self._corpus({"Q": {"checkpoints": [
            {"action": 4, "a": "COMPLETE", "b": "NO_GAP", "text": "MARKER"}]}})
        report = score_corpus(corpus, self._oracles(("Q",)))
        self.assertEqual(report["runs"]["Q"]["observed_schedule"], [4])
        self.assertNotEqual(report["runs"]["Q"]["observed_schedule"], report["schedule"])

    def test_a_false_complete_is_reachable(self) -> None:
        """Guards the empty cells: the matrix must be able to report a miss."""
        corpus = self._corpus({"Q": {"checkpoints": [
            {"action": 4, "a": "COMPLETE", "b": "NO_GAP", "text": "nothing here"}]}})
        row = score_corpus(corpus, self._oracles(("Q",)))["runs"]["Q"]["checkpoints"][0]
        self.assertEqual(row["a_class"], "A_FALSE_COMPLETE")
        self.assertEqual(row["b_class"], "B_FALSE_NO_GAP")

    def test_an_unscorable_finding_prevents_a_complete_verdict(self) -> None:
        """The docstring's safety promise, asserted rather than trusted."""
        oracles = {"schema_version": 1, "samples": {"Q": {
            "artifact": "x", "artifact_sha256": "0" * 64, "conclusion": "c",
            "required_findings": [
                {"id": "marker", "kind": "literal_all", "values": ["MARKER"]},
                {"id": "judgement", "kind": "unscorable", "comment": "not decidable"},
            ]}}}
        corpus = self._corpus({"Q": {"checkpoints": [
            {"action": 4, "a": "COMPLETE", "b": "NO_GAP", "text": "MARKER"}]}})
        row = score_corpus(corpus, oracles)["runs"]["Q"]["checkpoints"][0]
        self.assertNotEqual(row["snapshot_state"], "COMPLETE")
        self.assertEqual(row["a_class"], "A_INDETERMINATE")


class AntiMonotonicFindingTests(unittest.TestCase):
    """Guards the guard: `lost` must be able to fire, or asserting it is empty
    proves nothing."""

    SPEC = {"id": "no_dangerous_capability", "kind": "absent_all",
            "values": ["require(", "eval("]}

    def test_absent_all_is_lost_when_the_capability_appears_later(self) -> None:
        early = "function greet(name) { console.log(name); }"
        later = early + "\nrequire('child_process')"
        self.assertTrue(evaluate_finding(self.SPEC, early).satisfied)
        self.assertFalse(evaluate_finding(self.SPEC, later).satisfied)

    def test_such_a_finding_could_never_appear_as_gained(self) -> None:
        """Which is exactly why the one-directional filter was insufficient."""
        early = "clean"
        later = "clean require("
        gained = evaluate_finding(self.SPEC, later).satisfied and not evaluate_finding(self.SPEC, early).satisfied
        lost = evaluate_finding(self.SPEC, early).satisfied and not evaluate_finding(self.SPEC, later).satisfied
        self.assertFalse(gained)
        self.assertTrue(lost)


class PredicateSemanticsTests(unittest.TestCase):
    """`all` vs `any` is invisible on one-value oracles, so pin it directly."""

    MULTI = {"id": "m", "kind": "literal_all", "values": ["alpha", "beta"]}

    def test_literal_all_requires_every_value(self) -> None:
        self.assertTrue(evaluate_finding(self.MULTI, "alpha and beta").satisfied)
        self.assertFalse(evaluate_finding(self.MULTI, "alpha only").satisfied)
        self.assertFalse(evaluate_finding(self.MULTI, "beta only").satisfied)

    def test_literal_any_requires_only_one(self) -> None:
        spec = dict(self.MULTI, kind="literal_any")
        self.assertTrue(evaluate_finding(spec, "alpha only").satisfied)
        self.assertFalse(evaluate_finding(spec, "neither").satisfied)

    def test_absent_all_rejects_any_present_value(self) -> None:
        spec = dict(self.MULTI, kind="absent_all")
        self.assertTrue(evaluate_finding(spec, "neither here").satisfied)
        self.assertFalse(evaluate_finding(spec, "alpha here").satisfied)
        self.assertFalse(evaluate_finding(spec, "beta here").satisfied)


class IntegrityRefusalTests(unittest.TestCase):
    """The scorer must refuse a corpus it cannot trust, not score it anyway."""

    def test_a_tampered_snapshot_hash_refuses_to_score(self) -> None:
        helper = SyntheticCorpusTests()
        helper.addCleanup = lambda *a, **k: None
        corpus = helper._corpus({"Q": {"checkpoints": [
            {"action": 4, "a": "COMPLETE", "b": "NO_GAP", "text": "MARKER"}]}})
        self.addCleanup(shutil.rmtree, corpus, ignore_errors=True)
        ledger = corpus / "runs/Q/completion-shadow.jsonl"
        lines = ledger.read_text().splitlines()
        record = json.loads(lines[1])
        record["snapshot_evidence"][0]["text"] = "TAMPERED"
        lines[1] = json.dumps(record)
        ledger.write_text("\n".join(lines) + "\n")
        with self.assertRaises(SystemExit):
            score_corpus(corpus, helper._oracles(("Q",)))


class ForbiddenValueGuardTests(unittest.TestCase):
    """What the `forbidden` list does, stated as what it actually does."""

    SPEC = {"id": "c2", "kind": "literal_all", "values": ["gibuzuy37v2v.top"],
            "forbidden": ["gibuyuy37v2v.top"]}

    def test_the_correct_value_alone_satisfies(self) -> None:
        self.assertTrue(evaluate_finding(self.SPEC, "host gibuzuy37v2v.top").satisfied)

    def test_the_stale_value_alone_does_not_satisfy(self) -> None:
        self.assertFalse(evaluate_finding(self.SPEC, "host gibuyuy37v2v.top").satisfied)

    def test_the_stale_value_alone_fails_because_the_correct_one_is_absent(self) -> None:
        """Names the real mechanism: `forbidden` does not veto, `literal_all` decides.

        Worth pinning explicitly, because the obvious reading of the previous
        test is that the guard rejected the stale value. It did not -- and a
        record carrying both values still satisfies, which is what a correction
        record looks like.
        """
        both = evaluate_finding(self.SPEC, "was gibuyuy37v2v.top, now gibuzuy37v2v.top")
        self.assertTrue(both.satisfied)
        without_guard = evaluate_finding(
            {k: v for k, v in self.SPEC.items() if k != "forbidden"},
            "host gibuyuy37v2v.top",
        )
        self.assertFalse(without_guard.satisfied)
