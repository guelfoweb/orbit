"""The completion oracle must be monotonic under append-only analysis state.

A completion oracle exists to answer "is the request satisfied yet?". If
appending evidence can *unsatisfy* it, the question has no stable answer and
nothing built on it can be qualified.

The defect these tests pin was real and observed: `no_dangerous_capability` is
an absence property of the artifact, but was decided by searching the cumulative
evidence narrative. An analysis step explaining that the file makes no network
call has to name the calls it does not make -- and the predicate then failed
against its own explanation. On a real captured trajectory the oracle went
COMPLETE at action 1 and INCOMPLETE at actions 2-5, with no evidence lost.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from score_completion_shadow import (  # noqa: E402
    PREDICATE_SCOPES,
    evaluate_finding,
    predicate_scope,
    score_text,
)

ORACLES = json.loads(
    (ROOT / "scripts/evaluation/completion_shadow_oracles.json").read_text()
)
AUTHENTIC = Path(
    "/home/guelfoweb/LAB/orbit-checkpoints/INVALID-authentic-capture-20260828/authentic_states"
)


class PredicateScopeTests(unittest.TestCase):
    """Every predicate must declare what it is allowed to inspect."""

    def test_every_kind_in_use_has_a_scope(self) -> None:
        for sample in ORACLES["samples"].values():
            for spec in sample["required_findings"]:
                with self.subTest(spec["id"]):
                    self.assertIn(spec["kind"], PREDICATE_SCOPES)
                    self.assertNotEqual(predicate_scope(spec), "UNKNOWN")

    def test_absence_predicates_are_negative_source_facts(self) -> None:
        for sample in ORACLES["samples"].values():
            for spec in sample["required_findings"]:
                if spec["kind"] == "absent_all":
                    self.assertEqual(predicate_scope(spec), "NEGATIVE_SOURCE_FACT")

    def test_a_spec_may_declare_a_derived_absence(self) -> None:
        """Absence of a DERIVED fact must not be forced onto source bytes."""
        spec = {"id": "no_gap_reported", "kind": "absent_all",
                "values": ["UNRESOLVED"], "scope": "DERIVED_FACT"}
        self.assertEqual(predicate_scope(spec), "DERIVED_FACT")
        # Scored against the narrative, as a derived fact should be.
        self.assertFalse(evaluate_finding(spec, "UNRESOLVED gap remains").satisfied)
        self.assertTrue(evaluate_finding(spec, "all questions answered").satisfied)

    def test_an_unrecognised_declared_scope_is_unknown(self) -> None:
        spec = {"id": "x", "kind": "absent_all", "values": ["y"], "scope": "MADE_UP"}
        self.assertEqual(predicate_scope(spec), "UNKNOWN")

    def test_a_negative_source_fact_refuses_the_narrative(self) -> None:
        """Without source bytes it is unscorable, never guessed from prose."""
        spec = {"id": "x", "kind": "absent_all", "values": ["require("]}
        result = evaluate_finding(spec, "clean narrative")
        self.assertTrue(result.unscorable)
        self.assertFalse(result.satisfied)

    def test_negative_fact_ignores_every_forbidden_data_source(self) -> None:
        """Narrative, later evidence and the final report are all excluded.

        Constructed rather than corpus-derived on purpose: this corpus's report
        happens not to name the forbidden tokens, so it cannot demonstrate the
        property. The adversarial case is the one that matters.
        """
        spec = {"id": "x", "kind": "absent_all", "values": ["fetch(", "eval("]}
        source = "function greet(n){ console.log(n); }"
        forbidden_sources = {
            "cumulative narrative": "The artifact never calls fetch( or eval(.",
            "later evidence prose": "Step 4 confirmed absence of eval( usage.",
            "final report text": "## Findings\nNo fetch( or eval( is present.",
        }
        for where, text in forbidden_sources.items():
            with self.subTest(where):
                result = evaluate_finding(spec, text, source_text=source)
                self.assertTrue(
                    result.satisfied,
                    f"{where} was allowed to unsatisfy a negative source fact",
                )

    def test_the_same_text_would_have_broken_the_old_scoring(self) -> None:
        """Guards the guard: these strings really are hostile to the old rule."""
        values = ["fetch(", "eval("]
        narrative = "The artifact never calls fetch( or eval(."
        old_rule = not any(v.lower() in narrative.lower() for v in values)
        self.assertFalse(old_rule)

    def test_positive_predicates_still_read_the_narrative(self) -> None:
        """Derived facts are established BY analysis, so narrative is correct."""
        spec = {"id": "x", "kind": "literal_all", "values": ["FOUND"]}
        self.assertTrue(evaluate_finding(spec, "the analysis FOUND it").satisfied)


class ScorerSuppliesSourceTests(unittest.TestCase):
    """The scorer must hand the artifact to negative source predicates.

    Fail-closed is right, but a scorer that never supplies the source turns
    every run INDETERMINATE -- correct in principle, useless in practice, and
    it silently erased a previously-merged COMPLETE conclusion.
    """

    CORPUS = Path(
        "/home/guelfoweb/LAB/orbit-checkpoints/completion-shadow-lossless-ac-20260827-175528"
    )

    def setUp(self) -> None:
        if not self.CORPUS.is_dir():
            self.skipTest("preserved corpus not present")

    def test_load_run_carries_the_immutable_artifact(self) -> None:
        from score_completion_shadow import load_run

        data = load_run(self.CORPUS, "A")
        self.assertIsNotNone(data["source_text"])
        self.assertIn("function greet", data["source_text"])

    def test_run_a_scores_complete_not_indeterminate(self) -> None:
        """The regression this guards: absent source made every row INDETERMINATE."""
        from score_completion_shadow import score_corpus

        oracles = json.loads(
            (ROOT / "scripts/evaluation/completion_shadow_oracles.json").read_text()
        )
        rows = score_corpus(self.CORPUS, oracles)["runs"]["A"]["checkpoints"]
        for row in rows:
            with self.subTest(action=row["action"]):
                self.assertEqual(row["snapshot_state"], "COMPLETE")
                # Cumulative gets its own assertion: source can be threaded to
                # one call and not the other, and only this catches that.
                self.assertEqual(row["cumulative_state"], "COMPLETE")
                self.assertEqual(row["a_class"], "A_TRUE_COMPLETE")


class NoNarrativeFallbackTests(unittest.TestCase):
    """score_text must not quietly substitute the narrative for missing source.

    `source_text or text` looks harmless and reintroduces the whole defect:
    with no artifact preserved, absence would once again be decided against the
    analyst's own commentary.
    """

    SPEC = {
        "required_findings": [
            {"id": "no_cap", "kind": "absent_all", "values": ["fetch("]},
        ]
    }

    def test_missing_source_is_unscorable_not_narrative_scored(self) -> None:
        clean = "the artifact does not call anything unusual"
        score = score_text(self.SPEC, clean, 0)
        self.assertEqual(score.unscorable, ["no_cap"])
        self.assertEqual(score.state, "INDETERMINATE")

    def test_hostile_narrative_still_unscorable_without_source(self) -> None:
        """Text naming the token must not flip it either way when source is absent."""
        hostile = "the artifact never calls fetch( at all"
        score = score_text(self.SPEC, hostile, 0)
        self.assertEqual(score.unscorable, ["no_cap"])
        self.assertNotIn("no_cap", score.missing)

    def test_absent_source_never_reaches_the_predicate_at_all(self) -> None:
        """Directly pins the guard, not just its consequence.

        A mutant that keeps `text = source_text` when source exists but deletes
        the None-guard leaves absence decided against narrative in exactly the
        case where no artifact was preserved. Asserting the returned reason
        catches that; asserting only the satisfied/missing split does not.
        """
        spec = {"id": "no_cap", "kind": "absent_all", "values": ["fetch("]}
        result = evaluate_finding(spec, "text mentioning fetch( freely")
        self.assertTrue(result.unscorable)
        self.assertIn("source", result.detail.lower())

    def test_with_source_it_becomes_scorable_again(self) -> None:
        hostile = "the artifact never calls fetch( at all"
        score = score_text(self.SPEC, hostile, 0, source_text="function greet(){}")
        self.assertEqual(score.unscorable, [])
        self.assertEqual(score.state, "COMPLETE")


class SourceProvenanceTests(unittest.TestCase):
    """Which bytes count as "the source" is itself load-bearing.

    Absence is now decided against the artifact, so accepting the wrong file --
    above all `final_report.txt`, the narrative-like text whose exclusion is the
    entire point -- would reintroduce the defect through a different door.
    """

    CORPUS = Path(
        "/home/guelfoweb/LAB/orbit-checkpoints/completion-shadow-lossless-ac-20260827-175528"
    )

    def setUp(self) -> None:
        if not self.CORPUS.is_dir():
            self.skipTest("preserved corpus not present")
        self.tmp = Path(tempfile.mkdtemp(prefix="source-prov-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        shutil.copytree(self.CORPUS, self.tmp / "corpus")
        self.corpus = self.tmp / "corpus"

    def test_source_is_the_artifact_not_the_report(self) -> None:
        from score_completion_shadow import load_run

        data = load_run(self.corpus, "A")
        report = (self.corpus / "runs/A/final_report.txt").read_text()
        self.assertNotEqual(data["source_text"], report)
        self.assertEqual(
            data["source_text"],
            (self.corpus / "runs/A/artifact.js").read_text(),
        )

    def test_a_tampered_artifact_refuses_to_score(self) -> None:
        from score_completion_shadow import load_run

        path = self.corpus / "runs/A/artifact.js"
        path.write_text(path.read_text() + "\n// appended\n")
        with self.assertRaises(SystemExit):
            load_run(self.corpus, "A")

    def test_an_artifact_swapped_for_the_report_refuses_to_score(self) -> None:
        """The exact substitution that would resurrect narrative scoring."""
        from score_completion_shadow import load_run

        report = (self.corpus / "runs/A/final_report.txt").read_text()
        (self.corpus / "runs/A/artifact.js").write_text(report)
        with self.assertRaises(SystemExit):
            load_run(self.corpus, "A")

    def test_ambiguous_artifacts_refuse_to_score(self) -> None:
        """Two candidates means no defensible choice, so make none."""
        from score_completion_shadow import load_run

        (self.corpus / "runs/A/artifact.txt").write_text("decoy")
        with self.assertRaises(SystemExit):
            load_run(self.corpus, "A")

    def _blank_hash(self) -> None:
        run = self.corpus / "runs/A/run.json"
        data = json.loads(run.read_text())
        data["artifact_sha256_before"] = ""
        run.write_text(json.dumps(data))

    def test_a_corpus_recording_no_hash_refuses_to_score(self) -> None:
        """The guard must not weaken itself when the corpus vouches for nothing.

        This defect lived in corpus SHAPE, not in code, so no source mutation
        expresses it: `if expected and ...` silently verified nothing whenever
        the field was blank.
        """
        from score_completion_shadow import load_run

        self._blank_hash()
        with self.assertRaises(SystemExit) as caught:
            load_run(self.corpus, "A")
        # The REASON matters, not just the raise. Without the explicit guard a
        # blank hash still trips the mismatch branch by accident -- same
        # outcome here, but it would waive verification anywhere the recorded
        # hash is blank AND the comparison is later loosened.
        self.assertIn("records no", str(caught.exception))

    def test_a_blank_hash_cannot_launder_the_final_report(self) -> None:
        """The swap test only bit because the hash was intact; close that door."""
        from score_completion_shadow import load_run

        report = (self.corpus / "runs/A/final_report.txt").read_text()
        (self.corpus / "runs/A/artifact.js").write_text(report)
        self._blank_hash()
        with self.assertRaises(SystemExit) as caught:
            load_run(self.corpus, "A")
        self.assertIn("records no", str(caught.exception))

    def test_an_empty_artifact_refuses_even_with_a_matching_hash(self) -> None:
        """Absence over nothing is vacuously true; refuse rather than pass."""
        import hashlib

        from score_completion_shadow import load_run

        path = self.corpus / "runs/A/artifact.js"
        path.write_text("")
        run = self.corpus / "runs/A/run.json"
        data = json.loads(run.read_text())
        data["artifact_sha256_before"] = hashlib.sha256(b"").hexdigest()
        run.write_text(json.dumps(data))
        with self.assertRaises(SystemExit):
            load_run(self.corpus, "A")

    def test_the_whole_hash_is_compared_not_a_prefix(self) -> None:
        """A truncated comparison is weaker than it looks; pin the full digest."""
        import hashlib

        from score_completion_shadow import load_run

        path = self.corpus / "runs/A/artifact.js"
        real = path.read_bytes()
        digest = hashlib.sha256(real).hexdigest()
        run = self.corpus / "runs/A/run.json"
        data = json.loads(run.read_text())
        # Same first 16 hex chars, different tail: only a full comparison fails.
        data["artifact_sha256_before"] = digest[:16] + ("0" * 48)
        run.write_text(json.dumps(data))
        with self.assertRaises(SystemExit):
            load_run(self.corpus, "A")

    def test_a_directory_artifact_refuses_cleanly(self) -> None:
        from score_completion_shadow import load_run

        path = self.corpus / "runs/A/artifact.js"
        path.unlink()
        path.mkdir()
        with self.assertRaises(SystemExit):
            load_run(self.corpus, "A")

    def test_the_unmodified_corpus_still_loads(self) -> None:
        """Guards the guard: the refusals above must not be unconditional."""
        from score_completion_shadow import load_run

        self.assertIsNotNone(load_run(self.corpus, "A")["source_text"])


class MonotonicityInvariantTests(unittest.TestCase):
    """COMPLETE -> INCOMPLETE must be unreachable without supersession."""

    LEGAL = {
        ("INCOMPLETE", "INCOMPLETE"),
        ("INCOMPLETE", "COMPLETE"),
        ("COMPLETE", "COMPLETE"),
    }

    def _trajectory(self, label: str, source: Path) -> list[str]:
        states = sorted(AUTHENTIC.glob(f"{label}_action*"))
        if not states:
            self.skipTest("authentic capture not present")
        source_text = source.read_text(errors="replace")
        out = []
        for state in states:
            index = json.loads((state / "evidence" / "index.json").read_text())
            text = "\n".join(
                (state / "evidence" / f"{eid}.txt").read_text(errors="replace")
                for eid in index
                if (state / "evidence" / f"{eid}.txt").exists()
            )
            out.append(
                score_text(ORACLES["samples"][label], text, 0, source_text=source_text).state
            )
        return out

    def test_trivial_trajectory_never_regresses(self) -> None:
        """The exact trajectory that exhibited the defect."""
        states = self._trajectory("A", ROOT / "workdir/samples/trivial_greeting_demo.js")
        self.assertGreaterEqual(len(states), 2, "need a transition to test")
        for i in range(1, len(states)):
            with self.subTest(step=i, transition=(states[i - 1], states[i])):
                self.assertIn((states[i - 1], states[i]), self.LEGAL)

    def test_powershell_trajectory_never_regresses(self) -> None:
        states = self._trajectory("C", ROOT / "workdir/samples/peXF7I6W.ps1")
        for i in range(1, len(states)):
            with self.subTest(step=i, transition=(states[i - 1], states[i])):
                self.assertIn((states[i - 1], states[i]), self.LEGAL)

    def test_the_defect_is_actually_gone_not_merely_untested(self) -> None:
        """Guards the guard: the old scoring really did regress here."""
        states = sorted(AUTHENTIC.glob("A_action*"))
        if len(states) < 2:
            self.skipTest("authentic capture not present")
        spec = [
            s for s in ORACLES["samples"]["A"]["required_findings"]
            if s["id"] == "no_dangerous_capability"
        ][0]
        later = states[1]
        index = json.loads((later / "evidence" / "index.json").read_text())
        narrative = "\n".join(
            (later / "evidence" / f"{eid}.txt").read_text(errors="replace")
            for eid in index
            if (later / "evidence" / f"{eid}.txt").exists()
        )
        # Old behaviour: absence decided against narrative -> fails.
        old = not any(v.lower() in narrative.lower() for v in spec["values"])
        self.assertFalse(old, "expected the historical defect to be reproducible")
        # New behaviour: decided against source -> holds.
        source = (ROOT / "workdir/samples/trivial_greeting_demo.js").read_text()
        self.assertTrue(evaluate_finding(spec, narrative, source_text=source).satisfied)


if __name__ == "__main__":
    unittest.main()
