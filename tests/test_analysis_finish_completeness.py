"""FINISH must not accept RESOLVED when its own completion data says work
remains.

The IBAN failure: a run asked "what does eval(MMGCLZ) decode to?", ran one
action whose evidence held only the still-unresolved `String.fromCharCode(N-J7f,
...)` expression, then issued `finish(status=resolved)` -- and the runtime
accepted it, leaving no open question, so the run stopped while its own report
said the value was unresolved. The user had to type "ok decode J7f" to continue
locally-available work.

The fix is a structured completion contract, not semantic understanding: a
RESOLVED decision must carry a non-empty answer and cite evidence that exists,
and must not simultaneously declare an unresolved dependency (a child question,
or `remaining_unknown`). These are structural contradictions the runtime rejects
without reading the artifact. A rejected RESOLVED becomes STILL_OPEN so the
question keeps its remaining action budget; boundedness is unchanged.

All witnesses are the controller's own state, never report prose.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from orbit.runtime.analysis_controller import (
    BLOCKED,
    OPEN,
    RESOLVED,
    AnalysisController,
    ControlError,
    parse_finish_call,
)


# --- unit level: the completion contract in parse_finish_call ---------------
class FinishContractTests(unittest.TestCase):
    def _parse(self, **kw):
        return parse_finish_call(kw)

    def test_resolved_requires_answer(self):
        # B7: RESOLVED with an empty answer is a contradiction.
        with self.assertRaises(ControlError):
            self._parse(status="resolved", evidence_ids=["ev_x"], answer_summary="")

    def test_resolved_without_ids_parses_but_needs_runtime_evidence(self):
        # Parse allows a resolution that states its answer but names no id: the
        # runtime cites the answering action's evidence (the legitimate pattern).
        # The evidence-existence requirement is enforced at close_active /
        # _apply_decision, proven in CloseActiveContractTests and the runtime
        # integration tests -- not here.
        d = self._parse(status="resolved", answer_summary="it decodes to X")
        self.assertEqual(d["status"], RESOLVED)
        self.assertEqual(d["evidence_ids"], ())

    def test_resolved_with_child_question_is_contradiction(self):
        # B7: declaring an unresolved dependency (a child) while claiming
        # RESOLVED contradicts itself.
        with self.assertRaises(ControlError):
            self._parse(
                status="resolved",
                evidence_ids=["ev_x"],
                answer_summary="answer",
                child_question={
                    "question": "resolve J-like constant",
                    "missing_fact": "the constant value",
                    "caused_by_evidence_id": "ev_x",
                },
            )

    def test_resolved_complete_is_accepted(self):
        d = self._parse(
            status="resolved", evidence_ids=["ev_x"], answer_summary="decodes to X"
        )
        self.assertEqual(d["status"], RESOLVED)

    def test_still_open_needs_nothing(self):
        d = self._parse(status="still_open")
        self.assertEqual(d["status"], OPEN)

    def test_blocked_needs_nothing(self):
        d = self._parse(status="blocked")
        self.assertEqual(d["status"], BLOCKED)

    def test_still_open_may_carry_child(self):
        # A child alongside still_open is the ordinary causal-follow-up path.
        d = self._parse(
            status="still_open",
            child_question={
                "question": "resolve the constant",
                "missing_fact": "value",
                "caused_by_evidence_id": "ev_x",
            },
        )
        self.assertEqual(d["status"], OPEN)
        self.assertIsInstance(d["child_question"], dict)


# --- controller level: close_active refuses a hollow RESOLVED ---------------
class CloseActiveContractTests(unittest.TestCase):
    def _one_open(self):
        c = AnalysisController()
        c.adopt_plan([{"question": "what does eval(x) decode to?",
                       "missing_fact": "the decoded value"}])
        c.activate_next()
        return c

    def test_resolved_without_evidence_stays_open(self):
        c = self._one_open()
        c.close_active(RESOLVED, evidence_ids=(), summary="decodes to X")
        self.assertEqual(c.states[c.active].status, OPEN)  # not resolved

    def test_resolved_without_summary_stays_open(self):
        c = self._one_open()
        c.close_active(RESOLVED, evidence_ids=("ev_x",), summary="")
        self.assertEqual(c.states[c.active].status, OPEN)

    def test_resolved_complete_resolves(self):
        c = self._one_open()
        c.close_active(RESOLVED, evidence_ids=("ev_x",), summary="decodes to X")
        self.assertEqual(c.states[c.active].status, RESOLVED)

    def test_blocked_needs_no_witness(self):
        c = self._one_open()
        c.close_active(BLOCKED, reason="ran out of attempts")
        self.assertEqual(c.states[c.active].status, BLOCKED)


# --- runtime level: _apply_decision requires evidence that EXISTS ------------
class ApplyDecisionEvidenceTests(unittest.TestCase):
    """B-T4: a RESOLVED whose cited evidence does not re-attest -- and whose
    answering step evidence does not re-attest either -- must not resolve. Proves
    the runtime never substitutes a non-existent evidence id to force closure."""

    def _runtime(self):
        import shutil
        import tempfile

        from orbit.runtime.analysis_runtime import AnalysisRuntime, acquire_analysis_source
        from orbit.runtime.evidence import EvidenceStore
        from tests.test_analysis_runtime import ScriptedBackend

        tmp = tempfile.mkdtemp(prefix="orbit-apply-")
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        original = Path(tmp) / "a.js"
        original.write_text("var x=1;\n")
        source = acquire_analysis_source(original, Path(tmp) / "owned")
        store = EvidenceStore(root=Path(tmp) / "evidence")
        rt = AnalysisRuntime(backend=ScriptedBackend(), source=source, evidence_store=store)
        self.addCleanup(rt.close)
        return rt

    def test_resolved_with_nonexistent_evidence_stays_open(self):
        rt = self._runtime()
        c = AnalysisController()
        c.adopt_plan([{"question": "what does it decode to?",
                       "missing_fact": "value"}])
        c.activate_next()
        # Both the cited id and the "step evidence" id are fabricated -- neither
        # re-attests in the empty store.
        rt._apply_decision(
            c,
            {"status": RESOLVED, "evidence_ids": ("ev_nope",),
             "answer_summary": "claims X", "child_question": None},
            "ev_also_nope",
        )
        self.assertEqual(c.states[c.active].status, OPEN)


# --- integration: the IBAN-shaped run cannot falsely close ------------------
class IbanShapeReproductionTests(unittest.TestCase):
    """Reproduce the first IBAN run's shape with a scripted backend and prove
    the runtime no longer accepts the hollow resolution. Witnesses are the
    controller's resolved/open sets on the run result -- never report prose."""

    def _make(self, *responses, plan_questions, finish_decisions):
        import shutil
        import tempfile

        from orbit.runtime.analysis_runtime import AnalysisRuntime, acquire_analysis_source
        from orbit.runtime.evidence import EvidenceStore
        from tests.test_analysis_runtime import ScriptedBackend

        tmp = tempfile.mkdtemp(prefix="orbit-finish-")
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        original = Path(tmp) / "artifact.js"
        original.write_text("var J=1;var M=String.fromCharCode(66-J);eval(M);\n")
        source = acquire_analysis_source(original, Path(tmp) / "owned")
        store = EvidenceStore(root=Path(tmp) / "evidence")
        backend = ScriptedBackend(
            *responses,
            plan_questions=plan_questions,
            finish_decisions=finish_decisions,
        )
        runtime = AnalysisRuntime(backend=backend, source=source, evidence_store=store)
        self.addCleanup(runtime.close)
        return runtime, backend

    def test_hollow_resolved_never_marks_question_resolved(self):
        # B-T1/B-T8: the exact first-IBAN shape -- one action whose evidence is
        # only the unresolved expression, then a hollow finish(resolved). The
        # question must NEVER end resolved on that; it is retried within its
        # bounded budget and then blocked (the scripted model keeps issuing the
        # hollow decision, which is refused every time). No false RESOLVED.
        from tests.test_analysis_runtime import tool_response

        runtime, backend = self._make(
            tool_response("print('String.fromCharCode(66-J) -- value not yet decoded')"),
            tool_response("print('String.fromCharCode(66-J) -- still not decoded')"),
            plan_questions=["What does eval(M) decode to?"],
            # The model keeps trying to close hollow; every one is refused.
            finish_decisions=[{"status": "resolved"}] * 6,
        )
        run = runtime.run_autonomous("Analyse this artifact.", finalize=False)
        # The one question must not be falsely resolved. It stays open or ends
        # blocked once its action budget is spent -- never resolved-with-nothing.
        self.assertEqual(run.resolved_questions, ())
        # And the run stayed bounded (did not loop unbounded on the refusal).
        self.assertLessEqual(run.actions_executed, 4)

    def test_run_stays_bounded_on_repeated_refusal(self):
        # B-T7/B-T8: the model keeps issuing still_open; the run spends the
        # question's bounded budget and stops, never looping unbounded.
        from tests.test_analysis_runtime import tool_response

        runtime, backend = self._make(
            tool_response("print('attempt 1')"),
            tool_response("print('attempt 2')"),
            tool_response("print('attempt 3 -- should never run: budget is 2')"),
            plan_questions=["What does eval(M) decode to?"],
            finish_decisions=[{"status": "still_open", "answer_summary": "seen"}] * 4,
        )
        run = runtime.run_autonomous("Analyse this artifact.", finalize=False)
        # MAX_ACTIONS_PER_QUESTION is 2, so the single question spends at most 2.
        self.assertLessEqual(run.actions_executed, 2)
        self.assertEqual(run.resolved_questions, ())

    def test_scripted_hollow_resolved_is_rejected_at_parse(self):
        # Direct proof at the contract boundary: the exact hollow decision the
        # first IBAN run issued is refused.
        with self.assertRaises(ControlError):
            parse_finish_call({"status": "resolved"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
