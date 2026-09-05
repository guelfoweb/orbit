"""The total dispatch bound for a control protocol failure.

A live Fattura run ended on `LlamaServerToolCallParseError` while closing Q2.
`_control_call` catches the first parse failure and dispatches one repaired
call, but that repair was not itself guarded: a second failure escaped the
control boundary and was caught by the step handler's `RecoverableBackendError`
clause -- which `ToolCallParseError` subclasses -- so the run reported a
backend error for something the server did correctly, and the question never
reached its own bounded close.

Every count here is witnessed at the BACKEND, never inferred from a runtime
counter: the bound is a claim about dispatches, and only the backend sees them.
"""

from __future__ import annotations

import unittest
from unittest import mock

from orbit.backend.base import RecoverableBackendError
from orbit.runtime.context_manager import ContextAdmissionError
from orbit.backend.llama_server import LlamaServerToolCallParseError
from orbit.runtime import analysis_controller
from orbit.runtime.analysis_runtime import FINISH_TOOL_NAME, PLAN_TOOL_NAME

from tests.test_analysis_controller_runtime import _Case, _Model, _question

#: One attempt plus one bounded repair. Asserted rather than derived: the
#: defect this file pins was two nested loops multiplying into four.
CONTROL_DISPATCH_BOUND = 2


class _DispatchWitness:
    """Counts every dispatch by tool, and scripts failures per phase.

    Wraps rather than replaces the fixture backend so ordinary calls behave
    exactly as they always did; only the counting and the scripted failures
    are added.
    """

    thinking = False

    def __init__(self, inner, *, finish=(), plan=()):
        self.inner = inner
        self.finish_script = list(finish)
        self.plan_script = list(plan)
        self.finish_dispatches = 0
        self.plan_dispatches = 0
        self.total_dispatches = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _fail(self, action):
        if action == "parse":
            raise LlamaServerToolCallParseError(
                "Failed to parse input at pos 41: <tool_call>"
            )
        if action == "outage":
            raise RecoverableBackendError("server unreachable")
        if action == "admission":
            raise ContextAdmissionError("context admission failed: too-large")
        if action == "interrupt":
            raise KeyboardInterrupt()

    def chat_stream(self, messages, **kwargs):
        self.total_dispatches += 1
        offered = [t["function"]["name"] for t in (kwargs.get("tools") or [])]
        if FINISH_TOOL_NAME in offered and self.finish_script:
            self.finish_dispatches += 1
            index = min(self.finish_dispatches - 1, len(self.finish_script) - 1)
            self._fail(self.finish_script[index])
        elif PLAN_TOOL_NAME in offered and self.plan_script:
            self.plan_dispatches += 1
            index = min(self.plan_dispatches - 1, len(self.plan_script) - 1)
            self._fail(self.plan_script[index])
        return self.inner.chat_stream(messages, **kwargs)


class ControlRepairBoundTestBase(unittest.TestCase):
    def _run(self, *, questions=1, finish=(), plan=(), capture_states=False):
        borrowed = _Case("run")
        borrowed.addCleanup = self.addCleanup
        runtime = borrowed._runtime(
            _Model(plan=[_question(f"q{i}") for i in range(questions)])
        )
        witness = _DispatchWitness(runtime.backend, finish=finish, plan=plan)
        runtime.backend = witness

        live = {}
        original = analysis_controller.AnalysisController.__init__

        def capture(self_, *args, **kwargs):
            original(self_, *args, **kwargs)
            live["controller"] = self_

        escaped = None
        patcher = (
            mock.patch.object(
                analysis_controller.AnalysisController, "__init__", capture
            )
            if capture_states else mock.patch.object(runtime, "close", runtime.close)
        )
        with patcher:
            try:
                run = runtime.run_autonomous(
                    "Analyse it.", finalize=False, max_model_calls=60
                )
            except BaseException as exc:  # noqa: BLE001 - the point of the test
                run, escaped = None, exc
        return run, witness, escaped, live.get("controller")


class TheDispatchBoundIsTotalTests(ControlRepairBoundTestBase):
    """One attempt, one repair. Never four, and never unbounded."""

    def test_a_valid_reply_costs_one_dispatch_per_question(self) -> None:
        run, witness, escaped, _ = self._run(questions=2)
        self.assertIsNone(escaped)
        self.assertEqual(witness.finish_dispatches, 0, "no failures scripted")
        self.assertEqual(list(run.resolved_questions), ["Q1", "Q2"])

    def test_one_failure_then_a_valid_reply_costs_two(self) -> None:
        run, witness, escaped, _ = self._run(
            questions=1, finish=("parse", "ok")
        )
        self.assertIsNone(escaped)
        # The failure and its repair: exactly the allowance, no more.
        self.assertEqual(witness.finish_dispatches, CONTROL_DISPATCH_BOUND)
        self.assertEqual(run.control_repairs, 1)

    def test_repeated_failures_never_exceed_the_bound(self) -> None:
        """The multiplication this file exists to prevent.

        `finish_question` bounds its attempts and `_control_call` bounds its
        repair; before this was contained the two nested loops multiplied
        into four dispatches for one question.
        """
        for questions in (1, 2, 3):
            with self.subTest(questions=questions):
                run, witness, escaped, _ = self._run(
                    questions=questions, finish=("parse",) * 20
                )
                self.assertIsNone(escaped)
                self.assertEqual(
                    witness.finish_dispatches,
                    CONTROL_DISPATCH_BOUND * questions,
                )

    def test_the_repair_counter_equals_the_dispatches_it_names(self) -> None:
        """Accounting must not claim a repair the backend never saw."""
        run, witness, _, _ = self._run(questions=2, finish=("parse",) * 20)
        # One repair per question, and two dispatches per question.
        self.assertEqual(run.control_repairs, 2)
        self.assertEqual(witness.finish_dispatches, 4)

    def test_the_plan_phase_is_bounded_the_same_way(self) -> None:
        run, witness, escaped, _ = self._run(questions=1, plan=("parse",) * 20)
        self.assertIsNone(escaped)
        # PLAN already reached `unsupported` on the parent; what was wrong
        # there was the cost -- its own `range(2)` multiplied against the
        # repair inside `_control_call`, so four dispatches bought a bound
        # of two. Pinned here so the two layers cannot drift apart again.
        self.assertEqual(witness.plan_dispatches, CONTROL_DISPATCH_BOUND)
        self.assertEqual(
            run.stop_reason, "autonomous control unsupported by this model"
        )


class RepeatedParseFailureClosesHonestlyTests(ControlRepairBoundTestBase):
    def test_no_parse_error_escapes_the_run(self) -> None:
        """It never reached the caller, and it never should.

        Measured on the parent too: the escape was from the CONTROL boundary,
        not from `run_autonomous` -- the step handler caught it as a backend
        error. So this assertion holds before and after, and the regression
        that matters is the domain, asserted below. Kept because a future
        change that let the exception reach the caller would be worse than
        either, and nothing else pins it.
        """
        run, _, escaped, _ = self._run(questions=2, finish=("parse",) * 20)
        self.assertIsNone(escaped, "the run contained its own protocol failure")
        self.assertIsNotNone(run)

    def test_it_is_not_reported_as_a_backend_error(self) -> None:
        """The domain the live run got wrong.

        `ToolCallParseError` subclasses `RecoverableBackendError`, so an
        escaped protocol failure was caught by the backend clause and blamed
        the server for a reply it delivered correctly.
        """
        run, _, _, _ = self._run(questions=2, finish=("parse",) * 20)
        self.assertNotIn("backend error", str(run.stop_reason))

    def test_the_question_is_blocked_not_resolved(self) -> None:
        run, _, _, controller = self._run(
            questions=2, finish=("parse",) * 20, capture_states=True
        )
        self.assertIsNotNone(controller)
        for question_id in controller.order:
            state = controller.states[question_id]
            self.assertEqual(state.status, "blocked")
            self.assertEqual(
                state.reason, "the completion state could not be read"
            )
        self.assertEqual(list(run.resolved_questions), [])

    def test_the_run_continues_to_the_next_question(self) -> None:
        """A blocked question ends itself, not the analysis."""
        _, witness, _, controller = self._run(
            questions=3, finish=("parse",) * 20, capture_states=True
        )
        # Every question was attempted, so the first failure did not end it.
        self.assertEqual(len(controller.order), 3)
        self.assertEqual(witness.finish_dispatches, CONTROL_DISPATCH_BOUND * 3)


class TheOtherDomainsAreUnchangedTests(ControlRepairBoundTestBase):
    def test_a_backend_outage_gets_no_protocol_repair(self) -> None:
        run, witness, escaped, _ = self._run(
            questions=1, finish=("outage",) * 20
        )
        self.assertIsNone(escaped)
        # One dispatch: an outage is not a protocol failure to repair.
        self.assertEqual(witness.finish_dispatches, 1)
        self.assertIn("backend error", str(run.stop_reason))
        self.assertEqual(run.control_repairs, 0)

    def test_an_admission_failure_keeps_its_domain(self) -> None:
        run, witness, escaped, _ = self._run(
            questions=1, finish=("admission",) * 20
        )
        self.assertIsNone(escaped)
        self.assertEqual(witness.finish_dispatches, 1)
        self.assertIn("ContextAdmissionError", str(run.stop_reason))

    def test_a_cancellation_is_still_a_cancellation(self) -> None:
        run, witness, escaped, _ = self._run(
            questions=1, finish=("interrupt",) * 20
        )
        self.assertIsNone(escaped, "KeyboardInterrupt stays contained")
        self.assertTrue(run.cancelled)
        self.assertEqual(run.stop_reason, "cancelled")
        self.assertEqual(witness.finish_dispatches, 1)

    # -- the SECOND dispatch keeps its own domain too ----------------------
    #
    # The repair dispatch is inside a `try` now, and what that clause catches
    # decides whether the other domains survive it. Widening it to
    # `BaseException` passes every test above -- the first dispatch is a parse
    # failure in all of them -- while swallowing a cancellation raised by the
    # repair itself. These three script the second dispatch specifically.

    def test_a_cancellation_during_the_repair_is_not_swallowed(self) -> None:
        run, witness, escaped, _ = self._run(
            questions=1, finish=("parse", "interrupt")
        )
        self.assertIsNone(escaped)
        self.assertTrue(run.cancelled, "the analyst stopped it, and it says so")
        self.assertEqual(run.stop_reason, "cancelled")
        self.assertEqual(witness.finish_dispatches, CONTROL_DISPATCH_BOUND)

    def test_an_outage_during_the_repair_keeps_the_backend_domain(self) -> None:
        run, witness, escaped, _ = self._run(
            questions=1, finish=("parse", "outage")
        )
        self.assertIsNone(escaped)
        self.assertIn("backend error", str(run.stop_reason))
        self.assertEqual(witness.finish_dispatches, CONTROL_DISPATCH_BOUND)

    def test_an_admission_failure_during_the_repair_keeps_its_domain(self) -> None:
        run, witness, escaped, _ = self._run(
            questions=1, finish=("parse", "admission")
        )
        self.assertIsNone(escaped)
        self.assertIn("ContextAdmissionError", str(run.stop_reason))
        self.assertEqual(witness.finish_dispatches, CONTROL_DISPATCH_BOUND)


if __name__ == "__main__":
    unittest.main()
