"""The one bounded replan must reach the STEP that spends it."""
import copy
import json
from unittest import mock

from orbit.runtime import analysis_runtime as ar
from orbit.runtime.analysis_controller import AnalysisController
from orbit.runtime.analysis_progress import NEW_CONTENT, NO_PROGRESS
from tests.test_analysis_autonomous import AutonomousTestBase, emit
from tests.test_analysis_runtime import ScriptedBackend, tool_response, prose_response
from tests.test_analysis_repair import ExactRecordingBackend
from tests import test_analysis_ioc_objectives as ioc_fixtures


class ReplanDeliveryTests(AutonomousTestBase):
    def backend(self, *responses, **kwargs):
        return ScriptedBackend(*responses, plan_questions=["Inspect local facts"], **kwargs)

    def step_requests(self, backend):
        return [messages for messages, tools in zip(backend.seen_messages, backend.seen_tools)
                if tools == [ar.ANALYSIS_TOOL_NAME]]

    def test_immediate_replan_receives_instruction_once_and_keeps_question_and_evidence(self):
        same = tool_response(emit("safe observation"))
        backend = self.backend(same, same, same)
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("Inspect", cover=False, finalize=False)
        requests = self.step_requests(backend)
        for messages in requests[:2]:
            self.assertNotIn(ar.AUTONOMOUS_REPLAN_MESSAGE,
                             [m.get("content") for m in messages])
        instruction = requests[2][-2]["content"]
        self.assertEqual(instruction, ar.AUTONOMOUS_REPLAN_MESSAGE)
        self.assertEqual(sum(m.get("content") == instruction for m in requests[2]), 1)
        self.assertIn("Inspect local facts", requests[2][-1]["content"])
        self.assertIn("What is missing: needs execution", requests[2][-1]["content"])
        self.assertTrue(any(m.get("role") == "tool" for m in requests[2]))
        self.assertEqual([p.classification for p in run.progress],
                         [NEW_CONTENT, NO_PROGRESS, NO_PROGRESS])
        self.assertEqual(run.model_calls, 5)
        self.assertEqual(run.replans, 1)
        self.assertEqual(run.actions_executed, 1)
        self.assertTrue(run.stop_reason.startswith(ar.STOP_NO_PROGRESS))
        self.assertEqual(run.answered_unverified_questions, ())

    def test_replan_that_produces_new_evidence_continues(self):
        same = tool_response(emit("known"))
        backend = self.backend(same, same, tool_response(emit("new local fact")))
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("Inspect", cover=False, finalize=False)
        self.assertEqual([p.classification for p in run.progress],
                         [NEW_CONTENT, NO_PROGRESS, NEW_CONTENT])
        self.assertEqual(run.actions_executed, 2)
        self.assertNotIn(ar.STOP_NO_PROGRESS, run.stop_reason)
        self.assertEqual(self.step_requests(backend)[2][-2]["content"], ar.AUTONOMOUS_REPLAN_MESSAGE)

    def test_replan_preserves_explicit_evidence_requests_through_exact_admission(self):
        body = "EXACT-BODY-" + "safe-value-" * 400
        backend = ExactRecordingBackend(tool_response(emit(body)))
        runtime = self.runtime(backend)
        step = runtime.step("Capture a safe observation")
        controller = AnalysisController()
        controller.adopt_plan([{
            "question": (
                "Inspect exact evidence:" + step.evidence.evidence_id
                + " and evidence:" + step.evidence.metadata["raw_output_evidence_id"]
            ),
            "missing_fact": "Interpret the complete retained observation",
        }])
        question = controller.activate_next()
        original_history = copy.deepcopy(runtime.messages)
        for instruction in (None, ar.AUTONOMOUS_REPLAN_MESSAGE):
            with self.subTest(instruction=instruction):
                messages = runtime._resolve_messages(
                    controller, question, current_instruction=instruction,
                )
                admitted = runtime._admit(
                    messages, max_tokens=runtime.effective_max_tokens,
                    tools=[ar.ANALYSIS_TOOL_SCHEMA], next_action_reserve=0,
                )
                delivered = "\n".join(m.get("content", "") for m in admitted)
                self.assertIn("deterministic_evidence_rehydration:", delivered)
                self.assertIn(body, delivered)
                self.assertIn(question.question, delivered)
                self.assertIn(question.missing_fact, delivered)
                self.assertEqual(runtime.messages, original_history)

    def test_different_action_same_output_still_arms_the_replan(self):
        backend = self.backend(tool_response(emit("same")),
                               tool_response("x = 1\n" + emit("same")))
        # Keep another original question actionable after the first's two actions.
        backend._plan_questions.append("Inspect another local fact")
        backend._responses.append(prose_response("No further action."))
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("Inspect", cover=False, finalize=False)
        self.assertEqual(run.progress[1].classification, NO_PROGRESS)
        self.assertEqual(self.step_requests(backend)[2][-2]["content"], ar.AUTONOMOUS_REPLAN_MESSAGE)
        self.assertIn("Inspect another local fact", self.step_requests(backend)[2][-1]["content"])

    def test_refused_replan_is_not_retried_without_its_instruction(self):
        same = tool_response(emit("known"))
        backend = self.backend(same, same, same)
        runtime = self.runtime(backend)
        original = runtime._admit
        refused = []
        before = []

        def admit(messages, **kwargs):
            if messages[-2].get("content") == ar.AUTONOMOUS_REPLAN_MESSAGE:
                refused.append(copy.deepcopy(messages))
                before.append(copy.deepcopy(runtime.messages[:-1]))
                raise ar.ContextAdmissionError("required-context-does-not-fit")
            return original(messages, **kwargs)

        with mock.patch.object(runtime, "_admit", side_effect=admit), \
                mock.patch.object(runtime, "_report_narrative",
                                  return_value=ar.AnalysisReport("", 0)):
            run = runtime.run_autonomous("Inspect", cover=False)
        self.assertEqual(len(refused), 1)
        self.assertEqual(len(self.step_requests(backend)), 2)
        self.assertEqual(runtime.messages, before[0])
        self.assertEqual(runtime.analyst_turns, 2)
        self.assertTrue(run.stop_reason.startswith(ar.STOP_BACKEND_ERROR))
        self.assertIn("ContextAdmissionError", run.stop_reason)
        self.assertIsNotNone(run.final_report)
        self.assertTrue(run.final_report.document_complete)
        self.assertTrue(runtime.evidence_store.reattest_exact(run.steps[0].evidence.evidence_id))

    def test_no_replan_is_inserted_in_a_productive_run(self):
        backend = self.backend(tool_response(emit("one")), tool_response(emit("two")))
        run = self.runtime(backend).run_autonomous("Inspect", cover=False, finalize=False)
        self.assertEqual(run.replans, 0)
        self.assertTrue(all(p.classification == NEW_CONTENT for p in run.progress))
        self.assertTrue(all(ar.AUTONOMOUS_REPLAN_MESSAGE not in json.dumps(m)
                            for m in backend.seen_messages))

    def test_consumed_replan_is_not_reissued_when_stalled_question_is_blocked(self):
        same = tool_response(emit("known"))
        backend = self.backend(same, same, same, tool_response(emit("new fact")))
        backend._plan_questions.append("A different investigation")
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("Inspect", cover=False, finalize=False, max_model_calls=7)
        requests = self.step_requests(backend)
        self.assertEqual(requests[2][-2]["content"], ar.AUTONOMOUS_REPLAN_MESSAGE)
        self.assertIn("A different investigation", requests[3][-1]["content"])
        self.assertEqual(sum(m.get("content") == ar.AUTONOMOUS_REPLAN_MESSAGE
                             for m in requests[3]), 1)  # Committed history only.
        self.assertEqual(run.replans, 1)
        self.assertEqual([p.classification for p in run.progress[:4]],
                         [NEW_CONTENT, NO_PROGRESS, NO_PROGRESS, NEW_CONTENT])
        self.assertEqual(run.actions_executed, 2)

    def test_exact_ioc_does_not_skip_useful_work_or_certify_incomplete_discovery(self):
        backend = self.backend(tool_response(emit("local fact one")),
                               tool_response(emit("local fact two")))
        runtime = ioc_fixtures.ObjectiveTests.runtime(self, '<img src="https://other.invalid/"><script>location.href="https://fixed.invalid/a";</script>' ,
                                         backend)
        before = copy.deepcopy(runtime.ioc_checks())
        self.assertEqual(before[0]["state"], "RESOLVED_EXACT")
        self.assertFalse(runtime._ioc_closed(None))
        run = runtime.run_autonomous("Inspect", cover=False, finalize=False)
        self.assertEqual(run.actions_executed, 2)
        self.assertEqual(runtime.ioc_checks(), before)
        self.assertNotEqual(run.stop_reason, ar.STOP_IOC_CLOSED)

    def test_incomplete_discovery_stagnation_ends_without_certifying_completeness(self):
        same = tool_response(emit("same local observation"))
        backend = self.backend(same, same, same)
        runtime = ioc_fixtures.ObjectiveTests.runtime(self, '<img src="https://other.invalid/"><script>location.href="https://fixed.invalid/a";</script>' ,
                                         backend)
        with mock.patch.object(runtime, "_report_narrative", return_value=ar.AnalysisReport("", 0)):
            run = runtime.run_autonomous("Inspect", cover=False)
        self.assertTrue(run.stop_reason.startswith(ar.STOP_NO_PROGRESS))
        self.assertFalse(runtime._ioc_closed(None))
        self.assertEqual(runtime.ioc_checks()[0]["state"], "RESOLVED_EXACT")
        self.assertIn("discovery incomplete", run.final_report.text)
        self.assertIn("## IoC / Evidence", run.final_report.text)
        self.assertEqual(run.answered_unverified_questions, ())
