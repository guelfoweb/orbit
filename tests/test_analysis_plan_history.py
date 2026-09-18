"""PLAN owns call-local context, never an incomplete permanent-history turn."""
from __future__ import annotations

import copy
from unittest import mock

from orbit.backend.base import RecoverableBackendError, ToolCallParseError
from orbit.runtime.analysis_runtime import PLAN_TOOL_NAME
from orbit.runtime.context_manager import ContextAdmissionError
from tests.test_analysis_controller_runtime import _Backend, _Model
from tests.test_analysis_evidence_first import EvidenceFirstTestBase, OPENING, decodable


class PlanHistoryTests(EvidenceFirstTestBase):
    def make_runtime(self):
        return self.runtime(_Backend(_Model(plan=[])), decodable())

    def assert_committed(self, runtime, committed):
        self.assertEqual(runtime.messages, committed)
        self.assertEqual(runtime.analyst_turns, 0)
        history = "\n".join(str(m.get("content", "")) for m in runtime.messages)
        for _stage, record in runtime.transform_stages:
            self.assertEqual(history.count(record.evidence_id), 1)
        self.assertNotIn("Before running anything", history)

    def test_admission_fallback_preserves_committed_preamble_without_opening_turn(self):
        runtime = self.make_runtime()
        committed = copy.deepcopy(runtime.messages)
        self.assertTrue(runtime.transform_stages)
        self.assertTrue(committed[-1]["content"].startswith("Deterministic transformations"))
        admit = runtime._admit
        histories = []

        def refuse_first(messages, **kwargs):
            self.assertEqual(kwargs["tools"][0]["function"]["name"], PLAN_TOOL_NAME)
            # PLAN's extra user context exists only in the argument, not history.
            self.assertIn("Before running anything", str(messages))
            histories.append(copy.deepcopy(runtime.messages))
            self.assertEqual(runtime.analyst_turns, 0)
            if len(histories) == 1:
                self.assertEqual(runtime.messages, committed)
                raise ContextAdmissionError("required-context-does-not-fit")
            return admit(messages, **kwargs)

        with mock.patch.object(runtime, "_admit", side_effect=refuse_first):
            result = runtime.run_autonomous(
                OPENING, cover=False, finalize=False, max_model_calls=1)

        self.assertEqual(len(histories), 2, "the PLAN admission fallback must execute")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.actions_executed, 0)
        self.assertEqual(histories, [committed, committed])
        self.assert_committed(runtime, committed)
        sent = runtime.backend.chat_calls[0]["messages"]
        for _stage, record in runtime.transform_stages:
            self.assertIn(record.evidence_id, str(sent))

    def test_repeated_failed_fallback_is_idempotent(self):
        runtime = self.make_runtime()
        committed = copy.deepcopy(runtime.messages)
        for _ in range(3):
            with mock.patch.object(runtime, "_admit", side_effect=ContextAdmissionError(
                    "required-context-does-not-fit")) as admit:
                result = runtime.run_autonomous(OPENING, cover=False, finalize=False)
            self.assertEqual(admit.call_count, 2)
            self.assertEqual(result.model_calls, 0)
            self.assert_committed(runtime, committed)

    def test_other_plan_failures_own_no_history_cleanup(self):
        for error in (KeyboardInterrupt(), ToolCallParseError("malformed"),
                      TimeoutError("timeout"), RecoverableBackendError("unavailable")):
            with self.subTest(error=type(error).__name__):
                runtime = self.make_runtime()
                committed = copy.deepcopy(runtime.messages)
                with mock.patch.object(runtime, "_control_call", side_effect=error):
                    runtime.run_autonomous(OPENING, cover=False, finalize=False)
                self.assert_committed(runtime, committed)

    def test_actual_incomplete_step_still_closes_only_its_own_turn(self):
        for error in (KeyboardInterrupt(), RecoverableBackendError("unavailable")):
            with self.subTest(error=type(error).__name__):
                runtime = self.make_runtime()
                committed = copy.deepcopy(runtime.messages)

                def fail_step(messages, **kwargs):
                    self.assertEqual(runtime.messages[:-1], committed)
                    self.assertEqual(runtime.messages[-1]["role"], "user")
                    self.assertEqual(runtime.analyst_turns, 1)
                    raise error

                with mock.patch.object(runtime, "_admit", side_effect=fail_step):
                    runtime.run_autonomous(
                        OPENING, cover=False, plan=False, finalize=False)
                self.assert_committed(runtime, committed)
