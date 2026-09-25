"""STEP owns one admitted generation, one transient turn, and no future call."""
import copy
import json
from unittest import mock

from tests.test_analysis_controller_runtime import _Case, _Model, _question
from orbit.backend.base import TokenCount
from orbit.backend.llama_server import _parse_chat_result
from orbit.runtime import analysis_runtime as ar
from orbit.runtime.analysis_sandbox import AnalysisResult


class StepContractTests(_Case):
    def _response(self, reason="tool_calls"):
        return _parse_chat_result({"model": "safe-fixture", "choices": [{
            "finish_reason": reason, "message": {"content": "", "tool_calls": [{
                "id": "fixture", "type": "function", "function": {
                    "name": "execute_analysis", "arguments": json.dumps({"code": "print(1)"})}}]}}]})

    def _output(self, rt):
        return AnalysisResult(status="ok", code_sha256="1" * 64,
                              input_sha256=rt.source.sha256, stdout="1\n",
                              stderr="", exit_status=0, duration_seconds=0)

    def test_step_does_not_reserve_tokens_for_independently_admitted_call(self):
        rt = self._runtime(_Model([]))
        rt.context_tokens = 4096
        count = TokenCount(tokens=1577, context_tokens=4096,
                           rendered_hash="a" * 64, token_hash="b" * 64)
        with mock.patch.object(rt.backend, "count_chat_tokens", return_value=count), \
             mock.patch.object(rt.backend, "chat_stream", return_value=self._response()) as generate, \
             mock.patch.object(ar, "execute_analysis", return_value=self._output(rt)):
            step = rt.step("Read the safe fixture.")
        self.assertTrue(step.action_executed)
        self.assertEqual(rt.last_context_plan.input_limit, 1792)
        self.assertEqual(generate.call_args.kwargs["max_tokens"], 2048)

    def test_length_with_parsable_call_never_reaches_executor(self):
        rt = self._runtime(_Model([]))
        with mock.patch.object(rt.backend, "chat_stream", return_value=self._response("length")), \
             mock.patch.object(ar, "execute_analysis", return_value=self._output(rt)) as execute:
            step = rt.step("Read the safe fixture.")
        execute.assert_not_called()
        self.assertFalse(step.action_executed)
        self.assertIsNotNone(step.rejection)
        self.assertFalse(rt.evidence_store.records)

    def test_repeated_admission_refusal_restores_only_owned_turn(self):
        rt = self._runtime(_Model([]))
        rt.messages.append({"role": "user", "content": "Committed deterministic evidence preamble"})
        before = copy.deepcopy(rt.messages)
        turns = rt.analyst_turns
        with mock.patch.object(rt, "_admit", side_effect=ar.ContextAdmissionError("too-large")), \
             mock.patch.object(ar, "execute_analysis") as execute:
            for request in ("Evidence-first instruction", "Plain fallback", "Plain fallback"):
                with self.assertRaises(ar.ContextAdmissionError):
                    rt.step(request)
                self.assertEqual(rt.messages, before)
                self.assertEqual(rt.analyst_turns, turns)
        execute.assert_not_called()
        self.assertEqual(rt.model_calls, 0)

    def test_complete_calls_still_execute_with_provenance(self):
        for reason in ("stop", "tool_calls", "eos"):
            with self.subTest(reason=reason):
                rt = self._runtime(_Model([]))
                with mock.patch.object(rt.backend, "chat_stream", return_value=self._response(reason)), \
                     mock.patch.object(ar, "execute_analysis", return_value=self._output(rt)) as execute:
                    step = rt.step("Read the safe fixture.")
                execute.assert_called_once()
                self.assertTrue(step.action_executed)
                self.assertIsNotNone(step.evidence)
                self.assertEqual(rt.analyst_turns, 1)
                self.assertEqual(rt.messages[-1]["role"], "tool")

    def test_incomplete_calls_do_not_assign_ids_or_create_evidence(self):
        for reason in ("length", None, "empty_response", "unknown"):
            with self.subTest(reason=reason):
                rt = self._runtime(_Model([]))
                with mock.patch.object(rt.backend, "chat_stream", return_value=self._response(reason)), \
                     mock.patch.object(rt, "_with_canonical_call_ids", wraps=rt._with_canonical_call_ids) as ids, \
                     mock.patch.object(ar, "execute_analysis") as execute:
                    step = rt.step("Read it.")
                execute.assert_not_called()
                ids.assert_not_called()
                self.assertFalse(rt.evidence_store.records)
                self.assertFalse(ar._is_locally_repairable(step))
                self.assertTrue(step.rejection)
                self.assertFalse(any(m.get("tool_calls") for m in rt.messages))

    def test_incomplete_prose_does_not_claim_completion(self):
        rt = self._runtime(_Model([]))
        response = self._response("length")
        response.tool_calls.clear()
        with mock.patch.object(rt.backend, "chat_stream", return_value=response), \
             mock.patch.object(ar, "execute_analysis") as execute:
            step = rt.step("Read it.")
        self.assertTrue(step.rejection)
        execute.assert_not_called()

    def test_cancel_timeout_error_restore_history_and_keep_dispatch_cost(self):
        for reason, exception in (("cancelled", KeyboardInterrupt), ("canceled", KeyboardInterrupt),
                                  ("timeout", TimeoutError), ("error", ar.RecoverableBackendError)):
            with self.subTest(reason=reason):
                rt = self._runtime(_Model([]))
                before = copy.deepcopy(rt.messages)
                with mock.patch.object(rt.backend, "chat_stream", return_value=self._response(reason)), \
                     mock.patch.object(ar, "execute_analysis") as execute:
                    with self.assertRaises(exception):
                        rt.step("Read it.")
                execute.assert_not_called()
                self.assertEqual(rt.messages, before)
                self.assertEqual(rt.analyst_turns, 0)
                self.assertEqual(rt.model_calls, 1)

    def test_owned_cleanup_does_not_remove_a_foreign_appended_turn(self):
        rt = self._runtime(_Model([]))
        foreign = {"role": "user", "content": "Committed outside this STEP"}
        def refuse(*args, **kwargs):
            rt.messages.append(foreign)
            raise ar.ContextAdmissionError("refused")
        with mock.patch.object(rt, "_admit", side_effect=refuse):
            with self.assertRaises(ar.ContextAdmissionError):
                rt.step("Owned by STEP")
        self.assertIs(rt.messages[-1], foreign)
        self.assertEqual(rt.analyst_turns, 1)

    def test_admission_boundaries_and_explicit_smaller_caps(self):
        for native, configured, output in ((4096,4096,2048), (8192,8192,2048),
                                          (8192,4096,2048), (4096,8192,2048),
                                          (4096,4096,512), (4096,4096,4096)):
            allowance = min(native, configured) - min(output, 2048) - 256
            for tokens in (allowance, allowance + 1):
                with self.subTest(native=native, configured=configured, output=output, tokens=tokens):
                    rt = self._runtime(_Model([]))
                    rt.context_tokens = configured
                    rt.max_tokens = output
                    count = TokenCount(tokens=tokens, context_tokens=native,
                                       rendered_hash="a"*64, token_hash="b"*64)
                    with mock.patch.object(rt.backend, "count_chat_tokens", return_value=count), \
                         mock.patch.object(rt.backend, "chat_stream", return_value=self._response()) as generate, \
                         mock.patch.object(ar, "execute_analysis", return_value=self._output(rt)) as execute:
                        if tokens == allowance:
                            self.assertTrue(rt.step("Read.").action_executed)
                            self.assertEqual(generate.call_args.kwargs["max_tokens"], min(output,2048))
                        else:
                            before = copy.deepcopy(rt.messages)
                            with self.assertRaises(ar.ContextAdmissionError):
                                rt.step("Read.")
                            generate.assert_not_called()
                            execute.assert_not_called()
                            self.assertEqual(rt.messages, before)

    def test_autonomous_fallback_has_no_unanswered_turn_or_lost_preamble(self):
        rt = self._runtime(_Model([_question("first question"), _question("original objective")]))
        preamble = {"role": "user", "content": "Committed deterministic evidence preamble"}
        rt.messages.append(preamble)
        inputs = []
        original = rt._admit
        def admit(messages, **kwargs):
            if kwargs.get("tools") == [ar.ANALYSIS_TOOL_SCHEMA] and "original objective" in messages[-1]["content"]:
                inputs.append(copy.deepcopy(messages))
                raise ar.ContextAdmissionError("required-context-does-not-fit")
            return original(messages, **kwargs)
        with mock.patch.object(rt, "_admit", side_effect=admit), \
             mock.patch.object(ar, "execute_analysis", return_value=self._output(rt)) as execute:
            run = rt.run_autonomous("Analyse it.", cover=False, finalize=True)
        execute.assert_called_once()
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0], inputs[1])
        self.assertIn(preamble, rt.messages)
        self.assertEqual(rt.analyst_turns, 1)
        self.assertIn("ContextAdmissionError", run.stop_reason)
        self.assertTrue(run.final_report.text)
        self.assertIn("original objective", run.final_report.dossier_text)

    def test_incomplete_stream_is_rejected_by_actual_transport_path(self):
        import io
        from orbit.backend.llama_server import _parse_native_stream
        call = self._response().tool_calls[0]
        for done in (None, "length", "tool_calls"):
            rt = self._runtime(_Model([]))
            wire = 'event: tool_calls\ndata: ' + json.dumps({'tool_calls': [call]}) + '\n\n'
            if done:
                wire += 'event: done\ndata: ' + json.dumps({'finish_reason': done}) + '\n\n'
            response = _parse_native_stream(io.BytesIO(wire.encode()), on_delta=lambda _: None, on_progress=None)
            with mock.patch.object(rt.backend, "chat_stream", return_value=response), \
                 mock.patch.object(ar, "execute_analysis", return_value=self._output(rt)) as execute:
                step = rt.step("Read.")
            self.assertEqual(execute.call_count, int(done == "tool_calls"))
            self.assertEqual(step.action_executed, done == "tool_calls")

    def test_fallback_then_success_preserves_committed_history(self):
        rt = self._runtime(_Model([_question("q1"), _question("q2")]))
        original = rt._admit
        attempts = []
        def admit(messages, **kwargs):
            if kwargs.get("tools") == [ar.ANALYSIS_TOOL_SCHEMA] and "q2" in messages[-1]["content"]:
                attempts.append(copy.deepcopy(messages))
                if len(attempts) == 1:
                    raise ar.ContextAdmissionError("too-large")
            return original(messages, **kwargs)
        with mock.patch.object(rt, "_admit", side_effect=admit):
            run = self._run(rt, cover=False)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        self.assertEqual(run.actions_executed, 2)
        self.assertEqual(rt.analyst_turns, 2)
        self.assertEqual(run.answered_unverified_questions, ("Q1", "Q2"))

    def test_owned_action_repair_uses_same_step_budget_and_exact_error(self):
        from dataclasses import replace
        rt = self._runtime(_Model([_question("original objective")], decisions=[
            {"status": "still_open", "answer_summary": "The action failed."},
            {"status": "resolved", "answer_summary": "Observed 1."}]))
        failure = replace(self._output(rt), status="error", stdout="", exit_status=1,
                          stderr="Traceback (most recent call last):\nTypeError: safe fixture failure")
        with mock.patch.object(rt, "_admit", wraps=rt._admit) as admit, \
             mock.patch.object(ar, "execute_analysis", side_effect=[failure, self._output(rt)]):
            run = rt.run_autonomous("Analyse it.", cover=False, finalize=False)
        steps = [c for c in admit.call_args_list if c.kwargs.get("tools") == [ar.ANALYSIS_TOOL_SCHEMA]]
        self.assertEqual(len(steps), 2)
        self.assertTrue(all(c.kwargs["next_action_reserve"] == 0 for c in steps))
        self.assertTrue(all(c.kwargs["max_tokens"] == 2048 for c in steps))
        self.assertEqual(run.repairs, 1)
        requests = [c for c in rt.backend.chat_calls if c["tools"] == [ar.ANALYSIS_TOOL_SCHEMA]]
        repair = json.dumps(requests[1]["messages"])
        self.assertIn("TypeError: safe fixture failure", repair)
        self.assertIn("print(1)", repair)
        self.assertIn("original objective", repair)
        self.assertEqual(run.actions_executed, 2)

    def test_subsequent_finish_re_admits_and_refuses_safely_with_report(self):
        rt = self._runtime(_Model([_question("original objective")]))
        original = rt._admit
        def admit(messages, **kwargs):
            if kwargs.get("tools") == [ar.FINISH_TOOL_SCHEMA]:
                raise ar.ContextAdmissionError("required-context-does-not-fit")
            return original(messages, **kwargs)
        with mock.patch.object(rt, "_admit", side_effect=admit), \
             mock.patch.object(ar, "execute_analysis", return_value=self._output(rt)) as execute:
            run = rt.run_autonomous("Analyse it.", cover=False, finalize=True)
        execute.assert_called_once()
        self.assertEqual(run.answered_unverified_questions, ())
        self.assertEqual(run.resolved_questions, ())
        self.assertIn("ContextAdmissionError", run.stop_reason)
        self.assertTrue(run.final_report.document_complete)
        self.assertIn("original objective", run.final_report.dossier_text)
        self.assertTrue(rt.evidence_store.records)

    def test_ornith_near_boundary_keeps_more_history_not_less_evidence(self):
        rt = self._runtime(_Model([]))
        with mock.patch.object(rt.backend, "chat_stream", return_value=self._response()), \
             mock.patch.object(ar, "execute_analysis", return_value=self._output(rt)):
            first = rt.step("First safe action.")
        messages = [*copy.deepcopy(rt.messages), {"role": "user", "content": "Next question"}]
        before = copy.deepcopy(rt.messages)
        eid = first.evidence.evidence_id
        def count(ms, **kwargs):
            # Exact controlled counter: archived code is shorter. Both views
            # cite the same re-attested production EvidenceStore record.
            original_code = any('print(1)' in json.dumps(m.get('tool_calls', [])) for m in ms)
            return TokenCount(tokens=5800 if original_code else 5500, context_tokens=8192,
                              rendered_hash="a"*64, token_hash="b"*64)
        with mock.patch.object(rt.backend, "count_chat_tokens", side_effect=count):
            old = rt._admit(messages, max_tokens=2048, tools=[ar.ANALYSIS_TOOL_SCHEMA])
            self.assertEqual(rt.last_context_plan.compacted_turns, 1)
            with mock.patch.object(rt.backend, "chat_stream", return_value=self._response()) as generate, \
                 mock.patch.object(ar, "execute_analysis", return_value=self._output(rt)):
                rt.step("Next question", controller_messages=messages)
            new = generate.call_args.args[0]
        self.assertEqual(rt.last_context_plan.compacted_turns, 0)
        self.assertEqual(new, messages)
        self.assertNotEqual(old, new)
        self.assertIn(eid, json.dumps(old))
        self.assertIn(eid, json.dumps(new))
        self.assertIsNotNone(rt.evidence_store.reattest_exact(eid))
        self.assertEqual(rt.messages[:len(before)], before)
