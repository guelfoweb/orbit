"""FINISH shares exact context capacity; incomplete generation commits no proposal."""
import copy
import hashlib
import json
from dataclasses import replace
from unittest import mock

from orbit.backend.base import TokenCount, RecoverableBackendError
from orbit.backend.llama_server import LlamaServerToolCallParseError
from orbit.runtime.analysis_controller import AnalysisController, parse_finish_call
from orbit.runtime.analysis_runtime import FINISH_TOOL_SCHEMA, PLAN_TOOL_SCHEMA, ContextAdmissionError
from tests.test_analysis_controller_runtime import _Case, _Model, _Backend, _question


class ExactFinishBackend(_Backend):
    def __init__(self, tokens=2022, *, context=4096):
        super().__init__(_Model(plan=[]))
        self.tokens = tokens
        self.context = context
        self.counts = []

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        self.counts.append(copy.deepcopy((messages, tools, thinking)))
        n = self.tokens(messages) if callable(self.tokens) else self.tokens
        digest = hashlib.sha256(json.dumps([messages, tools, thinking], sort_keys=True).encode()).hexdigest()
        return TokenCount(tokens=n, context_tokens=self.context,
                          rendered_hash=digest, token_hash=digest)


class FinishBudgetTests(_Case):
    def setup_finish(self, tokens=2022, **kw):
        rt = self._runtime(_Model(plan=[]))
        rt.backend = ExactFinishBackend(tokens, **kw)
        rt.context_tokens = 4096
        return rt

    def test_complete_request_uses_only_exact_remaining_capacity(self):
        for n, cap in [(1912, 1928), (2022, 1818), (1965, 1875), (1700, 2048)]:
            with self.subTest(input=n):
                rt = self.setup_finish(n)
                history = copy.deepcopy(rt.messages)
                rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
                call = rt.backend.chat_calls[-1]
                self.assertEqual(call['kwargs']['max_tokens'], cap)
                self.assertEqual(rt.last_context_plan.input_limit, 4096 - 256 - cap)
                self.assertEqual(call['messages'], list(rt.last_context_plan.messages))
                self.assertEqual(rt.messages, history)
                self.assertTrue(all(tools == [FINISH_TOOL_SCHEMA] for _, tools, _ in rt.backend.counts))

    def test_explicit_lower_generation_and_context_limits_win(self):
        rt = self.setup_finish()
        rt.max_tokens = 512
        rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
        self.assertEqual(rt.backend.chat_calls[-1]['kwargs']['max_tokens'], 512)
        rt = self.setup_finish(context=8192)
        rt.context_tokens = 3072
        rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
        self.assertEqual(rt.backend.chat_calls[-1]['kwargs']['max_tokens'], 794)
        rt = self.setup_finish(tokens=1000, context=2048)
        rt.context_tokens = 2048
        rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
        self.assertEqual(rt.backend.chat_calls[-1]['kwargs']['max_tokens'], 792)

    def test_plan_remains_on_existing_fixed_budget(self):
        rt = self.setup_finish()
        with self.assertRaises(ContextAdmissionError):
            rt._control_dispatch(rt.messages, PLAN_TOOL_SCHEMA)
        self.assertEqual(rt.model_calls, 0)
        self.assertEqual(rt.backend.chat_calls, [])

    def test_exhausted_or_nonpositive_output_capacity_never_dispatches(self):
        for n, maximum in [(3840, 2048), (3900, 2048), (2022, 0), (2022, -1)]:
            with self.subTest(tokens=n, maximum=maximum):
                rt = self.setup_finish(n)
                rt.max_tokens = maximum
                with self.assertRaises(ContextAdmissionError):
                    rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
                self.assertFalse(rt.backend.chat_calls)
                if n == 3840:
                    self.assertFalse(rt.last_context_plan.admitted)
                    self.assertEqual(rt.last_context_plan.reason, 'finish-output-capacity-exhausted')

    def test_unknown_or_changed_exact_identity_does_not_enable_fallback(self):
        rt = self.setup_finish()
        orig = rt.backend.count_chat_tokens
        def changed(*args, **kwargs):
            v = orig(*args, **kwargs)
            return replace(v, rendered_hash=str(len(rt.backend.counts)))
        with mock.patch.object(rt.backend, 'count_chat_tokens', side_effect=changed):
            with self.assertRaises(ContextAdmissionError):
                rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
        self.assertEqual(rt.model_calls, 0)

    def test_final_admission_rejects_changed_capacity_before_dispatch(self):
        rt = self.setup_finish()
        def changed(messages):
            return 2022 if len(rt.backend.counts) <= 4 else 2023
        rt.backend.tokens = changed
        with self.assertRaises(ContextAdmissionError):
            rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
        self.assertEqual(rt.model_calls, 0)

    def test_protocol_repair_recounts_its_own_complete_request(self):
        rt = self.setup_finish(lambda ms: 2050 if any('previous control response' in str(m.get('content')) for m in ms) else 2022)
        original = rt.backend.chat_stream
        seen = []
        def generate(messages, **kw):
            seen.append(kw['max_tokens'])
            if len(seen) == 1:
                raise LlamaServerToolCallParseError('unparseable control')
            return original(messages, **kw)
        with mock.patch.object(rt.backend, 'chat_stream', side_effect=generate):
            args, text = rt._control_call(rt.messages, FINISH_TOOL_SCHEMA, repair_budget=1)
        self.assertIsNotNone(args)
        self.assertEqual(seen, [1818, 1790])
        self.assertEqual(rt.model_calls, 2)
        self.assertEqual(rt.control_repairs, 1)

    def test_complete_control_is_accepted_but_incomplete_control_never_applied(self):
        for reason in ['tool_calls', 'stop', 'eos', 'length', None]:
            with self.subTest(reason=reason):
                rt = self.setup_finish(1000)
                c = AnalysisController();c.adopt_plan([_question('Original broad question')]);q=c.activate_next()
                original = rt.backend.chat_stream
                def generate(messages, **kw):
                    return replace(original(messages, **kw), finish_reason=reason)
                with mock.patch.object(rt.backend, 'chat_stream', side_effect=generate), mock.patch.object(rt, '_apply_decision') as apply:
                    rt.finish_question(c, q, 'bounded observation', 'ev_not_attested', max_calls=2)
                if reason in ['tool_calls', 'stop', 'eos']:
                    apply.assert_called_once()
                else:
                    apply.assert_not_called()
                    self.assertEqual(c.states[q.id].summary, '')
                    self.assertNotEqual(c.states[q.id].status, 'answered_unverified')
                    self.assertEqual(rt.model_calls, 1)

    def test_valid_1930_token_control_under_smaller_cap_cannot_commit_after_length(self):
        # Retained counterexample: production tokenizer counts this JSON at
        # 1930 tokens. Structural validity is not proof that generation ended.
        args = {'status': 'still_open', 'answer_summary': '\U00010000' * 480}
        parse_finish_call(args)
        rt = self.setup_finish(1912)
        c = AnalysisController();c.adopt_plan([_question('Keep the original objective')]);q=c.activate_next()
        original = rt.backend.chat_stream
        def generate(messages, **kw):
            response = original(messages, **kw)
            response.tool_calls[0]['function']['arguments'] = json.dumps(args, ensure_ascii=False, separators=(',', ':'))
            return replace(response, finish_reason='length', completion_tokens=kw['max_tokens'])
        with mock.patch.object(rt.backend, 'chat_stream', side_effect=generate), mock.patch.object(rt, '_apply_decision') as apply:
            rt.finish_question(c, q, 'observation', 'ev', max_calls=2)
        apply.assert_not_called()
        self.assertEqual(rt.backend.chat_calls[-1]['kwargs']['max_tokens'], 1928)
        self.assertEqual(c.states[q.id].summary, '')

    def test_mixed_protocol_and_schema_failure_cannot_multiply_repairs(self):
        rt = self.setup_finish(1000)
        c = AnalysisController();c.adopt_plan([_question('Original objective')]);q=c.activate_next()
        original = rt.backend.chat_stream
        dispatches = []
        def generate(messages, **kw):
            dispatches.append(kw['max_tokens'])
            if len(dispatches) == 1:
                raise LlamaServerToolCallParseError('first malformed control')
            response = original(messages, **kw)
            response.tool_calls[0]['function']['arguments'] = '{"status":"invalid"}'
            return response
        with mock.patch.object(rt.backend, 'chat_stream', side_effect=generate):
            calls = rt.finish_question(c, q, 'observation', 'ev', max_calls=2)
        self.assertEqual(len(dispatches), 2)
        self.assertEqual(calls, 2)
        self.assertEqual(c.states[q.id].status, 'blocked')
        self.assertEqual(c.states[q.id].summary, '')

    def test_cutoff_after_internal_repair_gets_no_outer_retry(self):
        rt = self.setup_finish(1000)
        c = AnalysisController();c.adopt_plan([_question('Original objective')]);q=c.activate_next()
        original = rt.backend.chat_stream
        dispatches = []
        def generate(messages, **kw):
            dispatches.append(kw['max_tokens'])
            if len(dispatches) == 1:
                raise LlamaServerToolCallParseError('malformed first control')
            return replace(original(messages, **kw), finish_reason='length')
        with mock.patch.object(rt.backend, 'chat_stream', side_effect=generate), mock.patch.object(rt, '_apply_decision') as apply:
            rt.finish_question(c, q, 'observation', 'ev', max_calls=2)
        apply.assert_not_called()
        self.assertEqual(len(dispatches), 2)
        self.assertIn('length', c.states[q.id].reason)

    def test_previous_blocked_plan_cannot_override_failed_evidence_rehydration(self):
        rt = self.setup_finish()
        try:
            rt._admit(rt.messages, max_tokens=2048, tools=[FINISH_TOOL_SCHEMA], next_action_reserve=0)
        except ContextAdmissionError:
            pass
        self.assertEqual(rt.last_context_plan.reason, 'required-context-does-not-fit')
        with mock.patch.object(rt, '_with_evidence_rehydration', side_effect=ContextAdmissionError('evidence-rehydration-unavailable')):
            with self.assertRaisesRegex(ContextAdmissionError, 'evidence-rehydration-unavailable'):
                rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
        self.assertEqual(rt.model_calls, 0)

    def test_zero_remaining_call_allowance_does_not_dispatch(self):
        rt = self.setup_finish()
        c = AnalysisController();c.adopt_plan([_question('Original objective')]);q=c.activate_next()
        self.assertEqual(rt.finish_question(c, q, 'observation', 'ev', max_calls=0), 0)
        self.assertEqual(rt.model_calls, 0)
        self.assertEqual(c.states[q.id].status, 'open')

    def test_returned_outage_and_cancel_stop_the_run_and_preserve_evidence(self):
        from orbit.runtime import analysis_runtime as ar
        for reason, error in [('cancelled', KeyboardInterrupt), ('error', RecoverableBackendError), ('timeout', TimeoutError)]:
            with self.subTest(reason=reason):
                rt = self._runtime(_Model(plan=[_question('Original broad question'), _question('Second question')]))
                original = rt.backend.chat_stream
                seen = []
                def generate(messages, **kw):
                    response = original(messages, **kw)
                    names = [t['function']['name'] for t in (kw.get('tools') or [])]
                    seen.append(names)
                    if names == [ar.FINISH_TOOL_NAME]:
                        return replace(response, finish_reason=reason)
                    return response
                def action(**kw):
                    return ar.AnalysisResult(status='ok', code_sha256='c'*64,
                        input_sha256=rt.source.sha256, stdout='retained observation',
                        stderr='', exit_status=0, duration_seconds=0.1)
                with mock.patch.object(rt.backend, 'chat_stream', side_effect=generate), mock.patch.object(ar, 'execute_analysis', side_effect=action):
                    run = rt.run_autonomous('Analyse it.', cover=False, finalize=True)
                self.assertEqual(run.actions_executed, 1)
                self.assertEqual(run.answered_unverified_questions, ())
                self.assertIn('retained observation', run.final_report.dossier_text)
                self.assertTrue(run.final_report.document_complete)
                self.assertEqual(run.cancelled, reason == 'cancelled')
                if reason == 'cancelled':
                    self.assertEqual(seen[-1], [ar.FINISH_TOOL_NAME])
                    self.assertEqual(run.final_report.model_calls, 0)
                else:
                    self.assertIn('backend error', run.stop_reason)
                    # Existing outage policy may make its one optional report
                    # call, but must not investigate the next question.
                    self.assertEqual(sum(x == [ar.ANALYSIS_TOOL_NAME] for x in seen), 1)

    def test_rehydrated_active_evidence_is_frozen_through_cap_fallback(self):
        rt = self.setup_finish()
        original = rt._with_evidence_rehydration
        seen = []
        def hydrate(messages, **kw):
            seen.append(copy.deepcopy(messages))
            hydrated, ids = original(messages, **kw)
            return [*hydrated, {'role': 'system', 'content': 'deterministic_evidence_rehydration:\nEXACT_REHYDRATED_BYTES'}], ids
        with mock.patch.object(rt, '_with_evidence_rehydration', side_effect=hydrate):
            rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
        self.assertEqual(len(seen), 1)
        sent = rt.backend.chat_calls[-1]['messages']
        self.assertIn('EXACT_REHYDRATED_BYTES', sent[-1]['content'])
        self.assertTrue(all(ms == sent for ms, _, _ in rt.backend.counts))

    def test_loss_of_exact_count_refuses_before_generation(self):
        rt = self.setup_finish()
        original = rt.backend.count_chat_tokens
        def count(messages, **kw):
            if len(rt.backend.counts) >= 2:
                return None
            return original(messages, **kw)
        with mock.patch.object(rt.backend, 'count_chat_tokens', side_effect=count):
            with self.assertRaises(ContextAdmissionError):
                rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
        self.assertEqual(rt.model_calls, 0)

    def test_context_consumed_by_safety_never_dispatches(self):
        for ctx in (128, 256):
            rt = self.setup_finish(tokens=100, context=ctx)
            rt.context_tokens = ctx
            with self.assertRaises(ContextAdmissionError):
                rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
            self.assertEqual(rt.model_calls, 0)

    def test_existing_compaction_is_preserved_and_its_cost_remains_visible(self):
        rt = self._runtime(_Model(plan=[_question('Original question')]))
        self._run(rt, cover=False)
        history = copy.deepcopy(rt.messages)
        def tokens(messages):
            archived = any('archived' in call['function']['arguments']
                           for m in messages for call in m.get('tool_calls', []))
            return 2022 if archived else 2200
        rt.backend = ExactFinishBackend(tokens)
        rt.context_tokens = 4096
        before = rt.context_compactions
        rt._control_dispatch([*rt.messages, {'role': 'user', 'content': 'Finish this question.'}], FINISH_TOOL_SCHEMA)
        self.assertEqual(rt.last_context_plan.tokens_before, 2200)
        self.assertEqual(rt.last_context_plan.tokens_after, 2022)
        self.assertEqual(rt.last_context_plan.compacted_turns, 1)
        self.assertTrue(rt.last_context_plan.externalized_evidence_ids)
        self.assertEqual(rt.context_compactions, before + 1)
        self.assertEqual(rt.backend.chat_calls[-1]['kwargs']['max_tokens'], 1818)
        self.assertEqual(rt.messages, history)

    def test_cutoff_keeps_prior_open_answer_and_citations(self):
        for reason in ('length', 'empty_response'):
            with self.subTest(reason=reason):
                rt = self._runtime(_Model(plan=[_question('Original broad question')]))
                run = self._run(rt, cover=False)
                eid = run.last_step.evidence.evidence_id
                c = AnalysisController()
                c.adopt_plan([_question('Original broad question')])
                q = c.activate_next()
                rt._apply_decision(c, parse_finish_call({
                    'status': 'still_open', 'answer_summary': 'Established partial observation; wider question remains open.',
                    'evidence_ids': [eid],
                }), eid)
                prior = copy.deepcopy(c.states[q.id])
                self.assertEqual(prior.status, 'open')
                self.assertEqual(prior.evidence_ids, (eid,))
                rt.backend = ExactFinishBackend(1000)
                original = rt.backend.chat_stream
                with mock.patch.object(rt.backend, 'chat_stream', side_effect=lambda *a, **kw: replace(original(*a, **kw), finish_reason=reason)):
                    rt.finish_question(c, q, 'Later bounded observation', eid, max_calls=2)
                state = c.states[q.id]
                self.assertEqual(state.status, 'blocked')
                self.assertEqual(state.summary, prior.summary)
                self.assertEqual(state.evidence_ids, prior.evidence_ids)
                self.assertEqual(len(c.questions), 1)

    def test_empty_control_uses_existing_repair_and_recounts_its_cap(self):
        # Native Qwen: 38 generated tokens ended at EOG, but the unavailable
        # read_artifact tool was correctly rejected; the response was empty.
        rt = self._runtime(_Model(plan=[_question('Original broad question')]))
        run = self._run(rt, cover=False)
        eid = run.last_step.evidence.evidence_id
        rt.backend = ExactFinishBackend(lambda ms: 2070 if any('That could not be used' in str(m.get('content')) for m in ms) else 2022)
        c = AnalysisController();c.adopt_plan([_question('Original broad question')]);q=c.activate_next()
        original = rt.backend.chat_stream
        caps = []
        def generate(messages, **kw):
            caps.append(kw['max_tokens'])
            response = original(messages, **kw)
            if len(caps) == 1:
                return replace(response, content='', tool_calls=[], finish_reason='empty_response', completion_tokens=38)
            return response
        with mock.patch.object(rt.backend, 'chat_stream', side_effect=generate):
            calls = rt.finish_question(c, q, 'Bounded observation', eid, max_calls=2)
        self.assertEqual(calls, 2)
        self.assertEqual(caps, [1818, 1770])
        self.assertEqual(c.repairs, 1)
        self.assertEqual(c.states[q.id].status, 'answered_unverified')
        self.assertEqual(c.states[q.id].evidence_ids, (eid,))
        self.assertEqual(c.questions[q.id].question, 'Original broad question')

    def test_empty_response_label_cannot_authorize_attached_control(self):
        rt = self.setup_finish(1000)
        c=AnalysisController();c.adopt_plan([_question('Original question')]);q=c.activate_next()
        original = rt.backend.chat_stream
        with mock.patch.object(rt.backend, 'chat_stream', side_effect=lambda *a, **kw: replace(original(*a, **kw), finish_reason='empty_response')), mock.patch.object(rt, '_apply_decision') as apply:
            calls = rt.finish_question(c, q, 'Observation', 'ev', max_calls=2)
        apply.assert_not_called()
        self.assertEqual(calls, 2)
        self.assertEqual(c.states[q.id].status, 'blocked')

    def test_empty_after_internal_repair_cannot_get_a_third_dispatch(self):
        rt=self.setup_finish(1000)
        c=AnalysisController();c.adopt_plan([_question('Original question')]);q=c.activate_next()
        original=rt.backend.chat_stream
        seen=[]
        def generate(*a, **kw):
            seen.append(kw['max_tokens'])
            if len(seen)==1:
                raise LlamaServerToolCallParseError('malformed first control')
            return replace(original(*a, **kw), tool_calls=[], finish_reason='empty_response')
        with mock.patch.object(rt.backend,'chat_stream',side_effect=generate):
            calls=rt.finish_question(c,q,'Observation','ev',max_calls=2)
        self.assertEqual(calls,2)
        self.assertEqual(len(seen),2)
        self.assertEqual(c.states[q.id].status,'blocked')

    def test_empty_response_cannot_repair_without_a_remaining_call(self):
        rt=self.setup_finish(1000)
        c=AnalysisController();c.adopt_plan([_question('Original question')]);q=c.activate_next()
        original=rt.backend.chat_stream
        with mock.patch.object(rt.backend,'chat_stream',side_effect=lambda *a,**kw: replace(original(*a,**kw),tool_calls=[],finish_reason='empty_response')):
            calls=rt.finish_question(c,q,'Observation','ev',max_calls=1)
        self.assertEqual(calls,1)
        self.assertEqual(c.repairs,0)
        self.assertEqual(c.states[q.id].status,'blocked')
