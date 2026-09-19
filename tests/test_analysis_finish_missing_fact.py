"""FINISH preserves the active question's registered requirement, not proof."""
import copy
from dataclasses import replace
from unittest import mock

from orbit.backend.llama_server import LlamaServerToolCallParseError
from orbit.runtime.analysis_controller import AnalysisController, ControlError
from orbit.runtime.analysis_runtime import FINISH_TOOL_SCHEMA, ContextAdmissionError
from tests.test_analysis_controller_runtime import _Case, _Model, _question
from tests.test_analysis_finish_budget import ExactFinishBackend


class FinishMissingFactTests(_Case):
    def setup_finish(self, missing='The exact line count printed by the action.'):
        model = _Model(plan=[])
        rt = self._runtime(model)
        c = AnalysisController()
        c.adopt_plan([{'question': 'How many lines does this fixture have?',
                       'missing_fact': missing}])
        q = c.activate_next()
        record = rt.evidence_store.add('execute_analysis', 'LINE_COUNT 3', metadata={
            'tool_call_id': 'count_action', 'user_turn_id': 'count_turn',
            'produced_by_phase': 'analysis_step', 'input_sha256': rt.source.sha256,
        })
        return rt, model, c, q, record.evidence_id

    def assert_requirement(self, messages, q):
        matches = [m for m in messages
                   if f'The question was: {q.question}\n' in str(m.get('content', ''))]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['role'], 'user')
        self.assertIn(f'What was missing: {q.missing_fact}\n', matches[0]['content'])
        self.assertEqual(matches[0]['content'].count(q.missing_fact), 1)
        self.assertFalse(any(q.missing_fact in str(m.get('content', ''))
                             for m in messages if m.get('role') == 'system'))

    def test_original_requirement_reaches_finish_without_rewriting_or_history_mutation(self):
        missing = 'The raw source, not its preview.\r\nKeep "caffè" and its byte offsets.'
        rt, model, c, q, eid = self.setup_finish(missing)
        before = copy.deepcopy(c)
        history = copy.deepcopy(rt.messages)
        records = copy.deepcopy(rt.evidence_store.records)
        self.assertIn(q.missing_fact, rt._resolve_messages(c, q)[-1]['content'])
        view = rt._finish_messages(q, 'A bounded preview.', eid)
        self.assert_requirement(view, q)
        rt._control_dispatch(view, FINISH_TOOL_SCHEMA)
        self.assert_requirement(rt.backend.chat_calls[-1]['messages'], q)
        self.assertEqual(q.missing_fact, missing)
        self.assertEqual(c, before)
        self.assertEqual(rt.messages, history)
        self.assertEqual(rt.evidence_store.records, records)

    def test_requirement_belongs_to_question_and_does_not_leak_into_next_session(self):
        rt, model, c, q, eid = self.setup_finish('FIRST_REGISTERED_REQUIREMENT')
        second = replace(q, id='Q2', question='Second original question?',
                         missing_fact='SECOND_REGISTERED_REQUIREMENT')
        fresh = replace(q, missing_fact='NEW_SESSION_REQUIREMENT')
        history = copy.deepcopy(rt.messages)
        for current in (q, second, fresh):
            view = rt._finish_messages(current, 'Observation', eid)
            self.assert_requirement(view, current)
            for other in (q, second, fresh):
                if other != current:
                    self.assertNotIn(other.missing_fact, str(view))
        self.assertEqual(rt.messages, history)
        self.assertEqual(c.questions, {q.id: q})

    def test_missing_or_empty_plan_requirement_is_not_invented(self):
        for entry in ({'question': 'Missing requirement'},
                      {'question': 'Empty requirement', 'missing_fact': ''},
                      {'question': 'Null requirement', 'missing_fact': None}):
            with self.subTest(entry=entry):
                c = AnalysisController()
                before = copy.deepcopy(c)
                with self.assertRaises(ControlError):
                    c.adopt_plan([_question('Valid first entry'), entry])
                self.assertEqual(c, before)

    def test_schema_repair_preserves_requirement_once_and_cannot_change_it(self):
        rt, model, c, q, eid = self.setup_finish()
        model.decisions = [
            {'status': 'resolved', 'answer_summary': 'Partial answer',
             'child_question': {'question': 'Still missing?', 'missing_fact': 'Other',
                                'caused_by_evidence_id': eid}},
            {'status': 'still_open', 'answer_summary': 'The required count is not yet known.'},
        ]
        self.assertEqual(rt.finish_question(c, q, 'A bounded observation', eid), 2)
        for call in rt.backend.chat_calls:
            self.assert_requirement(call['messages'], q)
        self.assertIn('cannot also declare a child_question',
                      rt.backend.chat_calls[-1]['messages'][-1]['content'])
        self.assertEqual(c.questions, {q.id: q})
        self.assertEqual(c.states[q.id].status, 'open')
        self.assertEqual(c.repairs, 1)

    def test_protocol_repair_preserves_requirement_once(self):
        rt, model, c, q, eid = self.setup_finish()
        model.decisions = [{'status': 'still_open'}]
        generate = rt.backend.chat_stream
        seen = []
        def dispatch(messages, **kw):
            seen.append(copy.deepcopy(messages))
            if len(seen) == 1:
                raise LlamaServerToolCallParseError('retained malformed control')
            return generate(messages, **kw)
        with mock.patch.object(rt.backend, 'chat_stream', side_effect=dispatch):
            self.assertEqual(rt.finish_question(c, q, 'A bounded observation', eid), 2)
        for messages in seen:
            self.assert_requirement(messages, q)
        self.assertEqual(rt.control_repairs, 1)
        self.assertEqual(c.questions, {q.id: q})
        self.assertEqual(c.states[q.id].status, 'open')

    def test_compaction_and_adaptive_cap_preserve_requirement_and_canonical_history(self):
        rt = self._runtime(_Model(plan=[_question('Original question')]))
        run = self._run(rt, cover=False)
        history = copy.deepcopy(rt.messages)
        def tokens(messages):
            archived = any('archived' in call['function']['arguments']
                           for m in messages for call in m.get('tool_calls', []))
            return 2022 if archived else 2200
        rt.backend = ExactFinishBackend(tokens)
        rt.context_tokens = 4096
        c = AnalysisController()
        c.adopt_plan([{'question': 'Original question',
                       'missing_fact': 'An exact requirement preserved through admission.'}])
        q = c.activate_next()
        view = rt._finish_messages(q, 'Observation', run.last_step.evidence.evidence_id)
        rt._control_dispatch(view, FINISH_TOOL_SCHEMA)
        sent = rt.backend.chat_calls[-1]
        self.assert_requirement(sent['messages'], q)
        self.assertEqual(rt.last_context_plan.compacted_turns, 1)
        self.assertEqual(sent['kwargs']['max_tokens'], 1818)
        self.assertEqual(rt.last_context_plan.input_limit + 1818 + 256, 4096)
        self.assertEqual(rt.messages, history)
        self.assertEqual(c.questions[q.id], q)

    def test_requirement_is_not_proof_but_a_supported_answer_remains_acceptable(self):
        rt, model, c, q, eid = self.setup_finish()
        model.decisions = [{'status': 'resolved', 'answer_summary': 'The action printed LINE_COUNT 3.',
                            'evidence_ids': [eid]}]
        self.assertEqual(rt.finish_question(c, q, 'LINE_COUNT 3', eid), 1)
        self.assert_requirement(rt.backend.chat_calls[-1]['messages'], q)
        self.assertEqual(c.states[q.id].status, 'answered_unverified')
        self.assertEqual(c.states[q.id].evidence_ids, (eid,))
        self.assertEqual(c.counts()['resolved'], 0)
        self.assertEqual(c.questions[q.id], q)

    def test_evidence_reference_in_requirement_uses_existing_reattestation(self):
        rt, model, c, q, eid = self.setup_finish()
        q = replace(q, missing_fact=f'The exact count in evidence:{eid}.')
        view = rt._finish_messages(q, 'Observation', eid)
        rt._control_dispatch(view, FINISH_TOOL_SCHEMA)
        self.assert_requirement(rt.backend.chat_calls[-1]['messages'], q)
        self.assertTrue(any('LINE_COUNT 3' in str(m.get('content', ''))
                            for m in rt.backend.chat_calls[-1]['messages']))
        self.assertTrue(rt.evidence_store.discard(eid))
        dispatched = rt.model_calls
        with self.assertRaisesRegex(ContextAdmissionError, 'evidence-rehydration-unavailable'):
            rt._control_dispatch(view, FINISH_TOOL_SCHEMA)
        self.assertEqual(rt.model_calls, dispatched)
        self.assertEqual(q.missing_fact, f'The exact count in evidence:{eid}.')

    def test_truncated_supported_control_cannot_commit_a_decision(self):
        rt, model, c, q, eid = self.setup_finish()
        model.decisions = [{'status': 'resolved', 'answer_summary': 'LINE_COUNT 3',
                            'evidence_ids': [eid]}]
        generate = rt.backend.chat_stream
        with mock.patch.object(rt.backend, 'chat_stream',
                               side_effect=lambda *a, **kw: replace(generate(*a, **kw), finish_reason='length')):
            rt.finish_question(c, q, 'LINE_COUNT 3', eid)
        self.assert_requirement(rt.backend.chat_calls[-1]['messages'], q)
        self.assertEqual(c.questions, {q.id: q})
        self.assertEqual(c.states[q.id].summary, '')
        self.assertEqual(c.states[q.id].evidence_ids, ())
        self.assertEqual(rt.model_calls, 1)
