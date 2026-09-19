"""FINISH advertises its completion contract and repairs without partial commits."""
import copy
from dataclasses import replace
from unittest import mock

from orbit.runtime.analysis_controller import AnalysisController, ControlError, parse_finish_call
from orbit.runtime.analysis_runtime import FINISH_TOOL_SCHEMA
from tests.test_analysis_controller_runtime import _Case, _Model, _question


class FinishInterfaceTests(_Case):
    def setup_question(self):
        model = _Model(plan=[])
        rt = self._runtime(model)
        record = rt.evidence_store.add('execute_analysis', 'A bounded fixture observation.', metadata={
            'tool_call_id': 'fixture_action_1', 'user_turn_id': 'fixture_turn_1',
            'produced_by_phase': 'analysis_step', 'input_sha256': rt.source.sha256,
        })
        self.assertEqual(rt.evidence_store.reattest_exact(record.evidence_id), 'A bounded fixture observation.')
        c = AnalysisController()
        c.adopt_plan([_question('What is the full content and structure of the artifact?')])
        q = c.activate_next()
        child = {'question': 'What do the encoded values mean?',
                 'missing_fact': 'The values have not been established.',
                 'caused_by_evidence_id': record.evidence_id}
        return rt, model, c, q, record.evidence_id, child

    def test_delivered_schema_exposes_the_existing_cross_field_contract(self):
        rt, model, c, q, eid, child = self.setup_question()
        model.decisions = [{'status': 'still_open'}]
        rt.finish_question(c, q, 'Observation', eid)
        schema = rt.backend.chat_calls[-1]['tools'][0]
        self.assertEqual(schema, FINISH_TOOL_SCHEMA)
        description = schema['function']['description']
        self.assertEqual(description, (
            'Report what the action just run established about the question you '
            'were working on. Answer `still_open` if it did not settle the question '
            'and `blocked` if it cannot be settled -- both are real answers and '
            'the report will say so. '
            'Use `resolved` only with a non-empty answer_summary and no child_question.'
        ))

    def test_parser_combinations_preserve_the_proposal(self):
        _, _, _, _, eid, child = self.setup_question()
        for status, summary, dependency, valid in [
            ('resolved', 'A proposed answer.', None, True),
            ('resolved', '', None, False),
            ('resolved', 'A proposed answer.', child, False),
            ('still_open', '', child, True),
            ('blocked', '', None, True),
            ('blocked', '', child, True),
        ]:
            with self.subTest(status=status, summary=summary, child=dependency):
                args = {'status': status, 'answer_summary': summary, 'evidence_ids': [eid]}
                if dependency is not None:
                    args['child_question'] = copy.deepcopy(dependency)
                original = copy.deepcopy(args)
                if valid:
                    parsed = parse_finish_call(args)
                    self.assertEqual(parsed['answer_summary'], summary)
                    self.assertEqual(parsed['child_question'], dependency)
                else:
                    with self.assertRaises(ControlError):
                        parse_finish_call(args)
                self.assertEqual(args, original)

    def test_repair_delivers_exact_rejection_before_any_controller_commit(self):
        for repaired_status in ('still_open', 'resolved', 'blocked'):
            with self.subTest(status=repaired_status):
                rt, model, c, q, eid, child = self.setup_question()
                invalid = {'status': 'resolved', 'answer_summary': 'Partial answer.',
                           'evidence_ids': [eid], 'child_question': child}
                valid = {'status': repaired_status, 'answer_summary': 'Model proposal.', 'evidence_ids': [eid]}
                if repaired_status == 'still_open':
                    valid['child_question'] = child
                model.decisions = [invalid, valid]
                before = copy.deepcopy(c)
                history = copy.deepcopy(rt.messages)
                records = copy.deepcopy(rt.evidence_store.records)
                with self.assertRaises(ControlError) as error:
                    parse_finish_call(invalid)
                generate = rt.backend.chat_stream
                def probe(messages, **kw):
                    self.assertEqual(kw['tools'], [FINISH_TOOL_SCHEMA])
                    if rt.model_calls == 2:
                        # Only the bounded repair counter may change on refusal.
                        expected = copy.deepcopy(before)
                        expected.repairs = 1
                        self.assertEqual(c, expected)
                        self.assertIn(str(error.exception), messages[-1]['content'])
                        self.assertIn(q.question, str(messages))
                    return generate(messages, **kw)
                with mock.patch.object(rt.backend, 'chat_stream', side_effect=probe):
                    calls = rt.finish_question(c, q, 'Observation', eid, max_calls=2)
                self.assertEqual(calls, 2)
                self.assertEqual(c.repairs, 1)
                self.assertEqual(rt.messages, history)
                self.assertEqual(rt.evidence_store.records, records)
                self.assertEqual(c.questions[q.id], q)
                self.assertEqual(c.states[q.id].status, {'resolved': 'answered_unverified', 'still_open': 'open', 'blocked': 'blocked'}[repaired_status])
                self.assertEqual(c.counts()['resolved'], 0)
                self.assertEqual(len(c.questions), 2 if repaired_status == 'still_open' else 1)
                if repaired_status == 'still_open':
                    self.assertEqual(c.questions['Q1.1'].caused_by, eid)

    def test_repeated_contradiction_does_not_commit_answer_or_child(self):
        rt, model, c, q, eid, child = self.setup_question()
        model.decisions = [{'status': 'resolved', 'answer_summary': 'Unsupported answer.',
                            'evidence_ids': [eid], 'child_question': child}] * 2
        questions = copy.deepcopy(c.questions)
        calls = rt.finish_question(c, q, 'Observation', eid, max_calls=2)
        self.assertEqual(calls, 2)
        self.assertEqual(c.repairs, 1)
        self.assertEqual(c.questions, questions)
        self.assertEqual(c.states[q.id].status, 'blocked')
        self.assertEqual(c.states[q.id].summary, '')
        self.assertEqual(c.states[q.id].evidence_ids, ())
        self.assertEqual(c.rejected_children, 0)

    def test_interrupted_valid_dependency_never_commits_a_child(self):
        for reason in ('length', 'cancelled', 'error'):
            with self.subTest(reason=reason):
                rt, model, c, q, eid, child = self.setup_question()
                model.decisions = [{'status': 'still_open', 'child_question': child}]
                generate = rt.backend.chat_stream
                from orbit.backend.base import RecoverableBackendError
                with mock.patch.object(rt.backend, 'chat_stream', side_effect=lambda *a, **kw: replace(generate(*a, **kw), finish_reason=reason)):
                    if reason == 'cancelled':
                        with self.assertRaises(KeyboardInterrupt):
                            rt.finish_question(c, q, 'Observation', eid, max_calls=2)
                    elif reason == 'error':
                        with self.assertRaises(RecoverableBackendError):
                            rt.finish_question(c, q, 'Observation', eid, max_calls=2)
                    else:
                        rt.finish_question(c, q, 'Observation', eid, max_calls=2)
                self.assertEqual(len(c.questions), 1)
                self.assertEqual(c.states[q.id].summary, '')
                self.assertEqual(rt.model_calls, 1)
