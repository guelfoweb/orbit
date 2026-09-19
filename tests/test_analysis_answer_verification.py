"""A model answer can stop work without certifying its narrative."""
from __future__ import annotations

import unittest
from dataclasses import asdict
from unittest import mock

from orbit.runtime.analysis_controller import AnalysisController, QuestionState, RESOLVED
from orbit.runtime.analysis_runtime import AutonomousRunResult, AnalysisRuntime, STOP_LEDGER_EXHAUSTED
from orbit.terminal.analysis_mode import format_progress_event
from orbit.runtime.analysis_runtime import AnalysisProgressEvent
from orbit.runtime.analysis_sandbox import AnalysisResult
from tests.test_analysis_controller_runtime import _Case, _Model, _question


class AnswerVerificationTests(unittest.TestCase):
    def controller(self):
        c = AnalysisController()
        c.adopt_plan([{
            "question": "What does the artifact's source code contain, and are the five decoded payloads consistent with it?",
            "missing_fact": "Only bounded source observations are available.",
        }])
        c.activate_next()
        return c

    def test_valid_reference_does_not_verify_whole_source_negative(self):
        c = self.controller()
        original = c.questions['Q1']
        claim = "The raw source contains no plaintext URLs or PowerShell text."
        c.close_active(RESOLVED, evidence_ids=('ev_excerpt',), summary=claim)
        self.assertEqual(c.states['Q1'].status, 'answered_unverified')
        self.assertEqual(c.questions['Q1'], original)
        self.assertEqual(c.states['Q1'].summary, claim)
        self.assertEqual(c.states['Q1'].evidence_ids, ('ev_excerpt',))
        self.assertIsNone(c.activate_next())
        self.assertTrue(c.exhausted)
        self.assertEqual(c.counts()['resolved'], 0)
        self.assertEqual(c.counts()['answered_unverified'], 1)
        self.assertIn('[ANSWERED_UNVERIFIED]', c.dossier())
        self.assertIn('unverified model summary', c.dossier())

    def test_legacy_question_state_is_explicitly_unverified(self):
        state = QuestionState(status='resolved', evidence_ids=('ev_old',), summary='37 bytes')
        self.assertEqual(state.status, 'answered_unverified')
        self.assertEqual(state.legacy_status, 'resolved')
        self.assertEqual(asdict(state)['summary'], '37 bytes')
        c = self.controller()
        c.states['Q1'] = state
        self.assertIn('legacy status: resolved', c.dossier())
        self.assertEqual(c.counts()['resolved'], 0)

    def test_legacy_result_never_exports_resolved_as_verified(self):
        run = AutonomousRunResult(steps=(), progress=(), stop_reason='done', model_calls=0,
                                  actions_executed=0, resolved_questions=('Q1',))
        data = asdict(run)
        self.assertEqual(data['resolved_questions'], ())
        self.assertEqual(data['legacy_resolved_questions'], ('Q1',))
        self.assertEqual(data['answered_unverified_questions'], ('Q1',))
        self.assertIn('Q1', data['unverified_questions'])

    def test_all_answered_report_still_receives_dossier(self):
        c = self.controller()
        c.close_active(RESOLVED, evidence_ids=('ev_excerpt',), summary='37 bytes')
        request = AnalysisRuntime._final_question(STOP_LEDGER_EXHAUSTED, (), c.dossier())
        self.assertIn(c.questions['Q1'].question, request)
        self.assertIn('37 bytes', request)
        self.assertIn('unverified', request)

    def test_legacy_progress_is_not_rendered_as_proof(self):
        text = format_progress_event(AnalysisProgressEvent(phase='analysis_finish', event='resolved'))
        self.assertIn('unverified', text.lower())

    def test_open_and_blocked_retain_existing_limits(self):
        for status in ('open', 'blocked'):
            with self.subTest(status=status):
                c = self.controller()
                c.close_active(status, summary='not established')
                self.assertEqual(c.states['Q1'].status, status)
                self.assertEqual(c.counts()['actions'], 0)
                self.assertEqual(c.repairs, 0)
                self.assertEqual(c.open_ids, ['Q1'] if status == 'open' else [])

    def test_hollow_answer_remains_open(self):
        for refs, summary in [((), 'answer'), (('ev',), '')]:
            c = self.controller()
            c.close_active(RESOLVED, evidence_ids=refs, summary=summary)
            self.assertEqual(c.states['Q1'].status, 'open')


class RuntimeAnswerVerificationTests(_Case):
    def test_real_run_exports_answer_without_verified_resolution(self):
        claim = 'The raw source contains no plaintext URLs or PowerShell text.'
        model = _Model(plan=[_question('Describe the entire source, including all commands.')],
                       decisions=[{'status': 'resolved', 'answer_summary': claim}])
        rt = self._runtime(model)
        events = []
        run = self._run(rt, cover=False, on_event=events.append)
        self.assertEqual(run.answered_unverified_questions, ('Q1',))
        self.assertEqual(run.unverified_questions, ('Q1',))
        self.assertEqual(run.resolved_questions, ())
        self.assertEqual(run.legacy_resolved_questions, ())
        self.assertEqual(run.open_questions, ())
        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(run.model_calls, 3)
        self.assertEqual(run.repairs, 0)
        self.assertIn('answered_unverified', [e.event for e in events])
        self.assertNotIn('resolved', [e.event for e in events])
        eid = run.last_step.evidence.evidence_id
        self.assertIsNotNone(rt.evidence_store.reattest_exact(eid))

    def test_missing_or_withdrawn_evidence_cannot_support_even_an_answer(self):
        for withdrawn in (False, True):
            with self.subTest(withdrawn=withdrawn):
                rt = self._runtime(_Model(plan=[_question('What is known?')]))
                run = self._run(rt, cover=False)
                eid = run.last_step.evidence.evidence_id
                if withdrawn:
                    self.assertTrue(rt.evidence_store.discard(eid))
                else:
                    (rt.evidence_store.root / f'{eid}.txt').write_text('altered')
                c = AnalysisController()
                c.adopt_plan([_question('What is known?')]); c.activate_next()
                rt._apply_decision(c, {'status': 'resolved', 'evidence_ids': [eid],
                                      'answer_summary': 'an answer'}, eid)
                self.assertEqual(c.states['Q1'].status, 'open')
                self.assertEqual(c.states['Q1'].evidence_ids, ())

    def test_ornith_37_claim_does_not_change_verified_snapshot_bytes(self):
        data = bytes.fromhex('616c70686120626574610d0a4575726f20e282ac3b2063616666c3a80d0a6c617374206279746521')
        text = data.decode('utf-8')
        self.assertEqual(len(data), 40)
        self.assertEqual(len(text), 37)
        model = _Model(plan=[_question('Read the entire file and determine its contents.')],
                       decisions=[{'status': 'resolved', 'answer_summary':
                                   'the file is 37 bytes, not 40 as stated in the task.'}])
        rt = self._runtime(model, data=data)
        output = f'len: 37\nrepr: {text!r}\nhex: {data.hex()}\n'
        result = AnalysisResult(
            status='ok', code_sha256='c' * 64, input_sha256=rt.source.sha256,
            stdout=output, stderr='', exit_status=0, duration_seconds=0.1,
        )
        with mock.patch('orbit.runtime.analysis_runtime.execute_analysis', return_value=result):
            run = rt.run_autonomous('Read this artifact.', cover=False, finalize=False)
        self.assertEqual(rt.source.size_bytes, 40)
        self.assertEqual(rt.source.snapshot_path.read_bytes(), data)
        self.assertEqual(run.answered_unverified_questions, ('Q1',))
        self.assertEqual(run.resolved_questions, ())
        self.assertEqual(rt.source_delivery.representation, 'repr_contained')
        record = run.last_step.evidence
        self.assertEqual(record.metadata['analysis_source_bytes'], 40)
        self.assertIsNotNone(rt.evidence_store.reattest_exact(record.evidence_id))


if __name__ == '__main__':
    unittest.main()
