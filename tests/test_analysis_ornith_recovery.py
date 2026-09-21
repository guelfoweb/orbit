"""Regressions from the retained Ornith source-acquisition/reread trajectory."""
import copy
from dataclasses import replace
from unittest import mock

from orbit.runtime import analysis_runtime as ar
from orbit.runtime.analysis_controller import Question
from orbit.runtime.analysis_tools_shim import ORBIT_TOOLS_SOURCE
from tests.test_analysis_source_delivery import _Case, SOURCE, NUMBERED
from tests.test_analysis_reacquisition_runtime import _result


class RecoveryTests(_Case):
    def _step(self, runtime, result, code=None):
        return super()._step(runtime, replace(result, input_sha256=runtime.source.sha256), code)

    def test_archive_named_by_suppression_is_readable_by_next_action(self):
        rt = self._runtime()
        first = self._step(rt, _result(stdout='FULL SOURCE\n' + SOURCE + '\nEND'))
        raw_id = first.raw_output_evidence_id
        self._step(rt, _result(stdout=NUMBERED), code='print("numbered source")')
        seen = {}
        self.backend.code = f'import orbit_tools\nprint(orbit_tools.read_evidence("evidence:{raw_id}"))'
        def sandbox(**kw):
            seen.update(kw)
            helper = {}
            exec(ORBIT_TOOLS_SOURCE, helper)
            helper['_EVIDENCE_INPUTS'] = kw.get('evidence_inputs', {})
            value = helper['read_evidence']('evidence:' + raw_id)
            self.assertEqual(value, rt.evidence_store.reattest_exact(raw_id))
            return _result(stdout='A selected finding, not another source copy.')
        with mock.patch.object(ar, 'execute_analysis', sandbox):
            step = rt.step('Inspect the existing archive.')
        self.assertTrue(step.action_executed)
        self.assertIn(raw_id, seen['evidence_inputs'])
        self.assertEqual(rt.actions_executed, 2)
        self.assertFalse(rt.source_covered)

    def test_suppressed_execution_is_not_executed_again_on_identical_state(self):
        rt = self._runtime()
        self._step(rt, _result(stdout=SOURCE), code='print("first source acquisition")')
        second = self._step(rt, _result(stdout=NUMBERED), code='print("numbered source")')
        third = self._step(rt, _result(stdout=NUMBERED), code='print("numbered source")')
        self.assertIsNotNone(second.suppressed_duplicate_of)
        self.assertEqual(self.dispatched, 2, 'a suppressed execution still established its fingerprint')
        self.assertFalse(third.action_executed)
        self.assertEqual(rt.actions_executed, 1)

    def test_finish_traceback_reference_does_not_request_archive_rehydration(self):
        rt = self._runtime()
        first = self._step(rt, _result(stdout=SOURCE))
        q = Question('Q1', 'Which lines delete files?', 'The exact deletion operations.')
        obs = f'File "main.py", read_evidence("evidence:{first.raw_output_evidence_id}")\nValueError: unavailable'
        messages = rt._finish_messages(q, obs, first.evidence.evidence_id)
        with mock.patch.object(rt, '_bounded_rehydration_block', return_value=('UNREQUESTED', {})) as hydrate:
            sent, ids = rt._with_evidence_rehydration(messages)
        hydrate.assert_not_called()
        self.assertEqual(ids, ())
        self.assertEqual(sent, messages)
        self.assertIn(obs, sent[-1]['content'])

    def test_new_authorized_input_changes_fingerprint_then_stabilizes(self):
        rt = self._runtime()
        code = 'print("source")'
        self._step(rt, _result(stdout=SOURCE), code)
        self._step(rt, _result(stdout=SOURCE), code)
        self._step(rt, _result(stdout=SOURCE), code)
        self.assertEqual(self.dispatched, 2)
        raw_id = rt.source_delivery.raw_evidence_id
        note = rt.messages[-1]['content']
        self.assertIn("read_evidence('" + raw_id + "')", note)
        self.assertNotIn('Reuse it: name', note)

    def test_source_archive_authorization_is_removed_by_any_broken_binding(self):
        for fault in ('parent_withdrawn', 'raw_withdrawn', 'raw_altered', 'parent_altered',
                      'rewind', 'snapshot', 'raw_call', 'raw_turn', 'raw_code',
                      'raw_source', 'parent_source', 'wrong_raw', 'other_session'):
            with self.subTest(fault=fault):
                rt = self._runtime()
                first = self._step(rt, _result(stdout=SOURCE))
                parent = first.evidence; raw_id = first.raw_output_evidence_id
                raw = rt.evidence_store.records[raw_id]
                if fault == 'parent_withdrawn': rt.evidence_store.discard(parent.evidence_id)
                elif fault == 'raw_withdrawn': rt.evidence_store.discard(raw_id)
                elif fault == 'raw_altered': (rt.evidence_store.root / (raw_id+'.txt')).write_text('altered')
                elif fault == 'parent_altered': (rt.evidence_store.root / (parent.evidence_id+'.txt')).write_text('altered')
                elif fault == 'rewind': rt.messages = rt.messages[:2]
                elif fault == 'snapshot': rt.source.snapshot_path.write_bytes(b'different snapshot')
                elif fault == 'raw_call': rt.evidence_store.records[raw_id] = replace(raw, tool_call_id='other')
                elif fault == 'raw_turn': rt.evidence_store.records[raw_id] = replace(raw, user_turn_id='other')
                elif fault in ('raw_code', 'raw_source'):
                    key = 'code_sha256' if fault == 'raw_code' else 'analysis_source_sha256'
                    rt.evidence_store.records[raw_id] = replace(raw, metadata={**raw.metadata,key:'other'})
                elif fault == 'parent_source':
                    rt.evidence_store.records[parent.evidence_id] = replace(parent, metadata={**parent.metadata,'analysis_source_sha256':'other'})
                elif fault == 'wrong_raw': rt.messages[-1]['source_delivered']['raw_evidence_id']='ev_unrelated'
                elif fault == 'other_session':
                    other = self._runtime();other.evidence_store = rt.evidence_store;rt = other
                self.backend = rt.backend
                self.backend.code = 'print("different authorized action")'
                with mock.patch.object(ar, 'execute_analysis', return_value=_result(stdout='unrelated')) as execute:
                    rt.step('Continue')
                self.assertEqual(execute.call_count, 1)
                self.assertNotIn(raw_id, execute.call_args.kwargs.get('evidence_inputs', {}))

    def test_helper_never_interprets_paths_or_unknown_references(self):
        helper={};exec(ORBIT_TOOLS_SOURCE,helper);helper['_EVIDENCE_INPUTS']={'ev_authorized':'caffè\r\n終\x00'}
        for value in ('ev_authorized','evidence:ev_authorized'):
            self.assertEqual(helper['read_evidence'](value),'caffè\r\n終\x00')
        for value in ('/etc/passwd','/workspace/evidence/ev_authorized','evidence:../secret',
                      'evidence:evidence:ev_authorized','ev_unknown',None):
            with self.assertRaises(ValueError):helper['read_evidence'](value)

    def test_withdrawn_suppressed_record_is_not_reused_as_current_evidence(self):
        rt=self._runtime();self._step(rt,_result(stdout=SOURCE),'print("first")')
        second=self._step(rt,_result(stdout=NUMBERED),'print("listing")')
        rt.evidence_store.discard(second.evidence.evidence_id)
        third=self._step(rt,_result(stdout=NUMBERED),'print("listing")')
        self.assertEqual(self.dispatched,3)
        self.assertIsNotNone(third.raw_output_evidence_id)
        self.assertNotEqual(third.evidence.evidence_id, second.evidence.evidence_id)

    def test_large_raw_output_is_not_silently_truncated_into_helper_input(self):
        rt = self._runtime(data=(b'x\r\n' * 24000))
        self.backend.per_char = .001
        first = self._step(rt, _result(stdout=rt.source.snapshot_path.read_text()))
        # read_text normalizes CRLF, so it is not a complete source proof.
        self.assertIsNone(rt.source_delivery)
        exact=rt.source.snapshot_path.read_bytes().decode()
        first=self._step(rt,_result(stdout=exact))
        self.assertIsNotNone(rt.source_delivery)
        self.backend.code='print("next")'
        with mock.patch.object(ar,'execute_analysis',return_value=_result(stdout='bounded finding')) as execute:
            rt.step('Continue')
        self.assertNotIn(first.raw_output_evidence_id,execute.call_args.kwargs.get('evidence_inputs',{}))

    def test_skipped_large_acquisition_does_not_offer_unavailable_helper_input(self):
        rt = self._runtime(data=b'x\r\n' * 24000)
        self.backend.per_char = .001
        source = rt.source.snapshot_path.read_bytes().decode()
        self._step(rt, _result(stdout=source), 'print("first acquisition")')
        self._step(rt, _result(stdout=source), 'print("another acquisition")')
        skipped = self._step(rt, _result(stdout=source), 'print("another acquisition")')
        self.assertFalse(skipped.action_executed)
        self.assertEqual(self.dispatched, 2)
        note = rt.messages[-1]['content']
        self.assertNotIn('read_evidence(', note)
        self.assertIn('not available through read_evidence', note)


from orbit.backend.llama_server import LlamaServerToolCallParseError
from orbit.runtime.analysis_controller import AnalysisController
from tests.test_analysis_controller_runtime import _Case as ControllerCase, _Model


class FinishObservationOwnershipTests(ControllerCase):
    def test_actual_finish_and_protocol_repair_do_not_retrieve_ids_from_errors(self):
        model=_Model(plan=[])
        rt=self._runtime(model)
        record=rt.evidence_store.add('execute_analysis_raw','UNREQUESTED LARGE ARCHIVE '*10000,
            metadata={'tool_call_id':'previous','user_turn_id':'previous','produced_by_phase':'analysis_action_raw'})
        c=AnalysisController();c.adopt_plan([{'question':'How many lines?', 'missing_fact':'The exact count.'}]);q=c.activate_next()
        model.decisions=[{'status':'still_open','answer_summary':'The action failed before determining the count.'}]
        original=rt.backend.chat_stream;calls=[]
        def dispatch(messages,**kw):
            calls.append(copy.deepcopy(messages))
            if len(calls)==1: raise LlamaServerToolCallParseError('invalid output mentions evidence:'+record.evidence_id)
            return original(messages,**kw)
        history=copy.deepcopy(rt.messages)
        with mock.patch.object(rt.backend,'chat_stream',side_effect=dispatch), mock.patch.object(rt,'_bounded_rehydration_block',side_effect=AssertionError('data is not a request')):
            self.assertEqual(rt.finish_question(c,q,'Traceback: read_evidence("evidence:'+record.evidence_id+'") failed.',record.evidence_id),2)
        self.assertEqual(c.states[q.id].status,'open')
        self.assertEqual(rt.messages,history)
        self.assertEqual(rt.control_repairs,1)
        self.assertEqual(rt.model_calls,2)
        self.assertTrue(all('UNREQUESTED LARGE ARCHIVE' not in str(ms) for ms in calls))
        self.assertEqual(c.questions[q.id],q)
