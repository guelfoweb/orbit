"""FINISH citations are data; only structurally owned requests may retrieve."""
import copy
from dataclasses import replace
from unittest import mock

from orbit.backend.llama_server import LlamaServerToolCallParseError
from orbit.runtime.analysis_controller import AnalysisController, ControlError
from orbit.runtime.analysis_runtime import FINISH_TOOL_SCHEMA, ContextAdmissionError
from tests.test_analysis_controller_runtime import _Case, _Model, _question


class FinishReferenceOwnershipTests(_Case):
    def setup_case(self):
        rt = self._runtime(_Model([], [{'status': 'still_open'}]))
        c = AnalysisController()
        c.adopt_plan([_question('Original question')])
        q = c.activate_next()
        return rt, c, q

    def record(self, rt, body='OWNED EXACT BODY', *, source=None):
        rt.analyst_turns += 1
        call = rt._with_canonical_call_ids([{
            'type': 'function', 'function': {'name': 'execute_analysis',
                                          'arguments': '{"code":"print(1)"}'}}])[0]
        record = rt.evidence_store.add('execute_analysis', body, metadata={
            'tool_call_id': call['id'], 'user_turn_id': f'turn_{rt.analyst_turns}',
            'produced_by_phase': 'analysis_action',
            'analysis_source_sha256': source or rt.source.sha256})
        rt.messages.extend([{'role': 'user', 'content': 'Perform the action'},
                            {'role': 'assistant', 'content': '', 'tool_calls': [call]}])
        rt._append_tool_result(call, body, record=record)
        return record

    def dispatch(self, rt, q, observation, record, **kwargs):
        view = rt._finish_messages(q, observation, record.evidence_id, **kwargs)
        rt._control_dispatch(view, FINISH_TOOL_SCHEMA)
        return view, rt.backend.chat_calls[-1]['messages']

    def test_diagnostic_mentions_do_not_request_bytes(self):
        for where in ('Traceback (most recent call last):', 'stdout:', 'stderr:',
                      'diagnostic:', 'model said:', 'excerpt:'):
            with self.subTest(where=where):
                rt, c, q = self.setup_case()
                unrelated = self.record(rt, 'UNRELATED RAW BODY' * 100)
                owned = self.record(rt)
                observation = f'{where}\nread_evidence("evidence:{unrelated.evidence_id}")'
                history = copy.deepcopy(rt.messages)
                records = copy.deepcopy(rt.evidence_store.records)
                with mock.patch.object(rt, '_bounded_rehydration_block',
                                       side_effect=AssertionError('incidental reference read')):
                    _, sent = self.dispatch(rt, q, observation, owned)
                self.assertIn(observation, sent[-1]['content'])
                self.assertEqual(rt.messages, history)
                self.assertEqual(rt.evidence_store.records, records)
                self.assertEqual(c.states[q.id].status, 'open')

    def test_unowned_and_invented_question_mentions_are_not_authority(self):
        rt, _, q = self.setup_case()
        unrelated = self.record(rt, 'OTHER QUESTION')
        owned = self.record(rt)
        for eid in (unrelated.evidence_id, 'ev_000000000000_0000000000000000'):
            with self.subTest(eid=eid):
                question = replace(q, missing_fact=f'Read evidence:{eid}')
                with mock.patch.object(rt, '_bounded_rehydration_block',
                                       side_effect=AssertionError('unowned reference read')):
                    self.dispatch(rt, question, 'Observation', owned)

    def test_owned_action_reference_rehydrates_exactly_once(self):
        rt, _, q = self.setup_case()
        owned = self.record(rt)
        q = replace(q, question=f'Read evidence:{owned.evidence_id}',
                    missing_fact=f'Use evidence:{owned.evidence_id}')
        _, sent = self.dispatch(rt, q, 'Observation', owned)
        self.assertEqual(rt.last_rehydration_diag['requested_ids'], [owned.evidence_id])
        self.assertEqual(sum('deterministic_evidence_rehydration:' in str(m.get('content'))
                             for m in sent), 1)
        self.assertIn('OWNED EXACT BODY', sent[-1]['content'])

    def test_owned_id_mentioned_only_by_observation_is_still_not_a_request(self):
        rt, _, q = self.setup_case()
        owned = self.record(rt)
        with mock.patch.object(rt, '_bounded_rehydration_block',
                               side_effect=AssertionError('citation requested bytes')):
            self.dispatch(rt, q, f'stdout: evidence:{owned.evidence_id}', owned)

    def test_owned_withdrawn_request_cannot_dispatch(self):
        rt, _, q = self.setup_case()
        owned = self.record(rt)
        q = replace(q, missing_fact=f'Read evidence:{owned.evidence_id}')
        view = rt._finish_messages(q, 'Observation', owned.evidence_id)
        rt.evidence_store.discard(owned.evidence_id)
        with self.assertRaises(ContextAdmissionError):
            rt._control_dispatch(view, FINISH_TOOL_SCHEMA)
        self.assertEqual(rt.model_calls, 0)

    def test_repair_error_cannot_introduce_a_request(self):
        rt, c, q = self.setup_case()
        unrelated = self.record(rt, 'UNRELATED')
        owned = self.record(rt)
        generate = rt.backend.chat_stream
        calls = []
        def reply(messages, **kwargs):
            calls.append(copy.deepcopy(messages))
            if len(calls) == 1:
                raise LlamaServerToolCallParseError(f'bad evidence:{unrelated.evidence_id}')
            return generate(messages, **kwargs)
        with mock.patch.object(rt.backend, 'chat_stream', side_effect=reply), \
             mock.patch.object(rt, '_bounded_rehydration_block',
                               side_effect=AssertionError('repair text read')):
            self.assertEqual(rt.finish_question(c, q, 'Observation', owned.evidence_id), 2)
        self.assertEqual(rt.control_repairs, 1)
        self.assertEqual(c.states[q.id].status, 'open')

    def test_other_session_and_snapshot_request_rejected(self):
        rt, _, q = self.setup_case()
        owned = self.record(rt)
        q = replace(q, missing_fact=f'Read evidence:{owned.evidence_id}')
        view = rt._finish_messages(q, 'Observation', owned.evidence_id)
        foreign, _, _ = self.setup_case()
        with self.assertRaises(ContextAdmissionError):
            foreign._control_dispatch(view, FINISH_TOOL_SCHEMA)
        rt.source = replace(rt.source, sha256='f' * 64)
        with self.assertRaises(ContextAdmissionError):
            rt._control_dispatch(view, FINISH_TOOL_SCHEMA)
        self.assertEqual(foreign.model_calls + rt.model_calls, 0)

    def test_user_turn_and_tool_provenance_mismatch_rejected(self):
        for field in ('user_turn_id', 'tool_call_id', 'name', 'content'):
            with self.subTest(field=field):
                rt, _, q = self.setup_case()
                owned = self.record(rt)
                q = replace(q, missing_fact=f'Read evidence:{owned.evidence_id}')
                view = rt._finish_messages(q, 'Observation', owned.evidence_id)
                rt.messages[-1][field] = 'wrong'
                with self.assertRaises(ContextAdmissionError):
                    rt._finish_rehydration_ids(view)
                with self.assertRaises(ContextAdmissionError):
                    rt._control_dispatch(view, FINISH_TOOL_SCHEMA)
                self.assertEqual(rt.model_calls, 0)

    def test_structural_child_reference_is_owned(self):
        rt, c, q = self.setup_case()
        parent = self.record(rt, 'PARENT EXACT BODY')
        child = c.accept_child('Follow-up', f'Read evidence:{parent.evidence_id}',
                               parent.evidence_id, {parent.evidence_id})
        current = self.record(rt)
        _, sent = self.dispatch(rt, child, 'Observation', current)
        self.assertIn('PARENT EXACT BODY', sent[-1]['content'])

    def test_non_finish_explicit_request_is_unchanged(self):
        rt, _, _ = self.setup_case()
        record = self.record(rt)
        messages = [*rt.messages, {'role': 'user', 'content': f'Read evidence:{record.evidence_id}'}]
        sent, ids = rt._with_evidence_rehydration(messages)
        self.assertEqual(ids, (record.evidence_id,))
        self.assertIn('OWNED EXACT BODY', sent[-1]['content'])

    def test_question_state_owned_reference_survives_both_repairs(self):
        for repair in ('protocol', 'schema'):
            with self.subTest(repair=repair):
                rt, c, q = self.setup_case()
                prior = self.record(rt, 'PRIOR OWNED BODY')
                current = self.record(rt)
                q = replace(q, missing_fact=f'Read evidence:{prior.evidence_id}')
                c.questions[q.id] = q
                c.states[q.id].evidence_ids = (prior.evidence_id,)
                generate = rt.backend.chat_stream
                sent = []
                def reply(messages, **kwargs):
                    sent.append(copy.deepcopy(messages))
                    if len(sent) == 1:
                        if repair == 'protocol':
                            raise LlamaServerToolCallParseError('bad control evidence:ev_invented')
                        rt.backend.model.decisions.insert(0, {'status': 'invalid'})
                    return generate(messages, **kwargs)
                with mock.patch.object(rt.backend, 'chat_stream', side_effect=reply):
                    self.assertEqual(rt.finish_question(c, q, 'Observation', current.evidence_id), 2)
                self.assertTrue(all('PRIOR OWNED BODY' in m[-1]['content'] for m in sent))
                self.assertEqual(c.states[q.id].status, 'open')
                self.assertEqual(rt.last_rehydration_diag['requested_ids'], [prior.evidence_id])

    def test_wrong_question_cannot_own_current_action(self):
        rt, c, q = self.setup_case()
        record = self.record(rt)
        for other in (replace(q, id='Q9'), replace(q, missing_fact='Changed question')):
            with self.subTest(other=other):
                with self.assertRaises(ControlError):
                    rt.finish_question(c, other, 'Observation', record.evidence_id)
                self.assertEqual(rt.model_calls, 0)
        c.active = None
        with self.assertRaises(ControlError):
            rt.finish_question(c, q, 'Observation', record.evidence_id)
        self.assertEqual(rt.model_calls, 0)

    def test_older_action_cannot_be_handed_off_as_current(self):
        rt, _, q = self.setup_case()
        old = self.record(rt)
        self.record(rt, 'OTHER QUESTION ACTION')
        q = replace(q, missing_fact=f'Read evidence:{old.evidence_id}')
        with self.assertRaises(ContextAdmissionError):
            self.dispatch(rt, q, 'Observation', old)
        self.assertEqual(rt.model_calls, 0)
        # Distinguish turn equality from the latest-result identity check.
        rt.analyst_turns = 1
        with self.assertRaises(ContextAdmissionError):
            self.dispatch(rt, q, 'Observation', old)

    def test_current_result_with_incompatible_user_turn_is_rejected(self):
        rt, _, q = self.setup_case()
        current = self.record(rt)
        rt.analyst_turns += 1
        q = replace(q, missing_fact=f'Read evidence:{current.evidence_id}')
        with self.assertRaises(ContextAdmissionError):
            self.dispatch(rt, q, 'Observation', current)

    def test_wrong_record_snapshot_and_changed_snapshot_bytes_rejected(self):
        for changed_file in (False, True):
            with self.subTest(changed_file=changed_file):
                rt, _, q = self.setup_case()
                owned = self.record(rt, source=None if changed_file else 'e' * 64)
                q = replace(q, missing_fact=f'Read evidence:{owned.evidence_id}')
                if changed_file:
                    rt.source.snapshot_path.write_bytes(b'changed')
                with self.assertRaises(ContextAdmissionError):
                    self.dispatch(rt, q, 'Observation', owned)
                self.assertEqual(rt.model_calls, 0)

    def test_rewind_and_new_turn_invalidate_request(self):
        for change in ('rewind', 'turn', 'store'):
            with self.subTest(change=change):
                rt, _, q = self.setup_case()
                record = self.record(rt)
                q = replace(q, missing_fact=f'Read evidence:{record.evidence_id}')
                view = rt._finish_messages(q, 'Observation', record.evidence_id)
                if change == 'rewind':
                    del rt.messages[-3:]
                elif change == 'turn':
                    rt.analyst_turns += 1
                else:
                    from orbit.runtime.evidence import EvidenceStore
                    rt.evidence_store = EvidenceStore(rt.evidence_store.root)
                    rt.evidence_store.load_index()
                with self.assertRaises(ContextAdmissionError):
                    rt._control_dispatch(view, FINISH_TOOL_SCHEMA)
                self.assertEqual(rt.model_calls, 0)

    def test_repair_reattests_owned_bytes_instead_of_trusting_cached_selection(self):
        rt, c, q = self.setup_case()
        record = self.record(rt)
        q = replace(q, missing_fact=f'Read evidence:{record.evidence_id}')
        c.questions[q.id] = q
        def fail(messages, **kwargs):
            rt.evidence_store.discard(record.evidence_id)
            raise LlamaServerToolCallParseError('invalid control')
        with mock.patch.object(rt.backend, 'chat_stream', side_effect=fail):
            with self.assertRaises(ContextAdmissionError):
                rt.finish_question(c, q, 'Observation', record.evidence_id)
        self.assertEqual(rt.model_calls, 1)
        self.assertEqual(c.states[q.id].status, 'open')

    def test_modified_or_unavailable_sidecar_cannot_rehydrate(self):
        for change in ('alter', 'delete'):
            with self.subTest(change=change):
                rt, _, q = self.setup_case()
                record = self.record(rt)
                q = replace(q, missing_fact=f'Read evidence:{record.evidence_id}')
                path = rt.evidence_store.root / f'{record.evidence_id}.txt'
                if change == 'alter':
                    path.write_text('ALTERED')
                else:
                    path.unlink()
                with self.assertRaises(ContextAdmissionError):
                    self.dispatch(rt, q, 'Observation', record)
                self.assertEqual(rt.model_calls, 0)

    def test_ambiguous_or_missing_request_does_not_use_text(self):
        rt, _, q = self.setup_case()
        record = self.record(rt)
        q = replace(q, missing_fact=f'Read evidence:{record.evidence_id}')
        view = rt._finish_messages(q, 'Observation', record.evidence_id)
        with self.assertRaises(ContextAdmissionError):
            rt._control_dispatch([*view, copy.deepcopy(view[-1])], FINISH_TOOL_SCHEMA)
        malformed = copy.deepcopy(view)
        malformed[-1]['analysis_finish_request'] = None
        with self.assertRaises(ContextAdmissionError):
            rt._control_dispatch(malformed, FINISH_TOOL_SCHEMA)
        del view[-1]['analysis_finish_request']
        with mock.patch.object(rt, '_bounded_rehydration_block',
                               side_effect=AssertionError('unowned request')):
            rt._control_dispatch(view, FINISH_TOOL_SCHEMA)

    def test_directly_associated_registered_transform_remains_available(self):
        from tests.test_analysis_evidence_delivery import EvidenceDeliveryTests
        case = EvidenceDeliveryTests()
        self.addCleanup(case.doCleanups)
        rt, backend = case.runtime()
        stage, record = rt.transform_stages[0]
        q = _question('Follow-up about decoded evidence')
        c = AnalysisController(); c.adopt_plan([q]); parent = c.activate_next()
        child = c.accept_child('Original child question', f'Read evidence:{record.evidence_id}',
                               record.evidence_id, {record.evidence_id})
        view = rt._finish_messages(child, 'Observation', '')
        admitted, _ = rt._admit_finish(view, FINISH_TOOL_SCHEMA)
        self.assertEqual(rt.last_rehydration_diag['requested_ids'], [record.evidence_id])
        self.assertIn(stage.output, admitted[-1]['content'])

    def test_owned_office_module_uses_existing_bounded_delivery(self):
        from pathlib import Path
        from tests.test_analysis_ole import PreflightIntegrationTests, SAMPLE
        from tests.test_analysis_controller_runtime import _Backend
        case = PreflightIntegrationTests(); self.addCleanup(case.doCleanups)
        rt = case._runtime(Path(SAMPLE).read_bytes())
        rt.backend = _Backend(_Model([]))
        module, record = rt.office_modules[0]
        c = AnalysisController(); c.adopt_plan([_question('Office question')]); c.activate_next()
        child = c.accept_child('Original macro question', f'Read evidence:{record.evidence_id}',
                               record.evidence_id, {record.evidence_id})
        view = rt._finish_messages(child, 'Observation', '')
        admitted, _ = rt._admit_finish(view, FINISH_TOOL_SCHEMA)
        self.assertEqual(rt.last_rehydration_diag['requested_ids'], [record.evidence_id])
        self.assertIn(module.source[:100], admitted[-1]['content'])
        self.assertTrue(rt.last_rehydration_diag['windowed'])
        for change in ('registry', 'revoked'):
            with self.subTest(change=change):
                if change == 'registry':
                    rt.office_modules = [(replace(module, source='different'), record)]
                else:
                    rt.office_modules = [(module, record)]
                    rt.evidence_store.discard(record.evidence_id)
                with self.assertRaises(ContextAdmissionError):
                    rt._admit_finish(view, FINISH_TOOL_SCHEMA)

    def test_owned_raw_sibling_requires_its_existing_action_lineage(self):
        from orbit.runtime.analysis_sandbox import AnalysisResult
        for change in (None, 'call', 'turn', 'code', 'source', 'revoked', 'rewind'):
            with self.subTest(change=change):
                rt, c, q = self.setup_case()
                rt.analyst_turns = 1
                call = rt._with_canonical_call_ids([{'type': 'function', 'function': {
                    'name': 'execute_analysis', 'arguments': '{"code":"print(1)"}'}}])[0]
                result = AnalysisResult(status='ok', code_sha256='b' * 64,
                    input_sha256=rt.source.sha256, stdout='EXACT RAW OUTPUT',
                    stderr='', exit_status=0, duration_seconds=.01)
                bounded, raw = rt._record_action_evidence(call, result, 'Bounded observation')
                if change in ('call', 'turn', 'code', 'source'):
                    field = {'call': 'tool_call_id', 'turn': 'user_turn_id',
                             'code': 'code_sha256', 'source': 'analysis_source_sha256'}[change]
                    raw = rt.evidence_store.add('execute_analysis_raw', 'EXACT RAW OUTPUT',
                        metadata={**raw.metadata, field: 'different'})
                    bounded = rt.evidence_store.add('execute_analysis', 'Bounded observation',
                        metadata={**bounded.metadata, 'raw_output_evidence_id': raw.evidence_id})
                rt.messages.extend([{'role': 'user', 'content': 'Action'},
                    {'role': 'assistant', 'content': '', 'tool_calls': [call]}])
                rt._append_tool_result(call, 'Bounded observation', record=bounded)
                child = c.accept_child('Raw question', f'Read evidence:{raw.evidence_id}',
                                       raw.evidence_id, {raw.evidence_id})
                current = self.record(rt)
                view = rt._finish_messages(child, 'Observation', current.evidence_id)
                if change == 'revoked':
                    rt.evidence_store.discard(raw.evidence_id)
                elif change == 'rewind':
                    del rt.messages[-6:-3]
                if change is None:
                    admitted, _ = rt._admit_finish(view, FINISH_TOOL_SCHEMA)
                    self.assertIn('EXACT RAW OUTPUT', admitted[-1]['content'])
                    self.assertEqual(rt.last_rehydration_diag['requested_ids'], [raw.evidence_id])
                else:
                    with self.assertRaises(ContextAdmissionError):
                        rt._admit_finish(view, FINISH_TOOL_SCHEMA)
