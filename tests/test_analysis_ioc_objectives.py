"""Production-seam tests: PLAN and FINISH cannot hide a runtime IoC objective."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from orbit.runtime.analysis_runtime import AnalysisRuntime, acquire_analysis_source, STOP_LEDGER_EXHAUSTED
from orbit.runtime.analysis_controller import AnalysisController, OPEN, BLOCKED
from orbit.runtime.evidence import EvidenceStore
from tests.test_analysis_runtime import ScriptedBackend, tool_response, prose_response
from tests.test_analysis_ioc_proof import fixture

LOCAL_UNKNOWN = "function host(){return ['example.invalid'].join('');}location.href='https://'+host();"


class ObjectiveTests(unittest.TestCase):
    def runtime(self, source=LOCAL_UNKNOWN, backend=None):
        temp=tempfile.TemporaryDirectory(prefix='orbit-ioc-objective-')
        self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        original=root/'source.html'
        original.write_bytes(source.encode())
        rt=AnalysisRuntime(backend=backend or ScriptedBackend(plan_questions=[]),
            source=acquire_analysis_source(original,root/'snapshot'), evidence_store=EvidenceStore(root/'evidence'))
        self.addCleanup(rt.close)
        return rt

    def test_empty_plan_does_not_skip_runtime_objective(self):
        backend=ScriptedBackend(tool_response('print("bounded local inspection one")'),
            tool_response('print("bounded local inspection two")'),plan_questions=[],
            finish_decisions=[{'status':'resolved','answer_summary':'A plausible unverified URL'},
                              {'status':'resolved','answer_summary':'Still no static proof'}])
        rt=self.runtime(backend=backend)
        run=rt.run_autonomous('Analyse.',cover=False,finalize=False)
        self.assertEqual(run.actions_executed,2)
        self.assertNotEqual(run.stop_reason,STOP_LEDGER_EXHAUSTED)
        self.assertIn('IOC',run.open_questions)
        self.assertNotIn('IOC',run.answered_unverified_questions)
        outcome=rt._report_runs[-1]['questions'][0][1]
        self.assertEqual(outcome['status'],BLOCKED)
        self.assertIn('2-action',outcome['reason'])
        self.assertEqual(rt.ioc_checks(include_outcome=True)[0]['state'],'BLOCKED')
        self.assertEqual(backend.calls,2)

    def test_runtime_objective_precedes_model_questions_and_prevents_complete(self):
        rt=self.runtime()
        controller=AnalysisController()
        rt._ioc_register(controller)
        controller.adopt_plan([{'question':'Explain the layout','missing_fact':'interpretation'}])
        self.assertEqual(controller.activate_next().id,'IOC')
        self.assertEqual(controller.questions['Q1'].question,'Explain the layout')
        self.assertTrue(rt._ioc_pending(controller))
        self.assertFalse(controller.exhausted)

    def test_prose_complete_cannot_bypass_pending_destination(self):
        backend = ScriptedBackend(prose_response('The investigation is complete.'),
            tool_response('print("local inspection still required")'), plan_questions=[])
        rt = self.runtime(backend=backend)
        run = rt.run_autonomous('Analyse.', cover=False, max_model_calls=4, finalize=False)
        self.assertEqual(run.actions_executed, 1)
        self.assertNotEqual(run.stop_reason, 'model returned prose with no action')
        self.assertIn('IOC', run.open_questions)

    def test_model_cannot_close_or_spawn_children_for_runtime_objective(self):
        rt=self.runtime()
        controller=AnalysisController()
        rt._ioc_register(controller);controller.adopt_plan([]);controller.activate_next()
        record=rt.evidence_store.add('execute_analysis','https://invented.invalid',metadata={
            'tool_call_id':'a','user_turn_id':'turn_1','produced_by_phase':'analysis_step'})
        for status in ('resolved','blocked','open'):
            with self.subTest(status=status):
                rt._apply_decision(controller,{'status':status,'answer_summary':'Unverified proposal',
                    'evidence_ids':[record.evidence_id], 'child_question':{
                        'question':'Different objective','missing_fact':'other', 'caused_by_evidence_id':record.evidence_id}},record.evidence_id)
                self.assertEqual(controller.states['IOC'].status,OPEN)
                self.assertEqual(controller.order,['IOC'])
                self.assertEqual(controller.states['IOC'].summary,'Unverified proposal')
                self.assertNotIn('https://invented.invalid',{i.value for i in rt.canonical_indicators()})

    def test_proven_destination_needs_no_extra_action_and_precedes_narrative(self):
        rt=self.runtime(fixture())
        controller=AnalysisController();rt._ioc_register(controller)
        self.assertEqual(controller.order,[])
        self.assertEqual(rt.ioc_checks()[0]['state'],'RESOLVED_EXACT')
        self.assertIn('https://example.invalid/download',{i.value for i in rt.canonical_indicators()})
        report=rt.report(generate_narrative=False)
        self.assertIn('https://example.invalid/download',report.text)
        self.assertTrue(report.document_complete)
        self.assertIn('not effective browser mutation, observed execution or network contact',report.text)
        self.assertEqual(rt.model_calls,0)
        self.assertEqual(rt.actions_executed,0)

    def test_runtime_input_can_be_blocked_without_futile_local_action(self):
        rt=self.runtime('location.href=navigator.userAgent;')
        controller=AnalysisController();rt._ioc_register(controller)
        self.assertEqual(controller.states['IOC'].status,BLOCKED)
        self.assertIn('runtime input',controller.states['IOC'].reason)
        self.assertFalse(rt._ioc_pending(controller))

    def test_destination_proof_is_not_deduplicated_against_plain_value(self):
        url = 'https://example.invalid/a'
        encoded = ','.join(str(ord(c)) for c in url)
        rt = self.runtime(f'<p>String.fromCharCode({encoded})</p><script>location.href="{url}";</script>')
        kinds = {s.kind for s, _r in rt.transform_stages if s.output == url}
        self.assertIn('js_fromcharcode_offset', kinds)
        self.assertIn('js_destination_proof', kinds)
        self.assertEqual(rt.ioc_checks()[0]['state'], 'RESOLVED_EXACT')

    def test_evidence_json_round_trip_retains_proof_and_report(self):
        rt = self.runtime(fixture())
        facts = rt.ioc_checks()
        report = rt.report(generate_narrative=False)
        rt.evidence_store.load_index()
        self.assertEqual(rt.ioc_checks(), facts)
        restored = rt.report(generate_narrative=False)
        self.assertTrue(restored.document_complete)
        self.assertEqual(restored.text, report.text)

    def test_global_zero_budget_preserves_unresolved_objective(self):
        rt=self.runtime()
        run=rt.run_autonomous('Analyse.',cover=False,max_model_calls=0,finalize=False)
        self.assertEqual(run.actions_executed,0)
        self.assertIn('IOC',run.open_questions)
        self.assertIn('model call',rt.ioc_checks(include_outcome=True)[0]['reason'])

    def test_proven_host_ip_and_unrelated_data_are_distinct(self):
        for prop, value, kind in [('hostname', 'example.invalid', 'hostname'), ('host', '192.0.2.4', 'IP')]:
            rt = self.runtime(f'location.{prop}="{value}";')
            self.assertEqual(rt.ioc_checks()[0]['state'], 'RESOLVED_EXACT')
            self.assertIn((kind, value), {(i.kind, i.value) for i in rt.canonical_indicators()})
        rt = self.runtime('var name="example.invalid";var address="192.0.2.4";')
        self.assertEqual(rt.ioc_checks(), [])
        self.assertEqual(rt.canonical_indicators(), [])

    def test_existing_indicator_bound_preserves_all_proof_records(self):
        from orbit.runtime.analysis_indicators import MAX_INDICATORS
        source = '// ' + ' '.join(f'https://example.invalid/{i}' for i in range(MAX_INDICATORS))
        rt = self.runtime(source + '\nlocation.hostname="example.invalid";')
        self.assertEqual(len(rt.canonical_indicators()), MAX_INDICATORS)
        self.assertEqual(rt.ioc_checks()[0]['state'], 'RESOLVED_EXACT')
        self.assertEqual(rt.ioc_checks()[0]['value'], 'example.invalid')

    def test_oversized_input_has_coverage_limit_not_invented_objective(self):
        from orbit.runtime.analysis_ioc_proof import MAX_INPUT_CHARS
        rt = self.runtime('x' * (MAX_INPUT_CHARS + 1))
        self.assertEqual(rt.ioc_checks(), [])
        controller = AnalysisController(); rt._ioc_register(controller)
        self.assertEqual(controller.order, [])
        report = rt.report(generate_narrative=False)
        self.assertIn('no absence of network destinations', report.text)

    def test_cancel_error_and_incomplete_generation_cannot_close_objective(self):
        from dataclasses import replace
        from orbit.backend.llama_server import LlamaServerError
        for mode in ('cancel', 'error', 'length'):
            class Backend(ScriptedBackend):
                def chat_stream(self, messages, **kwargs):
                    offered = [t['function']['name'] for t in kwargs.get('tools') or []]
                    if 'execute_analysis' in offered:
                        if mode == 'cancel': raise KeyboardInterrupt()
                        if mode == 'error': raise LlamaServerError('fixture unavailable')
                        return replace(tool_response('raise RuntimeError("must not run")'), finish_reason='length')
                    return super().chat_stream(messages, **kwargs)
            with self.subTest(mode=mode):
                rt = self.runtime(backend=Backend(plan_questions=[]))
                with patch('orbit.runtime.analysis_runtime.execute_analysis') as executor:
                    run = rt.run_autonomous('Analyse.', cover=False, max_model_calls=4, finalize=False)
                executor.assert_not_called()
                self.assertEqual(run.actions_executed, 0)
                self.assertIn('IOC', run.open_questions)
                self.assertEqual(rt.ioc_checks(include_outcome=True)[0]['state'], 'BLOCKED')
                self.assertIn('Network destination', rt.report(generate_narrative=False).text)

    def test_existing_action_repair_retains_objective_and_limits(self):
        from tests.test_analysis_repair import ExactRecordingBackend, _tool_call
        backend = ExactRecordingBackend(_tool_call('raise TypeError("fixture failure")'),
            _tool_call('print("bounded local inspection")'), plan_questions=[])
        rt = self.runtime(backend=backend)
        run = rt.run_autonomous('Analyse.', cover=False, finalize=False)
        self.assertEqual(run.actions_executed, 2)
        self.assertEqual(run.repairs, 1)
        self.assertIn('IOC', run.open_questions)
        self.assertNotIn('IOC', run.answered_unverified_questions)
        self.assertIn('fixture failure', str(backend.action_requests[-1]))

    def test_evidence_proof_snapshot_session_and_history_are_reattested(self):
        for change in ('withdraw','body','links','source','store','history',
                       'metadata_source','metadata_input','metadata_output','metadata_root','phase'):
            with self.subTest(change=change):
                rt=self.runtime(fixture())
                stage,record=rt.transform_stages[0]
                if change=='withdraw':rt.evidence_store.records.pop(record.evidence_id)
                elif change=='body':(rt.evidence_store.root/f'{record.evidence_id}.txt').write_text('https://wrong.invalid')
                elif change=='links':record.metadata['ioc_proofs']=[]
                elif change=='source':
                    rt.source.snapshot_path.chmod(0o600);rt.source.snapshot_path.write_text('other')
                elif change=='store':
                    import shutil
                    other = rt.evidence_store.root.parent/'other-session'
                    shutil.copytree(rt.evidence_store.root, other)
                    rt.evidence_store=EvidenceStore(other);rt.evidence_store.load_index()
                elif change.startswith('metadata_'):
                    key={'metadata_source':'analysis_source_sha256','metadata_input':'input_sha256',
                         'metadata_output':'output_sha256','metadata_root':'ioc_store_root'}[change]
                    record.metadata[key]='wrong'
                elif change=='phase':
                    from dataclasses import replace
                    other=replace(record,produced_by_phase='analysis_step')
                    rt.transform_stages[0]=(stage,other)
                    rt.evidence_store.records[record.evidence_id]=other
                else:rt.messages=[m for m in rt.messages if 'analysis_transform_ids' not in m]
                self.assertNotEqual(rt.ioc_checks()[0]['state'],'RESOLVED_EXACT')
                self.assertNotIn(stage.output,{i.value for i in rt.canonical_indicators()})
                controller=AnalysisController();rt._ioc_register(controller)
                self.assertEqual(controller.states['IOC'].status,OPEN)

if __name__ == '__main__':unittest.main()
