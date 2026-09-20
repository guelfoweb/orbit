"""Explicit reads of authorized transformations cross the real sandbox boundary."""
import copy
import hashlib
import unittest
from unittest import mock

from tests import test_analysis_evidence_delivery as delivery_tests
from tests.test_analysis_evidence_first import decodable
from tests.test_analysis_runtime import tool_response
from orbit.runtime.analysis_runtime import FINISH_TOOL_SCHEMA
from orbit.runtime import analysis_sandbox
from orbit.runtime.analysis_controller import AnalysisController
from orbit.runtime.analysis_progress import ProgressLedger
from orbit.runtime.analysis_tools_shim import MAX_READ_BYTES
import tempfile
from pathlib import Path



class EvidenceReaderTests(unittest.TestCase):
    runtime = delivery_tests.EvidenceDeliveryTests.runtime

    def action(self, rt, backend, eid):
        code = f'import orbit_tools\nprint(orbit_tools.read_evidence({eid!r}), end="")'
        backend._responses = backend._responses[:backend.calls] + [tool_response(code)]
        return code, rt.step('Read the registered transform output.')

    def test_budget_withheld_body_is_read_without_path_or_recomputation(self):
        value = 'café\r\n終' + ' exact-value' * 180
        rt, b = self.runtime(decodable(value))
        b.context = 6000
        record = rt.transform_stages[0][1]
        history = copy.deepcopy(rt.messages)
        code, step = self.action(rt, b, record.evidence_id)
        receipts = [d for m in b.seen_messages[0] for d in m.get('analysis_evidence_delivery', [])]
        self.assertEqual(receipts[0]['status'], 'not_delivered')
        self.assertEqual(step.result.status, 'ok')
        self.assertEqual(step.result.stdout.encode(), value.encode())
        self.assertEqual(step.result.code_sha256, hashlib.sha256(code.encode()).hexdigest())
        self.assertIsNone(step.suppressed_duplicate_of)
        self.assertIn(value, step.evidence and rt.evidence_store.reattest_exact(step.evidence.evidence_id))
        supplied = step.evidence.metadata['sandbox_evidence_inputs']
        self.assertEqual(supplied[0]['evidence_id'], record.evidence_id)
        self.assertEqual(supplied[0]['byte_range'], [0, len(value.encode())])
        self.assertEqual(supplied[0]['sha256'], record.raw_sha256)
        self.assertEqual(rt.messages[:len(history)], history)

    def test_missing_tampered_wrong_source_and_unrelated_records_are_unavailable(self):
        for fault in ('withdrawn','tampered','source','unrelated'):
            with self.subTest(fault=fault):
                rt,b=self.runtime()
                _,r=rt.transform_stages[0];eid=r.evidence_id
                if fault=='withdrawn': rt.evidence_store.discard(eid)
                if fault=='tampered': (rt.evidence_store.root/(eid+'.txt')).write_text('forged')
                if fault=='source':
                    rt.source.snapshot_path.chmod(0o600);rt.source.snapshot_path.write_bytes(b'other snapshot')
                if fault=='unrelated': eid=rt.evidence_store.add('execute_analysis','private other output').evidence_id
                _,step=self.action(rt,b,eid)
                self.assertEqual(step.result.status,'error')
                self.assertIn('not available',step.result.stderr)
                self.assertNotIn('private other output',step.result.stdout)

    def test_evidence_does_not_leak_to_another_runtime(self):
        first,_=self.runtime(decodable('first-session-secret'))
        second,b=self.runtime(decodable('second-session'))
        _,step=self.action(second,b,first.transform_stages[0][1].evidence_id)
        self.assertEqual(step.result.status,'error')
        self.assertNotIn('first-session-secret',step.result.stdout)

    def test_paths_are_not_silently_converted_to_evidence_reads(self):
        rt,b=self.runtime();eid=rt.transform_stages[0][1].evidence_id
        b._responses=[tool_response(f'import orbit_tools\nprint(orbit_tools.read_file("evidence:{eid}"))')]
        step=rt.step('Read the value.')
        self.assertEqual(step.result.status,'error')
        self.assertIn('FileNotFoundError', step.result.stderr)

    def test_retiring_reader_input_changes_duplicate_identity(self):
        value='not-yet-delivered'*140
        rt,b=self.runtime(decodable(value));b.context=6000;eid=rt.transform_stages[0][1].evidence_id
        _,first=self.action(rt,b,eid)
        self.assertEqual(first.result.status,'ok')
        rt.evidence_store.discard(eid)
        _,second=self.action(rt,b,eid)
        self.assertIsNone(second.suppressed_duplicate_of)
        self.assertEqual(second.result.status,'error')

    def test_body_already_delivered_to_step_remains_redundant(self):
        rt,b=self.runtime();eid=rt.transform_stages[0][1].evidence_id
        _,step=self.action(rt,b,eid)
        self.assertEqual(step.result.status,'ok')
        self.assertIsNotNone(step.suppressed_duplicate_of)


    def test_autonomous_reader_delivers_withheld_body_to_finish(self):
        value = 'safe decoded text ' * 45 + 'END'
        rt, b = self.runtime(decodable(value))
        b.context = 6000
        eid = rt.transform_stages[0][1].evidence_id
        b._plan_questions = ['What is the exact decoded text?']
        b._responses = [tool_response(
            f'import orbit_tools\nprint(orbit_tools.read_evidence({eid!r}), end="")')]
        b._finish_decisions = [{'status':'resolved', 'answer_summary':value,
                                'evidence_ids':[eid]}]
        run = rt.run_autonomous('Inspect the stored output.', cover=False,
                                max_model_calls=3, finalize=False)
        self.assertNotIn('error', run.stop_reason, run.stop_reason)
        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(b.finish_calls, 1)
        step_messages = next(m for m,t in zip(b.seen_messages,b.seen_tools)
                             if t == ['execute_analysis'])
        self.assertTrue(all(r['status']=='not_delivered' for m in step_messages
                            for r in m.get('analysis_evidence_delivery', [])))
        finish = next(m for m,t in zip(b.seen_messages,b.seen_tools)
                      if t == ['finish_analysis_question'])
        self.assertIn(value, '\n'.join(m.get('content','') for m in finish))
        self.assertTrue(run.answered_unverified_questions)
        self.assertFalse(run.resolved_questions)

    def test_snapshot_change_at_executor_boundary_refuses_before_launch(self):
        rt,b=self.runtime(); eid=rt.transform_stages[0][1].evidence_id
        real=analysis_sandbox.execute_analysis
        def changed(**kw):
            rt.source.snapshot_path.chmod(0o600)
            rt.source.snapshot_path.write_bytes(b'changed after authorization')
            return real(**kw)
        with mock.patch('orbit.runtime.analysis_runtime.execute_analysis', side_effect=changed), \
             mock.patch.object(analysis_sandbox.subprocess, 'Popen') as launch, \
             mock.patch.object(analysis_sandbox, 'sandbox_preflight'):
            _,step=self.action(rt,b,eid)
        self.assertFalse(step.action_executed)
        self.assertIn('do not match', step.rejection)
        launch.assert_not_called()

    def test_progress_identity_includes_reader_inputs(self):
        rt,b=self.runtime(decodable('long value '*160));b.context=6000
        eid=rt.transform_stages[0][1].evidence_id
        _,step=self.action(rt,b,eid)
        ledger=ProgressLedger()
        before=ledger._strategy_fingerprint(step,step.evidence,step.result.code_sha256)
        from dataclasses import replace
        changed=replace(step.evidence,metadata={**step.evidence.metadata,
                         'sandbox_evidence_inputs_sha256':'another attested input'})
        after=ledger._strategy_fingerprint(step,changed,step.result.code_sha256)
        self.assertNotEqual(before,after)
        self.assertEqual(before,ledger._strategy_fingerprint(step,step.evidence,step.result.code_sha256))


class SandboxEvidenceInputTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        self.source=Path(directory.name)/'source';self.source.write_bytes(b'safe snapshot')
        self.digest=hashlib.sha256(self.source.read_bytes()).hexdigest()

    def execute(self, body, code=None):
        return analysis_sandbox.execute_analysis(source_path=self.source,
            code=code or "import orbit_tools\nprint(orbit_tools.read_evidence('ev_test'),end='')",
            evidence_inputs={'ev_test':body}, evidence_source_sha256=self.digest)

    def test_values_are_inert_exact_data_not_python_or_paths(self):
        body="café\r\n終\x00';raise RuntimeError('injected')#"
        result=self.execute(body)
        self.assertEqual(result.status,'ok')
        self.assertEqual(result.stdout.encode(),body.encode())
        self.assertFalse(result.artifacts)

    def test_available_mapping_cannot_be_mutated_in_place(self):
        result=self.execute('original', "import orbit_tools\norbit_tools._EVIDENCE_INPUTS['ev_test']='forged'")
        self.assertEqual(result.status,'error')
        self.assertIn('mappingproxy',result.stderr)

    def test_full_read_limit_has_no_silent_truncation(self):
        body='é'*(MAX_READ_BYTES//2)
        result=self.execute(body,"import orbit_tools,hashlib\nprint(hashlib.sha256(orbit_tools.read_evidence('ev_test').encode()).hexdigest())")
        self.assertEqual(result.status,'ok')
        self.assertEqual(result.stdout.strip(),hashlib.sha256(body.encode()).hexdigest())
        with mock.patch.object(analysis_sandbox.subprocess,'Popen') as launch, \
             mock.patch.object(analysis_sandbox, 'sandbox_preflight'):
            with self.assertRaisesRegex(ValueError,'complete-read bound'): self.execute(body+'!')
        launch.assert_not_called()

    def test_aggregate_limit_is_checked_before_executor(self):
        with mock.patch.object(analysis_sandbox.subprocess,'Popen') as launch, \
             mock.patch.object(analysis_sandbox, 'sandbox_preflight'):
            with self.assertRaisesRegex(ValueError,'aggregate'):
                analysis_sandbox.execute_analysis(source_path=self.source,code='print(1)',
                    evidence_source_sha256=self.digest,
                    evidence_inputs={f'ev_{i}':'x'*MAX_READ_BYTES for i in range(129)})
        launch.assert_not_called()

if __name__=='__main__': unittest.main()
