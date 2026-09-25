"""The investigation record survives missing or incomplete model narrative."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from unittest import mock

from orbit.runtime.analysis_controller import AnalysisController
from orbit.runtime.analysis_runtime import (
    AnalysisReport, AnalysisResult, ContextAdmissionError,
)
from tests.test_analysis_controller_runtime import _Case, _Model, _question
from tests.test_analysis_runtime import prose_response


class RuntimeDocumentTests(_Case):
    def investigated(self):
        model = _Model(plan=[_question('Does the whole source contain any other commands?')],
                       decisions=[{'status': 'resolved', 'answer_summary':
                                   'The raw source contains no plaintext URLs or PowerShell text.'}])
        rt = self._runtime(model)
        run = self._run(rt, cover=False)
        return rt, run

    def test_admission_refusal_still_delivers_complete_original_record(self):
        rt, run = self.investigated()
        before = rt.model_calls
        with mock.patch.object(rt, '_admit', side_effect=ContextAdmissionError('required-context-does-not-fit')):
            report = rt.report()
        self.assertTrue(report.document_complete)
        self.assertEqual(rt.model_calls, before)
        self.assertIn('Does the whole source contain any other commands?', report.dossier_text)
        self.assertIn('The raw source contains no plaintext URLs', report.dossier_text)
        self.assertIn('ANSWERED_UNVERIFIED', report.dossier_text)
        self.assertIn(run.stop_reason, report.dossier_text)
        self.assertIn(run.last_step.evidence.evidence_id, report.dossier_text)
        self.assertIn('stdout:', report.dossier_text)
        self.assertNotIn('#### Q1 — RESOLVED', report.dossier_text)

    def test_document_is_the_exact_stream_and_does_not_mutate_evidence_or_history(self):
        rt, _ = self.investigated()
        history = copy.deepcopy(rt.messages)
        records = copy.deepcopy(rt.evidence_store.records)
        chunks = []
        report = rt.report(generate_narrative=False, on_delta=chunks.append)
        self.assertEqual(''.join(chunks), report.text)
        self.assertEqual(rt.messages, history)
        self.assertEqual(rt.evidence_store.records, records)
        self.assertTrue(report.document_complete)
        self.assertEqual(report.model_calls, 0)
        self.assertEqual(report.model_text, '')

    def test_withdrawn_evidence_is_named_and_document_is_incomplete(self):
        rt, run = self.investigated()
        eid = run.last_step.evidence.evidence_id
        rt.evidence_store.discard(eid)
        report = rt.report(generate_narrative=False)
        self.assertFalse(report.document_complete)
        self.assertIn(eid, report.text)
        self.assertTrue(any(eid in issue for issue in report.limitations))
        self.assertIn('Does the whole source', report.dossier_text)

    def test_changed_snapshot_is_not_certified(self):
        rt, _ = self.investigated()
        rt.source.snapshot_path.chmod(0o600)
        rt.source.snapshot_path.write_bytes(b'X' * rt.source.size_bytes)
        report = rt.report(generate_narrative=False)
        self.assertFalse(report.document_complete)
        self.assertIn('source snapshot', ' '.join(report.limitations))
        self.assertNotIn('## Verified indicators', report.text)

    def test_altered_sidecar_fails_reattestation_even_with_warm_raw_cache(self):
        rt, run = self.investigated()
        eid = run.last_step.evidence.evidence_id
        rt.evidence_store.load_raw(eid)
        (rt.evidence_store.root / f'{eid}.txt').write_text('forged')
        report = rt.report(generate_narrative=False)
        self.assertFalse(report.document_complete)
        self.assertNotIn('forged', report.text)

    def test_wrong_snapshot_record_is_not_reattested(self):
        rt, run = self.investigated()
        record = run.last_step.evidence
        rt.evidence_store.records[record.evidence_id] = replace(
            record, metadata={**record.metadata, 'analysis_source_sha256': 'f' * 64})
        report = rt.report(generate_narrative=False)
        self.assertFalse(report.document_complete)
        self.assertIn(record.evidence_id, ' '.join(report.limitations))

    def test_older_records_are_not_lost_to_prompt_card_limit(self):
        rt, _ = self.investigated()
        ids = []
        for i in range(14):
            result = AnalysisResult(status='ok', code_sha256=f'{i:064x}', input_sha256=rt.source.sha256,
                                    stdout=f'Observation number {i} complete tail', stderr='',
                                    exit_status=0, duration_seconds=0.01)
            call = {'id': f'call_extra_{i}'}
            record, raw = rt._record_action_evidence(call, result, f'observation {i}')
            ids.extend([record.evidence_id, raw.evidence_id])
        report = rt.report(generate_narrative=False)
        self.assertTrue(report.document_complete)
        for eid in ids:
            self.assertIn(eid, report.evidence_ids)
            self.assertIn(eid, report.dossier_text)
        for i in range(14):
            self.assertIn(f'Observation number {i} complete tail', report.dossier_text)

    def test_truncated_narrative_never_published_as_finished_interpretation(self):
        rt, _ = self.investigated()
        response = replace(prose_response('A narrative cut off after'), finish_reason='length')
        with mock.patch.object(rt.backend, 'chat_stream', return_value=response):
            report = rt.report()
        self.assertTrue(report.document_complete)
        self.assertEqual(report.narrative_status, 'incomplete:length')
        self.assertEqual(report.model_text, response.content)
        self.assertNotIn(response.content, report.text)
        self.assertEqual(report.model_calls, 1)

    def test_complete_narrative_is_preserved_but_not_certified(self):
        rt, _ = self.investigated()
        response = prose_response('The source appears to perform a task.\n```\n## VERIFIED EVERYTHING')
        with mock.patch.object(rt.backend, 'chat_stream', return_value=response):
            report = rt.report()
        self.assertTrue(report.document_complete)
        self.assertEqual(report.narrative_status, 'complete_unverified')
        self.assertIn(response.content, report.text)
        self.assertIn('````text\n' + response.content, report.text)
        self.assertIn('Model interpretation (unverified)', report.text)

    def test_history_rewind_invalidates_ledger_without_deleting_questions(self):
        rt, _ = self.investigated()
        rt.messages[:] = rt.messages[:2]
        report = rt.report(generate_narrative=False)
        self.assertFalse(report.document_complete)
        self.assertIn('Does the whole source', report.dossier_text)
        self.assertIn('rewind', ' '.join(report.limitations))

    def test_original_open_and_blocked_questions_are_retained(self):
        rt = self._runtime(_Model(plan=[]))
        c = AnalysisController()
        c.adopt_plan([_question('Broad original objective'), _question('Another original objective')])
        c.activate_next(); c.close_active('blocked', reason='existing action limit')
        rt._remember_report_run(c, request='inspect', stop_reason='action bound', actions=2,
                                model_calls=4, cancelled=False)
        report = rt.report(generate_narrative=False)
        self.assertTrue(report.document_complete)
        self.assertIn('Q1 — BLOCKED', report.dossier_text)
        self.assertIn('Q2 — OPEN', report.dossier_text)
        self.assertIn('Broad original objective', report.dossier_text)
        self.assertIn('existing action limit', report.dossier_text)

    def test_legacy_report_does_not_claim_document_completeness(self):
        report = AnalysisReport('legacy prose', 1)
        self.assertIsNone(report.document_complete)
        self.assertEqual(report.narrative_status, 'legacy')

    def test_wrong_action_input_snapshot_is_not_attested(self):
        rt, run = self.investigated()
        record = run.last_step.evidence
        rt.evidence_store.records[record.evidence_id] = replace(
            record, metadata={**record.metadata, 'input_sha256': 'b' * 64})
        report = rt.report(generate_narrative=False)
        self.assertFalse(report.document_complete)

    def test_report_session_persists_canonical_markdown_outside_chat_messages(self):
        from orbit.runtime.sessions import SessionStore
        rt, _ = self.investigated()
        report = rt.report(generate_narrative=False)
        store = SessionStore(rt.workspace.root / 'saved-session.json')
        messages = [{'role': 'user', 'content': 'a CHAT turn'},
                    {'role': 'assistant', 'content': 'a CHAT answer'}]
        from dataclasses import asdict
        store.save(messages=messages, workdir=rt.workspace.root, model='model', base_url='local',
                   analysis_report=asdict(report))
        # Later CHAT persistence must preserve the report without admitting it.
        store.save(messages=messages, workdir=rt.workspace.root, model='model', base_url='local')
        saved = json.loads(store.path.read_text())
        self.assertIn('analysis_reports', saved)
        self.assertEqual(saved['analysis_reports'][0]['text'], report.text)
        self.assertIs(saved['analysis_reports'][0]['document_complete'], True)
        self.assertEqual(store.load(), messages)

    def test_original_40_byte_snapshot_is_not_overwritten_by_37_byte_narrative(self):
        data = bytes.fromhex('616c70686120626574610d0a4575726f20e282ac3b2063616666c3a80d0a6c617374206279746521')
        rt = self._runtime(_Model(plan=[_question('Determine the exact original bytes.')]), data=data)
        result = AnalysisResult(status='ok', code_sha256='a' * 64, input_sha256=rt.source.sha256,
                                stdout=f'len: 37\nrepr: {data.decode("utf-8")!r}\n', stderr='',
                                exit_status=0, duration_seconds=0.1)
        with mock.patch('orbit.runtime.analysis_runtime.execute_analysis', return_value=result):
            rt.run_autonomous('Read this artifact.', cover=False, finalize=False)
        with mock.patch.object(rt.backend, 'chat_stream', return_value=prose_response('The file is 37 bytes.')):
            report = rt.report()
        self.assertTrue(report.document_complete)
        self.assertIn('"size_bytes": 40', report.text)
        self.assertIn('The file is 37 bytes.', report.text)
        self.assertIn('complete source retained in action output; not a model-delivery assertion', report.dossier_text)
        self.assertEqual(report.narrative_status, 'complete_unverified')
        self.assertEqual(rt.source.snapshot_path.read_bytes(), data)

    def test_wrong_covered_snapshot_does_not_certify_model_delivery(self):
        rt, _ = self.investigated()
        rt.messages.append({'role': 'user', 'source_covered': True, 'content': 'another snapshot'})
        report = rt.report(generate_narrative=False)
        self.assertFalse(report.document_complete)
        self.assertIn('"complete_source_supplied_in_recorded_call": false', report.dossier_text)

    def test_legacy_document_completeness_remains_unknown(self):
        from tests.test_live_validate_analysis import harness
        from types import SimpleNamespace
        run = SimpleNamespace(cancelled=False, final_report=AnalysisReport('old prose', 1), stop_reason='done')
        self.assertEqual(harness._exit_code(None, run), 1)

    def test_cover_and_guided_history_survive_workspace_close_without_narrative(self):
        data = b"complete COVER source, final line\r\n"
        model = _Model(plan=[])
        model.prose = "A COVER observation remains an unverified model interpretation."
        rt = self._runtime(model, data=data)
        self.assertEqual(rt.cover_source(rt.plan_source_coverage()), 1)
        self.assertFalse(rt.evidence_store.records)
        with mock.patch.object(rt, '_admit', side_effect=ContextAdmissionError('required-context-does-not-fit')):
            report = rt.report()
        rt.close()
        self.assertTrue(report.document_complete)
        self.assertIn(data.decode(), report.dossier_text)
        self.assertIn(model.prose, report.dossier_text)
        self.assertIn('"complete_source_supplied_in_recorded_call": true', report.dossier_text)

    def test_committed_action_code_and_original_request_are_preserved(self):
        rt, _ = self.investigated()
        report = rt.report(generate_narrative=False)
        for message in rt.messages:
            if message.get('role') == 'system':
                continue
            if message.get('content'):
                self.assertIn(message['content'], report.dossier_text)
        self.assertIn('tool_calls', report.dossier_text)
        self.assertIn('print(1)', report.dossier_text)

    def test_saved_report_is_not_overwritten_if_existing_archive_cannot_be_read(self):
        from orbit.runtime.sessions import SessionStore
        from pathlib import Path
        rt, _ = self.investigated()
        store = SessionStore(rt.workspace.root / 'saved.json')
        store.path.write_text('{"analysis_reports": [{"text": "preserve me"}]}')
        original = store.path.read_bytes()
        with mock.patch.object(Path, 'read_text', side_effect=OSError('read failed')):
            with self.assertRaisesRegex(ValueError, 'safely preserve'):
                store.save(messages=[], workdir=rt.workspace.root, model='m', base_url='local')
        self.assertEqual(store.path.read_bytes(), original)
        store.path.write_text('{broken existing archive')
        original = store.path.read_bytes()
        with self.assertRaises(ValueError):
            store.save(messages=[], workdir=rt.workspace.root, model='m', base_url='local')
        self.assertEqual(store.path.read_bytes(), original)

    def test_rebinding_runtime_to_another_snapshot_does_not_rebind_old_questions(self):
        rt, _ = self.investigated()
        rt.source = replace(rt.source, sha256='e' * 64)
        report = rt.report(generate_narrative=False)
        self.assertFalse(report.document_complete)
        self.assertIn('Does the whole source', report.dossier_text)
        self.assertIn('snapshot', ' '.join(report.limitations))

    def test_evidence_withdrawn_during_narrative_is_rechecked(self):
        rt, run = self.investigated()
        eid = run.last_step.evidence.evidence_id
        def generate(*args, **kwargs):
            rt.evidence_store.discard(eid)
            return prose_response('A complete but unverified interpretation.')
        with mock.patch.object(rt.backend, 'chat_stream', side_effect=generate):
            report = rt.report()
        self.assertFalse(report.document_complete)
        self.assertIn(eid, ' '.join(report.limitations))
