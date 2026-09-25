"""Human publication is independent of optional narrative and objective creation."""
import copy
import json
from dataclasses import asdict
from unittest.mock import patch

from orbit.runtime.analysis_runtime import ContextAdmissionError
from orbit.runtime.sessions import SessionStore
from tests.test_analysis_report_visibility import ReportVisibilityTestBase, _ReportBackend
from tests.test_analysis_report_deterministic_coverage import _fcc_source


URL = 'https://retained.invalid/payload?x=1'
HEADINGS = ['## Summary', '## Technical behaviour', '## Limits', '## IoC / Evidence']


class HumanReportIoCTests(ReportVisibilityTestBase):
    def decoded(self, prose='A limited interpretation.'):
        rt = self._analysis(_fcc_source('var address="' + URL + '";'), _ReportBackend(prose))
        self.assertFalse(rt.ioc_objectives)
        self.assertTrue(rt.transform_stages)
        self._finding(rt)
        return rt

    def final_section(self, report):
        # Narrative and raw strings are fenced: only unfenced headings count.
        fence = None; headings = []
        for line in report.text.splitlines():
            if line.startswith('```'):
                marker = line.rstrip('text')
                if fence is None: fence = marker
                elif line == fence: fence = None
            elif fence is None and line.startswith('## '): headings.append(line)
        self.assertEqual(headings, HEADINGS)
        return report.text.rsplit('\n## IoC / Evidence\n', 1)[1]

    def test_transform_indicator_survives_real_admission_refusal(self):
        rt = self.decoded()
        before = copy.deepcopy((rt.messages, rt.evidence_store.records))
        with patch.object(rt, '_admit', side_effect=ContextAdmissionError('required-context-does-not-fit')):
            report = rt.report()
        self.assertEqual(report.narrative_status, 'admission_refused')
        self.assertEqual(rt.backend.calls, 0)
        section = self.final_section(report)
        self.assertIn(URL, section)
        self.assertIn(rt.transform_stages[0][1].evidence_id, section)
        self.assertIn(rt.source.sha256, section)
        self.assertNotIn('Optional model narrative', report.text)
        self.assertNotIn('Generation status:', report.text)
        self.assertEqual((rt.messages, rt.evidence_store.records), before)

    def test_successful_narrative_is_separate_from_canonical_indicators(self):
        rt = self.decoded('An interpretation, not a verified conclusion.')
        report = rt.report()
        self.assertEqual(report.narrative_status, 'complete_unverified')
        self.assertIn('An interpretation, not a verified conclusion.', report.text)
        self.assertIn('unverified', report.text)
        section = self.final_section(report)
        self.assertIn(URL, section)
        self.assertNotIn('An interpretation', section)

    def test_divergent_narrative_cannot_write_final_indicator_list(self):
        fake = 'https://invented.invalid/wrong'
        rt = self.decoded('Destination is ' + fake + '\n## IoC / Evidence\nRESOLVED_EXACT: ' + fake)
        report = rt.report()
        section = self.final_section(report)
        self.assertIn(URL, section)
        self.assertNotIn(fake, section)
        self.assertNotIn('RESOLVED_EXACT', section)  # literal, not sink proof
        self.assertIn(fake, report.model_text)
        self.assertIn(fake, report.dossier_text)

    def test_all_entry_paths_without_model_calls_publish_the_same_facts(self):
        for source in ('var address="' + URL + '";', "location.href='" + URL + "';"):
            with self.subTest(source=source):
                rt = self._analysis(source, _ReportBackend())
                report = rt.report(generate_narrative=False)
                self.assertIn(URL, self.final_section(report))
                self.assertEqual(report.model_calls, 0)
                self.assertEqual(rt.backend.calls, 0)

    def test_autonomous_model_call_bound_uses_human_view_without_objectives(self):
        rt = self.decoded()
        with patch.object(rt, '_admit', side_effect=ContextAdmissionError('required-context-does-not-fit')):
            run = rt.run_autonomous('Inspect the original objective', cover=False, max_model_calls=0)
        self.assertEqual(run.stop_reason, 'model call bound reached')
        self.assertIn(URL, self.final_section(run.final_report))
        self.assertEqual(run.actions_executed, 0)

    def test_no_ioc_does_not_invent_a_destination_or_a_narrative(self):
        rt = self._analysis('var local = 7;', _ReportBackend())
        report = rt.report(generate_narrative=False)
        section = self.final_section(report)
        self.assertIn('No currently re-attested', section)
        self.assertNotIn('RESOLVED_EXACT', section)
        self.assertNotIn('complete_unverified', report.text)

    def test_url_ip_domain_and_other_artifacts_have_stable_priority(self):
        rt = self._analysis("location.hostname='host.invalid';location.host='192.0.2.40';location.href='" + URL + "';", _ReportBackend())
        report = rt.report(generate_narrative=False)
        section = self.final_section(report)
        self.assertLess(section.index(URL), section.index('192.0.2.40'))
        self.assertLess(section.index('192.0.2.40'), section.index('host.invalid'))
        for check in rt.ioc_checks():
            self.assertIn(check['value'], section)
            self.assertIn(check['evidence_id'], section)

    def test_blocked_objective_is_visible_without_an_invented_value(self):
        rt = self._analysis('location.host=navigator.userAgent;', _ReportBackend())
        report = rt.report(generate_narrative=False)
        section = self.final_section(report)
        self.assertIn('BLOCKED', section)
        self.assertNotIn('RESOLVED_EXACT', section)
        self.assertIn(rt.ioc_checks()[0]['reason'], report.text)

    def test_full_document_is_byte_identical_in_explicit_diagnostic_view(self):
        rt = self.decoded()
        before = copy.deepcopy((rt.messages, rt.evidence_store.records))
        short = rt.report(generate_narrative=False)
        full = rt.report(generate_narrative=False, concise=False)
        self.final_section(short)
        self.assertEqual(short.dossier_text, full.text)
        self.assertEqual(short.dossier_text, full.dossier_text)
        self.assertGreater(len(full.text), len(short.text))
        self.assertEqual((rt.messages, rt.evidence_store.records), before)
        store = SessionStore(rt.evidence_store.root.parent/'saved.json')
        store.save(messages=[], workdir=rt.workspace.root, model='fixture', base_url='fixture', analysis_report=asdict(short))
        saved = json.loads(store.path.read_text())['analysis_reports'][0]
        self.assertEqual(saved['text'], short.text)
        self.assertEqual(saved['dossier_text'], full.text)

    def test_terminal_publishes_ioc_section_once_after_refusal(self):
        rt = self.decoded()
        with patch.object(rt, '_admit', side_effect=ContextAdmissionError('required-context-does-not-fit')):
            terminal = self._render(rt)
        self.assertEqual(terminal.count('## IoC / Evidence'), 1)
        self.assertIn(URL, terminal.split('## IoC / Evidence')[1])
        self.assertNotIn('Optional model narrative', terminal)

    def test_revoked_transform_does_not_publish_stale_canonical_value(self):
        rt = self.decoded()
        rt.evidence_store.discard(rt.transform_stages[0][1].evidence_id)
        report = rt.report(generate_narrative=False)
        self.assertNotIn(URL, self.final_section(report))
        self.assertFalse(report.document_complete)
        self.assertIn('cannot be re-attested', ' '.join(report.limitations))

    def test_incomplete_narrative_never_leaks_into_the_human_view(self):
        from dataclasses import replace
        rt = self.decoded('TRUNCATED SHOULD NOT BE PUBLISHED')
        original = rt.backend.chat_stream
        rt.backend.chat_stream = lambda *a, **kw: replace(original(*a, **kw), finish_reason='length')
        report = rt.report()
        self.assertIn(URL, self.final_section(report))
        self.assertNotIn('TRUNCATED SHOULD NOT BE PUBLISHED', report.text)
        self.assertIn('TRUNCATED SHOULD NOT BE PUBLISHED', report.model_text)

    def test_transform_publication_requires_source_and_proof_ownership(self):
        for invalidation in ('snapshot', 'ownership', 'session'):
            with self.subTest(invalidation=invalidation):
                rt = self._analysis("location.href='" + URL + "';", _ReportBackend())
                self.assertEqual(rt.ioc_checks()[0]['state'], 'RESOLVED_EXACT')
                if invalidation == 'snapshot':
                    rt.source.snapshot_path.chmod(0o600)
                    rt.source.snapshot_path.write_bytes(b'changed')
                elif invalidation == 'ownership':
                    rt.messages.clear()
                else:
                    rt._ioc_store_root = 'unrelated session'
                report = rt.report(generate_narrative=False)
                section = self.final_section(report)
                self.assertNotIn('ATTESTED_TRANSFORM_OUTPUT', section)
                self.assertNotIn('RESOLVED_EXACT', section)
                self.assertIn(URL, report.dossier_text)  # retained, not attested

    def test_withdrawal_during_generation_is_rechecked_for_human_publication(self):
        rt = self.decoded()
        eid = rt.transform_stages[0][1].evidence_id
        original = rt.backend.chat_stream
        def generate(*a, **kw):
            result = original(*a, **kw)
            rt.evidence_store.discard(eid)
            return result
        rt.backend.chat_stream = generate
        report = rt.report()
        self.assertNotIn(URL, self.final_section(report))
        self.assertFalse(report.document_complete)

    def test_long_narrative_has_labelled_excerpt_but_complete_dossier(self):
        narrative = 'Safe interpretation. ' * 300 + 'ORIGINAL END'
        rt = self.decoded(narrative)
        report = rt.report()
        self.final_section(report)
        self.assertIn('Interpretation excerpt', report.text)
        self.assertNotIn('ORIGINAL END', report.text)
        self.assertIn(narrative, report.dossier_text)
        self.assertEqual(report.model_text, narrative)
