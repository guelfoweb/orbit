"""IoC-first stop uses re-attested objectives, never model prose or a count."""
import copy
import json
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

from orbit.runtime.analysis_controller import AnalysisController, BLOCKED, OPEN
from orbit.runtime.sessions import SessionStore
from tests import test_analysis_ioc_objectives as fixtures
LOCAL_UNKNOWN = fixtures.LOCAL_UNKNOWN
from orbit.runtime.analysis_runtime import AnalysisReport
from tests.test_analysis_runtime import ScriptedBackend, tool_response


class IoCCompletionTests(unittest.TestCase):
    runtime = fixtures.ObjectiveTests.runtime

    def test_closed_inventory_stops_before_cover_plan_action_and_narrative(self):
        for source in ("location.href='https://one.invalid/a';",
                       "location.href='https://one.invalid/a';location.hostname='two.invalid';",
                       "location.href='https://one.invalid/a';location.host=navigator.userAgent;"):
            with self.subTest(source=source):
                rt = self.runtime(source)
                before = copy.deepcopy(rt.messages)
                with patch.object(rt, 'cover_source') as cover, patch.object(rt, 'plan_analysis', wraps=rt.plan_analysis) as plan, \
                     patch('orbit.runtime.analysis_runtime.execute_analysis') as execute, \
                     patch.object(rt, '_report_narrative', return_value=AnalysisReport('unverified', 0)) as narrative:
                    run = rt.run_autonomous('Identify network destinations.')
                self.assertEqual(run.model_calls, 0)
                self.assertEqual(run.actions_executed, 0)
                self.assertEqual(run.stop_reason, 'known network destination objectives closed')
                cover.assert_not_called(); plan.assert_not_called(); execute.assert_not_called(); narrative.assert_not_called()
                self.assertEqual(rt.messages, before)
                self.assertTrue(run.final_report.document_complete)
                self.assertEqual(run.final_report.narrative_status, 'not_requested')

    def test_no_objectives_is_not_vacuous_completion(self):
        rt = self.runtime('var unrelated = 42;', ScriptedBackend(plan_questions=[]))
        run = rt.run_autonomous('Analyse.', cover=False, finalize=False)
        self.assertEqual(run.plan_calls, 1)
        self.assertNotEqual(run.stop_reason, 'known network destination objectives closed')

    def test_one_open_among_exact_objectives_must_be_investigated(self):
        rt = self.runtime("location.href='https://one.invalid/a';location.host=7;",
                          ScriptedBackend(tool_response('print("inspect local dependencies")'),
                                          plan_questions=[]))
        self.assertEqual([d['state'] for d in rt.ioc_checks()], ['RESOLVED_EXACT', 'OPEN'])
        run = rt.run_autonomous('Analyse.', cover=False, max_model_calls=2, finalize=False)
        self.assertEqual(run.actions_executed, 1)
        self.assertNotEqual(run.stop_reason, 'known network destination objectives closed')

    def test_final_budget_block_does_not_certify_discovery_and_report_is_retained(self):
        rt = self.runtime(backend=ScriptedBackend(tool_response('print("checked dependency")'),
                                               plan_questions=[]))
        with patch.object(rt, '_report_narrative', return_value=AnalysisReport('No optional synthesis.', 0)) as narrative:
            run = rt.run_autonomous('Analyse.', cover=False, max_model_calls=2)
        self.assertEqual(run.model_calls, 2)
        self.assertEqual(run.stop_reason, 'model call bound reached')
        self.assertEqual(rt.ioc_checks(include_outcome=True)[0]['state'], 'BLOCKED')
        narrative.assert_called_once()
        self.assertEqual(run.final_report.narrative_status, 'not_needed')
        self.assertIn('Static destination discovery incomplete', run.final_report.text)

    def test_open_nonactionable_needs_concrete_block_reason(self):
        rt = self.runtime()
        c = AnalysisController(); rt._ioc_register(c)
        c.states['IOC'].actions = 2
        self.assertFalse(rt._ioc_closed(c))
        c.states['IOC'].status = BLOCKED
        self.assertFalse(rt._ioc_closed(c))
        c.states['IOC'].reason = 'reached the 2-action limit for one question'
        self.assertTrue(rt._known_ioc_objectives_closed(c))
        self.assertFalse(rt._ioc_closed(c))  # budget exhaustion cannot certify discovery

    def test_new_open_objective_cannot_inherit_old_blocked_outcome(self):
        rt = self.runtime()
        c = AnalysisController(); rt._ioc_register(c)
        c.states['IOC'].status = BLOCKED; c.states['IOC'].reason = 'local dependency unavailable'
        self.assertTrue(rt._known_ioc_objectives_closed(c))
        self.assertFalse(rt._ioc_closed(c))
        rt.ioc_objectives += (replace(rt.ioc_objectives[0], start=123, end=130),)
        self.assertFalse(rt._ioc_closed(c))
        rt._ioc_register(c)
        self.assertEqual(c.states['IOC'].status, OPEN)
        self.assertFalse(rt._ioc_closed(c))
        self.assertEqual(c.states['IOC'].actions, 0)

    def test_action_ceiling_blocks_ioc_work_without_certifying_unsupported_discovery(self):
        backend = ScriptedBackend(tool_response('print("inspect dependency one")'),
                                  tool_response('print("inspect dependency two")'),
                                  plan_questions=['Explain the whole artifact'])
        rt = self.runtime(backend=backend)
        with patch.object(rt, '_report_narrative', return_value=AnalysisReport('No optional synthesis.', 0)) as narrative:
            run = rt.run_autonomous('Analyse.', cover=False, max_actions=2)
        self.assertEqual(run.actions_executed, 2)
        self.assertNotEqual(run.stop_reason, 'known network destination objectives closed')
        narrative.assert_called_once()
        states = dict((q['id'], s) for q, s in rt._report_runs[-1]['questions'])
        self.assertEqual(states['Q1']['status'], OPEN)
        self.assertEqual(states['Q1']['actions'], 0)
        self.assertEqual(states['IOC']['status'], BLOCKED)

    def test_new_objective_does_not_reset_used_action_budget(self):
        rt = self.runtime()
        c = AnalysisController(); rt._ioc_register(c)
        c.states['IOC'].actions = 2
        rt.ioc_objectives += (replace(rt.ioc_objectives[0], start=123, end=130),)
        rt._ioc_register(c)
        self.assertEqual(c.states['IOC'].actions, 2)
        self.assertEqual(c.states['IOC'].status, BLOCKED)
        self.assertIn('2-action limit', c.states['IOC'].reason)

    def test_prior_run_block_does_not_relabel_newly_discovered_objectives(self):
        rt = self.runtime()
        c = AnalysisController(); rt._ioc_register(c)
        c.states['IOC'].status = BLOCKED; c.states['IOC'].reason = 'local dependency unavailable'
        rt._remember_report_run(c, request='Analyse.', stop_reason='bounded stop', actions=2, model_calls=3, cancelled=False)
        self.assertEqual(rt.ioc_checks(include_outcome=True)[0]['state'], 'BLOCKED')
        rt.ioc_objectives += (replace(rt.ioc_objectives[0], start=123, end=130),)
        self.assertEqual(rt.ioc_checks(include_outcome=True)[-1]['state'], 'OPEN')

    def test_real_discovery_limit_is_not_closed_inventory(self):
        rt = self.runtime(';'.join(f"location.href='https://example.invalid/{i}'" for i in range(33)) + ';')
        c = AnalysisController(); rt._ioc_register(c)
        self.assertTrue(any(d.property == 'scan' for d in rt.ioc_objectives))
        c.states['IOC'].status = BLOCKED; c.states['IOC'].reason = 'reached the 2-action limit for one question'
        self.assertFalse(rt._ioc_closed(c))

    def test_revoked_exact_proof_keeps_its_own_unavailability_reason(self):
        rt = self.runtime("location.href='https://one.invalid/a';location.host=navigator.userAgent;")
        c = AnalysisController(); rt._ioc_register(c)
        c.states['IOC'].status = BLOCKED; c.states['IOC'].reason = 'other destination unavailable'
        rt._remember_report_run(c, request='Analyse.', stop_reason='bounded stop', actions=2, model_calls=3, cancelled=False)
        self.assertEqual([d['state'] for d in rt.ioc_checks()], ['RESOLVED_EXACT', 'BLOCKED'])
        rt.evidence_store.discard(rt.ioc_checks()[0]['evidence_id'])
        check = rt.ioc_checks(include_outcome=True)[0]
        self.assertEqual(check['state'], 'OPEN')
        self.assertIn('no longer re-attests', check['reason'])
        self.assertFalse(rt._ioc_closed(c))

    def test_mid_action_trusted_producer_closure_skips_finish_and_narrative(self):
        rt = self.runtime("location.href='https://one.invalid/a';", backend=ScriptedBackend(tool_response('print("bounded local inspection")'),
                                               plan_questions=['Interpret everything']))
        real_step = rt.step
        reattest = rt.evidence_store.reattest_exact
        available = False
        def step(*args, **kwargs):
            nonlocal available
            result = real_step(*args, **kwargs)
            # The identical owned proof becomes re-attestable. A narrative or
            # manual objective-state replacement cannot certify discovery.
            available = True
            return result
        with patch.object(rt.evidence_store, 'reattest_exact', side_effect=lambda eid: reattest(eid) if available else None), \
             patch.object(rt, 'step', side_effect=step), patch.object(rt, 'finish_question') as finish, \
             patch.object(rt, '_report_narrative') as narrative:
            run = rt.run_autonomous('Analyse.', cover=False)
        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(len(run.steps), 1)
        self.assertEqual(len(run.progress), 1)
        self.assertEqual(run.stop_reason, 'known network destination objectives closed')
        finish.assert_not_called(); narrative.assert_not_called()

    def test_revocation_source_session_and_scan_limits_prevent_exact_stop(self):
        for mutation in ('revoke', 'source', 'session', 'scan'):
            with self.subTest(mutation=mutation):
                rt = self.runtime("location.href='https://one.invalid/a';")
                self.assertTrue(rt._ioc_closed(None))
                if mutation == 'revoke': rt.evidence_store.discard(rt.ioc_checks()[0]['evidence_id'])
                if mutation == 'source':
                    rt.source.snapshot_path.chmod(0o600)
                    rt.source.snapshot_path.write_bytes(b'changed')
                if mutation == 'session': rt._ioc_store_root = 'different session'
                if mutation == 'scan': rt._ioc_scan_incomplete = 'bounded discovery incomplete'
                self.assertFalse(rt._ioc_closed(None))

    def test_blocked_inventory_also_requires_source_and_session_identity(self):
        for mutation in ('source', 'session'):
            with self.subTest(mutation=mutation):
                rt = self.runtime('location.host=navigator.userAgent;')
                self.assertEqual(rt.ioc_checks()[0]['state'], 'BLOCKED')
                self.assertTrue(rt._ioc_closed(None))
                if mutation == 'source':
                    rt.source.snapshot_path.chmod(0o600)
                    rt.source.snapshot_path.write_bytes(b'x' * rt.source.size_bytes)
                else:
                    rt._ioc_store_root = 'different session'
                self.assertFalse(rt._ioc_closed(None))

    def test_narrative_question_keeps_its_scope_and_open_state(self):
        rt = self.runtime("location.href='https://one.invalid/a';")
        c = AnalysisController(); c.adopt_plan([{'question':'Explain every behaviour.', 'missing_fact':'unobserved behaviour'}])
        before = copy.deepcopy(asdict(c))
        self.assertTrue(rt._ioc_closed(c))
        self.assertEqual(asdict(c), before)
        self.assertEqual(c.states['Q1'].status, OPEN)

    def test_human_report_is_short_dossier_is_complete_and_session_persists_both(self):
        rt = self.runtime("location.href='https://one.invalid/a';")
        c = AnalysisController(); c.adopt_plan([{'question':'Original broad objective', 'missing_fact':'Full behavioural explanation'}])
        c.states['Q1'].summary = 'An unsupported narrative claim'
        rt._remember_report_run(c, request='original request', stop_reason='known network destination objectives closed', actions=0, model_calls=0, cancelled=False)
        before = copy.deepcopy(rt.messages)
        report = rt.report(generate_narrative=False, concise=True)
        self.assertEqual([line for line in report.text.splitlines() if line.startswith('## ')],
                         ['## Summary', '## Technical behaviour', '## Limits', '## IoC / Evidence'])
        self.assertNotIn('An unsupported narrative claim', report.text)
        self.assertNotIn('proof_chain', report.text)
        for text in ('Original broad objective', 'Full behavioural explanation', 'An unsupported narrative claim', 'proof_chain'):
            self.assertIn(text, report.dossier_text)
        for text in ('https://one.invalid/a', 'RESOLVED_EXACT', rt.source.sha256,
                     rt.ioc_checks()[0]['evidence_id'], 'js_destination_proof'):
            self.assertIn(text, report.text)
        self.assertEqual(rt.messages, before)
        store = SessionStore(rt.evidence_store.root.parent/'session.json')
        store.save(messages=[], workdir=rt.workspace.root, model='fixture', base_url='fixture', analysis_report=asdict(report))
        saved = json.loads(store.path.read_text())['analysis_reports'][0]
        self.assertEqual(saved['text'], report.text)
        self.assertEqual(saved['dossier_text'], report.dossier_text)

    def test_exact_malformed_url_operand_does_not_crash_report(self):
        rt = self.runtime("location.href='https://[invalid/a';")
        run = rt.run_autonomous('Analyse.')
        self.assertEqual(run.model_calls, 0)
        self.assertIn('https://[invalid/a', run.final_report.text)
        self.assertIn('"hostname_from_URL": null', run.final_report.text)
