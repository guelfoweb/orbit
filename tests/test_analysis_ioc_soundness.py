"""Closed DOM premises and positive discovery, using inert independent sources."""
import hashlib
import json
import unittest
from pathlib import Path

from orbit.runtime import analysis_ioc_proof as proof
from orbit.runtime.analysis_controller import AnalysisController, BLOCKED
from tests import test_analysis_ioc_objectives as fixtures


CASES = json.loads((Path(__file__).parent / 'fixtures/analysis_ioc_soundness.json').read_text())


class AuditGate(unittest.TestCase):
    runtime = fixtures.ObjectiveTests.runtime


def audit_value(case):
    def test(self):
        runtime = self.runtime(case['source'])
        raw = case['source'].encode()
        for check in runtime.ioc_checks():
            if check['state'] == 'RESOLVED_EXACT':
                self.assertIn(check['value'], case['exact'])
                self.assertEqual(runtime.evidence_store.reattest_exact(check['evidence_id']), check['value'])
            for _role, lo, hi, digest in check['proof_chain']:
                self.assertEqual(hashlib.sha256(raw[lo:hi]).hexdigest(), digest)
    return test


def audit_closure(case):
    def test(self):
        runtime = self.runtime(case['source'])
        controller = AnalysisController()
        runtime._ioc_register(controller)
        self.assertFalse(runtime._ioc_closed(controller))
        if 'IOC' in controller.states:
            controller.states['IOC'].status = BLOCKED
            controller.states['IOC'].reason = 'reached the 2-action limit for one question'
            self.assertFalse(runtime._ioc_closed(controller))
    return test


for case in CASES:
    if case['scope'] != 'manual':
        setattr(AuditGate, 'test_' + case['id'] + '_no_false_exact', audit_value(case))
    if case['forbidden_close']:
        setattr(AuditGate, 'test_' + case['id'] + '_discovery', audit_closure(case))


class SoundnessTests(unittest.TestCase):
    runtime = fixtures.ObjectiveTests.runtime

    def test_static_conditional_excludes_dead_destination(self):
        for condition, expected in [('true', 'https://a.invalid/'), ('false', 'https://b.invalid/')]:
            code = "if("+condition+"){location.href='https://a.invalid/';}else{location.href='https://b.invalid/';}"
            self.assertEqual([d.value for d in proof.destinations(code) if d.state == 'RESOLVED_EXACT'], [expected])

    def test_unknown_alternative_is_not_two_exact_destinations(self):
        code = "if(navigator.userAgent){location.href='https://a.invalid/';}else{location.href='https://b.invalid/';}"
        self.assertFalse(any(d.state == 'RESOLVED_EXACT' for d in proof.destinations(code)))

    def test_positive_discovery_and_complete_early_stop(self):
        for source in ["location.href='https://a.invalid/';", "location.host=navigator.userAgent;",
                       '<a id="live"></a><script>document.getElementById("live").href="https://a.invalid/";</script>']:
            with self.subTest(source=source):
                runtime = self.runtime(source)
                self.assertTrue(runtime._ioc_discovery_complete())
                self.assertTrue(runtime._ioc_closed(None))
                run = runtime.run_autonomous('Identify destinations.', cover=False)
                self.assertEqual((run.model_calls, run.actions_executed), (0, 0))
                self.assertTrue(run.final_report.text.rstrip().endswith('```'))
                self.assertIn('## IoC / Evidence', run.final_report.text)

    def test_every_unsupported_fixture_is_discovery_incomplete(self):
        for case in CASES:
            if case['id'] in ('01','02','03','04','05','07','11','12','13','16','17','23','24','26','27','28','29','31','32','33'):
                with self.subTest(case=case['id']):
                    runtime = self.runtime(case['source'])
                    self.assertFalse(runtime._ioc_discovery_complete())
                    self.assertTrue(runtime._ioc_scan_incomplete)

    def test_positive_certificate_cannot_follow_changed_inventory(self):
        runtime = self.runtime("location.href='https://a.invalid/';")
        self.assertTrue(runtime._ioc_discovery_complete())
        runtime.ioc_objectives = ()
        self.assertFalse(runtime._ioc_discovery_complete())

    def test_incomplete_discovery_keeps_exact_evidence_and_normal_controller(self):
        runtime = self.runtime('<img src="https://other.invalid/"><script>location.href="https://a.invalid/";</script>')
        check = runtime.ioc_checks()[0]
        self.assertEqual(check['state'], 'RESOLVED_EXACT')
        self.assertFalse(runtime._ioc_closed(None))
        run = runtime.run_autonomous('Identify destinations.', cover=False, finalize=False, max_model_calls=1)
        self.assertEqual(run.plan_calls, 1)
        self.assertNotEqual(run.stop_reason, 'known network destination objectives closed')
        self.assertEqual(runtime.evidence_store.reattest_exact(check['evidence_id']), 'https://a.invalid/')

    def test_html_script_view_must_be_classic_and_byte_faithful(self):
        for source in [
            '<script language="vbscript">location.href="https://ghost.invalid/";</script>',
            '<script nomodule>location.href="https://ghost.invalid/";</script>',
            '<script>var marker="<!--<script>";location.href="https://ghost.invalid/";</script>',
            '<script>location.href="https://host.invalid/a\x00b";</script>',
        ]:
            with self.subTest(source=source):
                result = proof.scan_destinations(source)
                self.assertFalse(result.complete)
                self.assertFalse(any(d.state == 'RESOLVED_EXACT' for d in result.objectives))
        # The HTML tokenizer limitation must not invent a new JS restriction.
        self.assertTrue(proof.scan_destinations('var marker="<!--<script>";location.href="https://live.invalid/";').complete)

    def test_missing_or_foreign_certificate_cannot_authorize_stop(self):
        from dataclasses import replace
        for change in ('missing', 'source', 'incomplete'):
            with self.subTest(change=change):
                runtime = self.runtime("location.href='https://live.invalid/';")
                scan = runtime._ioc_discovery
                runtime._ioc_discovery = (None if change == 'missing' else
                    replace(scan, source_sha256='0' * 64) if change == 'source' else
                    replace(scan, complete=False, reason='unclassified source'))
                self.assertTrue(runtime._known_ioc_objectives_closed(None))
                self.assertFalse(runtime._ioc_closed(None))

    def test_unknown_guards_never_certify_complete_discovery(self):
        scan = proof.scan_destinations("if(navigator.userAgent){location.href='https://conditional.invalid/';}")
        self.assertFalse(scan.complete)
        self.assertTrue(scan.reason)

    def test_dom_membership_is_conservative_for_unproved_tree_structure(self):
        for source in [
            '<a id="one"><a id="two"></a></a>',
            '<select><a id="one"></a></select>',
            '<a id="one"/>',
            '<p><div><a id="one"></a></div></p>',
            '<div><a id="one"></a></span>',
        ]:
            source += '<script>document.getElementById("one").href="https://ghost.invalid/";</script>'
            with self.subTest(source=source):
                scan = proof.scan_destinations(source)
                self.assertFalse(scan.complete)
                self.assertFalse(any(d.state == 'RESOLVED_EXACT' for d in scan.objectives))

    def test_attribute_context_must_not_invent_a_dom_id(self):
        for raw_id, incorrect_id in [('x&notit;', 'x¬it;'), ('x&copycat', 'x©cat'), ('x&not=foo', 'x¬=foo')]:
            source = (f'<a id="{raw_id}"></a><script>document.getElementById("{incorrect_id}")'
                      '.href="https://ghost.invalid/";</script>')
            with self.subTest(raw_id=raw_id):
                scan = proof.scan_destinations(source)
                self.assertFalse(scan.complete)
                self.assertFalse(any(d.state == 'RESOLVED_EXACT' for d in scan.objectives))
