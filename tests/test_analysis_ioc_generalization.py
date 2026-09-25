"""Sample-independent fixtures for the supported static destination class.

These tests contain no canonical corpus, filename, decoder names or values.
All source strings are inert: only the production static producer is invoked.
"""
import hashlib
import tempfile
import unittest
from pathlib import Path

from orbit.runtime.analysis_controller import AnalysisController, OPEN, BLOCKED
from orbit.runtime.analysis_ioc_proof import destinations
from orbit.runtime.analysis_runtime import AnalysisRuntime, acquire_analysis_source
from orbit.runtime.evidence import EvidenceStore
from tests.test_analysis_runtime import ScriptedBackend


def encoded_fixture(value, key, separator, prefix):
    """Alpha-renamed helpers/properties/locals, independently encoded constants."""
    data = separator.join(str(ord(char) ^ key) for char in value)
    p = prefix
    return f'''<a id="{p}Anchor">safe fixture</a><script>
function {p}Pack({p}Text) {{return {{
  {p}Mask:{key}, {p}Size:{p}Text.split('{separator}').length,
  {p}Items:{p}Text.split('{separator}')}};}}
function {p}Fold({p}Fn,{p}Input,{p}Result) {{
 var {p}Cursor=0;
 while({p}Cursor<{p}Fn({p}Input).{p}Size){{
 {p}Result+=String.fromCharCode((+{p}Fn({p}Input).{p}Items[{p}Cursor])^{p}Fn({p}Input).{p}Mask);
 {p}Cursor++;
 }}return {p}Result;
}}
var {p}Data='{data}';
function {p}Target(){{var {p}Initial='';return {p}Fold({p}Pack,{p}Data,{p}Initial);}}
var {p}Element=document.getElementById('{p}Anchor');
{p}Element.onclick=function({p}Event){{{p}Event.target.href={p}Target();}};
</script>'''


class GeneralizationTests(unittest.TestCase):
    def runtime(self, source):
        directory = tempfile.TemporaryDirectory(prefix='orbit-generic-ioc-')
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        path = root / 'arbitrary-input.html'
        path.write_bytes(source.encode('utf-8'))
        runtime = AnalysisRuntime(
            backend=ScriptedBackend(plan_questions=[]),
            source=acquire_analysis_source(path, root / 'snapshot'),
            evidence_store=EvidenceStore(root / 'evidence'))
        self.addCleanup(runtime.close)
        return runtime

    def assert_exact(self, source, value, roles):
        runtime = self.runtime(source)
        checks = runtime.ioc_checks()
        self.assertEqual(len(checks), 1)
        check = checks[0]
        self.assertEqual((check['state'], check['value']), ('RESOLVED_EXACT', value))
        self.assertTrue(set(roles) <= {link[0] for link in check['proof_chain']})
        raw = source.encode('utf-8')
        self.assertEqual(check['source_sha256'], hashlib.sha256(raw).hexdigest())
        for role, start, end, digest in check['proof_chain']:
            self.assertLess(start, end, role)
            self.assertEqual(hashlib.sha256(raw[start:end]).hexdigest(), digest)
        self.assertEqual(runtime.evidence_store.reattest_exact(check['evidence_id']), value)
        self.assertIn(value, {item.value for item in runtime.canonical_indicators()})
        controller = AnalysisController()
        runtime._ioc_register(controller)
        self.assertEqual(controller.order, [])
        self.assertEqual(runtime.model_calls, 0)
        self.assertEqual(runtime.actions_executed, 0)

    def test_literals_concatenations_and_pure_functions(self):
        cases = [
            ("location.href='https://alpha.invalid/v2?q=7';", 'https://alpha.invalid/v2?q=7', {'literal'}),
            ("var base='http://';var tail='beta.invalid/data';location.href=base+tail;", 'http://beta.invalid/data', {'binding', 'concat'}),
            ("function routeQ(){var prefixQ='https://';return prefixQ+'gamma.invalid/end';}window.location.href=routeQ();", 'https://gamma.invalid/end', {'wrapper', 'call'}),
        ]
        for source, value, roles in cases:
            with self.subTest(value=value):
                self.assert_exact(source, value, roles | {'sink', 'effect_scope'})

    def test_decoder_alpha_renaming_constants_and_delimiters(self):
        for key, separator, prefix, value in [
            (0, '|', 'north', 'https://north.invalid/zero'),
            (7, ';', 'south', 'http://south.invalid/seven?q=2'),
            (93, ':', 'east', 'https://east.invalid/ninety-three'),
            (4095, ',', 'west', 'https://west.invalid/large'),
        ]:
            with self.subTest(key=key, prefix=prefix):
                self.assert_exact(encoded_fixture(value, key, separator, prefix), value,
                                  {'decoder', 'provider', 'decode_call', 'sink', 'effect_scope'})

    def test_derived_domain_and_ip(self):
        for source, value in [
            ("var suffixA='.invalid';location.hostname='delta'+suffixA;", 'delta.invalid'),
            ("function numericHost(){return '198.51.100.'+'23';}location.host=numericHost();", '198.51.100.23'),
        ]:
            with self.subTest(value=value):
                self.assert_exact(source, value, {'sink', 'effect_scope', 'concat'})

    def test_ambiguous_and_partial_local_proofs_remain_open(self):
        cases = [
            "function routeZ(){if(navigator.userAgent){return 'https://one.invalid';}else{return 'https://two.invalid';}}location.href=routeZ();",
            "function joinZ(){return ['https://','three.invalid'].join('');}location.href=joinZ();",
            encoded_fixture('https://four.invalid/', 29, ',', 'amber').replace('amberCursor++;', 'amberCursor+=2;'),
            "location.href='https://'+absentLocalHelper();",
        ]
        for source in cases:
            with self.subTest(source=source):
                runtime = self.runtime(source)
                checks = runtime.ioc_checks()
                self.assertTrue(checks)
                self.assertTrue(all(c['state'] == 'OPEN' and c['value'] is None
                                    and not c['proof_chain'] for c in checks))
                controller = AnalysisController()
                runtime._ioc_register(controller)
                self.assertEqual(controller.states['IOC'].status, OPEN)
                self.assertTrue(runtime._ioc_pending(controller))
                self.assertFalse(controller.exhausted)

    def test_runtime_dependent_values_are_blocked_with_reason(self):
        for source in [
            'location.href=navigator.userAgent;',
            "var valueZ=window.localStorage.getItem('user-choice');location.href=valueZ;",
        ]:
            with self.subTest(source=source):
                runtime = self.runtime(source)
                check = runtime.ioc_checks()[0]
                self.assertEqual(check['state'], 'BLOCKED')
                self.assertIsNone(check['value'])
                self.assertEqual(check['proof_chain'], [])
                self.assertIn('runtime input', check['reason'])
                controller = AnalysisController()
                runtime._ioc_register(controller)
                self.assertEqual(controller.states['IOC'].status, BLOCKED)

    def test_unconnected_values_do_not_acquire_sink_proofs(self):
        for source in [
            "var unrelated='https://unused.invalid/value';",
            "var address='203.0.113.81';var domain='unused.invalid';",
            "var scheme='https://';",
        ]:
            with self.subTest(source=source):
                self.assertEqual(destinations(source), [])
                runtime = self.runtime(source)
                self.assertEqual(runtime.ioc_checks(), [])
                self.assertFalse(any(s.kind == 'js_destination_proof' for s, _r in runtime.transform_stages))


if __name__ == '__main__':
    unittest.main()
