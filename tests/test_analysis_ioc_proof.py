"""Inert-source positive and adversarial gates for the closed destination proof."""
import unittest
from pathlib import Path
from orbit.runtime.analysis_ioc_proof import destinations


def fixture(host='example.invalid/download', key=17):
    data = ','.join(str(ord(c) ^ key) for c in host)
    return '''<html><a id="download">Download</a><script>
function parts(s) {return {values:s.split(','),count:s.split(',').length,key:(+('%s'))};}
function decode(parts,s,out) {var i=0; while(i<parts(s).count){out+=String.fromCharCode((+parts(s).values[i])^parts(s).key);i++;}return out;}
var encoded='%s';
function destination(){var out='';return decode(parts,encoded,out);}
var link=document.getElementById('download');
link.addEventListener('mouseover',function(event){var scheme='https:'; event.target.href=scheme+'//'+destination();});
</script></html>''' % (key, data)


class ProofTests(unittest.TestCase):
    def resolved(self, source):
        return [d for d in destinations(source) if d.state == 'RESOLVED_EXACT']

    def test_complete_literal_concat_and_pure_local_function(self):
        for code in [
            "location.href='https://example.invalid/a';",
            "location.href='https://'+'example.invalid/a';",
            "function host(){return 'example.invalid/a';} location.href='https://'+host();",
            fixture(),
        ]:
            with self.subTest(code=code):
                result = self.resolved(code)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].value, 'https://example.invalid/a' if code != fixture() else 'https://example.invalid/download')
                self.assertTrue(any(link[0] == 'sink' for link in result[0].links))
                self.assertTrue(any(link[0] == 'literal' for link in result[0].links))

    def test_five_frozen_counterexamples(self):
        cases = [
            'var scheme="https://"; function f({scheme}){location.href=scheme+"example.invalid";} f({scheme:"ftp://"});',
            'var scheme="https://"; var obj={f(scheme){location.href=scheme+"example.invalid";}}; obj.f("ftp://");',
            fixture().replace('var link=', 'var f=eval; f(\'encoded="0"\'); var link='),
            fixture().replace('var link=', 'var f=Function; f(\'encoded="0"\')(); var link='),
            fixture(key=21).replace("(+('21'))", '021'),
        ]
        for code in cases:
            with self.subTest(code=code):
                self.assertTrue(destinations(code))
                self.assertEqual(self.resolved(code), [])

    def test_unknown_effects_scopes_and_receivers_do_not_prove_a_destination(self):
        for code in [
            "var obj={};obj.href='https://example.invalid/a';",
            "function h(){if(user){return 'a.invalid';}else{return 'b.invalid';}}location.href='https://'+h();",
            "var scheme='https://';scheme&&='ftp://';location.href=scheme+'example.invalid';",
            "var scheme='https://';scheme>>=0;location.href=scheme+'example.invalid';",
            "var scheme='https://';if(false){var scheme='ftp://';}location.href=scheme+'example.invalid';",
            "function host(){return\n 'example.invalid';}location.href='https://'+host();",
            "var document=other;var a=document.getElementById('a');a.href='https://example.invalid';",
            "function f(){var hidden=evil();return 'example.invalid';}location.href='https://'+f();",
            '<script src="remote.js"></script>'+fixture(),
            '<body onload="evil()">'+fixture(),
            fixture().replace('var encoded=', 'var encoded=').replace('var link=', "String.fromCharCode=other;var link="),
            fixture().replace('i<parts(s).count', 'i<unknown'),
        ]:
            with self.subTest(code=code):
                self.assertEqual(self.resolved(code), [])

    def test_unproved_sink_rhs_effect_prevents_every_destination_proof(self):
        for expression in ['eval("scheme=\'ftp://\'")', 'Function("scheme=\'ftp://\'")()']:
            source='var scheme="https://";location.href='+expression+';location.href=scheme+"example.invalid";'
            self.assertEqual(self.resolved(source), [])

    def test_runtime_value_is_blocked_not_guessed(self):
        result = destinations('location.href=navigator.userAgent;')
        self.assertEqual(result[0].state, 'BLOCKED')
        self.assertIn('runtime input', result[0].reason)

    def test_unsupported_local_decoder_stays_open(self):
        result = destinations("function f(){return ['a.invalid'].join('');}location.href='https://'+f();")
        self.assertEqual(result[0].state, 'OPEN')
        self.assertIsNone(result[0].value)

    def test_unrelated_schemes_domain_and_ip_are_not_destinations(self):
        self.assertEqual(destinations("var x='https://';var host='example.invalid';var ip='192.0.2.4';"), [])
        self.assertEqual(destinations("var text='location.href=\"https://example.invalid\";';"), [])
        self.assertEqual(destinations("location.href='relative/page';"), [])

    def test_closed_html_and_js_semantics_reject_false_sinks(self):
        for source in [
            '<script type="text/plain" type="text/javascript">location.href="https://example.invalid/";</script>',
            '<template><script>location.href="https://example.invalid/";</script></template>',
            '<noscript><script>location.href="https://example.invalid/";</script></noscript>',
            'function true(){return "example.invalid";}location.href="https://"+true();',
            '<div id="x"></div><script>var a=document.getElementById("x");a.href="https://example.invalid/";</script>',
            fixture().replace('count:', '__proto__:').replace('.count', '.__proto__'),
            fixture().replace('key:', '__proto__:').replace('.key', '.__proto__'),
        ]:
            with self.subTest(source=source): self.assertEqual(self.resolved(source), [])

    def test_proof_ranges_are_original_utf8_bytes(self):
        source='<p>è</p>\r\n<script>\r\nlocation.href="https://example.invalid/";\r\n</script>'
        result=self.resolved(source)[0]
        import hashlib
        raw=source.encode()
        for role, lo, hi, digest in result.links:
            self.assertEqual(hashlib.sha256(raw[lo:hi]).hexdigest(),digest)
        self.assertEqual(raw[result.start:result.start+5],b'.href')

    def test_lexical_ownership_and_line_terminators_are_closed(self):
        for source in [
            '<a id="a"></a><script>var a=document.getElementById("a");a.onclick=function(event){var event="no";event.target.href="https://example.invalid/";};</script>',
            '<a id="a"></a><script>var a=document.getElementById("a");a.onclick=function(event){const event="no";event.target.href="https://example.invalid/";};</script>',
            'function h(){return\u2028"example.invalid";}location.href="https://"+h();',
            'var scheme="https://";//comment\u2028scheme="ftp://";\nlocation.href=scheme+"example.invalid";',
            'function h(){return\u2029"example.invalid";}location.href="https://"+h();',
            'const x="one";const x="two";location.href="https://example.invalid";',
            '"use strict";var eval="one";location.href="https://example.invalid";',
        ]:
            with self.subTest(source=source): self.assertEqual(self.resolved(source), [])

    def test_script_and_dom_declaration_order(self):
        for source in [
            '<script>var a=document.getElementById("a");a.href="https://example.invalid";</script><a id="a"></a>',
            '<script>location.href="https://"+host();</script><script>function host(){return "example.invalid";}</script>',
            '<script>location.href=</script><script>"https://example.invalid";</script>',
            'location.href=host();var x="https://example.invalid";function host(){return x;}',
            'function host(){return x;}location.href=host();var x="https://example.invalid";',
            fixture().replace('i++;', 'i\n++;'),
            fixture().replace('i++;', 'i/*\n*/++;'),
        ]:
            with self.subTest(source=source): self.assertEqual(self.resolved(source), [])
        self.assertEqual(len(self.resolved('location.href="https://"+host();function host(){return "example.invalid";}')),1)
        self.assertEqual(len(self.resolved('location.href=host();function host(){var x="https://example.invalid";return x;}')),1)

    def test_hostile_numbers_and_dependency_dag_are_bounded(self):
        source='var n='+'1'*5000+';location.href="https://example.invalid/";'
        self.assertEqual(self.resolved(source),[])
        source="var x0='';"+''.join(f'var x{i}=x{i-1}+x{i-1};' for i in range(1,30))
        source+='location.href="https://example.invalid/"+x29;'
        self.assertEqual(self.resolved(source),[])

    def test_canonical_html_proof_retains_both_sinks(self):
        path = Path(__file__).resolve().parents[1] / 'workdir/samples/IT5440738233991.html'
        if not path.exists(): self.skipTest('canonical local fixture absent')
        source=path.read_bytes().decode('utf-8')
        result = self.resolved(source)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0].value,result[1].value)
        self.assertEqual(len({d.links[-1] for d in result}),2)
        self.assertEqual(result[0].value,'https://ytrFS9dtI3BlLhgyBsMaCWj3rJQ2ik7gg3lC.debbydot.com/aAey404bxI?files=IT5440738233991.html,IT5440738233991.zip,KeoIjRLNS0J4a.js')

if __name__ == '__main__': unittest.main()
