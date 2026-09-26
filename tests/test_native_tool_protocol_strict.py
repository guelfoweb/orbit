"""Real vendored PEG/renderer negative and positive gates, without weights/inference."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'src/orbit/native_llama/vendor'
SOURCE = VENDOR / 'source/llama.cpp'


def schema(name, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': 'Inert fixture.',
            'parameters': {'type': 'object', 'properties': properties,
                           'required': required, 'additionalProperties': False}}}


TOOLS = [schema('inspect_text', {'text': {'type': 'string'}}, ['text']),
         schema('batch', {'values': {'type': 'array', 'items': {'type': 'number'}},
                          'enabled': {'type': 'boolean'}, 'note': {'type': ['string', 'null']}},
                ['values', 'enabled', 'note']),
         schema('record', {'data': {'type': 'object', 'properties': {'count': {'type': 'integer'}},
                                    'required': ['count'], 'additionalProperties': False}}, ['data'])]


def wire(name='inspect_text', arguments=None):
    return '<tool_call><function='+name+'>'+json.dumps(
        {'text': 'safe'} if arguments is None else arguments, ensure_ascii=False)+'</function></tool_call>'


def xml(text='safe'):
    return '<tool_call>\n<function=inspect_text>\n<parameter=text>\n'+text+'\n</parameter>\n</function>\n</tool_call>'


class NativeToolProtocolStrictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('c++') or not (VENDOR / 'lib/libllama-common.so').exists():
            raise unittest.SkipTest('native build/compiler required for real PEG tests')
        cls.tmp = tempfile.TemporaryDirectory(prefix='orbit-protocol-test.')
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.probe = Path(cls.tmp.name) / 'probe'
        cmd = ['c++', '-std=c++17', str(ROOT/'tests/native/tool_protocol_probe.cpp'),
               '-I'+str(SOURCE/'common'), '-I'+str(SOURCE/'include'),
               '-I'+str(SOURCE/'ggml/include'), '-I'+str(SOURCE/'vendor'),
               '-L'+str(VENDOR/'lib'), '-Wl,-rpath,'+str(VENDOR/'lib'),
               '-lllama-common', '-lllama', '-o', str(cls.probe)]
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.returncode:
            raise RuntimeError(build.stderr)
        cls.templates = {
            'json': (ROOT/'tests/fixtures/mimo26_chat_template.jinja').read_text(),
            'xml': (ROOT/'tests/fixtures/ornith15_chat_template.jinja').read_text(),
        }
        # Same Qwen XML contract without any reasoning capability/prefill.
        cls.templates['coder'] = """{# XML protocol: <tool_call> <function= <parameter= #}
{%- for message in messages %}{{ '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{% endfor %}
{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}
"""

    def parse(self, text, *, protocol='json', thinking=False, partial=False, tools=None):
        request = {'template': self.templates[protocol], 'text': text,
                   'messages': [{'role': 'user', 'content': 'Offline protocol check.'}],
                   'tools': TOOLS if tools is None else tools,
                   'thinking': thinking, 'partial': partial}
        run = subprocess.run([str(self.probe)], input=json.dumps(request)+'\n',
                             capture_output=True, text=True, check=True)
        return json.loads(run.stdout)

    def reject(self, text, **kwargs):
        parsed = self.parse(text, **kwargs)
        self.assertTrue('error' in parsed or not parsed['calls'], parsed)

    def test_complete_json_three_schemas_preserve_values(self):
        for name, args in [('inspect_text', {'text': '"q" \\ \r\nè 日本語'}),
                           ('batch', {'values': [1, -2.5], 'enabled': False, 'note': None}),
                           ('record', {'data': {'count': 9}})]:
            with self.subTest(name=name):
                p = self.parse(wire(name, args))
                self.assertNotIn('error', p)
                self.assertEqual(json.loads(p['calls'][0]['arguments']), args)

    def test_complete_xml_remains_valid(self):
        p = self.parse(xml(), protocol='xml')
        self.assertNotIn('error', p)
        self.assertEqual(json.loads(p['calls'][0]['arguments']), {'text': 'safe'})

    def test_partial_stream_is_not_final_authority(self):
        for protocol, text in [('json', wire()), ('xml', xml())]:
            for incomplete in (text[:-2], text.split('</function>')[0],
                               text.removesuffix('</tool_call>')):
                with self.subTest(protocol=protocol, incomplete=incomplete):
                    self.parse(incomplete, protocol=protocol, partial=True)
                    self.reject(incomplete, protocol=protocol)

    def test_valid_call_followed_by_incomplete_envelope_rejected(self):
        for protocol, text in [('json', wire()), ('xml', xml())]:
            self.reject(text+'<tool_call><function=', protocol=protocol)
            self.reject(text+'</tool_call>', protocol=protocol)

    def test_reasoning_off_unexpected_blocks_are_not_authority(self):
        for protocol, text in [('json', wire()), ('xml', xml())]:
            for prefix in ('', 'Visible preface. '):
                for close in ('', '</think>'):
                    with self.subTest(protocol=protocol, prefix=prefix, close=close):
                        self.reject(prefix+'<think>Synthetic reasoning '+text+close, protocol=protocol)

    def test_reasoning_off_tools_off_never_exposes_reasoning(self):
        for protocol in ('json', 'xml'):
            for partial in (False, True):
                p = self.parse('<think>Private fixture</think>Visible.', protocol=protocol,
                               tools=[], partial=partial)
                self.assertTrue('error' in p or 'Private fixture' not in p['content'])

    def test_thinking_on_hidden_tool_is_never_a_call(self):
        for protocol, text in [('json', wire()), ('xml', xml())]:
            for close in ('', '</think>Visible.'):
                self.reject('Synthetic reasoning '+text+close, protocol=protocol, thinking=True)

    def test_genuine_tool_after_closed_reasoning_remains_valid(self):
        for protocol, text in [('json', wire()), ('xml', xml())]:
            with self.subTest(protocol=protocol):
                p = self.parse('Synthetic reasoning</think>'+text, protocol=protocol, thinking=True)
                self.assertNotIn('error', p)
                self.assertEqual(len(p['calls']), 1)
                self.assertIn('Synthetic reasoning', p['reasoning'])

    def test_control_delimiters_inside_arguments_are_inert_data(self):
        text = '</function></tool_call><think>inert</think>'
        p = self.parse(wire(arguments={'text': text}))
        self.assertNotIn('error', p)
        self.assertEqual(json.loads(p['calls'][0]['arguments'])['text'], text)
        p = self.parse(xml('<think>inert</think>'), protocol='xml')
        self.assertNotIn('error', p)
        self.assertEqual(json.loads(p['calls'][0]['arguments'])['text'], '<think>inert</think>')

    def test_invalid_function_and_malformed_json_rejected(self):
        for text in (wire('not_a_tool'), wire('bad name'),
                     '<tool_call><function=inspect_text>{"text":}</function></tool_call>'):
            self.reject(text)

    def test_non_thinking_xml_profile_rejects_unexpected_reasoning(self):
        self.reject('<think>Hidden '+xml(), protocol='coder')
        valid = self.parse(xml(), protocol='coder')
        self.assertNotIn('error', valid)
        self.assertEqual(len(valid['calls']), 1)
        inert = self.parse(xml('<think>inert</think>'), protocol='coder')
        self.assertNotIn('error', inert)
        self.assertEqual(json.loads(inert['calls'][0]['arguments'])['text'], '<think>inert</think>')
