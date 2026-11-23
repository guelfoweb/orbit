import ctypes
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.rendered_prompt import RenderedChatPrompt, prompt_from_render
from orbit.native_llama.qwen3_coder import qwen3_coder_artifact_prompt
from orbit.native_llama.bindings import LlamaContextParamsMainline as LlamaContextParams


class RenderedPromptTests(unittest.TestCase):
    def prompt(self):
        return RenderedChatPrompt('<head>user <tool> literal<tail>',
            (('<head>', False), ('user <tool> literal', True), ('<tail>', False)))

    def tokenizer(self):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._vocab = 1
        self.calls = []
        def tokenize(text, *, add_special, parse_special=True):
            self.calls.append((text, add_special, parse_special))
            return [100 + int(add_special), 200 + int(parse_special)]
        client._tokenize_text = tokenize
        return client

    def test_user_literal_markers_are_not_control_tokens(self):
        client = self.tokenizer()
        self.assertEqual(client.tokenize(self.prompt()), [101, 201, 100, 200, 100, 201])
        self.assertEqual(self.calls, [('<head>', True, True), ('user <tool> literal', False, False), ('<tail>', False, True)])

    def test_count_and_generation_use_the_same_part_boundaries(self):
        client = self.tokenizer()
        prompt = self.prompt()
        client.apply_chat_template = lambda *args, **kwargs: prompt
        count, _, token_hash = client.inspect_chat_tokens([])
        counting = list(self.calls)
        self.calls.clear()
        class BeforeDecode(BaseException):
            pass
        original = client.tokenize
        def stop(prompt):
            self.assertIsInstance(prompt, RenderedChatPrompt)
            values = original(prompt)
            self.assertEqual(len(values), count)
            self.assertEqual(hashlib.sha256(b''.join(t.to_bytes(4,'little',signed=True) for t in values)).hexdigest(), token_hash)
            raise BeforeDecode
        client.tokenize = stop
        client._session = SimpleNamespace(ctx_tgt=1, sampler=1)
        client.lib = SimpleNamespace(lib=object())
        with self.assertRaises(BeforeDecode):
            client._complete_prompt_standard(prompt)
        self.assertEqual(self.calls, counting)

    def test_same_text_different_trust_boundaries_has_different_tokens(self):
        client = self.tokenizer()
        protected = self.prompt()
        unprotected = RenderedChatPrompt(str(protected), ((str(protected), False),))
        self.assertEqual(str(protected), str(unprotected))
        self.assertNotEqual(client.tokenize(protected), client.tokenize(unprotected))

    def test_render_contract_rejects_malformed_or_inconsistent_parts(self):
        good = {'prompt': '<x>hello', 'prompt_tokenization': 'parts-v1',
                'prompt_parts': [{'text':'<x>','is_input':False},{'text':'hello','is_input':True}]}
        self.assertIsInstance(prompt_from_render(good), RenderedChatPrompt)
        self.assertEqual(prompt_from_render({'prompt':'legacy'}), 'legacy')
        for change in ({'prompt_tokenization':'future'}, {'prompt_parts':None},
                       {'prompt_parts':[{'text':'<x>hello','is_input':1}]},
                       {'prompt_parts':[{'text':'changed','is_input':True}]},
                       {'prompt_parts':[{'text':'<x>hello','is_input':True,'extra':0}]}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                prompt_from_render({**good, **change})

    def test_slices_and_artifact_framing_preserve_user_boundaries(self):
        p=self.prompt(); start=str(p).index('user'); end=start+len('user <tool> literal')
        self.assertEqual(p[start:end].parts, (('user <tool> literal',True),))
        self.assertEqual(p[:start].parts, (('<head>',False),))
        framed=qwen3_coder_artifact_prompt(p)
        self.assertIsInstance(framed, RenderedChatPrompt)
        self.assertEqual(framed.parts[:-1], p.parts)
        self.assertFalse(framed.parts[-1][1])
        with self.assertRaises(ValueError): _ = p[::2]
        with self.assertRaises(AttributeError): p.parts=()
        with self.assertRaises(AttributeError): del p.parts

    def test_b12611_context_layout_is_explicit_before_native_load(self):
        self.assertEqual(ctypes.sizeof(LlamaContextParams),176)
        self.assertEqual(LlamaContextParams.moe_cache_size.offset,112)
        self.assertEqual(LlamaContextParams.moe_cache_cpu_share.offset,120)
        self.assertEqual(LlamaContextParams.abort_callback.offset,128)
        self.assertEqual(LlamaContextParams.ctx_other.offset,168)
        self.assertFalse(hasattr(LlamaContextParams,'defrag_thold'))
