"""Required FINISH: real render/grammar/parser with controlled logits, no inference."""
import copy
import ctypes as C
import json
import math
import unittest
from pathlib import Path
from unittest import mock

from orbit.backend.llama_server import LlamaServerBackend, _parse_chat_result
from orbit.native_llama.bindings import LlamaLibrary, ChatBridgeLibrary, LLAMA_LOAD_MODE_MMAP
from orbit.native_llama.chat_bridge import chat_bridge_filename
from orbit.native_llama.events import NativeTimings
from orbit.native_llama.model_registry import resolve_models_dir
from orbit.native_llama.paths import DEFAULT_VENDOR_BUILD_BIN, DEFAULT_VENDOR_LIB_DIR
from orbit.native_server.app import OrbitNativeServer
from orbit.native_server.protocol import openai_chat_response, parse_chat_request
from orbit.runtime.analysis_controller import AnalysisController, ControlError, parse_finish_call
from orbit.runtime.analysis_runtime import FINISH_TOOL_SCHEMA, PLAN_TOOL_SCHEMA
from tests.test_analysis_controller_runtime import _Case, _Model, _question
from tests.test_analysis_finish_budget import ExactFinishBackend
from tests import test_native_qwen_profile as qwen_tests


class ConstraintTransportTests(_Case):
    def test_finish_opt_in_and_repair_use_same_contract_for_count_and_generation(self):
        rt = self._runtime(_Model(plan=[])); rt.constrain_finish = True
        backend = ExactFinishBackend(2404); rt.backend = backend
        original = backend.count_chat_tokens
        choices = []
        def counter(messages, *, tool_choice='auto', **kwargs):
            choices.append(tool_choice)
            return original(messages, **kwargs)
        backend.count_chat_tokens = counter
        for repair in (False, True):
            ms = rt.messages + ([{'role': 'user', 'content': 'Call the control again.'}] if repair else [])
            rt._control_dispatch(ms, FINISH_TOOL_SCHEMA)
            self.assertEqual(backend.chat_calls[-1]['kwargs']['tool_choice'], 'required')
            self.assertEqual(backend.chat_calls[-1]['kwargs']['max_tokens'], 1436)
        self.assertEqual(set(choices), {'required'})
        rt.constrain_finish = False
        rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
        self.assertNotIn('tool_choice', backend.chat_calls[-1]['kwargs'])
        rt.constrain_finish = True; backend.tokens = 100
        rt._control_dispatch(rt.messages, PLAN_TOOL_SCHEMA)
        self.assertNotIn('tool_choice', backend.chat_calls[-1]['kwargs'])
        self.assertEqual(choices[-1], 'auto')

    def test_required_contract_is_preserved_and_unsupported_requests_fail(self):
        payload = {'messages': [{'role': 'user', 'content': 'Control'}],
                   'tools': [FINISH_TOOL_SCHEMA], 'tool_choice': 'required'}
        self.assertEqual(parse_chat_request(payload).tool_choice, 'required')
        for change in ({'tools': []}, {'tool_choice': 'none'}, {'thinking': True},
                       {'artifact_content': True}, {'stop': '</tool_call>'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_chat_request({**payload, **change})
        self.assertEqual(parse_chat_request({'messages': payload['messages']}).tool_choice, 'auto')

    def test_old_server_cannot_ignore_required_contract(self):
        from orbit.backend.llama_server import LlamaServerError
        backend = LlamaServerBackend(base_url="http://old-server", timeout=1)
        with (mock.patch.object(backend, '_is_orbit_native_backend', return_value=True),
              mock.patch.object(backend, '_props_or_empty', return_value={'backend': 'orbit-native'}),
              mock.patch.object(backend, '_post_native_stream') as dispatch):
            with self.assertRaises(LlamaServerError):
                backend.chat_stream([], tools=[FINISH_TOOL_SCHEMA], tool_choice='required',
                                    max_tokens=100, temperature=0, on_delta=lambda _: None)
            with self.assertRaises(LlamaServerError):
                backend.count_chat_tokens([], tools=[FINISH_TOOL_SCHEMA], tool_choice='required')
            dispatch.assert_not_called()


# Test-only mirrors of public llama.h, used solely to supply controlled logits.
class _Token(C.Structure):
    _fields_ = [('id', C.c_int32), ('logit', C.c_float), ('p', C.c_float)]
class _Candidates(C.Structure):
    _fields_ = [('data', C.POINTER(_Token)), ('size', C.c_size_t),
                ('selected', C.c_int64), ('sorted', C.c_bool)]


def wire(status='still_open', summary='Bounded finding: café 世界.', extra=''):
    return ('<tool_call>\n<function=finish_analysis_question>\n<parameter=status>\n'
            + status + '\n</parameter>\n<parameter=answer_summary>\n' + summary
            + '\n</parameter>\n' + extra + '</function>\n</tool_call>')


class NativeConstraintTests(_Case):
    model_relative = "unsloth--Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf"
    @classmethod
    def setUpClass(cls):
        model_path = resolve_models_dir().path / cls.model_relative
        if not model_path.exists():
            raise unittest.SkipTest('local Qwen vocabulary unavailable')
        # The existing in-process bridge tests use the build family when it
        # exists. Share that canonical root; never reset native ownership or
        # fall back from an invalid family to a different loaded runtime.
        runtime = (DEFAULT_VENDOR_BUILD_BIN
                   if (DEFAULT_VENDOR_BUILD_BIN / chat_bridge_filename()).exists()
                   else DEFAULT_VENDOR_LIB_DIR)
        cls.binding = LlamaLibrary(runtime)
        cls.lib = cls.binding.lib
        cls.lib.llama_sampler_apply.argtypes = [C.c_void_p, C.POINTER(_Candidates)]
        cls.lib.llama_sampler_apply.restype = None
        params = cls.lib.llama_model_default_params(); params.vocab_only = True
        params.load_mode = LLAMA_LOAD_MODE_MMAP
        cls.model = cls.lib.llama_model_load_from_file(str(model_path).encode(), params)
        assert cls.model
        cls.vocab = cls.lib.llama_model_get_vocab(cls.model)
        cls.bridge = ChatBridgeLibrary(runtime, runtime / chat_bridge_filename())
        cls.bridge_ctx = cls.bridge.create(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.bridge.free(cls.bridge_ctx)
        cls.lib.llama_model_free(cls.model)

    def client(self):
        client = qwen_tests.NativeQwenProfileTests()._client()
        client.lib = self.binding; client._model = self.model; client._vocab = self.vocab
        client.chat_bridge = self.bridge; client._chat_bridge_context = self.bridge_ctx
        # Shared vocabulary only; clients must not take ownership of class handles.
        self.addCleanup(lambda: setattr(client, '_model', None))
        self.addCleanup(lambda: setattr(client, '_chat_bridge_context', None))
        return client

    def consume(self, client, sampler, text):
        for token in client._tokenize_text(text, add_special=False):
            data = (_Token * 1)(_Token(token, 1.0, 0.0))
            candidates = _Candidates(data, 1, -1, False)
            self.lib.llama_sampler_apply(sampler, C.byref(candidates))
            if not math.isfinite(data[0].logit):
                return False
            self.lib.llama_sampler_accept(sampler, token)
        return True

    def test_native_grammar_all_statuses_roundtrip_and_excludes_other_tools(self):
        client = self.client(); messages = [{'role': 'user', 'content': 'Control'}]
        auto = client.apply_chat_template(messages, tools=[FINISH_TOOL_SCHEMA])
        required = client.apply_chat_template(messages, tools=[FINISH_TOOL_SCHEMA], tool_choice='required')
        self.assertEqual(auto, required)
        for status in ('resolved', 'still_open', 'blocked'):
            for extra in ('', '<parameter=evidence_ids>\n["ev_é"]\n</parameter>\n'):
                text = wire(status, extra=extra)
                sampler = client._create_required_tool_sampler()
                try:
                    self.assertTrue(self.consume(client, sampler, text))
                finally:
                    self.lib.llama_sampler_free(sampler)
                parsed = client._parse_profile_output(text, partial=False)
                args = json.loads(parsed.tool_calls[0]['function']['arguments'])
                self.assertEqual(args['status'], status)
                self.assertEqual(args['answer_summary'], 'Bounded finding: café 世界.')
                parse_finish_call(args)
        for name in ('execute_analysis', 'finish_analysis_questions', 'other'):
            sampler = client._create_required_tool_sampler()
            try:
                self.assertFalse(self.consume(client, sampler, wire().replace('finish_analysis_question', name)))
            finally:
                self.lib.llama_sampler_free(sampler)
        child = {'question': 'Original further question?', 'missing_fact': 'Need more evidence.', 'caused_by_evidence_id': 'ev_x'}
        text = wire('resolved', extra='<parameter=child_question>\n'+json.dumps(child)+'\n</parameter>\n')
        sampler = client._create_required_tool_sampler()
        try:
            self.assertTrue(self.consume(client, sampler, text))
        finally:
            self.lib.llama_sampler_free(sampler)
        args = json.loads(client._parse_profile_output(text, partial=False).tool_calls[0]['function']['arguments'])
        with self.assertRaises(ControlError):
            parse_finish_call(args)  # Grammar is not the completion validator.

    def test_required_grammar_failure_is_explicit(self):
        client = self.client()
        for rendered in ({}, {'tool_choice': 'required', 'grammar_lazy': True, 'grammar': 'root ::= "x"'}):
            client._active_profile_render = rendered
            with self.assertRaises(RuntimeError): client._create_required_tool_sampler()
        with self.assertRaisesRegex(RuntimeError, 'generation prompt'):
            self.bridge.required_sampler(self.vocab, 'root ::= "x"', 'y')
        with self.assertRaises(RuntimeError):
            self.bridge.required_sampler(self.vocab, 'not a grammar', '')

    def test_real_dispatch_server_bridge_sampler_and_next_request_isolation(self):
        rt = self._runtime(_Model(plan=[])); rt.constrain_finish = True
        client = self.client()
        with mock.patch('orbit.native_server.app.safe_native_capability_manifest', return_value={}):
            server = OrbitNativeServer(client=client, model_alias='test-vocab')
        backend = LlamaServerBackend(base_url='http://in-memory', timeout=1, model='test-vocab')
        rt.backend = backend
        backend._props_cache = {'backend': 'orbit-native', 'required_tool_decoding': True}
        native_requests = []
        def post(path, payload):
            if path == '/props': return {'backend': 'orbit-native', 'required_tool_decoding': True}
            self.assertEqual(path, '/tokens/count')
            req = parse_chat_request(payload)
            return server.count_chat_tokens(req.messages, tools=req.tools, thinking=False, tool_choice=req.tool_choice)
        def stream(path, payload, **kwargs):
            native_requests.append(copy.deepcopy(payload))
            return _parse_chat_result(openai_chat_response(server.chat(payload)))
        seen = []
        def generate(prompt, **kw):
            sampler = kw.get('sampler_override'); seen.append(sampler is not None)
            if rt.constrain_finish:
                self.assertIsNotNone(sampler)
                self.assertTrue(self.consume(client, sampler, wire()))
            else:
                self.assertIsNone(sampler)
            kw['on_token'](wire())
            return NativeTimings(len(client.tokenize(prompt)), 44, 0, len(client.tokenize(prompt)), 0, 0)
        with (mock.patch.object(backend, '_post_json', side_effect=post),
              mock.patch.object(backend, '_is_orbit_native_backend', return_value=True),
              mock.patch.object(backend, '_post_native_stream', side_effect=stream),
              mock.patch.object(client, '_ensure_prompt_cache_mode'),
              mock.patch.object(client, 'complete_prompt', side_effect=generate),
              mock.patch.object(client.lib.lib, 'llama_sampler_free', wraps=client.lib.lib.llama_sampler_free) as free):
            result = rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
            self.assertEqual(result.tool_calls[0]['function']['name'], 'finish_analysis_question')
            self.assertEqual(native_requests[-1]['tool_choice'], 'required')
            self.assertFalse(client._session.continuation_ready)
            self.assertEqual(free.call_count, 1)
            rt.constrain_finish = False
            rt._control_dispatch(rt.messages, FINISH_TOOL_SCHEMA)
            self.assertEqual(native_requests[-1]['tool_choice'], 'auto')
            self.assertEqual(seen, [True, False])
            self.assertEqual(free.call_count, 1)

    def test_sampler_freed_after_error_or_cancel(self):
        client = self.client()
        for error in (RuntimeError('decode failed'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                with (mock.patch.object(client, '_ensure_prompt_cache_mode'),
                      mock.patch.object(client, 'complete_prompt', side_effect=error),
                      mock.patch.object(client.lib.lib, 'llama_sampler_free', wraps=client.lib.lib.llama_sampler_free) as free):
                    with self.assertRaises(type(error)):
                        client.complete_chat_text([{'role': 'user', 'content': 'Control'}],
                            tools=[FINISH_TOOL_SCHEMA], thinking=False, tool_choice='required')
                    self.assertEqual(free.call_count, 1)
                    self.assertFalse(client._session.continuation_ready)

    def test_actual_generation_loop_preserves_primed_sampler(self):
        client = self.client()
        client.apply_chat_template([{'role': 'user', 'content': 'Control'}],
                                   tools=[FINISH_TOOL_SCHEMA], tool_choice='required')
        sampler = client._create_required_tool_sampler()
        ids = client._tokenize_text(wire(), add_special=False)
        client._session.ctx_tgt = 1
        client._session.sampler = sampler
        self.addCleanup(lambda: setattr(client._session, 'ctx_tgt', None))
        self.addCleanup(lambda: setattr(client._session, 'sampler', None))
        index = 0
        def controlled_sample(chain, context, position):
            nonlocal index
            token = ids[index]; index += 1
            data = (_Token * 1)(_Token(token, 1.0, 0.0))
            candidates = _Candidates(data, 1, -1, False)
            self.lib.llama_sampler_apply(chain, C.byref(candidates))
            self.assertTrue(math.isfinite(data[0].logit), 'generation reset the primed grammar')
            self.lib.llama_sampler_accept(chain, token)
            return token
        try:
            with (mock.patch.object(self.lib, 'llama_sampler_sample', side_effect=controlled_sample),
                  mock.patch.object(self.lib, 'llama_decode', return_value=0),
                  mock.patch('orbit.native_llama.client.kv_diag_enabled', return_value=False)):
                generated, _, cancelled = client._generate_from_current_context(
                    max_tokens=len(ids), sampler_override=sampler)
            self.assertEqual(generated, len(ids)); self.assertFalse(cancelled)
            self.assertEqual(client.last_committed_generated_tokens, ids)
        finally:
            self.lib.llama_sampler_free(sampler)


class OrnithNativeConstraintTests(NativeConstraintTests):
    model_relative = "ornith-ai--Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf"

    def client(self):
        from dataclasses import replace
        from tests.test_ornith_route_prefix import _profile
        client = super().client()
        client.model_profile = _profile()
        client.config = replace(client.config, ornith_analysis_prefix_reuse_enabled=True)
        client._model_metadata_identity = {'general.file_type': '15'}
        return client

    def test_prefix_reference_probes_restore_required_contract(self):
        client = self.client()
        messages = [{'role': 'system', 'content': 'Static analysis control.'},
                    {'role': 'user', 'content': 'Evidence observation. ' * 200}]
        prompt = client.apply_chat_template(messages, tools=[FINISH_TOOL_SCHEMA], tool_choice='required')
        with mock.patch.object(client.chat_bridge, 'render', wraps=client.chat_bridge.render) as render:
            client._qwen_route_anchor_plan_for_prompt(messages, tools=[FINISH_TOOL_SCHEMA],
                thinking=False, prompt=prompt, analysis_lineage=True, tool_choice='required')
            self.assertGreaterEqual(render.call_count, 2)
            self.assertEqual(render.call_args.kwargs['tool_choice'], 'required')
        self.assertEqual(client._active_profile_render['prompt'], prompt)
        sampler = client._create_required_tool_sampler()
        try:
            self.assertTrue(self.consume(client, sampler, wire()))
        finally:
            self.lib.llama_sampler_free(sampler)
