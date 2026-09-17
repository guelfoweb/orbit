"""Qwen38 uses existing synchronous Qwen prewarm at unchanged decode calls.

These deterministic lifecycle tests model full KV + recurrent state. Native
byte/logit/greedy equivalence and the segmentation controls are recorded in
QWEN38-ALIGNED-ROUTE-PREWARM-25; token counters alone are not that proof.
"""
from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from orbit.native_llama.bindings import LLAMA_LOAD_MODE_MMAP, LLAMA_LAZY_MODE_ON
from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.model_profiles import NativeModelProfile, QWEN38_FLASH_NEXT_PROFILE_ID
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.qwen_route_prefix import derive_qwen_route_prefix_spec
from orbit.native_server import app
from tests.test_post_final_route_shadow import ShadowLib, _LibHolder
from tests.test_qwen38_rolling_route_reuse import fold

SYSTEM = "s" * 1100

def tokenize(text):
    return [ord(c) for c in text]

class Bridge:
    def render(self, _context, messages, _tools, *, thinking):
        text = str(messages[0]["content"]) + "\n<user>"
        for m in messages[1:]:
            text += f"<{m['role']}>{m['content']}</{m['role']}>"
        return {"prompt": text + "<assistant>", "generation_prompt": "<assistant>"}

    def free(self, _context):
        pass

class DerivationTests(unittest.TestCase):
    def derive(self, length, user="hello", alignment=64):
        render = lambda u: "s" * length + u + "<assistant>"
        prompt = render(user)
        return derive_qwen_route_prefix_spec(system_prompt="s" * length,
            full_prompt=prompt, full_tokens=tokenize(prompt), render_reference=render,
            tokenize=tokenize, decode_alignment=alignment)

    def test_actual_token_lcp_selects_longest_call_boundary(self):
        for length in (63, 64, 95, 960, 1023, 1024, 1048, 1088, 1100):
            for user in ("hi", "France?", "東京", "", " leading"):
                spec, reason = self.derive(length, user)
                if length < 64:
                    self.assertIsNone(spec)
                else:
                    self.assertIsNone(reason)
                    self.assertEqual(len(spec.prefix_tokens), length // 64 * 64)
                    self.assertEqual(spec.invariant_token_count, length)
                    self.assertEqual(spec.prefix_tokens, tuple(tokenize("s" * (length // 64 * 64))))

    def test_alignment_and_short_actual_prompt_fail_closed(self):
        self.assertEqual(self.derive(1048, alignment=0)[1], "invalid_decode_alignment")
        spec, reason = derive_qwen_route_prefix_spec(system_prompt="system",
            full_prompt="short", full_tokens=[1], render_reference=lambda u: "s" * 1048 + u,
            tokenize=tokenize, decode_alignment=64)
        self.assertIsNone(spec)
        self.assertEqual(reason, "route_prompt_too_short")

    def test_exact_tokens_required_even_if_text_is_identical(self):
        text = "s" * 1048 + "hi"
        spec, reason = derive_qwen_route_prefix_spec(system_prompt="s" * 1048,
            full_prompt=text, full_tokens=[999] + tokenize(text)[1:],
            render_reference=lambda u: "s" * 1048 + u, tokenize=tokenize, decode_alignment=64)
        self.assertIsNone(spec)
        self.assertEqual(reason, "production_prefix_mismatch")


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        model = Path(self.temp.name) / "model-00001-of-00003.gguf"
        for i in range(1, 4):
            model.with_name(f"model-{i:05}-of-00003.gguf").write_bytes(b"model")
        paths = NativeLlamaPaths(llama_root=Path('/llama'), build_bin=Path('/llama/build/bin'),
            library=Path('/llama/build/bin/libllama.so'), model=model, model_id='test')
        config = NativeClientConfig(context_tokens=4096, threads=10, threads_batch=10,
            batch_size=256, ubatch_size=128, progress_step=64, gpu_layers=0,
            use_extra_bufts=False, load_mode=LLAMA_LOAD_MODE_MMAP, lazy_mode=LLAMA_LAZY_MODE_ON,
            load_mtp=False, qwen_route_prefix_reuse_enabled=True)
        with mock.patch('orbit.native_llama.client.LlamaLibrary'):
            self.client = c = NativeLlamaClient(paths, config)
        c.model_profile = NativeModelProfile(profile_id=QWEN38_FLASH_NEXT_PROFILE_ID,
            family='qwen3.8-flash-next', model_name='Qwen3.8 Flash Next', architecture='qwen4exp',
            renderer='llama.cpp-jinja', reasoning_protocol='qwen-think', tool_call_protocol='qwen3.6-xml',
            history_serialization='qwen-leading-system-only', verified=True, failure_reason=None,
            template_source='gguf-embedded-official', template_sha256='f' * 64,
            thinking_supported=True, mtp_supported=False, gemma_prefix_reuse_supported=False,
            route_prefix_reuse_supported=True, verified_quantization='UD-IQ1_M')
        c._model_metadata_identity = {'general.file_type': '31', 'general.architecture': 'qwen4exp',
            'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen35'}
        c.chat_bridge = Bridge()
        c._chat_bridge_context = c._model = c._vocab = 1
        c.tokenize = tokenize
        c._qwen_backend_build_identity = lambda: 'build'
        self.lib = ShadowLib()
        c.lib = _LibHolder(self.lib)
        c._session.ctx_tgt = object()

    def plan(self, user='hi', system=SYSTEM, tools=None):
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        prompt = self.client.apply_chat_template(messages, tools=tools, thinking=False)
        return self.client._qwen_route_anchor_plan_for_prompt(messages, tools=tools,
            thinking=False, prompt=prompt), tokenize(prompt)

    def prewarm(self):
        result = self.client.capture_qwen3_coder_route_prefix_prefill_only(system_prompt=SYSTEM)
        self.assertTrue(result.succeeded, result)
        self.assertEqual(self.lib.kv, [])
        self.assertFalse(result.sampler_touched)
        self.assertFalse(result.session_history_touched)
        return result

    def test_qualified_profile_and_existing_kill_switch(self):
        plan, tokens = self.plan()
        self.assertEqual(plan.prefix_tokens, tokens[:len(plan.prefix_tokens)])
        self.assertEqual(len(plan.prefix_tokens), plan.spec.invariant_token_count // 64 * 64)
        self.assertTrue(self.client.qwen_route_prefix_reuse_status()['enabled'])
        self.client.config = replace(self.client.config, qwen_route_prefix_reuse_enabled=False)
        self.assertIsNone(self.plan()[0])
        self.assertEqual(self.client.capture_qwen3_coder_route_prefix_prefill_only(system_prompt=SYSTEM).skip_reason,
                         'route_prefix_reuse_disabled')

    def test_unqualified_state_configs_invalidate_existing_checkpoint(self):
        original = self.client.config
        for key, value in {'context_tokens':8192, 'threads':6, 'threads_batch':6,
                'batch_size':128, 'ubatch_size':64, 'progress_step':32, 'gpu_layers':1,
                'use_extra_bufts':True, 'low_memory':True, 'lazy_mode':0, 'load_mode':0,
                'load_mtp':True, 'use_mtp_experimental':True}.items():
            with self.subTest(key=key):
                self.client.config = original
                self.prewarm()
                self.client.config = replace(original, **{key:value})
                self.assertIsNone(self.plan()[0])
                self.assertFalse(self.client._qwen_route_prefix_anchor_state.valid)
                self.assertFalse(self.client.qwen_route_prefix_reuse_status()['enabled'])

    def test_quantization_and_unverified_profile_rejected(self):
        self.client._model_metadata_identity['general.file_type'] = '15'
        self.assertIsNone(self.plan()[0])
        self.client.model_profile = replace(self.client.model_profile, verified=False)
        self.assertEqual(self.client.capture_qwen3_coder_route_prefix_prefill_only(system_prompt=SYSTEM).skip_reason,
                         'model_profile_ineligible')

    def test_all_state_affecting_inputs_change_existing_identity(self):
        plan, _ = self.plan()
        base = self.client._qwen_route_prefix_state_kwargs(plan.spec)
        original = self.client.config
        for field, value in {'context_tokens':8192, 'batch_size':128, 'ubatch_size':64,
                'progress_step':32, 'threads':6, 'threads_batch':6, 'gpu_layers':1,
                'use_extra_bufts':True, 'load_mode':0, 'lazy_mode':0, 'load_mtp':True}.items():
            with self.subTest(field=field):
                self.client.config = replace(original, **{field:value})
                self.assertNotEqual(base, self.client._qwen_route_prefix_state_kwargs(plan.spec))
        self.client.config = original
        self.client.model_profile = replace(self.client.model_profile, template_sha256='other')
        self.assertNotEqual(base, self.client._qwen_route_prefix_state_kwargs(plan.spec))
        self.client._qwen_backend_build_identity = lambda: 'other-build'
        self.assertNotEqual(base, self.client._qwen_route_prefix_state_kwargs(plan.spec))

    def test_changed_second_shard_rejects_old_blob_and_missing_third_fails_closed(self):
        self.prewarm()
        shard = self.client.paths.model.with_name('model-00002-of-00003.gguf')
        shard.write_bytes(b'different-model')
        plan, _ = self.plan()
        self.assertEqual(self.client._prepare_memory_with_qwen_route_anchor(plan), (0, 0))
        self.assertFalse(self.client._qwen_route_prefix_anchor_state.valid)
        self.client.paths.model.with_name('model-00003-of-00003.gguf').unlink()
        self.assertIsNone(self.plan()[0])
        self.assertEqual(self.client._qwen_route_prefix_status.failure_reason, 'model_identity_unavailable')

    def test_template_backend_and_tokenizer_mismatch_restore_fall_cold(self):
        for field in ('template', 'backend', 'tokenizer'):
            with self.subTest(field=field):
                self.client.reset_session_state()
                self.prewarm()
                if field == 'template':
                    self.client.model_profile = replace(self.client.model_profile, template_sha256='changed')
                elif field == 'backend':
                    self.client._qwen_backend_build_identity = lambda: 'changed-build'
                else:
                    self.client._model_metadata_identity['tokenizer.ggml.pre'] = 'changed'
                plan, _ = self.plan()
                self.assertEqual(self.client._prepare_memory_with_qwen_route_anchor(plan), (0, 0))
                self.assertEqual(self.lib.kv, [])

    def test_system_tool_contract_change_never_restores_old_prefix(self):
        self.prewarm()
        old = self.client._qwen_route_prefix_anchor_state.checkpoint_data
        plan, _ = self.plan(system='changed tools ' + SYSTEM)
        self.assertFalse(self.client._qwen_route_prefix_anchor_state.valid)
        self.assertNotEqual(self.client._prepare_memory_with_qwen_route_anchor(plan)[1], len(plan.prefix_tokens))
        self.assertNotEqual(old, self.client._qwen_route_prefix_anchor_state.checkpoint_data)
        self.assertIsNone(self.plan(tools=[{'type':'function', 'function':{'name':'changed'}}])[0])

    def test_two_users_restore_full_recurrent_state_and_preserve_cold_call_layout(self):
        self.prewarm()
        for user in ('hi', 'What is the capital of France?'):
            plan, tokens = self.plan(user)
            self.lib.decode([999, 888])  # unrelated live state must be replaced
            processed, reused = self.client._prepare_memory_with_qwen_route_anchor(plan)
            self.assertEqual(reused, len(plan.prefix_tokens))
            self.lib.decode_calls.clear()
            arr = (ctypes.c_int32 * len(tokens))(*tokens)
            self.client._decode_prompt_range(arr, processed=processed, end=len(tokens), step=64, total=len(tokens))
            self.assertEqual(self.lib.kv, tokens)
            self.assertEqual(self.lib.recurrent, fold(tokens))
            self.assertEqual([64] * (processed // 64) + [len(x) for x in self.lib.decode_calls],
                             [len(tokens[i:i+64]) for i in range(0, len(tokens), 64)])
            self.assertFalse(self.lib.seq_rm_calls)

    def test_failed_restore_clears_native_state_and_falls_cold(self):
        self.prewarm()
        def partial_restore(*args):
            self.lib.decode([777])
            return 0
        self.lib.llama_state_seq_set_data = partial_restore
        self.lib.decode([999])
        plan, _ = self.plan()
        self.assertEqual(self.client._prepare_memory_with_qwen_route_anchor(plan), (0, 0))
        self.assertEqual(self.lib.kv, [])
        self.assertFalse(self.client._qwen_route_prefix_anchor_state.valid)

    def test_cancelled_or_failed_partial_capture_never_publishes_and_can_retry(self):
        decode = self.lib.llama_decode
        def cancel(ctx, batch):
            result = decode(ctx, batch)
            self.client.cancel()
            return result
        self.lib.llama_decode = cancel
        result = self.client.capture_qwen3_coder_route_prefix_prefill_only(system_prompt=SYSTEM)
        self.assertFalse(result.succeeded)
        self.assertFalse(self.client._qwen_route_prefix_anchor_state.valid)
        self.assertEqual(self.lib.kv, [])
        self.lib.llama_decode = decode
        self.lib.fail_decode = True
        result = self.client.capture_qwen3_coder_route_prefix_prefill_only(system_prompt=SYSTEM)
        self.assertFalse(result.succeeded)
        self.assertFalse(self.client._qwen_route_prefix_anchor_state.valid)
        self.lib.fail_decode = False
        self.prewarm()

    def test_cancel_during_serialization_does_not_publish_checkpoint(self):
        get_data = self.lib.llama_state_seq_get_data
        def cancel_after_copy(*args):
            size = get_data(*args)
            self.client.cancel()
            return size
        self.lib.llama_state_seq_get_data = cancel_after_copy
        result = self.client.capture_qwen3_coder_route_prefix_prefill_only(system_prompt=SYSTEM)
        self.assertFalse(result.succeeded)
        self.assertFalse(self.client._qwen_route_prefix_anchor_state.valid)
        self.assertEqual(self.lib.kv, [])

    def test_request_disconnect_callback_bounds_lazy_prefix_capture(self):
        # A disconnected request may have waited for the server lock while
        # cancel() fired. Request start clears that flag; the callback stays
        # latched. Drive the actual completion path with no startup checkpoint.
        self.lib.llama_time_us = lambda: 1000
        self.client._session.sampler = object()
        for batches in (0, 1):
            with self.subTest(batches=batches):
                self.client.reset_session_state()
                self.lib.decode_calls.clear()
                self.client.cancel()
                messages = [{'role':'system','content':SYSTEM}, {'role':'user','content':'hi'}]
                with mock.patch.object(self.client, '_generate_from_current_context',
                                       return_value=(0, 0.0, True)):
                    result = self.client.complete_chat(messages, max_tokens=1,
                        thinking=False, qwen_route_prefix_anchor=True,
                        should_cancel=lambda: len(self.lib.decode_calls) >= batches)
                self.assertTrue(result.cancelled)
                self.assertEqual(len(self.lib.decode_calls), batches)
                self.assertFalse(self.client._qwen_route_prefix_anchor_state.valid)
                self.assertFalse(self.client._rolling_route_anchor_state.valid)
                self.assertEqual(self.lib.kv, [])

    def test_reset_cancel_failed_completion_and_shutdown_destroy_checkpoint(self):
        for action in (self.client.reset_session_state, self.client.cancel,
                       lambda: self.client._invalidate_coder_route_prefix_after_failed_completion('error')):
            self.prewarm()
            action()
            self.assertFalse(self.client._qwen_route_prefix_anchor_state.valid)
            self.client.reset_session_state()
        self.prewarm()
        self.client.lib.configure_expert_usage = mock.Mock()
        self.lib.llama_free = mock.Mock()
        self.lib.llama_model_free = mock.Mock()
        self.client.close()
        self.assertFalse(self.client._qwen_route_prefix_anchor_state.valid)
        self.assertIsNone(self.client._session.ctx_tgt)

    def test_synchronous_server_dispatch_and_no_analysis_capture(self):
        with mock.patch.object(app, 'ROUTE_SYSTEM_PROMPT', SYSTEM):
            result = app.prewarm_startup_route_prefix(self.client)
        self.assertTrue(result.succeeded)
        self.assertEqual(app.route_prefix_prewarm_mode({}), 'startup')
        self.assertEqual(app.route_prefix_prewarm_mode({'ORBIT_KV_PREFIX_PREWARM':'background'}), 'off')
        self.assertFalse(app.prewarm_startup_analysis_prefix(self.client).succeeded)

    def test_prewarm_rolling_and_long_reply_shadow_compose(self):
        self.prewarm()
        blob = self.client._qwen_route_prefix_anchor_state.checkpoint_data
        plan, tokens = self.plan()
        identity = self.client._rolling_route_identity(tools=None)
        self.client._rolling_route_identity_cache = identity
        messages = [{'role':'system','content':SYSTEM}, {'role':'user','content':'hi'}]
        self.assertFalse(self.client._rolling_outranks_route_prefix(tokens,
            rolling_route_eligible=True, rolling_route_identity=identity))
        processed, _ = self.client._prepare_memory_with_qwen_route_anchor(plan)
        self.lib.decode(tokens[processed:])
        self.client._session.cached_prompt_tokens = list(tokens)
        self.client._capture_whole_prompt_checkpoint(tokens, identity, render_messages=messages, render_tools=[])
        self.client._session.committed_sequence_tokens = list(tokens)
        reply = 'long reply ' * 45
        self.lib.llama_memory_clear('mem', True)
        self.lib.decode(tokenize('different final system' + reply))
        result = self.client.advance_route_checkpoint_after_final(reply, should_cancel=lambda:False)
        self.assertEqual(result['status'], 'advanced', result)
        nxt = messages + [{'role':'assistant','content':reply}, {'role':'user','content':'ok'}]
        tokens2 = tokenize(self.client.apply_chat_template(nxt, tools=None, thinking=False))
        self.assertTrue(self.client._rolling_outranks_route_prefix(tokens2,
            rolling_route_eligible=True, rolling_route_identity=identity))
        reused = self.client._prepare_memory_with_ornith_rolling_route_anchor(tokens2)
        self.assertGreater(reused, len(tokens))
        self.lib.decode(tokens2[reused:])
        self.assertEqual(self.lib.kv, tokens2)
        self.assertEqual(self.lib.recurrent, fold(tokens2))
        self.assertIs(self.client._qwen_route_prefix_anchor_state.checkpoint_data, blob)
        self.client.reset_session_state()
        self.assertFalse(self.client._rolling_route_anchor_state.valid)
        self.assertFalse(self.client._rolling_anchor_store().route_shadow_state.valid)
        self.assertFalse(self.client._qwen_route_prefix_anchor_state.valid)

if __name__ == '__main__':
    unittest.main()
