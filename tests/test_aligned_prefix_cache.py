from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
import threading
from itertools import count

from orbit.native_llama.aligned_prefix_cache import (
    AlignedPrefixCache,Snapshot,PREFIX_TOKENS,MAX_CHECKPOINT_BYTES,enabled,qualified_config,
)
from orbit.native_llama.client import NativeLlamaClient,NativeClientConfig
from orbit.native_llama.events import NativeTimings
from orbit.native_llama.model_profiles import INKLING2_PROFILE_ID
from orbit.native_llama.cache_profile_policy import cache_policy_for_profile


class Native:
    def __init__(self):
        self.size=4;self.write=4;self.read=4;self.restored=[]
    def llama_state_seq_get_size(self,ctx,seq):return self.size
    def llama_state_seq_get_data(self,ctx,buffer,size,seq):
        buffer[:]=b'KV42';return self.write
    def llama_state_seq_set_data(self,ctx,buffer,size,seq):
        self.restored.append(bytes(buffer));return self.read


class AlignedPrefixCacheTests(unittest.TestCase):
    def setUp(self):
        env=patch.dict('os.environ',{'ORBIT_ALIGNED_PREFIX_REUSE':'1'})
        env.start();self.addCleanup(env.stop)
        self.tokens=list(range(PREFIX_TOKENS));self.prompt=self.tokens+[2048,2049]
        self.lib=Native();self.cache=AlignedPrefixCache()
    def capture(self):return self.cache.capture(self.lib,1,self.tokens,'identity')

    def test_exact_prefix_and_full_identity_authorize_restore(self):
        self.assertTrue(self.capture());self.assertTrue(self.cache.restore(self.lib,1,self.prompt,'identity'))
        self.assertEqual(self.lib.restored,[b'KV42']);self.assertEqual(self.cache.restores,1)

    def test_short_inputs_changed_tokens_and_identity_never_reach_native_restore(self):
        self.capture()
        for tokens,identity in [(self.tokens,'identity'),([99]+self.prompt[1:],'identity'),(self.prompt,'different-context')]:
            self.assertFalse(self.cache.restore(self.lib,1,tokens,identity))
        self.assertEqual(self.lib.restored,[])
        self.assertTrue(self.cache.matches(self.prompt,'identity'))

    def test_capture_requires_exact_boundary_and_bounded_nonempty_state(self):
        self.assertFalse(self.cache.capture(self.lib,1,self.tokens[:-1],'identity'))
        for size in (0,-1,MAX_CHECKPOINT_BYTES+1):
            self.lib.size=size
            with patch('orbit.native_llama.aligned_prefix_cache.c_ubyte') as allocator:
                self.assertFalse(self.capture());self.assertIsNone(self.cache.snapshot)
                allocator.__mul__.assert_not_called()

    def test_partial_capture_and_restore_invalidate_checkpoint(self):
        self.lib.write=3;self.assertFalse(self.capture());self.assertIsNone(self.cache.snapshot)
        self.lib.write=4;self.capture();self.lib.read=3
        self.assertFalse(self.cache.restore(self.lib,1,self.prompt,'identity'));self.assertIsNone(self.cache.snapshot)

    def test_native_io_exceptions_never_leave_an_eligible_state(self):
        self.lib.llama_state_seq_get_data=Mock(side_effect=RuntimeError('write failed'))
        self.assertFalse(self.capture());self.assertIsNone(self.cache.snapshot)
        self.lib=Native();self.capture();self.lib.llama_state_seq_set_data=Mock(side_effect=RuntimeError('read failed'))
        self.assertFalse(self.cache.restore(self.lib,1,self.prompt,'identity'));self.assertIsNone(self.cache.snapshot)

    def test_invalid_empty_or_oversized_snapshot_cannot_restore(self):
        self.cache.snapshot=Snapshot('identity',tuple(self.tokens),b'')
        self.assertFalse(self.cache.restore(self.lib,1,self.prompt,'identity'))
        self.assertEqual(self.lib.restored,[])

    def test_clear_releases_checkpoint_without_resetting_observed_counts(self):
        self.capture();self.cache.clear('session_reset')
        self.assertIsNone(self.cache.snapshot);self.assertEqual(self.cache.captures,1)
        self.assertFalse(self.cache.matches(self.prompt,'identity'))

    def test_kill_switch_invalid_values_disable_safely(self):
        self.assertTrue(enabled({}))
        for value in ('0','false','off','invalid',''):
            self.assertFalse(enabled({'ORBIT_ALIGNED_PREFIX_REUSE':value}))

    def test_configuration_matrix_is_narrow_and_preserves_call_alignment(self):
        cfg=NativeClientConfig(use_extra_bufts=False);self.assertTrue(qualified_config(cfg))
        for name,value in [('threads',4),('threads_batch',8),('context_tokens',4096),('batch_size',128),('ubatch_size',64),('progress_step',32),('gpu_layers',1),('low_memory',True),('use_extra_bufts',None),('use_mtp_experimental',True),('mtp_probe_enabled',True),('moe_expert_usage_enabled',True)]:
            with self.subTest(name=name):self.assertFalse(qualified_config(replace(cfg,**{name:value})))

    def client(self):
        c=object.__new__(NativeLlamaClient)
        c.model_profile=SimpleNamespace(profile_id=INKLING2_PROFILE_ID,verified=True)
        c.config=NativeClientConfig(use_extra_bufts=False)
        c._session=SimpleNamespace(ctx_tgt=1,mtp_enabled=False,prompt_cache_mode='chat:thinking=off')
        c.lib=SimpleNamespace(lib=self.lib);c._aligned_prefix_cache=self.cache
        return c

    def test_only_explicit_verified_profile_no_thinking_or_mtp_can_enter(self):
        c=self.client();self.assertTrue(c._aligned_prefix_eligible(thinking=False))
        self.assertFalse(c._aligned_prefix_eligible(thinking=True))
        c._persistent_mtp_runtime=object();self.assertFalse(c._aligned_prefix_eligible(thinking=False));c._persistent_mtp_runtime=None
        c._session.mtp_enabled=True;self.assertFalse(c._aligned_prefix_eligible(thinking=False));c._session.mtp_enabled=False
        c.model_profile.verified=False;self.assertFalse(c._aligned_prefix_eligible(thinking=False));c.model_profile.verified=True
        for profile_id in ('orbit-ornith15-native-v1','research-exact-model','orbit-inkling2-mini-native-v2',None):
            c.model_profile.profile_id=profile_id;self.assertFalse(c._aligned_prefix_eligible(thinking=False))
        self.assertFalse(cache_policy_for_profile('orbit-ornith15-native-v1').aligned_prompt_checkpoint)

    def test_full_context_identity_changes_never_alias_a_checkpoint(self):
        c=self.client();c.paths=SimpleNamespace(model="/models/first.gguf")
        c.model_profile.template_sha256="template1";c._model_metadata_identity={"architecture":"inkling2_loop"}
        c._reset_generation=0;c._session.session_id="session1"
        c._qwen_backend_build_identity=Mock(return_value="native-build1")
        original=c._aligned_prefix_identity()
        changes=[(c._session,"ctx_tgt",2),(c._session,"session_id","session2"),
                 (c.paths,"model","/models/second.gguf"),(c.model_profile,"template_sha256","template2"),
                 (c,"_reset_generation",1),(c,"config",replace(c.config,threads=4)),
                 (c,"_model_metadata_identity",{"architecture":"other"})]
        for owner,name,value in changes:
            before=getattr(owner,name);setattr(owner,name,value)
            self.assertNotEqual(c._aligned_prefix_identity(),original,name)
            setattr(owner,name,before)
        c._qwen_backend_build_identity.return_value="native-build2"
        self.assertNotEqual(c._aligned_prefix_identity(),original)

    def test_failed_restore_clears_physical_memory_before_cold_fallback(self):
        self.capture();self.lib.read=3;c=self.client();trace=[]
        c._clear_target_memory=lambda:trace.append('clear')
        c._prepare_memory_for_prompt=lambda tokens:trace.append('cold') or 0
        self.assertEqual(c._prepare_memory_with_aligned_prefix(self.prompt,'identity'),0)
        self.assertEqual(trace,['clear','clear','cold']);self.assertIsNone(self.cache.snapshot)

    def test_successful_restore_publishes_only_checkpoint_as_committed(self):
        self.capture();c=self.client();c._clear_target_memory=Mock()
        def prepare(tokens):
            self.assertEqual(c._session.committed_sequence_tokens,self.tokens)
            self.assertEqual(c._session.cached_prompt_tokens,self.tokens)
            return PREFIX_TOKENS
        c._prepare_memory_for_prompt=prepare
        self.assertEqual(c._prepare_memory_with_aligned_prefix(self.prompt,'identity'),PREFIX_TOKENS)
        c._clear_target_memory.assert_called_once()

    def lifecycle_client(self):
        c=self.client();c.cancel_event=threading.Event();c._reset_generation=0
        c._session.cached_prompt_tokens=[];c._session.sampler=None
        c.lib.configure_expert_usage=Mock();c.lib.lib.llama_get_memory=Mock(return_value=1)
        c.lib.lib.llama_memory_clear=Mock();c.lib.lib.llama_free=Mock()
        for name in ("_invalidate_committed_sequence","_invalidate_final_prefix","_invalidate_qwen_route_prefix",
                     "_invalidate_qwen36_shell_tool_prefix","_invalidate_rolling_route_anchor","_free_persistent_mtp_session"):
            setattr(c,name,Mock())
        c.chat_bridge=None;c._chat_bridge_context=None;c._mtmd_ctx=None;c._model=None
        return c

    def test_idle_cancel_immediately_invalidates_a_populated_checkpoint(self):
        c=self.lifecycle_client();self.capture();c.cancel()
        self.assertIsNone(self.cache.snapshot);self.assertTrue(c.cancel_event.is_set())
        c.reset_cancel();self.assertFalse(self.cache.matches(self.prompt,'identity'))

    def test_reset_and_close_release_existing_snapshot(self):
        c=self.lifecycle_client();self.capture();c.reset_session_state()
        self.assertIsNone(self.cache.snapshot)
        self.capture();c.close();self.assertIsNone(self.cache.snapshot)

    def test_dynamic_kill_switch_releases_snapshot_before_next_generation(self):
        c=self.lifecycle_client();self.capture()
        c._complete_prompt_standard=Mock(return_value=NativeTimings(10,0,0,10,1,0))
        with patch.dict('os.environ',{'ORBIT_ALIGNED_PREFIX_REUSE':'0'}):
            c.complete_prompt('prompt',max_tokens=1,allow_mtp_experimental=False)
        self.assertIsNone(self.cache.snapshot)
        self.assertNotIn('aligned_prefix_reuse',c._complete_prompt_standard.call_args.kwargs)

    def standard_client(self):
        c=self.lifecycle_client();c._session.sampler=1;c._vocab=1
        c._session.session_id='session';c._session.committed_sequence_tokens=None
        c._qwen3_coder_native_protocol=lambda:False
        c._aligned_prefix_identity=lambda:'identity'
        c._route_anchor_plan=Mock(return_value=None);c._final_prefix_plan=Mock(return_value=None)
        ticks=count();c.lib.lib.llama_time_us=lambda:next(ticks)*1000
        ranges=[]
        def decode(array,*,processed,end,step,total,**kwargs):
            ranges.append((processed,end,step));return end
        c._decode_prompt_range=decode
        c._generate_from_current_context=Mock(return_value=(0,0.0,False))
        c._commit_sequence=Mock()
        c.tokenize=Mock(return_value=self.prompt)
        return c,ranges

    def test_standard_prefill_captures_on_unchanged_boundary_then_reuses_without_recapture(self):
        c,ranges=self.standard_client()
        first=c._complete_prompt_standard('prompt',max_tokens=0,aligned_prefix_reuse=True)
        self.assertEqual(ranges,[(0,1024,64),(1024,1026,64)])
        self.assertEqual(first.reused_prompt_tokens,0);self.assertEqual(self.cache.captures,1)
        ranges.clear()
        warm=c._complete_prompt_standard('prompt',max_tokens=0,aligned_prefix_reuse=True)
        self.assertEqual(ranges,[(1024,1026,64)])
        self.assertEqual(warm.reused_prompt_tokens,1024);self.assertEqual(warm.evaluated_prompt_tokens,2)
        self.assertEqual(self.cache.captures,1)
        # A short intervening final does not evict the useful route checkpoint.
        c.tokenize.return_value=[9000,9001];c._session.cached_prompt_tokens=[]
        c._complete_prompt_standard('short',max_tokens=0,aligned_prefix_reuse=True)
        self.assertIsNotNone(self.cache.snapshot);self.assertEqual(self.cache.captures,1)

    def test_partial_resident_append_does_not_create_a_non_cold_checkpoint(self):
        c,ranges=self.standard_client()
        c._prepare_memory_for_prompt=Mock(return_value=17)
        c._complete_prompt_standard('prompt',max_tokens=0,aligned_prefix_reuse=True)
        self.assertEqual(ranges,[(17,1026,64)])
        self.assertIsNone(self.cache.snapshot)

    def test_unavailable_identity_falls_back_to_normal_prefill(self):
        c,ranges=self.standard_client()
        c._aligned_prefix_identity=Mock(side_effect=RuntimeError('unavailable native identity'))
        result=c._complete_prompt_standard('prompt',max_tokens=0,aligned_prefix_reuse=True)
        self.assertEqual(ranges,[(0,1026,64)])
        self.assertEqual(result.reused_prompt_tokens,0)
        self.assertIsNone(self.cache.snapshot)

    def test_cancel_and_prefill_exception_drop_checkpoint_and_release_inflight(self):
        for fail in (True,False):
            c=self.client();self.capture();c.reset_cancel=Mock();c._thinking_enabled=lambda thinking:False
            c._aligned_prefix_eligible=Mock(return_value=True);c._invalidate_committed_sequence=Mock()
            c._invalidate_coder_route_prefix_after_failed_completion=Mock()
            c._complete_prompt_standard=Mock(side_effect=RuntimeError('prefill aborted')) if fail else Mock(return_value=NativeTimings(1100,0,0,1100,1,0,cancelled=True))
            if fail:
                with self.assertRaises(RuntimeError):c.complete_prompt('prompt',max_tokens=1,allow_mtp_experimental=False)
            else:c.complete_prompt('prompt',max_tokens=1,allow_mtp_experimental=False)
            self.assertIsNone(self.cache.snapshot);self.assertFalse(c._session.in_flight)

if __name__=='__main__':unittest.main()
