from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from tests import test_aligned_prefix_cache as fixtures
from orbit.native_llama.aligned_prefix_cache import gpu_enabled, gpu_config, qualified_config, AlignedPrefixCache
from orbit.native_llama.client import NativeLlamaClient, NativeClientConfig
from orbit.native_llama.model_profiles import INKLING2_PROFILE_ID

class GPUPrefixCacheTests(unittest.TestCase):
    def setUp(self):
        env=patch.dict('os.environ',{'ORBIT_ALIGNED_PREFIX_REUSE':'1','ORBIT_GPU_ALIGNED_PREFIX_REUSE':'1'})
        env.start();self.addCleanup(env.stop)
    def client(self):
        c=object.__new__(NativeLlamaClient)
        c.paths=SimpleNamespace(model='/model.gguf',accelerator='intel-gpu',abi_profile='mainline-b12611')
        c.config=NativeClientConfig(threads=1,threads_batch=1,gpu_layers=-1,use_extra_bufts=False)
        c.model_profile=SimpleNamespace(verified=True,profile_id=INKLING2_PROFILE_ID,template_sha256='template')
        c._gpu_prefix_backend_identity='gpu-build-1'
        c._acceleration_status={'active':'intel-gpu','gpu_weight_bytes':11_000_000_000,'device':'Intel PTL'}
        c._session=SimpleNamespace(ctx_tgt=1,session_id='session',mtp_enabled=False,prompt_cache_mode='chat:thinking=off')
        c._reset_generation=0;c._model_metadata_identity={'architecture':'inkling2_loop'}
        c._qwen_backend_build_identity=Mock(return_value='llama-build')
        return c

    def test_gpu_has_separate_exact_layout(self):
        c=self.client();self.assertTrue(gpu_config(c.config));self.assertFalse(qualified_config(c.config))
        self.assertTrue(qualified_config(NativeClientConfig(use_extra_bufts=False)))
        for key,value in [('threads',6),('threads_batch',6),('gpu_layers',0),('context_tokens',4096),('batch_size',128),('ubatch_size',64),('progress_step',32),('use_extra_bufts',True),('low_memory',True),('use_mtp_experimental',True),('mtp_probe_enabled',True),('mtp_dry_run_enabled',True),('mtp_accept_probe_enabled',True),('mtp_decode_probe_enabled',True),('moe_expert_usage_enabled',True)]:
            with self.subTest(key=key):self.assertFalse(gpu_config(replace(c.config,**{key:value})))

    def test_master_and_gpu_switches_fail_closed(self):
        self.assertFalse(gpu_enabled({}))
        self.assertTrue(gpu_enabled({'ORBIT_GPU_ALIGNED_PREFIX_REUSE':'1'}))
        for value in ['0','false','invalid','']:
            self.assertFalse(gpu_enabled({'ORBIT_GPU_ALIGNED_PREFIX_REUSE':value}))
            self.assertFalse(gpu_enabled({'ORBIT_ALIGNED_PREFIX_REUSE':value,'ORBIT_GPU_ALIGNED_PREFIX_REUSE':'1'}))
        c=self.client()
        with patch.dict('os.environ',{'ORBIT_GPU_ALIGNED_PREFIX_REUSE':'0'}):
            self.assertFalse(c._aligned_prefix_eligible(thinking=False))
            c.paths.accelerator='cpu';c.config=NativeClientConfig(use_extra_bufts=False)
            self.assertTrue(c._aligned_prefix_eligible(thinking=False))

    def test_verified_gpu_buffers_and_bound_native_build_required(self):
        with patch.dict('os.environ',{'ORBIT_ALIGNED_PREFIX_REUSE':'1','ORBIT_GPU_ALIGNED_PREFIX_REUSE':'1'}):
            c=self.client();self.assertTrue(c._aligned_prefix_eligible(thinking=False))
            for owner,key,value in [(c.paths,'abi_profile','legacy'),(c,'_gpu_prefix_backend_identity',None),(c.model_profile,'verified',False),(c.model_profile,'profile_id','research-model')]:
                old=getattr(owner,key);setattr(owner,key,value);self.assertFalse(c._aligned_prefix_eligible(thinking=False));setattr(owner,key,old)
            c._acceleration_status['active']='cpu';self.assertFalse(c._aligned_prefix_eligible(thinking=False))
            c._acceleration_status['active']='intel-gpu';c._acceleration_status['gpu_weight_bytes']=0
            self.assertFalse(c._aligned_prefix_eligible(thinking=False))

    def test_reasoning_tools_mode_and_mtp_remain_ineligible(self):
        c=self.client()
        self.assertFalse(c._aligned_prefix_eligible(thinking=True))
        with patch('orbit.native_llama.client.current_reasoning_effort',return_value='max'):
            self.assertFalse(c._aligned_prefix_eligible(thinking=False))
        c._session.prompt_cache_mode='chat:tools';self.assertFalse(c._aligned_prefix_eligible(thinking=False))
        c._session.prompt_cache_mode='chat:thinking=off';c._session.mtp_enabled=True
        self.assertFalse(c._aligned_prefix_eligible(thinking=False))

    def test_gpu_build_device_context_and_cpu_identities_do_not_alias(self):
        c=self.client();original=c._aligned_prefix_identity()
        for owner,key,value in [(c,'_gpu_prefix_backend_identity','gpu-build-2'),(c,'_acceleration_status',dict(c._acceleration_status,device='other device')),(c._session,'ctx_tgt',2),(c._session,'session_id','other'),(c,'_reset_generation',1),(c.paths,'accelerator','cpu')]:
            old=getattr(owner,key);setattr(owner,key,value);self.assertNotEqual(c._aligned_prefix_identity(),original);setattr(owner,key,old)

    def test_native_reuse_requires_exact_gpu_identity(self):
        c=self.client();lib=fixtures.Native();lib.llama_state_seq_set_data=Mock(wraps=lib.llama_state_seq_set_data)
        c.lib=SimpleNamespace(lib=lib);c._aligned_prefix_cache=AlignedPrefixCache()
        prefix=list(range(1024));prompt=prefix+[9999];identity=c._aligned_prefix_identity()
        self.assertTrue(c._aligned_prefix_cache.capture(lib,1,prefix,identity))
        c._clear_target_memory=Mock();c._prepare_memory_for_prompt=Mock(return_value=0)
        c._gpu_prefix_backend_identity='different-gpu-build'
        self.assertEqual(c._prepare_memory_with_aligned_prefix(prompt,c._aligned_prefix_identity()),0)
        lib.llama_state_seq_set_data.assert_not_called()
        c._gpu_prefix_backend_identity='gpu-build-1';c._prepare_memory_for_prompt.return_value=1024
        self.assertEqual(c._prepare_memory_with_aligned_prefix(prompt,c._aligned_prefix_identity()),1024)
        lib.llama_state_seq_set_data.assert_called_once()
