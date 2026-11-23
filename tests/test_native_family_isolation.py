"""Artifact-free boundaries for independently versioned native runtimes."""
import ctypes
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from orbit.native_llama import bindings, paths, model_download
from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.family_build import CPU_DISABLED_BACKENDS, _verify_cpu_cache
from orbit.native_llama.model_registry import get_manifest, load_registry
from orbit.native_llama.native_families import NativeFamily
from orbit.native_llama.native_names import runtime_library_filename


class NativeFamilyIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = paths.NativeLlamaPaths(None, self.root, self.root/'libllama.so',
            self.root/'model.gguf', abi_profile='mainline-b12611')

    def test_exact_context_abis_remain_distinct(self):
        old = bindings.context_params_type('legacy')
        new = bindings.context_params_type('mainline-b12611')
        self.assertEqual(ctypes.sizeof(old), 160)
        self.assertEqual(ctypes.sizeof(new), 176)
        self.assertTrue(hasattr(old, 'defrag_thold'))
        self.assertFalse(hasattr(new, 'defrag_thold'))
        self.assertEqual(new.moe_cache_size.offset, 112)
        self.assertEqual(new.moe_cache_cpu_share.offset, 120)
        with self.assertRaises(RuntimeError):
            bindings.context_params_type('guess')

    def test_same_directory_cannot_switch_abi_in_either_order(self):
        commits = {'legacy':'41abbfd599fbdd3470fcae0a1fb6530ad8403cd7',
                   'mainline-b12611':'690b87d88151dbd775bb6eff5581663688785928'}
        identity = self.root/(runtime_library_filename('orbit-mtmd-bridge')+'.identity.json')
        for first, second in (('legacy','mainline-b12611'), ('mainline-b12611','legacy')):
            with self.subTest(first=first), patch.object(bindings, '_RUNTIME_FAMILY_ROOT', None), \
                    patch.object(bindings, '_RUNTIME_ABI_PROFILE', None), \
                    patch.object(bindings, '_require_runtime_prefix', return_value=()), \
                    patch.object(bindings, 'MtmdLibrary'), \
                    patch.object(bindings.LlamaLibrary, '_load_library') as load, \
                    patch.object(bindings.LlamaLibrary, '_configure_api'), \
                    patch.object(bindings.LlamaLibrary, '_configure_expert_usage_api'):
                identity.write_text(json.dumps({'build':{'upstream_commit':commits[first]}}))
                bindings.LlamaLibrary(self.root, abi_profile=first)
                before = load.call_count
                # Even replacement metadata cannot reconfigure an already loaded CDLL.
                identity.write_text(json.dumps({'build':{'upstream_commit':commits[second]}}))
                with self.assertRaisesRegex(RuntimeError, 'ABI profile conflict'):
                    bindings.LlamaLibrary(self.root, abi_profile=second)
                self.assertEqual(load.call_count, before)

    def test_wrong_artifact_abi_is_rejected_before_native_loading(self):
        identity = self.root/(runtime_library_filename('orbit-mtmd-bridge')+'.identity.json')
        identity.write_text(json.dumps({'build':{'upstream_commit':
            '690b87d88151dbd775bb6eff5581663688785928'}}))
        with patch.object(bindings, 'load_native_cdll') as load:
            with self.assertRaisesRegex(RuntimeError, 'ABI profile disagree'):
                bindings.LlamaLibrary(self.root)
            load.assert_not_called()

    def test_chat_first_binds_abi_even_without_mtmd_identity(self):
        lib = Mock()
        lib.orbit_chat_bridge_api_version.return_value = bindings.CHAT_BRIDGE_API_VERSION
        with patch.object(bindings, '_RUNTIME_FAMILY_ROOT', None), \
                patch.object(bindings, '_RUNTIME_ABI_PROFILE', None), \
                patch.object(bindings, '_require_runtime_prefix', return_value=()), \
                patch.object(bindings, 'validate_chat_bridge_artifact', return_value={
                    'upstream_commit':'690b87d88151dbd775bb6eff5581663688785928'}), \
                patch.object(bindings, 'load_native_cdll', return_value=lib) as load, \
                patch.object(bindings.ChatBridgeLibrary, '_configure_api'):
            bindings.ChatBridgeLibrary(self.root, self.root/'chat.so')
            before = load.call_count
            with self.assertRaisesRegex(RuntimeError, 'ABI profile conflict'):
                bindings.LlamaLibrary(self.root)
            self.assertEqual(load.call_count, before)

    def test_unknown_chat_first_cannot_guess_abi(self):
        with patch.object(bindings, '_RUNTIME_FAMILY_ROOT', None), \
                patch.object(bindings, '_RUNTIME_ABI_PROFILE', None), \
                patch.object(bindings, 'validate_chat_bridge_artifact', return_value={
                    'upstream_commit':'f'*40}), \
                patch.object(bindings, 'load_native_cdll') as load:
            with self.assertRaisesRegex(RuntimeError, 'no selected core ABI'):
                bindings.ChatBridgeLibrary(self.root, self.root/'chat.so')
            load.assert_not_called()

    def test_official_cpu_family_rejects_unsupported_requests_before_loading(self):
        flags = ('gpu_layers','low_memory','use_mtp_experimental','mtp_probe_enabled',
                 'mtp_dry_run_enabled','mtp_accept_probe_enabled','mtp_decode_probe_enabled')
        with patch('orbit.native_llama.client.LlamaLibrary') as load:
            for flag in flags:
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'CPU/text'):
                    NativeLlamaClient(self.paths, replace(NativeClientConfig(), **{flag:1}))
            for field in ('mmproj_model','draft_mtp_model'):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'CPU/text'):
                    NativeLlamaClient(replace(self.paths, **{field:self.root/'unsupported.gguf'}))
            load.assert_not_called()

    def test_missing_or_invalid_bridges_never_reach_model_setup(self):
        with patch('orbit.native_llama.client.LlamaLibrary') as load:
            with self.assertRaisesRegex(RuntimeError, 'both identity-verified'):
                NativeLlamaClient(self.paths)
            with patch('orbit.native_llama.client._resolve_mtmd_bridge_path', return_value=self.root/'mtmd'), \
                    patch('orbit.native_llama.client._resolve_chat_bridge_path', return_value=self.root/'chat'), \
                    patch('orbit.native_llama.client.validate_chat_bridge_artifact',
                          side_effect=RuntimeError('artifact identity mismatch')):
                with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                    NativeLlamaClient(self.paths)
            load.assert_not_called()

    def test_missing_family_does_not_fall_back_to_legacy(self):
        family = NativeFamily('mainline-support', self.root/'mainline', 'mainline-b12611')
        with patch.object(paths, 'native_family', return_value=family):
            with self.assertRaisesRegex(FileNotFoundError, 'mainline-support'):
                paths._resolve_native_runtime(None, family_name='mainline-support')

    def test_cached_accelerator_on_cannot_be_accepted_as_cpu(self):
        cache = self.root/'CMakeCache.txt'
        off = {name:'OFF' for name in CPU_DISABLED_BACKENDS}
        def write(values):
            cache.write_text('\n'.join(f'GGML_{key}:BOOL={value}' for key,value in values.items()))
        write(off)
        _verify_cpu_cache(cache)
        for name in CPU_DISABLED_BACKENDS:
            with self.subTest(name=name):
                write(off | {name:'ON'})
                with self.assertRaisesRegex(RuntimeError, 'accelerator'):
                    _verify_cpu_cache(cache)

    def test_registered_download_pins_remote_variant_and_keeps_local_name(self):
        manifest = get_manifest('inkling2-mini-ud-q4-k-m')
        self.assertEqual(manifest.native_family, 'mainline-support')
        self.assertIsNone(manifest.mtp)
        self.assertIsNone(manifest.mmproj)
        expected = ('https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/'
            '800c946d7b9c830c86290df49afbbe91db8455f5/Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file')
        for spec in ('unsloth/Inkling-2-Mini-GGUF',
                     'unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_M.gguf'):
            with self.subTest(spec=spec), patch.object(model_download, 'fetch_resumable') as fetch:
                result = model_download.download_model(spec, models_dir=self.root)
                self.assertEqual(result.url, expected)
                self.assertEqual(result.path.name, 'Inkling-2-Mini-UD-Q4_K_M.gguf')
                self.assertEqual(fetch.call_args.args, (expected, result.path))

    def test_registry_rejects_unknown_family_and_unsafe_remote_identity(self):
        packaged = Path(__file__).resolve().parents[1]/'src/orbit/native_llama/model_registry.json'
        original = json.loads(packaged.read_text())
        for change in ({'native_family':'unknown'}, {'native_family':'legacy'}, {'target':{'remote_file':'../model.gguf'}},
                       {'target':{'revision':'main'}}, {'target':{'revision':True}}):
            data = json.loads(json.dumps(original))
            if 'target' in change:
                data['models'][-1]['target'].update(change['target'])
            else:
                data['models'][-1].update(change)
            registry = self.root/'registry.json'
            registry.write_text(json.dumps(data))
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_registry(registry)

    def test_inkling_metadata_and_template_identity_are_exact(self):
        from orbit.native_llama.model_profiles import detect_native_model_profile
        metadata = {
            'general.name':'Inkling-2-Mini', 'general.architecture':'inkling2_loop',
            'general.file_type':'15', 'general.quantization_version':'2',
            'tokenizer.ggml.model':'gpt2', 'tokenizer.ggml.pre':'inkling2',
            'tokenizer.ggml.eos_token_id':'200006', 'tokenizer.ggml.eot_token_id':'200006',
            'tokenizer.ggml.add_bos_token':'false', 'tokenizer.ggml.add_eos_token':'false',
            'inkling2_loop.block_count':'24', 'inkling2_loop.loop.count':'3',
            'inkling2_loop.loop.start':'6', 'inkling2_loop.loop.end':'18',
            'inkling2_loop.expert_count':'128', 'inkling2_loop.expert_used_count':'8',
        }
        template = (Path(__file__).parent/'fixtures/inkling2_mini_chat_template.jinja').read_text()
        accepted = detect_native_model_profile(metadata, template)
        self.assertTrue(accepted.verified)
        self.assertFalse(accepted.mtp_supported)
        self.assertEqual(accepted.supported_capabilities, frozenset({'chat','tools'}))
        for key in metadata:
            with self.subTest(key=key):
                self.assertFalse(detect_native_model_profile(metadata | {key:'changed'}, template).verified)
        self.assertFalse(detect_native_model_profile(metadata, template+' ').verified)
