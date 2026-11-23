"""Intel GPU is explicit, isolated, and never a silent CPU fallback."""
from argparse import Namespace
from ctypes import c_void_p
import hashlib
import io
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orbit.native_llama.acceleration import selected_family, select_intel_device, efficient_host_cpus, loaded_intel_status
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.llama_provenance import source_tree_sha256
from orbit.native_llama.vulkan_build import prepare_source, require_shader_compiler
from orbit.native_server import app


class IntelGPUChoiceTests(unittest.TestCase):
    def test_mainline_checkout_contains_the_complete_attested_source(self):
        root=Path(__file__).resolve().parents[1]/'src/orbit/native_llama/vendor/families/mainline_support'
        manifest=json.loads((root/'LLAMA_PROVENANCE.json').read_text())
        self.assertEqual(source_tree_sha256(root/'source/llama.cpp'),manifest['source_tree_sha256'])

    def test_family_selection_is_explicit_and_scoped(self):
        self.assertEqual(selected_family('legacy','cpu'),'legacy')
        self.assertEqual(selected_family('mainline-support','cpu'),'mainline-support')
        self.assertEqual(selected_family('mainline-support','intel-gpu'),'mainline-vulkan')
        for family, accelerator in [('legacy','intel-gpu'),('mainline-support','auto')]:
            with self.assertRaises(ValueError):selected_family(family,accelerator)
        self.assertEqual(app.build_parser().parse_args([]).accelerator,'cpu')

    def test_hybrid_affinity_never_escapes_allowed_cpus(self):
        freqs={0:4800,1:4900,4:3800,5:3800,6:3800,12:3400,13:3400}
        self.assertEqual(efficient_host_cpus(set(freqs),freqs),{4,5})
        self.assertEqual(efficient_host_cpus({0,1,5,6,12,13},freqs),{5,6})
        self.assertIsNone(efficient_host_cpus({0,1},{0:4000,1:4000}))
        self.assertIsNone(efficient_host_cpus({0},freqs))

    def lib(self, devices):
        lib=Mock()
        lib.ggml_backend_dev_count.return_value=len(devices)
        lib.ggml_backend_dev_get.side_effect=lambda i:i+1
        lib.ggml_backend_dev_type.side_effect=lambda d:devices[d-1][0]
        lib.ggml_backend_dev_name.side_effect=lambda d:devices[d-1][1]
        lib.ggml_backend_dev_description.side_effect=lambda d:devices[d-1][2]
        return lib

    def test_device_selection_rejects_software_other_gpu_and_ambiguity(self):
        cpu=(0,b'CPU',b'Intel CPU');software=(0,b'Vulkan1',b'llvmpipe')
        other=(2,b'Vulkan0',b'AMD Radeon');intel=(2,b'Vulkan2',b'Intel(R) Graphics (PTL)')
        self.assertEqual(select_intel_device(self.lib([cpu,software,other,intel])),(4,'Vulkan2','Intel(R) Graphics (PTL)'))
        for devices in [[cpu,software],[other],[intel,intel]]:
            with self.assertRaisesRegex(RuntimeError,'exactly one'):select_intel_device(self.lib(devices))

    def test_active_gpu_status_requires_native_layer_assignment(self):
        lib=Mock();lib.orbit_llama_model_gpu_layers.return_value=0;lib.orbit_llama_model_gpu_bytes.return_value=11_000_000_000
        with self.assertRaisesRegex(RuntimeError,'fallback refused'):
            loaded_intel_status(lib,object(),'Vulkan0','Intel GPU')
        lib.orbit_llama_model_gpu_layers.return_value=49
        lib.orbit_llama_model_gpu_bytes.return_value=0
        with self.assertRaisesRegex(RuntimeError,'weight buffers'):
            loaded_intel_status(lib,object(),'Vulkan0','Intel GPU')
        lib.orbit_llama_model_gpu_bytes.return_value=11_000_000_000
        self.assertEqual(loaded_intel_status(lib,object(),'Vulkan0','Intel GPU')['offloaded_layers'],49)

    def test_gpu_requires_full_offload_before_loading_libraries(self):
        p=NativeLlamaPaths(None,Path('/missing'),Path('/missing/libllama.so'),Path('/model.gguf'),
                           abi_profile='mainline-b12611',accelerator='intel-gpu')
        with patch('orbit.native_llama.client.LlamaLibrary') as loader:
            with self.assertRaisesRegex(ValueError,'full GPU'):NativeLlamaClient(p,NativeClientConfig())
            loader.assert_not_called()

    def test_real_gpu_start_configures_host_before_client_construction(self):
        paths=NativeLlamaPaths(None,Path('/missing'),Path('/missing/libllama.so'),Path('/model.gguf'))
        with patch.object(app,'_select_startup_model',return_value=0), \
             patch.object(app,'resolve_bootstrap_paths',return_value=paths), \
             patch.object(app,'detect_topology',return_value=SimpleNamespace(machine_model='test')), \
             patch('orbit.native_llama.acceleration.configure_gpu_host',return_value=[4,5]) as host, \
             patch.object(app,'NativeLlamaClient',side_effect=RuntimeError('stop before native load')) as loader, \
             redirect_stderr(io.StringIO()):
            self.assertEqual(app.run_server(['--accelerator','intel-gpu','--ctx','8192']),1)
            host.assert_called_once()
            loader.assert_called_once()

    def test_gpu_preview_does_not_promise_cpu_calibration(self):
        args=app.build_parser().parse_args(['--accelerator','intel-gpu','--show-profile'])
        with patch.object(app,'_resolve_preview_target',return_value=('model','/model',('identity',0),False)), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(app._show_profile(args),0)
        self.assertNotIn('will calibrate',out.getvalue())
        self.assertIn('CPU calibration is not used',out.getvalue())

    def test_gpu_does_not_borrow_cpu_calibration_or_qualified_profile(self):
        args=Namespace(accelerator='intel-gpu',threads=None,threads_batch=None,batch=None,ubatch=None)
        with patch.object(app,'_model_identity_for_profile',side_effect=AssertionError('CPU calibration path')):
            r=app._resolve_startup_profile(args)
        self.assertEqual((r.profile.threads,r.profile.threads_batch,r.profile.batch,r.profile.ubatch),(1,1,256,128))
        self.assertTrue(r.calibrated)
        self.assertIsNone(r.cache_path)
        self.assertIsNone(app._qualified_startup_profile(args))
        args.threads=2
        self.assertEqual(app._resolve_startup_profile(args).profile.threads,2)
        args.threads=0
        with self.assertRaises(ValueError):app._resolve_startup_profile(args)

    def fixture(self, root):
        base=root/'mainline_support';gpu=root/'mainline_vulkan'
        src=base/'source/llama.cpp';src.mkdir(parents=True);(src/'input.cpp').write_text('original')
        def manifest(tree):return {'format':1,'upstream_commit':'a'*40,'upstream_tag':'b1','source_tree_sha256':source_tree_sha256(tree),'patchset_sha256':'b'*64,'patched_paths':[]}
        (base/'LLAMA_PROVENANCE.json').write_text(json.dumps(manifest(src)))
        (gpu/'overlay').mkdir(parents=True);(gpu/'overlay/input.cpp').write_text('patched')
        desired=root/'expected';desired.mkdir();(desired/'input.cpp').write_text('patched')
        (gpu/'LLAMA_PROVENANCE.json').write_text(json.dumps(manifest(desired)))
        (gpu/'OVERLAY.json').write_text(json.dumps({'base_source_tree_sha256':source_tree_sha256(src),'files':[
            {'path':'input.cpp','base_sha256':hashlib.sha256(b'original').hexdigest(),'overlay_sha256':hashlib.sha256(b'patched').hexdigest()}]}))
        return base,gpu

    def test_overlay_is_reproducible_preserves_cpu_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,gpu=self.fixture(Path(tmp));source=prepare_source(gpu)
            self.assertEqual((base/'source/llama.cpp/input.cpp').read_text(),'original')
            self.assertEqual((source/'input.cpp').read_text(),'patched')
            self.assertEqual(prepare_source(gpu),source)
            (source/'input.cpp').write_text('tampered')
            with self.assertRaisesRegex(RuntimeError,'provenance|PROVENANCE'):prepare_source(gpu)
            self.assertEqual((prepare_source(gpu,clean=True)/'input.cpp').read_text(),'patched')
            (gpu/'overlay/input.cpp').write_text('wrong patch')
            with self.assertRaisesRegex(RuntimeError,'overlay identity'):prepare_source(gpu)

    def test_shader_feature_failure_is_not_a_successful_build(self):
        with patch('orbit.native_llama.vulkan_build.subprocess.run',return_value=Mock(returncode=1,stderr='extension not supported')):
            with self.assertRaisesRegex(RuntimeError,'--glslc'):require_shader_compiler('/compiler',Path('/source'))

if __name__=='__main__':unittest.main()
