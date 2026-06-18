from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_server.app import OrbitNativeServer


class NativeRuntimePropsTests(unittest.TestCase):
    def _paths(self) -> NativeLlamaPaths:
        return NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/target.gguf"),
            mmproj_model=Path("/models/mmproj.gguf"),
            draft_mtp_model=None,
            multimodal_available=True,
            multimodal_fallback_reason=None,
            mtp_available=False,
            fallback_reason="draft-mtp-missing",
            model_id="gemma4-12b-it-q4km",
        )

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_runtime_info_matches_native_config_baseline(self, _mocked_lib) -> None:
        client = NativeLlamaClient(
            self._paths(),
            NativeClientConfig(
                context_tokens=8192,
                threads=6,
                threads_batch=6,
                batch_size=256,
                ubatch_size=128,
            ),
        )
        server = OrbitNativeServer(client=client, model_alias="m")

        runtime = server.runtime_info()

        self.assertEqual(runtime["threads"], 6)
        self.assertEqual(runtime["threads_batch"], 6)
        self.assertEqual(runtime["ctx_size"], 8192)
        self.assertEqual(runtime["batch_size"], 256)
        self.assertEqual(runtime["ubatch_size"], 128)
        self.assertEqual(runtime["parallel_slots"], 1)
        self.assertEqual(runtime["thinking_mode"], "off")

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_server_props_include_multimodal_paths(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        client.supports_vision = True
        client.supports_audio = True
        server = OrbitNativeServer(client=client, model_alias="m")

        self.assertEqual(server.client.paths.mmproj_model, Path("/models/mmproj.gguf"))
        self.assertTrue(server.client.paths.multimodal_available)
        self.assertTrue(server.client.supports_vision)
        self.assertTrue(server.client.supports_audio)


if __name__ == "__main__":
    unittest.main()
