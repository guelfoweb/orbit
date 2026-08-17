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

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_token_counts_do_not_touch_completion_state(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(context_tokens=16384))
        client.count_text_tokens = mock.Mock(return_value=17)
        client.inspect_chat_tokens = mock.Mock(return_value=(29, "a" * 64, "b" * 64))
        client.inspect_artifact_content_tokens = mock.Mock(return_value=(31, "c" * 64, "d" * 64))
        server = OrbitNativeServer(client=client, model_alias="m")

        text = server.count_text_tokens("hello")
        chat = server.count_chat_tokens(
            [{"role": "user", "content": "hello"}],
            tools=[],
            thinking=False,
        )
        artifact = server.count_artifact_content_tokens(
            [{"role": "user", "content": "artifact"}],
        )

        self.assertEqual(text, {"tokens": 17, "context_tokens": 16384})
        self.assertEqual(
            chat,
            {
                "tokens": 29,
                "context_tokens": 16384,
                "rendered_hash": "a" * 64,
                "token_hash": "b" * 64,
            },
        )
        self.assertEqual(
            artifact,
            {
                "tokens": 31,
                "context_tokens": 16384,
                "rendered_hash": "c" * 64,
                "token_hash": "d" * 64,
            },
        )
        client.count_text_tokens.assert_called_once_with("hello")
        client.inspect_chat_tokens.assert_called_once_with(
            [{"role": "user", "content": "hello"}],
            tools=[],
            thinking=False,
        )
        client.inspect_artifact_content_tokens.assert_called_once_with(
            [{"role": "user", "content": "artifact"}],
        )

    @mock.patch("orbit.native_server.app.safe_native_capability_manifest")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_server_caches_bounded_native_capability_manifest(self, _mocked_lib, build_manifest) -> None:
        build_manifest.return_value = {
            "schema_version": 1,
            "profile_id": "orbit-gemma4-native-v1",
            "status": "verified",
            "behavior_enforced": False,
        }
        client = NativeLlamaClient(self._paths(), NativeClientConfig())

        server = OrbitNativeServer(client=client, model_alias="m")

        self.assertEqual(server.native_backend_capabilities["status"], "verified")
        build_manifest.assert_called_once_with(client, final_system_prompt=mock.ANY)


if __name__ == "__main__":
    unittest.main()
