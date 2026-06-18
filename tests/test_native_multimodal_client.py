from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.events import NativeTimings
from orbit.native_llama.paths import NativeLlamaPaths


class NativeMultimodalClientTests(unittest.TestCase):
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
    def test_complete_chat_routes_media_to_multimodal_prefill(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        client.supports_vision = True
        expected = NativeTimings(10, 3, 0, 10, 12.0, 34.0, False)

        with (
            mock.patch.object(client, "apply_chat_template", return_value="<bos>prompt") as apply_template,
            mock.patch.object(client, "_complete_prompt_multimodal", return_value=expected) as complete_multimodal,
        ):
            timings = client.complete_chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}},
                        ],
                    }
                ],
                max_tokens=32,
            )

        self.assertEqual(timings, expected)
        apply_template.assert_called_once()
        complete_multimodal.assert_called_once()

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_complete_chat_rejects_image_when_multimodal_support_is_missing(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        client.supports_vision = False

        with self.assertRaisesRegex(RuntimeError, "image input is not supported"):
            client.complete_chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}},
                        ],
                    }
                ],
                max_tokens=32,
            )

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_multimodal_chunk_batch_size_is_capped_by_ubatch(self, _mocked_lib) -> None:
        client = NativeLlamaClient(
            self._paths(),
            NativeClientConfig(batch_size=256, ubatch_size=128),
        )

        self.assertEqual(client._multimodal_chunk_batch_size(), 128)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_multimodal_chunk_batch_size_is_at_least_one(self, _mocked_lib) -> None:
        client = NativeLlamaClient(
            self._paths(),
            NativeClientConfig(batch_size=0, ubatch_size=0),
        )

        self.assertEqual(client._multimodal_chunk_batch_size(), 1)


if __name__ == "__main__":
    unittest.main()
