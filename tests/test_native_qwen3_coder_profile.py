from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest import mock

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.events import NativeTimings
from orbit.native_llama.model_profiles import NativeModelProfile, QWEN3_CODER_PROFILE_ID
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.qwen3_coder import QWEN3_CODER_ARTIFACT_SYSTEM_PROMPT


def _coder_profile() -> NativeModelProfile:
    return NativeModelProfile(
        profile_id=QWEN3_CODER_PROFILE_ID,
        family="qwen3-coder",
        model_name="Qwen3-Coder-30B-A3B-Instruct",
        architecture="qwen3moe",
        renderer="llama.cpp-jinja",
        reasoning_protocol="none",
        tool_call_protocol="qwen3-coder-xml",
        history_serialization="qwen3-coder-chatml",
        verified=True,
        failure_reason=None,
        template_source="gguf-embedded-official",
        template_sha256="a" * 64,
        thinking_supported=False,
        mtp_supported=False,
        gemma_prefix_reuse_supported=False,
        verified_quantization="Q4_K_M",
        artifact_content_protocol="qwen3-coder-json-string-v1",
    )


class NativeQwen3CoderProfileTests(unittest.TestCase):
    def _client(self) -> NativeLlamaClient:
        paths = NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/qwen3-coder.gguf"),
            model_id="legacy-path",
        )
        with mock.patch("orbit.native_llama.client.LlamaLibrary"):
            client = NativeLlamaClient(paths, NativeClientConfig(qwen_route_prefix_reuse_enabled=True))
        client.model_profile = _coder_profile()
        client._model = 1  # type: ignore[assignment]
        client._vocab = 1  # type: ignore[assignment]
        client.apply_chat_template = mock.Mock(return_value="<|im_start|>assistant\n")  # type: ignore[method-assign]
        client._create_qwen3_coder_artifact_sampler = mock.Mock(return_value=123)  # type: ignore[method-assign]
        return client

    def _complete(
        self,
        client: NativeLlamaClient,
        generated: str,
        *,
        output_tokens: int = 8,
        max_tokens: int = 128,
        cancelled: bool = False,
    ):
        timings = NativeTimings(20, output_tokens, 0, 20, 1.0, 2.0, cancelled)
        emitted: list[str] = []

        def complete(prompt: str, **kwargs):
            self.assertTrue(prompt.endswith('<|im_start|>assistant\n"'))
            self.assertFalse(kwargs["allow_mtp_experimental"])
            self.assertFalse(kwargs["thinking"])
            self.assertIsNone(kwargs.get("qwen_route_anchor_plan"))
            self.assertEqual(kwargs["sampler_override"], 123)
            self.assertEqual(kwargs["utf8_errors"], "strict")
            kwargs["on_token"](generated)
            return timings

        messages = [
            {"role": "system", "content": "generic artifact instruction"},
            {"role": "user", "content": "generate the selected artifact"},
        ]
        with mock.patch.object(client, "complete_prompt", side_effect=complete):
            result = client.complete_artifact_text(
                messages,
                max_tokens=max_tokens,
                on_token=emitted.append,
            )
        rendered_messages = client.apply_chat_template.call_args.args[0]  # type: ignore[union-attr]
        self.assertEqual(rendered_messages[0]["content"], QWEN3_CODER_ARTIFACT_SYSTEM_PROMPT)
        self.assertEqual(messages[0]["content"], "generic artifact instruction")
        return result, emitted

    @staticmethod
    def _wire(content: str) -> str:
        return json.dumps(content, ensure_ascii=False)[1:]

    def test_generative_text_formats_preserve_content_without_outer_fence(self) -> None:
        fixtures = (
            "<!doctype html>\n<script>const x = 1;</script>",
            "export const ready = true;\n",
            "def ready():\n\treturn 'caffè 日本語 🚀'\n",
            '{"enabled":true,"label":"日本語"}',
            "# Notes\n\nPlain text.\n",
            "server:\n  port: 8080\n",
            "[server]\nport = 8080\n",
            "<tool_call>{\"tool\":\"inert\"}</tool_call>\n",
        )
        for content in fixtures:
            with self.subTest(content=content[:24]):
                client = self._client()
                result, emitted = self._complete(client, self._wire(content))
                self.assertEqual(result.content, content)
                self.assertEqual("".join(emitted), content)

    def test_internal_and_terminal_markdown_fences_are_preserved(self) -> None:
        client = self._client()
        content = "# Example\n\n```python\nprint('ok')\n```\n"

        result, _ = self._complete(client, self._wire(content))

        self.assertEqual(result.content, content)
        self.assertEqual(result.content.count("```"), 2)

    def test_no_whitespace_newline_or_unicode_normalization_occurs(self) -> None:
        client = self._client()
        content = "\t caffè e\u0301 日本語 🚀 \r\n\r\n"

        result, _ = self._complete(client, self._wire(content))

        self.assertEqual(result.content, content)

    def test_missing_terminal_boundary_fails_closed(self) -> None:
        client = self._client()

        with self.assertRaisesRegex(RuntimeError, "malformed JSON-string framing"):
            self._complete(client, "const ready = true;")

    def test_malformed_terminal_boundary_fails_closed(self) -> None:
        client = self._client()

        with self.assertRaisesRegex(RuntimeError, "data outside JSON-string framing"):
            self._complete(client, self._wire("const ready = true;") + "unexpected")

    def test_content_ending_in_transport_like_fence_cannot_consume_framing(self) -> None:
        client = self._client()
        content = "# Example\n\n```python\nprint('ok')\n```"

        result, _ = self._complete(client, self._wire(content))

        self.assertEqual(result.content, content)

    def test_quotes_backslashes_controls_and_protocol_like_text_are_reversible(self) -> None:
        client = self._client()
        content = 'JSON={"tool":"noop"}\nXML=<|im_end|>\nPATH=C:\\\\tmp\\file\n\tend\r\n'

        result, _ = self._complete(client, self._wire(content))

        self.assertEqual(result.content, content)

    def test_invalid_unicode_scalar_fails_closed(self) -> None:
        client = self._client()

        with self.assertRaisesRegex(RuntimeError, "invalid UTF-8 content"):
            self._complete(client, '\\ud800"')

    def test_length_termination_does_not_release_generated_content(self) -> None:
        client = self._client()

        result, emitted = self._complete(client, self._wire("partial"), output_tokens=128, max_tokens=128)

        self.assertEqual(result.content, "")
        self.assertEqual(emitted, [])

    def test_cancelled_generation_does_not_release_generated_content(self) -> None:
        client = self._client()

        result, emitted = self._complete(client, self._wire("partial"), cancelled=True)

        self.assertEqual(result.content, "")
        self.assertEqual(emitted, [])

    def test_profile_sampler_is_freed_after_success(self) -> None:
        client = self._client()

        self._complete(client, self._wire("complete"))

        client.lib.lib.llama_sampler_free.assert_called_once_with(123)

    def test_profile_sampler_is_freed_after_generation_error(self) -> None:
        client = self._client()

        with mock.patch.object(client, "complete_prompt", side_effect=RuntimeError("backend failed")):
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                client.complete_artifact_text(
                    [
                        {"role": "system", "content": "generic artifact instruction"},
                        {"role": "user", "content": "generate the selected artifact"},
                    ],
                    max_tokens=128,
                )

        client.lib.lib.llama_sampler_free.assert_called_once_with(123)

    def test_profile_sampler_is_freed_after_framing_error(self) -> None:
        client = self._client()

        with self.assertRaisesRegex(RuntimeError, "malformed JSON-string framing"):
            self._complete(client, "unterminated")

        client.lib.lib.llama_sampler_free.assert_called_once_with(123)

    def test_sampler_creation_owns_grammar_and_greedy_sampler(self) -> None:
        client = self._client()
        lib = client.lib.lib
        lib.llama_sampler_chain_init.return_value = 101
        lib.llama_sampler_init_grammar.return_value = 102
        lib.llama_sampler_init_greedy.return_value = 103

        sampler = NativeLlamaClient._create_qwen3_coder_artifact_sampler(client)

        self.assertEqual(sampler.value, 101)
        lib.llama_sampler_chain_add.assert_has_calls([mock.call(101, 102), mock.call(101, 103)])
        lib.llama_sampler_free.assert_not_called()

    def test_sampler_creation_frees_chain_when_grammar_initialization_fails(self) -> None:
        client = self._client()
        lib = client.lib.lib
        lib.llama_sampler_chain_init.return_value = 101
        lib.llama_sampler_init_grammar.return_value = None

        with self.assertRaisesRegex(RuntimeError, "failed to create Qwen3-Coder artifact grammar"):
            NativeLlamaClient._create_qwen3_coder_artifact_sampler(client)

        lib.llama_sampler_free.assert_called_once_with(101)

    def test_sampler_creation_frees_chain_when_greedy_initialization_fails(self) -> None:
        client = self._client()
        lib = client.lib.lib
        lib.llama_sampler_chain_init.return_value = 101
        lib.llama_sampler_init_grammar.return_value = 102
        lib.llama_sampler_init_greedy.return_value = None

        with self.assertRaisesRegex(RuntimeError, "failed to create Qwen3-Coder artifact greedy sampler"):
            NativeLlamaClient._create_qwen3_coder_artifact_sampler(client)

        lib.llama_sampler_free.assert_called_once_with(101)

    def test_profile_sampler_is_not_accepted_twice(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1
        client._session.sampler = 11
        lib = client.lib.lib
        lib.llama_sampler_sample.return_value = 151645
        lib.llama_vocab_is_eog.return_value = True
        lib.llama_time_us.side_effect = [0, 1000]

        generated, _elapsed, cancelled = client._generate_from_current_context(
            max_tokens=1,
            sampler_override=123,
            utf8_errors="strict",
        )

        self.assertEqual(generated, 0)
        self.assertFalse(cancelled)
        lib.llama_sampler_accept.assert_not_called()

    def test_default_sampler_keeps_existing_accept_behavior(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1
        client._session.sampler = 11
        lib = client.lib.lib
        lib.llama_sampler_sample.return_value = 151645
        lib.llama_vocab_is_eog.return_value = True
        lib.llama_time_us.side_effect = [0, 1000]

        client._generate_from_current_context(max_tokens=1)

        lib.llama_sampler_accept.assert_called_once_with(11, 151645)

    def test_profile_decoder_rejects_invalid_utf8_token_piece(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1
        client._session.sampler = 11
        client._token_to_bytes = mock.Mock(return_value=b"\xa1")  # type: ignore[method-assign]
        lib = client.lib.lib
        lib.llama_sampler_sample.return_value = 94
        lib.llama_vocab_is_eog.return_value = False

        with self.assertRaises(UnicodeDecodeError):
            client._generate_from_current_context(
                max_tokens=1,
                sampler_override=123,
                utf8_errors="strict",
            )

    def test_default_decoder_keeps_existing_replacement_behavior(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1
        client._session.sampler = 11
        client._token_to_bytes = mock.Mock(return_value=b"\xa1")  # type: ignore[method-assign]
        lib = client.lib.lib
        lib.llama_sampler_sample.return_value = 94
        lib.llama_vocab_is_eog.return_value = False
        lib.llama_decode.return_value = 0
        lib.llama_time_us.side_effect = [0, 1000]
        emitted: list[str] = []

        generated, _elapsed, cancelled = client._generate_from_current_context(
            max_tokens=1,
            on_token=emitted.append,
        )

        self.assertEqual(generated, 1)
        self.assertFalse(cancelled)
        self.assertEqual(emitted, ["\ufffd"])

    def test_unexpected_message_sequence_fails_closed(self) -> None:
        client = self._client()

        with self.assertRaisesRegex(RuntimeError, "unexpected message sequence"):
            client.complete_artifact_text(
                [{"role": "user", "content": "missing system"}],
                max_tokens=128,
            )

    def test_qwen36_route_checkpoint_is_not_eligible(self) -> None:
        client = self._client()
        client._model_metadata_identity = {"general.file_type": "15"}

        status = client.qwen_route_prefix_reuse_status()

        self.assertFalse(status["enabled"])
        self.assertEqual(status["profile_identity"], QWEN3_CODER_PROFILE_ID)

    def test_thinking_is_rejected_before_rendering(self) -> None:
        client = self._client()

        with self.assertRaisesRegex(RuntimeError, "thinking is unsupported"):
            client._thinking_enabled(True)

    def test_profile_history_serialization_does_not_use_qwen36_rewrite(self) -> None:
        client = self._client()
        messages = [
            {"role": "system", "content": "initial"},
            {"role": "user", "content": "request"},
            {"role": "system", "content": "bounded evidence"},
        ]

        serialized = client._serialize_profile_messages(messages)

        self.assertEqual(serialized, messages)
        self.assertIs(serialized, messages)

    def test_multimodal_projector_is_rejected_for_unqualified_profile(self) -> None:
        client = self._client()
        client.paths = NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/qwen3-coder.gguf"),
            mmproj_model=Path("/models/unqualified-mmproj.gguf"),
            model_id="legacy-path",
        )

        with self.assertRaisesRegex(RuntimeError, "multimodal input is unsupported"):
            client._initialize_multimodal_context()


if __name__ == "__main__":
    unittest.main()
