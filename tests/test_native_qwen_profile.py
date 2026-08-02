from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest import mock

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.events import NativeTimings
from orbit.native_llama.model_profiles import NativeModelProfile, QWEN36_PROFILE_ID
from orbit.native_llama.paths import NativeLlamaPaths


class _FakeChatBridge:
    build_identity = {"schema_version": 1}

    def __init__(self) -> None:
        self.render_calls: list[tuple[list[dict[str, object]], list[dict[str, object]], bool]] = []

    def render(self, _context, messages, tools, *, thinking: bool):
        self.render_calls.append((messages, tools, thinking))
        return {
            "prompt": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "format": "peg-native",
            "supports_thinking": True,
            "additional_stops": [],
        }

    def parse(self, _context, generated_text: str, *, partial: bool):
        del partial
        if "<tool_call>" in generated_text:
            return {
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "exec_shell_full_command", "arguments": '{"command":"pwd"}'},
                    }
                ],
            }
        reasoning = "private route analysis" if "<think>" in generated_text else ""
        content = generated_text.split("</think>", maxsplit=1)[-1].strip() if reasoning else generated_text
        return {"content": content, "reasoning_content": reasoning, "tool_calls": []}


def _qwen_profile() -> NativeModelProfile:
    return NativeModelProfile(
        profile_id=QWEN36_PROFILE_ID,
        family="qwen3.6",
        model_name="Qwen3.6-35B-A3B",
        architecture="qwen35moe",
        renderer="llama.cpp-jinja",
        reasoning_protocol="qwen-think",
        tool_call_protocol="qwen3.6-xml",
        history_serialization="qwen-leading-system-only",
        verified=True,
        failure_reason=None,
        template_source="gguf-embedded-official",
        template_sha256="a" * 64,
        thinking_supported=True,
        mtp_supported=False,
        gemma_prefix_reuse_supported=False,
    )


class NativeQwenProfileTests(unittest.TestCase):
    def _client(self) -> NativeLlamaClient:
        paths = NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/qwen.gguf"),
            model_id="legacy-path",
        )
        with mock.patch("orbit.native_llama.client.LlamaLibrary"):
            client = NativeLlamaClient(paths, NativeClientConfig())
        client.model_profile = _qwen_profile()
        client.chat_bridge = _FakeChatBridge()  # type: ignore[assignment]
        client._chat_bridge_context = 1  # type: ignore[assignment]
        client._model = 1  # type: ignore[assignment]
        client._vocab = 1  # type: ignore[assignment]
        return client

    def test_render_uses_bridge_and_preserves_assistant_tool_history(self) -> None:
        client = self._client()
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "tool", "name": "x", "content": "ok"},
        ]

        prompt = client.apply_chat_template(messages, tools=[{"type": "function", "function": {"name": "x"}}], thinking=False)

        self.assertIn("</think>", prompt)
        bridge = client.chat_bridge
        assert isinstance(bridge, _FakeChatBridge)
        rendered_messages, rendered_tools, thinking = bridge.render_calls[0]
        self.assertEqual(rendered_messages, messages)
        self.assertEqual(rendered_tools[0]["function"]["name"], "x")
        self.assertFalse(thinking)

    def test_nonleading_system_evidence_is_preserved_in_place_as_qwen_input(self) -> None:
        client = self._client()
        messages = [
            {"role": "system", "content": "initial"},
            {"role": "user", "content": "request"},
            {"role": "system", "content": "exact evidence"},
        ]

        client.apply_chat_template(messages, thinking=False)

        bridge = client.chat_bridge
        assert isinstance(bridge, _FakeChatBridge)
        rendered_messages = bridge.render_calls[0][0]
        self.assertEqual([message["role"] for message in rendered_messages], ["system", "user", "user"])
        self.assertEqual(rendered_messages[-1]["content"], "exact evidence")
        self.assertEqual(messages[-1]["role"], "system")

    def test_reasoning_is_never_emitted_or_returned_as_visible_content(self) -> None:
        client = self._client()
        emitted: list[str] = []
        timings = NativeTimings(10, 5, 0, 10, 1.0, 2.0, False)

        def complete(_messages, **kwargs):
            kwargs["on_token"]("<think>private route analysis</think>\n\n{\"route\":\"CHAT\"}")
            return timings

        with (
            mock.patch.object(client, "complete_chat", side_effect=complete),
            mock.patch.object(client, "_content_token_count", return_value=3),
        ):
            result = client.complete_chat_text(
                [{"role": "user", "content": "hello"}],
                max_tokens=16,
                thinking=False,
                on_token=emitted.append,
            )

        self.assertEqual(result.content, '{"route":"CHAT"}')
        self.assertEqual(result.reasoning_content, "private route analysis")
        self.assertEqual("".join(emitted), result.content)
        self.assertNotIn("private", "".join(emitted))

    def test_qwen_tool_call_arguments_remain_exact_json_for_canonical_gate(self) -> None:
        client = self._client()

        parsed = client._parse_profile_output(
            "<tool_call><function=exec_shell_full_command><parameter=command>pwd</parameter></function></tool_call>",
            partial=False,
        )

        self.assertEqual(parsed.content, "")
        self.assertEqual(len(parsed.tool_calls), 1)
        function = parsed.tool_calls[0]["function"]
        self.assertEqual(function["name"], "exec_shell_full_command")
        self.assertEqual(json.loads(function["arguments"]), {"command": "pwd"})

    def test_tools_available_final_prose_is_emitted_after_structural_parse(self) -> None:
        client = self._client()
        emitted: list[str] = []
        timings = NativeTimings(10, 4, 0, 10, 1.0, 2.0, False)

        def complete(_messages, **kwargs):
            kwargs["on_token"]("The exact tool output is QWEN_XML_OK.")
            return timings

        with (
            mock.patch.object(client, "complete_chat", side_effect=complete),
            mock.patch.object(client, "_content_token_count", return_value=0),
        ):
            result = client.complete_chat_text(
                [{"role": "tool", "name": "x", "content": "QWEN_XML_OK"}],
                max_tokens=16,
                tools=[{"type": "function", "function": {"name": "x"}}],
                thinking=False,
                on_token=emitted.append,
            )

        self.assertEqual(result.content, "The exact tool output is QWEN_XML_OK.")
        self.assertEqual("".join(emitted), result.content)

    def test_malformed_bridge_tool_call_fails_closed(self) -> None:
        client = self._client()
        assert client.chat_bridge is not None
        client.chat_bridge.parse = mock.Mock(  # type: ignore[method-assign]
            return_value={"content": "", "reasoning_content": "", "tool_calls": [{"function": {"name": "x", "arguments": {}}}]}
        )

        with self.assertRaisesRegex(RuntimeError, "invalid tool arguments"):
            client._parse_profile_output("bad", partial=False)

    def test_qwen_profile_disables_mtp_and_gemma_prefix_reuse(self) -> None:
        client = self._client()

        self.assertFalse(client.final_prefix_experiment_status()["enabled"])
        self.assertFalse(client.model_profile.mtp_supported)
        self.assertFalse(client.model_profile.gemma_prefix_reuse_supported)

    def test_rejected_partial_memory_removal_falls_back_to_cold_prefill(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1  # type: ignore[assignment]
        client._session.cached_prompt_tokens = [1, 2, 3, 4]
        native = client.lib.lib
        native.llama_get_memory.return_value = 2
        native.llama_memory_seq_rm.return_value = False

        reused = client._prepare_memory_for_prompt([1, 2, 9])

        self.assertEqual(reused, 0)
        native.llama_memory_seq_rm.assert_called_once_with(2, 0, 2, -1)
        native.llama_memory_clear.assert_called_once_with(2, True)

    def test_supported_partial_memory_removal_preserves_lcp(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1  # type: ignore[assignment]
        client._session.cached_prompt_tokens = [1, 2, 3, 4]
        native = client.lib.lib
        native.llama_get_memory.return_value = 2
        native.llama_memory_seq_rm.return_value = True

        reused = client._prepare_memory_for_prompt([1, 2, 9])

        self.assertEqual(reused, 2)
        native.llama_memory_clear.assert_not_called()

    def test_reset_clears_qwen_continuation_and_parser_projection_state(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1  # type: ignore[assignment]
        client._session.continuation_ready = True
        client._active_profile_render = {"prompt": "old"}
        client._profile_last_raw_output = "private and visible output"
        client._profile_last_parsed_content = "visible output"
        client.lib.lib.llama_get_memory.return_value = 2

        client.reset_session_state()

        self.assertFalse(client._session.continuation_ready)
        self.assertIsNone(client._active_profile_render)
        self.assertEqual(client._profile_last_raw_output, "")
        self.assertEqual(client._profile_last_parsed_content, "")


if __name__ == "__main__":
    unittest.main()
