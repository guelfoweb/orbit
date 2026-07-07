from __future__ import annotations

import unittest
from unittest import mock
import json
import os
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.llama_server import (
    LlamaServerBackend,
    _enrich_model_info_with_props,
    _parse_chat_result,
    _parse_chat_stream,
    _parse_model_info,
    _parse_native_stream,
)
from orbit.backend.base import ChatResult
from orbit.backend.payloads import ChatPayloadOptions, build_chat_payload
from orbit.backend import model_names
from orbit.runtime.kv_diag import model_call_context
from urllib.error import URLError


class FakeStream:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def __iter__(self):
        return iter(line.encode("utf-8") for line in self.lines)


class FakeNativeStreamWithTrailingNoise:
    def __iter__(self):
        yield b'event: delta\n'
        yield b'data: {"text":"ok"}\n'
        yield b'\n'
        yield b'event: metrics\n'
        yield b'data: {"usage":{"prompt_tokens":10,"completion_tokens":1},"timings":{"predicted_per_second":2.0}}\n'
        yield b'\n'
        yield b'event: done\n'
        yield b'data: {"finish_reason":"stop","model":"gemma4"}\n'
        yield b'\n'
        while True:
            yield b': keep-alive\n'


class LlamaServerBackendTests(unittest.TestCase):
    def test_backend_connection_error_mentions_backend_server(self) -> None:
        backend = LlamaServerBackend(base_url="http://127.0.0.1:12120", model="fake", timeout=1)

        with self.assertRaisesRegex(Exception, "cannot connect to backend server at http://127.0.0.1:12120"):
            with mock.patch("orbit.backend.llama_server.urlopen", side_effect=URLError("[Errno 111] Connection refused")):
                backend._get_json("/health")

    def test_continue_current_uses_native_continue_stream_endpoint_for_non_stream_call(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1, thinking=True)
                self.seen_path: str | None = None
                self.seen_payload: dict[str, object] | None = None

            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "orbit-native"}

            def _post_native_stream(self, path: str, payload: dict[str, Any], *, on_delta, on_progress):
                self.seen_path = path
                self.seen_payload = payload
                return _parse_native_stream(
                    FakeStream(
                        [
                            'event: delta\n',
                            'data: {"text":"continued"}\n',
                            '\n',
                            'event: metrics\n',
                            'data: {"usage":{"prompt_tokens":0,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":0}},"timings":{"predicted_per_second":2.0}}\n',
                            '\n',
                            'event: done\n',
                            'data: {"finish_reason":"stop","model":"gemma4"}\n',
                            '\n',
                        ]
                    ),
                    on_delta=on_delta,
                    on_progress=on_progress,
                )

        backend = Backend()
        result = backend.continue_current(max_tokens=24)

        self.assertEqual(backend.seen_path, "/chat/continue/stream")
        self.assertEqual(backend.seen_payload, {"max_tokens": 24, "thinking": True, "stream": True})
        self.assertEqual(result.content, "continued")
        self.assertEqual(result.finish_reason, "stop")

    def test_chat_uses_native_stream_endpoint_for_non_stream_call(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1, thinking=True)
                self.seen_path: str | None = None
                self.seen_payload: dict[str, object] | None = None

            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "orbit-native"}

            def _post_native_stream(self, path: str, payload: dict[str, Any], *, on_delta, on_progress):
                self.seen_path = path
                self.seen_payload = payload
                return _parse_native_stream(
                    FakeStream(
                        [
                            'event: delta\n',
                            'data: {"text":"hello"}\n',
                            '\n',
                            'event: metrics\n',
                            'data: {"usage":{"prompt_tokens":10,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":0}},"timings":{"predicted_per_second":2.0}}\n',
                            '\n',
                            'event: done\n',
                            'data: {"finish_reason":"stop","model":"gemma4"}\n',
                            '\n',
                        ]
                    ),
                    on_delta=on_delta,
                    on_progress=on_progress,
                )

        backend = Backend()
        result = backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=32)

        self.assertEqual(backend.seen_path, "/chat/stream")
        self.assertEqual(result.content, "hello")
        self.assertEqual(result.finish_reason, "stop")

    def test_enrich_model_info_with_native_props_adds_context_and_capabilities(self) -> None:
        enriched = _enrich_model_info_with_props(
            None,
            {
                "backend": "orbit-native",
                "model_id": "gemma4-12b-it-q4km",
                "model_path": "/models/target.gguf",
                "ctx_size": 8192,
                "supports_vision": True,
                "supports_audio": True,
                "multimodal_available": True,
            },
        )

        assert enriched is not None
        self.assertEqual(enriched.context_length, 8192)
        self.assertIn("completion", enriched.capabilities)
        self.assertIn("vision", enriched.capabilities)
        self.assertIn("audio", enriched.capabilities)
        self.assertIn("multimodal", enriched.capabilities)

    def test_chat_stream_uses_native_stream_for_orbit_backend_even_with_tools(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1)
                self.path: str | None = None
                self.stream_kind: str | None = None

            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "orbit-native"}

            def _post_native_stream(self, path, payload, *, on_delta, on_progress):
                self.path = path
                self.stream_kind = "native"
                return _parse_native_stream(
                    FakeStream(
                        [
                            'event: progress.prefill\n',
                            'data: {"current":10,"total":20,"percent":50}\n',
                            '\n',
                            'event: done\n',
                            'data: {"finish_reason":"tool_calls"}\n',
                            '\n',
                        ]
                    ),
                    on_delta=on_delta,
                    on_progress=on_progress,
                )

            def _post_stream(self, path, payload, *, on_delta):
                self.path = path
                self.stream_kind = "openai"
                raise AssertionError("openai stream should not be used")

        progress: list[tuple[str, int, int, int]] = []
        backend = Backend()
        backend.chat_stream(
            [{"role": "user", "content": "read note.txt"}],
            temperature=0,
            max_tokens=32,
            tools=[{"type": "function", "function": {"name": "exec_shell_full_command"}}],
            on_delta=lambda _text: None,
            on_progress=lambda item: progress.append((item.phase, item.current, item.total, item.percent)),
        )

        self.assertEqual(backend.stream_kind, "native")
        self.assertEqual(backend.path, "/chat/stream")
        self.assertEqual(progress, [("prefill", 10, 20, 50)])

    def test_chat_payload_enables_prompt_cache(self) -> None:
        payload = build_chat_payload(
            ChatPayloadOptions(
                model="gemma4",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=32,
            )
        )

        self.assertIs(payload["cache_prompt"], True)
        self.assertNotIn("stream", payload)

    def test_server_tools_are_cached(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1)
                self.calls = 0
                self.paths: list[str] = []

            def _get_json(self, path: str):
                self.calls += 1
                self.paths.append(path)
                return [{"tool": "read_file", "definition": {"type": "function"}}]

        backend = Backend()

        self.assertEqual(len(backend.server_tools()), 1)
        self.assertEqual(len(backend.server_tools()), 1)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(backend.paths, ["/tools"])

    def test_chat_payload_adds_stream_and_tool_options(self) -> None:
        payload = build_chat_payload(
            ChatPayloadOptions(
                model="gemma4",
                messages=[{"role": "user", "content": "list files"}],
                temperature=0,
                max_tokens=32,
                tools=[{"type": "function", "function": {"name": "list_files"}}],
                stream=True,
            )
        )

        self.assertIs(payload["cache_prompt"], True)
        self.assertIs(payload["stream"], True)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertIs(payload["parallel_tool_calls"], False)
        self.assertIs(payload["parse_tool_calls"], True)

    def test_chat_payload_carries_thinking_flag(self) -> None:
        payload = build_chat_payload(
            ChatPayloadOptions(
                model="gemma4",
                messages=[{"role": "user", "content": "think"}],
                temperature=0,
                max_tokens=32,
                thinking=True,
            )
        )

        self.assertTrue(payload["thinking"])

    def test_route_prefix_anchor_payload_is_limited_to_route_tools_on(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1)
                self.payloads: list[dict[str, object]] = []

            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "orbit-native"}

            def _post_native_stream(self, _path: str, payload: dict[str, object], *, on_delta, on_progress) -> ChatResult:
                self.payloads.append(payload)
                return ChatResult(
                    content="ok",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=1,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = Backend()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORBIT_KV_PREFIX_ANCHOR", None)
            os.environ.pop("ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT", None)
            backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)
            with model_call_context(phase="route", tools_mode="on"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)
            with model_call_context(phase="route", tools_mode="off"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)
            with model_call_context(phase="final_from_tool", tools_mode="on"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)

        self.assertNotIn("route_prefix_anchor", backend.payloads[0])
        self.assertTrue(backend.payloads[1]["route_prefix_anchor"])
        self.assertNotIn("route_prefix_anchor", backend.payloads[2])
        self.assertNotIn("route_prefix_anchor", backend.payloads[3])

    def test_native_kv_diag_payload_carries_phase_only_when_enabled(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1)
                self.payloads: list[dict[str, object]] = []

            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "orbit-native"}

            def _post_native_stream(self, _path: str, payload: dict[str, object], *, on_delta, on_progress) -> ChatResult:
                self.payloads.append(payload)
                return ChatResult(
                    content="ok",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=1,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = Backend()
        with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "0"}, clear=False):
            with model_call_context(phase="route", tools_mode="on"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)
        with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1"}, clear=False):
            with model_call_context(phase="final_from_tool", tools_mode="on"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)

        self.assertNotIn("_orbit_kv_phase", backend.payloads[0])
        self.assertEqual(backend.payloads[1]["_orbit_kv_phase"], "final_from_tool")
        self.assertEqual(backend.payloads[1]["_orbit_kv_tools_mode"], "on")

    def test_allow_mtp_false_payload_is_limited_to_native_tools_final_fallbacks(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1)
                self.payloads: list[dict[str, object]] = []

            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "orbit-native"}

            def _post_native_stream(self, _path: str, payload: dict[str, object], *, on_delta, on_progress) -> ChatResult:
                self.payloads.append(payload)
                return ChatResult(
                    content="ok",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=1,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = Backend()
        backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)
        with model_call_context(phase="chat_final_retry", tools_mode="on"):
            backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)
        with model_call_context(phase="final_from_tool", tools_mode="on"):
            backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)
        with model_call_context(phase="final_from_tool_retry", tools_mode="on"):
            backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)
        with model_call_context(phase="chat_final_retry", tools_mode="off"):
            backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)

        self.assertNotIn("allow_mtp_experimental", backend.payloads[0])
        self.assertFalse(backend.payloads[1]["allow_mtp_experimental"])
        self.assertFalse(backend.payloads[2]["allow_mtp_experimental"])
        self.assertFalse(backend.payloads[3]["allow_mtp_experimental"])
        self.assertNotIn("allow_mtp_experimental", backend.payloads[4])

    def test_allow_mtp_payload_is_not_sent_to_non_native_backend(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1)
                self.payload: dict[str, object] | None = None

            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "openai-compatible"}

            def _post_json(self, _path: str, payload: dict[str, object]) -> dict[str, object]:
                self.payload = payload
                return {
                    "model": "fake",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }

        backend = Backend()
        with model_call_context(phase="chat_final_retry", tools_mode="on"):
            backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)

        self.assertIsNotNone(backend.payload)
        assert backend.payload is not None
        self.assertNotIn("allow_mtp_experimental", backend.payload)

    def test_route_prefix_anchor_legacy_experiment_flag_still_enables_auto_mode(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1)
                self.payload: dict[str, object] | None = None

            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "orbit-native"}

            def _post_native_stream(self, _path: str, payload: dict[str, object], *, on_delta, on_progress) -> ChatResult:
                self.payload = payload
                return ChatResult(
                    content="ok",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=1,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = Backend()
        with mock.patch.dict(os.environ, {"ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT": "1"}, clear=False):
            os.environ.pop("ORBIT_KV_PREFIX_ANCHOR", None)
            with model_call_context(phase="route", tools_mode="on"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)

        self.assertIsNotNone(backend.payload)
        assert backend.payload is not None
        self.assertTrue(backend.payload["route_prefix_anchor"])

    def test_route_prefix_anchor_off_wins_over_legacy_flag(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1)
                self.payload: dict[str, object] | None = None

            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "orbit-native"}

            def _post_native_stream(self, _path: str, payload: dict[str, object], *, on_delta, on_progress) -> ChatResult:
                self.payload = payload
                return ChatResult(
                    content="ok",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=1,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = Backend()
        with mock.patch.dict(
            os.environ,
            {"ORBIT_KV_PREFIX_ANCHOR": "off", "ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT": "1"},
            clear=False,
        ):
            with model_call_context(phase="route", tools_mode="on"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)

        self.assertIsNotNone(backend.payload)
        assert backend.payload is not None
        self.assertNotIn("route_prefix_anchor", backend.payload)

    def test_route_prefix_anchor_invalid_mode_falls_back_to_off(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1)
                self.payload: dict[str, object] | None = None

            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "orbit-native"}

            def _post_native_stream(self, _path: str, payload: dict[str, object], *, on_delta, on_progress) -> ChatResult:
                self.payload = payload
                return ChatResult(
                    content="ok",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=1,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = Backend()
        with mock.patch.dict(os.environ, {"ORBIT_KV_PREFIX_ANCHOR": "maybe"}, clear=False):
            os.environ.pop("ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT", None)
            with model_call_context(phase="route", tools_mode="on"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)

        self.assertIsNotNone(backend.payload)
        assert backend.payload is not None
        self.assertNotIn("route_prefix_anchor", backend.payload)

    def test_route_prefix_anchor_payload_is_not_sent_to_non_native_backend(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1)
                self.payload: dict[str, object] | None = None

            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "openai-compatible"}

            def _post_json(self, _path: str, payload: dict[str, object]) -> dict[str, object]:
                self.payload = payload
                return {
                    "model": "fake",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }

        backend = Backend()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORBIT_KV_PREFIX_ANCHOR", None)
            os.environ.pop("ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT", None)
            with model_call_context(phase="route", tools_mode="on"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)

        self.assertIsNotNone(backend.payload)
        assert backend.payload is not None
        self.assertNotIn("route_prefix_anchor", backend.payload)

    def test_parse_chat_result_extracts_content_and_metrics(self) -> None:
        result = _parse_chat_result(
            {
                "model": "gemma4",
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 8},
                },
                "timings": {
                    "prompt_per_second": 12.5,
                    "predicted_per_second": 3.4,
                },
            }
        )

        self.assertEqual(result.content, "hello")
        self.assertEqual(result.model, "gemma4")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(result.completion_tokens, 3)
        self.assertEqual(result.cached_tokens, 8)
        self.assertEqual(result.prompt_tokens_per_second, 12.5)
        self.assertEqual(result.generation_tokens_per_second, 3.4)

    def test_parse_chat_result_extracts_tool_calls(self) -> None:
        result = _parse_chat_result(
            {
                "model": "gemma4",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "list_files", "arguments": "{\"path\":\".\"}"},
                                }
                            ],
                        },
                    }
                ],
            }
        )

        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.tool_calls[0]["id"], "call-1")

    def test_parse_chat_result_converts_raw_tool_call_content(self) -> None:
        result = _parse_chat_result(
            {
                "model": "gemma4",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '<|tool_call>call:exec_shell_command{command:<|"|>cat server-tool-test.txt<|"|>}<tool_call|>',
                        },
                    }
                ],
            }
        )

        self.assertEqual(result.content, "")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "exec_shell_command")
        self.assertEqual(result.tool_calls[0]["function"]["arguments"], '{"command": "cat server-tool-test.txt"}')

    def test_parse_chat_result_converts_raw_tool_call_with_inner_quotes(self) -> None:
        result = _parse_chat_result(
            {
                "model": "gemma4",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '<|tool_call>call:exec_shell_full_command{command:<|"|>strings -a samples/suspicious_dropper_demo.js && grep -E "http://|https://|[0-9]{1,3}\\.[0-9]{1,3}" samples/suspicious_dropper_demo.js | sort | uniq<|"|>}<tool_call|>',
                        },
                    }
                ],
            }
        )

        self.assertEqual(result.content, "")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "exec_shell_full_command")
        self.assertEqual(
            result.tool_calls[0]["function"]["arguments"],
            '{"command": "strings -a samples/suspicious_dropper_demo.js && grep -E \\"http://|https://|[0-9]{1,3}\\\\.[0-9]{1,3}\\" samples/suspicious_dropper_demo.js | sort | uniq"}',
        )

    def test_parse_native_stream_stops_at_done_without_waiting_for_eof(self) -> None:
        deltas: list[str] = []

        result = _parse_native_stream(
            FakeNativeStreamWithTrailingNoise(),
            on_delta=deltas.append,
            on_progress=None,
        )

        self.assertEqual("".join(deltas), "ok")
        self.assertEqual(result.content, "ok")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.model, "gemma4")
        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(result.completion_tokens, 1)

    def test_parse_model_info_extracts_capabilities_and_meta(self) -> None:
        info = _parse_model_info(
            {
                "models": [{"model": "served", "capabilities": ["completion", "multimodal"]}],
                "data": [{"id": "served-id", "meta": {"n_ctx": 8192, "n_params": 12_000_000_000, "size": 7_000_000_000}}],
            }
        )

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.id, "served-id")
        self.assertEqual(info.capabilities, ("completion", "multimodal"))
        self.assertEqual(info.context_length, 8192)
        self.assertEqual(info.parameter_count, 12_000_000_000)
        self.assertEqual(info.size_bytes, 7_000_000_000)

    def test_parse_model_info_resolves_hash_id_from_manifest(self) -> None:
        digest = "c" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "manifests"
            manifest = root / "registry.ollama.ai" / "library" / "gemma4" / "12b"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"layers": [{"digest": f"sha256:{digest}"}]}), encoding="utf-8")
            original = model_names.default_manifest_roots
            model_names.default_manifest_roots = lambda: [root]
            try:
                info = _parse_model_info(
                    {
                        "models": [{"model": f"sha256-{digest}", "capabilities": ["completion"]}],
                        "data": [{"id": f"sha256-{digest}", "meta": {"n_ctx": 8192}}],
                    },
                    model_path=f"/models/blobs/sha256-{digest}",
                )
            finally:
                model_names.default_manifest_roots = original

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.id, "gemma4:12b")

    def test_parse_chat_stream_emits_text_deltas(self) -> None:
        emitted: list[str] = []

        result = _parse_chat_stream(
            FakeStream(
                [
                    'data: {"model":"gemma4","choices":[{"delta":{"content":"hel"},"finish_reason":null}]}\n',
                    'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}\n',
                    "data: [DONE]\n",
                ]
            ),
            on_delta=emitted.append,
        )

        self.assertEqual(emitted, ["hel", "lo"])
        self.assertEqual(result.content, "hello")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.prompt_tokens, 2)

    def test_parse_chat_stream_accumulates_tool_call_deltas(self) -> None:
        result = _parse_chat_stream(
            FakeStream(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"read_","arguments":"{\\"path\\""}}]},"finish_reason":null}]}\n',
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"file","arguments":":\\"note.txt\\"}"}}]},"finish_reason":"tool_calls"}]}\n',
                    "data: [DONE]\n",
                ]
            ),
            on_delta=lambda text: None,
        )

        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "read_file")
        self.assertEqual(result.tool_calls[0]["function"]["arguments"], "{\"path\":\"note.txt\"}")

    def test_parse_chat_stream_suppresses_and_converts_raw_tool_call_content(self) -> None:
        emitted: list[str] = []

        result = _parse_chat_stream(
            FakeStream(
                [
                    'data: {"choices":[{"delta":{"content":"<|tool_"},"finish_reason":null}]}\n',
                    'data: {"choices":[{"delta":{"content":"call>call:exec_shell_command{command:<|\\"|>cat server-tool-test.txt<|\\"|>}<tool_call|>"},"finish_reason":"stop"}]}\n',
                    "data: [DONE]\n",
                ]
            ),
            on_delta=emitted.append,
        )

        self.assertEqual(emitted, [])
        self.assertEqual(result.content, "")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "exec_shell_command")
        self.assertEqual(result.tool_calls[0]["function"]["arguments"], '{"command": "cat server-tool-test.txt"}')

    def test_parse_chat_stream_suppresses_embedded_raw_tool_call_content(self) -> None:
        emitted: list[str] = []

        result = _parse_chat_stream(
            FakeStream(
                [
                    'data: {"choices":[{"delta":{"content":"Need more. <|tool_"},"finish_reason":null}]}\n',
                    'data: {"choices":[{"delta":{"content":"call>call:read_file{path:<|\\"|>samples/suspicious_dropper_demo.js<|\\"|>}<tool_call|> Done."},"finish_reason":"stop"}]}\n',
                    "data: [DONE]\n",
                ]
            ),
            on_delta=emitted.append,
        )

        self.assertEqual("".join(emitted), "Need more.  Done.")
        self.assertIn("<|tool_call>", result.content)
        self.assertEqual(result.tool_calls, [])

    def test_parse_native_stream_emits_progress_and_metrics(self) -> None:
        emitted: list[str] = []
        progress = []

        result = _parse_native_stream(
            FakeStream(
                [
                    'event: progress.prefill\n',
                    'data: {"current":243,"total":935,"percent":25}\n',
                    '\n',
                    'event: delta\n',
                    'data: {"text":"hel"}\n',
                    '\n',
                    'event: progress.generation\n',
                    'data: {"current":1,"total":32,"percent":3}\n',
                    '\n',
                    'event: delta\n',
                    'data: {"text":"lo"}\n',
                    '\n',
                    'event: metrics\n',
                    'data: {"usage":{"prompt_tokens":935,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":12}},"timings":{"prompt_per_second":14.7,"predicted_per_second":3.2}}\n',
                    '\n',
                    'event: done\n',
                    'data: {"finish_reason":"stop"}\n',
                    '\n',
                ]
            ),
            on_delta=emitted.append,
            on_progress=lambda item: progress.append((item.phase, item.current, item.total, item.percent)),
        )

        self.assertEqual(emitted, ["hel", "lo"])
        self.assertEqual(progress, [("prefill", 243, 935, 25), ("generation", 1, 32, 3)])
        self.assertEqual(result.content, "hello")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.prompt_tokens, 935)
        self.assertEqual(result.completion_tokens, 2)
        self.assertEqual(result.cached_tokens, 12)
        self.assertEqual(result.prompt_tokens_per_second, 14.7)
        self.assertEqual(result.generation_tokens_per_second, 3.2)

    def test_parse_native_stream_converts_raw_tool_call_and_suppresses_delta(self) -> None:
        emitted: list[str] = []

        result = _parse_native_stream(
            FakeStream(
                [
                    'event: progress.prefill\n',
                    'data: {"current":243,"total":935,"percent":25}\n',
                    '\n',
                    'event: delta\n',
                    'data: {"text":"<|tool_call>call:exec_shell_full_command{command:<|\\"|>cat note.txt<|\\"|>}<tool_call|>"}\n',
                    '\n',
                    'event: done\n',
                    'data: {"finish_reason":"tool_calls"}\n',
                    '\n',
                ]
            ),
            on_delta=emitted.append,
            on_progress=lambda _item: None,
        )

        self.assertEqual(emitted, [])
        self.assertEqual(result.content, "")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "exec_shell_full_command")
        self.assertEqual(result.tool_calls[0]["function"]["arguments"], '{"command": "cat note.txt"}')


if __name__ == "__main__":
    unittest.main()
