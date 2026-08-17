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
    LlamaServerError,
    _enrich_model_info_with_props,
    _parse_chat_result,
    _parse_chat_stream,
    _parse_model_info,
    _parse_native_stream,
    _final_prefix_experiment_requested,
    _qwen_route_prefix_anchor_requested,
    _qwen36_shell_tool_prefix_anchor_requested,
)
from orbit.backend.base import ChatResult
from orbit.backend.payloads import (
    ARTIFACT_CONTENT_PROTOCOL_ID,
    ARTIFACT_CONTENT_PROTOCOL_VERSION,
    ChatPayloadOptions,
    build_chat_payload,
)
from orbit.backend import model_names
from orbit.runtime.kv_diag import model_call_context
from orbit.runtime.shell_guardrails import exec_shell_full_definition
from urllib.error import HTTPError, URLError


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
    def test_static_analysis_session_reset_requires_confirmed_empty_native_state(self) -> None:
        backend = LlamaServerBackend(base_url="http://localhost", model="fake", timeout=1)
        backend._props_cache = {"backend": "orbit-native"}
        with mock.patch.object(
            backend,
            "_post_json",
            return_value={"status": "reset", "cached_tokens": 0, "in_flight": False},
        ) as post:
            error = backend.reset_static_analysis_session()

        self.assertIsNone(error)
        post.assert_called_once_with("/session/reset", {})

        for response in (
            {"status": "reset", "cached_tokens": 1, "in_flight": False},
            {"status": "reset", "cached_tokens": 0, "in_flight": True},
            {"status": "not-reset", "cached_tokens": 0, "in_flight": False},
        ):
            with self.subTest(response=response), mock.patch.object(
                backend, "_post_json", return_value=response
            ):
                self.assertIn(
                    "was not confirmed", backend.reset_static_analysis_session() or ""
                )

    def test_static_analysis_session_reset_fails_closed_without_native_support(self) -> None:
        backend = LlamaServerBackend(base_url="http://localhost", model="fake", timeout=1)
        backend._props_cache = {"backend": "other"}
        self.assertIn("requires native", backend.reset_static_analysis_session() or "")

        backend._props_cache = {"backend": "orbit-native"}
        with mock.patch.object(
            backend, "_post_json", side_effect=LlamaServerError("offline")
        ):
            self.assertIn("reset failed", backend.reset_static_analysis_session() or "")

    def test_artifact_content_stream_is_native_content_only(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", model="fake", timeout=1)
                self.payload = None

            def _props_or_empty(self) -> dict[str, object]:
                return {
                    "backend": "orbit-native",
                    "artifact_content_protocol": {
                        "id": ARTIFACT_CONTENT_PROTOCOL_ID,
                        "version": ARTIFACT_CONTENT_PROTOCOL_VERSION,
                        "literal_stream": True,
                    },
                }

            def _post_native_stream(
                self,
                path,
                payload,
                *,
                on_delta,
                on_progress,
                literal_content=False,
            ):
                del on_delta, on_progress
                self.assert_path = path
                self.payload = payload
                self.literal_content = literal_content
                return ChatResult("content", "fake", "stop", [], 1, 1, 0, None, None)

        backend = Backend()
        result = backend.artifact_content_stream(
            [{"role": "user", "content": "content only"}],
            temperature=0,
            max_tokens=2048,
            on_delta=lambda _text: None,
        )

        self.assertEqual(result.content, "content")
        self.assertEqual(backend.assert_path, "/chat/stream")
        self.assertTrue(backend.payload["artifact_content"])
        self.assertFalse(backend.payload["thinking"])
        self.assertNotIn("tools", backend.payload)
        self.assertTrue(backend.literal_content)

    def test_artifact_content_stream_rejects_external_backend(self) -> None:
        backend = LlamaServerBackend(base_url="http://localhost", model="fake", timeout=1)
        with mock.patch.object(backend, "_is_orbit_native_backend", return_value=False):
            with self.assertRaisesRegex(LlamaServerError, "native Orbit backend"):
                backend.artifact_content_stream(
                    [{"role": "user", "content": "content only"}],
                    temperature=0,
                    max_tokens=32,
                    on_delta=lambda _text: None,
                )

    def test_artifact_content_stream_rejects_native_backend_without_verified_protocol(self) -> None:
        backend = LlamaServerBackend(base_url="http://localhost", model="fake", timeout=1)
        with mock.patch.object(backend, "_props_or_empty", return_value={"backend": "orbit-native"}):
            with self.assertRaisesRegex(LlamaServerError, "verified artifact content protocol"):
                backend.artifact_content_stream(
                    [{"role": "user", "content": "content only"}],
                    temperature=0,
                    max_tokens=32,
                    on_delta=lambda _text: None,
                )

    def test_native_token_count_uses_non_generating_endpoint(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", timeout=1)
                self.requests: list[tuple[str, dict[str, object]]] = []

            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "orbit-native"}

            def _post_json(self, path: str, payload: dict[str, object]):
                self.requests.append((path, payload))
                return {
                    "tokens": 123,
                    "context_tokens": 8192,
                    "rendered_hash": "a" * 64,
                    "token_hash": "b" * 64,
                }

        backend = Backend()
        chat = backend.count_chat_tokens([{"role": "user", "content": "hello"}], thinking=False)
        text_count = backend.count_text_tokens("hello")

        assert chat is not None and text_count is not None
        self.assertEqual((chat.tokens, chat.context_tokens), (123, 8192))
        self.assertEqual((chat.rendered_hash, chat.token_hash), ("a" * 64, "b" * 64))
        self.assertEqual((text_count.tokens, text_count.context_tokens), (123, 8192))
        self.assertEqual([request[0] for request in backend.requests], ["/tokens/count", "/tokens/count"])
        self.assertEqual(backend.requests[0][1]["mode"], "chat")
        self.assertEqual(backend.requests[1][1], {"mode": "text", "text": "hello"})

    def test_external_backend_does_not_attempt_native_token_count(self) -> None:
        class Backend(LlamaServerBackend):
            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "external"}

            def _post_json(self, path: str, payload: dict[str, object]):
                raise AssertionError("native token endpoint must not be called")

        backend = Backend(base_url="http://localhost", timeout=1)
        self.assertIsNone(backend.count_chat_tokens([{"role": "user", "content": "hello"}]))
        self.assertIsNone(backend.count_text_tokens("hello"))

    def test_result_observer_receives_every_completed_backend_result(self) -> None:
        class Backend(LlamaServerBackend):
            def display_model_name(self) -> str:
                return "display-model"

        backend = Backend(base_url="http://localhost", timeout=1)
        observed: list[ChatResult] = []
        backend.set_result_observer(observed.append)
        source = ChatResult("ok", "source", "stop", [], 100, 5, 20, 10.0, 2.0)

        result = backend._with_display_model(source)

        self.assertEqual(result.model, "display-model")
        self.assertEqual(observed, [result])
        backend.set_result_observer(None)
        backend._with_display_model(source)
        self.assertEqual(observed, [result])

    def test_observer_failure_does_not_change_successful_result(self) -> None:
        class Backend(LlamaServerBackend):
            def display_model_name(self) -> str:
                return "display-model"

        backend = Backend(base_url="http://localhost", timeout=1)
        backend.set_result_observer(lambda _result: (_ for _ in ()).throw(RuntimeError("observer failed")))
        source = ChatResult("ok", "source", "stop", [], 100, 5, 20, 10.0, 2.0)

        self.assertEqual(backend._with_display_model(source).content, "ok")

    def test_failed_backend_call_is_observed_and_original_error_is_preserved(self) -> None:
        backend = LlamaServerBackend(base_url="http://localhost", timeout=1)
        failures: list[str] = []
        backend.set_failure_observer(lambda: failures.append("failed"))

        with self.assertRaisesRegex(LlamaServerError, "boom"):
            backend._observe_call(lambda: (_ for _ in ()).throw(LlamaServerError("boom")))

        self.assertEqual(failures, ["failed"])

    def test_failure_observer_error_does_not_replace_original_error(self) -> None:
        backend = LlamaServerBackend(base_url="http://localhost", timeout=1)
        backend.set_failure_observer(
            lambda: (_ for _ in ()).throw(RuntimeError("observer failed"))
        )

        with self.assertRaisesRegex(LlamaServerError, "original"):
            backend._observe_call(
                lambda: (_ for _ in ()).throw(LlamaServerError("original"))
            )

    def test_cancelled_backend_call_is_observed(self) -> None:
        backend = LlamaServerBackend(base_url="http://localhost", timeout=1)
        failures: list[str] = []
        backend.set_failure_observer(lambda: failures.append("cancelled"))

        with self.assertRaises(KeyboardInterrupt):
            backend._observe_call(lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

        self.assertEqual(failures, ["cancelled"])

    def test_backend_props_overlay_runtime_tool_healing_diagnostics(self) -> None:
        class Backend(LlamaServerBackend):
            def _props_or_empty(self) -> dict[str, object]:
                return {"backend": "orbit-native", "tool_call_healing_repair_count": 0}

        runtime_status = {
            "tool_call_healing_enabled": True,
            "tool_call_healing_source": "default",
            "tool_call_healing_config_error": None,
            "tool_call_healing_repair_count": 3,
            "tool_call_healing_rejection_count": 2,
            "tool_call_healing_last_rules": ["remove_trailing_comma"],
        }
        with mock.patch(
            "orbit.backend.llama_server.tool_call_healing_status",
            return_value=runtime_status,
        ):
            props = Backend(base_url="http://localhost", timeout=1).backend_props()

        self.assertEqual(props["backend"], "orbit-native")
        self.assertEqual(props["tool_call_healing_repair_count"], 3)
        self.assertEqual(props["tool_call_healing_last_rules"], ["remove_trailing_comma"])

    def test_backend_props_stays_empty_when_server_props_are_unavailable(self) -> None:
        class Backend(LlamaServerBackend):
            def _props_or_empty(self) -> dict[str, object]:
                return {}

        self.assertEqual(Backend(base_url="http://localhost", timeout=1).backend_props(), {})

    def test_transient_props_failure_does_not_disable_native_route_anchor(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", timeout=1)
                self.props_calls = 0
                self.payload = None

            def _get_json(self, path: str):
                if path != "/props":
                    raise AssertionError(f"unexpected path: {path}")
                self.props_calls += 1
                if self.props_calls == 1:
                    try:
                        raise URLError("server is still starting")
                    except URLError as exc:
                        raise LlamaServerError("server is still starting") from exc
                return {"backend": "orbit-native"}

            def display_model_name(self):
                return None

            def _post_native_stream(self, path, payload, *, on_delta, on_progress):
                del on_delta, on_progress
                self.assert_path = path
                self.payload = payload
                return ChatResult("ok", "fake", "stop", [], 10, 1, 0, None, None)

        backend = Backend()
        self.assertEqual(backend._props_or_empty(), {})
        self.assertIsNone(backend._props_cache)

        with model_call_context(phase="route", tools_mode="on"):
            backend.chat(
                [{"role": "system", "content": "route"}, {"role": "user", "content": "hi"}],
                temperature=0,
                max_tokens=32,
            )

        self.assertEqual(backend.props_calls, 2)
        self.assertEqual(backend.assert_path, "/chat/stream")
        self.assertTrue(backend.payload["qwen_route_prefix_anchor"])
        self.assertTrue(backend._is_orbit_native_backend())
        self.assertEqual(backend.props_calls, 2)

    def test_repeated_transient_props_failures_retry_once_per_operation(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self) -> None:
                super().__init__(base_url="http://localhost", timeout=1)
                self.props_calls = 0

            def _get_json(self, path: str):
                if path != "/props":
                    raise AssertionError(f"unexpected path: {path}")
                self.props_calls += 1
                try:
                    raise URLError("server is still starting")
                except URLError as exc:
                    raise LlamaServerError("server is still starting") from exc

        backend = Backend()
        for expected_calls in range(1, 4):
            self.assertEqual(backend._props_or_empty(), {})
            self.assertEqual(backend.props_calls, expected_calls)
            self.assertIsNone(backend._props_cache)

    def test_successful_props_response_is_cached_and_fails_closed(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self, response) -> None:
                super().__init__(base_url="http://localhost", timeout=1)
                self.response = response
                self.props_calls = 0

            def _get_json(self, path: str):
                if path != "/props":
                    raise AssertionError(f"unexpected path: {path}")
                self.props_calls += 1
                return self.response

        cases = (
            ({"backend": "orbit-native"}, True, {"backend": "orbit-native"}),
            ({"backend": "external"}, False, {"backend": "external"}),
            (["malformed"], False, {}),
        )
        for response, expected_native, expected_cache in cases:
            with self.subTest(response=response):
                backend = Backend(response)
                self.assertEqual(backend._is_orbit_native_backend(), expected_native)
                self.assertEqual(backend._is_orbit_native_backend(), expected_native)
                self.assertEqual(backend.props_calls, 1)
                self.assertEqual(backend._props_cache, expected_cache)

    def test_permanent_props_failure_is_cached_and_fails_closed(self) -> None:
        class Backend(LlamaServerBackend):
            def __init__(self, cause) -> None:
                super().__init__(base_url="http://localhost", timeout=1)
                self.cause = cause
                self.props_calls = 0

            def _get_json(self, path: str):
                if path != "/props":
                    raise AssertionError(f"unexpected path: {path}")
                self.props_calls += 1
                try:
                    raise self.cause
                except BaseException as exc:
                    raise LlamaServerError("permanent props failure") from exc

        causes = (
            HTTPError("http://localhost/props", 404, "not found", {}, None),
            json.JSONDecodeError("invalid JSON", "x", 0),
        )
        for cause in causes:
            with self.subTest(cause=type(cause).__name__):
                backend = Backend(cause)
                self.assertFalse(backend._is_orbit_native_backend())
                self.assertFalse(backend._is_orbit_native_backend())
                self.assertEqual(backend.props_calls, 1)
                self.assertEqual(backend._props_cache, {})

    def test_native_separated_reasoning_requires_verified_qwen_profile(self) -> None:
        class Backend(LlamaServerBackend):
            props: dict[str, object] = {}

            def _props_or_empty(self) -> dict[str, object]:
                return self.props

        backend = Backend(base_url="http://localhost", timeout=1)
        backend.props = {
            "model_compatibility": {
                "verified": True,
                "reasoning_protocol": "qwen-think",
            }
        }
        self.assertTrue(backend.uses_native_separated_reasoning())

        backend.props = {
            "model_compatibility": {
                "verified": True,
                "reasoning_protocol": "gemma4-control-channel",
            }
        }
        self.assertFalse(backend.uses_native_separated_reasoning())

        backend.props = {
            "model_compatibility": {
                "verified": False,
                "reasoning_protocol": "qwen-think",
            }
        }
        self.assertFalse(backend.uses_native_separated_reasoning())

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

    def test_qwen_route_prefix_request_is_bounded_to_native_route_tools_on(self) -> None:
        with mock.patch.dict(os.environ, {"ORBIT_QWEN_ROUTE_PREFIX_REUSE": "1"}, clear=True):
            with model_call_context(phase="route", tools_mode="on"):
                self.assertTrue(_qwen_route_prefix_anchor_requested(native_backend=True))
            with model_call_context(phase="route", tools_mode="off"):
                self.assertFalse(_qwen_route_prefix_anchor_requested(native_backend=True))
            with model_call_context(phase="chat_final", tools_mode="on"):
                self.assertFalse(_qwen_route_prefix_anchor_requested(native_backend=True))
            with model_call_context(phase="route", tools_mode="on"):
                self.assertFalse(_qwen_route_prefix_anchor_requested(native_backend=False))

    def test_qwen_route_prefix_kill_switch_and_invalid_value_disable_request(self) -> None:
        for value in ("0", "invalid"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ,
                {
                    "ORBIT_QWEN_ROUTE_PREFIX_REUSE": value,
                    "ORBIT_QWEN3_CODER_ROUTE_PREFIX_REUSE": value,
                },
                clear=True,
            ), model_call_context(phase="route", tools_mode="on"):
                self.assertFalse(_qwen_route_prefix_anchor_requested(native_backend=True))

    def test_qwen36_shell_prefix_request_has_exact_phase_mode_and_schema(self) -> None:
        tools = [exec_shell_full_definition()]
        with mock.patch.dict(os.environ, {"ORBIT_QWEN36_SHELL_TOOL_PREFIX_REUSE": "1"}, clear=True):
            with model_call_context(phase="tool_call", tools_mode="on"):
                self.assertTrue(
                    _qwen36_shell_tool_prefix_anchor_requested(
                        native_backend=True,
                        tools=tools,
                        thinking=False,
                    )
                )
                self.assertFalse(
                    _qwen36_shell_tool_prefix_anchor_requested(
                        native_backend=True,
                        tools=tools,
                        thinking=True,
                    )
                )
                changed = [exec_shell_full_definition()]
                changed[0]["function"]["parameters"]["properties"]["timeout"]["maximum"] = 16
                self.assertFalse(
                    _qwen36_shell_tool_prefix_anchor_requested(
                        native_backend=True,
                        tools=changed,
                        thinking=False,
                    )
                )
            for phase in (
                "route",
                "tool_call_retry",
                "artifact_content",
                "verify_artifact",
                "post_tool_route",
                "final_from_tool",
            ):
                with self.subTest(phase=phase), model_call_context(
                    phase=phase,
                    tools_mode="on",
                ):
                    self.assertFalse(
                        _qwen36_shell_tool_prefix_anchor_requested(
                            native_backend=True,
                            tools=tools,
                            thinking=False,
                        )
                    )
            with model_call_context(phase="tool_call", tools_mode="off"):
                self.assertFalse(
                    _qwen36_shell_tool_prefix_anchor_requested(
                        native_backend=True,
                        tools=tools,
                        thinking=False,
                    )
                )

    def test_qwen36_shell_prefix_kill_switch_disables_request(self) -> None:
        tools = [exec_shell_full_definition()]
        for value in ("0", "invalid"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ,
                {"ORBIT_QWEN36_SHELL_TOOL_PREFIX_REUSE": value},
                clear=True,
            ), model_call_context(phase="tool_call", tools_mode="on"):
                self.assertFalse(
                    _qwen36_shell_tool_prefix_anchor_requested(
                        native_backend=True,
                        tools=tools,
                        thinking=False,
                    )
                )

    def test_final_prefix_experiment_payload_is_limited_to_native_final_tool_phase(self) -> None:
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
        with mock.patch.dict(os.environ, {"ORBIT_FINAL_PREFIX_EXPERIMENT": "1"}, clear=False):
            with model_call_context(phase="route", tools_mode="on"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)
            with model_call_context(phase="final_from_tool", tools_mode="off"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)
            with model_call_context(phase="final_from_tool", tools_mode="on"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)
        with mock.patch.dict(
            os.environ,
            {"ORBIT_FINAL_PREFIX_REUSE": "0", "ORBIT_FINAL_PREFIX_EXPERIMENT": "1"},
            clear=True,
        ):
            with model_call_context(phase="final_from_tool", tools_mode="on"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)
        with mock.patch.dict(os.environ, {"ORBIT_FINAL_PREFIX_REUSE": "1"}, clear=True):
            with model_call_context(phase="final_from_tool", tools_mode="on"):
                backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=16)

        self.assertNotIn("final_prefix_experiment", backend.payloads[0])
        self.assertNotIn("final_prefix_experiment", backend.payloads[1])
        self.assertTrue(backend.payloads[2]["final_prefix_experiment"])
        self.assertNotIn("final_prefix_experiment", backend.payloads[3])
        self.assertTrue(backend.payloads[4]["final_prefix_experiment"])

    def test_final_prefix_stable_configuration_controls_native_payload_request(self) -> None:
        cases = (
            ({}, True),
            ({"ORBIT_FINAL_PREFIX_EXPERIMENT": "1"}, True),
            ({"ORBIT_FINAL_PREFIX_REUSE": "1"}, True),
            ({"ORBIT_FINAL_PREFIX_REUSE": "0", "ORBIT_FINAL_PREFIX_EXPERIMENT": "1"}, False),
            ({"ORBIT_FINAL_PREFIX_REUSE": "1", "ORBIT_FINAL_PREFIX_EXPERIMENT": "0"}, True),
            ({"ORBIT_FINAL_PREFIX_REUSE": "invalid", "ORBIT_FINAL_PREFIX_EXPERIMENT": "1"}, False),
        )
        for env, expected in cases:
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True), model_call_context(
                phase="final_from_tool", tools_mode="on"
            ):
                self.assertIs(_final_prefix_experiment_requested(native_backend=True), expected)

    def test_final_prefix_request_remains_ineligible_without_tools_or_native_backend(self) -> None:
        with mock.patch.dict(os.environ, {"ORBIT_FINAL_PREFIX_REUSE": "1"}, clear=True):
            with model_call_context(phase="final_from_tool", tools_mode="off"):
                self.assertFalse(_final_prefix_experiment_requested(native_backend=True))
            with model_call_context(phase="final_from_tool", tools_mode="on"):
                self.assertFalse(_final_prefix_experiment_requested(native_backend=False))

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

    @mock.patch("orbit.backend.llama_server.time.monotonic_ns", return_value=1_250_000_000)
    def test_parse_chat_stream_records_first_output_event_once(self, monotonic_ns) -> None:
        result = _parse_chat_stream(
            FakeStream(
                [
                    'data: {"choices":[{"delta":{"content":"first"},"finish_reason":null}]}\n',
                    'data: {"choices":[{"delta":{"content":"second"},"finish_reason":"stop"}],"timings":{"backend_ttft_ms":200.0}}\n',
                    "data: [DONE]\n",
                ]
            ),
            on_delta=lambda _text: None,
            request_started_ns=1_000_000_000,
        )

        self.assertEqual(result.backend_ttft_ms, 200.0)
        self.assertEqual(result.stream_ttft_ms, 250.0)
        monotonic_ns.assert_called_once_with()

    @mock.patch("orbit.backend.llama_server.time.monotonic_ns")
    def test_parse_chat_stream_keeps_ttft_null_without_output(self, monotonic_ns) -> None:
        result = _parse_chat_stream(
            FakeStream(
                [
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"backend_ttft_ms":null}}\n',
                    "data: [DONE]\n",
                ]
            ),
            on_delta=lambda _text: None,
            request_started_ns=1_000_000_000,
        )

        self.assertIsNone(result.backend_ttft_ms)
        self.assertIsNone(result.stream_ttft_ms)
        monotonic_ns.assert_not_called()

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
                    'data: {"current":1011,"total":1703,"percent":59,"evaluated_current":243,"evaluated_total":935,"cached_tokens":768,"elapsed_seconds":7.75,"tokens_per_second":31.4}\n',
                    '\n',
                    'event: delta\n',
                    'data: {"text":"hel"}\n',
                    '\n',
                    'event: progress.generation\n',
                    'data: {"current":1,"total":32,"percent":3,"elapsed_seconds":0.2,"tokens_per_second":5.0}\n',
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
            on_progress=progress.append,
        )

        self.assertEqual(emitted, ["hel", "lo"])
        self.assertEqual((progress[0].evaluated_current, progress[0].evaluated_total), (243, 935))
        self.assertEqual(progress[0].cached_tokens, 768)
        self.assertEqual(progress[0].tokens_per_second, 31.4)
        self.assertEqual(progress[1].current, 1)
        self.assertEqual(progress[1].elapsed_seconds, 0.2)
        self.assertEqual(progress[1].tokens_per_second, 5.0)
        self.assertEqual(result.content, "hello")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.prompt_tokens, 935)
        self.assertEqual(result.completion_tokens, 2)
        self.assertEqual(result.cached_tokens, 12)
        self.assertEqual(result.prompt_tokens_per_second, 14.7)
        self.assertEqual(result.generation_tokens_per_second, 3.2)

    @mock.patch("orbit.backend.llama_server.time.monotonic_ns", return_value=1_400_000_000)
    def test_parse_native_stream_keeps_backend_and_stream_ttft_distinct(self, monotonic_ns) -> None:
        result = _parse_native_stream(
            FakeStream(
                [
                    'event: progress.prefill\n',
                    'data: {"current":1,"total":2,"percent":50}\n',
                    '\n',
                    'event: delta\n',
                    'data: {"text":"first"}\n',
                    '\n',
                    'event: delta\n',
                    'data: {"text":"second"}\n',
                    '\n',
                    'event: metrics\n',
                    'data: {"usage":{},"timings":{"backend_ttft_ms":350.0}}\n',
                    '\n',
                    'event: done\n',
                    'data: {"finish_reason":"stop"}\n',
                    '\n',
                ]
            ),
            on_delta=lambda _text: None,
            on_progress=lambda _progress: None,
            request_started_ns=1_000_000_000,
        )

        self.assertEqual(result.backend_ttft_ms, 350.0)
        self.assertEqual(result.stream_ttft_ms, 400.0)
        monotonic_ns.assert_called_once_with()

    def test_parse_chat_result_rejects_invalid_ttft_metrics(self) -> None:
        result = _parse_chat_result(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                "timings": {"backend_ttft_ms": float("nan")},
            }
        )

        self.assertIsNone(result.backend_ttft_ms)

    def test_parse_native_stream_drops_nonfinite_or_invalid_live_metrics(self) -> None:
        progress = []

        _parse_native_stream(
            FakeStream(
                [
                    'event: progress.prefill\n',
                    'data: {"current":1,"total":2,"percent":50,"evaluated_current":true,"evaluated_total":2,"cached_tokens":-1,"elapsed_seconds":NaN,"tokens_per_second":true}\n',
                    '\n',
                    'event: done\n',
                    'data: {"finish_reason":"stop"}\n',
                    '\n',
                ]
            ),
            on_delta=lambda _text: None,
            on_progress=progress.append,
        )

        self.assertIsNone(progress[0].evaluated_current)
        self.assertEqual(progress[0].evaluated_total, 2)
        self.assertIsNone(progress[0].cached_tokens)
        self.assertIsNone(progress[0].elapsed_seconds)
        self.assertIsNone(progress[0].tokens_per_second)

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

    def test_parse_native_stream_literal_mode_preserves_raw_tool_markers(self) -> None:
        emitted: list[str] = []
        literal = '<|tool_call>call:exec_shell_full_command{command:<|"|>inert<|"|>}<tool_call|>'

        result = _parse_native_stream(
            FakeStream(
                [
                    'event: delta\n',
                    f'data: {json.dumps({"text": literal})}\n',
                    '\n',
                    'event: done\n',
                    'data: {"finish_reason":"stop"}\n',
                    '\n',
                ]
            ),
            on_delta=emitted.append,
            on_progress=None,
            literal_content=True,
        )

        self.assertEqual(emitted, [literal])
        self.assertEqual(result.content, literal)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.finish_reason, "stop")

    def test_parse_native_stream_keeps_qwen_reasoning_and_tool_calls_out_of_content(self) -> None:
        emitted: list[str] = []

        result = _parse_native_stream(
            FakeStream(
                [
                    'event: reasoning\n',
                    'data: {"text":"private planning"}\n',
                    '\n',
                    'event: tool_calls\n',
                    'data: {"tool_calls":[{"id":"","type":"function","function":{"name":"system_info","arguments":"{}"}}]}\n',
                    '\n',
                    'event: metrics\n',
                    'data: {"usage":{"prompt_tokens":20,"completion_tokens":8,"prompt_tokens_details":{"cached_tokens":0}},"timings":{}}\n',
                    '\n',
                    'event: done\n',
                    'data: {"finish_reason":"tool_calls"}\n',
                    '\n',
                ]
            ),
            on_delta=emitted.append,
            on_progress=None,
        )

        self.assertEqual(emitted, [])
        self.assertEqual(result.content, "")
        self.assertEqual(result.reasoning_content, "private planning")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.tool_calls[0]["function"], {"name": "system_info", "arguments": "{}"})


if __name__ == "__main__":
    unittest.main()
