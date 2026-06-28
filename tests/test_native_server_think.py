from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from orbit.native_server.app import OrbitNativeHandler, OrbitNativeServer, _DisconnectWatcher
from orbit.native_server.protocol import openai_chat_response, parse_chat_request


@dataclass
class _FakeTimings:
    prompt_tokens: int = 1
    output_tokens: int = 1
    reused_prompt_tokens: int = 0
    evaluated_prompt_tokens: int = 1
    prefill_ms: float = 0.0
    generation_ms: float = 0.0
    cancelled: bool = False


@dataclass
class _FakeCompletion:
    content: str = "ok"
    timings: _FakeTimings = field(default_factory=_FakeTimings)
    stopped_by_stop: bool = False
    completed_after_thought: bool = False


class _FakeClient:
    def __init__(self, thinking: bool) -> None:
        self.config = type("Config", (), {"thinking": thinking})()
        self.calls: list[bool] = []
        self.paths = type("Paths", (), {})()
        self.mtp_probe = type("Probe", (), {"enabled": False, "initialized": False, "error": None})()
        self.mtp_dry_run = type("Probe", (), {"enabled": False, "success": False, "draft_tokens": 0, "error": None})()
        self.mtp_accept_probe = type("Probe", (), {"enabled": False, "success": False, "draft_tokens": 0, "accepted_tokens": 0, "error": None})()
        self.mtp_decode_probe = type("Probe", (), {"enabled": False, "success": False, "error": None})()
        self.last_mtp_completion = type("Probe", (), {"success": False})()
        self.mtp_fallback_reason = None

    def complete_chat_text(
        self,
        messages,
        *,
        max_tokens,
        stop,
        tools,
        thinking,
        route_prefix_anchor=False,
        on_progress=None,
        on_token=None,
        should_cancel=None,
    ):
        self.calls.append(thinking)
        return _FakeCompletion()

    def session_snapshot(self, session_id: str):
        return type(
            "Snapshot",
            (),
            {
                "session_id": session_id,
                "backend_mode": "no-mtp",
                "cached_tokens": 0,
                "in_flight": False,
                "cancel_requested": False,
                "mtp_enabled": False,
                "mtp_initialized": False,
                "mtp_failure_reason": None,
            },
        )()


class NativeServerThinkTests(unittest.TestCase):
    def test_disconnect_watcher_disarm_skips_cancel_callback(self) -> None:
        called: list[str] = []
        watcher = _DisconnectWatcher(None, lambda: called.append("cancel"))  # type: ignore[arg-type]

        watcher.disarm()
        watcher._mark_disconnected()

        self.assertEqual(called, [])
        self.assertTrue(watcher.is_set())

    def test_json_ignores_client_disconnect_during_write(self) -> None:
        class _BrokenWriter:
            def write(self, _body: bytes) -> None:
                raise BrokenPipeError("client disconnected")

        class _DummyHandler:
            def __init__(self) -> None:
                self.wfile = _BrokenWriter()
                self.statuses: list[int] = []
                self.headers: list[tuple[str, str]] = []
                self.ended = False

            def send_response(self, status: int) -> None:
                self.statuses.append(status)

            def send_header(self, key: str, value: str) -> None:
                self.headers.append((key, value))

            def end_headers(self) -> None:
                self.ended = True

        handler = _DummyHandler()

        OrbitNativeHandler._json(handler, {"status": "ok"})

        self.assertEqual(handler.statuses, [200])
        self.assertTrue(handler.ended)

    def test_request_thinking_overrides_client_default(self) -> None:
        server = OrbitNativeServer(client=_FakeClient(thinking=False), model_alias="m")

        result = server.complete(parse_chat_request({"messages": [{"role": "user", "content": "hello"}], "thinking": True}))

        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(server.client.calls, [True])

    def test_client_default_thinking_applies_when_request_is_unspecified(self) -> None:
        server = OrbitNativeServer(client=_FakeClient(thinking=True), model_alias="m")

        result = server.complete(parse_chat_request({"messages": [{"role": "user", "content": "hello"}]}))

        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(server.client.calls, [True])

    def test_server_returns_postprocessed_completion_content_not_raw_chunks(self) -> None:
        client = _FakeClient(thinking=False)

        def fake_complete(_messages, **kwargs):
            kwargs["on_token"]("### Reasoning\nplan\n\n**Final Answer:**\nfinal")
            return _FakeCompletion(content="final")

        client.complete_chat_text = fake_complete
        server = OrbitNativeServer(client=client, model_alias="m")

        result = server.complete(parse_chat_request({"messages": [{"role": "user", "content": "hello"}], "thinking": False}))

        self.assertEqual(result["content"], "final")

    def test_server_uses_stop_when_completion_closed_thought_after_continuation(self) -> None:
        client = _FakeClient(thinking=True)

        def fake_complete(_messages, **kwargs):
            kwargs["on_token"]("<|channel>thought\nx<channel|>4")
            return _FakeCompletion(
                content="<|channel>thought\nx<channel|>4",
                timings=_FakeTimings(output_tokens=94),
                completed_after_thought=True,
            )

        client.complete_chat_text = fake_complete
        server = OrbitNativeServer(client=client, model_alias="m")

        result = server.complete(parse_chat_request({"messages": [{"role": "user", "content": "hello"}], "thinking": True, "max_tokens": 48}))

        self.assertEqual(result["finish_reason"], "stop")

    def test_server_marks_empty_native_completion_as_empty_response(self) -> None:
        client = _FakeClient(thinking=True)

        def fake_complete(_messages, **kwargs):
            return _FakeCompletion(
                content="",
                timings=_FakeTimings(output_tokens=0, cancelled=False),
            )

        client.complete_chat_text = fake_complete
        server = OrbitNativeServer(client=client, model_alias="m")

        result = server.complete(parse_chat_request({"messages": [{"role": "user", "content": "hello"}], "thinking": True}))

        self.assertEqual(result["finish_reason"], "empty_response")

    def test_server_marks_open_thought_without_final_as_length(self) -> None:
        client = _FakeClient(thinking=True)

        def fake_complete(_messages, **kwargs):
            kwargs["on_token"]("<|channel>thought\npartial plan")
            return _FakeCompletion(
                content="<|channel>thought\npartial plan",
                timings=_FakeTimings(output_tokens=32, cancelled=False),
                completed_after_thought=False,
            )

        client.complete_chat_text = fake_complete
        server = OrbitNativeServer(client=client, model_alias="m")

        result = server.complete(parse_chat_request({"messages": [{"role": "user", "content": "hello"}], "thinking": True, "max_tokens": 512}))

        self.assertEqual(result["finish_reason"], "length")

    def test_server_keeps_cancelled_finish_reason_even_when_content_is_empty(self) -> None:
        client = _FakeClient(thinking=True)

        def fake_complete(_messages, **kwargs):
            return _FakeCompletion(
                content="",
                timings=_FakeTimings(output_tokens=0, cancelled=True),
            )

        client.complete_chat_text = fake_complete
        server = OrbitNativeServer(client=client, model_alias="m")

        result = server.complete(parse_chat_request({"messages": [{"role": "user", "content": "hello"}], "thinking": True}))

        self.assertEqual(result["finish_reason"], "cancelled")

    def test_error_result_handles_continue_payload_without_messages(self) -> None:
        server = OrbitNativeServer(client=_FakeClient(thinking=False), model_alias="m")

        result = server.error_result("no active continuation state", {"max_tokens": 32})

        self.assertEqual(result["finish_reason"], "error")
        self.assertEqual(result["session_id"], "default")
        self.assertIn("no active continuation state", result["content"])

    def test_openai_shape_stays_correct_when_thinking_mode_changes_between_serial_calls(self) -> None:
        client = _FakeClient(thinking=False)
        seen_calls: list[bool] = []

        def fake_complete(_messages, **kwargs):
            thinking = kwargs["thinking"]
            seen_calls.append(thinking)
            if thinking:
                return _FakeCompletion(
                    content='<|channel>thought\nplan<channel|>I was developed by Google DeepMind.',
                    timings=_FakeTimings(output_tokens=24),
                )
            return _FakeCompletion(
                content="I was developed by Google DeepMind.",
                timings=_FakeTimings(output_tokens=8),
            )

        client.complete_chat_text = fake_complete
        server = OrbitNativeServer(client=client, model_alias="m")

        first = openai_chat_response(
            server.chat(
                {
                    "messages": [{"role": "user", "content": "who designed you?"}],
                    "thinking": False,
                    "max_tokens": 64,
                }
            )
        )
        second = openai_chat_response(
            server.chat(
                {
                    "messages": [{"role": "user", "content": "who designed you?"}],
                    "thinking": True,
                    "max_tokens": 64,
                }
            )
        )

        self.assertEqual(seen_calls, [False, True])
        self.assertEqual(first["choices"][0]["finish_reason"], "stop")
        self.assertEqual(first["choices"][0]["message"]["content"], "I was developed by Google DeepMind.")
        self.assertEqual(second["choices"][0]["finish_reason"], "stop")
        self.assertIn("<|channel>thought", second["choices"][0]["message"]["content"])
        self.assertIn("I was developed by Google DeepMind.", second["choices"][0]["message"]["content"])


if __name__ == "__main__":
    unittest.main()
