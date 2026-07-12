from __future__ import annotations

import unittest

from orbit.native_server.protocol import (
    DEFAULT_SESSION_ID,
    native_chat_response,
    openai_chat_response,
    parse_chat_request,
    parse_continue_request,
    trim_at_stop,
    validate_session_id,
)


class NativeServerProtocolTests(unittest.TestCase):
    def test_parse_chat_request_defaults_to_default_session(self) -> None:
        request = parse_chat_request(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 12,
                "temperature": 0.1,
            }
        )

        self.assertEqual(request.messages, [{"role": "user", "content": "hello"}])
        self.assertEqual(request.max_tokens, 12)
        self.assertEqual(request.temperature, 0.1)
        self.assertEqual(request.session_id, DEFAULT_SESSION_ID)
        self.assertIsNone(request.thinking)
        self.assertEqual(request.stop, ())
        self.assertEqual(request.tools, [])
        self.assertFalse(request.route_prefix_anchor)
        self.assertIsNone(request.allow_mtp_experimental)
        self.assertFalse(request.final_prefix_experiment)

    def test_parse_chat_request_accepts_thinking_flag(self) -> None:
        request = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "thinking": True})

        self.assertTrue(request.thinking)
        self.assertEqual(parse_chat_request({"messages": [{"role": "user", "content": "x"}], "thinking": False}).thinking, False)

    def test_parse_chat_request_accepts_stop_string_or_list(self) -> None:
        self.assertEqual(parse_chat_request({"messages": [{"role": "user", "content": "x"}], "stop": "END"}).stop, ("END",))
        self.assertEqual(
            parse_chat_request({"messages": [{"role": "user", "content": "x"}], "stop": ["A", "", 1, "B"]}).stop,
            ("A", "B"),
        )

    def test_parse_continue_request_accepts_defaults_and_options(self) -> None:
        request = parse_continue_request({"max_tokens": 32, "thinking": True, "stop": ["A", "", "B"], "stream": True})

        self.assertEqual(request.max_tokens, 32)
        self.assertTrue(request.thinking)
        self.assertEqual(request.stop, ("A", "B"))
        self.assertTrue(request.stream)

        default_request = parse_continue_request({})
        self.assertEqual(default_request.max_tokens, 256)
        self.assertIsNone(default_request.thinking)
        self.assertEqual(default_request.stop, ())
        self.assertFalse(default_request.stream)

    def test_parse_chat_request_keeps_tool_definitions(self) -> None:
        tool = {"type": "function", "function": {"name": "exec_shell_full_command"}}
        request = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "tools": [tool, "bad"]})

        self.assertEqual(request.tools, [tool])

    def test_parse_chat_request_accepts_route_prefix_anchor_flag(self) -> None:
        request = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "route_prefix_anchor": True})
        self.assertTrue(request.route_prefix_anchor)

    def test_parse_chat_request_accepts_allow_mtp_flag(self) -> None:
        disabled = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "allow_mtp_experimental": False})
        enabled = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "allow_mtp_experimental": True})
        ignored = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "allow_mtp_experimental": "false"})

        self.assertFalse(disabled.allow_mtp_experimental)
        self.assertTrue(enabled.allow_mtp_experimental)
        self.assertIsNone(ignored.allow_mtp_experimental)

    def test_parse_chat_request_accepts_final_prefix_experiment_flag(self) -> None:
        enabled = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "final_prefix_experiment": True})
        ignored = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "final_prefix_experiment": "true"})

        self.assertTrue(enabled.final_prefix_experiment)
        self.assertFalse(ignored.final_prefix_experiment)

    def test_parse_chat_request_rejects_malformed_payloads(self) -> None:
        bad_payloads = [
            {},
            {"messages": "hello"},
            {"messages": []},
            {"messages": ["bad"]},
            {"messages": [{"content": "missing role"}]},
            {"messages": [{"role": "user", "content": "hello"}], "max_tokens": "bad"},
        ]

        for payload in bad_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    parse_chat_request(payload)

    def test_parse_chat_request_accepts_legacy_prompt_string(self) -> None:
        request = parse_chat_request({"prompt": "hello"})

        self.assertEqual(request.messages, [{"role": "user", "content": "hello"}])

    def test_validate_session_id_is_explicitly_single_session_for_now(self) -> None:
        validate_session_id(DEFAULT_SESSION_ID)
        with self.assertRaises(ValueError):
            validate_session_id("other")

    def test_trim_at_stop_removes_stop_sequence(self) -> None:
        content, stopped = trim_at_stop("hello STOP ignored", ["STOP"])
        self.assertEqual(content, "hello ")
        self.assertTrue(stopped)

    def test_native_response_includes_reuse_and_evaluated_tokens(self) -> None:
        response = native_chat_response(
            content="ok",
            model="m",
            finish_reason="stop",
            session_id=DEFAULT_SESSION_ID,
            prompt_tokens=10,
            completion_tokens=2,
            reused_prompt_tokens=7,
            evaluated_prompt_tokens=3,
            prefill_ms=300.0,
            generation_ms=100.0,
            cancelled=False,
        )

        details = response["usage"]["prompt_tokens_details"]
        self.assertEqual(details["cached_tokens"], 7)
        self.assertEqual(details["reused_tokens"], 7)
        self.assertEqual(details["evaluated_tokens"], 3)
        self.assertEqual(response["timings"]["prompt_per_second"], 10.0)

    def test_openai_response_keeps_expected_shape(self) -> None:
        native = native_chat_response(
            content="hello",
            model="m",
            finish_reason="stop",
            session_id=DEFAULT_SESSION_ID,
            prompt_tokens=1,
            completion_tokens=1,
            reused_prompt_tokens=0,
            evaluated_prompt_tokens=1,
            prefill_ms=1.0,
            generation_ms=1.0,
            cancelled=False,
        )

        response = openai_chat_response(native)
        self.assertEqual(response["choices"][0]["message"]["content"], "hello")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"]["prompt_tokens"], 1)

    def test_openai_response_preserves_empty_response_finish_reason(self) -> None:
        native = native_chat_response(
            content="",
            model="m",
            finish_reason="empty_response",
            session_id=DEFAULT_SESSION_ID,
            prompt_tokens=10,
            completion_tokens=0,
            reused_prompt_tokens=5,
            evaluated_prompt_tokens=5,
            prefill_ms=25.0,
            generation_ms=0.0,
            cancelled=False,
        )

        response = openai_chat_response(native)
        self.assertEqual(response["choices"][0]["message"]["content"], "")
        self.assertEqual(response["choices"][0]["finish_reason"], "empty_response")
        self.assertEqual(response["usage"]["completion_tokens"], 0)

    def test_openai_response_preserves_cancelled_finish_reason(self) -> None:
        native = native_chat_response(
            content="",
            model="m",
            finish_reason="cancelled",
            session_id=DEFAULT_SESSION_ID,
            prompt_tokens=10,
            completion_tokens=0,
            reused_prompt_tokens=5,
            evaluated_prompt_tokens=5,
            prefill_ms=25.0,
            generation_ms=0.0,
            cancelled=True,
        )

        response = openai_chat_response(native)
        self.assertEqual(response["choices"][0]["message"]["content"], "")
        self.assertEqual(response["choices"][0]["finish_reason"], "cancelled")
        self.assertEqual(response["timings"]["predicted_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
