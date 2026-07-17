from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from orbit.backend.base import ChatResult, Message
from orbit.post_tool_final_reuse_config import resolve_post_tool_final_reuse
from orbit.runtime import ChatRuntime
from orbit.runtime.post_tool_final_reuse import evaluate_post_tool_final_reuse


def _result(
    content: str,
    *,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, object]] | None = None,
) -> ChatResult:
    return ChatResult(
        content=content,
        model="fake",
        finish_reason=finish_reason,
        tool_calls=tool_calls or [],
        prompt_tokens=20,
        completion_tokens=4,
        cached_tokens=4,
        prompt_tokens_per_second=100.0,
        generation_tokens_per_second=10.0,
    )


def _tool_call(call_id: str, command: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "exec_shell_full_command",
            "arguments": f'{{"command":"{command}"}}',
        },
    }


def _named_tool_call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, separators=(",", ":"))},
    }


class SequenceBackend:
    def __init__(self, results: list[ChatResult | BaseException]) -> None:
        self.results = list(results)
        self.calls = 0

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.calls += 1
        if not self.results:
            raise AssertionError("unexpected model call")
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools=None,
        on_delta=None,
        on_progress=None,
    ) -> ChatResult:
        result = self.chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
        if on_delta is not None and result.content:
            on_delta(result.content)
        return result


class PostToolFinalReuseConfigTests(unittest.TestCase):
    def test_resolver_is_on_by_default_and_invalid_values_disable(self) -> None:
        default = resolve_post_tool_final_reuse({})
        self.assertEqual((default.enabled, default.source), (True, "default"))
        self.assertEqual(resolve_post_tool_final_reuse({"ORBIT_POST_TOOL_FINAL_REUSE": "1"}).enabled, True)
        disabled = resolve_post_tool_final_reuse({"ORBIT_POST_TOOL_FINAL_REUSE": "0"})
        self.assertEqual((disabled.enabled, disabled.source), (False, "stable"))
        invalid = resolve_post_tool_final_reuse({"ORBIT_POST_TOOL_FINAL_REUSE": "yes"})
        self.assertEqual((invalid.enabled, invalid.validation_error), (False, "invalid_boolean"))


class PostToolFinalReuseEligibilityTests(unittest.TestCase):
    def _decision(self, content: str = "The operation completed successfully.", **overrides):
        values = {
            "content": content,
            "finish_reason": "stop",
            "tool_calls": [],
            "phase": "post_tool_route",
            "messages": [{"role": "tool", "content": "ok", "name": "exec_shell_full_command"}],
            "tool_rounds": 1,
            "output_was_suppressed": True,
            "pending_internal_request": False,
            "executed_internal_tool_prompt": False,
            "shell_error_pending": False,
            "shadow_attempt_detected": False,
        }
        values.update(overrides)
        return evaluate_post_tool_final_reuse(**values)

    def test_accepts_only_complete_suppressed_first_pass_prose(self) -> None:
        self.assertTrue(self._decision().eligible)
        cases = {
            "retry": {"phase": "tool_call_retry"},
            "length": {"finish_reason": "length"},
            "tool": {"tool_calls": [_tool_call("call-2", "pwd")]},
            "empty": {"content": ""},
            "streamed": {"output_was_suppressed": False},
            "pending": {"pending_internal_request": True},
            "guard": {"executed_internal_tool_prompt": True},
            "error": {"shell_error_pending": True},
            "attempt": {"shadow_attempt_detected": True},
            "no_result": {"messages": [{"role": "user", "content": "run it"}]},
        }
        for name, overrides in cases.items():
            with self.subTest(name=name):
                self.assertFalse(self._decision(**overrides).eligible)

    def test_rejects_technical_or_incomplete_content(self) -> None:
        contents = (
            '<|tool_call>{"name":"system_info","arguments":{}}<tool_call|>',
            '<tool_call>{"name":"system_info"}</tool_call>',
            '{"name":"system_info","arguments":{}}',
            '```json\n{"name":"system_info","arguments":{}}\n```',
            'For example: {"name":"system_info","arguments":{}}',
            "Result:",
            "<|channel>thought<channel|>",
        )
        for content in contents:
            with self.subTest(content=content):
                self.assertFalse(self._decision(content).eligible)


class PostToolFinalReuseRuntimeTests(unittest.TestCase):
    def _run(
        self,
        env_value: str | None,
        results: list[ChatResult],
        *,
        on_final_delta=None,
        prompt: str = "list files",
        tool_names: tuple[str, ...] = ("exec_shell_full_command",),
    ):
        backend = SequenceBackend(results)
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {}, clear=False) as environ:
            if env_value is None:
                environ.pop("ORBIT_POST_TOOL_FINAL_REUSE", None)
            else:
                environ["ORBIT_POST_TOOL_FINAL_REUSE"] = env_value
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_with_tools(
                prompt,
                temperature=0,
                max_tokens=64,
                workdir=Path(tmp),
                tool_names=tool_names,
                on_final_delta=on_final_delta,
            )
        return backend, runtime, result

    def test_enabled_reuses_exact_model_prose_and_avoids_one_call(self) -> None:
        prose = "The directory listing completed successfully."
        emitted: list[str] = []
        backend, runtime, result = self._run(
            "1",
            [
                _result("", finish_reason="tool_calls", tool_calls=[_tool_call("call-1", "ls -F")]),
                _result(prose),
            ],
            on_final_delta=emitted.append,
        )
        self.assertEqual((backend.calls, result.content), (2, prose))
        self.assertEqual(runtime.messages[-1], {"role": "assistant", "content": prose})
        self.assertEqual(runtime.post_tool_final_reuse_reused_count, 1)
        self.assertEqual(runtime.post_tool_final_reuse_avoided_model_calls, 1)
        self.assertEqual(emitted, [prose])

    def test_disabled_and_invalid_configuration_keep_existing_three_call_path(self) -> None:
        for value in ("0", "invalid"):
            with self.subTest(value=value):
                backend, runtime, result = self._run(
                    value,
                    [
                        _result("", finish_reason="tool_calls", tool_calls=[_tool_call("call-1", "ls -F")]),
                        _result("Post-tool prose."),
                        _result("Final answer."),
                    ],
                )
                self.assertEqual((backend.calls, result.content), (3, "Final answer."))
                self.assertEqual(runtime.post_tool_final_reuse_reused_count, 0)

    def test_incomplete_prose_falls_back_to_existing_final(self) -> None:
        backend, runtime, result = self._run(
            "1",
            [
                _result("", finish_reason="tool_calls", tool_calls=[_tool_call("call-1", "ls -F")]),
                _result("Result:"),
                _result("The final result is complete."),
            ],
        )
        self.assertEqual((backend.calls, result.content), (3, "The final result is complete."))
        self.assertEqual(runtime.post_tool_final_reuse_fallback_count, 1)
        self.assertTrue(runtime.post_tool_final_reuse_last_reason.startswith("incomplete_"))

    def test_post_tool_length_records_fallback_and_uses_existing_final(self) -> None:
        backend, runtime, result = self._run(
            "1",
            [
                _result("", finish_reason="tool_calls", tool_calls=[_tool_call("call-1", "ls -F")]),
                _result("An unfinished response", finish_reason="length"),
                _result("The final response is complete."),
            ],
        )
        self.assertEqual((backend.calls, result.content), (3, "The final response is complete."))
        self.assertEqual(runtime.post_tool_final_reuse_fallback_count, 1)
        self.assertEqual(runtime.post_tool_final_reuse_last_reason, "finish_reason")

    def test_cancelled_post_tool_result_is_not_reused(self) -> None:
        backend, runtime, result = self._run(
            "1",
            [
                _result("", finish_reason="tool_calls", tool_calls=[_tool_call("call-1", "ls -F")]),
                _result("Cancelled partial response.", finish_reason="cancelled"),
                _result("The final response is complete."),
            ],
        )
        self.assertEqual((backend.calls, result.content), (3, "The final response is complete."))
        self.assertEqual(runtime.post_tool_final_reuse_reused_count, 0)
        self.assertEqual(runtime.post_tool_final_reuse_last_reason, "finish_reason")

    def test_timeout_during_post_tool_route_never_reuses(self) -> None:
        backend = SequenceBackend(
            [
                _result("", finish_reason="tool_calls", tool_calls=[_tool_call("call-1", "ls -F")]),
                TimeoutError("synthetic timeout"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"ORBIT_POST_TOOL_FINAL_REUSE": "1"}, clear=False
        ):
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            with self.assertRaises(TimeoutError):
                runtime.ask_with_tools(
                    "list files",
                    temperature=0,
                    max_tokens=64,
                    workdir=Path(tmp),
                    tool_names=("exec_shell_full_command",),
                )
        self.assertEqual(backend.calls, 2)
        self.assertEqual(runtime.post_tool_final_reuse_reused_count, 0)
        self.assertEqual(runtime.post_tool_final_reuse_fallback_count, 0)

    def test_shell_error_uses_existing_direct_final_without_reuse(self) -> None:
        backend, runtime, result = self._run(
            "1",
            [
                _result("", finish_reason="tool_calls", tool_calls=[_tool_call("call-1", "false")]),
                _result("The command failed."),
            ],
        )
        self.assertEqual((backend.calls, result.content), (2, "The command failed."))
        self.assertEqual(runtime.post_tool_final_reuse_reused_count, 0)
        self.assertEqual(runtime.post_tool_final_reuse_fallback_count, 0)

    def test_second_tool_is_not_skipped(self) -> None:
        backend, runtime, result = self._run(
            "1",
            [
                _result("", finish_reason="tool_calls", tool_calls=[_tool_call("call-1", "printf first")]),
                _result("", finish_reason="tool_calls", tool_calls=[_tool_call("call-2", "printf second")]),
                _result("Both commands completed successfully."),
            ],
        )
        self.assertEqual((backend.calls, result.content), (3, "Both commands completed successfully."))
        self.assertEqual(runtime.post_tool_final_reuse_reused_count, 1)
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 2)

    def test_native_system_and_directory_families_can_reuse_complete_prose(self) -> None:
        cases = (
            (
                "system_info",
                {},
                "Show the computer specifications.",
                "The requested computer specifications are shown above.",
            ),
            (
                "list_directory",
                {"path": "."},
                "List this directory.",
                "The directory listing is complete.",
            ),
        )
        for name, arguments, prompt, prose in cases:
            with self.subTest(name=name):
                backend, runtime, result = self._run(
                    "1",
                    [
                        _result(
                            "",
                            finish_reason="tool_calls",
                            tool_calls=[_named_tool_call("call-1", name, arguments)],
                        ),
                        _result(prose),
                    ],
                    prompt=prompt,
                    tool_names=("exec_shell_full_command", name),
                )
                self.assertEqual((backend.calls, result.content), (2, prose))
                self.assertEqual(runtime.post_tool_final_reuse_reused_count, 1)

    def test_reset_clears_reuse_diagnostics(self) -> None:
        _, runtime, _ = self._run(
            "1",
            [
                _result("", finish_reason="tool_calls", tool_calls=[_tool_call("call-1", "ls -F")]),
                _result("The directory listing completed successfully."),
            ],
        )
        runtime.reset()
        self.assertEqual(runtime.post_tool_final_reuse_reused_count, 0)
        self.assertIsNone(runtime.post_tool_final_reuse_last_reason)


if __name__ == "__main__":
    unittest.main()
