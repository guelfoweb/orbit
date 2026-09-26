"""Completion state is execution authority, independently of parser success."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from orbit.backend.base import ChatResult
from orbit.runtime import ChatRuntime
from orbit.runtime.tools import ToolResult


CALL = {"id": "safe-fixture", "type": "function",
        "function": {"name": "system_info", "arguments": "{}"}}


def completion(reason="stop", calls=(), content=""):
    return ChatResult(content, "fixture", reason, copy.deepcopy(list(calls)),
                      10, 3, 0, None, None)


class Backend:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return next(self.results, completion(content="Done."))


class ToolCompletionAuthorityTests(unittest.TestCase):
    def run_case(self, result, *, auto=False, gate="1", preceding=()):
        backend = Backend([*preceding, result])
        runtime = ChatRuntime(backend=backend, system_prompt=None)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {
            "ORBIT_TOOL_CALL_CANONICAL_GATE": gate,
            "ORBIT_POST_TOOL_FINAL_REUSE": "0",
        }), mock.patch("orbit.runtime.tool_backends.execute_tool",
                       return_value=ToolResult("system_info", "Fixture result.")) as execute:
            kwargs = dict(temperature=0, max_tokens=128, max_loops=2, workdir=Path(tmp))
            if auto:
                final = runtime.ask_auto("Inspect system information.",
                                         allowed_tool_names=("system_info",), **kwargs)
            else:
                final = runtime.ask_with_tools("Inspect system information.",
                                               tool_names=("system_info",), **kwargs)
        return final, execute.call_count, runtime.messages, backend.calls

    def test_stopped_and_tool_calls_completions_dispatch(self):
        for reason in ("stop", "tool_calls"):
            for auto in (False, True):
                with self.subTest(reason=reason, auto=auto):
                    _, count, _, _ = self.run_case(completion(reason, [CALL]), auto=auto)
                    self.assertEqual(count, 1)

    def test_incomplete_completion_never_dispatches_or_commits_call(self):
        for reason in ("length", "cancelled", "timeout", "error", "incomplete",
                       "truncated", "content_filter", None, "unknown"):
            for auto in (False, True):
                for gate in ("0", "1"):
                    with self.subTest(reason=reason, auto=auto, gate=gate):
                        final, count, history, _ = self.run_case(
                            completion(reason, [CALL], "Partial trailing output"),
                            auto=auto, gate=gate)
                        self.assertEqual(count, 0)
                        self.assertFalse(final.tool_calls)
                        self.assertFalse(any(m.get("tool_calls") or m["role"] == "tool"
                                             for m in history))

    def test_textual_tool_normalization_cannot_launder_terminal_state(self):
        raw = json.dumps({"name": "system_info", "arguments": {}})
        for reason in ("cancelled", "timeout", "incomplete", "length"):
            with self.subTest(reason=reason):
                _, count, _, _ = self.run_case(completion(reason, content=raw))
                self.assertEqual(count, 0)

    def test_retry_result_gets_its_own_completion_check(self):
        for reason in ("length", "cancelled", "timeout"):
            with self.subTest(reason=reason):
                _, count, _, _ = self.run_case(completion(reason, [CALL]),
                                               preceding=[completion("length", content="Partial")])
                self.assertEqual(count, 0)

    def test_complete_retry_can_execute(self):
        _, count, _, _ = self.run_case(completion("tool_calls", [CALL]),
                                       preceding=[completion("length", content="Partial")])
        self.assertEqual(count, 1)

    def test_post_tool_cancel_preserves_finalization_without_another_execution(self):
        call = {"id": "first", "type": "function", "function": {
            "name": "exec_shell_full_command", "arguments": '{"command":"printf ok"}'}}
        for second_has_call in (False, True):
            with self.subTest(second_has_call=second_has_call):
                backend = Backend([completion("tool_calls", [call]),
                                   completion("cancelled", [call] if second_has_call else [],
                                              "Cancelled partial response."),
                                   completion(content="Final from committed evidence.")])
                runtime = ChatRuntime(backend=backend, system_prompt=None)
                with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {
                    "ORBIT_POST_TOOL_FINAL_REUSE": "1",
                }), mock.patch("orbit.runtime.tool_backends.execute_tool",
                               return_value=ToolResult("exec_shell_full_command", "ok")) as execute:
                    final = runtime.ask_with_tools(
                        "Inspect the fixture.", temperature=0, max_tokens=128, max_loops=3,
                        workdir=Path(tmp), tool_names=("exec_shell_full_command",))
                self.assertEqual(execute.call_count, 1)
                self.assertEqual(sum(bool(m.get('tool_calls')) for m in runtime.messages), 1)
                self.assertFalse(final.tool_calls)
                self.assertEqual(backend.calls, 2 if second_has_call else 3)
                self.assertEqual(final.finish_reason, 'cancelled' if second_has_call else 'stop')
