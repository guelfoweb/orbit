from __future__ import annotations

import json
import os
from pathlib import Path
import random
import string
import tempfile
import unittest
from unittest import mock

from orbit.backend.base import ChatResult, Message
from orbit.runtime import ChatRuntime
from orbit.runtime.tool_backends import HybridToolExecutor
from orbit.runtime.tool_contract import validate_canonical_tool_call
from orbit.runtime.tools import ToolResult, default_tool_names, tool_definitions
from orbit.tool_contract_config import resolve_tool_call_canonical_gate


class CanonicalToolContractTests(unittest.TestCase):
    def _decision(
        self,
        name: object,
        arguments: object,
        *,
        allowed: tuple[str, ...] | None = None,
        workdir: Path | None = None,
        user_prompt: str = "perform the operation",
    ):
        return validate_canonical_tool_call(
            name,
            arguments,
            tool_definitions=tool_definitions(),
            allowed_tool_names=allowed if allowed is not None else default_tool_names(),
            workdir=workdir or Path.cwd(),
            user_prompt=user_prompt,
        )

    def test_config_is_default_on_and_invalid_fails_closed(self) -> None:
        self.assertTrue(resolve_tool_call_canonical_gate({}).enabled)
        self.assertFalse(resolve_tool_call_canonical_gate({"ORBIT_TOOL_CALL_CANONICAL_GATE": "0"}).enabled)
        self.assertTrue(resolve_tool_call_canonical_gate({"ORBIT_TOOL_CALL_CANONICAL_GATE": "1"}).enabled)
        invalid = resolve_tool_call_canonical_gate({"ORBIT_TOOL_CALL_CANONICAL_GATE": "true"})
        self.assertFalse(invalid.enabled)
        self.assertEqual(invalid.validation_error, "invalid_canonical_gate_value")

    def test_all_published_tool_schemas_match_the_strict_contract(self) -> None:
        definitions = {item["function"]["name"]: item["function"]["parameters"] for item in tool_definitions()}
        self.assertEqual(set(definitions), set(default_tool_names()))
        for name, schema in definitions.items():
            with self.subTest(name=name):
                self.assertEqual(schema["type"], "object")
                self.assertIs(schema["additionalProperties"], False)
                self.assertIn("required", schema)
                self.assertIsInstance(schema["required"], list)
        self.assertEqual(definitions["exec_shell_full_command"]["required"], ["command"])
        self.assertEqual(definitions["fetch_url"]["required"], ["url"])
        self.assertEqual(definitions["list_directory"]["required"], [])
        self.assertEqual(definitions["system_info"]["required"], [])
        self.assertEqual(definitions["exec_shell_full_command"]["properties"]["timeout"]["maximum"], 15)
        self.assertEqual(definitions["fetch_url"]["properties"]["timeout"]["maximum"], 15)
        self.assertEqual(definitions["list_directory"]["properties"]["max_entries"]["maximum"], 1000)
        self.assertEqual(definitions["list_directory"]["properties"]["max_depth"]["maximum"], 20)

    def test_api_reports_stage_outcomes_and_stable_rejections(self) -> None:
        cases = (
            ("system_info", {}, "accepted", None),
            ("unknown", {}, "rejected_permission", "tool_not_enabled"),
            ("fetch_url", {}, "rejected_schema", "missing_required"),
            ("fetch_url", {"url": 3}, "rejected_schema", "type_mismatch"),
            ("system_info", {"extra": True}, "rejected_schema", "additional_property"),
            ("list_directory", {"max_entries": 2000}, "rejected_guardrail", "limit_out_of_range"),
        )
        for name, arguments, terminal, code in cases:
            with self.subTest(name=name, arguments=arguments):
                decision = self._decision(name, arguments)
                self.assertEqual((decision.terminal_decision, decision.rejection_code), (terminal, code))

        denied = self._decision("system_info", {}, allowed=("fetch_url",))
        self.assertEqual((denied.terminal_decision, denied.rejection_code), ("rejected_permission", "tool_not_enabled"))
        unknown = self._decision("unknown", {}, allowed=("unknown",))
        self.assertEqual((unknown.terminal_decision, unknown.rejection_code), ("rejected_schema", "unknown_tool"))
        policy = self._decision(
            "exec_shell_full_command",
            {"command": "rm -f note.txt"},
            user_prompt="show note.txt",
        )
        self.assertEqual((policy.terminal_decision, policy.rejection_code), ("rejected_policy", "policy_read_only_mutation"))

    def test_duplicate_keys_and_ambiguous_argument_shapes_are_rejected(self) -> None:
        duplicate = self._decision("system_info", '{"include_cpu":true,"include_cpu":false}')
        array = self._decision("system_info", "[]")
        empty = self._decision("system_info", "")

        self.assertEqual((duplicate.terminal_decision, duplicate.rejection_code), ("rejected_parse", "duplicate_key"))
        self.assertEqual(array.rejection_code, "arguments_not_object")
        self.assertEqual(empty.rejection_code, "arguments_not_object")

    def test_gate_off_preserves_legacy_and_gate_on_rejects_legacy_dependencies(self) -> None:
        cases = (
            ("system_info", {"extra": True}, "additional_property"),
            ("system_info", {"include_cpu": "yes"}, "type_mismatch"),
            ("list_directory", {"max_entries": 2000}, "limit_out_of_range"),
            ("fetch_url", {}, "missing_required"),
            ("exec_shell_full_command", {"command": "printf ok", "timeout": "slow"}, "type_mismatch"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=None,
                workdir=Path(tmp),
                allowed_tool_names=default_tool_names(),
                user_prompt="perform the operation",
            )
            for name, arguments, code in cases:
                with self.subTest(name=name, arguments=arguments):
                    with mock.patch.dict(os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": "0"}, clear=False), mock.patch(
                        "orbit.runtime.tool_backends.execute_tool",
                        return_value=ToolResult(name=name, content="legacy-executed"),
                    ) as execute:
                        off = executor.execute(name, arguments, chunk_budget={})
                    self.assertEqual(off.terminal_outcome, "executed")
                    execute.assert_called_once()

                    with mock.patch.dict(os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": "1"}, clear=False), mock.patch(
                        "orbit.runtime.tool_backends.execute_tool"
                    ) as execute:
                        on = executor.execute(name, arguments, chunk_budget={})
                    self.assertEqual(on.terminal_reason, code)
                    execute.assert_not_called()

    def test_gate_off_on_equivalence_for_valid_calls_across_all_tools(self) -> None:
        cases = (
            ("system_info", {"include_cpu": False}),
            ("list_directory", {"path": ".", "recursive": True, "max_depth": 2}),
            ("fetch_url", {"url": "https://example.invalid/path", "timeout": 5}),
            ("exec_shell_full_command", {"command": "printf canonical-ok", "timeout": 5}),
        )
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=None,
                workdir=Path(tmp),
                allowed_tool_names=default_tool_names(),
                user_prompt="perform the operation",
            )
            for name, arguments in cases:
                observed = []
                for enabled in ("0", "1"):
                    with mock.patch.dict(os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": enabled}, clear=False), mock.patch(
                        "orbit.runtime.tool_backends.execute_tool",
                        return_value=ToolResult(name=name, content="same-result"),
                    ) as execute:
                        result = executor.execute(name, json.dumps(arguments), chunk_budget={})
                    call = execute.call_args
                    observed.append((result, call.args, call.kwargs))
                self.assertEqual(observed[0], observed[1])

    def test_gate_rejects_duplicate_keys_before_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": "1"}, clear=False
        ), mock.patch("orbit.runtime.tool_backends.execute_tool") as execute:
            result = HybridToolExecutor(
                backend=None,
                workdir=Path(tmp),
                allowed_tool_names=("system_info",),
            ).execute(
                "system_info",
                '{"include_cpu":true,"include_cpu":false}',
                chunk_budget={},
            )

        self.assertEqual((result.terminal_outcome, result.terminal_reason), ("rejected_parse", "duplicate_key"))
        execute.assert_not_called()

    def test_runtime_valid_tool_path_is_equivalent_off_on(self) -> None:
        class Backend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="", model="fake", finish_reason="tool_calls",
                        tool_calls=[{
                            "id": "call-1", "type": "function",
                            "function": {"name": "exec_shell_full_command", "arguments": '{"command":"printf gate-ok"}'},
                        }],
                        prompt_tokens=10, completion_tokens=2, cached_tokens=0,
                        prompt_tokens_per_second=None, generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="gate complete", model="fake", finish_reason="stop", tool_calls=[],
                    prompt_tokens=10, completion_tokens=2, cached_tokens=4,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        outcomes = []
        with tempfile.TemporaryDirectory() as tmp:
            for enabled in ("0", "1"):
                backend = Backend()
                tool_events = []
                result_events = []
                with mock.patch.dict(os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": enabled}, clear=False):
                    runtime = ChatRuntime(backend=backend, system_prompt=None)
                    result = runtime.ask_with_tools(
                        "print gate-ok", temperature=0, max_tokens=32, workdir=Path(tmp),
                        tool_names=("exec_shell_full_command",),
                        on_tool_call=lambda name, arguments: tool_events.append((name, arguments)),
                        on_tool_result=lambda name, chars, source, content: result_events.append((name, chars, source, content)),
                    )
                outcomes.append((result.content, result.finish_reason, backend.calls, tool_events, result_events))

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][:3], ("gate complete", "stop", 3))

    def test_gate_does_not_promote_unknown_tool_or_json_example(self) -> None:
        class Backend:
            def __init__(self, *, structured: bool) -> None:
                self.structured = structured
                self.calls = 0

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                self.calls += 1
                if self.structured and self.calls == 1:
                    return ChatResult(
                        content="", model="fake", finish_reason="tool_calls",
                        tool_calls=[{
                            "id": "call-web", "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query":"fixture"}'},
                        }],
                        prompt_tokens=5, completion_tokens=2, cached_tokens=0,
                        prompt_tokens_per_second=None, generation_tokens_per_second=None,
                    )
                if self.structured:
                    return ChatResult(
                        content="rejection reported", model="fake", finish_reason="stop", tool_calls=[],
                        prompt_tokens=5, completion_tokens=2, cached_tokens=0,
                        prompt_tokens_per_second=None, generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content='{"name":"system_info","arguments":{}}', model="fake", finish_reason="stop",
                    tool_calls=[], prompt_tokens=5, completion_tokens=5, cached_tokens=0,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            for structured in (True, False):
                with self.subTest(structured=structured), mock.patch.dict(
                    os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": "1"}, clear=False
                ), mock.patch("orbit.runtime.tool_backends.execute_tool") as execute:
                    runtime = ChatRuntime(backend=Backend(structured=structured), system_prompt=None)
                    tool_results = []
                    result = runtime.ask_with_tools(
                        "fixture request", temperature=0, max_tokens=16, workdir=Path(tmp),
                        tool_names=default_tool_names(),
                        on_tool_result=lambda name, chars, source, content: tool_results.append(content),
                    )
                execute.assert_not_called()
                if structured:
                    self.assertEqual(result.content, "rejection reported")
                    self.assertEqual(tool_results, ["error: tool not available for this turn: web_search"])
                else:
                    self.assertEqual(result.content, '{"name":"system_info","arguments":{}}')

    def test_gate_on_rejects_multiple_calls_without_executor(self) -> None:
        class Backend:
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                return ChatResult(
                    content="", model="fake", finish_reason="tool_calls",
                    tool_calls=[
                        {"id": "a", "type": "function", "function": {"name": "system_info", "arguments": "{}"}},
                        {"id": "b", "type": "function", "function": {"name": "system_info", "arguments": "{}"}},
                    ],
                    prompt_tokens=5, completion_tokens=4, cached_tokens=0,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": "1"}, clear=False
        ), mock.patch("orbit.runtime.tool_backends.execute_tool") as execute:
            result = ChatRuntime(backend=Backend(), system_prompt=None).ask_with_tools(
                "show specs twice", temperature=0, max_tokens=16, workdir=Path(tmp),
                tool_names=("system_info",),
            )

        execute.assert_not_called()
        self.assertEqual(result.finish_reason, "stop")
        self.assertIn("multiple_candidates", result.content)

    def test_gate_off_on_timeout_cancel_and_reset_are_equivalent(self) -> None:
        class TimeoutBackend:
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                raise TimeoutError("contract timeout")

        class CancelBackend:
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                return ChatResult(
                    content="", model="fake", finish_reason="cancelled", tool_calls=[],
                    prompt_tokens=1, completion_tokens=0, cached_tokens=0,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        outcomes = []
        with tempfile.TemporaryDirectory() as tmp:
            for enabled in ("0", "1"):
                with mock.patch.dict(os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": enabled}, clear=False):
                    timeout_runtime = ChatRuntime(backend=TimeoutBackend(), system_prompt=None)
                    with self.assertRaisesRegex(TimeoutError, "contract timeout"):
                        timeout_runtime.ask_with_tools(
                            "show specs", temperature=0, max_tokens=16, workdir=Path(tmp), tool_names=("system_info",)
                        )
                    timeout_runtime.reset()
                    cancel_runtime = ChatRuntime(backend=CancelBackend(), system_prompt=None)
                    cancelled = cancel_runtime.ask_with_tools(
                        "show specs", temperature=0, max_tokens=16, workdir=Path(tmp), tool_names=("system_info",)
                    )
                    cancel_runtime.reset()
                    outcomes.append((timeout_runtime.messages, cancelled.finish_reason, cancel_runtime.messages))

        self.assertEqual(outcomes, [([], "cancelled", []), ([], "cancelled", [])])

    def test_property_valid_inputs_are_value_preserving(self) -> None:
        rng = random.Random(149)
        alphabet = string.ascii_letters + string.digits + " -_./?=&'"
        for _ in range(300):
            command = "printf %s " + json.dumps("".join(rng.choice(alphabet) for _ in range(30)))
            arguments = {
                "command": command,
                "timeout": rng.randint(1, 10),
                "max_output_size": rng.randint(1, 4096),
            }
            decision = self._decision("exec_shell_full_command", json.dumps(arguments))
            self.assertTrue(decision.accepted)
            self.assertIsNotNone(decision.normalized_call)
            self.assertEqual(decision.normalized_call.name, "exec_shell_full_command")
            self.assertEqual(decision.normalized_call.arguments, arguments)

    def test_executor_success_error_and_shell_policy_remain_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": "1"}, clear=False
        ):
            executor = HybridToolExecutor(
                backend=None,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                user_prompt="perform the operation",
            )
            success = executor.execute("exec_shell_full_command", {"command": "printf ok"}, chunk_budget={})
            error = executor.execute("exec_shell_full_command", {"command": "sh -c 'exit 7'"}, chunk_budget={})
            policy = HybridToolExecutor(
                backend=None,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                user_prompt="show note.txt",
            ).execute("exec_shell_full_command", {"command": "rm -f note.txt"}, chunk_budget={})

        self.assertEqual(success.terminal_outcome, "executed")
        self.assertEqual(error.terminal_outcome, "runtime_error")
        self.assertEqual((policy.terminal_outcome, policy.terminal_reason), ("rejected_policy", "policy_read_only_mutation"))

    def test_gate_reuses_shell_policy_once_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": "1"}, clear=False
        ), mock.patch(
            "orbit.runtime.tool_contract.validate_read_only_shell_mutation",
            return_value=None,
        ) as read_only, mock.patch(
            "orbit.runtime.tool_contract.validate_shell_full_contract",
            return_value=None,
        ) as contract, mock.patch(
            "orbit.runtime.tool_backends.execute_tool",
            return_value=ToolResult("exec_shell_full_command", "ok"),
        ):
            result = HybridToolExecutor(
                backend=None,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                user_prompt="run printf ok",
            ).execute("exec_shell_full_command", {"command": "printf ok"}, chunk_budget={})

        self.assertEqual(result.terminal_outcome, "executed")
        read_only.assert_called_once()
        contract.assert_called_once()

    def test_runtime_preflight_is_the_only_canonical_validation(self) -> None:
        class Backend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="", model="fake", finish_reason="tool_calls",
                        tool_calls=[{
                            "id": "call-1", "type": "function",
                            "function": {"name": "system_info", "arguments": "{}"},
                        }],
                        prompt_tokens=10, completion_tokens=2, cached_tokens=0,
                        prompt_tokens_per_second=None, generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done", model="fake", finish_reason="stop", tool_calls=[],
                    prompt_tokens=10, completion_tokens=1, cached_tokens=4,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        from orbit.runtime import tool_loop as tool_loop_module
        original = tool_loop_module.validate_canonical_tool_call_payload
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": "1"}, clear=False
        ), mock.patch(
            "orbit.runtime.tool_loop.validate_canonical_tool_call_payload", wraps=original
        ) as preflight, mock.patch(
            "orbit.runtime.tool_backends.validate_canonical_tool_call",
            side_effect=AssertionError("executor repeated canonical validation"),
        ), mock.patch(
            "orbit.runtime.tool_backends.execute_tool",
            return_value=ToolResult("system_info", "fixture specs"),
        ):
            backend = Backend()
            result = ChatRuntime(backend=backend, system_prompt=None).ask_with_tools(
                "show specs", temperature=0, max_tokens=32, workdir=Path(tmp),
                tool_names=("system_info",),
            )

        self.assertEqual((result.content, result.finish_reason, backend.calls), ("done", "stop", 2))
        preflight.assert_called_once()
