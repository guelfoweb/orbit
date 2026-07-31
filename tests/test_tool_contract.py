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
from orbit.runtime.shell_guardrails import validate_tool_no_mutation_policy
from orbit.runtime.tool_backends import HybridToolExecutor
from orbit.runtime.tool_contract import validate_canonical_tool_call
from orbit.runtime.tools import (
    ToolResult,
    agent_tool_names,
    default_tool_names,
    execute_tool,
    tool_definitions,
)
from orbit.tool_contract_config import resolve_tool_call_canonical_gate


class CanonicalToolContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._post_tool_final_reuse_env = mock.patch.dict(
            os.environ,
            {"ORBIT_POST_TOOL_FINAL_REUSE": "0"},
        )
        self._post_tool_final_reuse_env.start()
        self.addCleanup(self._post_tool_final_reuse_env.stop)

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
        definitions = {
            item["function"]["name"]: item["function"]["parameters"]
            for item in tool_definitions(agent_tool_names())
        }
        self.assertEqual(set(definitions), set(agent_tool_names()))
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
        self.assertEqual(definitions["apply_patch"]["required"], ["patch"])
        self.assertEqual(definitions["exec_shell_full_command"]["properties"]["timeout"]["maximum"], 15)
        self.assertEqual(definitions["fetch_url"]["properties"]["timeout"]["maximum"], 15)
        self.assertEqual(definitions["list_directory"]["properties"]["max_entries"]["maximum"], 1000)
        self.assertEqual(definitions["list_directory"]["properties"]["max_depth"]["maximum"], 20)
        self.assertIs(definitions["apply_patch"]["additionalProperties"], False)

    def test_apply_patch_uses_the_same_schema_permission_policy_and_operational_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "item.txt").write_text("before\n", encoding="utf-8")
            patch = "--- item.txt\n+++ item.txt\n@@ -1 +1 @@\n-before\n+after\n"

            accepted = validate_canonical_tool_call(
                "apply_patch",
                {"patch": patch},
                tool_definitions=tool_definitions(agent_tool_names()),
                allowed_tool_names=agent_tool_names(),
                workdir=root,
                user_prompt="Change item.txt from before to after.",
            )
            denied = validate_canonical_tool_call(
                "apply_patch",
                {"patch": patch},
                tool_definitions=tool_definitions(agent_tool_names()),
                allowed_tool_names=default_tool_names(),
                workdir=root,
                user_prompt="Change item.txt from before to after.",
            )
            read_only = validate_canonical_tool_call(
                "apply_patch",
                {"patch": patch},
                tool_definitions=tool_definitions(agent_tool_names()),
                allowed_tool_names=agent_tool_names(),
                workdir=root,
                user_prompt="Read item.txt.",
            )
            invalid = validate_canonical_tool_call(
                "apply_patch",
                {"patch": patch.replace("-before", "-missing")},
                tool_definitions=tool_definitions(agent_tool_names()),
                allowed_tool_names=agent_tool_names(),
                workdir=root,
                user_prompt="Change item.txt from before to after.",
            )

        self.assertTrue(accepted.accepted)
        self.assertEqual((denied.terminal_decision, denied.rejection_code), ("rejected_permission", "tool_not_enabled"))
        self.assertEqual((read_only.terminal_decision, read_only.rejection_code), ("rejected_policy", "policy_read_only_mutation"))
        self.assertEqual((invalid.terminal_decision, invalid.rejection_code), ("rejected_guardrail", "context_mismatch"))

    def test_apply_patch_obeys_global_and_mixed_constraints_but_ignores_inert_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "item.txt").write_text("before\n", encoding="utf-8")
            patch = "--- item.txt\n+++ item.txt\n@@ -1 +1 @@\n-before\n+after\n"
            cases = (
                ("Analyze without changing any files.", False, "read-only request rejected"),
                ("Inspect without changing files, then fix item.txt.", False, "mixed or scoped"),
                ('Change item.txt to contain "without changing any files".', True, None),
            )
            for prompt, accepted, message in cases:
                with self.subTest(prompt=prompt):
                    decision = validate_canonical_tool_call(
                        "apply_patch",
                        {"patch": patch},
                        tool_definitions=tool_definitions(agent_tool_names()),
                        allowed_tool_names=agent_tool_names(),
                        workdir=root,
                        user_prompt=prompt,
                    )
                    self.assertIs(decision.accepted, accepted)
                    if message is not None:
                        self.assertIn(message, decision.policy_outcome.message or "")

    def test_explicit_no_mutation_policy_rejects_before_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "orbit.runtime.tool_backends.execute_tool"
        ) as execute:
            result = HybridToolExecutor(
                backend=None,
                workdir=Path(tmp),
                allowed_tool_names=default_tool_names(),
                user_prompt="Analyze the project without changing any files.",
            ).execute(
                "exec_shell_full_command",
                {"command": "cd . && touch marker.txt"},
                chunk_budget={},
            )

        self.assertEqual((result.terminal_outcome, result.terminal_reason), ("rejected_policy", "policy_read_only_mutation"))
        execute.assert_not_called()

    def test_no_mutation_policy_survives_canonical_and_healing_kill_switches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "item.txt"
            patch = "--- item.txt\n+++ item.txt\n@@ -1 +1 @@\n-before\n+after\n"
            for gate in ("0", "1"):
                for healing in ("0", "1"):
                    with self.subTest(gate=gate, healing=healing), mock.patch.dict(
                        os.environ,
                        {
                            "ORBIT_TOOL_CALL_CANONICAL_GATE": gate,
                            "ORBIT_TOOL_CALL_HEALING": healing,
                        },
                        clear=False,
                    ):
                        target.write_text("before\n", encoding="utf-8")
                        executor = HybridToolExecutor(
                            backend=None,
                            workdir=root,
                            allowed_tool_names=agent_tool_names(),
                            user_prompt="Analyze without changing any files.",
                        )
                        shell = executor.execute(
                            "exec_shell_full_command",
                            {"command": "cat README.md"},
                            chunk_budget={},
                        )
                        applied = executor.execute(
                            "apply_patch",
                            {"patch": patch},
                            chunk_budget={},
                        )

                        self.assertEqual(
                            (shell.terminal_outcome, shell.terminal_reason),
                            ("rejected_policy", "policy_read_only_mutation"),
                        )
                        self.assertEqual(
                            (applied.terminal_outcome, applied.terminal_reason),
                            ("rejected_policy", "policy_read_only_mutation"),
                        )
                        self.assertEqual(target.read_text(encoding="utf-8"), "before\n")

    def test_common_global_forms_are_enforced_with_canonical_gate_on_and_off(self) -> None:
        prompts = (
            "Keep all files unchanged.",
            "Files must not be modified.",
            "Please refrain from changing files.",
            "Never modify files.",
            "You must not modify files.",
            "Avoid modifying files.",
            "Don\u2019t modify files.",
            "Without altering files, report what you find.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for gate in ("0", "1"):
                for prompt in prompts:
                    with self.subTest(gate=gate, prompt=prompt), mock.patch.dict(
                        os.environ,
                        {"ORBIT_TOOL_CALL_CANONICAL_GATE": gate},
                        clear=False,
                    ):
                        result = HybridToolExecutor(
                            backend=None,
                            workdir=root,
                            allowed_tool_names=default_tool_names(),
                            user_prompt=prompt,
                        ).execute(
                            "exec_shell_full_command",
                            {"command": "touch marker.txt"},
                            chunk_budget={},
                        )

                        self.assertEqual(result.terminal_outcome, "rejected_policy")
                        self.assertFalse((root / "marker.txt").exists())

    def test_descriptive_no_changes_text_allows_mutation_with_gate_on_or_off(self) -> None:
        prompt = "There are no file changes in the current status. Create report.txt."
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for gate in ("0", "1"):
                with self.subTest(gate=gate), mock.patch.dict(
                    os.environ,
                    {"ORBIT_TOOL_CALL_CANONICAL_GATE": gate},
                    clear=False,
                ):
                    marker = root / "marker.txt"
                    marker.unlink(missing_ok=True)
                    result = HybridToolExecutor(
                        backend=None,
                        workdir=root,
                        allowed_tool_names=default_tool_names(),
                        user_prompt=prompt,
                    ).execute(
                        "exec_shell_full_command",
                        {"command": "printf report > marker.txt"},
                        chunk_budget={},
                    )

                    self.assertEqual(result.terminal_outcome, "executed")
                    self.assertEqual(marker.read_text(encoding="utf-8"), "report")

    def test_ambiguous_markdown_lists_fail_closed_with_gate_on_or_off(self) -> None:
        prompts = (
            "Use the following bullets:\n- Do not modify files.\n- Delete old.txt.",
            "Create report.md with the following bullets:\n- do not modify files",
            "Generate report.md with the following bullets:\n- do not modify files",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for gate in ("0", "1"):
                for prompt in prompts:
                    with self.subTest(gate=gate, prompt=prompt), mock.patch.dict(
                        os.environ,
                        {"ORBIT_TOOL_CALL_CANONICAL_GATE": gate},
                        clear=False,
                    ):
                        marker = root / "marker.txt"
                        marker.unlink(missing_ok=True)
                        result = HybridToolExecutor(
                            backend=None,
                            workdir=root,
                            allowed_tool_names=default_tool_names(),
                            user_prompt=prompt,
                        ).execute(
                            "exec_shell_full_command",
                            {"command": "printf unsafe > marker.txt"},
                            chunk_budget={},
                        )

                        self.assertEqual(
                            (result.terminal_outcome, result.terminal_reason),
                            ("rejected_policy", "policy_read_only_mutation"),
                        )
                        self.assertIn("mixed or scoped mutation constraint", result.result.content)
                        self.assertFalse(marker.exists())

    def test_markdown_payload_headers_do_not_block_mutation_with_gate_on_or_off(self) -> None:
        prompts = (
            "Here is a Markdown example:\n- do not modify files",
            "The following Markdown payload:\n- do not modify files",
            "Copy the following Markdown into report.md:\n- do not modify files",
            "Output this Markdown:\n- do not modify files",
            "Return this Markdown:\n- do not modify files",
            "Provide this Markdown:\n- do not modify files",
            "Put this Markdown in report.md:\n- do not modify files",
            "Use the following Markdown payload:\n- do not modify files",
            "Replace report.md with this content:\n- do not modify files",
            "Append the following content to report.md:\n- do not modify files",
            "Paste this Markdown into report.md:\n- do not modify files",
            "Create report.txt with this payload:\ndo not modify files",
            "Create quote.txt containing \u201cdo not modify files\u201d.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for gate in ("0", "1"):
                for prompt in prompts:
                    with self.subTest(gate=gate, prompt=prompt), mock.patch.dict(
                        os.environ,
                        {"ORBIT_TOOL_CALL_CANONICAL_GATE": gate},
                        clear=False,
                    ):
                        marker = root / "marker.txt"
                        marker.unlink(missing_ok=True)
                        result = HybridToolExecutor(
                            backend=None,
                            workdir=root,
                            allowed_tool_names=default_tool_names(),
                            user_prompt=prompt,
                        ).execute(
                            "exec_shell_full_command",
                            {"command": "printf created > marker.txt"},
                            chunk_budget={},
                        )

                        self.assertEqual(result.terminal_outcome, "executed")
                        self.assertEqual(marker.read_text(encoding="utf-8"), "created")

    def test_no_mutation_policy_covers_agent_non_agent_and_direct_executor_paths(self) -> None:
        prompt = "Inspect the project without changing any files."
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "item.txt"
            target.write_text("before\n", encoding="utf-8")
            patch = "--- item.txt\n+++ item.txt\n@@ -1 +1 @@\n-before\n+after\n"

            for allowed in (default_tool_names(), agent_tool_names()):
                with self.subTest(allowed=allowed):
                    result = HybridToolExecutor(
                        backend=None,
                        workdir=root,
                        allowed_tool_names=allowed,
                        user_prompt=prompt,
                    ).execute(
                        "exec_shell_full_command",
                        {"command": "printf bypass > marker.txt"},
                        chunk_budget={},
                    )
                    self.assertEqual(result.terminal_outcome, "rejected_policy")
                    self.assertFalse((root / "marker.txt").exists())

            direct_shell = execute_tool(
                "exec_shell_full_command",
                {"command": "printf bypass > marker.txt"},
                workdir=root,
                user_prompt=prompt,
            )
            direct_patch = execute_tool(
                "apply_patch",
                {"patch": patch},
                workdir=root,
                user_prompt=prompt,
            )

            self.assertIn("unrestricted shell command", direct_shell.content)
            self.assertIn("read-only request rejected file patch", direct_patch.content)
            self.assertFalse((root / "marker.txt").exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "before\n")

    def test_shell_bypass_matrix_never_reaches_direct_dispatch(self) -> None:
        commands = (
            "./cat README.md",
            "/bin/cat README.md",
            "env cat README.md",
            "command cat README.md",
            "sed -n 'w marker.txt' README.md",
            "sed -i 's/a/b/' README.md",
            "git diff --output=marker",
            "find . -exec touch marker.txt ';'",
            "printf marker | xargs touch",
            "python3 -c 'open(\"marker.txt\", \"w\").write(\"x\")'",
            "cat $(touch marker.txt)",
            "cat README.md > copy.txt",
            "grep Orbit README.md | tee copy.txt",
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "orbit.runtime.tools.execute_exec_shell_full_command"
        ) as shell:
            root = Path(tmp)
            for command in commands:
                with self.subTest(command=command):
                    result = execute_tool(
                        "exec_shell_full_command",
                        {"command": command},
                        workdir=root,
                        user_prompt="Analyze without changing any files.",
                    )
                    self.assertIn("unrestricted shell command", result.content)

            shell.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_structured_read_only_tools_remain_available_under_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "item.txt").write_text("content\n", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=None,
                workdir=root,
                allowed_tool_names=default_tool_names(),
                user_prompt="Inspect without changing any files.",
            )

            listing = executor.execute("list_directory", {"path": "."}, chunk_budget={})
            system = executor.execute("system_info", {}, chunk_budget={})

            self.assertEqual((listing.terminal_outcome, system.terminal_outcome), ("executed", "executed"))
            self.assertIn("item.txt", listing.result.content)
            self.assertIn("OS:", system.result.content)

    def test_tool_output_text_cannot_activate_no_mutation_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "orbit.runtime.tool_backends.execute_tool",
            side_effect=(
                ToolResult(name="exec_shell_full_command", content="without changing any files"),
                ToolResult(name="exec_shell_full_command", content="created"),
            ),
        ) as execute:
            executor = HybridToolExecutor(
                backend=None,
                workdir=Path(tmp),
                allowed_tool_names=default_tool_names(),
                user_prompt="Create marker.txt after reading the prior output.",
            )
            first = executor.execute(
                "exec_shell_full_command",
                {"command": "printf observation"},
                chunk_budget={},
            )
            second = executor.execute(
                "exec_shell_full_command",
                {"command": "touch marker.txt"},
                chunk_budget={},
            )

        self.assertEqual((first.terminal_outcome, second.terminal_outcome), ("executed", "executed"))
        self.assertEqual(execute.call_count, 2)

    def test_tool_output_prefix_is_never_reclassified_as_policy(self) -> None:
        content = "error: read-only request rejected unrestricted shell command"
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "orbit.runtime.tool_backends.execute_tool",
            return_value=ToolResult(name="exec_shell_full_command", content=content),
        ):
            result = HybridToolExecutor(
                backend=None,
                workdir=Path(tmp),
                allowed_tool_names=default_tool_names(),
                user_prompt="Create marker.txt.",
            ).execute(
                "exec_shell_full_command",
                {"command": "printf changed > marker.txt"},
                chunk_budget={},
            )

        self.assertEqual(
            (result.terminal_outcome, result.terminal_reason),
            ("runtime_error", "tool_error"),
        )
        self.assertEqual(result.result.content, content)

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

    def test_gate_and_dispatch_use_the_same_shell_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": "1"}, clear=False
        ), mock.patch(
            "orbit.runtime.tool_contract.validate_tool_no_mutation_policy",
            wraps=validate_tool_no_mutation_policy,
        ) as no_mutation, mock.patch(
            "orbit.runtime.tool_contract.validate_shell_full_contract",
            return_value=None,
        ) as contract, mock.patch(
            "orbit.runtime.tools.validate_tool_no_mutation_policy",
            wraps=validate_tool_no_mutation_policy,
        ) as executor_policy:
            result = HybridToolExecutor(
                backend=None,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                user_prompt="run printf ok",
            ).execute("exec_shell_full_command", {"command": "printf ok"}, chunk_budget={})

        self.assertEqual(result.terminal_outcome, "executed")
        no_mutation.assert_called_once()
        contract.assert_called_once()
        executor_policy.assert_called_once()

    def test_legacy_path_and_dispatch_use_the_same_shell_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"ORBIT_TOOL_CALL_CANONICAL_GATE": "0"}, clear=False
        ), mock.patch(
            "orbit.runtime.tool_backends.validate_tool_no_mutation_policy",
            wraps=validate_tool_no_mutation_policy,
        ) as no_mutation, mock.patch(
            "orbit.runtime.tools.validate_tool_no_mutation_policy",
            wraps=validate_tool_no_mutation_policy,
        ) as executor_policy:
            result = HybridToolExecutor(
                backend=None,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                user_prompt="run printf ok",
            ).execute("exec_shell_full_command", {"command": "printf ok"}, chunk_budget={})

        self.assertEqual(result.terminal_outcome, "executed")
        no_mutation.assert_called_once()
        executor_policy.assert_called_once()

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
