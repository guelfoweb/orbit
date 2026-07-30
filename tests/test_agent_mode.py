from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult, Message
from orbit.runtime import ChatRuntime
from orbit.runtime.agent_action_review import AGENT_ACTION_REVIEW_SYSTEM_PROMPT
from orbit.runtime.messages import (
    AGENT_ACTION_ANCHOR_TEMPLATE,
    AGENT_FINAL_COMPLETION_INSTRUCTION,
    AGENT_ROUTE_CONTROL_INSTRUCTION,
    AGENT_ROUTE_SYSTEM_PROMPT,
    AGENT_STRICT_TOOL_CALL_SYSTEM_PROMPT,
    AGENT_TOOL_CONTINUATION_SYSTEM_PROMPT,
    FINAL_FROM_TOOL_SYSTEM_PROMPT,
    ROUTE_SYSTEM_PROMPT,
    TOOL_CALL_SYSTEM_PROMPT,
)
from orbit.runtime.shell_guardrails import (
    SHELL_FULL_AGENT_MUTATION_VERIFICATION_PROMPT,
    SHELL_FULL_AGENT_SEMANTIC_COMPLETION_PROMPT,
)


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
        prompt_tokens=8,
        completion_tokens=2,
        cached_tokens=0,
        prompt_tokens_per_second=None,
        generation_tokens_per_second=None,
    )


def _review(decision: str, reason: str | None = None) -> ChatResult:
    payload = {
        "decision": decision,
        "reason": reason or "action is within the requested scope",
    }
    return _result(json.dumps(payload, separators=(",", ":")))


def _tool_call(name: str, arguments: dict[str, object], *, call_id: str) -> dict[str, object]:
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, separators=(",", ":")),
        },
    }


class ScriptedBackend:
    def __init__(self, outputs: list[ChatResult]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.messages_seen: list[list[Message]] = []

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.messages_seen.append(messages)
        if self.calls >= len(self.outputs):
            raise AssertionError(f"unexpected model call {self.calls + 1}")
        result = self.outputs[self.calls]
        self.calls += 1
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


class AgentModeTests(unittest.TestCase):
    def test_agent_route_final_control_uses_compact_final_without_post_tool_route(self) -> None:
        backend = ScriptedBackend(
            [
                _result('{"include_cpu":true,"include_os":true,"after":"final"}'),
                _result("OS and CPU reported."),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Show the OS and CPU, then report them.",
                temperature=0,
                max_tokens=128,
                workdir=Path(tmp),
                allowed_tool_names=("system_info", "exec_shell_full_command"),
                agent_mode=True,
            )

        self.assertEqual(result.content, "OS and CPU reported.")
        self.assertEqual(backend.calls, 2)
        self.assertEqual(backend.messages_seen[0][0]["content"], ROUTE_SYSTEM_PROMPT)
        self.assertEqual(backend.messages_seen[0][1]["content"], AGENT_ROUTE_CONTROL_INSTRUCTION)
        self.assertEqual(backend.messages_seen[1][0]["content"], FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertNotIn(AGENT_TOOL_CONTINUATION_SYSTEM_PROMPT, str(backend.messages_seen[1]))

    def test_agent_route_without_after_control_conservatively_continues(self) -> None:
        backend = ScriptedBackend(
            [
                _result('{"include_cpu":true,"include_os":true}'),
                _result("OS and CPU reported."),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Show the OS and CPU, then report them.",
                temperature=0,
                max_tokens=128,
                workdir=Path(tmp),
                allowed_tool_names=("system_info", "exec_shell_full_command"),
                agent_mode=True,
            )

        self.assertEqual(result.content, "OS and CPU reported.")
        self.assertEqual(backend.calls, 2)
        self.assertEqual(backend.messages_seen[1][0]["content"], AGENT_TOOL_CONTINUATION_SYSTEM_PROMPT)

    def test_agent_mode_reconsiders_canonical_rejection_before_execution(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "printf 'broken", "after": "continue"})),
                _result(json.dumps({"command": "pwd"})),
                _result("The working directory was reported."),
            ]
        )
        tool_results: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Report the working directory.",
                temperature=0,
                max_tokens=128,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_tool_result=lambda _name, _chars, _source, content: tool_results.append(content),
            )

        self.assertEqual(result.content, "The working directory was reported.")
        self.assertIn("invalid_shell_syntax", tool_results[0])
        self.assertEqual(backend.messages_seen[1][0]["content"], AGENT_STRICT_TOOL_CALL_SYSTEM_PROMPT)
        self.assertEqual(backend.messages_seen[2][0]["content"], AGENT_TOOL_CONTINUATION_SYSTEM_PROMPT)
        self.assertEqual(backend.calls, 3)

    def test_agent_mode_keeps_authorized_tools_after_initial_directory_discovery(self) -> None:
        backend = ScriptedBackend(
            [
                _result(
                    "",
                    tool_calls=[
                        {
                            "id": "route-list",
                            "function": {
                                "name": "list_directory",
                                "arguments": '{"path":".","recursive":false}',
                            },
                        }
                    ],
                ),
                _result(json.dumps({"command": "cat notes.txt"})),
                _result("The note contains alpha."),
            ]
        )
        tool_names: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "notes.txt").write_text("alpha\n", encoding="utf-8")
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Find the notes file, read it, and report its contents.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command", "list_directory"),
                agent_mode=True,
                on_final_delta=lambda _text: None,
                on_tool_call=lambda name, _arguments: tool_names.append(name),
            )

        self.assertEqual(result.content, "The note contains alpha.")
        self.assertEqual(tool_names, ["list_directory", "exec_shell_full_command"])
        self.assertEqual(backend.calls, 3)

    def test_agent_mode_retries_malformed_structural_route_without_executing_it(self) -> None:
        backend = ScriptedBackend(
            [
                _result('<|tool_call>call:{"path":".","recursive":false}<tool_call|>'),
                _result(json.dumps({"command": "pwd"})),
                _result("The current directory was confirmed."),
            ]
        )
        tool_calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Report the current working directory.",
                temperature=0,
                max_tokens=128,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_final_delta=lambda _text: None,
                on_tool_call=lambda _name, arguments: tool_calls.append(arguments),
            )

        self.assertEqual(result.content, "The current directory was confirmed.")
        self.assertEqual(runtime.agent_route_repairs, 1)
        self.assertEqual(tool_calls, ['{"command": "pwd"}'])
        self.assertEqual(backend.calls, 3)

    def test_agent_mode_fails_closed_when_structural_route_retry_is_still_invalid(self) -> None:
        backend = ScriptedBackend(
            [
                _result('<|tool_call>call:{"path":".","recursive":false}<tool_call|>'),
                _result("The request could not be completed."),
            ]
        )
        tool_calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Inspect the project.",
                temperature=0,
                max_tokens=128,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_tool_call=lambda _name, arguments: tool_calls.append(arguments),
            )

        self.assertEqual(result.content, "The request could not be completed.")
        self.assertEqual(runtime.agent_route_repairs, 1)
        self.assertEqual(tool_calls, [])
        self.assertEqual(backend.calls, 2)

    def test_agent_mode_reviews_registered_tool_name_used_as_shell_command(self) -> None:
        backend = ScriptedBackend(
            [
                _result('{"command":"list_directory","after":"continue"}'),
                _result(
                    "",
                    tool_calls=[
                        _tool_call(
                            "list_directory",
                            {"path": ".", "recursive": False},
                            call_id="list",
                        )
                    ],
                ),
                _result("The directory contains item.txt."),
            ]
        )
        tool_names: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "item.txt").write_text("value\n", encoding="utf-8")
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "List the current directory and report its files.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command", "list_directory"),
                agent_mode=True,
                on_tool_call=lambda name, _arguments: tool_names.append(name),
            )

        self.assertEqual(result.content, "The directory contains item.txt.")
        self.assertEqual(tool_names, ["list_directory"])
        self.assertEqual(runtime.agent_action_reviews, 0)
        self.assertEqual(runtime.agent_action_review_revisions, 1)
        self.assertEqual(backend.calls, 3)

    def test_default_path_still_hands_direct_evidence_to_final(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "cat facts.txt"})),
                _result("alpha"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "facts.txt").write_text("alpha\n", encoding="utf-8")
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Read facts.txt.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "alpha")
        self.assertEqual(backend.calls, 2)
        self.assertNotIn(AGENT_TOOL_CONTINUATION_SYSTEM_PROMPT, str(backend.messages_seen))
        self.assertNotIn(AGENT_FINAL_COMPLETION_INSTRUCTION, str(backend.messages_seen))

    def test_agent_mode_allows_model_to_continue_after_partial_evidence(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "grep -c ERROR service.log"})),
                _result(json.dumps({"command": "grep ERROR service.log | sort | uniq -c"})),
                _result("E_AUTH occurred once and E_CONN occurred twice."),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "service.log").write_text("ERROR E_CONN\nERROR E_AUTH\nERROR E_CONN\n", encoding="utf-8")
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Report each error code and its exact count from service.log.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_final_delta=lambda _text: None,
            )

        self.assertEqual(result.content, "E_AUTH occurred once and E_CONN occurred twice.")
        self.assertEqual(backend.calls, 3)
        self.assertEqual(backend.messages_seen[1][0]["content"], AGENT_TOOL_CONTINUATION_SYSTEM_PROMPT)
        self.assertIn(
            AGENT_ACTION_ANCHOR_TEMPLATE.format(
                user_prompt="Report each error code and its exact count from service.log."
            ),
            [message["content"] for message in backend.messages_seen[1]],
        )

    def test_agent_mode_recovers_from_repairable_shell_failure(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "sh -c 'exit 7'"})),
                _review("approve"),
                _result(json.dumps({"command": "printf 'ready\\n' > ready.txt"})),
                _review("approve"),
                _result(json.dumps({"command": "cat ready.txt"})),
                _result("OK"),
                _result("Created and verified ready.txt."),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Create ready.txt when absent and verify that it contains ready.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_final_delta=lambda _text: None,
            )
            content = (workdir / "ready.txt").read_text(encoding="utf-8")

        self.assertEqual(content, "ready\n")
        self.assertEqual(result.content, "Created and verified ready.txt.")
        self.assertEqual(runtime.agent_error_continuations, 1)
        self.assertEqual(runtime.mutation_verifications, 1)
        self.assertEqual(runtime.agent_semantic_completion_retries, 1)
        self.assertEqual(runtime.post_tool_final_reuse_reused_count, 1)
        self.assertEqual(backend.calls, 7)
        self.assertEqual(backend.messages_seen[1][0]["content"], AGENT_ACTION_REVIEW_SYSTEM_PROMPT)
        completion_messages = "\n".join(
            message["content"] for message in backend.messages_seen[-1]
        )
        self.assertIn(SHELL_FULL_AGENT_SEMANTIC_COMPLETION_PROMPT, completion_messages)
        self.assertIn(
            "Create ready.txt when absent and verify that it contains ready.",
            completion_messages,
        )

    def test_agent_mode_requires_tool_observation_for_mutation_verification(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "printf 'alpha\\n' > item.txt"})),
                _review("approve"),
                _result("The file is correct."),
                _result(json.dumps({"command": "cat item.txt"})),
                _result("OK"),
                _result("Created and verified item.txt."),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Create item.txt containing alpha and verify it.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_final_delta=lambda _text: None,
            )

        self.assertEqual(result.content, "Created and verified item.txt.")
        self.assertEqual(runtime.agent_verification_retries, 1)
        self.assertEqual(runtime.mutation_verification_repairs, 1)
        self.assertEqual(runtime.mutation_verification_failures, 0)
        verification_messages = "\n".join(
            message["content"] for message in backend.messages_seen[2]
        )
        self.assertIn(
            SHELL_FULL_AGENT_MUTATION_VERIFICATION_PROMPT,
            verification_messages,
        )
        self.assertIn(
            "Create item.txt containing alpha and verify it.",
            verification_messages,
        )
        self.assertEqual(backend.messages_seen[2][0]["content"], AGENT_STRICT_TOOL_CALL_SYSTEM_PROMPT)

    def test_agent_mode_rechecks_empty_semantic_verification_with_observable_evidence(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "printf 'alpha\\n' > item.txt"})),
                _review("approve"),
                _result(json.dumps({"command": "cat item.txt"})),
                _result(json.dumps({"command": "true"})),
                _review("approve"),
                _result(json.dumps({"command": "test \"$(cat item.txt)\" = alpha && printf 'verified\\n'"})),
                _result("Created and verified item.txt."),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Create item.txt containing alpha and verify it.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_final_delta=lambda _text: None,
            )

        self.assertEqual(result.content, "Created and verified item.txt.")
        self.assertEqual(runtime.mutation_verifications, 2)
        self.assertEqual(runtime.mutation_verification_failures, 0)
        self.assertEqual(backend.calls, 7)

    def test_agent_mode_reuses_complete_semantic_verification_prose(self) -> None:
        final = "Created item.txt with alpha and verified its exact content."
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "printf 'alpha\\n' > item.txt"})),
                _review("approve"),
                _result(json.dumps({"command": "cat item.txt"})),
                _result(final),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Create item.txt containing alpha and verify it.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_final_delta=lambda _text: None,
            )

        self.assertEqual(result.content, final)
        self.assertEqual(runtime.post_tool_final_reuse_reused_count, 1)
        self.assertEqual(runtime.post_tool_final_reuse_avoided_model_calls, 1)
        self.assertEqual(backend.calls, 4)

    def test_agent_mode_bounds_repeated_ok_semantic_completion(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "printf 'alpha\\n' > item.txt"})),
                _review("approve"),
                _result(json.dumps({"command": "cat item.txt"})),
                _result("OK"),
                _result("OK"),
                _result("The change was made, but the requested verification could not be confirmed."),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Create item.txt containing alpha and verify it.",
                temperature=0,
                max_tokens=128,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_final_delta=lambda _text: None,
            )

        self.assertIn("could not be confirmed", result.content)
        self.assertEqual(runtime.agent_semantic_completion_retries, 1)
        self.assertEqual(runtime.mutation_semantic_repair_failures, 1)
        self.assertEqual(backend.calls, 6)
        final_messages = backend.messages_seen[-1]
        self.assertEqual(final_messages[0]["content"], FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn(
            AGENT_FINAL_COMPLETION_INSTRUCTION,
            "\n".join(message["content"] for message in final_messages if message["role"] == "system"),
        )

    def test_agent_mode_rejects_mutating_verification_before_execution(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "printf 'alpha\\n' > item.txt"})),
                _review("approve"),
                _result(json.dumps({"command": "printf 'forged\\n' > item.txt"})),
                _result(json.dumps({"command": "cat item.txt"})),
                _result("OK"),
                _result("Created and verified item.txt."),
            ]
        )
        tool_calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            runtime.ask_auto(
                "Create item.txt containing alpha and verify it.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_final_delta=lambda _text: None,
                on_tool_call=lambda _name, arguments: tool_calls.append(arguments),
            )
            content = (workdir / "item.txt").read_text(encoding="utf-8")

        self.assertEqual(content, "alpha\n")
        self.assertEqual(len(tool_calls), 2)
        self.assertNotIn("forged", "\n".join(tool_calls))
        self.assertEqual(runtime.agent_verification_retries, 1)

    def test_agent_mode_revises_mutation_without_executing_original_proposal(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "echo -e 'alpha\\nbeta' > output.txt"})),
                _review("revise", "echo -e is not portable POSIX sh"),
                _result(json.dumps({"command": "printf 'alpha\\nbeta\\n' > output.txt"})),
                _review("approve"),
                _result(json.dumps({"command": "cat output.txt"})),
                _result("Created output.txt with the exact two lines and verified it."),
            ]
        )
        tool_calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Create output.txt with exactly two lines, alpha and beta, verify it, then report.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_final_delta=lambda _text: None,
                on_tool_call=lambda _name, arguments: tool_calls.append(arguments),
            )

            self.assertEqual((workdir / "output.txt").read_text(encoding="utf-8"), "alpha\nbeta\n")

        self.assertEqual(result.content, "Created output.txt with the exact two lines and verified it.")
        self.assertEqual(runtime.agent_action_reviews, 2)
        self.assertEqual(runtime.agent_action_review_revisions, 1)
        self.assertEqual(runtime.agent_action_review_approvals, 1)
        self.assertNotIn("echo -e", "\n".join(tool_calls))
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(backend.calls, 6)
        self.assertIn(
            AGENT_ACTION_ANCHOR_TEMPLATE.format(
                user_prompt=(
                    "Create output.txt with exactly two lines, alpha and beta, verify it, then report."
                )
            ),
            [message["content"] for message in backend.messages_seen[2]],
        )

    def test_agent_mode_applies_model_generated_exact_patch_then_verifies(self) -> None:
        patch = (
            "--- mathbox.py\n"
            "+++ mathbox.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def average(values):\n"
            "-    return sum(values) // len(values)\n"
            "+    return sum(values) / len(values)\n"
        )
        backend = ScriptedBackend(
            [
                _result("", tool_calls=[_tool_call("exec_shell_full_command", {"command": "cat mathbox.py"}, call_id="read")]),
                _result("", tool_calls=[_tool_call("apply_patch", {"patch": patch}, call_id="patch")]),
                _review("approve", "the exact patch is within the requested source edit"),
                _result("", tool_calls=[_tool_call("exec_shell_full_command", {"command": "cat mathbox.py"}, call_id="verify")]),
                _result("Changed integer division to true division and verified mathbox.py."),
            ]
        )
        tool_names: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "mathbox.py"
            target.write_text(
                "def average(values):\n    return sum(values) // len(values)\n",
                encoding="utf-8",
            )
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Fix mathbox.py so average uses true division, then verify the file.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command", "apply_patch"),
                agent_mode=True,
                on_tool_call=lambda name, _arguments: tool_names.append(name),
            )

            content = target.read_text(encoding="utf-8")

        self.assertEqual(content, "def average(values):\n    return sum(values) / len(values)\n")
        self.assertEqual(result.content, "Changed integer division to true division and verified mathbox.py.")
        self.assertEqual(tool_names, ["exec_shell_full_command", "apply_patch", "exec_shell_full_command"])
        self.assertEqual(runtime.agent_action_reviews, 1)
        self.assertEqual(runtime.agent_action_review_approvals, 1)
        self.assertEqual(runtime.mutation_verifications, 1)
        self.assertEqual(backend.calls, 5)

    def test_agent_verification_rejects_a_second_patch_without_executing_it(self) -> None:
        patch = "--- item.txt\n+++ item.txt\n@@ -1 +1 @@\n-before\n+after\n"
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "printf 'before\\n' > item.txt", "after": "continue"})),
                _review("approve", "the requested file creation is authorized"),
                _result("", tool_calls=[_tool_call("apply_patch", {"patch": patch}, call_id="forged-verification")]),
                _result("", tool_calls=[_tool_call("exec_shell_full_command", {"command": "cat item.txt"}, call_id="verify")]),
                _result("Created item.txt and verified its content."),
            ]
        )
        tool_names: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Create item.txt containing before and verify it.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command", "apply_patch"),
                agent_mode=True,
                on_tool_call=lambda name, _arguments: tool_names.append(name),
            )
            content = (workdir / "item.txt").read_text(encoding="utf-8")

        self.assertEqual(content, "before\n")
        self.assertEqual(result.content, "Created item.txt and verified its content.")
        self.assertEqual(tool_names, ["exec_shell_full_command", "exec_shell_full_command"])
        self.assertEqual(runtime.agent_verification_retries, 1)
        self.assertEqual(backend.calls, 5)

    def test_repeated_mutation_is_rejected_before_a_second_action_review(self) -> None:
        mutation = "printf 'value\\n' > item.txt"
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": mutation, "after": "continue"})),
                _review("approve"),
                _result(json.dumps({"command": "cat item.txt"})),
                _result(json.dumps({"command": mutation})),
                _result("The file was created and verified; the repeated mutation was not run."),
            ]
        )
        tool_calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Create item.txt containing value, verify it, and do not repeat the write.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_tool_call=lambda _name, arguments: tool_calls.append(arguments),
            )

            self.assertEqual((workdir / "item.txt").read_text(encoding="utf-8"), "value\n")

        self.assertEqual(
            result.content,
            "The file was created and verified; the repeated mutation was not run.",
        )
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(runtime.agent_action_reviews, 1)
        self.assertEqual(runtime.agent_action_review_approvals, 1)
        self.assertEqual(backend.calls, 5)

    def test_agent_mode_declines_mutating_json_example_without_tool_execution(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "touch should-not-exist.txt"})),
                _review("decline", "the command is quoted example data"),
                _result("The JSON is an inert example and was not executed."),
            ]
        )
        tool_calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                'Do not use tools. Explain this inert example: {"command":"touch should-not-exist.txt"}',
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_tool_call=lambda name, _arguments: tool_calls.append(name),
            )

            self.assertFalse((workdir / "should-not-exist.txt").exists())

        self.assertEqual(result.content, "The JSON is an inert example and was not executed.")
        self.assertEqual(tool_calls, [])
        self.assertEqual(runtime.agent_action_review_declines, 1)
        self.assertEqual(backend.calls, 3)

    def test_agent_mode_reviews_unknown_shell_effects_for_mutative_request(self) -> None:
        candidate = (
            "python3 -c \"import pandas as pd; "
            "pd.DataFrame([{'value': 1}]).to_csv('summary.csv', index=False)\""
        )
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": candidate})),
                _review("decline", "the optional dependency was not observed as available"),
                _result("The proposed action was not executed."),
            ]
        )
        tool_calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            runtime.ask_auto(
                "Create summary.csv from the supplied data.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_tool_call=lambda name, _arguments: tool_calls.append(name),
            )

            self.assertFalse((workdir / "summary.csv").exists())

        self.assertEqual(tool_calls, [])
        self.assertEqual(runtime.agent_action_reviews, 1)
        self.assertEqual(runtime.agent_action_review_declines, 1)

    def test_agent_mode_cancelled_action_review_has_no_side_effect(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "touch should-not-exist.txt"})),
                _result("", finish_reason="cancelled"),
            ]
        )
        tool_calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Create should-not-exist.txt.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_tool_call=lambda name, _arguments: tool_calls.append(name),
            )

            self.assertFalse((workdir / "should-not-exist.txt").exists())

        self.assertEqual(result.finish_reason, "cancelled")
        self.assertEqual(tool_calls, [])
        self.assertEqual(backend.calls, 2)

    def test_agent_mode_does_not_continue_non_repairable_shell_failure(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "printf 'Permission denied\\n' >&2; exit 1"})),
                _result("The command failed with permission denied."),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Run the requested command and report its result.",
                temperature=0,
                max_tokens=128,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
            )

        self.assertEqual(result.content, "The command failed with permission denied.")
        self.assertEqual(runtime.agent_error_continuations, 0)
        self.assertEqual(backend.calls, 2)

    def test_agent_mode_bounds_repairable_shell_error_continuations(self) -> None:
        backend = ScriptedBackend(
            [
                _result(json.dumps({"command": "sh -c 'exit 7'"})),
                _result(json.dumps({"command": "sh -c 'exit 8'"})),
                _result(json.dumps({"command": "sh -c 'exit 9'"})),
                _result("The task stopped after the bounded recovery attempts failed."),
            ]
        )
        tool_calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            result = runtime.ask_auto(
                "Run the workflow and recover from ordinary command failures when possible.",
                temperature=0,
                max_tokens=128,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                agent_mode=True,
                on_tool_call=lambda _name, arguments: tool_calls.append(arguments),
            )

        self.assertEqual(result.content, "The task stopped after the bounded recovery attempts failed.")
        self.assertEqual(runtime.agent_error_continuations, 2)
        self.assertEqual(len(tool_calls), 3)
        self.assertEqual(backend.calls, 4)


if __name__ == "__main__":
    unittest.main()
