from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path
from shutil import copyfile
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult, Message
from orbit.runtime import ChatRuntime
from orbit.runtime.evidence import (
    EvidenceStore,
    build_final_evidence_context,
    build_route_evidence_context,
    build_evidence_record,
    tool_evidence_ref,
)
from orbit.runtime.messages import FINAL_FROM_TOOL_SYSTEM_PROMPT, MEDIA_SYSTEM_PROMPT, TOOL_CALL_JSON_RETRY_PROMPT, TOOL_CALL_SYSTEM_PROMPT
from orbit.runtime.chat import (
    _RouteDecisionStreamAbort,
    _RouteNotCommandAbort,
    _has_list_like_tool_result,
    _should_attempt_route_repair,
    _should_skip_route_repair,
)
from orbit.runtime.capabilities import LocalCapabilities, LocalCapability
from orbit.runtime.media import AudioInput, ImageInput
from orbit.runtime.tool_loop import _should_guard_existing_file_rewrite
from orbit.runtime.turn_trace import ModelPhaseStart
from orbit.runtime.shell_guardrails import (
    SHELL_FULL_COMPLETION_GUARD_PROMPT,
    SHELL_FULL_CONTENT_EVIDENCE_GUARD_PROMPT,
    SHELL_FULL_EMPTY_RESULT_CHECK_PROMPT,
    SHELL_FULL_MINIMAL_PATCH_GUARD_PROMPT,
    SHELL_FULL_READ_ONLY_MUTATION_ERROR_PREFIX,
    SHELL_FULL_READ_ONLY_MUTATION_RETRY_PROMPT,
    SHELL_FULL_SEMANTIC_REPAIR_PROMPT,
    should_verify_shell_mutation,
)


class FakeBackend:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.calls = 0
        self.tools = None

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.calls += 1
        self.messages = messages
        self.tools = tools
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


class SequenceBackend:
    def __init__(self, results: list[ChatResult]) -> None:
        self.results = list(results)
        self.messages_by_call: list[list[Message]] = []
        self.max_tokens_by_call: list[int] = []

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.messages_by_call.append(messages)
        self.max_tokens_by_call.append(max_tokens)
        if not self.results:
            raise AssertionError("unexpected model call")
        return self.results.pop(0)


class RuntimeTests(unittest.TestCase):
    def test_reset_clears_evidence_store_memory_without_deleting_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add("exec_shell_full_command", "raw content", metadata={"command": "pwd"})
            runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None, evidence_store=store)
            runtime.messages = [{"role": "user", "content": "run pwd"}]

            runtime.reset()

            self.assertEqual(runtime.messages, [])
            self.assertEqual(store.recent_records(1), [])
            self.assertIsNone(build_route_evidence_context(store))
            self.assertIsNone(build_final_evidence_context(store))
            self.assertTrue((store.root / f"{record.evidence_id}.txt").exists())

    def test_chat_final_retry_uses_evidence_card_without_raw_tool_bloat(self) -> None:
        raw = "web_search_results: true\nquery: OpenAI\nresults:\n" + ("raw-result " * 500)
        record = build_evidence_record(
            "exec_shell_full_command",
            raw,
            {"command": 'orbit-web-search "OpenAI"'},
        )
        backend = SequenceBackend(
            [
                ChatResult(
                    content="This is prose, not route JSON.",
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=128,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
                ChatResult(
                    content="Google designed me.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=4,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            store.records[record.evidence_id] = record
            store.raw_cache[record.evidence_id] = raw
            runtime = ChatRuntime(backend=backend, system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "user", "content": "search online for OpenAI"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "function": {"name": "exec_shell_full_command", "arguments": '{"command":"orbit-web-search \\"OpenAI\\""}'}}]},
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "exec_shell_full_command",
                    "content": tool_evidence_ref(record),
                    "evidence_id": record.evidence_id,
                },
                {"role": "assistant", "content": "OpenAI is an AI company."},
            ]

            result = runtime.ask_auto(
                "who designed you?",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
            )

        self.assertEqual(result.content, "Google designed me.")
        retry_messages = backend.messages_by_call[1]
        rendered = "\n".join(str(message.get("content", "")) for message in retry_messages)
        self.assertIn("tool_evidence_card: true", rendered)
        self.assertIn("raw_ref:", rendered)
        self.assertLess(len(rendered), len(raw))
        self.assertNotIn(raw, rendered)

    def test_route_prompt_includes_recent_evidence_context_without_raw(self) -> None:
        raw = "web_search_results: true\nquery: OpenAI\nresults:\n" + ("raw-result " * 500)
        backend = SequenceBackend(
            [
                ChatResult(
                    content='{"route":"CHAT"}',
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=1,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
                ChatResult(
                    content="The previous search results discussed OpenAI.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=7,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add("exec_shell_full_command", raw, metadata={"command": 'orbit-web-search "OpenAI"'})
            runtime = ChatRuntime(backend=backend, system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "user", "content": "search online for OpenAI"},
                {"role": "tool", "tool_call_id": "call-1", "name": "exec_shell_full_command", "content": tool_evidence_ref(record), "evidence_id": record.evidence_id},
                {"role": "assistant", "content": "OpenAI is an AI company. " * 200},
            ]

            result = runtime.ask_auto("what did the search results say?", temperature=0, max_tokens=32, workdir=Path(tmp))

        self.assertEqual(result.content, "The previous search results discussed OpenAI.")
        route_rendered = "\n".join(str(message.get("content", "")) for message in backend.messages_by_call[0])
        final_rendered = "\n".join(str(message.get("content", "")) for message in backend.messages_by_call[1])
        self.assertIn("available_evidence:", route_rendered)
        self.assertIn("tool_evidence_card=true", route_rendered)
        self.assertIn("what did the search results say?", route_rendered)
        self.assertNotIn("choose CHAT", route_rendered)
        self.assertNotIn(record.raw_ref, route_rendered)
        self.assertNotIn(raw, route_rendered)
        self.assertNotIn("tool_evidence_ref: true", route_rendered)
        self.assertNotIn("OpenAI is an AI company. OpenAI is an AI company.", route_rendered)
        self.assertIn("evidence_context:", final_rendered)
        self.assertIn("evidence_excerpt", final_rendered)
        self.assertNotIn("tool_evidence_ref: true", final_rendered)
        self.assertNotIn("OpenAI is an AI company. OpenAI is an AI company.", final_rendered)

    def test_post_tool_route_window_excludes_full_history_and_audit_marker(self) -> None:
        raw = "\n".join(f"line-{index}" for index in range(120))
        backend = SequenceBackend(
            [
                ChatResult(
                    content='{"route":"CHAT"}',
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=1,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
                ChatResult(
                    content="The output listed line-0 through line-119.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=8,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add(
                "exec_shell_full_command",
                raw,
                metadata={"command": "python3 -c 'for i in range(120): print(i)'"},
            )
            runtime = ChatRuntime(backend=backend, system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "user", "content": "run the line printer"},
                {"role": "assistant", "content": "previous final answer " * 200},
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "exec_shell_full_command",
                    "content": tool_evidence_ref(record) + "\nlegacy raw " + raw,
                    "evidence_id": record.evidence_id,
                },
            ]

            result = runtime.ask_auto("summarize that output", temperature=0, max_tokens=32, workdir=Path(tmp))

        self.assertEqual(result.content, "The output listed line-0 through line-119.")
        route_messages = backend.messages_by_call[0]
        route_rendered = "\n".join(str(message.get("content", "")) for message in route_messages)
        self.assertIn("summarize that output", route_rendered)
        self.assertIn("available_evidence:", route_rendered)
        self.assertIn("tool_evidence_card=true", route_rendered)
        self.assertIn("result_available=true", route_rendered)
        self.assertNotIn(record.raw_ref, route_rendered)
        self.assertNotIn("legacy raw", route_rendered)
        self.assertNotIn("tool_evidence_ref: true", route_rendered)
        self.assertNotIn("previous final answer previous final answer", route_rendered)
        self.assertFalse(any(message.get("role") == "tool" for message in route_messages))

    def test_post_tool_chat_final_uses_evidence_window_without_full_history(self) -> None:
        raw = "\n".join(f"line-{index}" for index in range(120))
        backend = SequenceBackend(
            [
                ChatResult(
                    content='{"route":"CHAT"}',
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=1,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
                ChatResult(
                    content="The output listed line-0 through line-119.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=8,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add(
                "exec_shell_full_command",
                raw,
                metadata={"command": "python3 -c 'for i in range(120): print(i)'"},
            )
            runtime = ChatRuntime(backend=backend, system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "user", "content": "run the line printer"},
                {"role": "assistant", "content": "previous final answer " * 200},
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "exec_shell_full_command",
                    "content": tool_evidence_ref(record) + "\nlegacy raw " + raw,
                    "evidence_id": record.evidence_id,
                },
            ]

            result = runtime.ask_auto("summarize that output", temperature=0, max_tokens=32, workdir=Path(tmp))

        self.assertEqual(result.content, "The output listed line-0 through line-119.")
        final_messages = backend.messages_by_call[1]
        final_rendered = "\n".join(str(message.get("content", "")) for message in final_messages)
        self.assertIn("summarize that output", final_rendered)
        self.assertIn("evidence_context:", final_rendered)
        self.assertIn(record.raw_ref, final_rendered)
        self.assertNotIn("legacy raw", final_rendered)
        self.assertNotIn("tool_evidence_ref: true", final_rendered)
        self.assertNotIn("previous final answer previous final answer", final_rendered)
        self.assertFalse(any(message.get("role") == "tool" for message in final_messages))

    def test_post_tool_chat_final_keeps_short_previous_assistant_for_reference(self) -> None:
        raw = "exit_code: 0\nSTDOUT:\n./src/orbit/runtime/evidence.py\nSTDERR:\n(empty)"
        backend = SequenceBackend(
            [
                ChatResult(
                    content='{"route":"CHAT"}',
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=1,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
                ChatResult(
                    content="It was found in src/orbit/runtime/evidence.py.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=9,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add("exec_shell_full_command", raw, metadata={"command": "grep -r EvidenceStore ."})
            runtime = ChatRuntime(backend=backend, system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "user", "content": "search files for EvidenceStore"},
                {"role": "assistant", "content": "EvidenceStore was found in src/orbit/runtime/evidence.py."},
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "exec_shell_full_command",
                    "content": tool_evidence_ref(record),
                    "evidence_id": record.evidence_id,
                },
            ]

            result = runtime.ask_auto("where was it found?", temperature=0, max_tokens=32, workdir=Path(tmp))

        self.assertEqual(result.content, "It was found in src/orbit/runtime/evidence.py.")
        final_rendered = "\n".join(str(message.get("content", "")) for message in backend.messages_by_call[1])
        self.assertIn("EvidenceStore was found in src/orbit/runtime/evidence.py.", final_rendered)
        self.assertIn("where was it found?", final_rendered)
        self.assertIn("evidence_context:", final_rendered)
        self.assertIn(record.raw_ref, final_rendered)
        self.assertNotIn("tool_evidence_ref: true", final_rendered)

    def test_route_without_evidence_keeps_existing_history_window(self) -> None:
        backend = SequenceBackend(
            [
                ChatResult(
                    content='{"route":"CHAT"}',
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=1,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
                ChatResult(
                    content="Final answer.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=2,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            runtime.messages = [{"role": "assistant", "content": "older assistant context"}]
            result = runtime.ask_auto("hello?", temperature=0, max_tokens=32, workdir=Path(tmp))

        self.assertEqual(result.content, "Final answer.")
        route_rendered = "\n".join(str(message.get("content", "")) for message in backend.messages_by_call[0])
        self.assertIn("older assistant context", route_rendered)

    def test_chat_final_retry_gets_recent_final_evidence_without_raw(self) -> None:
        raw = "exit_code: 0\nSTDOUT:\n/home/example/project\nSTDERR:\n(empty)"
        backend = SequenceBackend(
            [
                ChatResult(
                    content="Plain prose route drift.",
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=128,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
                ChatResult(
                    content="That was /home/example/project.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=6,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add("exec_shell_full_command", raw, metadata={"command": "pwd"})
            runtime = ChatRuntime(backend=backend, system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "function": {"name": "exec_shell_full_command", "arguments": '{"command":"pwd"}'}}]},
                {"role": "tool", "tool_call_id": "call-1", "name": "exec_shell_full_command", "content": tool_evidence_ref(record), "evidence_id": record.evidence_id},
                {"role": "assistant", "content": "The command printed /home/example/project."},
            ]

            result = runtime.ask_auto("what directory was that?", temperature=0, max_tokens=32, workdir=Path(tmp))

        self.assertEqual(result.content, "That was /home/example/project.")
        retry_rendered = "\n".join(str(message.get("content", "")) for message in backend.messages_by_call[1])
        self.assertIn("evidence_context:", retry_rendered)
        self.assertIn("/home/example/project", retry_rendered)
        self.assertIn(record.raw_ref, retry_rendered)
        self.assertEqual(backend.max_tokens_by_call, [32, 192])

    def test_chat_final_retry_without_evidence_keeps_requested_budget(self) -> None:
        backend = SequenceBackend(
            [
                ChatResult(
                    content="Plain prose route drift.",
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=128,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
                ChatResult(
                    content="A short direct answer.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=5,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto("hello?", temperature=0, max_tokens=32, workdir=Path(tmp))

        self.assertEqual(result.content, "A short direct answer.")
        self.assertEqual(backend.max_tokens_by_call, [32, 32])

    def test_chat_final_retry_window_is_compact_and_excludes_tool_history_raw(self) -> None:
        raw = "exit_code: 0\nSTDOUT:\n/home/example/project\nSTDERR:\n(empty)"
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add("exec_shell_full_command", raw, metadata={"command": "pwd"})
            runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "user", "content": "run pwd"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "function": {"name": "exec_shell_full_command", "arguments": '{"command":"pwd"}'}}]},
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "exec_shell_full_command",
                    "content": tool_evidence_ref(record) + "\n" + ("legacy raw " * 300),
                    "evidence_id": record.evidence_id,
                },
                {"role": "assistant", "content": "The command printed /home/example/project."},
                {"role": "user", "content": "what directory was that?"},
            ]

            messages = runtime._chat_final_retry_messages()

        rendered = "\n".join(str(message.get("content", "")) for message in messages)
        self.assertIn("what directory was that?", rendered)
        self.assertIn("evidence_context:", rendered)
        self.assertIn("/home/example/project", rendered)
        self.assertIn(record.raw_ref, rendered)
        self.assertNotIn("legacy raw", rendered)
        self.assertFalse(any(message.get("role") == "tool" for message in messages))

    def test_chat_final_retry_medium_evidence_uses_compact_context(self) -> None:
        raw = "\n".join(f"line-{index}" for index in range(120))
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add(
                "exec_shell_full_command",
                raw,
                metadata={"command": "python3 -c 'for i in range(120): print(i)'"},
            )
            runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "user", "content": "run the line printer"},
                {"role": "assistant", "content": "previous final answer " * 200},
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "exec_shell_full_command",
                    "content": tool_evidence_ref(record) + "\nlegacy raw " + raw,
                    "evidence_id": record.evidence_id,
                },
                {"role": "user", "content": "summarize that output"},
            ]

            messages = runtime._chat_final_retry_messages()

        rendered = "\n".join(str(message.get("content", "")) for message in messages)
        self.assertIn("summarize that output", rendered)
        self.assertIn("evidence_context:", rendered)
        self.assertIn(record.raw_ref, rendered)
        self.assertIn(record.raw_sha256[:16], rendered)
        self.assertIn("line-0", rendered)
        self.assertIn("line-119", rendered)
        self.assertNotIn("legacy raw", rendered)
        self.assertNotIn("previous final answer previous final answer", rendered)
        self.assertFalse(any(message.get("role") == "tool" for message in messages))

    def test_final_from_tool_with_evidence_uses_compact_window(self) -> None:
        raw = "\n".join(f"line-{index}" for index in range(120))
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add(
                "exec_shell_full_command",
                raw,
                metadata={"command": "python3 -c 'for i in range(120): print(i)'"},
            )
            backend = FakeBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "user", "content": "run the line printer"},
                {"role": "assistant", "content": "previous answer " * 100},
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "exec_shell_full_command",
                    "content": tool_evidence_ref(record) + "\nlegacy raw " + raw,
                    "evidence_id": record.evidence_id,
                },
            ]

            runtime._answer_from_tool_results(
                temperature=0,
                max_tokens=64,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                loop=1,
                use_tool_prompt=True,
            )

        rendered = "\n".join(str(message.get("content", "")) for message in backend.messages)
        self.assertIn("run the line printer", rendered)
        self.assertIn("evidence_context:", rendered)
        self.assertIn(record.raw_ref, rendered)
        self.assertIn(record.raw_sha256[:16], rendered)
        self.assertIn("line-0", rendered)
        self.assertIn("line-119", rendered)
        self.assertNotIn("legacy raw", rendered)
        self.assertNotIn("previous answer previous answer", rendered)
        self.assertFalse(any(message.get("role") == "tool" for message in backend.messages))

    def test_final_from_tool_small_evidence_view_is_opt_in_without_tool_prompt(self) -> None:
        raw = "/home/example/project\n"
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add("exec_shell_full_command", raw, metadata={"command": "pwd"})
            backend = FakeBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "user", "content": "run pwd"},
                {"role": "assistant", "content": "previous answer " * 100},
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "exec_shell_full_command",
                    "content": tool_evidence_ref(record) + "\nlegacy raw " + raw,
                    "evidence_id": record.evidence_id,
                },
            ]

            runtime._answer_from_tool_results(
                temperature=0,
                max_tokens=64,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                loop=1,
                use_tool_prompt=False,
            )

        rendered = "\n".join(str(message.get("content", "")) for message in backend.messages)
        self.assertIn("run pwd", rendered)
        self.assertIn("evidence_context:", rendered)
        self.assertIn("/home/example/project", rendered)
        self.assertIn(record.raw_ref, rendered)
        self.assertNotIn("legacy raw", rendered)
        self.assertNotIn("previous answer previous answer", rendered)
        self.assertFalse(any(message.get("role") == "tool" for message in backend.messages))

    def test_final_from_tool_small_evidence_keeps_legacy_with_tool_prompt(self) -> None:
        raw = "/home/example/project\n"
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add("exec_shell_full_command", raw, metadata={"command": "pwd"})
            backend = FakeBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "user", "content": "run pwd"},
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "exec_shell_full_command",
                    "content": tool_evidence_ref(record),
                    "evidence_id": record.evidence_id,
                },
            ]

            runtime._answer_from_tool_results(
                temperature=0,
                max_tokens=64,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                loop=1,
                use_tool_prompt=True,
            )

        self.assertTrue(any(message.get("role") == "tool" for message in backend.messages))
        rendered = "\n".join(str(message.get("content", "")) for message in backend.messages)
        self.assertIn("evidence_context:", rendered)

    def test_final_from_tool_small_evidence_excludes_web_and_medium(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_store = EvidenceStore(Path(tmp) / "web.evidence")
            web_record = web_store.add(
                "exec_shell_full_command",
                "web_search_results: true\nquery: OpenAI\nresults:\n1. title: OpenAI",
                metadata={"command": 'orbit-web-search "OpenAI"'},
            )
            runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None, evidence_store=web_store)
            runtime.messages = [{"role": "user", "content": "search"}, {"role": "tool", "content": tool_evidence_ref(web_record)}]
            self.assertFalse(runtime._should_use_final_small_evidence_view(use_tool_prompt=False))

            medium_store = EvidenceStore(Path(tmp) / "medium.evidence")
            medium_record = medium_store.add(
                "exec_shell_full_command",
                "x" * 600,
                metadata={"command": "cat medium.txt"},
            )
            runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None, evidence_store=medium_store)
            runtime.messages = [{"role": "user", "content": "read"}, {"role": "tool", "content": tool_evidence_ref(medium_record)}]
            self.assertFalse(runtime._should_use_final_small_evidence_view(use_tool_prompt=False))

    def test_web_final_from_tool_uses_compact_web_view_without_full_history(self) -> None:
        class MaxTokenBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.max_tokens_seen: list[int] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.max_tokens_seen.append(max_tokens)
                return super().chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)

        raw = "\n".join(
            [
                "web_search_results: true",
                "query: OpenAI",
                "results:",
                "1. title: OpenAI",
                "   url: https://openai.com/",
                "   snippet: AI research and deployment.",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add(
                "exec_shell_full_command",
                raw,
                metadata={"command": 'orbit-web-search "OpenAI"'},
            )
            backend = MaxTokenBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "user", "content": "search online for OpenAI"},
                {"role": "assistant", "content": "previous answer " * 100},
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "exec_shell_full_command",
                    "content": tool_evidence_ref(record) + "\nlegacy raw " + raw,
                    "evidence_id": record.evidence_id,
                },
            ]

            runtime._answer_from_tool_results(
                temperature=0,
                max_tokens=128,
                on_final_delta=None,
                on_progress=None,
                on_model_step=None,
                loop=1,
                use_tool_prompt=False,
            )

        rendered = "\n".join(str(message.get("content", "")) for message in backend.messages)
        self.assertTrue(runtime._should_use_web_final_view(use_tool_prompt=False))
        self.assertFalse(runtime._should_use_web_final_view(use_tool_prompt=True))
        self.assertEqual(backend.max_tokens_seen, [192])
        self.assertIn("search online for OpenAI", rendered)
        self.assertIn("evidence_context:", rendered)
        self.assertIn("web_search_evidence: true", rendered)
        self.assertIn("AI research and deployment.", rendered)
        self.assertIn(record.raw_ref, rendered)
        self.assertIn(record.raw_sha256[:16], rendered)
        self.assertNotIn("legacy raw", rendered)
        self.assertNotIn("previous answer previous answer", rendered)
        self.assertFalse(any(message.get("role") == "tool" for message in backend.messages))

    def test_final_from_tool_without_evidence_store_keeps_legacy_window(self) -> None:
        backend = FakeBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)
        runtime.messages = [
            {"role": "user", "content": "run pwd"},
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "exec_shell_full_command",
                "content": "exit_code: 0\nSTDOUT:\n/home/example/project",
            },
        ]

        runtime._answer_from_tool_results(
            temperature=0,
            max_tokens=64,
            on_final_delta=None,
            on_progress=None,
            on_model_step=None,
            loop=1,
            use_tool_prompt=True,
        )

        self.assertTrue(any(message.get("role") == "tool" for message in backend.messages))

    def test_chat_final_completion_repair_uses_compact_search_evidence(self) -> None:
        raw = "\n".join(
            [
                "./src/orbit/runtime/evidence.py:42:class EvidenceStore:",
                "./src/orbit/runtime/tool_loop.py:17:from orbit.runtime.evidence import EvidenceStore",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add(
                "exec_shell_full_command",
                raw,
                metadata={"command": 'grep -R "EvidenceStore" .'},
            )
            runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None, evidence_store=store)
            runtime.messages = [
                {"role": "user", "content": "search files for EvidenceStore"},
                {
                    "role": "assistant",
                    "content": (
                        "The term EvidenceStore was found in ./src/orbit/runtime/evidence.py "
                        "and ./src/orbit/runtime/tool_loop.py. "
                        + ("extra detail " * 80)
                    ),
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "exec_shell_full_command",
                    "content": tool_evidence_ref(record) + "\nlegacy raw " + raw,
                    "evidence_id": record.evidence_id,
                },
                {"role": "user", "content": "where was it found?"},
            ]

            messages = runtime.chat_final_completion_repair_messages("Answer directly now.")

        assert messages is not None
        rendered = "\n".join(str(message.get("content", "")) for message in messages)
        self.assertIn("where was it found?", rendered)
        self.assertIn("evidence_context:", rendered)
        self.assertIn("./src/orbit/runtime/evidence.py", rendered)
        self.assertIn("EvidenceStore", rendered)
        self.assertIn(record.raw_ref, rendered)
        self.assertIn(record.raw_sha256[:16], rendered)
        self.assertIn("Answer directly now.", rendered)
        self.assertIn("The term EvidenceStore was found", rendered)
        self.assertNotIn("legacy raw", rendered)
        self.assertNotIn("extra detail extra detail extra detail extra detail", rendered)
        self.assertFalse(any(message.get("role") == "tool" for message in messages))

    def test_near_empty_route_length_skips_route_repair(self) -> None:
        backend = SequenceBackend(
            [
                ChatResult(
                    content="...",
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=1,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
                ChatResult(
                    content="That was /home/example/project.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=6,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto(
                "what directory was that?",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "That was /home/example/project.")
        self.assertEqual(len(backend.messages_by_call), 2)
        retry_rendered = "\n".join(str(message.get("content", "")) for message in backend.messages_by_call[1])
        self.assertIn("what directory was that?", retry_rendered)

    def test_partial_route_artifact_still_allows_route_repair(self) -> None:
        backend = SequenceBackend(
            [
                ChatResult(
                    content='{"route"',
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=2,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
                ChatResult(
                    content='{"route":"CHAT"}',
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=1,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
                ChatResult(
                    content="Final answer.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=2,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto(
                "what directory was that?",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "Final answer.")
        self.assertEqual(len(backend.messages_by_call), 3)

    def test_route_repair_detection_is_structural(self) -> None:
        self.assertFalse(
            _should_attempt_route_repair(
                ChatResult(
                    content="...",
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=1,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )
            )
        )
        self.assertTrue(
            _should_attempt_route_repair(
                ChatResult(
                    content='{"command"',
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=1,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )
            )
        )

    def test_route_repair_skips_empty_length_output_without_artifact(self) -> None:
        self.assertTrue(
            _should_skip_route_repair(
                ChatResult(
                    content="",
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=128,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )
            )
        )
        self.assertFalse(
            _should_skip_route_repair(
                ChatResult(
                    content='{"command"',
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=128,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )
            )
        )

    def test_exec_shell_wc_result_is_not_list_like(self) -> None:
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": {"command": "wc -l text/summary.txt"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "exec_shell_full_command",
                "content": "2 text/summary.txt",
            },
        ]

        self.assertFalse(_has_list_like_tool_result(messages))

    def test_exec_shell_ls_result_is_list_like(self) -> None:
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "name": "exec_shell_full_command",
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": {"command": "ls -F"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "exec_shell_full_command",
                "content": "README.md\nsrc/",
            },
        ]

        self.assertTrue(_has_list_like_tool_result(messages))

    def test_ask_with_image_builds_openai_multimodal_content(self) -> None:
        backend = FakeBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)
        image = ImageInput(path=Path("/tmp/test.png"), mime_type="image/png", data_url="data:image/png;base64,abc")

        runtime.ask("describe", temperature=0, max_tokens=32, images=[image])

        content = backend.messages[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "describe"})
        self.assertEqual(content[1], {"type": "image_url", "image_url": {"url": image.data_url}})

    def test_ask_with_audio_builds_openai_multimodal_content(self) -> None:
        backend = FakeBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)
        audio = AudioInput(path=Path("/tmp/test.wav"), format="wav", data="abc")

        runtime.ask("transcribe", temperature=0, max_tokens=32, audios=[audio])

        content = backend.messages[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "transcribe"})
        self.assertEqual(content[1], {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}})

    def test_restore_message_count_discards_partial_turn(self) -> None:
        runtime = ChatRuntime(backend=FakeBackend(), system_prompt="system")
        checkpoint = len(runtime.messages)
        runtime.messages.append({"role": "user", "content": "partial"})
        runtime.last_memory_refresh = None

        runtime.restore_message_count(checkpoint)

        self.assertEqual(len(runtime.messages), checkpoint)
        self.assertEqual(runtime.messages[-1]["content"], "system")

    def test_ask_emits_model_step_metrics(self) -> None:
        steps = []
        backend = FakeBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        runtime.ask("hello", temperature=0, max_tokens=32, on_model_step=steps.append)

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].loop, 1)
        self.assertEqual(steps[0].phase, "chat_final")
        self.assertEqual(steps[0].prompt_tokens, 1)
        self.assertEqual(steps[0].cached_tokens, 0)

    def test_capability_discovery_does_not_invoke_model_at_startup(self) -> None:
        backend = FakeBackend()

        ChatRuntime(backend=backend, system_prompt=None)

        self.assertEqual(backend.calls, 0)

    def test_continue_last_response_uses_chat_without_tools(self) -> None:
        backend = FakeBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)
        runtime.messages.append({"role": "assistant", "content": "partial answer"})

        result = runtime.continue_last_response(temperature=0, max_tokens=32)

        self.assertEqual(result.content, "ok")
        self.assertIsNone(getattr(backend, "tools", None))
        self.assertEqual(runtime.messages[-2]["role"], "user")
        self.assertIn("Continue exactly", runtime.messages[-2]["content"])
        self.assertEqual(runtime.messages[-1]["role"], "assistant")

    def test_continue_last_response_uses_open_reasoning_prompt_when_needed(self) -> None:
        backend = FakeBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)
        runtime.messages.append({"role": "assistant", "content": "<|channel>thought\npartial reasoning"})

        runtime.continue_last_response(temperature=0, max_tokens=32)

        self.assertEqual(runtime.messages[-2]["role"], "user")
        self.assertIn("stop reasoning now", runtime.messages[-2]["content"].lower())

    def test_local_capabilities_are_added_to_tool_prompts(self) -> None:
        backend = CaptureToolPromptBackend()
        runtime = ChatRuntime(
            backend=backend,
            system_prompt=None,
            local_capabilities=LocalCapabilities(
                (
                    LocalCapability("pdftotext", True, "/usr/bin/pdftotext", "document", "extract text from PDFs"),
                    LocalCapability("pandoc", False, None, "document", "convert supported document formats to text"),
                    LocalCapability("python3", True, "/usr/bin/python3", "python", "run Python helpers"),
                )
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            runtime.ask_with_tools(
                "read pdf/piccolo.pdf and summarize it",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
                max_loops=1,
                tool_names=("exec_shell_full_command",),
            )

        rendered = "\n".join(str(message.get("content", "")) for message in backend.calls[0])
        self.assertIn("Local tools available:", rendered)
        self.assertIn("pdftotext", rendered)
        self.assertIn("Unavailable: pandoc", rendered)

    def test_local_capabilities_are_added_to_normal_tool_prompts(self) -> None:
        backend = CaptureToolPromptBackend()
        runtime = ChatRuntime(
            backend=backend,
            system_prompt=None,
            local_capabilities=LocalCapabilities(
                (LocalCapability("pdftotext", True, "/usr/bin/pdftotext", "document", "extract text from PDFs"),)
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            runtime.ask_with_tools(
                "list files",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
                max_loops=1,
                tool_names=("exec_shell_full_command",),
            )

        rendered = "\n".join(str(message.get("content", "")) for message in backend.calls[0])
        self.assertIn("Local tools available:", rendered)

    def test_local_capabilities_are_not_added_to_tools_off_chat(self) -> None:
        backend = FakeBackend()
        runtime = ChatRuntime(
            backend=backend,
            system_prompt=None,
            local_capabilities=LocalCapabilities(
                (LocalCapability("pdftotext", True, "/usr/bin/pdftotext", "document", "extract text from PDFs"),)
            ),
        )

        runtime.ask("hello", temperature=0, max_tokens=32)

        rendered = "\n".join(str(message.get("content", "")) for message in backend.messages)
        self.assertNotIn("Local tools available:", rendered)

    def test_continue_last_response_uses_prompt_fallback_for_open_reasoning(self) -> None:
        class FallbackBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.continue_calls = 0
                self.chat_calls = 0
                self.thinking = True
                self.last_messages: list[Message] = []

            def continue_current(self, *, max_tokens: int, on_delta=None, on_progress=None) -> ChatResult:
                self.continue_calls += 1
                return ChatResult(
                    content="should not be used",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=0,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                self.last_thinking = self.thinking
                self.last_messages = messages
                return ChatResult(
                    content="Final answer from fallback.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=2,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = FallbackBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)
        runtime.messages.append({"role": "assistant", "content": "<|channel>thought\npartial reasoning"})

        result = runtime.continue_last_response(temperature=0, max_tokens=32)

        self.assertEqual(result.content, "Final answer from fallback.")
        self.assertEqual(backend.continue_calls, 0)
        self.assertEqual(backend.chat_calls, 1)
        self.assertEqual(runtime.messages[-1]["content"], "Final answer from fallback.")
        self.assertFalse(backend.last_thinking)
        self.assertIn("Stop reasoning now and write only the missing final answer.", backend.last_messages[-2]["content"])
        self.assertIn("Start the answer with 'Final answer:'", backend.last_messages[-2]["content"])

    def test_continue_last_response_returns_controlled_error_if_fallback_stays_thinking_like(self) -> None:
        class ThinkingFallbackBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.continue_calls = 0
                self.chat_calls = 0
                self.thinking = True

            def continue_current(self, *, max_tokens: int, on_delta=None, on_progress=None) -> ChatResult:
                self.continue_calls += 1
                raise AssertionError("native continue should not be used for thinking continuation")

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                return ChatResult(
                    content="<|channel>thought\nstill thinking",
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=2,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = ThinkingFallbackBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)
        runtime.messages.append({"role": "assistant", "content": "<|channel>thought\npartial reasoning"})

        result = runtime.continue_last_response(temperature=0, max_tokens=32)

        self.assertEqual(result.content, "error: model did not produce a final answer after continuation")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(backend.continue_calls, 0)
        self.assertEqual(backend.chat_calls, 1)
        self.assertFalse(runtime.can_continue_last_response())

    def test_ask_chat_retries_reasoning_only_stop_with_final_only_retry(self) -> None:
        class ReasoningThenFinalBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.chat_calls = 0
                self.thinking = True
                self.thinking_seen: list[bool] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                self.thinking_seen.append(self.thinking)
                if self.chat_calls == 1:
                    return ChatResult(
                        content="<|channel>thought\nprivate chain<channel|>",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=2,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Final answer: Dante Alighieri was an Italian poet.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=2,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = ReasoningThenFinalBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)

        result = runtime.ask_chat("Who is Dante Alighieri?", temperature=0, max_tokens=32)

        self.assertEqual(result.content, "Final answer: Dante Alighieri was an Italian poet.")
        self.assertEqual(backend.chat_calls, 2)
        self.assertEqual(backend.thinking_seen, [True, False])

    def test_ask_chat_retries_reasoning_only_length_with_final_only_retry(self) -> None:
        class ReasoningThenFinalBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.chat_calls = 0
                self.thinking = True
                self.thinking_seen: list[bool] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                self.thinking_seen.append(self.thinking)
                if self.chat_calls == 1:
                    return ChatResult(
                        content="<|channel>thought\nprivate chain<channel|>",
                        model="fake",
                        finish_reason="length",
                        tool_calls=[],
                        prompt_tokens=2,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Final answer: Dante Alighieri was an Italian poet.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=2,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = ReasoningThenFinalBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)

        result = runtime.ask_chat("Who is Dante Alighieri?", temperature=0, max_tokens=32)

        self.assertEqual(result.content, "Final answer: Dante Alighieri was an Italian poet.")
        self.assertEqual(backend.chat_calls, 2)
        self.assertEqual(backend.thinking_seen, [True, False])

    def test_ask_chat_rejects_reasoning_leakage_even_when_finish_reason_is_stop(self) -> None:
        class ReasoningLeakThenFinalBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.chat_calls = 0
                self.thinking = True
                self.thinking_seen: list[bool] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                self.thinking_seen.append(self.thinking)
                if self.chat_calls == 1:
                    return ChatResult(
                        content=(
                            '"What is the main difference between essay and wise?"\n'
                            'The user likely meant "essay" and "wise".\n'
                            "* **Possibility A:** compare essay and thesis.\n"
                            "* **Possibility B:** compare essay and wise.\n"
                            "An essay is a written composition, while wise means having good judgment."
                        ),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=2,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="An essay is a written composition, while wise means having good judgment.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=2,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = ReasoningLeakThenFinalBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)

        result = runtime.ask_chat("what is the main difference beetwen essay and wise?", temperature=0, max_tokens=64)

        self.assertEqual(
            result.content,
            "An essay is a written composition, while wise means having good judgment.",
        )
        self.assertEqual(backend.chat_calls, 2)
        self.assertEqual(backend.thinking_seen, [True, False])

    def test_ask_chat_returns_controlled_error_if_final_only_retry_still_leaks_reasoning(self) -> None:
        class ReasoningLeakTwiceBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.chat_calls = 0
                self.thinking = True

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                if self.chat_calls == 1:
                    return ChatResult(
                        content=(
                            "The user likely meant two different words.\n"
                            "* **Possibility A:** first interpretation.\n"
                            "* **Possibility B:** second interpretation.\n"
                            "Here is a partial answer."
                        ),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=2,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content=(
                        "The user likely meant two different words.\n"
                        "* **Possibility A:** first interpretation.\n"
                        "* **Possibility B:** second interpretation.\n"
                        "Here is another partial answer."
                    ),
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=2,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = ReasoningLeakTwiceBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)

        result = runtime.ask_chat("what is the main difference beetwen essay and wise?", temperature=0, max_tokens=64)

        self.assertEqual(result.content, "error: model did not produce a clean final answer")
        self.assertEqual(result.finish_reason, "stop")

    def test_ask_chat_think_on_streams_separate_thinking_then_final_answer_pass(self) -> None:
        class StreamingThinkingThenFinalBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.chat_calls = 0
                self.thinking = True
                self.thinking_seen: list[bool] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                raise AssertionError("streaming path expected")

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
                self.chat_calls += 1
                self.thinking_seen.append(self.thinking)
                assert on_delta is not None
                if self.chat_calls == 1:
                    on_delta("The user likely means \"essay\" and \"wise\".\n")
                    on_delta("* **Possibility A:** compare essay and thesis.\n")
                    on_delta("* **Possibility B:** compare essay and wise.\n")
                    return ChatResult(
                        content=(
                            "The user likely means \"essay\" and \"wise\".\n"
                            "* **Possibility A:** compare essay and thesis.\n"
                            "* **Possibility B:** compare essay and wise.\n"
                        ),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=2,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                on_delta("Final answer: An essay is a piece of writing, while wise describes good judgment.")
                return ChatResult(
                    content="Final answer: An essay is a piece of writing, while wise describes good judgment.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=2,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = StreamingThinkingThenFinalBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)
        emitted: list[str] = []

        result = runtime.ask_chat(
            "what is the main difference beetwen essay and wise?",
            temperature=0,
            max_tokens=64,
            on_final_delta=emitted.append,
        )

        self.assertEqual(
            result.content,
            "Final answer: An essay is a piece of writing, while wise describes good judgment.",
        )
        self.assertEqual(
            emitted,
            [
                "The user likely means \"essay\" and \"wise\".\n",
                "* **Possibility A:** compare essay and thesis.\n",
                "* **Possibility B:** compare essay and wise.\n",
                "Final answer: An essay is a piece of writing, while wise describes good judgment.",
            ],
        )
        self.assertEqual(backend.chat_calls, 2)
        self.assertEqual(backend.thinking_seen, [True, False])

    def test_ask_chat_emits_distinct_phase_reasons_for_reasoning_repair(self) -> None:
        class ReasoningThenFinalBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.chat_calls = 0
                self.thinking = True

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                if self.chat_calls == 1:
                    return ChatResult(
                        content="<|channel>thought\nprivate chain<channel|>",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=2,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.chat_calls == 2:
                    return ChatResult(
                        content=(
                            "The user likely meant two different words.\n"
                            "* **Possibility A:** first interpretation.\n"
                            "* **Possibility B:** second interpretation.\n"
                            "Here is a partial answer."
                        ),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=2,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Final answer: Dante Alighieri was an Italian poet.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=2,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = ReasoningThenFinalBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)
        phases: list[ModelPhaseStart] = []

        runtime.ask_chat(
            "Who is Dante Alighieri?",
            temperature=0,
            max_tokens=32,
            on_phase_start=phases.append,
        )

        self.assertEqual(
            [(phase.phase, phase.attempt, phase.reason) for phase in phases],
            [
                ("tool_plan", 1, "chat_thinking"),
                ("chat_final", 1, None),
                ("chat_final_completion_repair", 2, "reasoning_like"),
            ],
        )

    def test_chat_final_does_not_repair_plain_stop_answer(self) -> None:
        backend = SequenceBackend(
            [
                ChatResult(
                    content="The output contains sequential lines from line-0 through line-119",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=10,
                    completion_tokens=12,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )
            ]
        )
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        result = runtime._transport_environment().chat_final(
            [{"role": "user", "content": "summarize that output"}],
            temperature=0,
            max_tokens=64,
            on_final_delta=None,
            on_progress=None,
            on_model_step=None,
            on_phase_start=None,
            loop=1,
        )

        self.assertEqual(result.content, "The output contains sequential lines from line-0 through line-119")
        self.assertEqual(len(backend.messages_by_call), 1)

    def test_ask_chat_does_not_retry_complete_markdown_answer_with_trailing_bold_closer(self) -> None:
        class MarkdownCompleteBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.chat_calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                raise AssertionError("streaming path expected")

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
                self.chat_calls += 1
                assert on_delta is not None
                content = (
                    "I am a large language model, trained by Google. I can help answer questions and explain topics.\n\n"
                    "**What's on your mind today? Is there anything specific I can help you with?**"
                )
                on_delta(content)
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=2,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = MarkdownCompleteBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=False)
        emitted: list[str] = []
        phases: list[ModelPhaseStart] = []

        result = runtime.ask_chat(
            "hi, tell me something about yourself",
            temperature=0,
            max_tokens=512,
            on_final_delta=emitted.append,
            on_phase_start=phases.append,
        )

        self.assertEqual(backend.chat_calls, 1)
        self.assertEqual("".join(emitted), result.content)
        self.assertEqual(
            [(phase.phase, phase.attempt, phase.reason) for phase in phases],
            [("chat_final", 1, None)],
        )

    def test_continue_last_response_uses_prompt_fallback_for_bullet_reasoning_length(self) -> None:
        class BulletReasoningBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.continue_calls = 0
                self.chat_calls = 0
                self.thinking = True
                self.last_messages: list[Message] = []

            def continue_current(self, *, max_tokens: int, on_delta=None, on_progress=None) -> ChatResult:
                self.continue_calls += 1
                raise AssertionError("native continue should not be used for truncated reasoning prelude")

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                self.last_thinking = self.thinking
                self.last_messages = messages
                return ChatResult(
                    content="Final answer: I understand and will answer concisely.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=2,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = BulletReasoningBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)
        runtime.messages.append(
            {
                "role": "assistant",
                "content": "* Constraint 1: explain the plan.\n* Constraint 2: give the final answer.\n* Drafting the response:",
            }
        )
        runtime.client_state.last_content = runtime.messages[-1]["content"]
        runtime.last_visible_finish_reason = "length"

        result = runtime.continue_last_response(temperature=0, max_tokens=32)

        self.assertEqual(result.content, "Final answer: I understand and will answer concisely.")
        self.assertEqual(backend.continue_calls, 0)
        self.assertEqual(backend.chat_calls, 1)
        self.assertFalse(backend.last_thinking)
        self.assertIn("Stop reasoning now and write only the missing final answer.", backend.last_messages[-2]["content"])
        self.assertIn("Write exactly one short final-answer sentence.", backend.last_messages[-2]["content"])

    def test_continue_last_response_returns_controlled_error_if_forced_final_is_incomplete(self) -> None:
        class IncompleteForcedFinalBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.thinking = True

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                return ChatResult(
                    content="Final answer: I will now provide a concise final",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=2,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = IncompleteForcedFinalBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)
        runtime.messages.append({"role": "assistant", "content": "### Reasoning\npartial"})

        result = runtime.continue_last_response(temperature=0, max_tokens=32)

        self.assertEqual(result.content, "error: model did not produce a final answer after continuation")
        self.assertFalse(runtime.can_continue_last_response())

    def test_continue_last_response_uses_native_backend_continuation_after_length_even_without_open_reasoning(self) -> None:
        class NativeContinueBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.continue_calls = 0

            def continue_current(self, *, max_tokens: int, on_delta=None, on_progress=None) -> ChatResult:
                self.continue_calls += 1
                return ChatResult(
                    content="continued final answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=0,
                    completion_tokens=3,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = NativeContinueBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)
        runtime.messages.append({"role": "assistant", "content": "<|channel>thought\n<channel|>partial final"})
        runtime.last_visible_finish_reason = "length"

        result = runtime.continue_last_response(temperature=0, max_tokens=32)

        self.assertEqual(result.content, "continued final answer")
        self.assertEqual(backend.continue_calls, 1)
        self.assertEqual(runtime.messages[-1]["content"], "continued final answer")

    def test_ask_chat_uses_chat_prompt_without_tools(self) -> None:
        class ChatBackend:
            def __init__(self) -> None:
                self.messages: list[Message] = []
                self.tools_seen = object()

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.messages = messages
                self.tools_seen = tools
                return ChatResult(
                    content="chat",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=1,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = ChatBackend()
        runtime = ChatRuntime(backend=backend, system_prompt="route system")

        result = runtime.ask_chat("hello", temperature=0, max_tokens=32)

        self.assertEqual(result.content, "chat")
        self.assertIsNone(backend.tools_seen)
        self.assertIn("Answer normally", backend.messages[0]["content"])
        self.assertNotIn('{"command"', backend.messages[0]["content"])


class ToolCallingBackend:
    def __init__(self, tool_name: str = "exec_shell_full_command", arguments: str = "{\"command\":\"ls -F\"}") -> None:
        self.messages: list[Message] = []
        self.calls = 0
        self.tool_name = tool_name
        self.arguments = arguments

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.calls += 1
        self.messages = messages
        if self.calls == 1:
            return ChatResult(
                content="",
                model="fake",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": self.tool_name, "arguments": self.arguments},
                    }
                ],
                prompt_tokens=10,
                completion_tokens=2,
                cached_tokens=8,
                prompt_tokens_per_second=100.0,
                generation_tokens_per_second=10.0,
            )
        return ChatResult(
            content="done",
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=20,
            completion_tokens=3,
            cached_tokens=10,
            prompt_tokens_per_second=80.0,
            generation_tokens_per_second=9.0,
        )


class CaptureToolPromptBackend:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.calls.append(messages)
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


def _last_tool_message(runtime: ChatRuntime) -> Message:
    tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
    assert tool_messages
    return tool_messages[-1]


class ToolRuntimeTests(unittest.TestCase):
    def test_ask_with_tools_disables_backend_thinking_for_tool_plan_and_final_answer(self) -> None:
        class ThinkingAwareBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.thinking = True
                self.thinking_seen: list[bool] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.thinking_seen.append(self.thinking)
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"cat note.txt\"}"},
                            }
                        ],
                        prompt_tokens=None,
                        completion_tokens=None,
                        cached_tokens=None,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="<|channel>thought\nfrom tool result<channel|>done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=None,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello from file", encoding="utf-8")
            backend = ThinkingAwareBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools(
                "read note",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
            )

        self.assertEqual(result.finish_reason, "stop")
        self.assertGreaterEqual(len(backend.thinking_seen), 2)
        self.assertTrue(all(flag is False for flag in backend.thinking_seen))
        self.assertTrue(backend.thinking)

    def test_ask_auto_temporarily_disables_backend_thinking_for_route_phase(self) -> None:
        class ThinkingAwareRouteBackend:
            def __init__(self) -> None:
                self.thinking = True
                self.thinking_seen: list[bool] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.thinking_seen.append(self.thinking)
                return ChatResult(
                    content="plain answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=5,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = ThinkingAwareRouteBackend()
        runtime = ChatRuntime(backend=backend, system_prompt="route system", thinking_mode=True)

        result = runtime.ask_auto(
            "explain the plan before the final answer",
            temperature=0,
            max_tokens=128,
            workdir=Path("."),
            allowed_tool_names=("exec_shell_full_command",),
        )

        self.assertEqual(result.content, "plain answer")
        self.assertEqual(backend.thinking_seen, [False])
        self.assertTrue(backend.thinking)

    def test_ask_auto_respects_allowed_tool_subset(self) -> None:
        class FilesystemRouteBackend:
            def __init__(self) -> None:
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.tools_seen.append(tools)
                return ChatResult(
                    content='{"command":"cat README.md"}',
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=3,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = FilesystemRouteBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        result = runtime.ask_auto(
            "read README.md",
            temperature=0,
            max_tokens=32,
            workdir=Path("."),
            allowed_tool_names=("search_web", "fetch_url"),
        )

        self.assertEqual(result.finish_reason, "unsupported_command")
        self.assertIn("no suitable tool", result.content)
        self.assertEqual(backend.tools_seen, [None])

    def test_ask_auto_does_not_downgrade_file_edit_to_read_only_tool(self) -> None:
        class FileEditRouteBackend:
            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                return ChatResult(
                    content='{"command":"printf hello > note.txt"}',
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=3,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        runtime = ChatRuntime(backend=FileEditRouteBackend(), system_prompt=None)

        result = runtime.ask_auto(
            "create note.txt",
            temperature=0,
            max_tokens=32,
            workdir=Path("."),
            allowed_tool_names=("list_files", "read_file", "stat_path", "file_glob_search", "grep_search"),
        )

        self.assertEqual(result.finish_reason, "unsupported_command")
        self.assertIn("no suitable tool", result.content)

    def test_ask_auto_allows_file_edit_command_when_shell_full_is_enabled(self) -> None:
        class FileEditShellFullBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tools_seen.append(tools)
                if self.calls == 1:
                    content = '{"command":"mkdir -p lab && rmdir lab"}'
                else:
                    content = "use shell-full"
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=3,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = FileEditShellFullBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        result = runtime.ask_auto(
            "create then remove lab directory",
            temperature=0,
            max_tokens=32,
            workdir=Path("."),
            allowed_tool_names=("exec_shell_full_command",),
        )

        self.assertEqual(result.content, "use shell-full")
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(tool_messages[-1]["name"], "exec_shell_full_command")

    def test_ask_auto_allows_web_command_when_shell_full_is_enabled(self) -> None:
        class WebShellFullBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tools_seen.append(tools)
                if self.calls == 1:
                    content = '{"command":"curl https://example.com"}'
                else:
                    content = "use shell-full"
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=3,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = WebShellFullBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        result = runtime.ask_auto(
            "read https://example.com",
            temperature=0,
            max_tokens=32,
            workdir=Path("."),
            allowed_tool_names=("exec_shell_full_command",),
        )

        self.assertEqual(result.content, "use shell-full")
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(tool_messages[-1]["name"], "exec_shell_full_command")

    def test_ask_auto_uses_shell_command_for_datetime_when_tools_are_enabled(self) -> None:
        class TimeRouteBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tools_seen = None

            def server_tools(self):
                return [
                    {
                        "tool": "get_datetime",
                        "definition": {
                            "type": "function",
                            "function": {
                                "name": "get_datetime",
                                "description": "Get current local date/time.",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        },
                    }
                ]

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content='{"command":"printf ok"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=3,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                self.tools_seen = tools
                return ChatResult(
                    content="time answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=4,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = TimeRouteBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        result = runtime.ask_auto(
            "what time is it?",
            temperature=0,
            max_tokens=32,
            workdir=Path("."),
            allowed_tool_names=("exec_shell_full_command",),
        )

        self.assertEqual(result.content, "time answer")
        self.assertIsNone(backend.tools_seen)

    def test_ask_auto_returns_chat_without_tools(self) -> None:
        class ChatOnlyBackend:
            def __init__(self) -> None:
                self.tools_seen: list[object] = []
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.tools_seen.append(tools)
                self.messages_seen.append(messages)
                content = 'chat answer' if len(self.messages_seen) == 1 else "chat answer"
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=3,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = ChatOnlyBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        result = runtime.ask_auto("hello", temperature=0, max_tokens=32, workdir=Path("."))

        self.assertEqual(result.content, "chat answer")
        self.assertEqual(backend.tools_seen, [None])
        self.assertIn('{"command":"..."}', backend.messages_seen[0][0]["content"])
        self.assertEqual(runtime.messages[-1]["content"], "chat answer")

    def test_ask_auto_emits_short_probe_chat_response(self) -> None:
        class StreamingChatBackend:
            def __init__(self) -> None:
                self.max_tokens_seen: list[int] = []
                self.tools = object()

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.max_tokens_seen.append(max_tokens)
                self.tools = tools
                return ChatResult(
                    content="chat answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=3,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                raise AssertionError("short probe answer should not need streaming retry")

        emitted: list[str] = []
        backend = StreamingChatBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        result = runtime.ask_auto("hello", temperature=0, max_tokens=512, workdir=Path("."), on_final_delta=emitted.append)

        self.assertEqual(result.content, "chat answer")
        self.assertEqual(emitted, ["chat answer"])
        self.assertEqual(backend.max_tokens_seen, [64])
        self.assertIsNone(backend.tools)

    def test_ask_auto_streams_full_answer_after_truncated_probe(self) -> None:
        class StreamingRetryBackend:
            def __init__(self) -> None:
                self.chat_max_tokens: list[int] = []
                self.stream_max_tokens: list[int] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_max_tokens.append(max_tokens)
                return ChatResult(
                    content="partial",
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=3,
                    completion_tokens=64,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                assert on_delta is not None
                self.stream_max_tokens.append(max_tokens)
                on_delta("full ")
                on_delta("answer")
                return ChatResult(
                    content="full answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=3,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        emitted: list[str] = []
        backend = StreamingRetryBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        result = runtime.ask_auto("write a longer answer", temperature=0, max_tokens=512, workdir=Path("."), on_final_delta=emitted.append)

        self.assertEqual(result.content, "full answer")
        self.assertEqual(emitted, ["full ", "answer"])
        self.assertEqual(backend.chat_max_tokens, [64])
        self.assertEqual(backend.stream_max_tokens, [512])

    def test_ask_auto_retries_empty_chat_response_once(self) -> None:
        class EmptyThenAnswerBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls in {1, 2}:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=3,
                        completion_tokens=0,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="retry answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=3,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = EmptyThenAnswerBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        result = runtime.ask_auto("hello", temperature=0, max_tokens=64, workdir=Path("."))

        self.assertEqual(result.content, "retry answer")
        self.assertEqual(backend.calls, 3)

    def test_ask_auto_double_empty_returns_clear_error(self) -> None:
        class AlwaysEmptyBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                content = ""
                completion_tokens = 0
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=3,
                    completion_tokens=completion_tokens,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = AlwaysEmptyBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        result = runtime.ask_auto("hello", temperature=0, max_tokens=64, workdir=Path("."))

        self.assertEqual(result.finish_reason, "empty_response")
        self.assertIn("empty response twice", result.content)
        self.assertEqual(backend.calls, 3)

    def test_ask_auto_empty_tool_selection_falls_back_to_final_model_answer(self) -> None:
        class EmptyToolSelectionBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tools_seen.append(tools)
                if self.calls == 1:
                    content = '{"command":"printf ok"}'
                    completion_tokens = 7
                else:
                    content = "fallback final answer"
                    completion_tokens = 3
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=3,
                    completion_tokens=completion_tokens,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = EmptyToolSelectionBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        result = runtime.ask_auto("read a large file if present", temperature=0, max_tokens=64, workdir=Path("."))

        self.assertEqual(result.content, "fallback final answer")
        self.assertEqual(backend.calls, 2)
        self.assertIsNone(backend.tools_seen[0])
        self.assertIsNone(backend.tools_seen[1])

    def test_ask_auto_converts_generic_command_tool_call_inside_tool_loop(self) -> None:
        class GenericRouteToolCallBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content='{"command":"printf ok"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=3,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[
                            {
                                "id": "raw-tool-call-1",
                                "type": "function",
                                "function": {
                                    "name": "call",
                                    "arguments": '{"command":"ls -F"}',
                                },
                            }
                        ],
                        prompt_tokens=3,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="listed files",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=3,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("x", encoding="utf-8")
            backend = GenericRouteToolCallBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_auto("list files if needed", temperature=0, max_tokens=64, workdir=workdir)

        self.assertEqual(result.content, "listed files")
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["name"], "exec_shell_full_command")

    def test_ask_auto_commands_to_filesystem_tools(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tool_names_seen: list[tuple[str, ...]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tool_names_seen.append(tuple(tool["function"]["name"] for tool in tools or []))
                if self.calls == 1:
                    return ChatResult(
                        content='{"command":"printf ok"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "list_files", "arguments": "{\"path\":\".\"}"},
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("x", encoding="utf-8")
            backend = RoutedBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto("show files", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.tool_names_seen[0], ())
        self.assertEqual(backend.tool_names_seen[1], ())
        self.assertNotIn('{"command":"printf ok"}', [message.get("content") for message in runtime.messages])

    def test_ask_auto_allows_shell_full_after_generic_filesystem_command_when_enabled(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tool_names_seen: list[tuple[str, ...]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tool_names_seen.append(tuple(tool["function"]["name"] for tool in tools or []))
                if self.calls == 1:
                    return ChatResult(
                        content='{"command":"printf ok"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"printf ok\"}"},
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = RoutedBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "try to analyze it with apktool",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.tool_names_seen[0], ())
        self.assertEqual(backend.tool_names_seen[1], ())

    def test_ask_auto_allows_shell_full_when_safe_shell_command_is_not_enabled(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tool_names_seen: list[tuple[str, ...]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tool_names_seen.append(tuple(tool["function"]["name"] for tool in tools or []))
                if self.calls == 1:
                    return ChatResult(
                        content='{"command":"printf ok"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"printf ok\"}"},
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = RoutedBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "tell me the specs of this computer",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command", "list_files", "read_file"),
            )

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.tool_names_seen[0], ())
        self.assertEqual(backend.tool_names_seen[1], ())

    def test_ask_auto_routes_parenthesized_raw_shell_tool_call_to_tool_loop(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content='<|tool_call>call(shell, "orbit-web-search \\"Mario Nobile\\"")<tool_call|>',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=22,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": "{\"command\":\"orbit-web-search 'Mario Nobile'\"}",
                                },
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Mario Nobile is an Italian public official.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=9,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = RoutedBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "search online for information about Mario Nobile",
                temperature=0,
                max_tokens=64,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "Mario Nobile is an Italian public official.")
        self.assertEqual(backend.calls, 3)

    def test_ask_with_tools_normalizes_orbit_web_search_underscore_tool_call(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "orbit_web_search",
                                    "arguments": '{"query":"sample topic"}',
                                },
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="sample result",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = RoutedBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")
            with patch("orbit.runtime.shell_guardrails.search_web", return_value="web_search_results: true\nquery: sample topic"):
                result = runtime.ask_with_tools(
                    "search online for sample topic",
                    temperature=0,
                    max_tokens=64,
                    workdir=Path(tmp),
                )

        self.assertEqual(result.content, "sample result")
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["name"], "exec_shell_full_command")
        self.assertIn("web_search_results: true", str(tool_messages[0]["content"]))
        self.assertNotIn("tool not available", str(tool_messages[0]["content"]))

    def test_empty_web_search_route_uses_brief_final_prompt(self) -> None:
        class EmptyWebSearchBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.max_tokens_seen: list[int] = []
                self.final_messages: list[Message] | None = None

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.max_tokens_seen.append(max_tokens)
                if self.calls == 1:
                    return ChatResult(
                        content='{"command":"orbit-web-search \\"sample topic\\""}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=10,
                        completion_tokens=8,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                self.final_messages = messages
                return ChatResult(
                    content="No web search results were found for the query.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=38,
                    completion_tokens=10,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = EmptyWebSearchBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            with patch(
                "orbit.runtime.shell_guardrails.search_web",
                return_value="web_search_results: true\nresults: none",
            ):
                result = runtime.ask_auto(
                    "search online for sample topic",
                    temperature=0,
                    max_tokens=512,
                    workdir=Path(tmp),
                    allowed_tool_names=("exec_shell_full_command",),
                )

        self.assertEqual(result.content, "No web search results were found for the query.")
        self.assertEqual(backend.calls, 2)
        self.assertEqual(backend.max_tokens_seen, [64, 192])
        self.assertIsNotNone(backend.final_messages)
        self.assertEqual(backend.final_messages[0]["content"], FINAL_FROM_TOOL_SYSTEM_PROMPT)
        rendered = "\n".join(str(message.get("content", "")) for message in backend.final_messages or [])
        self.assertIn("web_search_results: true", rendered)
        self.assertIn("results: none", rendered)
        self.assertNotIn("Decide compactly whether the user request needs local tools", rendered)

    def test_new_web_turn_uses_latest_tool_evidence_not_previous_local_result(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.final_messages: list[Message] | None = None

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-web",
                                "type": "function",
                                "function": {
                                    "name": "orbit_web_search",
                                    "arguments": '{"query":"sample topic"}',
                                },
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                self.final_messages = messages
                return ChatResult(
                    content="sample web answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=3,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = RoutedBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")
            runtime.messages.extend(
                [
                    {"role": "user", "content": "read local fixture"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-old",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": json.dumps({"command": "cat fixture.txt"}),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-old",
                        "name": "exec_shell_full_command",
                        "content": "old_local_fixture_evidence: true\ncontent:\nold local content",
                    },
                    {"role": "assistant", "content": "old local answer"},
                ]
            )
            with patch("orbit.runtime.shell_guardrails.search_web", return_value="web_search_results: true\nquery: sample topic\nresults:\n- current web evidence"):
                result = runtime.ask_with_tools(
                    "search online for sample topic",
                    temperature=0,
                    max_tokens=64,
                    workdir=Path(tmp),
                )

        self.assertEqual(result.content, "sample web answer")
        self.assertIsNotNone(backend.final_messages)
        final_rendered = "\n".join(str(message.get("content", "")) for message in backend.final_messages or [])
        self.assertIn("current web evidence", final_rendered)
        self.assertNotIn("old_local_fixture_evidence", final_rendered)

    def test_ask_auto_retries_route_as_tool_call_after_length_without_decision(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.retry_messages: list[Message] | None = None

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="This answer wanders and never reaches a valid routing decision",
                        model="fake",
                        finish_reason="length",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=96,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    self.retry_messages = messages
                    return ChatResult(
                        content='{"command":"cat note.txt"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="hello from note",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=3,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello from note", encoding="utf-8")
            backend = RoutedBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "read note.txt",
                temperature=0,
                max_tokens=96,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "hello from note")
        self.assertEqual(backend.calls, 3)
        self.assertIsNotNone(backend.retry_messages)
        self.assertEqual(backend.retry_messages[0]["content"], TOOL_CALL_SYSTEM_PROMPT)

    def test_ask_auto_retries_explicit_web_search_prompt_when_route_answers_in_prose(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.retry_messages: list[Message] | None = None

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="I could not find anything useful from memory.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=12,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    self.retry_messages = messages
                    return ChatResult(
                        content="I still cannot route this correctly.",
                        model="fake",
                        finish_reason="length",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=96,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"printf 'Mario Nobile is a public official.'\"}"},
                            }
                        ],
                        prompt_tokens=7,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Mario Nobile is a public official.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=6,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = RoutedBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "search online for information about Mario Nobile",
                temperature=0,
                max_tokens=96,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "Mario Nobile is a public official.")
        self.assertEqual(backend.calls, 5)
        self.assertIsNotNone(backend.retry_messages)
        self.assertEqual(backend.retry_messages[0]["content"], TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("latest user request only", backend.retry_messages[-1]["content"])
        self.assertIn("ignore older tool results", backend.retry_messages[-1]["content"])

    def test_shell_full_analysis_metadata_only_command_gets_one_model_retry(self) -> None:
        class ShellFullAnalysisBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tools_seen: list[object] = []
                self.second_call_last_message: Message | None = None
                self.third_call_last_message: Message | None = None

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tools_seen.append(tools)
                if self.calls == 1:
                    return ChatResult(
                        content='{"command":"ls -R samples/"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    self.second_call_last_message = messages[-1]
                    return ChatResult(
                        content="I do not have access to the file content.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=8,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    self.third_call_last_message = messages[-1]
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": "{\"command\":\"sed -n '1,80p' samples/vulnerable_service.py\"}",
                                },
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="vulnerability found from source evidence",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            samples = workdir / "samples"
            samples.mkdir()
            (samples / "vulnerable_service.py").write_text("subprocess.run(cmd, shell=True)\n", encoding="utf-8")
            backend = ShellFullAnalysisBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "analyze the samples/vulnerable_service.py file and report vulnerabilities",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "vulnerability found from source evidence")
        self.assertEqual(backend.calls, 4)
        self.assertEqual(backend.tools_seen[0], None)
        self.assertIsNotNone(backend.tools_seen[1])
        self.assertIsNotNone(backend.tools_seen[2])
        self.assertEqual(backend.tools_seen[3], None)
        self.assertIsNotNone(backend.second_call_last_message)
        self.assertIn("Requested file: samples/vulnerable_service.py", backend.second_call_last_message["content"])
        self.assertIn("No usable content has been extracted from the requested file yet.", backend.second_call_last_message["content"])
        self.assertIsNotNone(backend.third_call_last_message)
        self.assertIn("Return only the tool call", backend.third_call_last_message["content"])

    def test_analysis_completion_guard_reconsiders_non_content_followup_after_evidence(self) -> None:
        class AnalysisCompletionBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.guard_messages: list[Message] | None = None
                self.final_messages: list[Message] | None = None

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "sed -n '1,80p' vulnerable_service.py"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": json.dumps({"command": "ls -R"}),
                                },
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    self.guard_messages = messages
                    return ChatResult(
                        content="I already have enough evidence.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=4,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                self.final_messages = messages
                return ChatResult(
                    content="The file is vulnerable because it executes shell commands with shell=True.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=8,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "vulnerable_service.py").write_text("subprocess.run(cmd, shell=True)\n", encoding="utf-8")
            backend = AnalysisCompletionBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "inspect vulnerable_service.py and explain the vulnerabilities",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertTrue(result.content.strip())
        self.assertIn(backend.calls, {3, 4})
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("shell=True", tool_messages[0]["content"])

    def test_analysis_completion_guard_does_not_block_followup_content_read(self) -> None:
        class FollowupContentBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "sed -n '1,40p' report.txt"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": json.dumps({"command": "sed -n '41,80p' report.txt"}),
                                },
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Final summary from both chunks.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=4,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "report.txt").write_text("A\n" * 120, encoding="utf-8")
            backend = FollowupContentBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "inspect report.txt and summarize it",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "Final summary from both chunks.")

    def test_file_recovery_guard_guides_model_to_candidate_read(self) -> None:
        class FileRecoveryBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.guard_messages: list[Message] | None = None
                self.guard_max_tokens: int | None = None

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                last_content = str(messages[-1].get("content"))
                if "Requested file not read yet." in last_content and self.guard_messages is None:
                    self.guard_messages = messages
                    self.guard_max_tokens = max_tokens
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "cat vulnerable_service.py"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content=json.dumps({"command": "find . -name \"vulnerable_service.py\""}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    return ChatResult(
                        content=json.dumps({"command": "ls -R"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 4:
                    return ChatResult(
                        content=json.dumps({"command": "sed -n '1,80p' ./samples/vulnerable_service.py"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=8,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="The file is vulnerable because it uses shell=True in subprocess.run.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=9,
                    completion_tokens=6,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            samples = workdir / "samples"
            samples.mkdir()
            (samples / "vulnerable_service.py").write_text("subprocess.run(cmd, shell=True)\n", encoding="utf-8")
            backend = FileRecoveryBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "inspect vulnerable_service.py and explain the vulnerabilities",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertIn("shell=True", result.content)
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 3)
        self.assertIn("No such file or directory", tool_messages[0]["content"])
        self.assertIn("./samples/vulnerable_service.py", tool_messages[1]["content"])
        self.assertIn("shell=True", tool_messages[2]["content"])
        self.assertIsNotNone(backend.guard_messages)
        self.assertEqual(backend.guard_max_tokens, 64)
        self.assertIn("Requested file not read yet.", backend.guard_messages[-1]["content"])
        self.assertIn("Requested file: vulnerable_service.py", backend.guard_messages[-1]["content"])
        self.assertIn("Direct read failure:", backend.guard_messages[-1]["content"])
        self.assertIn("./samples/vulnerable_service.py", backend.guard_messages[-1]["content"])

    def test_pdf_recovery_guard_prefers_robust_text_extraction_without_auto_running_it(self) -> None:
        class PdfRecoveryBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.guard_messages: list[Message] | None = None
                self.generic_shell_repairs = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                last_content = str(messages[-1].get("content"))
                if "Requested file not read yet." in last_content and self.guard_messages is None:
                    self.guard_messages = messages
                if "The previous shell command failed." in last_content:
                    self.generic_shell_repairs += 1
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "pdffind pdf/grande.pdf"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content=json.dumps({"command": "pdftotext pdf/grande.pdf - | head -n 20"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="The document is a presentation about eIDAS regulation and identity proofing.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=10,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "pdf").mkdir()
            copyfile(ROOT / "workdir" / "pdf" / "grande.pdf", workdir / "pdf" / "grande.pdf")
            backend = PdfRecoveryBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "Read pdf/grande.pdf and summarize the document topic in one concise sentence.",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertIn("identity proofing", result.content.lower())
        self.assertIsNotNone(backend.guard_messages)
        self.assertEqual(backend.generic_shell_repairs, 0)
        guard = str(backend.guard_messages[-1]["content"])
        self.assertIn("Requested file: pdf/grande.pdf", guard)
        self.assertIn("Requested file currently exists in the workdir: yes", guard)
        self.assertIn("Last failed command: pdffind pdf/grande.pdf", guard)
        self.assertIn("Last exit code:", guard)
        self.assertIn("preferably pdftotext", guard)
        self.assertIn("fallback such as strings", guard)
        self.assertIn("Do not conclude that the file is missing or unreadable yet.", guard)

    def test_file_recovery_guard_allows_final_not_found_answer(self) -> None:
        class MissingFileBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.guard_messages: list[Message] | None = None

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                last_content = str(messages[-1].get("content"))
                if "Requested file not read yet." in last_content and self.guard_messages is None:
                    self.guard_messages = messages
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "cat missing.py"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content=json.dumps({"command": "find . -name \"missing.py\""}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    return ChatResult(
                        content=json.dumps({"command": "ls -R"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="I could not find `missing.py` after a direct read and a targeted search.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=8,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            backend = MissingFileBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "inspect missing.py and explain the vulnerabilities",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertIn("could not find", result.content.lower())
        self.assertIsNotNone(backend.guard_messages)
        self.assertIn("Requested file: missing.py", backend.guard_messages[-1]["content"])

    def test_file_recovery_guard_does_not_block_recursive_listing_requests(self) -> None:
        class RecursiveListingBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "ls -R"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Directory listing complete.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=6,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            backend = RecursiveListingBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "list files recursively",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "Directory listing complete.")
        self.assertEqual(backend.calls, 2)
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)

    def test_direct_content_read_handoffs_immediately_to_final_from_tool(self) -> None:
        class DirectContentHandoffBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.final_messages: list[Message] | None = None

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": json.dumps({"command": "sed -n '1,80p' vulnerable_service.py"}),
                                },
                            }
                        ],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                self.final_messages = messages
                assert tools is None
                assert "shell-full output" in str(messages[-1].get("content"))
                return ChatResult(
                    content="The file is vulnerable because it uses shell=True in subprocess.run.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=6,
                    completion_tokens=8,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "vulnerable_service.py").write_text("subprocess.run(cmd, shell=True)\n", encoding="utf-8")
            backend = DirectContentHandoffBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_with_tools(
                "inspect vulnerable_service.py and explain the vulnerabilities",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                tool_names=("exec_shell_full_command",),
            )

        self.assertIn("shell=True", result.content)
        self.assertEqual(backend.calls, 2)
        self.assertIsNotNone(backend.final_messages)

    def test_candidate_paths_without_direct_content_do_not_handoff_to_final_from_tool(self) -> None:
        class CandidatePathBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.third_call_messages: list[Message] | None = None

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": json.dumps({"command": "cat vulnerable_service.py"}),
                                },
                            }
                        ],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": json.dumps({"command": "find . -name \"vulnerable_service.py\""}),
                                },
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    self.third_call_messages = messages
                    assert tools is not None
                    assert "shell-full output" not in str(messages[-1].get("content"))
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-3",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": json.dumps({"command": "sed -n '1,80p' ./samples/vulnerable_service.py"}),
                                },
                            }
                        ],
                        prompt_tokens=7,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                assert tools is None
                assert "shell-full output" in str(messages[-1].get("content"))
                return ChatResult(
                    content="The file is vulnerable because it uses shell=True in subprocess.run.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=8,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            samples = workdir / "samples"
            samples.mkdir()
            (samples / "vulnerable_service.py").write_text("subprocess.run(cmd, shell=True)\n", encoding="utf-8")
            backend = CandidatePathBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_with_tools(
                "inspect vulnerable_service.py and explain the vulnerabilities",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                tool_names=("exec_shell_full_command",),
            )

        self.assertIn("shell=True", result.content)
        self.assertEqual(backend.calls, 4)
        self.assertIsNotNone(backend.third_call_messages)

    def test_mutative_request_does_not_handoff_directly_to_final_from_tool(self) -> None:
        class MutativeNoHandoffBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.second_call_messages: list[Message] | None = None

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": json.dumps({"command": "printf 'updated\\n' > note.txt && cat note.txt"}),
                                },
                            }
                        ],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    self.second_call_messages = messages
                    assert tools is not None
                    assert "shell-full output" not in str(messages[-1].get("content"))
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": json.dumps({"command": "cat note.txt"}),
                                },
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            backend = MutativeNoHandoffBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_with_tools(
                "update note.txt with the word updated",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.calls, 4)
        self.assertIsNotNone(backend.second_call_messages)

    def test_shell_full_failed_initial_command_finalizes_from_error_evidence(self) -> None:
        class FailedShellCommandBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                self.tools_seen.append(tools)
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "command_that_does_not_exist_123"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    assert tools is None
                    rendered_messages = str(messages)
                    assert "evidence_context:" in rendered_messages
                    assert "tool_evidence_card: true" in rendered_messages
                    assert "shell_command_failed" in rendered_messages
                    assert "The previous shell command failed." not in rendered_messages
                    return ChatResult(
                        content="reported command failure",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=3,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="unexpected extra call",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            backend = FailedShellCommandBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "run a missing command",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "reported command failure")
        self.assertEqual(backend.calls, 2)
        self.assertIsNone(backend.tools_seen[0])
        self.assertIsNone(backend.tools_seen[1])
        self.assertEqual(len([message for message in runtime.messages if message.get("role") == "tool"]), 1)

    def test_shell_full_non_repairable_error_does_not_retry(self) -> None:
        class NonRepairableBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tools_seen.append(tools)
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "printf 'permission denied' >&2; exit 1"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="reported failure",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = NonRepairableBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "run a command",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "reported failure")
        self.assertEqual(backend.calls, 2)
        self.assertIsNone(backend.tools_seen[0])
        self.assertIsNone(backend.tools_seen[1])

    def test_shell_full_generic_nonzero_error_finalizes_without_repair_retry(self) -> None:
        class GenericErrorBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                self.tools_seen.append(tools)
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "printf 'custom parser exploded at token 7' >&2; exit 2"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    assert tools is None
                    rendered_messages = str(messages)
                    assert "evidence_context:" in rendered_messages
                    assert "tool_evidence_card: true" in rendered_messages
                    assert "custom parser exploded at token 7" in rendered_messages
                    assert "The previous shell command failed." not in rendered_messages
                    return ChatResult(
                        content="reported parser failure",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=3,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="unexpected extra call",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = GenericErrorBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "run custom command",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "reported parser failure")
        self.assertEqual(backend.calls, 2)
        self.assertIsNone(backend.tools_seen[0])
        self.assertIsNone(backend.tools_seen[1])

    def test_shell_full_repair_loop_stops_after_two_failed_retries(self) -> None:
        class AlwaysFailingBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tools_seen.append(tools)
                if self.calls <= 3:
                    commands = {
                        1: "grep '[' note.txt",
                        2: "grep '[[' note.txt",
                        3: "grep '[a-' note.txt",
                    }
                    return ChatResult(
                        content=json.dumps({"command": commands[self.calls]}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="final failure report",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("alpha\n", encoding="utf-8")
            backend = AlwaysFailingBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "search note.txt",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "final failure report")
        self.assertEqual(backend.calls, 4)
        self.assertIsNone(backend.tools_seen[0])
        self.assertIsNotNone(backend.tools_seen[1])
        self.assertIsNotNone(backend.tools_seen[2])
        self.assertIsNone(backend.tools_seen[3])

    def test_shell_full_empty_initial_result_gets_one_model_verification_chance(self) -> None:
        class EmptyShellResultBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                self.tools_seen.append(tools)
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "printf beta > note.txt"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": json.dumps({"command": "cat note.txt"}),
                                },
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    return ChatResult(
                        content="no further tool call",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="note updated",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            backend = EmptyShellResultBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "write beta into note.txt",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

            content = (workdir / "note.txt").read_text(encoding="utf-8")

        self.assertEqual(result.content, "note updated")
        self.assertEqual(content, "beta")
        self.assertEqual(backend.calls, 4)
        self.assertIsNone(backend.tools_seen[0])
        self.assertIsNotNone(backend.tools_seen[1])
        self.assertIsNotNone(backend.tools_seen[2])
        self.assertIsNone(backend.tools_seen[3])
        self.assertEqual(backend.messages_seen[1][-1]["content"], SHELL_FULL_EMPTY_RESULT_CHECK_PROMPT)
        self.assertEqual(runtime.mutation_verifications, 1)
        self.assertEqual(runtime.mutation_verification_repairs, 0)
        self.assertEqual(runtime.mutation_verification_failures, 0)

    def test_mutation_verification_sed_noop_requests_direct_evidence(self) -> None:
        class SedNoopBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "sed -i 's|<title>.*</title>|<title>test</title>|' index.html"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content=json.dumps({"command": "perl -0ne 'print $1 if m|<title>\\s*(.*?)\\s*</title>|s' index.html"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    return ChatResult(
                        content="verification inspected",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="not confirmed",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "index.html").write_text("<title>\nOld\n</title>\n", encoding="utf-8")
            runtime = ChatRuntime(backend=SedNoopBackend(), system_prompt="route system")

            result = runtime.ask_auto(
                'sostituisci il title di index.html con "test"',
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "not confirmed")
        self.assertEqual(runtime.mutation_verifications, 1)
        self.assertEqual(runtime.mutation_verification_failures, 0)
        self.assertIn("requested value or state", SHELL_FULL_EMPTY_RESULT_CHECK_PROMPT)
        self.assertIn("not only metadata, paths, tags, field names, or key names", SHELL_FULL_EMPTY_RESULT_CHECK_PROMPT)

    def test_mutation_verification_semantic_repair_completes_partial_change(self) -> None:
        class SemanticRepairBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    content = json.dumps({"command": "python3 -c 'from pathlib import Path; p=Path(\"backup.sh\"); p.write_text(p.read_text().replace(\"cp $1 $2\", \"cp \\\"$1\\\" \\\"$2\\\"\"))'"})
                elif self.calls == 2:
                    content = json.dumps({"command": "cat backup.sh"})
                elif self.calls == 3:
                    content = json.dumps({"command": "python3 -c 'from pathlib import Path; p=Path(\"backup.sh\"); s=p.read_text(); p.write_text(s.replace(\"#!/usr/bin/env bash\\n\", \"#!/usr/bin/env bash\\nset -euo pipefail\\n\")); print(p.read_text())'"})
                elif self.calls == 4:
                    return ChatResult(
                        content="script hardened",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=9,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                else:
                    return ChatResult(
                        content="done",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=10,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=5,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "backup.sh").write_text("#!/usr/bin/env bash\ncp $1 $2\n", encoding="utf-8")
            runtime = ChatRuntime(backend=SemanticRepairBackend(), system_prompt="route system")
            result = runtime.ask_auto(
                "Harden backup.sh for bash safety and correct quoting of arguments.",
                temperature=0,
                max_tokens=96,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )
            content = (workdir / "backup.sh").read_text(encoding="utf-8")

        self.assertEqual(result.content, "done")
        self.assertIn("set -euo pipefail", content)
        self.assertIn('cp "$1" "$2"', content)
        self.assertEqual(runtime.mutation_verifications, 1)
        self.assertEqual(runtime.mutation_semantic_repairs, 1)
        self.assertEqual(runtime.mutation_semantic_repair_commands, 1)
        self.assertEqual(runtime.mutation_semantic_repair_failures, 0)
        self.assertEqual(runtime.backend.messages_seen[2][-1]["content"], SHELL_FULL_SEMANTIC_REPAIR_PROMPT)

    def test_mutation_verification_rename_noop_requests_verification(self) -> None:
        class RenameNoopBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "mv note.txt note.txt 2>/dev/null || true"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content=json.dumps({"command": "ls -l note.txt"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    return ChatResult(
                        content="checked rename",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="rename inspected",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("alpha", encoding="utf-8")
            runtime = ChatRuntime(backend=RenameNoopBackend(), system_prompt="route system")
            runtime.ask_auto(
                "rename note.txt",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(runtime.mutation_verifications, 1)

    def test_mutation_verification_create_file_not_created(self) -> None:
        class CreateMissingBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content=json.dumps({"command": "touch missing/note.txt 2>/dev/null || true"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content=json.dumps({"command": "test -f missing/note.txt && echo exists || echo missing"}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    return ChatResult(
                        content="checked create",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="create not confirmed",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=CreateMissingBackend(), system_prompt="route system")
            result = runtime.ask_auto(
                "create missing/note.txt",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "create not confirmed")
        self.assertEqual(runtime.mutation_verifications, 1)

    def test_mutation_verification_failed_then_repair_succeeds(self) -> None:
        class VerificationRepairBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                commands = {
                    1: "printf alpha > note.txt",
                    2: "grep '[' note.txt",
                    3: "cat note.txt",
                }
                if self.calls in commands:
                    return ChatResult(
                        content=json.dumps({"command": commands[self.calls]}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="verified after repair",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=VerificationRepairBackend(), system_prompt="route system")
            result = runtime.ask_auto(
                "write alpha into note.txt",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "verified after repair")
        self.assertEqual(runtime.mutation_verifications, 1)
        self.assertEqual(runtime.mutation_verification_repairs, 1)
        self.assertEqual(runtime.mutation_verification_failures, 0)

    def test_mutation_verification_failed_then_repair_fails(self) -> None:
        class VerificationRepairFailsBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                commands = {
                    1: "printf alpha > note.txt",
                    2: "grep '[' note.txt",
                    3: "grep '[a-' note.txt",
                }
                if self.calls in commands:
                    return ChatResult(
                        content=json.dumps({"command": commands[self.calls]}),
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="verification failed",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=VerificationRepairFailsBackend(), system_prompt="route system")
            result = runtime.ask_auto(
                "write alpha into note.txt",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "verification failed")
        self.assertEqual(runtime.mutation_verifications, 1)
        self.assertEqual(runtime.mutation_verification_repairs, 1)
        self.assertEqual(runtime.mutation_verification_failures, 1)

    def test_mutation_detection_covers_sql_update_without_format_specific_runtime_validation(self) -> None:
        self.assertTrue(
            should_verify_shell_mutation(
                "sqlite3 app.db 'update users set name=\"x\" where id=999'",
                user_prompt="update the user",
            )
        )

    def test_completion_guard_continues_refactor_symbol_after_read_only_discovery(self) -> None:
        class CompletionGuardBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    content = json.dumps({"command": "grep -r \"slugify\" ."})
                elif self.calls == 2:
                    content = json.dumps(
                        {
                            "command": (
                                "python3 -c \"from pathlib import Path; "
                                "p=Path('names.py'); "
                                "p.write_text(p.read_text().replace('slugify', 'normalize_slug')); "
                                "print(p.read_text())\""
                            )
                        }
                    )
                elif self.calls == 3:
                    return ChatResult(
                        content="modification complete",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                else:
                    return ChatResult(
                        content="done",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=9,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=5,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "names.py").write_text(
                "def slugify(value):\n    return value\n\ndef make_id(value):\n    return slugify(value)\n",
                encoding="utf-8",
            )
            backend = CompletionGuardBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")
            result = runtime.ask_auto(
                "Rename slugify to normalize_slug across the codebase and update references.",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )
            updated = (workdir / "names.py").read_text(encoding="utf-8")

        self.assertEqual(result.content, "done")
        self.assertIn("def normalize_slug", updated)
        self.assertNotIn("slugify", updated)
        self.assertEqual(runtime.completion_guard_nudges, 1)
        self.assertEqual(runtime.completion_guard_commands, 1)
        self.assertEqual(runtime.completion_guard_successes, 1)
        self.assertEqual(runtime.completion_guard_failures, 0)
        self.assertEqual(backend.messages_seen[1][-1]["content"], SHELL_FULL_COMPLETION_GUARD_PROMPT)

    def test_content_evidence_guard_recovers_after_metadata_only_listing(self) -> None:
        class ContentEvidenceBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    content = json.dumps({"command": "ls -R"})
                elif self.calls == 2:
                    content = json.dumps({"command": "cat string_utils.py test_string_utils.py"})
                elif self.calls == 3:
                    content = json.dumps({"command": "python3 -c 'from pathlib import Path; p=Path(\"string_utils.py\"); p.write_text(\"def is_palindrome(value):\\n    normalized = value.lower().replace(\\\" \\\", \\\"\\\")\\n    return normalized == normalized[::-1]\\n\"); print(p.read_text())'"})
                elif self.calls == 4:
                    return ChatResult(
                        content="fixed",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=9,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                else:
                    return ChatResult(
                        content="done",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=10,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=5,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "string_utils.py").write_text("def is_palindrome(value):\n    return value == value[::-1]\n", encoding="utf-8")
            (workdir / "test_string_utils.py").write_text(
                "from string_utils import is_palindrome\n\n"
                "def test_palindrome_ignores_case_and_spaces():\n"
                "    assert is_palindrome('Never odd or even')\n",
                encoding="utf-8",
            )
            runtime = ChatRuntime(backend=ContentEvidenceBackend(), system_prompt="route system")
            result = runtime.ask_auto(
                "The test describes the desired behavior. Inspect files and fix the implementation.",
                temperature=0,
                max_tokens=128,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )
            updated = (workdir / "string_utils.py").read_text(encoding="utf-8")

        self.assertEqual(result.content, "done")
        self.assertIn("lower", updated)
        self.assertIn("replace", updated)
        self.assertEqual(runtime.content_evidence_guard_nudges, 1)
        self.assertEqual(runtime.content_evidence_guard_commands, 1)
        self.assertEqual(runtime.content_evidence_guard_successes, 1)
        self.assertEqual(runtime.content_evidence_guard_failures, 0)
        self.assertEqual(runtime.backend.messages_seen[1][-1]["content"], SHELL_FULL_CONTENT_EVIDENCE_GUARD_PROMPT)
        self.assertEqual(runtime.completion_guard_nudges, 0)

    def test_completion_guard_does_not_fire_for_read_only_analysis(self) -> None:
        class ReadOnlyBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    content = json.dumps({"command": "grep -n \"shell=True\" vulnerable_service.py"})
                elif self.calls == 2:
                    return ChatResult(
                        content="shell=True is vulnerable",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=4,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                else:
                    return ChatResult(
                        content="final review",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=5,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "vulnerable_service.py").write_text("subprocess.run(cmd, shell=True)\n", encoding="utf-8")
            backend = ReadOnlyBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")
            result = runtime.ask_auto(
                "Review vulnerable_service.py and report the vulnerable function. Do not modify files.",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "shell=True is vulnerable")
        self.assertEqual(runtime.completion_guard_nudges, 0)
        self.assertEqual(runtime.completion_guard_commands, 0)

    def test_completion_guard_covers_common_mutative_coding_tasks(self) -> None:
        cases = [
            (
                "config_update",
                "Update config.json: set timeout to 30 and enable features.cache.",
                {"config.json": '{"timeout":10,"features":{"cache":false}}\n'},
                "cat config.json",
                "python3 -c \"import json; p='config.json'; d=json.load(open(p)); d['timeout']=30; d['features']['cache']=True; open(p,'w').write(json.dumps(d)); print(open(p).read())\"",
                lambda workdir: '"timeout": 30' in (workdir / "config.json").read_text(encoding="utf-8")
                or '"timeout":30' in (workdir / "config.json").read_text(encoding="utf-8"),
            ),
            (
                "html_css_edit",
                "Change the page title to Dashboard and make the .btn color green.",
                {"index.html": "<title>Old</title>\n", "style.css": ".btn { color: blue; }\n"},
                "grep -r \"title\\|\\.btn\" .",
                "python3 -c \"from pathlib import Path; Path('index.html').write_text(Path('index.html').read_text().replace('<title>Old</title>','<title>Dashboard</title>')); Path('style.css').write_text(Path('style.css').read_text().replace('blue','green')); print(Path('index.html').read_text()+Path('style.css').read_text())\"",
                lambda workdir: "Dashboard" in (workdir / "index.html").read_text(encoding="utf-8")
                and "green" in (workdir / "style.css").read_text(encoding="utf-8"),
            ),
            (
                "python_edit",
                "Improve parse_port in service.py so invalid or out-of-range ports raise ValueError.",
                {"service.py": "def parse_port(value):\n    return int(value)\n"},
                "cat service.py",
                "printf '%s\\n' 'def parse_port(value):' '    port = int(value)' '    if not 1 <= port <= 65535:' '        raise ValueError(\"invalid port\")' '    return port' > service.py && cat service.py",
                lambda workdir: "ValueError" in (workdir / "service.py").read_text(encoding="utf-8")
                and "65535" in (workdir / "service.py").read_text(encoding="utf-8"),
            ),
            (
                "write_test",
                "Add a minimal pytest test for multiply(a, b) in math_utils.py.",
                {"math_utils.py": "def multiply(a, b):\n    return a * b\n"},
                "ls -F",
                "python3 -c \"from pathlib import Path; Path('test_math_utils.py').write_text('from math_utils import multiply\\n\\ndef test_multiply():\\n    assert multiply(2, 3) == 6\\n'); print(Path('test_math_utils.py').read_text())\"",
                lambda workdir: "multiply" in (workdir / "test_math_utils.py").read_text(encoding="utf-8"),
            ),
            (
                "fix_failed_test",
                "The test describes the desired behavior. Inspect files and fix the implementation.",
                {
                    "string_utils.py": "def is_palindrome(value):\n    return value == value[::-1]\n",
                    "test_string_utils.py": "from string_utils import is_palindrome\n\ndef test_palindrome_ignores_case_and_spaces():\n    assert is_palindrome('Never odd or even')\n",
                },
                "grep -R \"is_palindrome\" .",
                "python3 -c \"from pathlib import Path; Path('string_utils.py').write_text('def is_palindrome(value):\\n    normalized = value.lower().replace(\\\" \\\", \\\"\\\")\\n    return normalized == normalized[::-1]\\n'); print(Path('string_utils.py').read_text())\"",
                lambda workdir: "lower" in (workdir / "string_utils.py").read_text(encoding="utf-8")
                and "replace" in (workdir / "string_utils.py").read_text(encoding="utf-8"),
            ),
        ]

        class CompletionGuardCaseBackend:
            def __init__(self, read_command: str, mutate_command: str) -> None:
                self.calls = 0
                self.read_command = read_command
                self.mutate_command = mutate_command

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    content = json.dumps({"command": self.read_command})
                elif self.calls == 2:
                    content = json.dumps({"command": self.mutate_command})
                elif self.calls == 3:
                    return ChatResult(
                        content="modification complete",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=8,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                else:
                    return ChatResult(
                        content="done",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=9,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=5,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        for name, prompt, files, read_command, mutate_command, check in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                workdir = Path(tmp)
                for relative_path, content in files.items():
                    (workdir / relative_path).write_text(content, encoding="utf-8")
                runtime = ChatRuntime(
                    backend=CompletionGuardCaseBackend(read_command, mutate_command),
                    system_prompt="route system",
                )
                result = runtime.ask_auto(
                    prompt,
                    temperature=0,
                    max_tokens=64,
                    workdir=workdir,
                    allowed_tool_names=("exec_shell_full_command",),
                )

                self.assertEqual(result.content, "done")
                self.assertTrue(check(workdir))
                self.assertEqual(runtime.completion_guard_nudges, 1)
                self.assertEqual(runtime.completion_guard_commands, 1)
                self.assertEqual(runtime.completion_guard_successes, 1)
                self.assertEqual(runtime.completion_guard_failures, 0)

    def test_completion_guard_records_failure_when_nudge_does_not_return_command(self) -> None:
        class NoCommandBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    content = json.dumps({"command": "cat config.json"})
                elif self.calls == 2:
                    return ChatResult(
                        content="config.json has timeout 10",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=4,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                else:
                    return ChatResult(
                        content="not modified",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=5,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "config.json").write_text('{"timeout":10}\n', encoding="utf-8")
            runtime = ChatRuntime(backend=NoCommandBackend(), system_prompt="route system")
            result = runtime.ask_auto(
                "Update config.json: set timeout to 30.",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "not modified")
        self.assertEqual(runtime.completion_guard_nudges, 1)
        self.assertEqual(runtime.completion_guard_commands, 0)
        self.assertEqual(runtime.completion_guard_successes, 0)
        self.assertEqual(runtime.completion_guard_failures, 1)

    def test_minimal_patch_guard_retries_truncated_broad_rewrite_once(self) -> None:
        class MinimalPatchBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    content = json.dumps({"command": "cat script.sh"})
                elif self.calls == 2:
                    truncated = """{"command":"cat << 'EOF' > script.sh
#!/usr/bin/env bash
set -euo pipefail
cp "$1\""""
                    return ChatResult(
                        content=truncated,
                        model="fake",
                        finish_reason="length",
                        tool_calls=[
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": truncated,
                                },
                            }
                        ],
                        prompt_tokens=7,
                        completion_tokens=160,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                elif self.calls == 3:
                    content = json.dumps({"command": "python3 -c 'from pathlib import Path; p=Path(\"script.sh\"); s=p.read_text().replace(\"cp $1 $2\", \"cp quoted quoted\"); p.write_text(s); print(p.read_text())'"})
                elif self.calls == 4:
                    return ChatResult(
                        content="patched",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=9,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                else:
                    return ChatResult(
                        content="done",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=10,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=5,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "script.sh").write_text("#!/usr/bin/env bash\ncp $1 $2\n", encoding="utf-8")
            backend = MinimalPatchBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")
            result = runtime.ask_auto(
                "Harden script.sh for bash safety and correct quoting of arguments.",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )
            content = (workdir / "script.sh").read_text(encoding="utf-8")

        self.assertEqual(result.content, "done")
        self.assertIn("cp quoted quoted", content)
        self.assertEqual(runtime.minimal_patch_guard_nudges, 1)
        self.assertEqual(runtime.minimal_patch_guard_commands, 1)
        self.assertEqual(runtime.minimal_patch_guard_successes, 1)
        self.assertEqual(runtime.minimal_patch_guard_failures, 0)
        self.assertEqual(backend.messages_seen[2][-1]["content"], SHELL_FULL_MINIMAL_PATCH_GUARD_PROMPT)

    def test_minimal_patch_guard_intercepts_broad_rewrite_of_existing_file(self) -> None:
        class BroadRewriteBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    content = json.dumps(
                        {
                            "command": (
                                "cat << 'EOF' > copy_file.sh\n"
                                "#!/usr/bin/env bash\n"
                                "echo broad rewrite should not run\n"
                                "EOF"
                            )
                        }
                    )
                elif self.calls == 2:
                    content = json.dumps(
                        {
                            "command": (
                                "python3 -c 'from pathlib import Path; "
                                "p=Path(\"copy_file.sh\"); "
                                "s=p.read_text(); "
                                "s=s.replace(\"cp $1 $2\", \"set -euo pipefail\\ncp \\\"$1\\\" \\\"$2\\\"\"); "
                                "p.write_text(s); "
                                "print(p.read_text())'"
                            )
                        }
                    )
                elif self.calls == 3:
                    return ChatResult(
                        content="patched",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=9,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                else:
                    return ChatResult(
                        content="done",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=10,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=5,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "copy_file.sh").write_text("#!/usr/bin/env bash\ncp $1 $2\n", encoding="utf-8")
            backend = BroadRewriteBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")
            result = runtime.ask_auto(
                "Harden copy_file.sh for bash safety and correct quoting of arguments.",
                temperature=0,
                max_tokens=256,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )
            content = (workdir / "copy_file.sh").read_text(encoding="utf-8")

        self.assertEqual(result.content, "done")
        self.assertNotIn("broad rewrite should not run", content)
        self.assertIn("set -euo pipefail", content)
        self.assertIn('cp "$1" "$2"', content)
        self.assertEqual(runtime.minimal_patch_guard_nudges, 1)
        self.assertEqual(runtime.minimal_patch_guard_commands, 1)
        self.assertEqual(backend.messages_seen[1][-1]["content"], SHELL_FULL_MINIMAL_PATCH_GUARD_PROMPT)

    def test_minimal_patch_guard_detects_raw_invalid_json_broad_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "copy_file.sh").write_text("#!/usr/bin/env bash\ncp $1 $2\n", encoding="utf-8")
            raw_arguments = """{"command":"cat << 'EOF' > copy_file.sh
#!/usr/bin/env bash
set -euo pipefail
EOF"""
            tool_call = {
                "id": "call-1",
                "type": "function",
                "function": {"name": "exec_shell_full_command", "arguments": raw_arguments},
            }

            should_guard = _should_guard_existing_file_rewrite(
                tool_call,
                workdir=workdir,
                should_nudge_minimal_patch=lambda **kwargs: kwargs["existing_file_rewrite"],
            )

        self.assertTrue(should_guard)

    def test_minimal_patch_guard_does_not_fire_for_short_mutation(self) -> None:
        class ShortMutationBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    content = json.dumps({"command": "sed -i 's/old/new/' note.txt && cat note.txt"})
                else:
                    return ChatResult(
                        content="done",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=8,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content=content,
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=5,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("old\n", encoding="utf-8")
            runtime = ChatRuntime(backend=ShortMutationBackend(), system_prompt="route system")
            result = runtime.ask_auto(
                "replace old with new in note.txt",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "done")
        self.assertEqual(runtime.minimal_patch_guard_nudges, 0)

    def test_shell_full_allows_multiple_successive_command_rounds(self) -> None:
        class MultiShellFullBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tools_seen.append(tools)
                if self.calls == 1:
                    tool_call = {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"cat sample.txt\"}"},
                    }
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[tool_call],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    tool_call = {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"grep -n vulnerable sample.txt\"}"},
                    }
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[tool_call],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="final answer from multiple shell results",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "sample.txt").write_text("vulnerable=true\n", encoding="utf-8")
            backend = MultiShellFullBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools(
                "analyze sample.txt",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "final answer from multiple shell results")
        self.assertEqual(backend.calls, 3)
        self.assertIsNotNone(backend.tools_seen[0])
        self.assertIsNone(backend.tools_seen[1])
        self.assertIsNone(backend.tools_seen[2])
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)

    def test_ask_auto_streams_route_phase_when_progress_is_requested(self) -> None:
        class StreamingRouteBackend:
            def __init__(self) -> None:
                self.chat_calls = 0
                self.chat_stream_calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                raise AssertionError("route phase should use streaming when progress is requested")

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                self.chat_stream_calls += 1
                if self.chat_stream_calls == 1:
                    assert on_progress is not None
                    on_progress(type("P", (), {"phase": "prefill", "current": 12, "total": 48, "percent": 25})())
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"cat note.txt\"}"},
                            }
                        ],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                assert on_delta is not None
                on_delta("done")
                return ChatResult(
                    content="done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=6,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        progress: list[tuple[str, int, int, int]] = []
        emitted: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello", encoding="utf-8")
            backend = StreamingRouteBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "read note.txt",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                on_final_delta=emitted.append,
                on_progress=lambda item: progress.append((item.phase, item.current, item.total, item.percent)),
            )

        self.assertEqual(result.content, "done")
        self.assertEqual(emitted, ["done"])
        self.assertEqual(progress, [("prefill", 12, 48, 25)])
        self.assertEqual(backend.chat_calls, 0)
        self.assertEqual(backend.chat_stream_calls, 2)

    def test_ask_auto_aborts_streamed_route_prose_and_streams_final_retry(self) -> None:
        class StreamingRouteAnswerBackend:
            def __init__(self) -> None:
                self.chat_calls = 0
                self.chat_stream_calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                raise AssertionError("route phase should use streaming when progress is requested")

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                self.chat_stream_calls += 1
                assert on_delta is not None
                assert on_progress is not None
                on_progress(type("P", (), {"phase": "generation", "current": 12, "total": 128, "percent": 9})())
                if self.chat_stream_calls == 1:
                    on_delta("plain route prose")
                    return ChatResult(
                        content="plain route prose",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=3,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                on_delta("final answer")
                return ChatResult(
                    content="final answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=5,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        emitted: list[str] = []
        progress: list[tuple[str, int, int, int]] = []
        phases: list[tuple[str, str | None]] = []
        backend = StreamingRouteAnswerBackend()
        runtime = ChatRuntime(backend=backend, system_prompt="route system")

        result = runtime.ask_auto(
            "explain the plan before the final answer",
            temperature=0,
            max_tokens=128,
            workdir=Path("."),
            on_final_delta=emitted.append,
            on_progress=lambda item: progress.append((item.phase, item.current, item.total, item.percent)),
            on_phase_start=lambda phase: phases.append((phase.phase, phase.reason)),
            allowed_tool_names=("exec_shell_full_command",),
        )

        self.assertEqual(result.content, "final answer")
        self.assertEqual(emitted, ["final answer"])
        self.assertIn(("chat_final_retry", "route_not_command"), phases)
        self.assertEqual(progress, [("generation", 12, 128, 9), ("generation", 12, 128, 9)])
        self.assertEqual(backend.chat_calls, 0)
        self.assertEqual(backend.chat_stream_calls, 2)

    def test_route_stream_abort_stops_whitespace_only_route(self) -> None:
        route_filter = _RouteDecisionStreamAbort()

        with self.assertRaises(_RouteNotCommandAbort) as raised:
            for _ in range(8):
                route_filter.write(" ")

        self.assertEqual(raised.exception.chunks, 8)

    def test_route_stream_abort_allows_whitespace_before_command_prefix(self) -> None:
        route_filter = _RouteDecisionStreamAbort()

        for _ in range(3):
            route_filter.write(" ")
        route_filter.write("{")

    def test_ask_auto_discards_route_thought_and_runs_chat_final(self) -> None:
        class StreamingRouteThoughtBackend:
            def __init__(self) -> None:
                self.chat_calls = 0
                self.chat_stream_calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                raise AssertionError("streaming path expected")

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                self.chat_stream_calls += 1
                assert on_delta is not None
                assert on_progress is not None
                if self.chat_stream_calls == 1:
                    on_delta("<|channel>thought\ninternal route plan")
                    return ChatResult(
                        content="<|channel>thought\ninternal route plan",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=4,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                on_delta("<|channel>thought\nvisible thinking")
                on_delta("<channel|>visible answer")
                return ChatResult(
                    content="<|channel>thought\nvisible thinking<channel|>visible answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=9,
                    completion_tokens=6,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        emitted: list[str] = []
        backend = StreamingRouteThoughtBackend()
        runtime = ChatRuntime(backend=backend, system_prompt="route system")

        result = runtime.ask_auto(
            "who are you?",
            temperature=0,
            max_tokens=128,
            workdir=Path("."),
            on_final_delta=emitted.append,
            on_progress=lambda _item: None,
            allowed_tool_names=("exec_shell_full_command",),
        )

        self.assertEqual(result.content, "<|channel>thought\nvisible thinking<channel|>visible answer")
        self.assertEqual(emitted, ["<|channel>thought\nvisible thinking", "<channel|>visible answer"])
        self.assertEqual(backend.chat_calls, 0)
        self.assertEqual(backend.chat_stream_calls, 2)

    def test_ask_auto_discards_truncated_route_thought_and_streams_chat_final(self) -> None:
        class StreamingRouteLengthBackend:
            def __init__(self) -> None:
                self.chat_calls = 0
                self.chat_stream_calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                raise AssertionError("streaming path expected")

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                self.chat_stream_calls += 1
                assert on_delta is not None
                if self.chat_stream_calls == 1:
                    on_delta("<|channel>thought\ninternal route plan")
                    return ChatResult(
                        content="<|channel>thought\ninternal route plan",
                        model="fake",
                        finish_reason="length",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=4,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                on_delta("<|channel>thought\nvisible thinking")
                on_delta("<channel|>visible answer")
                return ChatResult(
                    content="<|channel>thought\nvisible thinking<channel|>visible answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=9,
                    completion_tokens=6,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        emitted: list[str] = []
        phases: list[tuple[str, bool, int | None, str | None]] = []
        backend = StreamingRouteLengthBackend()
        runtime = ChatRuntime(backend=backend, system_prompt="route system")

        result = runtime.ask_auto(
            "who are you?",
            temperature=0,
            max_tokens=128,
            workdir=Path("."),
            on_final_delta=emitted.append,
            on_progress=lambda _item: None,
            on_phase_start=lambda phase: phases.append((phase.phase, phase.streamed, phase.attempt, phase.reason)),
            allowed_tool_names=("exec_shell_full_command",),
        )

        self.assertEqual(result.content, "<|channel>thought\nvisible thinking<channel|>visible answer")
        self.assertEqual(emitted, ["<|channel>thought\nvisible thinking", "<channel|>visible answer"])
        self.assertIn(("chat_final_retry", True, 2, "route_not_command"), phases)
        self.assertEqual(backend.chat_calls, 0)
        self.assertEqual(backend.chat_stream_calls, 2)

    def test_complete_command_json_skips_tool_call_json_retry_round(self) -> None:
        class CompleteCommandBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    return ChatResult(
                        content='{"command":"printf ok"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = CompleteCommandBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                "try to analyze it with apktool",
                temperature=0,
                max_tokens=32,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.calls, 2)
        self.assertNotIn(TOOL_CALL_JSON_RETRY_PROMPT, [message.get("content") for messages in backend.messages_seen for message in messages])

    def test_ask_auto_uses_preferred_command_tool_when_valid(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tool_names_seen: list[tuple[str, ...]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tool_names_seen.append(tuple(tool["function"]["name"] for tool in tools or []))
                if self.calls == 1:
                    return ChatResult(
                        content='{"command":"cat note.txt"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"cat note.txt\"}"},
                            }
                        ],
                        prompt_tokens=6,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("x", encoding="utf-8")
            backend = RoutedBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto("read note.txt", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.tool_names_seen[0], ())
        self.assertEqual(backend.tool_names_seen[1], ())

    def test_ask_auto_executes_command_json_without_tool_selection_round(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tool_names_seen: list[tuple[str, ...]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tool_names_seen.append(tuple(tool["function"]["name"] for tool in tools or []))
                if self.calls == 1:
                    return ChatResult(
                        content='{"command":"cat note.txt"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=7,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("x", encoding="utf-8")
            backend = RoutedBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto("read note.txt", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.calls, 2)
        self.assertEqual(backend.tool_names_seen[0], ())
        self.assertEqual(backend.tool_names_seen[1], ())

    def test_ask_auto_rejects_metadata_only_file_read_after_prior_chat_turn(self) -> None:
        class MultiTurnBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    return ChatResult(
                        content="hello there",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=5,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content='{"command":"ls -F"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    if not messages[-1]["content"].startswith("Requested file not read yet."):
                        raise AssertionError(messages[-1]["content"])
                    return ChatResult(
                        content="I cannot confirm the content of the file yet.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 4:
                    if "still did not read the requested file contents" not in messages[-1]["content"]:
                        raise AssertionError(messages[-1]["content"])
                    return ChatResult(
                        content="I still cannot confirm the content of the file.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=8,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 5:
                    if "still did not read the requested file contents" not in messages[-1]["content"]:
                        raise AssertionError(messages[-1]["content"])
                    return ChatResult(
                        content='{"command":"cat README.md"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "README.md").write_text("Orbit README", encoding="utf-8")
            backend = MultiTurnBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            first = runtime.ask_auto("hi", temperature=0, max_tokens=32, workdir=workdir)
            second = runtime.ask_auto(
                "read the README.md file and explain it in a concise three-sentence list",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(first.content, "hello there")
        self.assertEqual(second.content, "done")
        self.assertEqual(backend.calls, 6)
        self.assertIn("hi", str(backend.messages_seen[1]))
        self.assertNotIn("I cannot confirm the content of the file yet.", str(backend.messages_seen[3]))
        self.assertNotIn("I still cannot confirm the content of the file.", str(backend.messages_seen[4]))

    def test_ask_auto_retries_direct_final_when_latest_turn_requires_file_content_after_listing(self) -> None:
        class FileEvidenceRouteBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    return ChatResult(
                        content="README.md contains project documentation.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    if "specific file's contents" not in messages[-1]["content"]:
                        raise AssertionError(messages[-1]["content"])
                    return ChatResult(
                        content='{"command":"cat README.md"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="The README content was read and summarized.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=8,
                    completion_tokens=3,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "README.md").write_text("Orbit README content", encoding="utf-8")
            backend = FileEvidenceRouteBackend()
            runtime = ChatRuntime(
                backend=backend,
                system_prompt="route system",
                messages=[
                    {"role": "system", "content": "route system"},
                    {"role": "user", "content": "list files in the workdir"},
                    {"role": "assistant", "content": "", "tool_calls": []},
                    {"role": "tool", "name": "list_directory", "tool_call_id": "call-list", "content": "README.md"},
                    {"role": "assistant", "content": "README.md"},
                ],
            )

            result = runtime.ask_auto(
                "read README.md and explain it",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "The README content was read and summarized.")
        self.assertEqual(backend.calls, 3)
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(tool_messages[-1]["name"], "exec_shell_full_command")
        self.assertIn("Orbit README content", tool_messages[-1]["content"])

    def test_ask_auto_uses_tool_loop_when_file_content_route_retry_still_has_no_decision(self) -> None:
        class FileEvidenceToolLoopBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tools_seen.append(tools)
                if self.calls == 1:
                    return ChatResult(
                        content="README.md appears to be documentation.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=6,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="I should inspect the file first.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=7,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"cat README.md\"}"},
                            }
                        ],
                        prompt_tokens=8,
                        completion_tokens=1,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="The README content was read after the fallback.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=9,
                    completion_tokens=3,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "README.md").write_text("Orbit README content", encoding="utf-8")
            backend = FileEvidenceToolLoopBackend()
            runtime = ChatRuntime(
                backend=backend,
                system_prompt="route system",
                messages=[
                    {"role": "system", "content": "route system"},
                    {"role": "user", "content": "list files in the workdir"},
                    {"role": "tool", "name": "list_directory", "tool_call_id": "call-list", "content": "README.md"},
                    {"role": "assistant", "content": "README.md"},
                ],
            )

            result = runtime.ask_auto(
                "read README.md and explain it",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "The README content was read after the fallback.")
        self.assertEqual(backend.calls, 4)
        self.assertIsNone(backend.tools_seen[0])
        self.assertIsNone(backend.tools_seen[1])
        self.assertIsNotNone(backend.tools_seen[2])
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(tool_messages[-1]["name"], "exec_shell_full_command")
        self.assertIn("Orbit README content", tool_messages[-1]["content"])

    def test_ask_with_tools_requires_real_url_fetch_before_answering(self) -> None:
        class UrlRecoveryBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    return ChatResult(
                        content="The page seems hypothetical because the date looks future-dated.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=8,
                        completion_tokens=4,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    if "Requested URL content not fetched yet." not in str(messages[-1]["content"]):
                        raise AssertionError(messages[-1]["content"])
                    return ChatResult(
                        content='{"command":"python3 -c \\"print(\'<html><body>Central thesis: human dignity and solidarity.</body></html>\')\\" https://example.com/doc"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=10,
                        completion_tokens=3,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Grounded final answer.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=12,
                    completion_tokens=3,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=UrlRecoveryBackend(), system_prompt="route system")
            result = runtime.ask_with_tools(
                "Fetch https://example.com/doc and explain the central thesis in Italian.",
                temperature=0,
                max_tokens=64,
                workdir=Path(tmp),
            )

        self.assertEqual(result.content, "Grounded final answer.")

    def test_ask_with_tools_requires_real_url_fetch_after_runtime_reset(self) -> None:
        class UrlRecoveryAfterResetBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    return ChatResult(
                        content="hello back",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=4,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="I cannot confirm the page contents yet.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=8,
                        completion_tokens=4,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    if "Requested URL content not fetched yet." not in str(messages[-1]["content"]):
                        raise AssertionError(messages[-1]["content"])
                    return ChatResult(
                        content='{"command":"python3 -c \\"print(\'<html><body>Central thesis: grounded evidence.</body></html>\')\\" https://example.com/doc"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=10,
                        completion_tokens=3,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Grounded after reset.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=12,
                    completion_tokens=3,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = UrlRecoveryAfterResetBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")
            runtime.ask("hi", temperature=0, max_tokens=32)
            runtime.reset()
            result = runtime.ask_with_tools(
                "Fetch https://example.com/doc and explain the central thesis in Italian.",
                temperature=0,
                max_tokens=64,
                workdir=Path(tmp),
            )

        self.assertEqual(result.content, "Grounded after reset.")

    def test_ask_with_tools_retries_when_url_fetch_only_returns_transfer_progress(self) -> None:
        class UrlProgressOnlyBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    return ChatResult(
                        content='{"command":"python3 -c \\"import sys; sys.stderr.write(\'% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\\\\n\'); sys.stderr.write(\'                                 Dload  Upload   Total   Spent    Left  Speed\\\\n\'); sys.stderr.write(\'  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0\\\\n\')\\" https://example.com/doc"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=8,
                        completion_tokens=4,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="I still cannot confirm the page contents yet.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=8,
                        completion_tokens=4,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    if "did not obtain usable content or a real HTTP/network failure" not in str(messages[-1]["content"]):
                        raise AssertionError(messages[-1]["content"])
                    return ChatResult(
                        content='{"command":"python3 -c \\"print(\'<html><body>Central thesis: grounded evidence.</body></html>\')\\" https://example.com/doc"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=10,
                        completion_tokens=3,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Grounded after retry.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=12,
                    completion_tokens=3,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=UrlProgressOnlyBackend(), system_prompt="route system")
            result = runtime.ask_with_tools(
                "Fetch https://example.com/doc and explain the central thesis in Italian.",
                temperature=0,
                max_tokens=64,
                workdir=Path(tmp),
            )

        self.assertEqual(result.content, "Grounded after retry.")

    def test_ask_with_tools_allows_grounded_url_failure_after_real_fetch_attempt(self) -> None:
        class UrlFailureBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content='{"command":"python3 -c \\"import sys; sys.stderr.write(\'HTTP 404 Not Found\\\\n\'); raise SystemExit(22)\\" https://example.com/missing"}',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=8,
                        completion_tokens=3,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="The URL returned an HTTP 404 error.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=10,
                    completion_tokens=4,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=UrlFailureBackend(), system_prompt="route system")
            result = runtime.ask_with_tools(
                "Fetch https://example.com/missing and summarize it.",
                temperature=0,
                max_tokens=64,
                workdir=Path(tmp),
            )

        self.assertEqual(result.content, "The URL returned an HTTP 404 error.")

    def test_ask_with_tools_accepts_fetch_url_success_as_url_evidence(self) -> None:
        class FetchUrlSuccessBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "fetch_url", "arguments": '{"url":"https://example.com/doc"}'},
                            }
                        ],
                        prompt_tokens=8,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Grounded final answer from fetch_url.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=10,
                    completion_tokens=3,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=FetchUrlSuccessBackend(), system_prompt="route system")
            result = runtime.ask_with_tools(
                "Fetch https://example.com/doc and summarize it.",
                temperature=0,
                max_tokens=64,
                workdir=Path(tmp),
            )

        self.assertEqual(result.content, "Grounded final answer from fetch_url.")

    def test_ask_with_tools_accepts_fetch_url_http_error_as_failure_evidence(self) -> None:
        class FetchUrlFailureBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "fetch_url", "arguments": '{"url":"https://example.com/missing"}'},
                            }
                        ],
                        prompt_tokens=8,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="The page returned HTTP 404.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=10,
                    completion_tokens=3,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=FetchUrlFailureBackend(), system_prompt="route system")
            result = runtime.ask_with_tools(
                "Fetch https://example.com/missing and summarize it.",
                temperature=0,
                max_tokens=64,
                workdir=Path(tmp),
            )

        self.assertEqual(result.content, "The page returned HTTP 404.")

    def test_ask_with_tools_does_not_force_fetch_for_non_fetch_url_question(self) -> None:
        class DomainOnlyBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                return ChatResult(
                    content="The domain is example.com.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=6,
                    completion_tokens=3,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        runtime = ChatRuntime(backend=DomainOnlyBackend(), system_prompt="route system")
        result = runtime.ask_with_tools(
            "What is the domain in this URL: https://example.com/foo?",
            temperature=0,
            max_tokens=64,
            workdir=Path("."),
        )

        self.assertEqual(result.content, "The domain is example.com.")

    def test_ask_auto_media_without_path_is_normal_chat_without_command_json(self) -> None:
        class MediaRouteBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                return ChatResult(
                    content='media requested',
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=5,
                    completion_tokens=1,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            backend = MediaRouteBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_auto("describe the image", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "media requested")
        self.assertEqual(backend.calls, 1)

    def test_ask_auto_attaches_referenced_media_without_command_json(self) -> None:
        class DirectMediaBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages: list[Message] = []
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages = messages
                self.tools_seen.append(tools)
                return ChatResult(
                    content="direct media answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=6,
                    completion_tokens=2,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "tiny.png").write_bytes(png_bytes)
            backend = DirectMediaBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_auto("describe tiny.png", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "direct media answer")
        self.assertEqual(backend.calls, 1)
        self.assertEqual(backend.tools_seen, [None])
        self.assertEqual(backend.messages[0]["content"], MEDIA_SYSTEM_PROMPT)
        user_content = backend.messages[1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[1]["type"], "image_url")

    def test_ask_with_tools_reinjects_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "file.txt").write_text("x", encoding="utf-8")
            backend = ToolCallingBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("list files", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.calls, 3)
        tool_message = _last_tool_message(runtime)
        self.assertIn("file.txt", tool_message["content"])

    def test_ask_with_tools_retries_empty_length_final_after_tool_result(self) -> None:
        class EmptyLengthFinalBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tools_seen.append(tools)
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                            "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"ls -l note.txt\"}"},
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=8,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="length",
                        tool_calls=[],
                        prompt_tokens=20,
                        completion_tokens=96,
                        cached_tokens=10,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="final from tool result",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=25,
                    completion_tokens=4,
                    cached_tokens=12,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("x", encoding="utf-8")
            backend = EmptyLengthFinalBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools(
                "show note.txt metadata",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "final from tool result")
        self.assertEqual(backend.calls, 3)
        self.assertIsNotNone(backend.tools_seen[0])
        self.assertIsNotNone(backend.tools_seen[1])
        self.assertIsNone(backend.tools_seen[2])

    def test_ask_with_tools_retries_incomplete_stop_final_after_tool_result(self) -> None:
        class IncompleteStopFinalBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tools_seen: list[object] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tools_seen.append(tools)
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"pdftotext report.pdf - | head -n 100\"}"},
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=8,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=16,
                        completion_tokens=8,
                        cached_tokens=9,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 3:
                    return ChatResult(
                        content="Il documento e una relazione tecnica per il servizio di gestione della rete QX",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=20,
                        completion_tokens=32,
                        cached_tokens=10,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Sintesi: relazione tecnica sulla gestione, manutenzione ed evoluzione della rete QX.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=24,
                    completion_tokens=18,
                    cached_tokens=12,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            backend = IncompleteStopFinalBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools(
                "analizza il PDF e fammi una sintesi",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "Sintesi: relazione tecnica sulla gestione, manutenzione ed evoluzione della rete QX.")
        self.assertEqual(backend.calls, 4)
        self.assertIsNotNone(backend.tools_seen[0])
        self.assertIsNone(backend.tools_seen[1])
        self.assertIsNone(backend.tools_seen[2])
        self.assertIsNone(backend.tools_seen[3])

    def test_ask_with_tools_can_read_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello from file", encoding="utf-8")
            backend = ToolCallingBackend(tool_name="exec_shell_full_command", arguments="{\"command\":\"cat note.txt\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("read note.txt", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "done")
        tool_message = _last_tool_message(runtime)
        self.assertIn("hello from file", tool_message["content"])

    def test_ask_with_tools_can_stat_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello", encoding="utf-8")
            backend = ToolCallingBackend(tool_name="exec_shell_full_command", arguments="{\"command\":\"stat note.txt && wc -c note.txt\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("stat note.txt", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "done")
        tool_message = _last_tool_message(runtime)
        self.assertIn("note.txt", tool_message["content"])
        self.assertIn("5 note.txt", tool_message["content"])

    def test_ask_with_tools_can_reinject_fetch_url_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            backend = ToolCallingBackend(tool_name="exec_shell_full_command", arguments="{\"command\":\"printf '<html><body>Hello web</body></html>'\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("fetch a URL", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "done")
        tool_message = _last_tool_message(runtime)
        self.assertIn("Hello web", tool_message["content"])

    def test_ask_with_tools_caps_fetch_url_final_answer(self) -> None:
        class FetchBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.max_tokens_seen: list[int] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.max_tokens_seen.append(max_tokens)
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"printf '<html><body>Hello web</body></html>'\"}"},
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=8,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="short web synthesis",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=20,
                    completion_tokens=4,
                    cached_tokens=10,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = FetchBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_with_tools(
                "summarize URL",
                temperature=0,
                max_tokens=512,
                workdir=Path(tmp),
                tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(result.content, "short web synthesis")
        self.assertEqual(backend.max_tokens_seen, [96, 96, 72])

    def test_ask_with_tools_retries_raw_tool_call_in_final_answer(self) -> None:
        class RawToolCallFinalBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"lscpu\"}"},
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=8,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content='<|tool_call>call:exec_shell_full_command{"command":"grep cpu /proc/cpuinfo"}<tool_call|>',
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=20,
                        completion_tokens=4,
                        cached_tokens=10,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="CPU information from tool result",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=22,
                    completion_tokens=5,
                    cached_tokens=10,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = RawToolCallFinalBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            steps = []
            result = runtime.ask_with_tools(
                "what cpu do I have?",
                temperature=0,
                max_tokens=512,
                workdir=Path(tmp),
                tool_names=("exec_shell_full_command",),
                on_model_step=steps.append,
            )

        self.assertEqual(result.content, "CPU information from tool result")
        self.assertEqual(backend.calls, 3)
        self.assertEqual(steps[-1].phase, "final_from_tool")

    def test_ask_with_tools_traces_tool_call_retry_in_final_answer(self) -> None:
        class ToolCallFinalBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"pwd\"}"},
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=8,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"ls\"}"},
                            }
                        ],
                        prompt_tokens=20,
                        completion_tokens=4,
                        cached_tokens=10,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="final answer from first tool result",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=22,
                    completion_tokens=5,
                    cached_tokens=10,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            backend = ToolCallFinalBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            steps = []
            result = runtime.ask_with_tools(
                "show current directory",
                temperature=0,
                max_tokens=512,
                workdir=Path(tmp),
                tool_names=("exec_shell_full_command",),
                on_model_step=steps.append,
            )

        self.assertEqual(result.content, "final answer from first tool result")
        self.assertEqual(backend.calls, 4)
        self.assertEqual(steps[-1].phase, "final_from_tool")

    def test_final_from_tool_uses_final_prompt_not_tool_call_prompt(self) -> None:
        class PromptCaptureBackend(ToolCallingBackend):
            def __init__(self) -> None:
                super().__init__(tool_name="exec_shell_full_command", arguments="{\"command\":\"cat note.txt\"}")
                self.system_prompts: list[str] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                system = messages[0].get("content") if messages and messages[0].get("role") == "system" else None
                if isinstance(system, str):
                    self.system_prompts.append(system)
                return super().chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello", encoding="utf-8")
            backend = PromptCaptureBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="system")

            runtime.ask_with_tools("read note", temperature=0, max_tokens=32, workdir=workdir, tool_names=("exec_shell_full_command",))

        self.assertEqual(backend.system_prompts[-1], FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertNotIn("When tools are available", backend.system_prompts[-1])

    def test_final_from_tool_streams_thought_channel_verbatim_to_renderer(self) -> None:
        class StreamingFinalThoughtBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                raise AssertionError("streaming path expected")

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"pwd\"}"},
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=8,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                assert on_delta is not None
                on_delta("<|channel>thought\nfrom tool result")
                on_delta("<channel|>final answer")
                return ChatResult(
                    content="<|channel>thought\nfrom tool result<channel|>final answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=22,
                    completion_tokens=5,
                    cached_tokens=10,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            emitted: list[str] = []
            backend = StreamingFinalThoughtBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_with_tools(
                "show current directory",
                temperature=0,
                max_tokens=512,
                workdir=Path(tmp),
                tool_names=("exec_shell_full_command",),
                on_final_delta=emitted.append,
            )

        self.assertEqual(result.content, "<|channel>thought\nfrom tool result<channel|>final answer")
        self.assertEqual(emitted, ["<|channel>thought\nfrom tool result", "<channel|>final answer"])

    def test_final_from_tool_uses_compact_retry_once_when_thinking_hits_length(self) -> None:
        class StreamingFinalThoughtLengthBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.continue_calls = 0
                self.thinking = True

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                raise AssertionError("streaming path expected")

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    assert on_delta is not None
                    on_delta("<|channel>thought\nplan before tools<channel|>ignored final")
                    return ChatResult(
                        content="<|channel>thought\nplan before tools<channel|>ignored final",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=8,
                        completion_tokens=5,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"pwd\"}"},
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=8,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                assert on_delta is not None
                on_delta("<|channel>thought\npartial")
                if self.calls == 3:
                    return ChatResult(
                        content="<|channel>thought\npartial",
                        model="fake",
                        finish_reason="length",
                        tool_calls=[],
                        prompt_tokens=22,
                        completion_tokens=5,
                        cached_tokens=10,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                on_delta("short final answer")
                return ChatResult(
                    content="short final answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=24,
                    completion_tokens=6,
                    cached_tokens=10,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

            def continue_current(self, *, max_tokens: int, on_delta=None, on_progress=None) -> ChatResult:
                self.continue_calls += 1
                raise AssertionError("final-from-tool should not use native continuation")

        with tempfile.TemporaryDirectory() as tmp:
            emitted: list[str] = []
            steps = []
            backend = StreamingFinalThoughtLengthBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)
            result = runtime.ask_with_tools(
                "show current directory",
                temperature=0,
                max_tokens=128,
                workdir=Path(tmp),
                tool_names=("exec_shell_full_command",),
                on_final_delta=emitted.append,
                on_model_step=steps.append,
            )

        self.assertEqual(backend.continue_calls, 0)
        self.assertEqual(result.content, "short final answer")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(emitted, ["plan before tools", "<|channel>thought\npartial", "short final answer"])
        self.assertIn(steps[-1].phase, {"final_from_tool", "final_from_tool_compact_retry"})

    def test_final_from_tool_non_stream_uses_compact_retry_once_when_thinking_hits_length(self) -> None:
        class FinalThoughtLengthBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.continue_calls = 0
                self.thinking = True

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"pwd\"}"},
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=8,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="<|channel>thought\npartial",
                        model="fake",
                        finish_reason="length",
                        tool_calls=[],
                        prompt_tokens=22,
                        completion_tokens=5,
                        cached_tokens=10,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="short final answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=24,
                    completion_tokens=6,
                    cached_tokens=10,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

            def continue_current(self, *, max_tokens: int, on_delta=None, on_progress=None) -> ChatResult:
                self.continue_calls += 1
                raise AssertionError("final-from-tool should not use native continuation")

        with tempfile.TemporaryDirectory() as tmp:
            backend = FinalThoughtLengthBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)
            result = runtime.ask_with_tools(
                "show current directory",
                temperature=0,
                max_tokens=128,
                workdir=Path(tmp),
                tool_names=("exec_shell_full_command",),
            )

        self.assertEqual(backend.continue_calls, 0)
        self.assertEqual(result.content, "short final answer")
        self.assertEqual(result.finish_reason, "stop")

    def test_ask_with_tools_streams_planning_thought_when_thinking_mode_is_on(self) -> None:
        class StreamingPlanningThoughtBackend:
            def __init__(self) -> None:
                self.chat_stream_calls = 0
                self.thinking = True

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                raise AssertionError("streaming path expected")

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                self.chat_stream_calls += 1
                assert on_delta is not None
                if self.chat_stream_calls == 1:
                    on_delta("<|channel>thought\nplan before tools<channel|>ignored final")
                    return ChatResult(
                        content="<|channel>thought\nplan before tools<channel|>ignored final",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=8,
                        completion_tokens=5,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.chat_stream_calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"pwd\"}"},
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                on_delta("<|channel>thought\nfrom tool result")
                on_delta("<channel|>final answer")
                return ChatResult(
                    content="<|channel>thought\nfrom tool result<channel|>final answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=22,
                    completion_tokens=5,
                    cached_tokens=10,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            emitted: list[str] = []
            backend = StreamingPlanningThoughtBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None, thinking_mode=True)
            result = runtime.ask_with_tools(
                "show current directory",
                temperature=0,
                max_tokens=512,
                workdir=Path(tmp),
                tool_names=("exec_shell_full_command",),
                on_final_delta=emitted.append,
            )

        self.assertEqual(result.content, "<|channel>thought\nfrom tool result<channel|>final answer")
        self.assertEqual(
            emitted,
            [
                "plan before tools",
                "<|channel>thought\nfrom tool result",
                "<channel|>final answer",
            ],
        )

    def test_ask_with_tools_can_write_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            backend = ToolCallingBackend(tool_name="exec_shell_full_command", arguments="{\"command\":\"printf hello > note.txt\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("create note.txt", temperature=0, max_tokens=32, workdir=workdir)

            self.assertEqual((workdir / "note.txt").read_text(encoding="utf-8"), "hello")

        self.assertEqual(result.content, "done")
        self.assertEqual(_last_tool_message(runtime)["content"], "")

    def test_ask_with_tools_can_append_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("first\n", encoding="utf-8")
            backend = ToolCallingBackend(tool_name="exec_shell_full_command", arguments="{\"command\":\"printf 'second\\n' >> note.txt\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("append note.txt", temperature=0, max_tokens=32, workdir=workdir)

            self.assertEqual((workdir / "note.txt").read_text(encoding="utf-8"), "first\nsecond\n")

        self.assertEqual(result.content, "done")
        self.assertEqual(_last_tool_message(runtime)["content"], "")

    def test_read_only_request_rejects_mutating_shell_command_before_execution(self) -> None:
        class ReadOnlyMutationBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.messages_seen.append(messages)
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"sed -i 's/beta/delta/' note.txt\"}"},
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=8,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"cat note.txt\"}"},
                            }
                        ],
                        prompt_tokens=12,
                        completion_tokens=2,
                        cached_tokens=8,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="read completed",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=20,
                    completion_tokens=3,
                    cached_tokens=10,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            backend = ReadOnlyMutationBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools(
                "read note.txt and explain it",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                tool_names=("exec_shell_full_command",),
            )
            content = (workdir / "note.txt").read_text(encoding="utf-8")

        self.assertEqual(result.content, "read completed")
        self.assertEqual(content, "alpha\nbeta\ngamma\n")
        self.assertEqual(backend.calls, 3)
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertTrue(tool_messages)
        self.assertIn(SHELL_FULL_READ_ONLY_MUTATION_ERROR_PREFIX, tool_messages[0]["content"])
        self.assertIn("beta", tool_messages[-1]["content"])
        self.assertEqual(backend.messages_seen[1][-1]["content"], SHELL_FULL_READ_ONLY_MUTATION_RETRY_PROMPT)

    def test_ask_with_tools_can_replace_unique_text_in_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello old world\n", encoding="utf-8")
            backend = ToolCallingBackend(tool_name="exec_shell_full_command", arguments="{\"command\":\"sed -i 's/old/new/' note.txt\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("replace old", temperature=0, max_tokens=32, workdir=workdir)

            self.assertEqual((workdir / "note.txt").read_text(encoding="utf-8"), "hello new world\n")

        self.assertEqual(result.content, "done")
        self.assertEqual(_last_tool_message(runtime)["content"], "")

    def test_ask_with_tools_can_read_file_chunk_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "large.txt").write_text("abcdef" * 50000, encoding="utf-8")
            backend = ToolCallingBackend(tool_name="exec_shell_full_command", arguments="{\"command\":\"cat large.txt\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("read chunk", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "done")
        tool_message = _last_tool_message(runtime)
        self.assertIn("shell_output_read_file: true", tool_message["content"])
        self.assertIn("large_file_excerpt: true", tool_message["content"])

    def test_ask_with_tools_records_memory_refresh_event(self) -> None:
        class MemoryThenAnswerBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if tools is None:
                    return ChatResult(
                        content="Durable memory.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=None,
                        completion_tokens=None,
                        cached_tokens=None,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=None,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = MemoryThenAnswerBackend()
        runtime = ChatRuntime(backend=backend, system_prompt="system", context_tokens=100)
        for index in range(20):
            runtime.messages.append({"role": "user", "content": f"question {index} " + ("x" * 60)})
            runtime.messages.append({"role": "assistant", "content": f"answer {index} " + ("y" * 60)})

        result = runtime.ask_with_tools("continue", temperature=0, max_tokens=32, workdir=Path("."))

        self.assertEqual(result.content, "done")
        self.assertIsNotNone(runtime.last_memory_refresh)
        self.assertTrue(runtime.last_memory_refresh.changed)
        self.assertIs(runtime.last_memory_refresh_attempt, runtime.last_memory_refresh)
        self.assertEqual(runtime.memory_refreshes, 1)
        self.assertGreater(runtime.total_memory_tokens_saved, 0)

    def test_memory_refresh_cooldown_avoids_back_to_back_refresh(self) -> None:
        class MemoryThenAnswerBackend:
            def __init__(self) -> None:
                self.memory_calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                if tools is None:
                    self.memory_calls += 1
                    return ChatResult(
                        content="Durable memory.",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=None,
                        completion_tokens=None,
                        cached_tokens=None,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=None,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        backend = MemoryThenAnswerBackend()
        runtime = ChatRuntime(backend=backend, system_prompt="system", context_tokens=100)
        for index in range(20):
            runtime.messages.append({"role": "user", "content": f"question {index} " + ("x" * 60)})
            runtime.messages.append({"role": "assistant", "content": f"answer {index} " + ("y" * 60)})

        runtime.ask_with_tools("continue", temperature=0, max_tokens=32, workdir=Path("."))
        runtime.ask_with_tools("continue again", temperature=0, max_tokens=32, workdir=Path("."))

        self.assertEqual(backend.memory_calls, 1)
        self.assertEqual(runtime.memory_refreshes, 1)

    def test_ask_with_tools_streams_final_text(self) -> None:
        class StreamingBackend:
            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                raise AssertionError("chat should not be used when streaming is requested")

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                assert on_delta is not None
                on_delta("hel")
                on_delta("lo")
                return ChatResult(
                    content="hello",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=None,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        emitted: list[str] = []
        runtime = ChatRuntime(backend=StreamingBackend(), system_prompt=None)

        result = runtime.ask_with_tools("say hello", temperature=0, max_tokens=32, workdir=Path("."), on_final_delta=emitted.append)

        self.assertEqual(emitted, ["hel", "lo"])
        self.assertEqual(result.content, "hello")
        self.assertEqual(runtime.messages[-1]["content"], "hello")

    def test_ask_with_tools_streaming_can_execute_tool_then_stream_final_text(self) -> None:
        class StreamingToolBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                raise AssertionError("chat should not be used when streaming is requested")

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"cat note.txt\"}"},
                            }
                        ],
                        prompt_tokens=None,
                        completion_tokens=None,
                        cached_tokens=None,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                assert on_delta is not None
                on_delta("done")
                return ChatResult(
                    content="done",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=None,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        emitted: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello from file", encoding="utf-8")
            backend = StreamingToolBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("read note", temperature=0, max_tokens=32, workdir=workdir, on_final_delta=emitted.append)

        self.assertEqual(result.content, "done")
        self.assertEqual(emitted, ["done"])
        self.assertEqual(backend.calls, 2)

    def test_ask_with_tools_streaming_retries_empty_final_then_streams_thought_and_answer(self) -> None:
        class EmptyThenThoughtFinalBackend:
            def __init__(self) -> None:
                self.chat_calls = 0
                self.chat_stream_calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.chat_calls += 1
                return ChatResult(
                    content="",
                    model="fake",
                    finish_reason="tool_calls",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"cat note.txt\"}"},
                        }
                    ],
                    prompt_tokens=None,
                    completion_tokens=None,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                self.chat_stream_calls += 1
                if self.chat_stream_calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"cat note.txt\"}"},
                            }
                        ],
                        prompt_tokens=None,
                        completion_tokens=None,
                        cached_tokens=None,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                assert on_delta is not None
                if self.chat_stream_calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="stop",
                        tool_calls=[],
                        prompt_tokens=None,
                        completion_tokens=0,
                        cached_tokens=None,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                on_delta("<|channel>thought\nfrom tool result")
                on_delta("<channel|>final answer")
                return ChatResult(
                    content="<|channel>thought\nfrom tool result<channel|>final answer",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=6,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        emitted: list[str] = []
        steps = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello from file", encoding="utf-8")
            backend = EmptyThenThoughtFinalBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools(
                "read note",
                temperature=0,
                max_tokens=64,
                workdir=workdir,
                on_final_delta=emitted.append,
                on_model_step=steps.append,
            )

        self.assertEqual(result.content, "<|channel>thought\nfrom tool result<channel|>final answer")
        self.assertEqual(emitted, ["<|channel>thought\nfrom tool result<channel|>final answer"])
        self.assertEqual(backend.chat_calls, 0)
        self.assertEqual(backend.chat_stream_calls, 3)
        phases = [step.phase for step in steps]
        self.assertEqual(phases, ["tool_call", "final_from_tool", "final_from_tool_retry"])

    def test_final_from_tool_emits_result_content_when_buffered_stream_has_no_deltas(self) -> None:
        class BufferedNoDeltaBackend:
            def __init__(self) -> None:
                self.chat_stream_calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                raise AssertionError("streaming path expected")

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                self.chat_stream_calls += 1
                if self.chat_stream_calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"orbit-web-search 'Mario Nobile'\"}"},
                            }
                        ],
                        prompt_tokens=None,
                        completion_tokens=None,
                        cached_tokens=None,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="Mario Nobile is an Italian public official.",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=9,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        emitted: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            backend = BufferedNoDeltaBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools(
                "search online for information about Mario Nobile",
                temperature=0,
                max_tokens=64,
                workdir=Path(tmp),
                on_final_delta=emitted.append,
            )

        self.assertEqual(result.content, "Mario Nobile is an Italian public official.")
        self.assertEqual(emitted, ["Mario Nobile is an Italian public official."])
        self.assertEqual(backend.chat_stream_calls, 2)

    def test_final_from_web_search_streams_and_stops_on_length_without_hidden_retry(self) -> None:
        class WebSearchLengthBackend:
            def __init__(self) -> None:
                self.chat_stream_calls = 0
                self.max_tokens_seen: list[int] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                raise AssertionError("streaming path expected")

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None, on_progress=None) -> ChatResult:
                self.chat_stream_calls += 1
                self.max_tokens_seen.append(max_tokens)
                if self.chat_stream_calls == 1:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "exec_shell_full_command", "arguments": "{\"command\":\"orbit-web-search 'sample topic'\"}"},
                            }
                        ],
                        prompt_tokens=None,
                        completion_tokens=None,
                        cached_tokens=None,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                assert on_delta is not None
                on_delta("partial web search answer")
                return ChatResult(
                    content="partial web search answer",
                    model="fake",
                    finish_reason="length",
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=72,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        emitted: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            backend = WebSearchLengthBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            with patch(
                "orbit.runtime.shell_guardrails.search_web",
                return_value="web_search_results: true\nquery: sample topic\nresults:\n- one result",
            ):
                result = runtime.ask_with_tools(
                    "search online for sample topic",
                    temperature=0,
                    max_tokens=512,
                    workdir=Path(tmp),
                    on_final_delta=emitted.append,
                )

        self.assertEqual(result.content, "partial web search answer")
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(emitted, ["partial web search answer"])
        self.assertEqual(backend.chat_stream_calls, 2)
        self.assertEqual(backend.max_tokens_seen, [96, 192])

    def test_ask_with_tools_emits_tool_call_event(self) -> None:
        events: list[tuple[str, str]] = []
        results: list[tuple[str, int, str, str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello", encoding="utf-8")
            backend = ToolCallingBackend(tool_name="exec_shell_full_command", arguments="{\"command\":\"cat note.txt\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            runtime.ask_with_tools(
                "read note",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                on_tool_call=lambda name, args: events.append((name, args)),
                on_tool_result=lambda name, chars, source, content: results.append((name, chars, source, content)),
            )

        self.assertEqual(events, [("exec_shell_full_command", "{\"command\":\"cat note.txt\"}")])
        self.assertEqual(results, [("exec_shell_full_command", 5, "orbit", "hello")])

    def test_ask_with_tools_emits_model_step_metrics(self) -> None:
        steps = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello", encoding="utf-8")
            backend = ToolCallingBackend(tool_name="exec_shell_full_command", arguments="{\"command\":\"cat note.txt\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            runtime.ask_with_tools(
                "read note",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                on_model_step=steps.append,
            )

        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].loop, 1)
        self.assertEqual(steps[0].phase, "tool_call")
        self.assertEqual(steps[0].cached_tokens, 8)
        self.assertEqual(steps[0].tool_calls, 1)
        self.assertEqual(steps[-1].phase, "final_from_tool")
        self.assertEqual(steps[-1].cached_tokens, 10)

    def test_ask_with_tools_stops_repeated_tool_call(self) -> None:
        class RepeatingToolBackend:
            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                return ChatResult(
                    content="",
                    model="fake",
                    finish_reason="tool_calls",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "list_files", "arguments": "{\"path\":\".\"}"},
                        }
                    ],
                    prompt_tokens=None,
                    completion_tokens=None,
                    cached_tokens=None,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        runtime = ChatRuntime(backend=RepeatingToolBackend(), system_prompt=None)

        result = runtime.ask_with_tools("list", temperature=0, max_tokens=32, workdir=Path("."))

        self.assertEqual(result.finish_reason, "tool_calls")

    def test_ask_with_tools_only_executes_allowed_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            backend = ToolCallingBackend(tool_name="read_file", arguments="{\"path\":\"note.txt\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools(
                "read note",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                tool_names=("exec_shell_full_command",),
            )

            self.assertFalse((workdir / "note.txt").exists())

        self.assertEqual(result.content, "done")
        tool_messages = [message for message in backend.messages if message.get("role") == "tool"]
        self.assertTrue(tool_messages)
        self.assertIn("tool not available for this turn: read_file", tool_messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
