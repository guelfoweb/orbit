from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult, Message
from orbit.runtime import ChatRuntime
from orbit.runtime.messages import FINAL_FROM_TOOL_SYSTEM_PROMPT, MEDIA_SYSTEM_PROMPT, TOOL_CALL_JSON_RETRY_PROMPT, TOOL_CALL_SYSTEM_PROMPT
from orbit.runtime.chat import _has_list_like_tool_result
from orbit.runtime.media import AudioInput, ImageInput
from orbit.runtime.shell_guardrails import SHELL_FULL_EMPTY_RESULT_CHECK_PROMPT, should_verify_shell_mutation


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


class RuntimeTests(unittest.TestCase):
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


def _last_tool_message(runtime: ChatRuntime) -> Message:
    tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
    assert tool_messages
    return tool_messages[-1]


class ToolRuntimeTests(unittest.TestCase):
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

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None) -> ChatResult:
                raise AssertionError("short probe answer should not need streaming retry")

        emitted: list[str] = []
        backend = StreamingChatBackend()
        runtime = ChatRuntime(backend=backend, system_prompt=None)

        result = runtime.ask_auto("hello", temperature=0, max_tokens=512, workdir=Path("."), on_final_delta=emitted.append)

        self.assertEqual(result.content, "chat answer")
        self.assertEqual(emitted, ["chat answer"])
        self.assertEqual(backend.max_tokens_seen, [128])
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

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None) -> ChatResult:
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
        self.assertEqual(backend.chat_max_tokens, [128])
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
        self.assertEqual(backend.calls, 5)
        self.assertEqual(backend.tools_seen[0], None)
        self.assertIsNotNone(backend.tools_seen[1])
        self.assertIsNotNone(backend.tools_seen[2])
        self.assertIsNotNone(backend.tools_seen[3])
        self.assertEqual(backend.tools_seen[4], None)
        self.assertIsNotNone(backend.second_call_last_message)
        self.assertIn("content/source/string evidence", backend.second_call_last_message["content"])
        self.assertIsNotNone(backend.third_call_last_message)
        self.assertIn("Return only the tool call", backend.third_call_last_message["content"])

    def test_shell_full_failed_initial_command_gets_one_model_retry(self) -> None:
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
                        content=json.dumps({"command": "sed -i 's/<title>.*</title>/<title>test</title>/' index.html"}),
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
                                    "arguments": json.dumps(
                                        {"command": "perl -0pi -e 's|<title>.*?</title>|<title>test</title>|s' index.html"}
                                    ),
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
                    content="title updated",
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
            (workdir / "index.html").write_text("<html><head><title>\nOld\n</title></head></html>\n", encoding="utf-8")
            backend = FailedShellCommandBackend()
            runtime = ChatRuntime(backend=backend, system_prompt="route system")

            result = runtime.ask_auto(
                'sostituisci il title della pagina con "test"',
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

            content = (workdir / "index.html").read_text(encoding="utf-8")

        self.assertEqual(result.content, "title updated")
        self.assertEqual(backend.calls, 4)
        self.assertIsNone(backend.tools_seen[0])
        self.assertIsNotNone(backend.tools_seen[1])
        self.assertIsNotNone(backend.tools_seen[2])
        self.assertIsNone(backend.tools_seen[3])
        retry_prompt = backend.messages_seen[1][-1]["content"]
        self.assertIn("The previous shell command failed.", retry_prompt)
        self.assertIn("Exit code:", retry_prompt)
        self.assertIn("STDOUT:", retry_prompt)
        self.assertIn("STDERR:", retry_prompt)
        self.assertIn('{"command":"..."}', retry_prompt)
        self.assertIn("<title>test</title>", content)

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

    def test_shell_full_generic_nonzero_error_gets_repair_retry(self) -> None:
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
                    return ChatResult(
                        content=json.dumps({"command": "printf repaired"}),
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
                    content="repaired",
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

        self.assertEqual(result.content, "repaired")
        self.assertEqual(backend.calls, 4)
        self.assertIsNone(backend.tools_seen[0])
        self.assertIsNotNone(backend.tools_seen[1])
        self.assertIsNotNone(backend.tools_seen[2])
        self.assertIsNone(backend.tools_seen[3])
        retry_prompt = backend.messages_seen[1][-1]["content"]
        self.assertIn("Exit code: 2", retry_prompt)
        self.assertIn("custom parser exploded at token 7", retry_prompt)

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
        self.assertEqual(backend.calls, 4)
        self.assertIsNotNone(backend.tools_seen[0])
        self.assertIsNotNone(backend.tools_seen[1])
        self.assertIsNotNone(backend.tools_seen[2])
        self.assertIsNone(backend.tools_seen[3])
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 2)

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
        self.assertEqual(backend.calls, 4)
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

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None) -> ChatResult:
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

            def chat_stream(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None, on_delta=None) -> ChatResult:
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
        self.assertEqual(backend.calls, 3)

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

        self.assertEqual(len(steps), 3)
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
