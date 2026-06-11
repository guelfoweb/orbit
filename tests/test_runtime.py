from __future__ import annotations

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
from orbit.runtime.messages import FINAL_FROM_TOOL_SYSTEM_PROMPT, MEDIA_SYSTEM_PROMPT
from orbit.runtime.chat import _has_list_like_tool_result
from orbit.runtime.media import AudioInput, ImageInput


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
                            "name": "exec_shell_command",
                            "arguments": {"command": "wc -l text/summary.txt"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "exec_shell_command",
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
                        "name": "exec_shell_command",
                        "function": {
                            "name": "exec_shell_command",
                            "arguments": {"command": "ls -F"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "exec_shell_command",
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


class ToolCallingBackend:
    def __init__(self, tool_name: str = "list_files", arguments: str = "{\"path\":\".\"}") -> None:
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


class ToolRuntimeTests(unittest.TestCase):
    def test_ask_auto_returns_chat_without_tools(self) -> None:
        class ChatOnlyBackend:
            def __init__(self) -> None:
                self.tools_seen: list[object] = []
                self.messages_seen: list[list[Message]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.tools_seen.append(tools)
                self.messages_seen.append(messages)
                content = '{"_route":"CHAT"}' if len(self.messages_seen) == 1 else "chat answer"
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
        self.assertEqual(backend.tools_seen, [None, None])
        self.assertIn("Classify the latest user request", backend.messages_seen[0][0]["content"])
        self.assertIn("Do not emit route JSON", backend.messages_seen[1][0]["content"])
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
        self.assertEqual(backend.chat_max_tokens, [64])
        self.assertEqual(backend.stream_max_tokens, [512])

    def test_ask_auto_retries_empty_chat_response_once(self) -> None:
        class EmptyThenAnswerBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content='{"_route":"CHAT"}',
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
                if self.calls == 1:
                    content = '{"_route":"CHAT"}'
                    completion_tokens = 1
                else:
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
                    content = '{"_route":"FILESYSTEM"}'
                    completion_tokens = 7
                elif self.calls in {2, 3}:
                    content = ""
                    completion_tokens = 0
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
        self.assertEqual(backend.calls, 4)
        self.assertIsNone(backend.tools_seen[0])
        self.assertIsNotNone(backend.tools_seen[1])
        self.assertIsNotNone(backend.tools_seen[2])
        self.assertIsNone(backend.tools_seen[3])

    def test_ask_auto_converts_generic_route_tool_call_inside_tool_loop(self) -> None:
        class GenericRouteToolCallBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content='{"_route":"FILESYSTEM"}',
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
                                    "arguments": '{"_route": "FILESYSTEM", "tool": "list_files"}',
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
        self.assertEqual(tool_messages[0]["name"], "list_files")
        self.assertIn("note.txt", tool_messages[0]["content"])

    def test_ask_auto_routes_to_filesystem_tools(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tool_names_seen: list[tuple[str, ...]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tool_names_seen.append(tuple(tool["function"]["name"] for tool in tools or []))
                if self.calls == 1:
                    return ChatResult(
                        content='{"_route":"FILESYSTEM"}',
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
        self.assertEqual(backend.tool_names_seen[1], ("list_files", "read_file", "exec_shell_command"))
        self.assertNotIn('{"_route":"FILESYSTEM"}', [message.get("content") for message in runtime.messages])

    def test_ask_auto_uses_preferred_route_tool_when_valid(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tool_names_seen: list[tuple[str, ...]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tool_names_seen.append(tuple(tool["function"]["name"] for tool in tools or []))
                if self.calls == 1:
                    return ChatResult(
                        content='{"_route":"FILESYSTEM","tool":"read_file"}',
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
                                "function": {"name": "read_file", "arguments": "{\"path\":\"note.txt\"}"},
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
        self.assertEqual(backend.tool_names_seen[1], ("read_file",))

    def test_ask_auto_executes_route_json_with_arguments_without_tool_selection_round(self) -> None:
        class RoutedBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.tool_names_seen: list[tuple[str, ...]] = []

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                self.tool_names_seen.append(tuple(tool["function"]["name"] for tool in tools or []))
                if self.calls == 1:
                    return ChatResult(
                        content='{"_route":"FILESYSTEM","tool":"read_file","path":"note.txt"}',
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

    def test_ask_auto_media_route_without_path_returns_clear_error(self) -> None:
        class MediaRouteBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                return ChatResult(
                    content='{"_route":"MEDIA"}',
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

        self.assertIn("no local image/audio path", result.content)
        self.assertEqual(backend.calls, 1)

    def test_ask_auto_attaches_referenced_media_without_route(self) -> None:
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
        self.assertEqual(backend.calls, 2)
        self.assertEqual(backend.messages[-2]["role"], "tool")
        self.assertIn("file.txt", backend.messages[-2]["content"])

    def test_ask_with_tools_can_read_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello from file", encoding="utf-8")
            backend = ToolCallingBackend(tool_name="read_file", arguments="{\"path\":\"note.txt\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("read note.txt", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.messages[-2]["role"], "tool")
        self.assertIn("hello from file", backend.messages[-2]["content"])

    def test_ask_with_tools_can_stat_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello", encoding="utf-8")
            backend = ToolCallingBackend(tool_name="stat_path", arguments="{\"path\":\"note.txt\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("stat note.txt", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.messages[-2]["role"], "tool")
        self.assertIn("type: file", backend.messages[-2]["content"])
        self.assertIn("size_bytes: 5", backend.messages[-2]["content"])

    def test_ask_with_tools_can_reinject_fetch_url_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            backend = ToolCallingBackend(tool_name="fetch_url", arguments="{\"url\":\"ftp://example.test\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("fetch a URL", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.messages[-2]["role"], "tool")
        self.assertIn("http/https", backend.messages[-2]["content"])

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
                                "function": {"name": "fetch_url", "arguments": "{\"url\":\"ftp://example.test\"}"},
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
                tool_names=("fetch_url",),
            )

        self.assertEqual(result.content, "short web synthesis")
        self.assertEqual(backend.max_tokens_seen, [96, 72])

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
                                "function": {"name": "exec_shell_command", "arguments": "{\"command\":\"lscpu\"}"},
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
                        content='<|tool_call>call:exec_shell_command{"command":"grep cpu /proc/cpuinfo"}<tool_call|>',
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
                tool_names=("exec_shell_command",),
                on_model_step=steps.append,
            )

        self.assertEqual(result.content, "CPU information from tool result")
        self.assertEqual(backend.calls, 3)
        self.assertEqual(steps[-1].phase, "final_from_tool_retry")
        self.assertEqual(steps[-1].retry_reason, "raw_tool_call")

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
                                "function": {"name": "exec_shell_command", "arguments": "{\"command\":\"pwd\"}"},
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
                                "function": {"name": "exec_shell_command", "arguments": "{\"command\":\"ls\"}"},
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
                tool_names=("exec_shell_command",),
                on_model_step=steps.append,
            )

        self.assertEqual(result.content, "final answer from first tool result")
        self.assertEqual(backend.calls, 3)
        self.assertEqual(steps[-1].phase, "final_from_tool_retry")
        self.assertEqual(steps[-1].retry_reason, "tool_call_in_final")

    def test_final_from_tool_uses_final_prompt_not_tool_call_prompt(self) -> None:
        class PromptCaptureBackend(ToolCallingBackend):
            def __init__(self) -> None:
                super().__init__(tool_name="read_file", arguments="{\"path\":\"note.txt\"}")
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

            runtime.ask_with_tools("read note", temperature=0, max_tokens=32, workdir=workdir, tool_names=("read_file",))

        self.assertEqual(backend.system_prompts[-1], FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertNotIn("When tools are available", backend.system_prompts[-1])

    def test_ask_with_tools_can_write_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            backend = ToolCallingBackend(tool_name="write_file", arguments="{\"path\":\"note.txt\",\"content\":\"hello\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("create note.txt", temperature=0, max_tokens=32, workdir=workdir)

            self.assertEqual((workdir / "note.txt").read_text(encoding="utf-8"), "hello")

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.messages[-2]["role"], "tool")
        self.assertIn("created: true", backend.messages[-2]["content"])

    def test_ask_with_tools_can_append_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("first\n", encoding="utf-8")
            backend = ToolCallingBackend(tool_name="append_file", arguments="{\"path\":\"note.txt\",\"content\":\"second\\n\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("append note.txt", temperature=0, max_tokens=32, workdir=workdir)

            self.assertEqual((workdir / "note.txt").read_text(encoding="utf-8"), "first\nsecond\n")

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.messages[-2]["role"], "tool")
        self.assertIn("appended: true", backend.messages[-2]["content"])

    def test_ask_with_tools_can_replace_unique_text_in_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello old world\n", encoding="utf-8")
            backend = ToolCallingBackend(tool_name="replace_in_file", arguments="{\"path\":\"note.txt\",\"old\":\"old\",\"new\":\"new\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("replace old", temperature=0, max_tokens=32, workdir=workdir)

            self.assertEqual((workdir / "note.txt").read_text(encoding="utf-8"), "hello new world\n")

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.messages[-2]["role"], "tool")
        self.assertIn("replaced: true", backend.messages[-2]["content"])

    def test_ask_with_tools_can_read_file_chunk_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "large.txt").write_text("abcdef" * 50000, encoding="utf-8")
            backend = ToolCallingBackend(tool_name="read_file", arguments="{\"path\":\"large.txt\",\"chunk_index\":1,\"chunk_chars\":2}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools("read chunk", temperature=0, max_tokens=32, workdir=workdir)

        self.assertEqual(result.content, "done")
        self.assertEqual(backend.messages[-2]["role"], "tool")
        self.assertIn("chunk_index: 1", backend.messages[-2]["content"])

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
                                "function": {"name": "read_file", "arguments": "{\"path\":\"note.txt\"}"},
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

    def test_ask_with_tools_emits_tool_call_event(self) -> None:
        events: list[tuple[str, str]] = []
        results: list[tuple[str, int, str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello", encoding="utf-8")
            backend = ToolCallingBackend(tool_name="read_file", arguments="{\"path\":\"note.txt\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            runtime.ask_with_tools(
                "read note",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                on_tool_call=lambda name, args: events.append((name, args)),
                on_tool_result=lambda name, chars, source: results.append((name, chars, source)),
            )

        self.assertEqual(events, [("read_file", "{\"path\":\"note.txt\"}")])
        self.assertEqual(results, [("read_file", 5, "orbit")])

    def test_ask_with_tools_emits_model_step_metrics(self) -> None:
        steps = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello", encoding="utf-8")
            backend = ToolCallingBackend(tool_name="read_file", arguments="{\"path\":\"note.txt\"}")
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
        self.assertEqual(steps[1].loop, 2)
        self.assertEqual(steps[1].phase, "final")
        self.assertEqual(steps[1].cached_tokens, 10)

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
            backend = ToolCallingBackend(tool_name="write_file", arguments="{\"path\":\"note.txt\",\"content\":\"hello\"}")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_with_tools(
                "read note",
                temperature=0,
                max_tokens=32,
                workdir=workdir,
                tool_names=("list_files", "read_file", "stat_path"),
            )

            self.assertFalse((workdir / "note.txt").exists())

        self.assertEqual(result.content, "done")
        tool_messages = [message for message in backend.messages if message.get("role") == "tool"]
        self.assertTrue(tool_messages)
        self.assertIn("tool not available for this turn: write_file", tool_messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
