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
from orbit.runtime.media import AudioInput, ImageInput


class FakeBackend:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.calls = 0

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.calls += 1
        self.messages = messages
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
        self.assertEqual(steps[0].phase, "final")
        self.assertEqual(steps[0].prompt_tokens, 1)
        self.assertEqual(steps[0].cached_tokens, 0)


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
        results: list[tuple[str, int]] = []
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
                on_tool_result=lambda name, chars: results.append((name, chars)),
            )

        self.assertEqual(events, [("read_file", "{\"path\":\"note.txt\"}")])
        self.assertEqual(results, [("read_file", 5)])

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

        self.assertEqual(result.finish_reason, "repeated_tool_call")
        self.assertIn("repeated tool call", result.content)


if __name__ == "__main__":
    unittest.main()
