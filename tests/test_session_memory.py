from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult, Message
from orbit.runtime.session_memory import MEMORY_MARKER, SUMMARY_MAX_TOKENS, maybe_refresh_memory, rebuild_with_memory


class MemoryBackend:
    def __init__(self, content: str = "User wants file inspection. README.md was listed.") -> None:
        self.content = content
        self.calls = 0
        self.last_messages: list[Message] = []

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.calls += 1
        self.last_messages = messages
        return ChatResult(
            content=self.content,
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )


class ToolCallingMemoryBackend(MemoryBackend):
    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.calls += 1
        return ChatResult(
            content="",
            model="fake",
            finish_reason="tool_calls",
            tool_calls=[{"id": "bad", "function": {"name": "list_files", "arguments": "{}"}}],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )


class SessionMemoryTests(unittest.TestCase):
    def test_memory_refresh_is_model_generated_and_rebuilds_history(self) -> None:
        messages: list[Message] = [{"role": "system", "content": "system"}]
        for index in range(80):
            messages.append({"role": "user", "content": f"question {index} " + ("x" * 80)})
            messages.append({"role": "assistant", "content": f"answer {index} " + ("y" * 80)})
        backend = MemoryBackend()

        result = maybe_refresh_memory(
            messages,
            backend=backend,
            context_tokens=1000,
            temperature=0,
        )

        self.assertTrue(result.changed)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(messages[0]["content"], "system")
        self.assertIn(MEMORY_MARKER, messages[1]["content"])
        self.assertLess(result.estimated_tokens_after, result.estimated_tokens_before)
        self.assertEqual(result.context_tokens, 1000)
        self.assertEqual(result.threshold_tokens, 850)
        self.assertGreaterEqual(result.elapsed_seconds, 0)

    def test_memory_refresh_does_not_store_internal_request(self) -> None:
        messages: list[Message] = [{"role": "system", "content": "system"}, {"role": "user", "content": "x" * 4000}]
        backend = MemoryBackend()

        maybe_refresh_memory(messages, backend=backend, context_tokens=1000, temperature=0)

        rendered = "\n".join(str(message.get("content", "")) for message in messages)
        self.assertNotIn("Create a concise durable session memory", rendered)

    def test_memory_generation_ignores_system_prompt_transcript(self) -> None:
        messages: list[Message] = [
            {"role": "system", "content": "system-only rule: always mention SYSTEM_SECRET"},
            {"role": "user", "content": "Remember user constraint C01 and inspected file alpha.txt."},
            {"role": "assistant", "content": "ok"},
        ]
        backend = MemoryBackend(content="User constraint C01. alpha.txt was inspected.")

        maybe_refresh_memory(messages, backend=backend, context_tokens=100, temperature=0, force=True)

        transcript = backend.last_messages[0]["content"]
        self.assertNotIn("SYSTEM_SECRET", transcript)
        self.assertIn("constraint C01", transcript)

    def test_memory_summary_budget_is_bounded_for_cpu_only_runtime(self) -> None:
        messages: list[Message] = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Remember user constraint C01." * 200},
        ]
        backend = MemoryBackend()

        maybe_refresh_memory(messages, backend=backend, context_tokens=100, temperature=0, force=True)

        self.assertEqual(SUMMARY_MAX_TOKENS, 256)

    def test_memory_refresh_is_noop_if_model_tries_tool_call(self) -> None:
        messages: list[Message] = [{"role": "system", "content": "system"}, {"role": "user", "content": "x" * 4000}]
        original = list(messages)

        result = maybe_refresh_memory(
            messages,
            backend=ToolCallingMemoryBackend(),
            context_tokens=1000,
            temperature=0,
        )

        self.assertFalse(result.changed)
        self.assertEqual(messages, original)

    def test_rebuild_keeps_recent_tail_from_user_boundary(self) -> None:
        messages: list[Message] = [{"role": "system", "content": "system"}]
        for index in range(20):
            messages.append({"role": "assistant", "content": f"old answer {index}"})
            messages.append({"role": "user", "content": f"recent question {index}"})

        rebuilt = rebuild_with_memory(messages, summary="durable state", context_tokens=600)

        self.assertEqual(rebuilt[0]["role"], "system")
        self.assertIn(MEMORY_MARKER, rebuilt[1]["content"])
        self.assertEqual(rebuilt[2]["role"], "user")

    def test_rebuild_drops_orphan_assistant_tail_without_user_boundary(self) -> None:
        messages: list[Message] = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old user " + ("x" * 1000)},
            {"role": "assistant", "content": "short orphan"},
        ]

        rebuilt = rebuild_with_memory(messages, summary="durable state", context_tokens=200)

        self.assertEqual(len(rebuilt), 2)
        self.assertIn(MEMORY_MARKER, rebuilt[1]["content"])

    def test_memory_generation_skips_recent_tail_that_will_be_kept(self) -> None:
        messages: list[Message] = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old durable fact " + ("x" * 1000)},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]
        backend = MemoryBackend()

        maybe_refresh_memory(messages, backend=backend, context_tokens=400, temperature=0, force=True)

        transcript = backend.last_messages[0]["content"]
        self.assertIn("old durable fact", transcript)
        self.assertNotIn("recent question", transcript)
        self.assertNotIn("recent answer", transcript)


if __name__ == "__main__":
    unittest.main()
