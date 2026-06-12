from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult
from orbit.runtime.final_policy import (
    build_final_tool_policy,
    final_from_tool_retry_reason,
    final_tool_retry_instruction,
    has_list_like_tool_result,
)


class FinalPolicyTests(unittest.TestCase):
    def test_list_like_policy_uses_compact_names_instruction(self) -> None:
        messages = [{"role": "tool", "name": "list_files", "content": "a\nb"}]

        policy = build_final_tool_policy(messages, max_tokens=512, streamed=False)

        self.assertEqual(policy.max_tokens, 96)
        self.assertIn("Return only the listed names", policy.messages[-1]["content"])

    def test_fetch_policy_caps_tokens_and_allows_length_retry_when_not_streamed(self) -> None:
        messages = [{"role": "tool", "name": "fetch_url", "content": "content"}]

        policy = build_final_tool_policy(messages, max_tokens=512, streamed=False)

        self.assertEqual(policy.max_tokens, 72)
        self.assertTrue(policy.length_retry_allowed)
        self.assertTrue(policy.web_fetch_result)
        self.assertIn("Write exactly two concise bullets", policy.messages[-1]["content"])

    def test_fetch_policy_disables_length_retry_when_streamed(self) -> None:
        messages = [{"role": "tool", "name": "fetch_url", "content": "content"}]

        policy = build_final_tool_policy(messages, max_tokens=512, streamed=True)

        self.assertFalse(policy.length_retry_allowed)

    def test_shell_list_command_is_detected_as_list_like(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "exec_shell_command",
                            "arguments": {"command": "ls -F"},
                        }
                    }
                ],
            },
            {"role": "tool", "name": "exec_shell_command", "content": "a\nb"},
        ]

        self.assertTrue(has_list_like_tool_result(messages))

    def test_final_retry_reason_detects_raw_tool_call(self) -> None:
        result = ChatResult(
            content="<|tool_call>call:x{}<tool_call|>",
            model="m",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

        reason = final_from_tool_retry_reason(result, length_retry_allowed=False)

        self.assertEqual(reason, "raw_tool_call")

    def test_final_retry_reason_detects_empty_length_even_when_length_retry_disabled(self) -> None:
        result = ChatResult(
            content="",
            model="m",
            finish_reason="length",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

        reason = final_from_tool_retry_reason(result, length_retry_allowed=False)

        self.assertEqual(reason, "empty_length")

    def test_final_tool_retry_instruction_is_unchanged(self) -> None:
        self.assertEqual(
            final_tool_retry_instruction()["content"],
            "Do not call tools. Provide a shorter final answer from the available tool result now.",
        )


if __name__ == "__main__":
    unittest.main()
