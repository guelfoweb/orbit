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
    is_operational_status_request,
)


class FinalPolicyTests(unittest.TestCase):
    def test_list_like_policy_uses_compact_names_instruction(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "exec_shell_full_command", "arguments": {"command": "ls -F"}}}],
            },
            {"role": "tool", "name": "exec_shell_full_command", "content": "a\nb"},
        ]

        policy = build_final_tool_policy(messages, max_tokens=512, streamed=False)

        self.assertEqual(policy.max_tokens, 96)
        self.assertIn("Return only the listed names", policy.messages[-1]["content"])

    def test_html_cleaned_policy_caps_tokens_and_allows_length_retry_when_not_streamed(self) -> None:
        messages = [{"role": "tool", "name": "exec_shell_full_command", "content": "shell_output_html_cleaned: true\ntext:\ncontent"}]

        policy = build_final_tool_policy(messages, max_tokens=512, streamed=False)

        self.assertEqual(policy.max_tokens, 72)
        self.assertTrue(policy.length_retry_allowed)
        self.assertTrue(policy.web_fetch_result)
        self.assertIn("Write exactly two concise bullets", policy.messages[-1]["content"])

    def test_html_cleaned_policy_disables_length_retry_when_streamed(self) -> None:
        messages = [{"role": "tool", "name": "exec_shell_full_command", "content": "shell_output_html_cleaned: true\ntext:\ncontent"}]

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
                            "name": "exec_shell_full_command",
                            "arguments": {"command": "ls -F"},
                        }
                    }
                ],
            },
            {"role": "tool", "name": "exec_shell_full_command", "content": "a\nb"},
        ]

        self.assertTrue(has_list_like_tool_result(messages))

    def test_shell_list_command_is_detected_with_serialized_arguments(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": '{"command":"ls -F"}',
                        }
                    }
                ],
            },
            {"role": "tool", "name": "exec_shell_full_command", "content": "a\nb"},
        ]

        self.assertTrue(has_list_like_tool_result(messages))

    def test_shell_full_policy_answers_latest_request_directly(self) -> None:
        messages = [
            {"role": "user", "content": "Use the shell output and answer with the exact first line only."},
            {"role": "tool", "name": "exec_shell_full_command", "content": "first line\nsecond line"},
        ]

        policy = build_final_tool_policy(messages, max_tokens=512, streamed=False)

        self.assertIn("shell-full output", policy.messages[-1]["content"])
        self.assertIn("latest user request directly", policy.messages[-1]["content"])
        self.assertIn("most recent relevant shell result", policy.messages[-1]["content"])
        self.assertIn("Do not call tools again", policy.messages[-1]["content"])
        self.assertIn("If the evidence is insufficient", policy.messages[-1]["content"])

    def test_operational_status_policy_prefers_recent_shell_evidence(self) -> None:
        messages = [
            {"role": "user", "content": "analyze index.html"},
            {"role": "tool", "name": "exec_shell_full_command", "content": "old noisy index.html analysis\n<title>Example</title>"},
            {"role": "assistant", "content": "old summary"},
            {"role": "user", "content": "is the new file saved? what was it renamed?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": '{"command":"ls -F cleaned_index.html"}',
                        }
                    }
                ],
            },
            {"role": "tool", "name": "exec_shell_full_command", "content": "cleaned_index.html"},
        ]

        policy = build_final_tool_policy(messages, max_tokens=512, streamed=False)

        self.assertEqual(policy.max_tokens, 96)
        self.assertIn("latest operational/status question", policy.messages[-1]["content"])
        self.assertIn("most recent relevant shell output", policy.messages[-1]["content"])
        self.assertIn("Ignore older tool results", policy.messages[-1]["content"])
        self.assertIn("Do not summarize file or page content", policy.messages[-1]["content"])

    def test_operational_status_policy_preserves_remove_confirmation(self) -> None:
        messages = [
            {"role": "user", "content": "remove index.html"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": {"command": "rm index.html && ls index.html"},
                        }
                    }
                ],
            },
            {"role": "tool", "name": "exec_shell_full_command", "content": "shell_command_failed: true\nexit_code: 2\nSTDOUT:\n(empty)\nSTDERR:\nls: cannot access 'index.html': No such file or directory"},
        ]

        policy = build_final_tool_policy(messages, max_tokens=512, streamed=False)

        self.assertIn("latest operational/status question", policy.messages[-1]["content"])
        self.assertIn("If recent evidence is insufficient", policy.messages[-1]["content"])

    def test_content_request_keeps_normal_shell_full_policy(self) -> None:
        messages = [
            {"role": "user", "content": "summarize cleaned_index.html"},
            {"role": "tool", "name": "exec_shell_full_command", "content": "<html><body>content</body></html>"},
        ]

        policy = build_final_tool_policy(messages, max_tokens=512, streamed=False)

        self.assertIn("shell-full output", policy.messages[-1]["content"])
        self.assertNotIn("latest operational/status question", policy.messages[-1]["content"])

    def test_operational_status_detector_excludes_explicit_content_requests(self) -> None:
        self.assertTrue(is_operational_status_request("is the new file saved? what was it renamed?"))
        self.assertTrue(is_operational_status_request("remove index.html"))
        self.assertFalse(is_operational_status_request("summarize cleaned_index.html"))
        self.assertFalse(is_operational_status_request("what is in cleaned_index.html?"))

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
