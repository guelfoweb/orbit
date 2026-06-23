from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult
from orbit.runtime.final_policy import (
    LONG_SHELL_ANALYSIS_FINAL_MAX_TOKENS,
    classify_final_answer_completeness,
    build_final_tool_policy,
    final_from_tool_compact_retry_reason,
    final_tool_compact_retry_max_tokens,
    final_from_tool_retry_reason,
    final_tool_compact_retry_instruction,
    final_tool_retry_instruction,
    has_list_like_tool_result,
    has_pdf_text_tool_result,
    is_repetitive_final_answer,
    is_compact_list_request,
    prepare_final_tool_messages,
    is_list_shell_command,
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
        self.assertFalse(policy.incomplete_retry_allowed)
        self.assertTrue(policy.web_fetch_result)
        self.assertIn("Write exactly two concise bullets", policy.messages[-1]["content"])

    def test_html_cleaned_policy_disables_length_retry_when_streamed(self) -> None:
        messages = [{"role": "tool", "name": "exec_shell_full_command", "content": "shell_output_html_cleaned: true\ntext:\ncontent"}]

        policy = build_final_tool_policy(messages, max_tokens=512, streamed=True)

        self.assertFalse(policy.length_retry_allowed)
        self.assertFalse(policy.incomplete_retry_allowed)

    def test_pdf_chunk_result_uses_large_excerpt_policy(self) -> None:
        messages = [
            {
                "role": "user",
                "content": 'Leggi il PDF "report.pdf" e fammi una sintesi.',
            },
            {
                "role": "tool",
                "name": "exec_shell_full_command",
                "content": (
                    "shell_output_pdf_text: true\n"
                    "path: report.pdf\n"
                    "extractor: pdftotext\n"
                    "chunk_index: 1\n"
                    "total_chunks: 4\n"
                    "chars: 3000-6000 of 9000\n"
                    "content:\nchunk one"
                ),
            }
        ]

        policy = build_final_tool_policy(messages, max_tokens=512, streamed=False)

        self.assertEqual(policy.max_tokens, 128)
        self.assertTrue(policy.length_retry_allowed)
        self.assertIn("Use only the already inspected chunk(s) or excerpt(s)", policy.messages[-1]["content"])

    def test_exhaustive_pdf_chunk_policy_allows_larger_final_budget(self) -> None:
        messages = [
            {
                "role": "user",
                "content": 'Analizza intero documento PDF "report.pdf" e fammi una sintesi dettagliata.',
            },
            {
                "role": "tool",
                "name": "exec_shell_full_command",
                "content": (
                    "shell_output_pdf_text: true\n"
                    "path: report.pdf\n"
                    "extractor: pdftotext\n"
                    "chunk_index: 2\n"
                    "total_chunks: 4\n"
                    "chars: 6000-9000 of 9000\n"
                    "content:\nchunk two"
                ),
            },
        ]

        policy = build_final_tool_policy(messages, max_tokens=512, streamed=False)

        self.assertEqual(policy.max_tokens, 256)
        self.assertTrue(policy.length_retry_allowed)
        self.assertIn("fuller synthesis", policy.messages[-1]["content"])

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

    def test_find_exec_cat_is_not_treated_as_listing(self) -> None:
        self.assertFalse(is_list_shell_command('find . -name "vulnerable_service.py" -exec cat {} +'))

    def test_rejected_ls_after_pdf_excerpt_is_not_treated_as_list_like(self) -> None:
        messages = [
            {"role": "user", "content": 'Read pdf/small.pdf and summarize the document topic in one concise sentence.'},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-pdf",
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": {"command": "pdftotext pdf/small.pdf - | sed -n '1,10p'"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-pdf",
                "name": "exec_shell_full_command",
                "content": "shell_output_pdf_text: true\npath: pdf/small.pdf\nextractor: pdftotext\nA short PDF about safety.",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-ls",
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": {"command": "ls -R pdf/"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-ls",
                "name": "exec_shell_full_command",
                "content": "error: shell-full analysis requests require content/source/string evidence",
            },
        ]

        policy = build_final_tool_policy(messages, max_tokens=128, streamed=False)

        self.assertFalse(has_list_like_tool_result(messages))
        self.assertFalse(is_compact_list_request(messages[0]["content"]))
        self.assertNotIn("Return only the listed names", policy.messages[-1]["content"])
        self.assertIn("PDF text extraction already succeeded", policy.messages[-1]["content"])

    def test_generic_recursive_listing_request_does_not_force_compact_names_mode(self) -> None:
        messages = [
            {"role": "user", "content": "List all files and directories in this workdir, including subdirectories."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-find",
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": {"command": "find . -maxdepth 10 -not -path '*/.*'"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-find", "name": "exec_shell_full_command", "content": ".\n./pdf\n./text"},
        ]

        policy = build_final_tool_policy(messages, max_tokens=128, streamed=False)

        self.assertTrue(has_list_like_tool_result(messages))
        self.assertFalse(is_compact_list_request(messages[0]["content"]))
        self.assertNotIn("Return only the listed names", policy.messages[-1]["content"])

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
        self.assertTrue(policy.incomplete_retry_allowed)

    def test_long_shell_analysis_policy_is_compact_and_bounded(self) -> None:
        messages = [
            {"role": "user", "content": "inspect vulnerable_service.py and explain the vulnerabilities"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": {"command": "find . -name \"vulnerable_service.py\" -exec cat {} +"},
                        }
                    }
                ],
            },
            {"role": "tool", "name": "exec_shell_full_command", "content": "x" * 1600},
        ]

        policy = build_final_tool_policy(messages, max_tokens=160, streamed=False)

        self.assertEqual(policy.max_tokens, LONG_SHELL_ANALYSIS_FINAL_MAX_TOKENS)
        self.assertIn("latest relevant shell-full output", policy.messages[-1]["content"])
        self.assertIn("exactly 4 short bullets", policy.messages[-1]["content"])
        self.assertIn("'- Finding: ... Fix: ...'", policy.messages[-1]["content"])
        self.assertIn("brief remediation", policy.messages[-1]["content"])
        self.assertEqual(final_tool_compact_retry_max_tokens(512, messages=policy.messages), 160)

    def test_long_shell_web_info_policy_stays_natural(self) -> None:
        messages = [
            {"role": "user", "content": "search online for information about Mario Nobile"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": {"command": "orbit-web-search 'Mario Nobile'"},
                        }
                    }
                ],
            },
            {"role": "tool", "name": "exec_shell_full_command", "content": "x" * 1600},
        ]

        policy = build_final_tool_policy(messages, max_tokens=160, streamed=False)

        self.assertNotIn("'- Finding: ... Fix: ...'", policy.messages[-1]["content"])
        self.assertNotIn("exactly 4 short bullets", policy.messages[-1]["content"])
        self.assertIn("Answer the latest user request directly and concisely", policy.messages[-1]["content"])

    def test_compact_retry_max_tokens_stays_default_for_non_shell_policy(self) -> None:
        messages = [
            {"role": "user", "content": 'Leggi il PDF "pdf/small.pdf" e fammi una sintesi dettagliata.'},
            {
                "role": "tool",
                "name": "exec_shell_full_command",
                "content": (
                    "shell_output_pdf_text: true\n"
                    "path: pdf/small.pdf\n"
                    "extractor: pdftotext\n"
                    "content:\n"
                    "Questo documento descrive una rete sicura con firewall e monitoraggio."
                ),
            },
        ]

        policy = build_final_tool_policy(messages, max_tokens=256, streamed=False)

        self.assertEqual(final_tool_compact_retry_max_tokens(512, messages=policy.messages), 160)

    def test_pdf_text_policy_treats_file_as_present_and_readable(self) -> None:
        messages = [
            {"role": "user", "content": 'Leggi il PDF "pdf/small.pdf" e fammi una sintesi dettagliata.'},
            {
                "role": "tool",
                "name": "exec_shell_full_command",
                "content": (
                    "shell_output_pdf_text: true\n"
                    "path: pdf/small.pdf\n"
                    "extractor: pdftotext\n"
                    "content:\n"
                    "Questo documento descrive una rete sicura con firewall e monitoraggio."
                ),
            },
        ]

        policy = build_final_tool_policy(messages, max_tokens=256, streamed=False)

        self.assertTrue(has_pdf_text_tool_result(messages))
        self.assertIn("PDF text extraction already succeeded", policy.messages[-1]["content"])
        self.assertIn("Treat the PDF file as present and readable", policy.messages[-1]["content"])
        self.assertIn("Do not claim the file is missing", policy.messages[-1]["content"])

    def test_pdf_text_brief_policy_caps_tokens_and_requests_one_sentence(self) -> None:
        messages = [
            {"role": "user", "content": "Read pdf/grande.pdf and summarize the document topic in one concise sentence."},
            {
                "role": "tool",
                "name": "exec_shell_full_command",
                "content": (
                    "shell_output_pdf_text: true\n"
                    "path: pdf/grande.pdf\n"
                    "extractor: pdftotext\n"
                    "content:\n"
                    + ("A" * 1800)
                ),
            },
        ]

        policy = build_final_tool_policy(messages, max_tokens=256, streamed=False)

        self.assertEqual(policy.max_tokens, 72)
        self.assertIn("exactly one concise sentence", policy.messages[-1]["content"])

    def test_prepare_final_tool_messages_prunes_older_failed_tool_attempts(self) -> None:
        messages = [
            {"role": "user", "content": "Read pdf/piccolo.pdf and summarize it."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "exec_shell_full_command", "arguments": {"command": "pdfflow pdf/piccolo.pdf"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "exec_shell_full_command",
                "content": "error: command not found",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-2",
                        "function": {"name": "exec_shell_full_command", "arguments": {"command": "pdftotext pdf/piccolo.pdf - | head -n 40"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-2",
                "name": "exec_shell_full_command",
                "content": "shell_output_pdf_text: true\npath: pdf/piccolo.pdf\nextractor: pdftotext\ncontent:\nUseful text",
            },
        ]

        prepared = prepare_final_tool_messages(messages)

        self.assertEqual(sum(1 for message in prepared if message.get("role") == "tool"), 1)
        self.assertIn("Useful text", str(prepared[-1].get("content")))
        self.assertNotIn("command not found", json.dumps(prepared, ensure_ascii=False))

    def test_prepare_final_tool_messages_compacts_large_pdf_tool_content(self) -> None:
        messages = [
            {"role": "user", "content": "Read pdf/grande.pdf and summarize the document topic in one concise sentence."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "exec_shell_full_command", "arguments": {"command": "pdftotext pdf/grande.pdf -"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "exec_shell_full_command",
                "content": (
                    "shell_output_pdf_text: true\npath: pdf/grande.pdf\nextractor: pdftotext\ncontent:\n"
                    + ("A" * 1800)
                ),
            },
        ]

        prepared = prepare_final_tool_messages(messages)
        content = str(prepared[-1].get("content"))

        self.assertIn("[output truncated for model context]", content)
        self.assertIn("shell_output_pdf_text: true", content)
        self.assertLess(len(content), 1300)

    def test_prepare_final_tool_messages_leaves_recursive_listing_verbatim(self) -> None:
        content = ".\n" + "\n".join(f"./file-{index}.txt" for index in range(200))
        messages = [
            {"role": "user", "content": "List all files and directories recursively."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "exec_shell_full_command", "arguments": {"command": "find . -maxdepth 10 -not -path '*/.*'"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "exec_shell_full_command", "content": content},
        ]

        prepared = prepare_final_tool_messages(messages)

        self.assertEqual(prepared[-1]["content"], content)

    def test_brief_shell_policy_caps_tokens_and_tightens_prompt(self) -> None:
        messages = [
            {"role": "user", "content": "Read text/summary.txt and summarize it in one concise sentence."},
            {
                "role": "tool",
                "name": "exec_shell_full_command",
                "content": "A concise source excerpt about AI safety and alignment.",
            },
        ]

        policy = build_final_tool_policy(messages, max_tokens=256, streamed=False)

        self.assertEqual(policy.max_tokens, 96)
        self.assertIn("one concise sentence", policy.messages[-1]["content"])

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

    def test_final_retry_reason_does_not_handle_semantic_incomplete_final(self) -> None:
        result = ChatResult(
            content="Il documento e una relazione tecnica per il servizio di gestione della rete QX",
            model="m",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

        reason = final_from_tool_retry_reason(
            result,
            length_retry_allowed=False,
            incomplete_retry_allowed=True,
        )

        self.assertIsNone(reason)

    def test_final_retry_reason_ignores_short_operational_or_path_like_outputs(self) -> None:
        path_result = ChatResult(
            content="/home/guelfoweb/LAB/orbit",
            model="m",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )
        short_result = ChatResult(
            content="saved as cleaned_index.html",
            model="m",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

        self.assertIsNone(final_from_tool_retry_reason(path_result, length_retry_allowed=False, incomplete_retry_allowed=True))
        self.assertIsNone(final_from_tool_retry_reason(short_result, length_retry_allowed=False, incomplete_retry_allowed=True))

    def test_final_tool_retry_instruction_is_unchanged(self) -> None:
        self.assertEqual(
            final_tool_retry_instruction()["content"],
            "Do not call tools. Provide a shorter final answer from the available tool result now.",
        )

    def test_final_tool_compact_retry_instruction_is_constrained(self) -> None:
        content = final_tool_compact_retry_instruction()["content"]
        self.assertIn("three to five sentences", content)
        self.assertIn("No repetition", content)
        self.assertIn("Do not call tools", content)

    def test_final_answer_completeness_detects_heading_stub(self) -> None:
        completeness = classify_final_answer_completeness("The file contains several vulnerabilities.\n\n### ")
        self.assertEqual(completeness.status, "malformed_markdown")

    def test_final_answer_completeness_detects_list_label_stub(self) -> None:
        completeness = classify_final_answer_completeness("1. **SQL Injection:**")
        self.assertEqual(completeness.status, "incomplete_stub")

    def test_final_answer_completeness_detects_unclosed_backtick(self) -> None:
        completeness = classify_final_answer_completeness("Use the variable `user_input to build the query.")
        self.assertEqual(completeness.status, "malformed_markdown")

    def test_final_answer_completeness_detects_reasoning_like_answer(self) -> None:
        completeness = classify_final_answer_completeness("Plan:\n1. Inspect the file\n2. Summarize the findings")
        self.assertEqual(completeness.status, "reasoning_like")

    def test_final_answer_completeness_detects_reasoning_leakage_with_possibilities(self) -> None:
        completeness = classify_final_answer_completeness(
            '"What is the main difference between essay and wise?"\n'
            "The user likely meant \"essay\" and \"wise\".\n"
            "* **Possibility A:** compare essay and thesis.\n"
            "* **Possibility B:** compare essay and wise.\n"
            "The main difference is that an essay is a piece of writing, while wise means having good judgment."
        )
        self.assertEqual(completeness.status, "reasoning_like")

    def test_final_answer_completeness_detects_reasoning_leakage_with_means_wording(self) -> None:
        completeness = classify_final_answer_completeness(
            '"What is the main difference between essay and wise?"\n'
            "The user likely means \"essay\" and \"wise\".\n"
            "* **Possibility A:** compare essay and thesis.\n"
            "* **Possibility B:** compare essay and wise.\n"
            "An essay is a piece of writing, while wise describes judgment."
        )
        self.assertEqual(completeness.status, "reasoning_like")

    def test_final_answer_completeness_detects_closed_thought_without_final_tail(self) -> None:
        completeness = classify_final_answer_completeness("<|channel>thought\nprivate chain<channel|>")
        self.assertEqual(completeness.status, "reasoning_like")

    def test_final_answer_completeness_accepts_complete_brief_answer(self) -> None:
        completeness = classify_final_answer_completeness(
            "The file is vulnerable to SQL injection and command injection due to unsanitized input handling."
        )
        self.assertEqual(completeness.status, "complete")

    def test_final_answer_completeness_allows_normal_possibility_wording(self) -> None:
        completeness = classify_final_answer_completeness(
            "One possibility is that the user is comparing an essay with the adjective \"wise\", but the direct difference is that an essay is a written composition while wise describes judgment."
        )
        self.assertEqual(completeness.status, "complete")

    def test_repetitive_final_answer_is_detected(self) -> None:
        content = (
            "The file contains several issues that need review. "
            "The file contains several issues that need review. "
            "The file contains several issues that need review."
        )
        self.assertTrue(is_repetitive_final_answer(content))

    def test_final_answer_completeness_detects_too_short_after_large_tool_result(self) -> None:
        messages = [
            {
                "role": "user",
                "content": "inspect vulnerable_service.py and explain the vulnerabilities",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "exec_shell_full_command",
                            "arguments": "{\"command\":\"cat samples/vulnerable_service.py\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "exec_shell_full_command",
                "tool_call_id": "call-1",
                "content": "x" * 1200,
            },
        ]
        completeness = classify_final_answer_completeness("Several flaws exist", messages=messages)
        self.assertEqual(completeness.status, "too_short_after_tool")

    def test_compact_retry_reason_detects_long_length_final(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "exec_shell_full_command", "arguments": {"command": "cat vulnerable_service.py"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "exec_shell_full_command", "content": "x" * 1600},
        ]
        result = ChatResult(
            content="Long answer that ran out of space before finishing the explanation.",
            model="m",
            finish_reason="length",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

        self.assertEqual(final_from_tool_compact_retry_reason(result, messages=messages), "length")

    def test_compact_retry_reason_detects_repetitive_final(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "exec_shell_full_command", "arguments": {"command": "cat vulnerable_service.py"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "exec_shell_full_command", "content": "x" * 1600},
        ]
        result = ChatResult(
            content=(
                "The application is vulnerable to injection and weak validation. "
                "The application is vulnerable to injection and weak validation. "
                "The application is vulnerable to injection and weak validation."
            ),
            model="m",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

        self.assertEqual(final_from_tool_compact_retry_reason(result, messages=messages), "repetition")


if __name__ == "__main__":
    unittest.main()
