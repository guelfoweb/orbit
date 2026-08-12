from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orbit.backend.base import ChatResult, TokenCount
from orbit.native_llama.model_discovery import ModelDiscoveryResult, ModelDiscoveryRow
from orbit.runtime import ChatRuntime
from orbit.runtime.command_evidence import AcquiredEvidence
from orbit.runtime.evidence import EvidenceStore
from orbit.terminal.command_actions import (
    build_list_action,
    build_models_action,
    build_read_action,
    build_search_action,
)
from orbit.terminal.command_registry import COMMANDS, commands_matching, resolve_command
from orbit.terminal.commands import help_text
from orbit.terminal import cli


class AnswerBackend:
    def __init__(self, *, prompt_tokens: int = 100, context_tokens: int = 8192) -> None:
        self.prompt_tokens = prompt_tokens
        self.context_tokens = context_tokens
        self.calls = 0
        self.messages_by_call: list[list[dict[str, object]]] = []

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        self.calls += 1
        self.messages_by_call.append(messages)
        return ChatResult(
            content="The acquired evidence supports this complete answer.",
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=self.prompt_tokens,
            completion_tokens=8,
            cached_tokens=0,
            prompt_tokens_per_second=10.0,
            generation_tokens_per_second=2.0,
        )

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None, on_delta=None, on_progress=None):
        result = self.chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
        if on_delta is not None:
            on_delta(result.content)
        return result

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        payload = repr(messages).encode("utf-8")
        return TokenCount(
            tokens=self.prompt_tokens,
            context_tokens=self.context_tokens,
            rendered_hash=hashlib.sha256(payload).hexdigest(),
            token_hash=hashlib.sha256(b"tokens:" + payload).hexdigest(),
        )

    def count_text_tokens(self, text):
        return TokenCount(tokens=max(1, len(text) // 4), context_tokens=self.context_tokens)


class SlashCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workdir = Path(temporary.name)

    def test_registry_is_unique_and_drives_help_and_prefix_filtering(self) -> None:
        names = [command.name for command in COMMANDS]

        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([command.name for command in commands_matching("/re")], ["/read", "/reset"])
        rendered = help_text()
        for name in ("/read", "/search", "/ls", "/models", "/help", "/status", "/props", "/clear", "/reset", "/exit"):
            self.assertIn(name, rendered)

    def test_resolve_command_keeps_quoted_arguments_for_handler(self) -> None:
        invocation = resolve_command('/search "CVE 2026" report.pdf summarize matches')

        self.assertEqual(invocation.spec.name, "/search")
        self.assertEqual(invocation.arguments, '"CVE 2026" report.pdf summarize matches')

    def test_read_local_text_is_bounded_and_does_not_request_model(self) -> None:
        (self.workdir / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")

        action = build_read_action("notes.txt", workdir=self.workdir)

        self.assertFalse(action.needs_model)
        self.assertIn("file_display_result: true", action.output)
        self.assertIn("coverage: complete", action.output)

    def test_read_large_text_uses_partial_bounded_display(self) -> None:
        (self.workdir / "large.txt").write_text("".join(f"line {index}\n" for index in range(300)), encoding="utf-8")

        action = build_read_action("large.txt", workdir=self.workdir)

        self.assertFalse(action.needs_model)
        self.assertIn("coverage: partial", action.output)
        self.assertIn("line_range: 1-100", action.output)

    def test_read_pdf_reuses_existing_safe_reader(self) -> None:
        with mock.patch("orbit.terminal.command_actions.read_pdf", return_value="pdf_text: true\ncontent:\nreport") as reader:
            action = build_read_action("report.pdf", workdir=self.workdir)

        reader.assert_called_once_with("report.pdf", arguments={}, workdir=self.workdir)
        self.assertEqual(action.output, "pdf_text: true\ncontent:\nreport")
        self.assertFalse(action.needs_model)

    def test_read_url_and_optional_prompt_use_deterministic_fetch(self) -> None:
        fetched = "status: ok\nurl: https://example.com/a\ntext_truncated: false\ntext:\nalpha"
        with mock.patch("orbit.terminal.command_actions.execute_fetch_url", return_value=fetched) as fetch:
            action = build_read_action("https://example.com/a explain risks", workdir=self.workdir)

        fetch.assert_called_once_with({"url": "https://example.com/a"})
        self.assertTrue(action.needs_model)
        self.assertEqual(action.prompt, "explain risks")
        self.assertEqual(action.evidence.content, fetched)

    def test_read_local_prompt_uses_full_document_admission(self) -> None:
        (self.workdir / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")

        action = build_read_action("notes.txt summarize the risks", workdir=self.workdir)

        self.assertTrue(action.needs_model)
        self.assertEqual(action.full_document_path, "notes.txt")
        self.assertIsNone(action.evidence)

    def test_search_web_is_deterministic_without_optional_prompt(self) -> None:
        with mock.patch("orbit.terminal.command_actions.search_web", return_value="web_search_results: true\nresults: none") as search:
            action = build_search_action('"CVE"', workdir=self.workdir)

        search.assert_called_once_with("CVE")
        self.assertFalse(action.needs_model)
        self.assertIn("web_search_results: true", action.output)

    def test_search_local_file_scans_complete_snapshot(self) -> None:
        (self.workdir / "report.txt").write_text("zero\nCVE-2026-0001\nlast\n", encoding="utf-8")

        action = build_search_action('"CVE" report.txt', workdir=self.workdir)

        self.assertFalse(action.needs_model)
        self.assertIn("scan_coverage=complete", action.output)
        self.assertIn("CVE-2026-0001", action.output)

    def test_search_pdf_reuses_existing_safe_pdf_command_path(self) -> None:
        evidence = "shell_output_pdf_text: true\npath: report.pdf\ncontent:\n1:CVE"
        with mock.patch("orbit.terminal.command_actions.execute_exec_shell_full_command", return_value=evidence) as execute:
            action = build_search_action('"CVE" report.pdf', workdir=self.workdir)

        execute.assert_called_once_with(
            {"command": "pdftotext report.pdf - | rg -e CVE"},
            workdir=self.workdir,
            user_prompt="search exactly for 'CVE' in report.pdf",
        )
        self.assertEqual(action.output, evidence)
        self.assertFalse(action.needs_model)

    def test_search_pdf_filters_extracted_text_literally(self) -> None:
        (self.workdir / "report.pdf").write_bytes(b"%PDF-1.4\n")
        with mock.patch(
            "orbit.runtime.shell_guardrails.extract_pdf_text",
            return_value=("alpha\nCVE[1] match\nomega\n", "test"),
        ):
            action = build_search_action('"CVE[1]" report.pdf', workdir=self.workdir)

        self.assertIn("CVE[1] match", action.output)
        self.assertNotIn("alpha", action.output)
        self.assertNotIn("omega", action.output)

    def test_search_url_searches_retrieved_page_not_the_web_again(self) -> None:
        fetched = "status: ok\nurl: https://example.com\ntext_truncated: false\ntext:\nfirst\nCVE match\nlast"
        with (
            mock.patch("orbit.terminal.command_actions.execute_fetch_url", return_value=fetched),
            mock.patch("orbit.terminal.command_actions.search_web") as global_search,
        ):
            action = build_search_action('"CVE" https://example.com summarize matches', workdir=self.workdir)

        global_search.assert_not_called()
        self.assertTrue(action.needs_model)
        self.assertEqual(action.prompt, "summarize matches")
        self.assertIn("CVE match", action.evidence.content)
        self.assertIn("retrieval_coverage=complete", action.evidence.content)
        self.assertEqual(action.evidence.arguments, {"url": "https://example.com"})

    def test_malformed_quotes_and_url_file_ambiguity_fail_closed(self) -> None:
        malformed = build_search_action('"unterminated', workdir=self.workdir)
        invalid_url = build_read_action("https:// summarize", workdir=self.workdir)

        self.assertTrue(malformed.output.startswith("error:"))
        self.assertIn("usage: /search", malformed.output)
        self.assertTrue(invalid_url.output.startswith("error:"))
        self.assertFalse(invalid_url.needs_model)

    def test_list_directory_is_confined_and_uses_current_workdir_by_default(self) -> None:
        (self.workdir / "a.txt").write_text("a", encoding="utf-8")

        listing = build_list_action("", workdir=self.workdir)
        escaped = build_list_action("../", workdir=self.workdir)

        self.assertIn("directory_listing: path=.", listing.output)
        self.assertIn("[file] a.txt", listing.output)
        self.assertIn("status=path_outside_workdir", escaped.output)
        self.assertFalse(listing.needs_model)

    def test_read_and_search_path_traversal_fail_closed(self) -> None:
        outside = self.workdir.parent / f"{self.workdir.name}-outside.txt"
        outside.write_text("CVE", encoding="utf-8")
        self.addCleanup(outside.unlink)

        relative = f"../{outside.name}"
        read = build_read_action(relative, workdir=self.workdir)
        search = build_search_action(f'"CVE" {relative}', workdir=self.workdir)

        self.assertTrue(read.output.startswith("error:"))
        self.assertTrue(search.output.startswith("error:"))
        self.assertFalse(read.needs_model)
        self.assertFalse(search.needs_model)

    def test_models_reuses_verified_discovery_output_without_loading_weights(self) -> None:
        discovered = ModelDiscoveryResult(
            rows=(ModelDiscoveryRow("Qwen3-Coder 30B-A3B", "AVAILABLE", "VERIFIED", "/models/qwen.gguf", "qwen"),),
            wall_ms=2.0,
            filesystem_scans=2,
            metadata_inspections=1,
        )
        with (
            mock.patch("orbit.terminal.command_actions.resolve_native_build_bin", return_value=Path("/native")),
            mock.patch("orbit.terminal.command_actions.discover_models", return_value=discovered) as discover,
        ):
            action = build_models_action("")

        discover.assert_called_once_with(build_bin=Path("/native"))
        self.assertIn("AVAILABLE", action.output)
        self.assertIn("VERIFIED", action.output)
        self.assertFalse(action.needs_model)

    def test_acquired_evidence_uses_exactly_one_normal_answer_path(self) -> None:
        backend = AnswerBackend()
        evidence_store = EvidenceStore(self.workdir / "evidence")
        runtime = ChatRuntime(backend=backend, evidence_store=evidence_store)
        evidence = AcquiredEvidence(
            tool_name="fetch_url",
            arguments={"url": "https://example.com"},
            content="status: ok\nurl: https://example.com\ntext:\nexact evidence",
            source="https://example.com",
        )

        result = runtime.answer_from_acquired_evidence(
            "explain the evidence",
            evidence=evidence,
            workdir=self.workdir,
            temperature=0.0,
            max_tokens=128,
        )

        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(backend.calls, 1)
        self.assertTrue(any(message.get("role") == "tool" for message in runtime.messages))
        self.assertEqual(runtime.messages[-1]["role"], "assistant")

    def test_optional_search_prompt_composes_with_one_model_call(self) -> None:
        (self.workdir / "report.txt").write_text("CVE-2026-0001\n", encoding="utf-8")
        action = build_search_action('"CVE" report.txt summarize matches', workdir=self.workdir)
        backend = AnswerBackend()
        runtime = ChatRuntime(backend=backend, evidence_store=EvidenceStore(self.workdir / "evidence"))

        result = runtime.answer_from_acquired_evidence(
            action.prompt,
            evidence=action.evidence,
            workdir=self.workdir,
            temperature=0.0,
            max_tokens=128,
        )

        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(backend.calls, 1)
        tool_call = next(message for message in runtime.messages if message.get("tool_calls"))
        self.assertIn('"command":"rg -n -F -- CVE report.txt"', tool_call["tool_calls"][0]["function"]["arguments"])

    def test_optional_prompt_failure_discards_acquired_evidence(self) -> None:
        class FailingBackend(AnswerBackend):
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                raise TimeoutError("timed out")

        store = EvidenceStore(self.workdir / "evidence")
        runtime = ChatRuntime(backend=FailingBackend(), evidence_store=store)
        evidence = AcquiredEvidence(
            "fetch_url",
            {"url": "https://example.com"},
            "status: ok\ntext:\nexact",
            "https://example.com",
        )

        with self.assertRaises(TimeoutError):
            runtime.answer_from_acquired_evidence(
                "summarize",
                evidence=evidence,
                workdir=self.workdir,
                temperature=0.0,
                max_tokens=128,
            )

        self.assertEqual(store.records, {})
        self.assertEqual(runtime.messages, [])

    def test_optional_prompt_interrupt_discards_acquired_evidence(self) -> None:
        class InterruptedBackend(AnswerBackend):
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                raise KeyboardInterrupt

        store = EvidenceStore(self.workdir / "evidence")
        runtime = ChatRuntime(backend=InterruptedBackend(), evidence_store=store)

        with self.assertRaises(KeyboardInterrupt):
            runtime.answer_from_acquired_evidence(
                "summarize",
                evidence=AcquiredEvidence(
                    "fetch_url",
                    {"url": "https://example.com"},
                    "status: ok\ntext:\nexact",
                    "https://example.com",
                ),
                workdir=self.workdir,
                temperature=0.0,
                max_tokens=128,
            )

        self.assertEqual(store.records, {})
        self.assertEqual(runtime.messages, [])

    def test_optional_prompt_cancel_discards_acquired_evidence(self) -> None:
        class CancelledBackend(AnswerBackend):
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                return ChatResult("cancelled", "fake", "cancelled", [], 10, 0, 0, None, None)

        store = EvidenceStore(self.workdir / "evidence")
        runtime = ChatRuntime(backend=CancelledBackend(), evidence_store=store)
        result = runtime.answer_from_acquired_evidence(
            "summarize",
            evidence=AcquiredEvidence(
                "fetch_url",
                {"url": "https://example.com"},
                "status: ok\ntext:\nx",
                "url",
            ),
            workdir=self.workdir,
            temperature=0.0,
            max_tokens=128,
        )

        self.assertEqual(result.finish_reason, "cancelled")
        self.assertEqual(store.records, {})
        self.assertEqual(runtime.messages, [])

    def test_one_shot_dispatch_uses_registry_and_plain_deterministic_action(self) -> None:
        (self.workdir / "a.txt").write_text("a", encoding="utf-8")
        config = mock.Mock(workdir=self.workdir)

        result = cli._handle_one_shot_command("/ls", mock.Mock(), config, mock.Mock())

        self.assertFalse(result.needs_model)
        self.assertNotIn("\x1b", result.output)
        self.assertNotIn("\r", result.output)

    def test_oversized_full_document_command_fails_closed_without_model_call(self) -> None:
        (self.workdir / "large.txt").write_text("x" * 20_000, encoding="utf-8")
        backend = AnswerBackend(prompt_tokens=8000, context_tokens=8192)
        runtime = ChatRuntime(backend=backend)

        result = runtime.answer_full_document_command(
            "summarize all of it",
            path="large.txt",
            workdir=self.workdir,
            temperature=0.0,
            max_tokens=512,
        )

        self.assertEqual(backend.calls, 0)
        self.assertIn("Document coverage: none", result.content)
        self.assertIn("requires at least", result.content)

    def test_fit_full_document_command_uses_one_model_call_and_complete_coverage(self) -> None:
        (self.workdir / "small.txt").write_text("alpha\nbeta\n", encoding="utf-8")
        backend = AnswerBackend(prompt_tokens=100, context_tokens=8192)
        runtime = ChatRuntime(backend=backend)

        result = runtime.answer_full_document_command(
            "summarize it",
            path="small.txt",
            workdir=self.workdir,
            temperature=0.0,
            max_tokens=128,
        )

        self.assertEqual(backend.calls, 1)
        self.assertIn("Document coverage: complete", result.content)
        self.assertIn("complete answer", result.content)


if __name__ == "__main__":
    unittest.main()
