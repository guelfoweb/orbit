from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult, TokenCount
from orbit.runtime import ChatRuntime
from orbit.runtime.document_tool import (
    FILE_DISPLAY_MARKER,
    execute_read_file,
)
from orbit.runtime.file_tools import read_full_document_snapshot
from orbit.runtime.evidence import EvidenceStore
from orbit.runtime.tools import tool_definitions
from orbit.runtime.full_document import (
    attest_full_document_snapshot,
    exact_coverage_notice,
    file_display_coverage_notice,
    full_document_blocked_notice,
    full_document_control_marker,
    full_document_messages,
    identify_full_document_request,
    parse_complete_file_display,
    parse_full_document_snapshot,
    required_full_document_context,
    round_context_requirement,
    targeted_search_coverage_notice,
    targeted_search_no_match_notice,
)


class PreflightBackend:
    def __init__(self, *, prompt_tokens: int = 1000, context_tokens: int = 8192) -> None:
        self.prompt_tokens = prompt_tokens
        self.context_tokens = context_tokens
        self.calls = 0
        self.count_calls = 0
        self.messages_by_call = []
        self.tools_by_call = []
        self.on_second_count = None
        self.on_full_call = None

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        self.calls += 1
        self.messages_by_call.append(messages)
        self.tools_by_call.append(tools)
        if self.calls == 1:
            return ChatResult(
                content='{"command":"sed -n 1,20p note.txt"}',
                model="fake",
                finish_reason="stop",
                tool_calls=[],
                prompt_tokens=100,
                completion_tokens=8,
                cached_tokens=0,
                prompt_tokens_per_second=None,
                generation_tokens_per_second=None,
            )
        if self.on_full_call is not None:
            self.on_full_call()
        return ChatResult(
            content="complete analysis",
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=self.prompt_tokens,
            completion_tokens=3,
            cached_tokens=0,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None, on_delta=None, on_progress=None):
        result = self.chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
        if on_delta is not None and result.content:
            on_delta(result.content)
        return result

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        self.count_calls += 1
        if self.count_calls == 2 and self.on_second_count is not None:
            self.on_second_count()
        return TokenCount(
            tokens=self.prompt_tokens,
            context_tokens=self.context_tokens,
            rendered_hash="a" * 64,
            token_hash="b" * 64,
        )

    def count_text_tokens(self, text):
        return TokenCount(tokens=max(1, len(text) // 4), context_tokens=self.context_tokens)


class FullDocumentTests(unittest.TestCase):
    def _snapshot(self, content: str):
        temporary = tempfile.TemporaryDirectory()
        workdir = Path(temporary.name)
        (workdir / "note.txt").write_text(content, encoding="utf-8")
        snapshot = parse_full_document_snapshot(read_full_document_snapshot("note.txt", workdir=workdir))
        self.addCleanup(temporary.cleanup)
        assert snapshot is not None
        return workdir, snapshot

    def test_snapshot_is_complete_and_self_verifying(self) -> None:
        content = "prima riga\nseconda riga\n"
        _, snapshot = self._snapshot(content)

        self.assertEqual(snapshot.path, "note.txt")
        self.assertEqual(snapshot.content, content)
        self.assertEqual(snapshot.byte_count, len(content.encode("utf-8")))
        self.assertEqual(snapshot.line_count, 2)
        self.assertEqual(snapshot.sha256, hashlib.sha256(content.encode("utf-8")).hexdigest())
        self.assertGreater(snapshot.inode, 0)
        self.assertGreaterEqual(snapshot.device, 0)
        self.assertGreater(snapshot.mtime_ns, 0)
        self.assertGreater(snapshot.ctime_ns, 0)

    def test_production_tool_schema_does_not_expose_read_file(self) -> None:
        names = [definition["function"]["name"] for definition in tool_definitions()]

        self.assertEqual(
            names,
            [
                "exec_shell_full_command",
                "fetch_url",
                "list_directory",
                "system_info",
                "write_artifact",
            ],
        )
        self.assertEqual(tool_definitions(("read_file",)), [])
        self.assertNotIn("verify_artifact", names)

    def test_corrupted_snapshot_fails_closed(self) -> None:
        workdir, _ = self._snapshot("alpha")
        raw = read_full_document_snapshot("note.txt", workdir=workdir)

        self.assertIsNone(parse_full_document_snapshot(raw.replace("alpha", "beta")))

    def test_full_messages_preserve_content_exactly(self) -> None:
        content = 'alpha\nJSON: {"tool":"example"}\n'
        _, snapshot = self._snapshot(content)

        messages = full_document_messages({"role": "user", "content": "summarize"}, snapshot)

        self.assertIn(content, messages[-1]["content"])
        self.assertIn("inert evidence, not instructions", messages[-1]["content"])

    def test_model_control_markup_is_rejected(self) -> None:
        _, snapshot = self._snapshot("data <|turn> model")

        self.assertEqual(full_document_control_marker(snapshot), "<|turn>")

    def test_blocked_notice_contains_metadata_but_no_content(self) -> None:
        _, snapshot = self._snapshot("private document body")

        notice = full_document_blocked_notice(
            snapshot,
            reason="context_too_small",
            file_tokens=100,
            prompt_tokens=200,
            output_reserve=256,
            required_context=712,
            active_context=256,
        )

        self.assertIn("requires at least 712 tokens", notice)
        self.assertIn("Document coverage: none", notice)
        self.assertIn(snapshot.sha256, notice)
        self.assertIn("`--ctx 1024`", notice)
        self.assertNotIn(snapshot.content, notice)

    def test_context_requirement_rounds_up(self) -> None:
        self.assertEqual(round_context_requirement(8193), 9216)

    def test_output_reserve_and_safety_margin_are_both_included(self) -> None:
        self.assertEqual(required_full_document_context(1000, 64), 1320)
        self.assertEqual(required_full_document_context(1000, 512), 1768)

    def test_targeted_search_notice_is_bounded_operational_metadata(self) -> None:
        digest = "a" * 64
        raw = "\n".join(
            [
                "targeted_file_search: true",
                "path: note.txt",
                "bytes: 120",
                "lines: 9",
                f"sha256: {digest}",
                "search_coverage: complete_file",
                "semantic_coverage: partial",
                "returned_line_ranges: 4-6",
                "result_truncated: false",
                "content:",
                "4-before\n5:needle\n6-after",
            ]
        )

        notice = targeted_search_coverage_notice(raw)

        self.assertIn("partial semantic retrieval", notice or "")
        self.assertIn("cannot establish absence", notice or "")
        self.assertIn("returned lines 4-6", notice or "")
        self.assertIn("lexical matches not reported", notice or "")
        self.assertIn(digest, notice or "")
        self.assertIsNone(targeted_search_coverage_notice(raw.replace(digest, "invalid")))

    def test_internal_display_notice_never_presents_a_page_as_complete(self) -> None:
        raw = "\n".join(
            [
                "shell_output_read_file: true",
                "original_command: cat note.txt",
                "file_display_result: true",
                "path: note.txt",
                "bytes: 10000",
                "lines: 500",
                f"sha256: {'a' * 64}",
                "coverage: partial",
                "selection: default",
                "line_range: 1-100",
                "next_cursor: v1:cursor",
                "content:",
                "alpha",
            ]
        )

        notice = file_display_coverage_notice(raw)

        self.assertIn("Document coverage: partial exact display", notice or "")
        self.assertIn("returned lines 1-100", notice or "")
        self.assertIn("does not represent the complete file", notice or "")

    def test_complete_display_parser_requires_exact_content_identity(self) -> None:
        workdir, _ = self._snapshot("alpha\nbeta\n")
        raw = execute_read_file({"path": "note.txt"}, workdir=workdir)

        display = parse_complete_file_display(raw)

        self.assertIsNotNone(display)
        assert display is not None
        self.assertEqual(display.content, "alpha\nbeta\n")
        self.assertEqual(display.byte_count, 11)
        self.assertIsNone(parse_complete_file_display(raw.replace("beta", "zeta")))
        self.assertIsNone(file_display_coverage_notice(raw.replace("beta", "zeta")))

    def test_zero_match_notice_refuses_definitive_negative(self) -> None:
        raw = "\n".join(
            [
                "targeted_file_search: true",
                "path: note.txt",
                "bytes: 10",
                "lines: 1",
                f"sha256: {'a' * 64}",
                "search_coverage: complete_file",
                "semantic_coverage: partial",
                "match_count: 0",
                "returned_line_ranges: unavailable",
                "result_truncated: false",
                "content:",
                "(no lexical matches)",
            ]
        )

        notice = targeted_search_no_match_notice(raw)

        self.assertIn("does not prove", notice or "")
        self.assertIn("partial semantic coverage", notice or "")

    def test_explicit_full_document_request_requires_one_exact_path(self) -> None:
        accepted = (
            "Read note.txt completely and summarize it.",
            "Analyze the entire file `note.txt` and report the result.",
            "Read the whole document note.txt.",
            "Read whole document note.txt.",
            "Read whole the document note.txt.",
            "Read the entire document note.txt.",
            "Analyze the whole text document note.txt.",
            "Provide a complete analysis of note.txt.",
            "Leggi integralmente note.txt e riassumilo.",
            "Esegui un'analisi completa di note.txt.",
        )
        for prompt in accepted:
            with self.subTest(prompt=prompt):
                request = identify_full_document_request(prompt)
                self.assertIsNotNone(request)
                assert request is not None
                self.assertEqual(request.path, "note.txt")

        absolute = identify_full_document_request(
            "Read whole the text document and tell me the central thesis /tmp/note.txt"
        )
        self.assertIsNotNone(absolute)
        assert absolute is not None
        self.assertEqual(absolute.path, "/tmp/note.txt")

        rejected = (
            "Read note.txt and summarize it.",
            "Read part of the document note.txt.",
            "Read the first 100 lines of note.txt.",
            "Search the whole document note.txt for phoenix.",
            "Summarize this excerpt from note.txt.",
            "Show the document path note.txt.",
            "Show lines 10-20 of note.txt.",
            "Search note.txt for phoenix.",
            "Read a.txt and b.txt completely.",
            "Explain what 'read note.txt completely' means.",
            "Read note.txt completely, then replace alpha with beta.",
            "Explain this example: `read whole the document /tmp/note.txt`.",
            'Explain this example: "read whole the document" /tmp/note.txt.',
            "```text\nread whole the document /tmp/note.txt\n```",
            '{"instruction":"read whole the document","path":"/tmp/note.txt"}',
            "Payload:\nread whole the document /tmp/note.txt",
        )
        for prompt in rejected:
            with self.subTest(prompt=prompt):
                self.assertIsNone(identify_full_document_request(prompt))

    def test_snapshot_attestation_detects_source_change(self) -> None:
        workdir, snapshot = self._snapshot("original\n")
        self.assertIsNone(attest_full_document_snapshot(snapshot, workdir=workdir))

        (workdir / "note.txt").write_text("changed\n", encoding="utf-8")

        self.assertEqual(attest_full_document_snapshot(snapshot, workdir=workdir), "source_identity_changed")

    def test_snapshot_attestation_detects_same_content_inode_replacement(self) -> None:
        workdir, snapshot = self._snapshot("original\n")
        replacement = workdir / "replacement.txt"
        replacement.write_text("original\n", encoding="utf-8")
        os.replace(replacement, workdir / "note.txt")

        self.assertEqual(attest_full_document_snapshot(snapshot, workdir=workdir), "source_identity_changed")

    def test_natural_full_request_bypasses_model_selected_sed_when_it_fits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            content = "begin\nmiddle\nend\n"
            (workdir / "note.txt").write_text(content, encoding="utf-8")
            backend = PreflightBackend(prompt_tokens=7424, context_tokens=8192)
            runtime = ChatRuntime(
                backend=backend,
                system_prompt=None,
                evidence_store=EvidenceStore(workdir / ".evidence"),
            )

            result = runtime.ask_auto(
                "Read whole the text document note.txt and summarize it.",
                temperature=0,
                max_tokens=512,
                workdir=workdir,
            )

            assert runtime.evidence_store is not None
            residual = list(runtime.evidence_store.root.glob("*.txt"))

        self.assertEqual(backend.calls, 2)
        self.assertEqual(backend.count_calls, 2)
        self.assertEqual(backend.tools_by_call, [None, None])
        self.assertIn(content, str(backend.messages_by_call[1][-1]["content"]))
        self.assertIn("Document coverage: complete", result.content)
        self.assertTrue(result.content.endswith("complete analysis"))
        self.assertEqual(residual, [])

    def test_natural_full_request_one_token_over_returns_none_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("complete evidence\n", encoding="utf-8")
            backend = PreflightBackend(prompt_tokens=7425, context_tokens=8192)
            runtime = ChatRuntime(
                backend=backend,
                system_prompt=None,
                evidence_store=EvidenceStore(workdir / ".evidence"),
            )

            result = runtime.ask_auto(
                "Analyze note.txt completely.",
                temperature=0,
                max_tokens=512,
                workdir=workdir,
            )

        self.assertEqual(backend.calls, 1)
        self.assertIn("Document coverage: none", result.content)
        self.assertIn("requires at least 8,193 tokens", result.content)
        self.assertIn("`--ctx 9216`", result.content)
        self.assertNotIn("complete evidence", result.content)

    def test_whole_text_document_request_uses_full_document_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "note.txt"
            target.write_text("beginning\nmiddle\nend\n", encoding="utf-8")
            backend = PreflightBackend(prompt_tokens=7425, context_tokens=8192)
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_auto(
                f"read whole the text document and tell me the central thesis {target}",
                temperature=0,
                max_tokens=512,
                workdir=workdir,
            )

        self.assertEqual(backend.calls, 1)
        self.assertEqual(backend.count_calls, 2)
        self.assertIn("Document coverage: none", result.content)
        self.assertIn("requires at least 8,193 tokens", result.content)
        self.assertNotIn("beginning", result.content)
        self.assertNotIn("central thesis", result.content.lower())

    def test_full_document_1024_budget_is_used_for_context_admission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("complete evidence\n", encoding="utf-8")
            backend = PreflightBackend(prompt_tokens=6913, context_tokens=8192)
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_auto(
                "Analyze note.txt completely.",
                temperature=0,
                max_tokens=1024,
                workdir=workdir,
            )

        self.assertEqual(backend.calls, 1)
        self.assertIn("Document coverage: none", result.content)
        self.assertIn("requires at least 8,193 tokens", result.content)
        self.assertIn("1024-token output reserve", result.content)
        self.assertNotIn("complete evidence", result.content)

    def test_preflight_rejects_changed_file_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "note.txt"
            target.write_text("original\n", encoding="utf-8")
            backend = PreflightBackend()
            backend.on_second_count = lambda: target.write_text("changed\n", encoding="utf-8")
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_auto(
                "Read note.txt completely.",
                temperature=0,
                max_tokens=512,
                workdir=workdir,
            )

        self.assertEqual(backend.calls, 1)
        self.assertIn("source_identity_changed", result.content)
        self.assertIn("Document coverage: none", result.content)

    def test_preflight_discards_model_output_when_file_changes_during_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "note.txt"
            target.write_text("original\n", encoding="utf-8")
            backend = PreflightBackend()
            backend.on_full_call = lambda: target.write_text("changed\n", encoding="utf-8")
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            deltas: list[str] = []

            result = runtime.ask_auto(
                "Read note.txt completely.",
                temperature=0,
                max_tokens=512,
                workdir=workdir,
                on_final_delta=deltas.append,
            )

        self.assertEqual(backend.calls, 2)
        self.assertIn("discarded because the source changed", result.content)
        self.assertIn("Document coverage: none for the current file", result.content)
        self.assertNotIn("complete analysis", result.content)
        self.assertNotIn("complete analysis", "".join(deltas))

    def test_preflight_rejects_tokenizer_template_identity_change(self) -> None:
        class ChangingIdentityBackend(PreflightBackend):
            def count_chat_tokens(self, messages, *, tools=None, thinking=False):
                value = super().count_chat_tokens(messages, tools=tools, thinking=thinking)
                if self.count_calls == 2:
                    return TokenCount(
                        tokens=value.tokens,
                        context_tokens=value.context_tokens,
                        rendered_hash="c" * 64,
                        token_hash=value.token_hash,
                    )
                return value

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("original\n", encoding="utf-8")
            backend = ChangingIdentityBackend()
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_auto(
                "Read note.txt completely.",
                temperature=0,
                max_tokens=512,
                workdir=workdir,
            )

        self.assertEqual(backend.calls, 1)
        self.assertIn("tokenizer_template_or_context_changed", result.content)

    def test_preflight_cleanup_runs_after_timeout(self) -> None:
        class TimeoutBackend(PreflightBackend):
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                if self.calls == 0:
                    return super().chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
                raise TimeoutError("timed out")

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("complete evidence\n", encoding="utf-8")
            backend = TimeoutBackend()
            runtime = ChatRuntime(
                backend=backend,
                system_prompt=None,
                evidence_store=EvidenceStore(workdir / ".evidence"),
            )

            with self.assertRaises(TimeoutError):
                runtime.ask_auto(
                    "Read note.txt completely.",
                    temperature=0,
                    max_tokens=512,
                    workdir=workdir,
                )

            assert runtime.evidence_store is not None
            self.assertEqual(list(runtime.evidence_store.root.glob("*.txt")), [])

    def test_preflight_cleanup_runs_after_cancelled_result(self) -> None:
        class CancelledBackend(PreflightBackend):
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                if self.calls == 0:
                    return super().chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
                self.calls += 1
                return ChatResult(
                    content="",
                    model="fake",
                    finish_reason="cancelled",
                    tool_calls=[],
                    prompt_tokens=self.prompt_tokens,
                    completion_tokens=0,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("complete evidence\n", encoding="utf-8")
            runtime = ChatRuntime(
                backend=CancelledBackend(),
                system_prompt=None,
                evidence_store=EvidenceStore(workdir / ".evidence"),
            )

            result = runtime.ask_auto(
                "Read note.txt completely.",
                temperature=0,
                max_tokens=512,
                workdir=workdir,
            )

            assert runtime.evidence_store is not None
            self.assertEqual(result.finish_reason, "cancelled")
            self.assertEqual(list(runtime.evidence_store.root.glob("*.txt")), [])

class InternalFilePaginationTests(unittest.TestCase):
    def test_display_returns_exact_page_with_complete_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            content = "alpha\nbeta\ngamma\n"
            (workdir / "note.txt").write_text(content, encoding="utf-8")

            result = execute_read_file(
                {"path": "note.txt", "start_line": 2, "line_count": 1},
                workdir=workdir,
            )

        self.assertTrue(result.startswith(FILE_DISPLAY_MARKER))
        self.assertIn("coverage: partial", result)
        self.assertIn("line_range: 2-2", result)
        self.assertRegex(result, r"next_cursor: v1:[0-9a-f]{64}:3")
        self.assertTrue(result.endswith("beta\n"))
        self.assertIn(hashlib.sha256(content.encode()).hexdigest(), result)

    def test_display_empty_file_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "empty.txt").write_text("", encoding="utf-8")
            result = execute_read_file({"path": "empty.txt"}, workdir=workdir)

        self.assertIn("coverage: complete", result)
        self.assertIn("line_range: none", result)

    def test_binary_and_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("alpha", encoding="utf-8")
            (workdir / "binary.txt").write_bytes(b"a\x00b")
            (workdir / "link.txt").symlink_to(workdir / "note.txt")

            binary = execute_read_file({"path": "binary.txt"}, workdir=workdir)
            symlink = execute_read_file({"path": "link.txt"}, workdir=workdir)

        self.assertIn("binary", binary)
        self.assertIn("does not follow symlinks", symlink)

    def test_cursor_paginates_without_gaps_and_rejects_changed_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "note.txt"
            content = "uno\ndue\ntré\nquattro\ncinque\n"
            target.write_text(content, encoding="utf-8")
            returned = ""
            cursor = None
            while True:
                arguments = {"path": "note.txt", "line_count": 2}
                if cursor is not None:
                    arguments["cursor"] = cursor
                page = execute_read_file(arguments, workdir=workdir)
                returned += page.split("\ncontent:\n", 1)[1]
                cursor_line = next(line for line in page.splitlines() if line.startswith("next_cursor: "))
                cursor = cursor_line.removeprefix("next_cursor: ")
                if cursor == "none":
                    break
            self.assertEqual(returned, content)

            first = execute_read_file({"path": "note.txt", "line_count": 1}, workdir=workdir)
            stale_cursor = next(
                line.removeprefix("next_cursor: ")
                for line in first.splitlines()
                if line.startswith("next_cursor: ")
            )
            target.write_text(content + "sei\n", encoding="utf-8")
            stale = execute_read_file({"path": "note.txt", "cursor": stale_cursor}, workdir=workdir)

        self.assertIn("file changed since the previous page", stale)

    def test_utf8_page_boundaries_preserve_exact_characters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "unicode.txt").write_text("😀 prima\n漢字 seconda\n", encoding="utf-8")
            first = execute_read_file({"path": "unicode.txt", "line_count": 1}, workdir=workdir)
            cursor = next(
                line.removeprefix("next_cursor: ")
                for line in first.splitlines()
                if line.startswith("next_cursor: ")
            )
            second = execute_read_file({"path": "unicode.txt", "cursor": cursor}, workdir=workdir)

        self.assertTrue(first.endswith("😀 prima\n"))
        self.assertTrue(second.endswith("漢字 seconda\n"))

    def test_special_file_and_permission_error_fail_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            fifo = workdir / "pipe.txt"
            os.mkfifo(fifo)
            fifo_result = execute_read_file({"path": "pipe.txt"}, workdir=workdir)
            with patch("orbit.runtime.file_tools.os.open", side_effect=PermissionError("denied")):
                permission_result = execute_read_file({"path": "missing.txt"}, workdir=workdir)

        self.assertIn("not a regular file", fifo_result)
        self.assertIn("unable to read file safely", permission_result)

    def test_concurrent_file_changes_discard_the_snapshot(self) -> None:
        actions = {
            "modified": lambda path: path.write_text("changed\n", encoding="utf-8"),
            "truncated": lambda path: path.write_bytes(b""),
            "replaced": lambda path: (path.unlink(), path.write_text("replacement\n", encoding="utf-8")),
            "deleted": lambda path: path.unlink(),
        }
        for label, action in actions.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                workdir = Path(tmp)
                target = workdir / "note.txt"
                target.write_text("original\n", encoding="utf-8")
                original_read = os.read
                changed = False

                def mutate_after_read(fd: int, count: int) -> bytes:
                    nonlocal changed
                    data = original_read(fd, count)
                    if not changed:
                        changed = True
                        action(target)
                    return data

                with patch("orbit.runtime.file_tools.os.read", side_effect=mutate_after_read):
                    result = execute_read_file({"path": "note.txt"}, workdir=workdir)

                self.assertTrue(result.startswith("error:"), result)
                self.assertIn("snapshot discarded", result)

    def test_symlink_replacement_race_discards_the_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            workdir = Path(tmp)
            target = workdir / "note.txt"
            target.write_text("inside\n", encoding="utf-8")
            outside_target = Path(outside) / "outside.txt"
            outside_target.write_text("outside\n", encoding="utf-8")
            original_read = os.read
            changed = False

            def replace_with_symlink(fd: int, count: int) -> bytes:
                nonlocal changed
                data = original_read(fd, count)
                if not changed:
                    changed = True
                    target.unlink()
                    target.symlink_to(outside_target)
                return data

            with patch("orbit.runtime.file_tools.os.read", side_effect=replace_with_symlink):
                result = execute_read_file({"path": "note.txt"}, workdir=workdir)

        self.assertTrue(result.startswith("error:"), result)
        self.assertNotIn("outside", result)


if __name__ == "__main__":
    unittest.main()
