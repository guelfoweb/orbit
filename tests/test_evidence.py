from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.evidence import (
    EvidenceStore,
    build_compact_final_evidence_context,
    build_final_evidence_context,
    build_post_tool_route_evidence_context,
    build_route_evidence_context,
    build_web_final_evidence_context,
    build_evidence_record,
    tool_evidence_ref,
)
from orbit.runtime.sessions import SessionStore


class EvidenceTests(unittest.TestCase):
    def test_record_has_id_hash_size_and_raw_ref(self) -> None:
        record = build_evidence_record("exec_shell_full_command", "hello\nworld", {"command": "printf hello"})

        self.assertTrue(record.evidence_id.startswith("ev_"))
        self.assertEqual(record.raw_chars, 11)
        self.assertEqual(record.raw_lines, 2)
        self.assertTrue(record.raw_ref.startswith("evidence:"))
        self.assertIn("sha256:", record.route_card)
        self.assertIn("hello", record.final_card)

    def test_sidecar_write_read_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add("exec_shell_full_command", "raw content", metadata={"command": "pwd"})

            self.assertEqual(store.load_raw(record.evidence_id), "raw content")
            self.assertTrue((store.root / "index.json").exists())
            self.assertTrue((store.root / f"{record.evidence_id}.txt").exists())

    def test_for_workdir_reloads_index_and_lazy_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            workdir = Path(tmp) / "workdir"
            workdir.mkdir()
            store = EvidenceStore.for_workdir(workdir, root=root)
            record = store.add("exec_shell_full_command", "raw content", metadata={"command": "pwd"})

            reloaded = EvidenceStore.for_workdir(workdir, root=root)

            self.assertEqual([item.evidence_id for item in reloaded.recent_records(1)], [record.evidence_id])
            self.assertEqual(reloaded.recent_records(1)[0].raw_ref, record.raw_ref)
            self.assertIn("command: pwd", reloaded.recent_records(1)[0].route_card)
            self.assertEqual(reloaded.load_raw(record.evidence_id), "raw content")

    def test_for_workdir_missing_sidecar_degrades_without_dropping_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            workdir = Path(tmp) / "workdir"
            workdir.mkdir()
            store = EvidenceStore.for_workdir(workdir, root=root)
            record = store.add("exec_shell_full_command", "raw content", metadata={"command": "pwd"})
            (store.root / f"{record.evidence_id}.txt").unlink()

            reloaded = EvidenceStore.for_workdir(workdir, root=root)
            loaded = reloaded.recent_records(1)[0]

            self.assertEqual(loaded.evidence_id, record.evidence_id)
            self.assertIn("raw_evidence_unavailable", reloaded.raw_excerpt(loaded))
            self.assertIn(record.raw_ref, reloaded.raw_excerpt(loaded))

    def test_same_raw_with_different_metadata_keeps_distinct_records_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            first = store.add("exec_shell_full_command", "same raw", metadata={"command": "printf one"})
            second = store.add("exec_shell_full_command", "same raw", metadata={"command": "printf two"})

            self.assertNotEqual(first.evidence_id, second.evidence_id)
            self.assertEqual(first.raw_sha256, second.raw_sha256)
            self.assertIn("command: printf one", first.route_card)
            self.assertIn("command: printf two", second.route_card)
            self.assertEqual(store.recent_records(2), [first, second])

            reloaded = EvidenceStore(store.root)
            reloaded.load_index()
            self.assertEqual([record.evidence_id for record in reloaded.recent_records(2)], [first.evidence_id, second.evidence_id])

    def test_recent_records_returns_latest_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            first = store.add("exec_shell_full_command", "one", metadata={"command": "printf one"})
            second = store.add("exec_shell_full_command", "two", metadata={"command": "printf two"})

            self.assertEqual(store.recent_records(1), [second])
            self.assertEqual(store.recent_records(2), [first, second])

    def test_large_shell_output_card_omits_raw_middle(self) -> None:
        content = "head\n" + ("x" * 3000) + "\ntail"
        record = build_evidence_record("exec_shell_full_command", content, {"command": "cat large.txt"})

        self.assertIn("command: cat large.txt", record.final_card)
        self.assertIn("head", record.final_card)
        self.assertIn("tail", record.final_card)
        self.assertNotIn("x" * 1000, record.final_card)
        self.assertIn("raw_ref:", record.final_card)

    def test_shell_small_stdout_is_visible_in_route_projection(self) -> None:
        content = "/home/guelfoweb/LAB/orbit/workdir\n"
        record = build_evidence_record("exec_shell_full_command", content, {"command": "pwd"})

        self.assertIn("stdout_excerpt: /home/guelfoweb/LAB/orbit/workdir", record.route_card)
        self.assertIn("stdout_excerpt=/home/guelfoweb/LAB/orbit/workdir", build_post_tool_route_evidence_context(_store_with(record, content)) or "")
        self.assertIn("raw_ref:", record.route_card)
        self.assertIn("sha256:", record.route_card)
        self.assertIn("size:", record.route_card)

    def test_shell_small_stderr_is_visible_in_route_projection(self) -> None:
        content = "shell_command_failed: true\nexit_code: 2\nSTDOUT:\n(empty)\nSTDERR:\npermission denied\n"
        record = build_evidence_record("exec_shell_full_command", content, {"command": "cat secret"})

        self.assertIn("stderr_excerpt: permission denied", record.route_card)
        self.assertIn("stderr_excerpt=permission denied", build_post_tool_route_evidence_context(_store_with(record, content)) or "")
        self.assertIn("raw_ref:", record.route_card)

    def test_shell_large_route_excerpt_is_bounded(self) -> None:
        content = "head\n" + ("x" * 3000) + "\ntail"
        record = build_evidence_record("exec_shell_full_command", content, {"command": "cat large.txt"})

        self.assertIn("stdout_excerpt:", record.route_card)
        self.assertIn("head", record.route_card)
        self.assertIn("tail", record.route_card)
        self.assertNotIn("x" * 1000, record.route_card)

    def test_post_tool_route_excerpt_respects_small_cap(self) -> None:
        content = "\n".join(f"line-{index}" for index in range(120))
        record = build_evidence_record("exec_shell_full_command", content, {"command": "python3 print-lines.py"})
        context = build_post_tool_route_evidence_context(_store_with(record, content)) or ""

        self.assertIn("stdout_excerpt=", context)
        excerpt = context.split("stdout_excerpt=", 1)[1]
        self.assertLessEqual(len(excerpt), 80)
        self.assertIn("[...bounded...]", excerpt)
        self.assertNotIn("line-20 | line-21 | line-22 | line-23 | line-24", excerpt)

    def test_post_tool_route_card_uses_compact_operational_metadata(self) -> None:
        content = "useful value\n"
        record = build_evidence_record("exec_shell_full_command", content, {"command": "printf useful"})
        context = build_post_tool_route_evidence_context(_store_with(record, content)) or ""

        self.assertIn("tool_evidence_card=true", context)
        self.assertIn("sz=", context)
        self.assertIn("cmd=printf useful", context)
        self.assertIn("stdout_excerpt=useful value", context)
        self.assertNotIn("raw_ref=", context)
        self.assertNotIn("hash=", context)

    def test_post_tool_route_omits_command_for_medium_shell_output(self) -> None:
        content = "\n".join(f"line-{index}" for index in range(120))
        record = build_evidence_record("exec_shell_full_command", content, {"command": "python3 print-lines.py"})
        context = build_post_tool_route_evidence_context(_store_with(record, content)) or ""

        self.assertNotIn("cmd=python3 print-lines.py", context)
        self.assertIn("stdout_excerpt=", context)

    def test_unknown_small_output_has_bounded_route_excerpt(self) -> None:
        record = build_evidence_record("custom_tool", "useful value\n", {})

        self.assertIn("stdout_excerpt: useful value", record.route_card)
        self.assertIn("raw_ref:", record.route_card)

    def test_web_search_card_contains_query_status_count_and_ref(self) -> None:
        content = "\n".join(
            [
                "web_search_results: true",
                "query: OpenAI",
                "results:",
                "1. title: OpenAI",
                "   url: https://openai.com/",
                "   snippet: AI research and deployment.",
            ]
        )

        record = build_evidence_record("exec_shell_full_command", content, {"command": 'orbit-web-search "OpenAI"'})

        self.assertEqual(record.kind, "web_search")
        self.assertIn("query: OpenAI", record.route_card)
        self.assertIn("result_count: 1", record.route_card)
        self.assertIn("top_domains: openai.com", record.route_card)
        self.assertIn("raw_ref:", record.route_card)

    def test_grep_path_line_output_contains_paths_lines_and_matches(self) -> None:
        content = "\n".join(
            [
                "STDOUT:",
                "src/orbit/runtime/evidence.py:17:class EvidenceStore:",
                "tests/test_evidence.py:23:EvidenceStore(Path(tmp))",
                "STDERR:",
                "(empty)",
            ]
        )

        record = build_evidence_record(
            "exec_shell_full_command",
            content,
            {"command": 'grep -R "EvidenceStore" src tests'},
        )

        self.assertEqual(record.kind, "grep_search")
        self.assertIn("query: EvidenceStore", record.route_card)
        self.assertIn("files_count: 2", record.route_card)
        self.assertIn("match_count: 2", record.route_card)
        self.assertIn("src/orbit/runtime/evidence.py", record.route_card)
        self.assertIn("src/orbit/runtime/evidence.py:17: class EvidenceStore:", record.final_card)
        self.assertIn(record.raw_ref, record.final_card)

    def test_grep_list_only_output_contains_file_list_without_fake_matches(self) -> None:
        content = "\n".join(
            [
                "STDOUT:",
                "src/orbit/runtime/evidence.py",
                "tests/test_evidence.py",
                "STDERR:",
                "(empty)",
            ]
        )

        record = build_evidence_record(
            "exec_shell_full_command",
            content,
            {"command": 'grep -l "EvidenceStore" .'},
        )

        self.assertIn("files_count: 2", record.route_card)
        self.assertIn("file_paths: src/orbit/runtime/evidence.py; tests/test_evidence.py", record.route_card)
        self.assertNotIn("match_count:", record.route_card)
        self.assertIn("- src/orbit/runtime/evidence.py", record.final_card)

    def test_failed_grep_with_empty_stdout_does_not_invent_matches(self) -> None:
        content = "\n".join(
            [
                "shell_command_failed: true",
                "exit_code: 1",
                "STDOUT:",
                "(empty)",
                "STDERR:",
                "(empty)",
            ]
        )

        record = build_evidence_record(
            "exec_shell_full_command",
            content,
            {"command": 'grep -r "EvidenceStore" .'},
        )

        self.assertEqual(record.kind, "grep_search")
        self.assertIn("status: error", record.route_card)
        self.assertIn("query: EvidenceStore", record.route_card)
        self.assertNotIn("match_count:", record.route_card)
        self.assertNotIn("files_count:", record.route_card)
        self.assertIn("raw_ref:", record.route_card)

    def test_unknown_large_tool_output_has_head_tail_hash_and_ref(self) -> None:
        content = "alpha\n" + ("body" * 1000) + "\nomega"

        record = build_evidence_record("custom_tool", content, {})

        self.assertEqual(record.kind, "unknown")
        self.assertIn("alpha", record.final_card)
        self.assertIn("omega", record.final_card)
        self.assertIn("sha256:", record.final_card)
        self.assertIn("raw_ref:", record.final_card)

    def test_session_save_load_keeps_card_promptable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            workdir = Path(tmp) / "workdir"
            workdir.mkdir()
            store = EvidenceStore.for_workdir(workdir, root=root)
            record = store.add("exec_shell_full_command", "raw " * 1000, metadata={"command": "cat big.txt"})
            session = SessionStore.for_workdir(workdir, root=root)
            session.save(
                messages=[
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "name": "exec_shell_full_command",
                        "content": tool_evidence_ref(record),
                        "evidence_id": record.evidence_id,
                    }
                ],
                workdir=workdir,
                model="m",
                base_url="u",
            )

            loaded = session.load()

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertIn("tool_evidence_ref: true", loaded[0]["content"])
            self.assertLess(len(str(loaded[0]["content"])), len("raw " * 1000))
            self.assertEqual(store.load_raw(record.evidence_id), "raw " * 1000)

    def test_tool_evidence_ref_is_audit_marker_not_card(self) -> None:
        record = build_evidence_record("exec_shell_full_command", "raw " * 1000, {"command": "cat big.txt"})

        marker = tool_evidence_ref(record)

        self.assertIn("tool_evidence_ref: true", marker)
        self.assertIn(record.raw_ref, marker)
        self.assertIn(record.evidence_id, marker)
        self.assertNotIn("tool_evidence_card: true", marker)
        self.assertNotIn("raw raw raw", marker)

    def test_route_context_exposes_store_cards_without_raw_or_policy_instruction(self) -> None:
        raw = "web_search_results: true\nquery: OpenAI\nresults:\n" + ("raw-result " * 500)
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add("exec_shell_full_command", raw, metadata={"command": 'orbit-web-search "OpenAI"'})

            context = build_route_evidence_context(store)

            self.assertIsNotNone(context)
            assert context is not None
            self.assertIn("available_evidence:", context)
            self.assertIn("tool_evidence_card: true", context)
            self.assertIn(record.raw_ref, context)
            self.assertNotIn("choose CHAT", context)
            self.assertNotIn("choose FINAL", context)
            self.assertNotIn(raw, context)
            self.assertLess(len(context), len(raw))

    def test_final_context_uses_bounded_excerpt_and_degrades_when_sidecar_missing(self) -> None:
        raw = "head\n" + ("body " * 500) + "\ntail"
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add("exec_shell_full_command", raw, metadata={"command": "cat big.txt"})

            context = build_final_evidence_context(store)
            self.assertIsNotNone(context)
            assert context is not None
            self.assertIn("evidence_context:", context)
            self.assertIn("bounded_raw_excerpt:", context)
            self.assertIn("head", context)
            self.assertIn("tail", context)
            self.assertNotIn("body " * 200, context)

            store.raw_cache.clear()
            (store.root / f"{record.evidence_id}.txt").unlink()
            missing = store.raw_excerpt(record)
            self.assertIn("raw_evidence_unavailable", missing)
            self.assertIn(record.raw_ref, missing)

    def test_final_context_does_not_duplicate_structured_grep_raw_excerpt(self) -> None:
        raw = "\n".join(
            [
                "STDOUT:",
                "src/orbit/runtime/evidence.py:17:class EvidenceStore:",
                "tests/test_evidence.py:23:EvidenceStore(Path(tmp))",
                "STDERR:",
                "(empty)",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add(
                "exec_shell_full_command",
                raw,
                metadata={"command": 'grep -R "EvidenceStore" src tests'},
            )

            context = build_final_evidence_context(store)

            self.assertIsNotNone(context)
            assert context is not None
            self.assertIn(record.raw_ref, context)
            self.assertIn("src/orbit/runtime/evidence.py:17: class EvidenceStore:", context)
            self.assertNotIn("bounded_raw_excerpt:", context)

    def test_final_context_does_not_duplicate_structured_web_raw_excerpt(self) -> None:
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

            context = build_final_evidence_context(store)

            self.assertIsNotNone(context)
            assert context is not None
            self.assertIn(record.raw_ref, context)
            self.assertIn("AI research and deployment.", context)
            self.assertNotIn("bounded_raw_excerpt:", context)

    def test_compact_final_context_does_not_duplicate_shell_excerpt(self) -> None:
        raw = "\n".join(f"line-{index}" for index in range(120))
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add(
                "exec_shell_full_command",
                raw,
                metadata={"command": "python3 print-lines.py"},
            )

            context = build_compact_final_evidence_context(store)

            self.assertIsNotNone(context)
            assert context is not None
            self.assertIn(record.raw_ref, context)
            self.assertIn(record.raw_sha256[:16], context)
            self.assertIn("stdout_excerpt:", context)
            self.assertIn("line-0", context)
            self.assertIn("line-119", context)
            self.assertNotIn("bounded_raw_excerpt:", context)

    def test_compact_final_context_keeps_error_raw_excerpt(self) -> None:
        raw = "shell_command_failed: true\nexit_code: 127\nSTDOUT:\n(empty)\nSTDERR:\nmissing command\n"
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            store.add("exec_shell_full_command", raw, metadata={"command": "missing-command"})

            context = build_compact_final_evidence_context(store)

            self.assertIsNotNone(context)
            assert context is not None
            self.assertIn("bounded_raw_excerpt:", context)
            self.assertIn("shell_command_failed: true", context)
            self.assertIn("missing command", context)

    def test_web_final_context_is_structured_bounded_and_keeps_refs(self) -> None:
        raw = "\n".join(
            [
                "web_search_results: true",
                "query: OpenAI",
                "results:",
                "1. title: OpenAI",
                "   url: https://openai.com/",
                "   snippet: " + ("AI research and deployment " * 40),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "session.evidence")
            record = store.add(
                "exec_shell_full_command",
                raw,
                metadata={"command": 'orbit-web-search "OpenAI"'},
            )

            context = build_web_final_evidence_context(store)

            self.assertIsNotNone(context)
            assert context is not None
            self.assertIn("web_search_evidence: true", context)
            self.assertIn("query: OpenAI", context)
            self.assertIn(record.raw_ref, context)
            self.assertIn(record.raw_sha256[:16], context)
            self.assertIn(f"size: {record.raw_chars} chars", context)
            self.assertIn("top_snippets:", context)
            self.assertNotIn("bounded_raw_excerpt:", context)
            self.assertNotIn("AI research and deployment " * 20, context)


def _store_with(record, content: str) -> EvidenceStore:
    store = EvidenceStore(Path("/tmp/orbit-test-unused-session.evidence"))
    store.records[record.evidence_id] = record
    store.raw_cache[record.evidence_id] = content
    return store


if __name__ == "__main__":
    unittest.main()
