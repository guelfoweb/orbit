from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.tools import (
    DEFAULT_CHUNK_CHARS,
    MAX_CHUNK_CALLS_PER_TURN,
    MAX_CHUNK_CHARS,
    MAX_APPEND_CHARS,
    MAX_REPLACE_CHARS,
    MAX_TEXT_FILE_BYTES_AFTER_APPEND,
    MAX_TEXT_FILE_BYTES_AFTER_REPLACE,
    MAX_WRITE_CHARS,
    tool_definitions,
    tool_names,
    execute_tool,
)
from orbit.runtime.web import MAX_FETCH_CHUNK_CALLS_PER_TURN


class ToolTests(unittest.TestCase):
    def test_list_files_is_confined_to_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()

            result = execute_tool("list_files", {"path": ".."}, workdir=workdir)

        self.assertIn("escapes workdir", result.content)

    def test_list_files_returns_bounded_directory_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "b.txt").write_text("b", encoding="utf-8")
            (workdir / "a").mkdir()

            result = execute_tool("list_files", {"path": "."}, workdir=workdir)

        self.assertEqual(result.content.splitlines(), ["a/", "b.txt"])

    def test_stat_path_returns_file_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello", encoding="utf-8")

            result = execute_tool("stat_path", {"path": "note.txt"}, workdir=workdir)

        self.assertIn("path: note.txt", result.content)
        self.assertIn("exists: true", result.content)
        self.assertIn("type: file", result.content)
        self.assertIn("size_bytes: 5", result.content)
        self.assertIn("modified:", result.content)

    def test_stat_path_returns_directory_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "docs").mkdir()

            result = execute_tool("stat_path", {"path": "docs"}, workdir=workdir)

        self.assertIn("exists: true", result.content)
        self.assertIn("type: directory", result.content)

    def test_stat_path_returns_missing_metadata_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            result = execute_tool("stat_path", {"path": "missing.txt"}, workdir=workdir)

        self.assertEqual(result.content, "path: missing.txt\nexists: false")

    def test_stat_path_is_confined_to_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()

            result = execute_tool("stat_path", {"path": ".."}, workdir=workdir)

        self.assertIn("escapes workdir", result.content)

    def test_make_directory_creates_nested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            result = execute_tool("make_directory", {"path": "build/output"}, workdir=workdir)

            self.assertTrue((workdir / "build" / "output").is_dir())

        self.assertIn("created: true", result.content)
        self.assertIn("type: directory", result.content)

    def test_make_directory_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()

            result = execute_tool("make_directory", {"path": "../outside"}, workdir=workdir)

        self.assertIn("escapes workdir", result.content)

    def test_delete_path_removes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "note.txt"
            target.write_text("x", encoding="utf-8")

            result = execute_tool("delete_path", {"path": "note.txt"}, workdir=workdir)

            self.assertFalse(target.exists())

        self.assertIn("deleted: true", result.content)
        self.assertIn("type: file", result.content)

    def test_delete_path_rejects_non_empty_directory_without_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "docs"
            target.mkdir()
            (target / "note.txt").write_text("x", encoding="utf-8")

            result = execute_tool("delete_path", {"path": "docs"}, workdir=workdir)

            self.assertTrue(target.exists())

        self.assertIn("recursive=true", result.content)

    def test_delete_path_removes_directory_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "docs"
            target.mkdir()
            (target / "note.txt").write_text("x", encoding="utf-8")

            result = execute_tool("delete_path", {"path": "docs", "recursive": True}, workdir=workdir)

            self.assertFalse(target.exists())

        self.assertIn("deleted: true", result.content)
        self.assertIn("type: directory", result.content)

    def test_delete_path_refuses_workdir_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            result = execute_tool("delete_path", {"path": ".", "recursive": True}, workdir=workdir)

            self.assertTrue(workdir.exists())

        self.assertIn("refusing to delete the workdir root", result.content)

    def test_delete_path_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()

            result = execute_tool("delete_path", {"path": "../outside", "recursive": True}, workdir=workdir)

        self.assertIn("escapes workdir", result.content)

    def test_read_file_reads_utf8_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.md").write_text("# Title\nhello", encoding="utf-8")

            result = execute_tool("read_file", {"path": "note.md"}, workdir=workdir)

        self.assertEqual(result.content, "# Title\nhello")

    def test_read_file_rejects_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "doc.pdf").write_bytes(b"%PDF-1.7")

            result = execute_tool("read_file", {"path": "doc.pdf"}, workdir=workdir)

        self.assertIn("PDF requires read_pdf", result.content)

    def test_read_file_rejects_binary_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "binary").write_bytes(b"a\x00b")

            result = execute_tool("read_file", {"path": "binary"}, workdir=workdir)

        self.assertIn("binary", result.content)

    def test_read_file_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()

            result = execute_tool("read_file", {"path": "../secret.txt"}, workdir=workdir)

        self.assertIn("escapes workdir", result.content)

    def test_read_file_large_file_returns_initial_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "large.txt").write_text("x" * (257 * 1024), encoding="utf-8")

            result = execute_tool("read_file", {"path": "large.txt"}, workdir=workdir)

        self.assertIn("large_file_excerpt: true", result.content)
        self.assertIn("content:", result.content)

    def test_read_file_chunk_mode_returns_real_chunk_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "large.txt").write_text("abcdef" * 50000, encoding="utf-8")

            result = execute_tool("read_file", {"path": "large.txt", "chunk_index": 1, "chunk_chars": 2}, workdir=workdir)

        self.assertIn("chunk_index: 1", result.content)
        self.assertIn("total_chunks:", result.content)

    def test_read_file_chunk_mode_rejects_oversized_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "large.txt").write_text("abcdef" * 50000, encoding="utf-8")

            result = execute_tool("read_file", {"path": "large.txt", "chunk_index": 0, "chunk_chars": 999999}, workdir=workdir)

        self.assertIn("chunk_chars too large", result.content)

    def test_read_file_chunk_limits_are_cpu_friendly(self) -> None:
        self.assertEqual(DEFAULT_CHUNK_CHARS, 6_000)
        self.assertEqual(MAX_CHUNK_CHARS, 12_000)

    def test_read_file_schema_keeps_chunk_index_optional(self) -> None:
        read_file = next(tool for tool in tool_definitions() if tool["function"]["name"] == "read_file")

        self.assertEqual(read_file["function"]["parameters"]["required"], ["path"])

    def test_tool_definitions_are_serialization_stable(self) -> None:
        first = json.dumps(tool_definitions(), sort_keys=True, separators=(",", ":"))
        second = json.dumps(tool_definitions(), sort_keys=True, separators=(",", ":"))

        self.assertEqual(first, second)
        self.assertIn("fetch_url", first)
        self.assertIn("list_files", first)
        self.assertIn("read_file", first)
        self.assertIn("search_web", first)
        self.assertIn("stat_path", first)

    def test_tool_names_match_tool_definitions(self) -> None:
        defined = tuple(tool["function"]["name"] for tool in tool_definitions())

        self.assertEqual(tool_names(), defined)

    def test_tool_definitions_can_be_filtered_by_name(self) -> None:
        defined = tuple(tool["function"]["name"] for tool in tool_definitions(("list_files", "read_file")))

        self.assertEqual(defined, ("list_files", "read_file"))

    def test_search_web_empty_query_is_rejected_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            result = execute_tool("search_web", {"query": ""}, workdir=workdir)

        self.assertIn("query must be a non-empty string", result.content)

    def test_search_web_rejects_invalid_site_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            result = execute_tool("search_web", {"query": "orbit", "site": "https://example.com"}, workdir=workdir)

        self.assertIn("bare domain", result.content)

    def test_write_file_creates_new_utf8_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            result = execute_tool("write_file", {"path": "note.md", "content": "# Hello\n"}, workdir=workdir)

            self.assertEqual((workdir / "note.md").read_text(encoding="utf-8"), "# Hello\n")

        self.assertIn("created: true", result.content)
        self.assertIn("chars: 8", result.content)

    def test_write_file_refuses_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.md").write_text("old", encoding="utf-8")

            result = execute_tool("write_file", {"path": "note.md", "content": "new"}, workdir=workdir)

            self.assertEqual((workdir / "note.md").read_text(encoding="utf-8"), "old")

        self.assertIn("refusing to overwrite", result.content)

    def test_write_file_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()

            result = execute_tool("write_file", {"path": "../note.md", "content": "x"}, workdir=workdir)

        self.assertIn("escapes workdir", result.content)

    def test_write_file_requires_existing_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            result = execute_tool("write_file", {"path": "missing/note.md", "content": "x"}, workdir=workdir)

        self.assertIn("parent directory does not exist", result.content)

    def test_write_file_rejects_binary_extension_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            extension_result = execute_tool("write_file", {"path": "image.png", "content": "x"}, workdir=workdir)
            binary_result = execute_tool("write_file", {"path": "note.txt", "content": "a\x00b"}, workdir=workdir)

        self.assertIn("unsupported file type", extension_result.content)
        self.assertIn("binary", binary_result.content)

    def test_write_file_rejects_oversized_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            result = execute_tool("write_file", {"path": "note.txt", "content": "x" * (MAX_WRITE_CHARS + 1)}, workdir=workdir)

        self.assertIn("content too large", result.content)

    def test_append_file_appends_to_existing_utf8_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("first\n", encoding="utf-8")

            result = execute_tool("append_file", {"path": "note.txt", "content": "second\n"}, workdir=workdir)

            self.assertEqual((workdir / "note.txt").read_text(encoding="utf-8"), "first\nsecond\n")

        self.assertIn("appended: true", result.content)
        self.assertIn("chars_added: 7", result.content)

    def test_append_file_requires_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            result = execute_tool("append_file", {"path": "note.txt", "content": "x"}, workdir=workdir)

        self.assertIn("Use write_file", result.content)

    def test_append_file_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()

            result = execute_tool("append_file", {"path": "../note.txt", "content": "x"}, workdir=workdir)

        self.assertIn("escapes workdir", result.content)

    def test_append_file_rejects_binary_extension_and_existing_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "image.png").write_text("not really png", encoding="utf-8")
            (workdir / "note.txt").write_bytes(b"a\x00b")

            extension_result = execute_tool("append_file", {"path": "image.png", "content": "x"}, workdir=workdir)
            binary_result = execute_tool("append_file", {"path": "note.txt", "content": "x"}, workdir=workdir)

        self.assertIn("unsupported file type", extension_result.content)
        self.assertIn("binary", binary_result.content)

    def test_append_file_rejects_oversized_append_and_final_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("x", encoding="utf-8")
            (workdir / "large.txt").write_text("x" * MAX_TEXT_FILE_BYTES_AFTER_APPEND, encoding="utf-8")

            append_result = execute_tool("append_file", {"path": "note.txt", "content": "x" * (MAX_APPEND_CHARS + 1)}, workdir=workdir)
            total_result = execute_tool("append_file", {"path": "large.txt", "content": "x"}, workdir=workdir)

        self.assertIn("content too large", append_result.content)
        self.assertIn("append would make file too large", total_result.content)

    def test_replace_in_file_replaces_unique_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("hello old world\n", encoding="utf-8")

            result = execute_tool("replace_in_file", {"path": "note.txt", "old": "old", "new": "new"}, workdir=workdir)

            self.assertEqual((workdir / "note.txt").read_text(encoding="utf-8"), "hello new world\n")

        self.assertIn("replaced: true", result.content)
        self.assertIn("matches: 1", result.content)

    def test_replace_in_file_rejects_missing_or_ambiguous_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("one two two\n", encoding="utf-8")

            missing = execute_tool("replace_in_file", {"path": "note.txt", "old": "three", "new": "x"}, workdir=workdir)
            ambiguous = execute_tool("replace_in_file", {"path": "note.txt", "old": "two", "new": "x"}, workdir=workdir)

            self.assertEqual((workdir / "note.txt").read_text(encoding="utf-8"), "one two two\n")

        self.assertIn("old text not found", missing.content)
        self.assertIn("ambiguous", ambiguous.content)

    def test_replace_in_file_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()

            result = execute_tool("replace_in_file", {"path": "../note.txt", "old": "a", "new": "b"}, workdir=workdir)

        self.assertIn("escapes workdir", result.content)

    def test_replace_in_file_rejects_binary_extension_and_existing_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "image.png").write_text("old", encoding="utf-8")
            (workdir / "note.txt").write_bytes(b"a\x00b")

            extension_result = execute_tool("replace_in_file", {"path": "image.png", "old": "old", "new": "new"}, workdir=workdir)
            binary_result = execute_tool("replace_in_file", {"path": "note.txt", "old": "a", "new": "b"}, workdir=workdir)

        self.assertIn("unsupported file type", extension_result.content)
        self.assertIn("binary", binary_result.content)

    def test_replace_in_file_rejects_oversized_text_and_final_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("old", encoding="utf-8")
            (workdir / "large.txt").write_text("old" + ("x" * (MAX_TEXT_FILE_BYTES_AFTER_REPLACE - 4)), encoding="utf-8")

            old_result = execute_tool("replace_in_file", {"path": "note.txt", "old": "x" * (MAX_REPLACE_CHARS + 1), "new": "x"}, workdir=workdir)
            new_result = execute_tool("replace_in_file", {"path": "note.txt", "old": "old", "new": "x" * (MAX_REPLACE_CHARS + 1)}, workdir=workdir)
            final_result = execute_tool("replace_in_file", {"path": "large.txt", "old": "old", "new": "y" * 16}, workdir=workdir)

        self.assertIn("old text too large", old_result.content)
        self.assertIn("new text too large", new_result.content)
        self.assertIn("replacement would make file too large", final_result.content)

    def test_fetch_url_chunk_budget_is_per_turn_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            budget = {"fetch_url_chunks": MAX_FETCH_CHUNK_CALLS_PER_TURN}

            result = execute_tool(
                "fetch_url",
                {"url": "http://example.test", "chunk_index": 0},
                workdir=workdir,
                chunk_budget=budget,
            )

        self.assertIn("fetch_url chunk budget exceeded", result.content)

    def test_read_file_chunk_mode_default_size_is_large_enough_for_real_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "large.txt").write_text("a" * (300 * 1024), encoding="utf-8")

            result = execute_tool("read_file", {"path": "large.txt", "chunk_index": 0}, workdir=workdir)

        self.assertIn(f"chars: 0-{DEFAULT_CHUNK_CHARS}", result.content)

    def test_read_file_chunk_budget_is_per_turn_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "large.txt").write_text("abcdef" * 50000, encoding="utf-8")
            budget = {"read_file_chunks": MAX_CHUNK_CALLS_PER_TURN}

            result = execute_tool(
                "read_file",
                {"path": "large.txt", "chunk_index": 0},
                workdir=workdir,
                chunk_budget=budget,
            )

        self.assertIn("chunk budget exceeded", result.content)

    def test_unknown_chunk_tool_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            result = execute_tool("read_chunk", {"path": "doc.pdf", "chunk_index": 0}, workdir=workdir)

        self.assertIn("unknown tool", result.content)


if __name__ == "__main__":
    unittest.main()
