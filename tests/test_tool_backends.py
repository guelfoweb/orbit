from __future__ import annotations

import sys
import tempfile
import unittest
from shutil import copyfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.tool_backends import HybridToolExecutor


class FakeServerTools:
    def __init__(self) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []

    def server_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "tool": "read_file",
                "definition": {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "server read",
                        "parameters": {"type": "object"},
                    },
                },
            }
        ]

    def execute_server_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.executed.append((name, arguments))
        return "server result"


class HybridToolExecutorTests(unittest.TestCase):
    def test_exposes_only_allowed_shell_full_definition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

            definitions = executor.tool_definitions()

        self.assertEqual([item["function"]["name"] for item in definitions], ["exec_shell_full_command"])
        description = definitions[0]["function"]["description"]
        self.assertIn("Local shell confined to the current workdir", description)
        self.assertIn("For URLs, use curl when content is needed", description)
        self.assertIn("Quote paths containing spaces", description)

    def test_ignores_server_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("read_file",),
            )

            execution = executor.execute("read_file", {"path": "note.txt"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("unknown tool", execution.result.content)
        self.assertEqual(backend.executed, [])

    def test_exec_shell_full_runs_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "printf x | wc -c"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertEqual(execution.result.content.strip(), "1")
        self.assertEqual(backend.executed, [])

    def test_exec_shell_full_rejects_unavailable_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=(),
            )

            execution = executor.execute("exec_shell_full_command", {"command": "pwd"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("tool not available", execution.result.content)

    def test_exec_shell_full_blocks_metadata_only_analysis_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                user_prompt="analyze samples/vulnerable_service.py for vulnerabilities",
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "ls -R samples/"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("require content/source/string evidence", execution.result.content)

    def test_exec_shell_full_allows_listing_when_not_analysis_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                user_prompt="list the samples directory",
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "ls -R ."},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertNotIn("require content/source/string evidence", execution.result.content)

    def test_exec_shell_full_rejects_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "cat /etc/passwd"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("must stay inside workdir", execution.result.content)

    def test_exec_shell_full_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "cat ../secret.txt"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("must stay inside workdir", execution.result.content)

    def test_exec_shell_full_rejects_cd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "cd .. && ls"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("directory-changing commands are not allowed", execution.result.content)

    def test_exec_shell_full_cat_large_text_uses_read_file_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "long.txt"
            target.write_text("alpha\n" * 2000, encoding="utf-8")
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute("exec_shell_full_command", {"command": "cat long.txt"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("shell_output_read_file: true", execution.result.content)
        self.assertIn("original_command: cat long.txt", execution.result.content)
        self.assertIn("chunk_index: 0", execution.result.content)

    def test_exec_shell_full_cat_small_text_keeps_raw_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "small.txt").write_text("alpha\nbeta\n", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute("exec_shell_full_command", {"command": "cat small.txt"}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertEqual(execution.result.content, "alpha\nbeta")

    def test_exec_shell_full_bounds_large_search_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "matches.txt").write_text(("Virgilio matched line\n" * 300), encoding="utf-8")
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "grep Virgilio matches.txt"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertLessEqual(len(execution.result.content.encode("utf-8")), 900)
        self.assertIn("[truncated]", execution.result.content)

    def test_exec_shell_full_cleans_html_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "printf '<!doctype html><html><body><h1>Title</h1><script>x()</script><p>Hello world</p></body></html>'"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("shell_output_html_cleaned: true", execution.result.content)
        self.assertIn("Title", execution.result.content)
        self.assertIn("Hello world", execution.result.content)
        self.assertNotIn("<html>", execution.result.content)

    def test_exec_shell_full_cleans_html_fragment_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {
                    "command": (
                        "printf '<tr><td>Dante Alighieri</td><td>Italian poet</td></tr>"
                        "<p>Divine Comedy</p>'"
                    )
                },
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("shell_output_html_cleaned: true", execution.result.content)
        self.assertIn("Dante Alighieri", execution.result.content)
        self.assertIn("Divine Comedy", execution.result.content)
        self.assertNotIn("<td>", execution.result.content)

    def test_exec_shell_full_preserves_html_source_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
                user_prompt="analyze the HTML source code of this page",
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "printf '<html><body><script>x()</script><p>Hello world</p></body></html>'"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("<script>", execution.result.content)
        self.assertNotIn("shell_output_html_cleaned: true", execution.result.content)

    def test_exec_shell_full_does_not_reinject_unreadable_html_fragment_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "printf '<script>var x=1; Dante Alighieri</script><td>Italian poet</td>'"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("shell_output_html_cleaned: true", execution.result.content)
        self.assertIn("Italian poet", execution.result.content)
        self.assertNotIn("<script>", execution.result.content)

    def test_exec_shell_full_extracts_pdf_text_with_pdftotext(self) -> None:
        source = ROOT / "workdir" / "pdf" / "piccolo.pdf"
        if not source.exists():
            self.skipTest("pdf fixture unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "pdf").mkdir()
            copyfile(source, workdir / "pdf" / "piccolo.pdf")
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "cat pdf/piccolo.pdf"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("shell_output_pdf_text: true", execution.result.content)
        self.assertIn("extractor: pdftotext", execution.result.content)
        self.assertNotIn("%PDF", execution.result.content)

    def test_exec_shell_full_extracts_large_pdf_as_chunk(self) -> None:
        source = ROOT / "workdir" / "pdf" / "grande.pdf"
        if not source.exists():
            self.skipTest("pdf fixture unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "pdf").mkdir()
            copyfile(source, workdir / "pdf" / "grande.pdf")
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "pdftotext pdf/grande.pdf -"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("shell_output_pdf_text: true", execution.result.content)
        self.assertIn("chunk_index: 0", execution.result.content)
        self.assertIn("total_chunks:", execution.result.content)

    def test_exec_shell_full_extracts_pdf_text_with_strings_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "sample.pdf"
            target.write_bytes(b"%PDF-1.4\nOrbit PDF fallback visible text\n%%EOF\n")
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

            with patch("orbit.runtime.shell_guardrails.shutil.which") as which:
                which.side_effect = lambda name: None if name == "pdftotext" else "/usr/bin/strings"
                execution = executor.execute(
                    "exec_shell_full_command",
                    {"command": "cat sample.pdf"},
                    chunk_budget={},
                )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("shell_output_pdf_text: true", execution.result.content)
        self.assertIn("extractor: strings", execution.result.content)
        self.assertIn("Orbit PDF fallback visible text", execution.result.content)
        self.assertNotIn("%PDF", execution.result.content)


if __name__ == "__main__":
    unittest.main()
