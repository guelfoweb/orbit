from __future__ import annotations

import sys
import io
import subprocess
import tempfile
import time
import unittest
from shutil import copyfile
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.tool_backends import HybridToolExecutor
from orbit.runtime.shell_guardrails import (
    SHELL_FULL_CONTRACT_ERROR_PREFIX,
    looks_like_transfer_progress_only,
    looks_like_url_content_evidence,
    looks_like_url_fetch_failure,
    validate_shell_full_contract,
)


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
    def test_shell_full_contract_rejects_metadata_only_analysis_in_italian(self) -> None:
        error = validate_shell_full_contract(
            {"command": "ls -F pdf/"},
            user_prompt='analizza intero documento PDF nella cartella "pdf/RELAZIONE TECNICA 1.pdf" e fammi una sintesi dettagliata',
        )

        self.assertIsNotNone(error)
        self.assertTrue(error.startswith(SHELL_FULL_CONTRACT_ERROR_PREFIX))

    def test_shell_full_contract_allows_content_evidence_analysis_in_italian(self) -> None:
        error = validate_shell_full_contract(
            {"command": 'pdftotext "pdf/RELAZIONE TECNICA 1.pdf" - | head -n 40'},
            user_prompt='analizza intero documento PDF nella cartella "pdf/RELAZIONE TECNICA 1.pdf" e fammi una sintesi dettagliata',
        )

        self.assertIsNone(error)

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
        self.assertIn("Unrestricted local shell launched from the current workdir", description)
        self.assertIn("access paths outside workdir", description)
        self.assertIn("orbit-web-search", description)
        self.assertIn("explicit URL content requests", description)
        self.assertIn("fetch_url", description)
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

    def test_exec_shell_full_runs_internal_web_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeServerTools()
            executor = HybridToolExecutor(
                backend=backend,
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

            with patch("orbit.runtime.shell_guardrails.search_web", return_value="web_search_results: true\nquery: Dante"):
                execution = executor.execute(
                    "exec_shell_full_command",
                    {"command": 'orbit-web-search "Dante"'},
                    chunk_budget={},
                )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("web_search_results: true", execution.result.content)
        self.assertEqual(backend.executed, [])

    def test_fetch_url_runs_internal_url_fetch(self) -> None:
        class FakeHeaders:
            def get_content_type(self) -> str:
                return "text/html"

            def get_content_charset(self) -> str:
                return "utf-8"

        class FakeResponse:
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def geturl(self) -> str:
                return "https://example.com/final"

            def getcode(self) -> int:
                return 200

            def read(self, amount: int = -1) -> bytes:
                del amount
                return b"<html><title>Example</title><body>Hello web</body></html>"

        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("fetch_url",),
            )

            with patch("orbit.runtime.web.urlopen", return_value=FakeResponse()):
                execution = executor.execute(
                    "fetch_url",
                    {"url": "https://example.com"},
                    chunk_budget={},
                )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("status: ok", execution.result.content)
        self.assertIn("Hello web", execution.result.content)

    def test_system_info_runs_internal_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("system_info",),
            )

            execution = executor.execute("system_info", {"include_disks": False}, chunk_budget={})

        self.assertEqual(execution.source, "orbit")
        self.assertIn("system_info:", execution.result.content)
        self.assertNotIn("Disk:", execution.result.content)

    def test_fetch_url_plain_text_returns_text(self) -> None:
        class FakeHeaders:
            def get_content_type(self) -> str:
                return "text/plain"

            def get_content_charset(self) -> str:
                return "utf-8"

        class FakeResponse:
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def geturl(self) -> str:
                return "https://example.com/plain"

            def getcode(self) -> int:
                return 200

            def read(self, amount: int = -1) -> bytes:
                del amount
                return b"plain text body"

        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("fetch_url",),
            )
            with patch("orbit.runtime.web.urlopen", return_value=FakeResponse()):
                execution = executor.execute("fetch_url", {"url": "https://example.com/plain"}, chunk_budget={})

        self.assertIn("status: ok", execution.result.content)
        self.assertIn("plain text body", execution.result.content)

    def test_fetch_url_json_returns_text(self) -> None:
        class FakeHeaders:
            def get_content_type(self) -> str:
                return "application/json"

            def get_content_charset(self) -> str:
                return "utf-8"

        class FakeResponse:
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def geturl(self) -> str:
                return "https://example.com/data.json"

            def getcode(self) -> int:
                return 200

            def read(self, amount: int = -1) -> bytes:
                del amount
                return b'{\"central_thesis\":\"human dignity\"}'

        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("fetch_url",),
            )
            with patch("orbit.runtime.web.urlopen", return_value=FakeResponse()):
                execution = executor.execute("fetch_url", {"url": "https://example.com/data.json"}, chunk_budget={})

        self.assertIn("status: ok", execution.result.content)
        self.assertIn("central_thesis", execution.result.content)

    def test_fetch_url_xml_returns_text(self) -> None:
        class FakeHeaders:
            def get_content_type(self) -> str:
                return "application/xml"

            def get_content_charset(self) -> str:
                return "utf-8"

        class FakeResponse:
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def geturl(self) -> str:
                return "https://example.com/data.xml"

            def getcode(self) -> int:
                return 200

            def read(self, amount: int = -1) -> bytes:
                del amount
                return b"<doc><thesis>human dignity</thesis></doc>"

        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("fetch_url",),
            )
            with patch("orbit.runtime.web.urlopen", return_value=FakeResponse()):
                execution = executor.execute("fetch_url", {"url": "https://example.com/data.xml"}, chunk_budget={})

        self.assertIn("status: ok", execution.result.content)
        self.assertIn("<doc><thesis>human dignity</thesis></doc>", execution.result.content)

    def test_exec_shell_full_rejects_empty_internal_web_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "orbit-web-search"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertIn("requires a query", execution.result.content)

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

    def test_validate_shell_full_contract_rejects_explicit_url_without_fetch(self) -> None:
        error = validate_shell_full_contract(
            {"command": 'orbit-web-search "vatican va encyclical"'},
            user_prompt="Fetch https://example.com/doc and explain the central thesis.",
        )

        self.assertIsNotNone(error)
        self.assertIn("explicit URL requests require direct URL fetch", error)

    def test_url_progress_meter_only_is_not_content_evidence(self) -> None:
        progress = (
            "% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n"
            "                                 Dload  Upload   Total   Spent    Left  Speed\n"
            "  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0\n"
        )

        self.assertTrue(looks_like_transfer_progress_only(progress))
        self.assertFalse(looks_like_url_content_evidence(progress))
        self.assertFalse(looks_like_url_fetch_failure(progress))

    def test_url_html_cleaned_output_is_content_evidence(self) -> None:
        content = "shell_output_html_cleaned: true\ntext:\nCentral thesis: human dignity and solidarity."

        self.assertTrue(looks_like_url_content_evidence(content))
        self.assertFalse(looks_like_url_fetch_failure(content))

    def test_url_http_404_text_is_failure_evidence(self) -> None:
        content = "HTTP/2 404\ncontent-type: text/html\nserver: envoy"

        self.assertTrue(looks_like_url_fetch_failure(content))
        self.assertFalse(looks_like_url_content_evidence(content))

    def test_fetch_url_404_returns_structured_http_error(self) -> None:
        error = HTTPError(
            "https://example.com/missing",
            404,
            "Not Found",
            None,
            io.BytesIO(b"<html><title>404</title><body>Not Found</body></html>"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("fetch_url",),
            )
            with patch("orbit.runtime.web.urlopen", side_effect=error):
                execution = executor.execute("fetch_url", {"url": "https://example.com/missing"}, chunk_budget={})

        self.assertIn("status: http_error", execution.result.content)
        self.assertIn("http_status: 404", execution.result.content)

    def test_fetch_url_timeout_returns_structured_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("fetch_url",),
            )
            with patch("orbit.runtime.web.urlopen", side_effect=TimeoutError()):
                execution = executor.execute("fetch_url", {"url": "https://example.com/slow"}, chunk_budget={})

        self.assertIn("status: timeout", execution.result.content)

    def test_fetch_url_dns_failure_returns_network_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("fetch_url",),
            )
            with patch("orbit.runtime.web.urlopen", side_effect=URLError("Name or service not known")):
                execution = executor.execute("fetch_url", {"url": "https://example.invalid"}, chunk_budget={})

        self.assertIn("status: network_error", execution.result.content)

    def test_fetch_url_binary_returns_unsupported_content(self) -> None:
        class FakeHeaders:
            def get_content_type(self) -> str:
                return "application/octet-stream"

            def get_content_charset(self):
                return None

        class FakeResponse:
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def geturl(self) -> str:
                return "https://example.com/file.bin"

            def getcode(self) -> int:
                return 200

            def read(self, amount: int = -1) -> bytes:
                del amount
                return b"\x00\x01\x02\x03"

        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("fetch_url",),
            )
            with patch("orbit.runtime.web.urlopen", return_value=FakeResponse()):
                execution = executor.execute("fetch_url", {"url": "https://example.com/file.bin"}, chunk_budget={})

        self.assertIn("status: unsupported_content", execution.result.content)

    def test_fetch_url_empty_body_returns_empty_body_status(self) -> None:
        class FakeHeaders:
            def get_content_type(self) -> str:
                return "text/html"

            def get_content_charset(self) -> str:
                return "utf-8"

        class FakeResponse:
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def geturl(self) -> str:
                return "https://example.com/empty"

            def getcode(self) -> int:
                return 200

            def read(self, amount: int = -1) -> bytes:
                del amount
                return b""

        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("fetch_url",),
            )
            with patch("orbit.runtime.web.urlopen", return_value=FakeResponse()):
                execution = executor.execute("fetch_url", {"url": "https://example.com/empty"}, chunk_budget={})

        self.assertIn("status: empty_body", execution.result.content)

    def test_fetch_url_truncates_large_text_payloads(self) -> None:
        class FakeHeaders:
            def get_content_type(self) -> str:
                return "text/plain"

            def get_content_charset(self) -> str:
                return "utf-8"

        class FakeResponse:
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def geturl(self) -> str:
                return "https://example.com/large"

            def getcode(self) -> int:
                return 200

            def read(self, amount: int = -1) -> bytes:
                del amount
                return ("paragraph " * 3000).encode("utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("fetch_url",),
            )
            with patch("orbit.runtime.web.urlopen", return_value=FakeResponse()):
                execution = executor.execute("fetch_url", {"url": "https://example.com/large"}, chunk_budget={})

        self.assertIn("status: ok", execution.result.content)
        self.assertIn("text_truncated: true", execution.result.content)
        self.assertLess(len(execution.result.content), 14_500)

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

    def test_exec_shell_full_allows_absolute_paths_because_shell_is_unrestricted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp).parent / "orbit-shell-outside.txt"
            outside.write_text("outside", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": f"cat {outside}"},
                chunk_budget={},
            )
            outside.unlink()

        self.assertEqual(execution.source, "orbit")
        self.assertEqual(execution.result.content, "outside")

    def test_exec_shell_full_allows_parent_traversal_because_shell_is_unrestricted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            outside = Path(tmp) / "secret.txt"
            outside.write_text("secret", encoding="utf-8")
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "cat ../secret.txt"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertEqual(execution.result.content, "secret")

    def test_exec_shell_full_allows_cd_because_shell_is_unrestricted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {"command": "cd .. && pwd"},
                chunk_budget={},
            )

        self.assertEqual(execution.source, "orbit")
        self.assertEqual(execution.result.content, str(Path(tmp)))

    def test_exec_shell_full_timeout_kills_background_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tempfile.gettempdir()) / f"orbit-child-finished-{Path(tmp).name}"
            marker.unlink(missing_ok=True)
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=Path(tmp),
                allowed_tool_names=("exec_shell_full_command",),
            )

            execution = executor.execute(
                "exec_shell_full_command",
                {
                    "command": (
                        "python3 -c 'import subprocess,time; "
                        f"subprocess.Popen([\"sh\",\"-c\",\"sleep 2; touch {marker}\"]); "
                        "time.sleep(20)'"
                    ),
                    "timeout": 1,
                },
                chunk_budget={},
            )
            time.sleep(3)
            child_survived = marker.exists()
            marker.unlink(missing_ok=True)

        self.assertIn("timed out after 1s", execution.result.content)
        self.assertFalse(child_survived)

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

    def test_exec_shell_full_pdf_head_and_tail_return_different_slices(self) -> None:
        sample_text = "\n".join(f"Line {index:03d}" for index in range(1, 41))
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "sample.pdf"
            target.write_bytes(b"%PDF-1.4\n")
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )
            with patch("orbit.runtime.shell_guardrails.extract_pdf_text", return_value=(sample_text, "pdftotext")):
                head_execution = executor.execute(
                    "exec_shell_full_command",
                    {"command": "pdftotext sample.pdf - | head -n 3"},
                    chunk_budget={},
                )
                tail_execution = executor.execute(
                    "exec_shell_full_command",
                    {"command": "pdftotext sample.pdf - | tail -n 3"},
                    chunk_budget={},
                )

        self.assertIn("Line 001", head_execution.result.content)
        self.assertIn("Line 003", head_execution.result.content)
        self.assertNotIn("Line 040", head_execution.result.content)
        self.assertIn("Line 038", tail_execution.result.content)
        self.assertIn("Line 040", tail_execution.result.content)
        self.assertNotEqual(head_execution.result.content, tail_execution.result.content)

    def test_exec_shell_full_pdf_sed_and_grep_filters_text(self) -> None:
        sample_text = "\n".join(
            [
                "alpha",
                "",
                "beta",
                "gamma",
                "Security requirement",
                "VPN requirement",
                "delta",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "sample.pdf"
            target.write_bytes(b"%PDF-1.4\n")
            executor = HybridToolExecutor(
                backend=FakeServerTools(),
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
            )
            with patch("orbit.runtime.shell_guardrails.extract_pdf_text", return_value=(sample_text, "pdftotext")):
                sed_execution = executor.execute(
                    "exec_shell_full_command",
                    {"command": "pdftotext sample.pdf - | sed -n '3,5p'"},
                    chunk_budget={},
                )
                grep_execution = executor.execute(
                    "exec_shell_full_command",
                    {"command": "pdftotext sample.pdf - | grep -iE 'Security|VPN'"},
                    chunk_budget={},
                )

        self.assertIn("beta", sed_execution.result.content)
        self.assertIn("Security requirement", sed_execution.result.content)
        self.assertNotIn("VPN requirement", sed_execution.result.content)
        self.assertIn("Security requirement", grep_execution.result.content)
        self.assertIn("VPN requirement", grep_execution.result.content)
        self.assertNotIn("alpha", grep_execution.result.content)

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

            with patch("orbit.runtime.file_tools.shutil.which") as which, patch(
                "orbit.runtime.file_tools.subprocess.run"
            ) as run:
                which.side_effect = lambda name: None if name == "pdftotext" else "/usr/bin/strings"
                run.return_value = subprocess.CompletedProcess(
                    args=["/usr/bin/strings", "-a", "-n", "8", str(target)],
                    returncode=0,
                    stdout="%PDF-1.4\nOrbit PDF fallback visible text\n%%EOF\n",
                    stderr="",
                )
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
