from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.tooling.registry import ToolRegistry
from orbit.tooling.web import WebTools


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "sub").mkdir()
        (self.root / "sub" / "note.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
        self.registry = ToolRegistry(workdir=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_read_file_is_bounded(self) -> None:
        result = self.registry.call("read_file", {"path": "sub/note.txt", "start_line": 2, "max_lines": 2})
        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], "b\nc")
        self.assertEqual(result["next_start_line"], 4)

    def test_read_file_reports_binary_input_cleanly(self) -> None:
        (self.root / "sub" / "blob.bin").write_bytes(b"\x00\xff\x10\x80")
        result = self.registry.call("read_file", {"path": "sub/blob.bin"})
        self.assertFalse(result["ok"])
        self.assertIn("binary", result["error"])

    def test_list_files_recursive(self) -> None:
        result = self.registry.call("list_files", {"path": ".", "recursive": True})
        self.assertTrue(result["ok"])
        paths = [entry["path"] for entry in result["entries"]]
        self.assertIn("sub", paths)
        self.assertIn("sub/note.txt", paths)
        self.assertIn("summary", result)
        self.assertIn("dirs:", result["summary"])
        self.assertIn("files:", result["summary"])

    def test_list_files_skips_noise_directories(self) -> None:
        (self.root / ".venv").mkdir()
        (self.root / ".venv" / "ignore.txt").write_text("x", encoding="utf-8")
        (self.root / "__pycache__").mkdir()
        (self.root / "__pycache__" / "mod.pyc").write_bytes(b"123")
        result = self.registry.call("list_files", {"path": ".", "recursive": True})
        self.assertTrue(result["ok"])
        paths = [entry["path"] for entry in result["entries"]]
        self.assertNotIn(".venv", paths)
        self.assertNotIn(".venv/ignore.txt", paths)
        self.assertNotIn("__pycache__", paths)
        self.assertNotIn("__pycache__/mod.pyc", paths)

    def test_list_files_shallow_summary_prefers_top_level_files(self) -> None:
        (self.root / "alpha.txt").write_text("x", encoding="utf-8")
        result = self.registry.call("list_files", {"path": ".", "recursive": False})
        self.assertTrue(result["ok"])
        self.assertIn("alpha.txt", result["summary"])

    def test_list_files_orders_implementation_before_metadata_and_archives(self) -> None:
        (self.root / "src").mkdir()
        (self.root / "src" / "agent.py").write_text("print('x')", encoding="utf-8")
        (self.root / "README.md").write_text("# docs", encoding="utf-8")
        (self.root / "pyproject.toml").write_text("[project]", encoding="utf-8")
        (self.root / "orbit-portable.zip").write_bytes(b"PK\x03\x04")
        result = self.registry.call("list_files", {"path": ".", "recursive": True})
        self.assertTrue(result["ok"])
        paths = [entry["path"] for entry in result["entries"]]
        self.assertLess(paths.index("src"), paths.index("README.md"))
        self.assertLess(paths.index("src/agent.py"), paths.index("README.md"))
        self.assertLess(paths.index("src/agent.py"), paths.index("pyproject.toml"))
        self.assertLess(paths.index("src/agent.py"), paths.index("orbit-portable.zip"))

    def test_stat_path_returns_file_metadata(self) -> None:
        result = self.registry.call("stat_path", {"path": "sub/note.txt"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], "sub/note.txt")
        self.assertEqual(result["type"], "file")
        self.assertEqual(result["size_bytes"], len("a\nb\nc\nd\n".encode("utf-8")))
        self.assertIn("modified_at", result)
        self.assertIn("mode", result)

    def test_stat_path_returns_bounded_directory_metadata_newest_first(self) -> None:
        older = self.root / "sub" / "older.txt"
        newer = self.root / "sub" / "newer.txt"
        older.write_text("old", encoding="utf-8")
        newer.write_text("new", encoding="utf-8")
        older_time = 1_700_000_000
        newer_time = older_time + 60
        os.utime(self.root / "sub" / "note.txt", (older_time - 60, older_time - 60))
        os.utime(older, (older_time, older_time))
        os.utime(newer, (newer_time, newer_time))

        result = self.registry.call("stat_path", {"path": "sub", "recursive": False, "max_entries": 2})
        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], "sub")
        self.assertEqual(result["type"], "dir")
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["total_entries"], 3)
        self.assertEqual(result["file_count"], 3)
        self.assertEqual(result["dir_count"], 0)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["entries"][0]["path"], "sub/newer.txt")

    def test_stat_path_rejects_escape(self) -> None:
        result = self.registry.call("stat_path", {"path": "../outside.txt"})
        self.assertFalse(result["ok"])
        self.assertIn("escapes workdir", result["error"])

    def test_tool_definitions_guide_codebase_analysis(self) -> None:
        definitions = {item["function"]["name"]: item["function"]["description"] for item in self.registry.definitions()}
        self.assertIn("reuse exactly", definitions["list_files"])
        self.assertIn("exact relative path", definitions["read_file"])
        self.assertIn("filesystem metadata", definitions["stat_path"])
        self.assertIn("generic online research", definitions["search_web"])
        self.assertIn("general web search tool", definitions["fetch_url"])

    def test_registry_filters_tools_by_category(self) -> None:
        web_definitions = self.registry.definitions_for_categories(("web",))
        web_names = [item["function"]["name"] for item in web_definitions]
        self.assertEqual(web_names, ["search_web", "fetch_url"])
        filesystem_definitions = self.registry.definitions_for_categories(("filesystem",))
        filesystem_names = [item["function"]["name"] for item in filesystem_definitions]
        self.assertIn("stat_path", filesystem_names)

    def test_make_directory_creates_nested_directory(self) -> None:
        result = self.registry.call("make_directory", {"path": "build/output"})
        self.assertTrue(result["ok"])
        self.assertTrue((self.root / "build" / "output").is_dir())
        self.assertEqual(result["path"], "build/output")

    def test_delete_path_removes_file(self) -> None:
        target = self.root / "sub" / "delete.txt"
        target.write_text("x", encoding="utf-8")
        result = self.registry.call("delete_path", {"path": "sub/delete.txt"})
        self.assertTrue(result["ok"])
        self.assertFalse(target.exists())
        self.assertEqual(result["type"], "file")

    def test_delete_path_removes_directory_recursively(self) -> None:
        target = self.root / "sub" / "nested"
        target.mkdir()
        (target / "keep.txt").write_text("x", encoding="utf-8")
        result = self.registry.call("delete_path", {"path": "sub/nested", "recursive": True})
        self.assertTrue(result["ok"])
        self.assertFalse(target.exists())
        self.assertEqual(result["type"], "dir")

    def test_delete_path_rejects_non_empty_directory_without_recursive(self) -> None:
        target = self.root / "sub" / "nested"
        target.mkdir()
        (target / "keep.txt").write_text("x", encoding="utf-8")
        result = self.registry.call("delete_path", {"path": "sub/nested"})
        self.assertFalse(result["ok"])
        self.assertIn("recursive=true", result["error"])

    def test_write_file_rejects_escape(self) -> None:
        result = self.registry.call("write_file", {"path": "../evil.txt", "content": "x"})
        self.assertFalse(result["ok"])
        self.assertIn("escapes workdir", result["error"])

    def test_replace_in_file_updates_existing_content(self) -> None:
        target = self.root / "sub" / "edit.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        result = self.registry.call(
            "replace_in_file",
            {"path": "sub/edit.txt", "old": "beta", "new": "gamma"},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], "sub/edit.txt")
        self.assertEqual(result["replaced"], 1)
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\ngamma\n")

    def test_replace_in_file_rejects_missing_target_text(self) -> None:
        target = self.root / "sub" / "edit.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        result = self.registry.call(
            "replace_in_file",
            {"path": "sub/edit.txt", "old": "delta", "new": "gamma"},
        )
        self.assertFalse(result["ok"])
        self.assertIn("target text not found", result["error"])

    def test_replace_in_file_rejects_ambiguous_target_without_replace_all(self) -> None:
        target = self.root / "sub" / "edit.txt"
        target.write_text("alpha\nbeta\nbeta\n", encoding="utf-8")
        result = self.registry.call(
            "replace_in_file",
            {"path": "sub/edit.txt", "old": "beta", "new": "gamma"},
        )
        self.assertFalse(result["ok"])
        self.assertIn("replace_all=true", result["error"])

    def test_replace_in_file_can_replace_all(self) -> None:
        target = self.root / "sub" / "edit.txt"
        target.write_text("alpha\nbeta\nbeta\n", encoding="utf-8")
        result = self.registry.call(
            "replace_in_file",
            {"path": "sub/edit.txt", "old": "beta", "new": "gamma", "replace_all": True},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["replaced"], 2)
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\ngamma\ngamma\n")

    def test_bash_blocks_shell_operators(self) -> None:
        result = self.registry.call("bash", {"command": "ls > out.txt"})
        self.assertFalse(result["ok"])
        self.assertIn("redirection", result["error"])

    def test_bash_runs_plain_command(self) -> None:
        result = self.registry.call("bash", {"command": "pwd"})
        self.assertTrue(result["ok"])
        self.assertIn(str(self.root), result["stdout"])

    def test_bash_normalizes_df_to_workspace_mount(self) -> None:
        def fake_run(args, **kwargs):
            self.assertEqual(args, ["df", "-h", "."])
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="Filesystem  Size Used Avail Use% Mounted on\n/dev/root 1G 1G 0 100% .\n", stderr="")

        with patch("orbit.tooling.shell.subprocess.run", side_effect=fake_run):
            result = self.registry.call("bash", {"command": "df -h"})
        self.assertTrue(result["ok"])
        self.assertIn("Mounted on", result["stdout"])

    def test_bash_runs_safe_pipeline(self) -> None:
        result = self.registry.call("bash", {"command": "printf 'a\\nb\\n' | head -1"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "a\n")

    def test_bash_allows_base64_decode_pipeline(self) -> None:
        result = self.registry.call("bash", {"command": "echo -n 'Y2lhbw==' | base64 -d"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "ciao")

    def test_bash_tolerates_sigpipe_when_pipeline_output_is_useful(self) -> None:
        result = self.registry.call("bash", {"command": "yes | head -1"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "y\n")

    def test_bash_allows_semicolon_inside_quoted_python_argument(self) -> None:
        result = self.registry.call("bash", {"command": "python3 -c \"print('a'); print('b')\""})
        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "a\nb\n")

    def test_bash_blocks_unsupported_pipe_filter(self) -> None:
        result = self.registry.call("bash", {"command": "pwd | cat"})
        self.assertFalse(result["ok"])
        self.assertIn("unsupported pipe filter", result["error"])

    def test_bash_blocks_shell_executors(self) -> None:
        result = self.registry.call("bash", {"command": "sh -c 'pwd'"})
        self.assertFalse(result["ok"])
        self.assertIn("blocked shell executor", result["error"])

    def test_bash_blocks_absolute_shell_executor(self) -> None:
        result = self.registry.call("bash", {"command": "/bin/sh -c 'pwd'"})
        self.assertFalse(result["ok"])
        self.assertIn("blocked shell executor", result["error"])

    def test_bash_blocks_env_shell_executor(self) -> None:
        result = self.registry.call("bash", {"command": "env sh -c 'pwd'"})
        self.assertFalse(result["ok"])
        self.assertIn("blocked shell executor", result["error"])

    def test_bash_allows_rm_inside_workdir(self) -> None:
        target = self.root / "sub" / "delete.txt"
        target.write_text("x", encoding="utf-8")
        result = self.registry.call("bash", {"command": "rm sub/delete.txt"})
        self.assertTrue(result["ok"])
        self.assertFalse(target.exists())

    def test_bash_blocks_rm_outside_workdir(self) -> None:
        result = self.registry.call("bash", {"command": "rm ../evil.txt"})
        self.assertFalse(result["ok"])
        self.assertIn("escapes workdir", result["error"])

    def test_fetch_url_rejects_invalid_scheme(self) -> None:
        result = self.registry.call("fetch_url", {"url": "file:///etc/passwd"})
        self.assertFalse(result["ok"])
        self.assertIn("http or https", result["error"])

    def test_fetch_url_extract_helpers(self) -> None:
        html = """
        <html>
          <head><title>Example Page</title></head>
          <body>
            <script>ignored()</script>
            <a href="/docs">Docs</a>
            <p>Hello <b>world</b></p>
          </body>
        </html>
        """
        title = WebTools._extract_title(html)
        text = WebTools._extract_text(html)
        links = WebTools._extract_links(html, "https://example.com/start", 5)
        self.assertEqual(title, "Example Page")
        self.assertIn("Hello world", text)
        self.assertEqual(links, ["https://example.com/docs"])

    def test_fetch_url_extract_text_strips_script_and_style_blocks(self) -> None:
        html = """
        <html>
          <head>
            <style>.hidden { display:none; }</style>
            <script>console.log("noise")</script>
          </head>
          <body>
            <p>Visible weather text</p>
          </body>
        </html>
        """
        text = WebTools._extract_text(html)
        self.assertIn("Visible weather text", text)
        self.assertNotIn("console.log", text)
        self.assertNotIn("display:none", text)

    def test_fetch_url_extract_highlights_finds_weather_like_snippets(self) -> None:
        text = (
            "La giornata sarà caratterizzata da cielo sereno o poco nuvoloso, "
            "temperatura minima di 14°C e massima di 29°C. "
            "Durante la giornata di oggi la temperatura massima verrà registrata alle ore 16."
        )
        highlights = WebTools._extract_highlights(text)
        self.assertTrue(highlights)
        self.assertTrue(any("14°C" in item and "29°C" in item for item in highlights))

    def test_search_web_extract_results(self) -> None:
        html = """
        <html>
          <body>
            <div class="result results_links results_links_deep web-result">
              <div class="links_main links_deep result__body">
                <h2 class="result__title">
                  <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fprofile">Dante Alighieri - Profile</a>
                </h2>
                <a class="result__url" href="https://example.com/profile">example.com/profile</a>
                <a class="result__snippet">Official profile page.</a>
              </div>
            </div>
          </body>
        </html>
        """
        results = WebTools._extract_search_results(html, max_results=5)
        self.assertEqual(
            results,
            [
                {
                    "title": "Dante Alighieri - Profile",
                    "url": "https://example.com/profile",
                    "snippet": "Official profile page.",
                    "display_url": "example.com/profile",
                }
            ],
        )

    def test_search_web_skips_duckduckgo_ad_redirects(self) -> None:
        html = """
        <html>
          <body>
            <div class="result results_links results_links_deep web-result">
              <div class="links_main links_deep result__body">
                <a class="result__a" href="https://duckduckgo.com/y.js?ad_provider=bingv7aa">Sponsored result</a>
              </div>
            </div>
            <div class="result results_links results_links_deep web-result">
              <div class="links_main links_deep result__body">
                <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Farticle">Real result</a>
                <a class="result__snippet">Useful organic result.</a>
              </div>
            </div>
          </body>
        </html>
        """
        results = WebTools._extract_search_results(html, max_results=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Real result")
        self.assertEqual(results[0]["url"], "https://example.org/article")

    def test_search_web_uses_form_encoded_post(self) -> None:
        captured = {}

        class InspectWebTools(WebTools):
            def _fetch_page(self, raw_url, *, timeout, form_data=None):
                captured["url"] = raw_url
                captured["timeout"] = timeout
                captured["form_data"] = form_data
                return raw_url, 200, "text/html", "<html></html>"

        tool = InspectWebTools()
        result = tool.search_web({"query": "Dante Alighieri", "timeout": 7})
        self.assertTrue(result["ok"])
        self.assertEqual(captured["url"], "https://html.duckduckgo.com/html/")
        self.assertEqual(captured["timeout"], 7)
        self.assertEqual(captured["form_data"], {"q": "Dante Alighieri"})

    def test_fetch_url_uses_browser_like_headers(self) -> None:
        captured = {}

        class DummyResponse:
            def __init__(self) -> None:
                self.status = 200
                self.headers = type(
                    "Headers",
                    (),
                    {
                        "get_content_type": lambda self: "text/html",
                        "get_content_charset": lambda self: "utf-8",
                    },
                )()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b"<html><title>Example</title><body>hello</body></html>"

            def geturl(self):
                return "https://example.com/"

        def fake_request(url, data=None, headers=None):
            captured["url"] = url
            captured["data"] = data
            captured["headers"] = dict(headers or {})
            return type("Request", (), {})()

        def fake_urlopen(req, timeout=None):
            captured["timeout"] = timeout
            return DummyResponse()

        with (
            patch("orbit.tooling.web.request.Request", side_effect=fake_request),
            patch("orbit.tooling.web.request.urlopen", side_effect=fake_urlopen),
        ):
            tool = WebTools()
            result = tool.fetch_url({"url": "https://example.com", "timeout": 5, "max_links": 0})

        self.assertTrue(result["ok"])
        self.assertEqual(captured["url"], "https://example.com")
        self.assertEqual(captured["timeout"], 5)
        self.assertIn("Mozilla/5.0", captured["headers"]["User-Agent"])
        self.assertIn("Chrome/124.0.0.0", captured["headers"]["User-Agent"])
        self.assertEqual(captured["headers"]["Accept-Language"], "en-US,en;q=0.9,it;q=0.8")
        self.assertEqual(captured["headers"]["Cache-Control"], "no-cache")
        self.assertEqual(captured["headers"]["Pragma"], "no-cache")

    def test_fetch_url_supports_chunk_offsets(self) -> None:
        class DummyResponse:
            def __init__(self) -> None:
                self.status = 200
                self.headers = type(
                    "Headers",
                    (),
                    {
                        "get_content_type": lambda self: "text/html",
                        "get_content_charset": lambda self: "utf-8",
                    },
                )()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                body = ("ABCDEFGHIJ1234567890" * 25).encode("utf-8")
                return (
                    b"<html><body>"
                    + body +
                    b"</body></html>"
                )

            def geturl(self):
                return "https://example.com/"

        with (
            patch("orbit.tooling.web.request.urlopen", return_value=DummyResponse()),
        ):
            tool = WebTools()
            first = tool.fetch_url({"url": "https://example.com", "timeout": 5, "max_chars": 200, "max_links": 0})
            second = tool.fetch_url(
                {"url": "https://example.com", "timeout": 5, "max_chars": 200, "max_links": 0, "start_char": 200}
            )

        self.assertTrue(first["ok"])
        self.assertTrue(first["has_more"])
        self.assertEqual(first["text"], "ABCDEFGHIJ1234567890" * 10)
        self.assertEqual(first["start_char"], 0)
        self.assertEqual(first["end_char"], 200)
        self.assertEqual(first["next_start_char"], 200)
        self.assertTrue(second["ok"])
        self.assertEqual(second["text"], "ABCDEFGHIJ1234567890" * 10)
        self.assertEqual(second["start_char"], 200)
        self.assertEqual(second["end_char"], 400)

    def test_fetch_url_literal_query_returns_bounded_matches(self) -> None:
        class DummyResponse:
            status = 200
            headers = type(
                "Headers",
                (),
                {
                    "get_content_type": lambda self: "text/html",
                    "get_content_charset": lambda self: "utf-8",
                },
            )()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b"<html><title>Doc</title><body><p>Alpha transhumanism beta.</p><p>Other text.</p></body></html>"

            def geturl(self):
                return "https://example.com/"

        with patch("orbit.tooling.web.request.urlopen", return_value=DummyResponse()):
            tool = WebTools()
            result = tool.fetch_url({"url": "https://example.com", "query": "transhumanism", "max_links": 0})

        self.assertTrue(result["ok"])
        self.assertEqual(result["query"], "transhumanism")
        self.assertEqual(result["query_mode"], "literal")
        self.assertEqual(result["match_count"], 1)
        self.assertTrue(result["has_query_matches"])
        self.assertIn("transhumanism", result["text"])
        self.assertIn("context", result["matches"][0])

    def test_fetch_url_concept_query_returns_keyword_overlap_candidates(self) -> None:
        text = (
            "<html><body>"
            "<p>Technology must preserve human dignity and freedom in the digital transition.</p>"
            "<p>Unrelated paragraph about weather.</p>"
            "</body></html>"
        )

        class DummyResponse:
            status = 200
            headers = type(
                "Headers",
                (),
                {
                    "get_content_type": lambda self: "text/html",
                    "get_content_charset": lambda self: "utf-8",
                },
            )()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return text.encode("utf-8")

            def geturl(self):
                return "https://example.com/"

        with patch("orbit.tooling.web.request.urlopen", return_value=DummyResponse()):
            tool = WebTools()
            result = tool.fetch_url(
                {
                    "url": "https://example.com",
                    "query": "human dignity and freedom",
                    "query_mode": "concept",
                    "max_links": 0,
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["query_mode"], "concept")
        self.assertEqual(result["match_count"], 1)
        self.assertIn("human dignity", result["matches"][0]["context"])

    def test_fetch_url_concept_query_handles_multiple_concepts(self) -> None:
        text = (
            "<html><body>"
            "<p>La dignità della persona deve essere custodita nella trasformazione digitale.</p>"
            "<p>La libertà richiede responsabilità e giustizia.</p>"
            "<p>L'intelligenza artificiale può aiutare l'uomo se resta trasparente.</p>"
            "</body></html>"
        )

        class DummyResponse:
            status = 200
            headers = type(
                "Headers",
                (),
                {
                    "get_content_type": lambda self: "text/html",
                    "get_content_charset": lambda self: "utf-8",
                },
            )()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return text.encode("utf-8")

            def geturl(self):
                return "https://example.com/"

        with patch("orbit.tooling.web.request.urlopen", return_value=DummyResponse()):
            tool = WebTools()
            result = tool.fetch_url(
                {
                    "url": "https://example.com",
                    "query": "dignity, freedom, and artificial intelligence",
                    "query_mode": "concept",
                    "max_links": 0,
                    "max_matches": 8,
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["query_mode"], "concept")
        self.assertGreaterEqual(result["match_count"], 3)
        contexts = "\n".join(item["context"] for item in result["matches"])
        self.assertIn("dignità", contexts)
        self.assertIn("libertà", contexts)
        self.assertIn("intelligenza artificiale", contexts)

    def test_fetch_url_concept_query_handles_synonymic_concepts(self) -> None:
        text = (
            "<html><body>"
            "<p>La dignità della persona resta il criterio fondamentale.</p>"
            "<p>La libertà della coscienza non può essere ridotta a controllo.</p>"
            "<p>I sistemi di intelligenza artificiale richiedono trasparenza.</p>"
            "</body></html>"
        )

        class DummyResponse:
            status = 200
            headers = type(
                "Headers",
                (),
                {
                    "get_content_type": lambda self: "text/html",
                    "get_content_charset": lambda self: "utf-8",
                },
            )()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return text.encode("utf-8")

            def geturl(self):
                return "https://example.com/"

        with patch("orbit.tooling.web.request.urlopen", return_value=DummyResponse()):
            tool = WebTools()
            result = tool.fetch_url(
                {
                    "url": "https://example.com",
                    "query": "human worth, free will, or AI systems",
                    "query_mode": "concept",
                    "max_links": 0,
                    "max_matches": 8,
                }
            )

        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["match_count"], 3)
        contexts = "\n".join(item["context"] for item in result["matches"])
        self.assertIn("dignità", contexts)
        self.assertIn("libertà", contexts)
        self.assertIn("intelligenza artificiale", contexts)
