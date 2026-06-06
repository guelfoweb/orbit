from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.web import _extract_site_filter, parse_search_results, search_web
from orbit.runtime.web import fetch_url


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"""
                <html>
                  <head>
                    <title>Test page</title>
                    <meta name="description" content="Useful test description.">
                    <script>secret()</script>
                    <style>body { color: red; }</style>
                  </head>
                  <body>
                    <nav>Navigation noise</nav>
                    <h1>Main heading</h1>
                    <p>First useful paragraph.</p>
                    <ul><li>First item</li><li>Second item</li></ul>
                    <footer>Footer noise</footer>
                  </body>
                </html>
                """
            )
            return
        if self.path == "/plain":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"plain text body")
            return
        if self.path == "/long":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            paragraphs = "".join(f"<p>Paragraph {index:03d} alpha beta gamma.</p>" for index in range(120))
            self.wfile.write(f"<html><body>{paragraphs}</body></html>".encode("utf-8"))
            return
        if self.path == "/binary":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(b"\x89PNG\r\n")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class FetchUrlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.thread.join(timeout=2)

    def test_fetch_url_rejects_non_http_urls(self) -> None:
        result = fetch_url("file:///etc/passwd")

        self.assertIn("http/https", result)

    def test_fetch_url_extracts_readable_html(self) -> None:
        result = fetch_url(f"{self.base_url}/html")

        self.assertIn("status: 200", result)
        self.assertIn("content_type: text/html", result)
        self.assertIn("chunk_index: 0", result)
        self.assertIn("total_chunks: 1", result)
        self.assertIn("chars:", result)
        self.assertIn("Test page", result)
        self.assertIn("Useful test description.", result)
        self.assertIn("Main heading", result)
        self.assertIn("- First item", result)
        self.assertNotIn("Navigation noise", result)
        self.assertNotIn("secret()", result)

    def test_fetch_url_reads_plain_text(self) -> None:
        result = fetch_url(f"{self.base_url}/plain")

        self.assertIn("content_type: text/plain", result)
        self.assertIn("plain text body", result)

    def test_fetch_url_supports_explicit_chunks(self) -> None:
        result = fetch_url(f"{self.base_url}/long", chunk_index=1, chunk_chars=500)

        self.assertIn("chunk_index: 1", result)
        self.assertIn("total_chunks:", result)
        self.assertIn("chars: 500-", result)
        self.assertIn("Paragraph", result)

    def test_fetch_url_rejects_out_of_range_chunk(self) -> None:
        result = fetch_url(f"{self.base_url}/plain", chunk_index=99, chunk_chars=500)

        self.assertIn("chunk_index out of range", result)

    def test_fetch_url_rejects_oversized_chunk(self) -> None:
        result = fetch_url(f"{self.base_url}/plain", chunk_index=0, chunk_chars=999999)

        self.assertIn("chunk_chars too large", result)

    def test_fetch_url_rejects_unsupported_content_type(self) -> None:
        result = fetch_url(f"{self.base_url}/binary")

        self.assertIn("unsupported content type", result)

    def test_search_web_rejects_invalid_max_results(self) -> None:
        result = search_web("orbit", max_results=99)

        self.assertIn("max_results", result)

    def test_search_web_rejects_invalid_site_and_timelimit_before_network(self) -> None:
        bad_site = search_web("orbit", site="https://example.com/path")
        bad_timelimit = search_web("orbit", timelimit="hour")

        self.assertIn("bare domain", bad_site)
        self.assertIn("timelimit", bad_timelimit)

    def test_search_web_extracts_inline_site_filter(self) -> None:
        query, site = _extract_site_filter("Dante Alighieri site:wikipedia.org")

        self.assertEqual(query, "Dante Alighieri")
        self.assertEqual(site, "wikipedia.org")

    def test_parse_search_results_extracts_structured_results(self) -> None:
        html = """
        <html><body>
          <a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=example.com">Sponsored result</a>
          <a class="result__snippet">Ad snippet</a>
          <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fa">Example A</a>
          <a class="result__snippet">First snippet</a>
          <a class="result__a" href="https://example.com/b">Example B</a>
          <div class="result__snippet">Second snippet</div>
        </body></html>
        """

        results = parse_search_results(html, max_results=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Example A")
        self.assertEqual(results[0]["url"], "https://example.com/a")
        self.assertEqual(results[0]["snippet"], "First snippet")

if __name__ == "__main__":
    unittest.main()
