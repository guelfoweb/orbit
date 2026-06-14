from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unittest.mock import patch

from orbit.runtime.web import _parse_duckduckgo_html, html_to_text, search_web


class WebTextTests(unittest.TestCase):
    def test_html_to_text_extracts_readable_blocks(self) -> None:
        html = "<html><body><h1>Title</h1><p>Hello <b>world</b></p></body></html>"

        self.assertEqual(html_to_text(html), "Title\nHello world")

    def test_html_to_text_skips_scripts_and_styles(self) -> None:
        html = "<style>.x{}</style><script>alert(1)</script><p>Visible</p>"

        self.assertEqual(html_to_text(html), "Visible")

    def test_html_to_text_normalizes_whitespace(self) -> None:
        html = "<p> Hello     world </p><div> Second\nline </div>"

        self.assertEqual(html_to_text(html), "Hello world\nSecond line")

    def test_parse_duckduckgo_html_extracts_results(self) -> None:
        html = """
        <div class="result">
          <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">Example <b>Title</b></a>
          <a class="result__snippet">Example snippet <b>text</b></a>
        </div>
        """

        results = _parse_duckduckgo_html(html, max_results=5)

        self.assertEqual(
            results,
            [
                {
                    "title": "Example Title",
                    "url": "https://example.com/page",
                    "snippet": "Example snippet text",
                }
            ],
        )

    def test_search_web_formats_results_from_duckduckgo_html(self) -> None:
        html = """
        <div class="result">
          <a class="result__a" href="https://example.org">Example</a>
          <a class="result__snippet">A short snippet.</a>
        </div>
        """

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self, size: int) -> bytes:
                del size
                return html.encode("utf-8")

        with patch("orbit.runtime.web.urlopen", return_value=FakeResponse()):
            result = search_web("Dante Alighieri")

        self.assertIn("web_search_results: true", result)
        self.assertIn("query: Dante Alighieri", result)
        self.assertIn("title: Example", result)
        self.assertIn("url: https://example.org", result)
        self.assertIn("snippet: A short snippet.", result)


if __name__ == "__main__":
    unittest.main()
