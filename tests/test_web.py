from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.web import html_to_text


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


if __name__ == "__main__":
    unittest.main()
