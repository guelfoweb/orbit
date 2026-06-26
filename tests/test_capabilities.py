from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.capabilities import (
    discover_local_capabilities,
)


class CapabilityTests(unittest.TestCase):
    def test_discovery_normalizes_available_and_missing_tools(self) -> None:
        paths = {
            "pdftotext": "/usr/bin/pdftotext",
            "python3": "/usr/bin/python3",
            "unzip": "/usr/bin/unzip",
        }

        capabilities = discover_local_capabilities(paths.get)

        self.assertTrue(capabilities.by_name("pdftotext").available)
        self.assertEqual(capabilities.by_name("pdftotext").path, "/usr/bin/pdftotext")
        self.assertTrue(capabilities.by_name("python3").available)
        self.assertFalse(capabilities.by_name("pandoc").available)
        self.assertIsNone(capabilities.by_name("pandoc").path)

    def test_prompt_summary_reports_available_and_unavailable_tools(self) -> None:
        capabilities = discover_local_capabilities({"pdftotext": "/usr/bin/pdftotext"}.get)

        summary = capabilities.format_prompt_summary()

        self.assertIn("Local tools available: pdftotext", summary)
        self.assertIn("Unavailable:", summary)
        self.assertIn("pandoc", summary)

    def test_prompt_summary_warns_against_assuming_external_tools(self) -> None:
        capabilities = discover_local_capabilities({"python3": "/usr/bin/python3"}.get)

        prompt = capabilities.format_prompt_summary()

        self.assertIn("Do not assume pdftotext", prompt)
        self.assertIn("Local tools available: python3", prompt)
        self.assertIn("verify availability before use", prompt)


if __name__ == "__main__":
    unittest.main()
