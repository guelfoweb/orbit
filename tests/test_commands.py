from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal.commands import help_text, set_max_tokens
from orbit.terminal.config import AppConfig


class CommandTests(unittest.TestCase):
    def test_help_mentions_max_tokens(self) -> None:
        self.assertIn("/max-tokens <n>", help_text())

    def test_set_max_tokens_without_value_reports_current_value(self) -> None:
        config = AppConfig(max_tokens=512)

        updated, message = set_max_tokens(config, "")

        self.assertEqual(updated.max_tokens, 512)
        self.assertEqual(message, "max_tokens: 512")

    def test_set_max_tokens_updates_runtime_config(self) -> None:
        config = AppConfig(max_tokens=512)

        updated, message = set_max_tokens(config, "2048")

        self.assertEqual(config.max_tokens, 512)
        self.assertEqual(updated.max_tokens, 2048)
        self.assertEqual(message, "max_tokens: 2048")

    def test_set_max_tokens_rejects_invalid_values(self) -> None:
        config = AppConfig(max_tokens=512)

        updated, message = set_max_tokens(config, "99999")

        self.assertEqual(updated.max_tokens, 512)
        self.assertIn("between 32 and 4096", message)


if __name__ == "__main__":
    unittest.main()
