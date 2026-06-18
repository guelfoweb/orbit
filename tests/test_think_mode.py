from __future__ import annotations

import unittest

from orbit.terminal.think_mode import DEFAULT_THINKING, normalize_think_spec, think_text


class ThinkModeTests(unittest.TestCase):
    def test_default_is_off(self) -> None:
        self.assertFalse(DEFAULT_THINKING)

    def test_normalize_think_spec_accepts_on_and_off(self) -> None:
        self.assertTrue(normalize_think_spec("on"))
        self.assertFalse(normalize_think_spec("off"))
        self.assertTrue(normalize_think_spec(True))
        self.assertFalse(normalize_think_spec(False))

    def test_think_text_reports_mode_and_usage(self) -> None:
        self.assertIn("/think off", think_text())
        self.assertIn("think: on", think_text(True))
        self.assertIn("think: off", think_text(False))

    def test_normalize_think_spec_rejects_invalid_value(self) -> None:
        with self.assertRaises(ValueError):
            normalize_think_spec("maybe")


if __name__ == "__main__":
    unittest.main()
