from __future__ import annotations

import unittest

from orbit.runtime.shell_guardrails import is_mutative_user_request


class ShellGuardrailsTests(unittest.TestCase):
    def test_set_enable_disable_are_mutative_requests(self) -> None:
        self.assertTrue(is_mutative_user_request("Set service.timeout to 30 in config.json."))
        self.assertTrue(is_mutative_user_request("Enable the service in settings.ini."))
        self.assertTrue(is_mutative_user_request("Disable debug mode in service.yaml."))

    def test_suggest_fixes_remains_read_only_when_negated(self) -> None:
        self.assertFalse(is_mutative_user_request("Suggest fixes for service.py but do not modify files."))


if __name__ == "__main__":
    unittest.main()
