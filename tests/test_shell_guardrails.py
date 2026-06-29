from __future__ import annotations

import unittest

from orbit.runtime.shell_guardrails import (
    is_mutating_shell_command,
    is_mutative_user_request,
    validate_read_only_shell_mutation,
)


class ShellGuardrailsTests(unittest.TestCase):
    def test_set_enable_disable_are_mutative_requests(self) -> None:
        self.assertTrue(is_mutative_user_request("Set service.timeout to 30 in config.json."))
        self.assertTrue(is_mutative_user_request("Enable the service in settings.ini."))
        self.assertTrue(is_mutative_user_request("Disable debug mode in service.yaml."))

    def test_suggest_fixes_remains_read_only_when_negated(self) -> None:
        self.assertFalse(is_mutative_user_request("Suggest fixes for service.py but do not modify files."))

    def test_read_only_request_rejects_mutating_shell_command(self) -> None:
        error = validate_read_only_shell_mutation(
            {"command": "sed -i 's/old/new/' note.txt"},
            user_prompt="read note.txt and explain it",
        )

        self.assertIsNotNone(error)
        self.assertIn("read-only request rejected", error or "")

    def test_read_only_path_with_edit_in_filename_is_not_mutative_intent(self) -> None:
        error = validate_read_only_shell_mutation(
            {"command": "sed -i 's/beta/delta/' workdir/edit-target.txt"},
            user_prompt="read workdir/edit-target.txt",
        )

        self.assertIsNotNone(error)

    def test_read_only_request_rejects_python_file_write(self) -> None:
        error = validate_read_only_shell_mutation(
            {"command": "python3 -c 'from pathlib import Path; Path(\"note.txt\").write_text(\"new\")'"},
            user_prompt="show note.txt",
        )

        self.assertIsNotNone(error)

    def test_explicit_edit_request_allows_mutating_shell_command(self) -> None:
        error = validate_read_only_shell_mutation(
            {"command": "sed -i 's/old/new/' note.txt"},
            user_prompt="change old to new in note.txt",
        )

        self.assertIsNone(error)

    def test_quoted_angle_brackets_are_not_shell_writes(self) -> None:
        self.assertFalse(is_mutating_shell_command("printf '<html><body>Hello</body></html>'"))


if __name__ == "__main__":
    unittest.main()
