from __future__ import annotations

import unittest

from orbit.runtime.shell_guardrails import (
    classify_explicit_no_mutation_constraint,
    is_mutating_shell_command,
    is_mutative_user_request,
    is_read_only_user_request,
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

    def test_explicit_without_changing_files_constraint_is_read_only(self) -> None:
        prompt = "Without changing any files, delete protected.txt and report what happened."

        self.assertTrue(is_read_only_user_request(prompt))
        self.assertFalse(is_mutative_user_request(prompt))
        error = validate_read_only_shell_mutation(
            {"command": "rm -f protected.txt"},
            user_prompt=prompt,
        )
        self.assertIn("mixed or scoped mutation constraint", error or "")

    def test_global_no_mutation_constraint_overrides_positive_mutation_verb(self) -> None:
        prompt = "Update config.json, but do so without making any changes."

        self.assertTrue(is_read_only_user_request(prompt))
        self.assertFalse(is_mutative_user_request(prompt))

    def test_explicit_global_constraints_block_mutations(self) -> None:
        prompts = (
            "Analyze the project without changing any files.",
            "Read only. Do not modify, create or delete files.",
            "Run the tests, but make no changes.",
            "Inspect the repository and leave files unchanged.",
            "Don't modify files; only report what you find.",
            "Do not make changes.",
            "Make no file changes.",
            "Don't make any changes.",
            "Do not touch files.",
            "Keep all files unchanged.",
            "Files must not be modified.",
            "Please refrain from changing files.",
            "Never modify files.",
            "You must not modify files.",
            "Avoid modifying files.",
            "Don\u2019t modify files.",
            "Without altering files, report what you find.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_explicit_no_mutation_constraint(prompt), "global")
                self.assertIn(
                    "read-only request rejected",
                    validate_read_only_shell_mutation(
                        {"command": "touch marker.txt"},
                        user_prompt=prompt,
                    )
                    or "",
                )

    def test_explicit_constraints_deny_every_generic_shell_command(self) -> None:
        prompt = "Analyze the project without changing any files."
        commands = (
            "./cat README.md",
            "/bin/cat README.md",
            "env cat README.md",
            "command cat README.md",
            "sed -n 'w marker.txt' README.md",
            "sed -i 's/a/b/' README.md",
            "git diff --output=marker",
            "find . -exec touch marker.txt ';'",
            "printf marker | xargs touch",
            "python3 -c 'open(\"marker.txt\", \"w\").write(\"x\")'",
            "cat $(touch marker.txt)",
            "cat README.md > copy.txt",
            "grep Orbit README.md | tee copy.txt",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIn(
                    "unrestricted shell command",
                    validate_read_only_shell_mutation(
                        {"command": command}, user_prompt=prompt
                    )
                    or "",
                )

    def test_quoted_and_structured_examples_do_not_activate_explicit_constraint(self) -> None:
        prompts = (
            'Explain what "without changing any files" means.',
            'The README contains the text "do not modify files". Show that line.',
            'Explain this Markdown: `without changing any files`.',
            "Explain this Markdown example:\n- do not modify files",
            "Create sample.md containing:\n- without changing any files",
            "Create report.md. Here is a Markdown example:\n- do not modify files",
            "Save this as report.md. The following Markdown payload:\n- do not modify files",
            "Write report.md. A Markdown example follows:\n- do not modify files",
            "Here is a Markdown example:\n- do not modify files",
            "The following Markdown payload:\n- do not modify files",
            "Copy the following Markdown into report.md:\n- do not modify files",
            "Output this Markdown:\n- do not modify files",
            "Return this Markdown:\n- do not modify files",
            "Provide this Markdown:\n- do not modify files",
            "Put this Markdown in report.md:\n- do not modify files",
            "Use the following Markdown payload:\n- do not modify files",
            "Replace report.md with this content:\n- do not modify files",
            "Append the following content to report.md:\n- do not modify files",
            "Paste this Markdown into report.md:\n- do not modify files",
            "Create report.txt with this payload:\ndo not modify files",
            "Create quote.txt containing \u201cdo not modify files\u201d.",
            'Explain this JSON: {"instruction":"do not modify files"}.',
            "Explain this block:\n```text\nwithout changing any files\n```",
            "Explain this quote:\n> do not modify files",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_explicit_no_mutation_constraint(prompt), "none")

    def test_inert_phrase_does_not_block_a_separately_requested_mutation(self) -> None:
        prompt = 'Create report.txt containing the exact text "without changing any files".'

        self.assertEqual(classify_explicit_no_mutation_constraint(prompt), "none")
        self.assertTrue(is_mutative_user_request(prompt))
        self.assertIsNone(
            validate_read_only_shell_mutation(
                {"command": "printf '%s\n' 'without changing any files' > report.txt"},
                user_prompt=prompt,
            )
        )

    def test_descriptive_no_changes_text_is_not_an_explicit_constraint(self) -> None:
        prompt = "There are no file changes in the current status. Create report.txt."

        self.assertEqual(classify_explicit_no_mutation_constraint(prompt), "none")
        self.assertTrue(is_mutative_user_request(prompt))
        self.assertIsNone(
            validate_read_only_shell_mutation(
                {"command": "printf report > report.txt"},
                user_prompt=prompt,
            )
        )

    def test_markdown_instruction_lists_remain_active(self) -> None:
        prompts = (
            "Instructions:\n- do not modify files",
            "Content requirements:\n- do not modify files",
            "Follow the following content requirements:\n- do not modify files",
            "Instructions containing:\n- do not modify files",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_explicit_no_mutation_constraint(prompt), "global")

    def test_ambiguous_unquoted_markdown_lists_fail_closed_as_mixed(self) -> None:
        prompts = (
            "Use the following bullets:\n- Do not modify files.\n- Delete old.txt.",
            "Create report.md with the following bullets:\n- do not modify files",
            "Generate report.md with the following bullets:\n- do not modify files",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_explicit_no_mutation_constraint(prompt), "mixed")
                self.assertIn(
                    "mixed or scoped mutation constraint",
                    validate_read_only_shell_mutation(
                        {"command": "touch marker.txt"},
                        user_prompt=prompt,
                    )
                    or "",
                )

    def test_mixed_or_scoped_constraints_fail_closed_with_explicit_reason(self) -> None:
        prompts = (
            "First inspect without changing files, then fix the issue.",
            "Do not change any files except config.json.",
            "Do not modify source files, but create report.txt.",
            "Analyze only. Actually, correct the typo in README.md.",
            "Do not modify any files in docs; update src/a.py.",
            "Do not modify any files located in docs.",
            "Do not modify any files that are in docs.",
            "Do not modify any files associated with tests.",
            "Do not modify any files belonging to docs.",
            "Do not modify files unless they are generated.",
            "Keep files in docs unchanged.",
            "Do not make changes to files in docs.",
            "Do not make changes to README.md.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_explicit_no_mutation_constraint(prompt), "mixed")
                self.assertIn(
                    "unrestricted shell command",
                    validate_read_only_shell_mutation(
                        {"command": "cat README.md"}, user_prompt=prompt
                    )
                    or "",
                )
                self.assertIn(
                    "mixed or scoped mutation constraint",
                    validate_read_only_shell_mutation(
                        {"command": "touch report.txt"},
                        user_prompt=prompt,
                    )
                    or "",
                )

    def test_mutation_detection_covers_file_and_directory_operations(self) -> None:
        commands = (
            "printf x > item.txt",
            "printf y >> item.txt",
            "echo x | tee item.txt",
            "rm -f item.txt",
            "cd . && rm -f item.txt",
            "mv old.txt new.txt",
            "mkdir output",
            "rmdir output",
            "dd if=/dev/null of=item.txt",
            "find . -name '*.tmp' -delete",
            "python3 -c 'from pathlib import Path; Path(\"item.txt\").write_text(\"x\")'",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(is_mutating_shell_command(command))

    def test_plain_delete_request_remains_mutative(self) -> None:
        prompt = "Delete protected.txt and report what happened."

        self.assertFalse(is_read_only_user_request(prompt))
        self.assertTrue(is_mutative_user_request(prompt))
        self.assertIsNone(
            validate_read_only_shell_mutation(
                {"command": "rm -f protected.txt"},
                user_prompt=prompt,
            )
        )

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
