from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from orbit.runtime.shell_guardrails import (
    classify_explicit_no_mutation_constraint,
    execute_exec_shell_full_command,
    is_mutating_shell_command,
    is_mutative_user_request,
    is_read_only_user_request,
    validate_read_only_shell_mutation,
)


class ShellGuardrailsTests(unittest.TestCase):
    def test_targeted_search_discards_result_when_source_changes_during_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "note.txt"
            target.write_text("alpha\nneedle\nomega\n" + "filler\n" * 2000, encoding="utf-8")

            def mutate(*args, **kwargs):
                target.write_text("alpha\nchanged\nomega\n", encoding="utf-8")
                return subprocess.CompletedProcess(args=args, returncode=0, stdout="2:needle\n", stderr="")

            with patch("orbit.runtime.shell_guardrails._run_shell_command", side_effect=mutate):
                result = execute_exec_shell_full_command(
                    {"command": "grep -n needle note.txt"},
                    workdir=workdir,
                )

        self.assertEqual(result, "error: source file changed during targeted search; result discarded")

    def test_overlapping_search_windows_are_deduplicated_into_one_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text(
                "a\nneedle one\nneedle two\nz\n" + "filler\n" * 2000,
                encoding="utf-8",
            )

            result = execute_exec_shell_full_command(
                {"command": "grep -n -C 1 needle note.txt"},
                workdir=workdir,
            )

        self.assertIn("returned_line_ranges: 1-4", result)
        self.assertEqual(result.count("returned_line_ranges:"), 1)
    def test_single_file_numbered_search_reports_exact_identity_and_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            content = "alpha\nneedle one\ncontext\nneedle two\nomega\n" + "filler\n" * 2000
            (workdir / "note file.txt").write_text(content, encoding="utf-8")

            result = execute_exec_shell_full_command(
                {"command": "grep -n -C 1 'needle' 'note file.txt'"},
                workdir=workdir,
            )

        self.assertIn("targeted_file_search: true", result)
        self.assertIn("path: note file.txt", result)
        self.assertIn(f"bytes: {len(content.encode('utf-8'))}", result)
        self.assertIn("lines: 2005", result)
        self.assertIn(f"sha256: {hashlib.sha256(content.encode('utf-8')).hexdigest()}", result)
        self.assertIn("search_coverage: complete_file", result)
        self.assertIn("semantic_coverage: partial", result)
        self.assertIn("returned_line_ranges: 1-5", result)
        self.assertIn("result_truncated: false", result)
        self.assertLessEqual(len(result.encode("utf-8")), 800)

    def test_exact_sed_line_range_uses_internal_page_with_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            content = "one\ntwo\nthree\nfour\n"
            (workdir / "note file.txt").write_text(content, encoding="utf-8")

            result = execute_exec_shell_full_command(
                {"command": "sed -n '2,3p' 'note file.txt'"},
                workdir=workdir,
            )

        self.assertIn("file_display_result: true", result)
        self.assertIn("coverage: partial", result)
        self.assertIn("line_range: 2-3", result)
        self.assertIn(hashlib.sha256(content.encode()).hexdigest(), result)
        self.assertTrue(result.endswith("two\nthree\n"))

    def test_exact_sed_range_rejects_more_than_two_hundred_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("line\n" * 300, encoding="utf-8")

            result = execute_exec_shell_full_command(
                {"command": "sed -n '1,201p' note.txt"},
                workdir=workdir,
            )

        self.assertIn("between 1 and 200 lines", result)

    def test_search_max_count_reports_partial_file_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("needle\nneedle\n" + "filler\n" * 2000, encoding="utf-8")

            result = execute_exec_shell_full_command(
                {"command": "grep -n -m 1 needle note.txt"},
                workdir=workdir,
            )

        self.assertIn("targeted_file_search: true", result)
        self.assertIn("search_coverage: partial_file", result)
        self.assertIn("returned_line_ranges: 1", result)

    def test_single_file_search_no_match_is_partial_semantic_evidence_not_shell_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            content = "alpha\nbeta\n" + "filler\n" * 2000
            (workdir / "note.txt").write_text(content, encoding="utf-8")

            result = execute_exec_shell_full_command(
                {"command": "grep -n absent note.txt"},
                workdir=workdir,
            )

        self.assertIn("targeted_file_search: true", result)
        self.assertIn("search_coverage: complete_file", result)
        self.assertIn("semantic_coverage: partial", result)
        self.assertIn("match_count: 0", result)
        self.assertIn("returned_line_ranges: unavailable", result)
        self.assertIn("(no lexical matches)", result)
        self.assertNotIn("shell_command_failed", result)

    def test_unnumbered_search_derives_ranges_only_from_exact_source_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text(
                "alpha\nneedle\nomega\n" + "filler\n" * 2000,
                encoding="utf-8",
            )

            result = execute_exec_shell_full_command(
                {"command": "grep needle note.txt"},
                workdir=workdir,
            )

        self.assertIn("targeted_file_search: true", result)
        self.assertIn("returned_line_ranges: 2", result)

    def test_short_single_file_search_keeps_existing_shell_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text("alpha\nneedle\nomega\n", encoding="utf-8")

            result = execute_exec_shell_full_command(
                {"command": "grep needle note.txt"},
                workdir=workdir,
            )

        self.assertEqual(result, "needle")
        self.assertNotIn("targeted_file_search: true", result)

    def test_quoted_regex_alternation_is_not_mistaken_for_a_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "note.txt").write_text(
                "alpha\nbeta\nomega\n" + "filler\n" * 2000,
                encoding="utf-8",
            )

            result = execute_exec_shell_full_command(
                {"command": 'grep -nE "alpha|omega" note.txt'},
                workdir=workdir,
            )

        self.assertIn("targeted_file_search: true", result)
        self.assertIn("returned_line_ranges: 1,3", result)

    def test_unstructured_search_forms_keep_existing_bounded_output(self) -> None:
        commands = (
            "grep -n needle note.txt other.txt",
            "grep -n needle note.txt | head -n 1",
            "grep -n needle note.txt > matches.txt",
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            for name in ("note.txt", "other.txt"):
                (workdir / name).write_text("needle\n", encoding="utf-8")
            for command in commands:
                with self.subTest(command=command):
                    result = execute_exec_shell_full_command({"command": command}, workdir=workdir)
                    self.assertNotIn("targeted_file_search: true", result)

    def test_successful_compound_cat_keeps_normal_bounded_shell_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            content = "alpha\n" * 2000
            (workdir / "note file.txt").write_text(content, encoding="utf-8")

            result = execute_exec_shell_full_command(
                {
                    "command": (
                        "sha256sum 'note file.txt' && wc -c 'note file.txt' && "
                        "wc -l 'note file.txt' && cat 'note file.txt'"
                    )
                },
                workdir=workdir,
            )

        self.assertNotIn("full_document_snapshot: true", result)
        self.assertIn(hashlib.sha256(content.encode()).hexdigest(), result)
        self.assertIn("[truncated]", result)
        self.assertLessEqual(len(result.encode("utf-8")), 12_012)

    def test_ambiguous_compound_cat_forms_do_not_claim_full_snapshot(self) -> None:
        commands = (
            "cat long.txt | head -n 2",
            "cat long.txt > copy.txt",
            "cat long.txt || true",
            "cat long.txt; printf done",
            "cat long.txt && cat other.txt",
            "cat long.txt && wc -l long.txt",
            "wc -l long.txt && cat -n long.txt",
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            content = "alpha\n" * 2000
            (workdir / "long.txt").write_text(content, encoding="utf-8")
            (workdir / "other.txt").write_text(content, encoding="utf-8")
            for command in commands:
                with self.subTest(command=command):
                    result = execute_exec_shell_full_command({"command": command}, workdir=workdir)
                    self.assertNotIn("full_document_snapshot: true", result)

    def test_invalid_utf8_search_result_is_not_enriched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "binary.txt").write_bytes(b"needle\n\xff\n")

            result = execute_exec_shell_full_command(
                {"command": "grep -a -n needle binary.txt"},
                workdir=workdir,
            )

        self.assertNotIn("targeted_file_search: true", result)

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
