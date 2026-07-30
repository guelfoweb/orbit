from __future__ import annotations

import os
from pathlib import Path
import random
import stat
import tempfile
import unittest

from orbit.runtime.file_patch import (
    MAX_PATCH_BYTES,
    _write_prepared_patch,
    execute_apply_patch,
    prepare_file_patch,
    validate_file_patch,
)


def _replacement_patch(path: str, old: str, new: str, *, line: int = 1) -> str:
    return f"--- {path}\n+++ {path}\n@@ -{line},1 +{line},1 @@\n-{old}\n+{new}\n"


class FilePatchTests(unittest.TestCase):
    def test_applies_one_exact_existing_file_patch_and_preserves_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "mathbox.py"
            target.write_text("def average(values):\n    return sum(values) // len(values)\n", encoding="utf-8")
            target.chmod(0o750)

            result = execute_apply_patch(
                {
                    "patch": (
                        "--- mathbox.py\n"
                        "+++ mathbox.py\n"
                        "@@ -1,2 +1,2 @@\n"
                        " def average(values):\n"
                        "-    return sum(values) // len(values)\n"
                        "+    return sum(values) / len(values)\n"
                    )
                },
                workdir=root,
            )

            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "def average(values):\n    return sum(values) / len(values)\n",
            )
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o750)
            self.assertIn("patch_applied: true", result)
            self.assertIn("path: mathbox.py", result)
            self.assertIn("hunks: 1", result)
            self.assertIn("added_lines: 1", result)
            self.assertIn("removed_lines: 1", result)

    def test_accepts_conventional_a_and_b_header_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "src" / "item.txt"
            target.parent.mkdir()
            target.write_text("before\n", encoding="utf-8")

            result = execute_apply_patch(
                {"patch": "--- a/src/item.txt\n+++ b/src/item.txt\n@@ -1 +1 @@\n-before\n+after\n"},
                workdir=root,
            )

            self.assertIn("patch_applied: true", result)
            self.assertEqual(target.read_text(encoding="utf-8"), "after\n")

    def test_preserves_literal_a_and_b_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("a", "b"):
                target = root / directory / "item.txt"
                target.parent.mkdir()
                target.write_text("before\n", encoding="utf-8")

                result = execute_apply_patch(
                    {
                        "patch": (
                            f"--- {directory}/item.txt\n"
                            f"+++ {directory}/item.txt\n"
                            "@@ -1 +1 @@\n"
                            "-before\n"
                            "+after\n"
                        )
                    },
                    workdir=root,
                )

                self.assertIn(f"path: {directory}/item.txt", result)
                self.assertEqual(target.read_text(encoding="utf-8"), "after\n")

    def test_rejects_context_mismatch_without_modifying_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "item.txt"
            target.write_text("actual\n", encoding="utf-8")
            patch = _replacement_patch("item.txt", "assumed", "changed")

            result = execute_apply_patch({"patch": patch}, workdir=root)

            self.assertEqual(result, "error: apply_patch rejected: context_mismatch")
            self.assertEqual(target.read_text(encoding="utf-8"), "actual\n")

    def test_rejects_path_escape_absolute_rename_create_and_multiple_files(self) -> None:
        cases = {
            "escape": "--- ../item.txt\n+++ ../item.txt\n@@ -1 +1 @@\n-a\n+b\n",
            "absolute": "--- /tmp/item.txt\n+++ /tmp/item.txt\n@@ -1 +1 @@\n-a\n+b\n",
            "rename": "--- item.txt\n+++ other.txt\n@@ -1 +1 @@\n-a\n+b\n",
            "create": "--- missing.txt\n+++ missing.txt\n@@ -0,0 +1 @@\n+new\n",
            "multiple": (
                "--- item.txt\n+++ item.txt\n@@ -1 +1 @@\n-a\n+b\n"
                "--- other.txt\n+++ other.txt\n@@ -1 +1 @@\n-x\n+y\n"
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "item.txt").write_text("a\n", encoding="utf-8")
            (root / "other.txt").write_text("x\n", encoding="utf-8")
            for label, patch in cases.items():
                with self.subTest(label=label):
                    result = execute_apply_patch({"patch": patch}, workdir=root)
                    self.assertTrue(result.startswith("error: apply_patch rejected:"))
            self.assertEqual(
                execute_apply_patch(
                    {"patch": "--- missing.txt\n+++ missing.txt\n@@ -0,0 +1 @@\n+new\n"},
                    workdir=root,
                ),
                "error: apply_patch rejected: target_not_found",
            )
            self.assertEqual((root / "item.txt").read_text(encoding="utf-8"), "a\n")
            self.assertEqual((root / "other.txt").read_text(encoding="utf-8"), "x\n")

    def test_rejects_symlink_binary_non_utf8_and_missing_final_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real.txt"
            real.write_text("value\n", encoding="utf-8")
            (root / "link.txt").symlink_to(real)
            (root / "binary.txt").write_bytes(b"a\x00b\n")
            (root / "invalid.txt").write_bytes(b"\xff\n")
            (root / "unterminated.txt").write_text("value", encoding="utf-8")
            cases = {
                "link.txt": "symlink_not_allowed",
                "binary.txt": "binary_target",
                "invalid.txt": "target_not_utf8",
                "unterminated.txt": "target_missing_final_newline",
            }
            for name, code in cases.items():
                with self.subTest(name=name):
                    patch = _replacement_patch(name, "value", "changed")
                    self.assertEqual(validate_file_patch(patch, workdir=root), code)

    def test_rejects_malformed_ambiguous_and_oversized_patches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "item.txt").write_text("a\n", encoding="utf-8")
            cases = {
                "missing_header": "@@ -1 +1 @@\n-a\n+b\n",
                "count_mismatch": "--- item.txt\n+++ item.txt\n@@ -1,2 +1 @@\n-a\n+b\n",
                "empty_hunk": "--- item.txt\n+++ item.txt\n@@ -1,0 +1,0 @@\n",
                "technical_marker": "--- item.txt\n+++ item.txt\n@@ -1 +1 @@\n-a\n+b\n\\ No newline at end of file\n",
                "carriage_return": "--- item.txt\r\n+++ item.txt\r\n@@ -1 +1 @@\r\n-a\r\n+b\r\n",
                "new_position_mismatch": "--- item.txt\n+++ item.txt\n@@ -1 +2 @@\n-a\n+b\n",
                "invalid_unicode": "\ud800",
                "oversized": "x" * (MAX_PATCH_BYTES + 1),
            }
            for label, patch in cases.items():
                with self.subTest(label=label):
                    self.assertIsNotNone(validate_file_patch(patch, workdir=root))

    def test_rejects_target_changed_between_preflight_and_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "item.txt"
            target.write_text("before\n", encoding="utf-8")
            prepared = prepare_file_patch(
                _replacement_patch("item.txt", "before", "after"),
                workdir=root,
            )
            target.write_text("external\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "target_changed"):
                _write_prepared_patch(prepared)

            self.assertEqual(target.read_text(encoding="utf-8"), "external\n")
            self.assertEqual(list(root.glob(".item.txt.orbit-*")), [])

    def test_property_style_single_line_replacements_preserve_all_other_lines(self) -> None:
        generator = random.Random(1729)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for repetition in range(40):
                lines = [f"line-{repetition}-{index}-{generator.randrange(10_000)}" for index in range(8)]
                changed_index = generator.randrange(len(lines))
                replacement = f"replacement-{repetition}-{generator.randrange(10_000)}"
                target = root / "item.txt"
                target.write_text("\n".join(lines) + "\n", encoding="utf-8")

                result = execute_apply_patch(
                    {
                        "patch": _replacement_patch(
                            "item.txt",
                            lines[changed_index],
                            replacement,
                            line=changed_index + 1,
                        )
                    },
                    workdir=root,
                )

                expected = list(lines)
                expected[changed_index] = replacement
                self.assertIn("patch_applied: true", result)
                self.assertEqual(target.read_text(encoding="utf-8"), "\n".join(expected) + "\n")


if __name__ == "__main__":
    unittest.main()
