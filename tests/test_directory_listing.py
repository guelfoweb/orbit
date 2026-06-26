from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.directory_listing import execute_list_directory


class DirectoryListingTests(unittest.TestCase):
    def test_non_recursive_listing_is_compact_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.txt").write_text("b", encoding="utf-8")
            (root / "a").mkdir()
            (root / "a" / "nested.txt").write_text("nested", encoding="utf-8")

            result = execute_list_directory({"path": "."}, workdir=root)

        self.assertIn("directory_listing: path=. recursive=false", result)
        self.assertIn("[dir] a/", result)
        self.assertIn("[file] b.txt", result)
        self.assertNotIn("nested.txt", result)
        self.assertLess(result.index("[dir] a/"), result.index("[file] b.txt"))

    def test_recursive_listing_honors_max_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a" / "b").mkdir(parents=True)
            (root / "a" / "one.txt").write_text("1", encoding="utf-8")
            (root / "a" / "b" / "two.txt").write_text("2", encoding="utf-8")

            result = execute_list_directory({"path": ".", "recursive": True, "max_depth": 2}, workdir=root)

        self.assertIn("[dir] a/", result)
        self.assertIn("[dir] a/b/", result)
        self.assertIn("[file] a/one.txt", result)
        self.assertNotIn("two.txt", result)

    def test_max_entries_truncates_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(5):
                (root / f"{index}.txt").write_text(str(index), encoding="utf-8")

            result = execute_list_directory({"path": ".", "max_entries": 2}, workdir=root)

        self.assertIn("shown=2", result)
        self.assertIn("total_seen=5", result)
        self.assertIn("truncated=true", result)

    def test_hidden_entries_are_excluded_by_default_and_included_on_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".secret").write_text("hidden", encoding="utf-8")
            (root / "shown.txt").write_text("shown", encoding="utf-8")

            default = execute_list_directory({"path": "."}, workdir=root)
            hidden = execute_list_directory({"path": ".", "include_hidden": True}, workdir=root)

        self.assertIn("shown.txt", default)
        self.assertNotIn(".secret", default)
        self.assertIn(".secret", hidden)

    def test_files_only_and_dirs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dir").mkdir()
            (root / "file.txt").write_text("x", encoding="utf-8")

            files = execute_list_directory({"path": ".", "files_only": True}, workdir=root)
            dirs = execute_list_directory({"path": ".", "dirs_only": True}, workdir=root)

        self.assertIn("[file] file.txt", files)
        self.assertNotIn("[dir] dir/", files)
        self.assertIn("[dir] dir/", dirs)
        self.assertNotIn("[file] file.txt", dirs)

    def test_symlink_is_listed_without_recursive_follow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "target").mkdir()
            (root / "target" / "inside.txt").write_text("x", encoding="utf-8")
            (root / "link").symlink_to(root / "target", target_is_directory=True)

            result = execute_list_directory({"path": ".", "recursive": True, "max_depth": 3}, workdir=root)

        self.assertIn("[symlink] link ->", result)
        self.assertIn("[file] target/inside.txt", result)
        self.assertNotIn("link/inside.txt", result)

    def test_missing_and_non_directory_paths_are_structured_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "file.txt").write_text("x", encoding="utf-8")

            missing = execute_list_directory({"path": "missing"}, workdir=root)
            file_path = execute_list_directory({"path": "file.txt"}, workdir=root)

        self.assertIn("error=true status=not_found", missing)
        self.assertIn("error=true status=not_directory", file_path)

    def test_outside_workdir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_list_directory({"path": ".."}, workdir=Path(tmp))

        self.assertIn("error=true status=path_outside_workdir", result)


if __name__ == "__main__":
    unittest.main()
