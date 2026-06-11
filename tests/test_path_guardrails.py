from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.path_guardrails import resolve_inside_workdir, validate_existing_file_path


class PathGuardrailsTests(unittest.TestCase):
    def test_resolve_inside_workdir_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve_inside_workdir("../outside.txt", workdir=Path(tmp))

        self.assertEqual(result, "error: path escapes workdir")

    def test_validate_existing_file_path_requires_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            directory = workdir / "dir"
            directory.mkdir()

            result = validate_existing_file_path("dir", workdir=workdir)

        self.assertEqual(result, "error: path is not a file: dir")

    def test_validate_existing_file_path_returns_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            target = workdir / "note.txt"
            target.write_text("hello", encoding="utf-8")

            result = validate_existing_file_path("note.txt", workdir=workdir)

        self.assertEqual(result, target.resolve())


if __name__ == "__main__":
    unittest.main()
