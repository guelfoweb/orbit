from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.native_llama import build_cli


class NativeBuildCliTests(unittest.TestCase):
    def test_validate_llama_root_requires_cmakelists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            message = build_cli._validate_llama_root(root)
        self.assertIn("does not look like a llama.cpp source tree", message)

    def test_main_fails_cleanly_when_no_source_tree_found(self) -> None:
        stream = io.StringIO()
        with (
            mock.patch("orbit.native_llama.build_cli.BUNDLED_SOURCE_ROOT", ROOT / "missing-bundled-llama"),
            contextlib.redirect_stderr(stream),
        ):
            code = build_cli.main([])

        self.assertEqual(code, 1)
        self.assertIn("bundled llama.cpp sources are missing from Orbit", stream.getvalue())

    def test_main_fails_cleanly_when_cmake_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)\n", encoding="utf-8")
            stream = io.StringIO()
            with (
                mock.patch("orbit.native_llama.build_cli.shutil.which", return_value=None),
                contextlib.redirect_stderr(stream),
            ):
                code = build_cli.main(["--llama-root", str(root)])

        self.assertEqual(code, 1)
        self.assertIn("cmake not found in PATH", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
