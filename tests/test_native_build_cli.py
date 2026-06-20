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
        self.assertIn("does not look like a llama.cpp checkout", message)

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

    def test_parser_accepts_source_dir_and_with_mtp_shim(self) -> None:
        args = build_cli.build_parser().parse_args(["--source-dir", "/tmp/src", "--with-mtp-shim", "--jobs", "4"])

        self.assertEqual(args.source_dir, Path("/tmp/src"))
        self.assertTrue(args.with_mtp_shim)
        self.assertEqual(args.jobs, 4)

    def test_resolve_source_root_uses_bundled_tree_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundled"
            root.mkdir(parents=True)
            (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)\n", encoding="utf-8")

            with mock.patch("orbit.native_llama.build_cli.BUNDLED_SOURCE_ROOT", root):
                resolved = build_cli._resolve_source_root(None, None)

        self.assertEqual(resolved, root)


if __name__ == "__main__":
    unittest.main()
