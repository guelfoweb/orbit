from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal import cli


class CliDownloadDispatchTests(unittest.TestCase):
    def test_orbit_help_mentions_download_command(self) -> None:
        parser = cli.build_parser()
        help_text = parser.format_help()

        self.assertIn("orbit download <repo-or-file.gguf>", help_text)
        self.assertIn("orbit download --mmproj <repo>", help_text)
        self.assertIn("orbit download --all <repo>", help_text)

    def test_main_dispatches_download_subcommand(self) -> None:
        with mock.patch("orbit.terminal.cli.native_download_main", return_value=7) as mocked:
            code = cli.main(["download", "--mmproj", "ggml-org/gemma-4-12B-it-GGUF"])

        self.assertEqual(code, 7)
        mocked.assert_called_once_with(["download", "--mmproj", "ggml-org/gemma-4-12B-it-GGUF"])


if __name__ == "__main__":
    unittest.main()
