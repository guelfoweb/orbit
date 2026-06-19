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
from orbit.native_llama import download_cli


class CliDownloadDispatchTests(unittest.TestCase):
    def test_orbit_help_mentions_download_command(self) -> None:
        parser = cli.build_parser()
        help_text = parser.format_help()

        self.assertIn("orbit build-native [options]", help_text)
        self.assertIn("orbit download <repo-or-file.gguf>", help_text)
        self.assertIn("orbit download --mmproj <repo>", help_text)
        self.assertIn("orbit download --all [repo]", help_text)
        self.assertIn("orbit server [options]", help_text)
        self.assertIn("orbit bench-core [options]", help_text)
        self.assertIn("orbit release-confidence [options]", help_text)

    def test_main_dispatches_download_subcommand(self) -> None:
        with mock.patch("orbit.terminal.cli.native_download_main", return_value=7) as mocked:
            code = cli.main(["download", "--mmproj", "ggml-org/gemma-4-12B-it-GGUF"])

        self.assertEqual(code, 7)
        mocked.assert_called_once_with(["download", "--mmproj", "ggml-org/gemma-4-12B-it-GGUF"])

    def test_main_dispatches_build_native_subcommand(self) -> None:
        with mock.patch("orbit.terminal.cli.native_build_main", return_value=5) as mocked:
            code = cli.main(["build-native", "--jobs", "6"])

        self.assertEqual(code, 5)
        mocked.assert_called_once_with(["--jobs", "6"])

    def test_main_dispatches_server_subcommand(self) -> None:
        with mock.patch("orbit.terminal.cli.run_server", return_value=11) as mocked:
            code = cli.main(["server", "--port", "11976"])

        self.assertEqual(code, 11)
        mocked.assert_called_once_with(["--port", "11976"])

    def test_main_dispatches_bench_core_subcommand(self) -> None:
        with mock.patch("orbit.terminal.cli.bench_core_main", return_value=12) as mocked:
            code = cli.main(["bench-core", "--base-url", "http://127.0.0.1:11976"])

        self.assertEqual(code, 12)
        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.args[0], ["--base-url", "http://127.0.0.1:11976"])

    def test_main_dispatches_release_confidence_subcommand(self) -> None:
        with mock.patch("orbit.terminal.cli.release_confidence_main", return_value=13) as mocked:
            code = cli.main(["release-confidence", "--list"])

        self.assertEqual(code, 13)
        mocked.assert_called_once_with(["--list"])

    def test_download_all_without_repo_uses_default_manifest(self) -> None:
        args = download_cli.build_parser().parse_args(["--all"])

        with mock.patch("orbit.native_llama.download_cli.download_all_for_repo") as mocked:
            mocked.return_value.results = ()
            code = download_cli._download(args)

        self.assertEqual(code, 0)
        mocked.assert_called_once_with("ggml-org/gemma-4-12B-it-GGUF", models_dir=mock.ANY)

    def test_download_without_spec_fails_when_not_all(self) -> None:
        args = download_cli.build_parser().parse_args([])

        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            code = download_cli._download(args)

        self.assertEqual(code, 1)
        self.assertIn("expected Hugging Face repo or repo/file", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
