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

    def test_parser_accepts_verbose_and_quiet(self) -> None:
        args = build_cli.build_parser().parse_args(["--verbose", "--jobs", "2"])
        self.assertTrue(args.verbose)
        self.assertFalse(args.quiet)

        args = build_cli.build_parser().parse_args(["--quiet"])
        self.assertTrue(args.quiet)
        self.assertFalse(args.verbose)

    def test_resolve_source_root_uses_bundled_tree_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundled"
            root.mkdir(parents=True)
            (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)\n", encoding="utf-8")

            with mock.patch("orbit.native_llama.build_cli.BUNDLED_SOURCE_ROOT", root):
                resolved = build_cli._resolve_source_root(None, None)

        self.assertEqual(resolved, root)

    def test_main_rejects_verbose_and_quiet_together(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            code = build_cli.main(["--verbose", "--quiet"])

        self.assertEqual(code, 1)
        self.assertIn("--verbose and --quiet cannot be used together", stream.getvalue())

    def test_run_verbose_streams_output(self) -> None:
        reporter = build_cli.BuildReporter(verbose=True, quiet=False)
        stream = io.StringIO()

        class FakeStdout:
            def readline(self) -> str:
                return ""

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = FakeStdout()
                self._poll_calls = 0

            def poll(self) -> int | None:
                self._poll_calls += 1
                return 0 if self._poll_calls > 1 else None

            def wait(self) -> int:
                return 0

        def fake_enqueue(_stream, output_queue):
            output_queue.put("line one\n")
            output_queue.put("line two\n")
            output_queue.put(None)

        with (
            mock.patch("orbit.native_llama.build_cli.subprocess.Popen", return_value=FakeProcess()),
            mock.patch("orbit.native_llama.build_cli._enqueue_process_output", side_effect=fake_enqueue),
            contextlib.redirect_stdout(stream),
        ):
            build_cli._run(["cmake", "--build", "x"], reporter=reporter, heartbeat_label="building")

        text = stream.getvalue()
        self.assertIn("line one", text)
        self.assertIn("line two", text)

    def test_run_quiet_suppresses_heartbeat_and_lines(self) -> None:
        reporter = build_cli.BuildReporter(verbose=False, quiet=True)
        stream = io.StringIO()

        class FakeStdout:
            def readline(self) -> str:
                return ""

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = FakeStdout()
                self._poll_calls = 0

            def poll(self) -> int | None:
                self._poll_calls += 1
                return 0 if self._poll_calls > 1 else None

            def wait(self) -> int:
                return 0

        def fake_enqueue(_stream, output_queue):
            output_queue.put("Built target llama\n")
            output_queue.put(None)

        with (
            mock.patch("orbit.native_llama.build_cli.subprocess.Popen", return_value=FakeProcess()),
            mock.patch("orbit.native_llama.build_cli._enqueue_process_output", side_effect=fake_enqueue),
            contextlib.redirect_stdout(stream),
        ):
            build_cli._run(["cmake", "--build", "x"], reporter=reporter, heartbeat_label="building")

        self.assertEqual(stream.getvalue(), "")

    def test_run_failure_reports_command_exit_code_and_tail(self) -> None:
        reporter = build_cli.BuildReporter(verbose=False, quiet=False)

        class FakeStdout:
            def readline(self) -> str:
                return ""

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = FakeStdout()
                self._poll_calls = 0

            def poll(self) -> int | None:
                self._poll_calls += 1
                return 7 if self._poll_calls > 1 else None

            def wait(self) -> int:
                return 7

        def fake_enqueue(_stream, output_queue):
            output_queue.put("first line\n")
            output_queue.put("last useful line\n")
            output_queue.put(None)

        with (
            mock.patch("orbit.native_llama.build_cli.subprocess.Popen", return_value=FakeProcess()),
            mock.patch("orbit.native_llama.build_cli._enqueue_process_output", side_effect=fake_enqueue),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                build_cli._run(["cmake", "--build", "x"], reporter=reporter, heartbeat_label="building")

        message = str(ctx.exception)
        self.assertIn("exit code 7", message)
        self.assertIn("cmake --build x", message)
        self.assertIn("last useful line", message)

    def test_main_success_prints_updated_next_port(self) -> None:
        stream = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(stream),
        ):
            root = Path(tmp) / "bundled"
            root.mkdir(parents=True)
            (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)\n", encoding="utf-8")
            with (
                mock.patch("orbit.native_llama.build_cli.BUNDLED_SOURCE_ROOT", root),
                mock.patch("orbit.native_llama.build_cli.shutil.which", return_value="/usr/bin/cmake"),
                mock.patch("orbit.native_llama.build_cli._run"),
                mock.patch("orbit.native_llama.build_cli._copy_runtime_libraries"),
                mock.patch("orbit.native_llama.build_cli._build_packaged_shims"),
                mock.patch("orbit.native_llama.build_cli.platform_runtime_libs", return_value=[]),
            ):
                code = build_cli.main(["--quiet"])

        self.assertEqual(code, 0)
        self.assertIn("next: orbit server --port 12120", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
