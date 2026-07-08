from __future__ import annotations

import io
from types import SimpleNamespace
import unittest
from unittest import mock
from contextlib import redirect_stdout

from orbit.dev import bench_core


class BenchCoreMetadataTests(unittest.TestCase):
    def test_metadata_on_prints_header_and_backend_props(self) -> None:
        completed_calls: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> SimpleNamespace:
            completed_calls.append(command)
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        with (
            mock.patch.object(bench_core, "_resolve_orbit_bin", return_value="/tmp/orbit"),
            mock.patch.object(bench_core, "_git_output", side_effect=["abc123", "v1"]),
            mock.patch.object(
                bench_core,
                "_fetch_backend_props",
                return_value={
                    "model": "gemma4:12b-it-native",
                    "ctx": 8192,
                    "threads": 6,
                    "threads_batch": 6,
                    "batch": 256,
                    "ubatch": 128,
                    "mtp_enabled": False,
                    "mtp_initialized": False,
                    "multimodal_available": True,
                    "gpu_layers": 0,
                    "ignored": "value",
                },
            ),
            mock.patch.object(bench_core.subprocess, "run", side_effect=fake_run),
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                code = bench_core.main(["--base-url", "http://127.0.0.1:12120", "--workdir", "workdir"])

        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("# orbit bench-core metadata", text)
        self.assertIn("orbit_commit: abc123", text)
        self.assertIn("orbit_tag: v1", text)
        self.assertIn("backend:", text)
        self.assertIn("  model: gemma4:12b-it-native", text)
        self.assertIn("  gpu_layers: 0", text)
        self.assertNotIn("ignored", text)
        self.assertEqual(len(completed_calls), len(bench_core.PROMPTS))

    def test_no_metadata_preserves_minimal_output(self) -> None:
        def fake_run(command: list[str], **_: object) -> SimpleNamespace:
            return SimpleNamespace(stdout="task output\n", stderr="", returncode=0)

        with (
            mock.patch.object(bench_core, "_resolve_orbit_bin", return_value="/tmp/orbit"),
            mock.patch.object(bench_core, "_print_metadata") as print_metadata,
            mock.patch.object(bench_core.subprocess, "run", side_effect=fake_run),
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                code = bench_core.main(["--no-metadata"])

        self.assertEqual(code, 0)
        self.assertNotIn("# orbit bench-core metadata", out.getvalue())
        print_metadata.assert_not_called()

    def test_backend_props_unavailable_does_not_fail(self) -> None:
        args = SimpleNamespace(base_url="http://127.0.0.1:12120", workdir="workdir", timeout=600, max_tokens=512)
        with (
            mock.patch.object(bench_core, "_git_output", side_effect=["abc123", "none"]),
            mock.patch.object(bench_core, "_fetch_backend_props", return_value=None),
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                bench_core._print_metadata(args, "/tmp/orbit")

        self.assertIn("backend_props: unavailable", out.getvalue())

    def test_git_output_commit_and_exact_tag(self) -> None:
        with mock.patch.object(
            bench_core.subprocess,
            "run",
            return_value=SimpleNamespace(stdout="abc123\n", stderr="", returncode=0),
        ):
            self.assertEqual(bench_core._git_output(["rev-parse", "HEAD"], fallback="unknown"), "abc123")

        with mock.patch.object(
            bench_core.subprocess,
            "run",
            return_value=SimpleNamespace(stdout="v0.0.1\n", stderr="", returncode=0),
        ):
            self.assertEqual(
                bench_core._git_output(["describe", "--tags", "--exact-match", "HEAD"], fallback="none"),
                "v0.0.1",
            )

    def test_git_output_fallbacks(self) -> None:
        with mock.patch.object(
            bench_core.subprocess,
            "run",
            return_value=SimpleNamespace(stdout="", stderr="", returncode=1),
        ):
            self.assertEqual(bench_core._git_output(["describe"], fallback="none"), "none")

        with mock.patch.object(bench_core.subprocess, "run", side_effect=OSError):
            self.assertEqual(bench_core._git_output(["rev-parse", "HEAD"], fallback="unknown"), "unknown")

    def test_task_arguments_are_unchanged(self) -> None:
        calls: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> SimpleNamespace:
            calls.append(command)
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        with (
            mock.patch.object(bench_core, "_resolve_orbit_bin", return_value="/tmp/orbit"),
            mock.patch.object(bench_core.subprocess, "run", side_effect=fake_run),
        ):
            with redirect_stdout(io.StringIO()):
                bench_core.main(
                    [
                        "--no-metadata",
                        "--base-url",
                        "http://backend",
                        "--workdir",
                        "workdir",
                        "--timeout",
                        "123",
                        "--max-tokens",
                        "456",
                    ]
                )

        self.assertEqual(len(calls), len(bench_core.PROMPTS))
        for call, (_, prompt) in zip(calls, bench_core.PROMPTS, strict=True):
            self.assertEqual(
                call,
                [
                    "/tmp/orbit",
                    "--base-url",
                    "http://backend",
                    "--workdir",
                    "workdir",
                    "--timeout",
                    "123",
                    "--max-tokens",
                    "456",
                    prompt,
                ],
            )


if __name__ == "__main__":
    unittest.main()
