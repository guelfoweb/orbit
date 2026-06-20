from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orbit.native_llama.mtp_dry_run import build_mtp_dry_run_helper, run_mtp_dry_run
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_server.app import build_parser


class NativeMtpDryRunTests(unittest.TestCase):
    def test_dry_run_returns_fallback_when_draft_is_missing(self) -> None:
        paths = NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/target.gguf"),
            draft_mtp_model=None,
            mtp_available=False,
            fallback_reason="draft-mtp-missing",
            model_id="gemma4-12b-it-q4km",
        )

        result = run_mtp_dry_run(llama_root=paths.llama_root, paths=paths, runner=lambda *args, **kwargs: None)

        self.assertTrue(result.enabled)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "draft-mtp-missing")
        self.assertEqual(result.draft_tokens, 0)

    def test_build_helper_reports_compile_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_root = root / "llama"
            (llama_root / "include").mkdir(parents=True)
            (llama_root / "common").mkdir(parents=True)
            (llama_root / "ggml/include").mkdir(parents=True)
            (llama_root / "src").mkdir(parents=True)
            (llama_root / "build/bin").mkdir(parents=True)

            def runner(cmd, **kwargs):
                class Completed:
                    returncode = 1
                    stdout = ""
                    stderr = "compile failed"
                return Completed()

            with mock.patch("orbit.native_llama.mtp_dry_run.packaged_shim_path", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "failed to build mtp dry run helper"):
                    build_mtp_dry_run_helper(llama_root=llama_root, build_dir=root / "out", runner=runner)

    def test_dry_run_reports_failure_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "out"
            build_dir.mkdir()
            helper = build_dir / "orbit-mtp-dry-run"
            helper.write_text("", encoding="utf-8")
            paths = NativeLlamaPaths(
                llama_root=root,
                build_bin=root / "build/bin",
                library=root / "build/bin/libllama.so",
                model=root / "target.gguf",
                draft_mtp_model=root / "draft.gguf",
                mtp_available=True,
                fallback_reason=None,
                model_id="gemma4-12b-it-q4km",
            )

            def runner(cmd, **kwargs):
                class Completed:
                    returncode = 1
                    stdout = json.dumps({"ok": False, "error": "failed to decode target prompt", "draft_tokens": 0})
                    stderr = ""
                return Completed()

            result = run_mtp_dry_run(llama_root=root, paths=paths, build_dir=build_dir, runner=runner)

        self.assertTrue(result.enabled)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "failed to decode target prompt")

    def test_dry_run_reports_success_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "out"
            build_dir.mkdir()
            helper = build_dir / "orbit-mtp-dry-run"
            helper.write_text("", encoding="utf-8")
            paths = NativeLlamaPaths(
                llama_root=root,
                build_bin=root / "build/bin",
                library=root / "build/bin/libllama.so",
                model=root / "target.gguf",
                draft_mtp_model=root / "draft.gguf",
                mtp_available=True,
                fallback_reason=None,
                model_id="gemma4-12b-it-q4km",
            )

            def runner(cmd, **kwargs):
                class Completed:
                    returncode = 0
                    stdout = json.dumps(
                        {
                            "ok": True,
                            "draft_tokens": 3,
                            "rss_before_kb": 100,
                            "rss_after_kb": 200,
                            "rss_peak_kb": 300,
                            "prompt_decode_s": 0.01,
                            "draft_s": 0.02,
                        }
                    )
                    stderr = ""
                return Completed()

            result = run_mtp_dry_run(llama_root=root, paths=paths, build_dir=build_dir, runner=runner)

        self.assertTrue(result.enabled)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(result.draft_tokens, 3)
        self.assertEqual(result.rss_peak_kb, 300)
        self.assertEqual(result.draft_s, 0.02)

    def test_parser_accepts_enable_mtp_dry_run_flag(self) -> None:
        args = build_parser().parse_args(["--enable-mtp-dry-run"])
        self.assertTrue(args.enable_mtp_dry_run)


if __name__ == "__main__":
    unittest.main()
from unittest import mock
