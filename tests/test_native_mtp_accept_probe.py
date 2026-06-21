from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orbit.native_llama.mtp_accept_probe import build_mtp_accept_probe_helper, run_mtp_accept_probe
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_server.app import build_parser


class NativeMtpAcceptProbeTests(unittest.TestCase):
    def test_accept_probe_returns_fallback_when_draft_is_missing(self) -> None:
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

        result = run_mtp_accept_probe(llama_root=paths.llama_root, paths=paths, runner=lambda *args, **kwargs: None)

        self.assertTrue(result.enabled)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "draft-mtp-missing")

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

            with mock.patch("orbit.native_llama.mtp_accept_probe.packaged_shim_path", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "failed to build mtp accept probe helper"):
                    build_mtp_accept_probe_helper(llama_root=llama_root, build_dir=root / "out", runner=runner)

    def test_accept_probe_reports_failure_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "out"
            build_dir.mkdir()
            helper = build_dir / "orbit-mtp-accept-probe"
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
                    stdout = json.dumps({"ok": False, "error": "failed to validate drafted tokens on target", "draft_tokens": 3})
                    stderr = ""
                return Completed()

            result = run_mtp_accept_probe(llama_root=root, paths=paths, build_dir=build_dir, runner=runner)

        self.assertTrue(result.enabled)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "failed to validate drafted tokens on target")

    def test_accept_probe_reports_success_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "out"
            build_dir.mkdir()
            helper = build_dir / "orbit-mtp-accept-probe"
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
                            "accepted_tokens": 2,
                            "rejected_tokens": 1,
                            "acceptance_ratio": 0.666667,
                            "target_decode_calls": 2,
                            "draft_decode_calls": 1,
                            "elapsed_ms": 12.5,
                            "rss_before_kb": 100,
                            "rss_after_kb": 200,
                            "rss_peak_kb": 300,
                        }
                    )
                    stderr = ""
                return Completed()

            result = run_mtp_accept_probe(llama_root=root, paths=paths, build_dir=build_dir, runner=runner)

        self.assertTrue(result.enabled)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(result.draft_tokens, 3)
        self.assertEqual(result.accepted_tokens, 2)
        self.assertEqual(result.rejected_tokens, 1)
        self.assertAlmostEqual(result.acceptance_ratio or 0.0, 0.666667, places=5)
        self.assertEqual(result.target_decode_calls, 2)

    def test_parser_accepts_enable_mtp_accept_probe_flag(self) -> None:
        args = build_parser().parse_args(["--enable-mtp-accept-probe"])
        self.assertTrue(args.enable_mtp_accept_probe)


if __name__ == "__main__":
    unittest.main()
from unittest import mock
