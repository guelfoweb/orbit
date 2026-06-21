from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orbit.native_llama.mtp_probe import build_mtp_probe_helper, run_mtp_probe
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_server.app import build_parser


class NativeMtpProbeTests(unittest.TestCase):
    def test_build_helper_prefers_packaged_probe_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packaged = Path(tmp) / "orbit-mtp-probe"
            packaged.write_text("", encoding="utf-8")
            with mock.patch("orbit.native_llama.mtp_probe.packaged_shim_path", return_value=packaged):
                helper = build_mtp_probe_helper(llama_root=None)

        self.assertEqual(helper, packaged)

    def test_probe_returns_fallback_when_draft_is_missing(self) -> None:
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

        result = run_mtp_probe(llama_root=paths.llama_root, paths=paths, runner=lambda *args, **kwargs: None)

        self.assertTrue(result.enabled)
        self.assertFalse(result.initialized)
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

            with mock.patch("orbit.native_llama.mtp_probe.packaged_shim_path", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "failed to build mtp probe helper"):
                    build_mtp_probe_helper(llama_root=llama_root, build_dir=root / "out", runner=runner)

    def test_build_helper_requires_legacy_root_when_no_packaged_artifact_exists(self) -> None:
        with mock.patch("orbit.native_llama.mtp_probe.packaged_shim_path", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "missing native build inputs for orbit-mtp-probe"):
                build_mtp_probe_helper(llama_root=None)

    def test_probe_reports_init_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "out"
            build_dir.mkdir()
            helper = build_dir / "orbit-mtp-probe"
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
                    stdout = json.dumps({"ok": False, "error": "failed to initialize speculative MTP state"})
                    stderr = ""
                return Completed()

            result = run_mtp_probe(llama_root=root, paths=paths, build_dir=build_dir, runner=runner)

        self.assertTrue(result.enabled)
        self.assertFalse(result.initialized)
        self.assertEqual(result.error, "failed to initialize speculative MTP state")

    def test_probe_reports_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "out"
            build_dir.mkdir()
            helper = build_dir / "orbit-mtp-probe"
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
                    stdout = json.dumps({"ok": True, "rss_before_kb": 100, "rss_after_kb": 200, "rss_peak_kb": 300})
                    stderr = ""
                return Completed()

            result = run_mtp_probe(llama_root=root, paths=paths, build_dir=build_dir, runner=runner)

        self.assertTrue(result.enabled)
        self.assertTrue(result.initialized)
        self.assertIsNone(result.error)
        self.assertEqual(result.rss_before_kb, 100)
        self.assertEqual(result.rss_peak_kb, 300)

    def test_parser_accepts_enable_mtp_probe_flag(self) -> None:
        args = build_parser().parse_args(["--enable-mtp-probe"])
        self.assertTrue(args.enable_mtp_probe)


if __name__ == "__main__":
    unittest.main()
