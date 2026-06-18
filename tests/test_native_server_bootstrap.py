from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orbit.native_server.app import build_parser, resolve_bootstrap_paths


class NativeServerBootstrapTests(unittest.TestCase):
    def test_bootstrap_defaults_to_model_id_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_root = root / "llama"
            models_dir = root / "models"
            build_bin = llama_root / "build/bin"
            target = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "gemma-4-12B-it-Q4_K_M.gguf"
            mmproj = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "mmproj-gemma-4-12B-it-Q8_0.gguf"
            build_bin.mkdir(parents=True)
            (build_bin / "libllama.so").write_text("", encoding="utf-8")
            target.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")

            args = build_parser().parse_args(["--llama-root", str(llama_root), "--models-dir", str(models_dir), "--hf-cache", str(root / "hf")])
            paths = resolve_bootstrap_paths(args)

        self.assertEqual(paths.model, target)
        self.assertEqual(paths.mmproj_model, mmproj)
        self.assertEqual(paths.model_id, "gemma4-12b-it-q4km")

    def test_parser_accepts_think_flag(self) -> None:
        args = build_parser().parse_args(["--think", "on"])

        self.assertEqual(args.think, "on")

    def test_bootstrap_supports_legacy_direct_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_root = root / "llama"
            build_bin = llama_root / "build/bin"
            target = root / "manual.gguf"
            mmproj = root / "manual-mmproj.gguf"
            build_bin.mkdir(parents=True)
            (build_bin / "libllama.so").write_text("", encoding="utf-8")
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")

            args = build_parser().parse_args(["--llama-root", str(llama_root), "--model", str(target), "--mmproj", str(mmproj)])
            paths = resolve_bootstrap_paths(args)

        self.assertEqual(paths.model, target.resolve())
        self.assertEqual(paths.mmproj_model, mmproj.resolve())
        self.assertEqual(paths.model_id, "legacy-path")
        self.assertEqual(paths.fallback_reason, "legacy-model-path")

    def test_bootstrap_with_model_id_and_draft_present_exposes_mtp_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_root = root / "llama"
            models_dir = root / "models"
            build_bin = llama_root / "build/bin"
            target = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "gemma-4-12B-it-Q4_K_M.gguf"
            mmproj = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "mmproj-gemma-4-12B-it-Q8_0.gguf"
            draft = models_dir / "unsloth--gemma-4-12b-it-GGUF" / "MTP/gemma-4-12b-it-Q8_0-MTP.gguf"
            build_bin.mkdir(parents=True)
            (build_bin / "libllama.so").write_text("", encoding="utf-8")
            target.parent.mkdir(parents=True)
            draft.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")
            draft.write_text("draft", encoding="utf-8")

            args = build_parser().parse_args(["--llama-root", str(llama_root), "--model-id", "gemma4-12b-it-q4km", "--models-dir", str(models_dir), "--hf-cache", str(root / "hf")])
            paths = resolve_bootstrap_paths(args)

        self.assertEqual(paths.model, target)
        self.assertEqual(paths.mmproj_model, mmproj)
        self.assertEqual(paths.draft_mtp_model, draft)
        self.assertTrue(paths.multimodal_available)
        self.assertTrue(paths.mtp_available)

    def test_bootstrap_errors_when_target_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_root = root / "llama"
            build_bin = llama_root / "build/bin"
            build_bin.mkdir(parents=True)
            (build_bin / "libllama.so").write_text("", encoding="utf-8")

            args = build_parser().parse_args(["--llama-root", str(llama_root), "--model-id", "gemma4-12b-it-q4km", "--models-dir", str(root / "models"), "--hf-cache", str(root / "hf")])
            with self.assertRaises(FileNotFoundError):
                resolve_bootstrap_paths(args)


if __name__ == "__main__":
    unittest.main()
