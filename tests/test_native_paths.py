from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from orbit.native_llama.paths import DEFAULT_VENDOR_LIB_DIR, LEGACY_MODEL_ID, resolve_legacy_paths, resolve_paths


class NativePathsTests(unittest.TestCase):
    def test_resolves_vendored_native_runtime_without_llama_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor_lib = root / "vendor/lib"
            models_dir = root / "models"
            target = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "gemma-4-12B-it-Q4_K_M.gguf"
            mmproj = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "mmproj-gemma-4-12B-it-Q8_0.gguf"
            vendor_lib.mkdir(parents=True)
            (vendor_lib / "libllama.so").write_text("", encoding="utf-8")
            target.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")

            with mock.patch("orbit.native_llama.paths.DEFAULT_VENDOR_LIB_DIR", vendor_lib), mock.patch(
                "orbit.native_llama.paths.BUNDLED_SOURCE_ROOT", root / "missing-bundled-source"
            ):
                paths = resolve_paths(llama_root=None, models_dir=models_dir, hf_cache=root / "hf")

        self.assertIsNone(paths.llama_root)
        self.assertEqual(paths.build_bin, vendor_lib)
        self.assertEqual(paths.library, vendor_lib / "libllama.so")
        self.assertEqual(paths.model, target)

    def test_resolves_vendored_runtime_and_preserves_bundled_source_root_for_builds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor_lib = root / "vendor/lib"
            bundled = root / "vendor/source/llama.cpp"
            models_dir = root / "models"
            target = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "gemma-4-12B-it-Q4_K_M.gguf"
            mmproj = models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "mmproj-gemma-4-12B-it-Q8_0.gguf"
            vendor_lib.mkdir(parents=True)
            (vendor_lib / "libllama.so").write_text("", encoding="utf-8")
            (bundled / "CMakeLists.txt").parent.mkdir(parents=True, exist_ok=True)
            (bundled / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
            target.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")

            with mock.patch("orbit.native_llama.paths.DEFAULT_VENDOR_LIB_DIR", vendor_lib), mock.patch(
                "orbit.native_llama.paths.BUNDLED_SOURCE_ROOT", bundled
            ):
                paths = resolve_paths(llama_root=None, models_dir=models_dir, hf_cache=root / "hf")

        self.assertEqual(paths.llama_root, bundled)
        self.assertEqual(paths.build_bin, vendor_lib)

    def test_resolves_target_from_models_dir(self) -> None:
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

            paths = resolve_paths(llama_root=llama_root, models_dir=models_dir, hf_cache=root / "hf")

        self.assertEqual(paths.model, target)
        self.assertEqual(paths.mmproj_model, mmproj)
        self.assertIsNone(paths.draft_mtp_model)
        self.assertTrue(paths.multimodal_available)
        self.assertFalse(paths.mtp_available)
        self.assertEqual(paths.fallback_reason, "draft-mtp-missing")

    def test_resolves_target_from_hf_cache_when_models_dir_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_root = root / "llama"
            hf_cache = root / "hf"
            build_bin = llama_root / "build/bin"
            target = hf_cache / "models--ggml-org--gemma-4-12B-it-GGUF/snapshots/abc/gemma-4-12B-it-Q4_K_M.gguf"
            build_bin.mkdir(parents=True)
            (build_bin / "libllama.so").write_text("", encoding="utf-8")
            target.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")

            paths = resolve_paths(llama_root=llama_root, models_dir=root / "models", hf_cache=hf_cache)

        self.assertEqual(paths.model, target)

    def test_raises_when_target_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_root = root / "llama"
            build_bin = llama_root / "build/bin"
            build_bin.mkdir(parents=True)
            (build_bin / "libllama.so").write_text("", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                resolve_paths(llama_root=llama_root, models_dir=root / "models", hf_cache=root / "hf")

    def test_raises_when_no_native_runtime_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_vendor = root / "vendor/lib"
            with mock.patch("orbit.native_llama.paths.DEFAULT_VENDOR_LIB_DIR", missing_vendor):
                with self.assertRaises(FileNotFoundError) as ctx:
                    resolve_paths(llama_root=None, models_dir=root / "models", hf_cache=root / "hf")

        self.assertIn("libllama.so not found", str(ctx.exception))
        self.assertIn("vendor/lib/libllama.so", str(ctx.exception))

    def test_exposes_draft_and_mtp_available_when_both_are_present(self) -> None:
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

            paths = resolve_paths(llama_root=llama_root, models_dir=models_dir, hf_cache=root / "hf")

        self.assertEqual(paths.model, target)
        self.assertEqual(paths.mmproj_model, mmproj)
        self.assertEqual(paths.draft_mtp_model, draft)
        self.assertTrue(paths.multimodal_available)
        self.assertTrue(paths.mtp_available)
        self.assertIsNone(paths.fallback_reason)

    def test_legacy_path_mode_keeps_direct_model_without_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_root = root / "llama"
            build_bin = llama_root / "build/bin"
            target = root / "manual.gguf"
            mmproj = root / "mmproj.gguf"
            build_bin.mkdir(parents=True)
            (build_bin / "libllama.so").write_text("", encoding="utf-8")
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")

            paths = resolve_legacy_paths(llama_root=llama_root, model=target, mmproj=mmproj)

        self.assertEqual(paths.model, target.resolve())
        self.assertEqual(paths.mmproj_model, mmproj.resolve())
        self.assertEqual(paths.model_id, LEGACY_MODEL_ID)
        self.assertTrue(paths.multimodal_available)
        self.assertFalse(paths.mtp_available)
        self.assertEqual(paths.fallback_reason, "legacy-model-path")


if __name__ == "__main__":
    unittest.main()
