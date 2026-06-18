from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from orbit.native_llama.model_download import (
    download_all_for_repo,
    DownloadRequest,
    download_model,
    huggingface_resolve_url,
    parse_huggingface_spec,
)
from orbit.native_llama.model_registry import (
    default_models_dir,
    find_project_root,
)


class NativeModelDownloadTests(unittest.TestCase):
    def test_project_root_models_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src/orbit").mkdir(parents=True)
            (root / "pyproject.toml").write_text("[project]\nname='orbit'\n", encoding="utf-8")

            self.assertEqual(find_project_root(root / "src"), root)
            self.assertEqual(default_models_dir(root / "src"), root / "models")

    def test_user_cache_models_dir_when_no_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(default_models_dir(Path(tmp)).name, "models")
            self.assertIn(".cache/orbit", str(default_models_dir(Path(tmp))))

    def test_parse_repo_uses_manifest_default_file(self) -> None:
        request = parse_huggingface_spec("ggml-org/gemma-4-12B-it-GGUF")

        self.assertEqual(request.repo, "ggml-org/gemma-4-12B-it-GGUF")
        self.assertEqual(request.file, "gemma-4-12B-it-Q4_K_M.gguf")

    def test_parse_repo_with_mmproj_preference_uses_manifest_projector(self) -> None:
        request = parse_huggingface_spec("ggml-org/gemma-4-12B-it-GGUF", prefer="mmproj")

        self.assertEqual(request.repo, "ggml-org/gemma-4-12B-it-GGUF")
        self.assertEqual(request.file, "mmproj-gemma-4-12B-it-Q8_0.gguf")

    def test_parse_explicit_gguf_file(self) -> None:
        request = parse_huggingface_spec("unsloth/gemma-4-12b-it-GGUF/MTP/gemma-4-12b-it-Q8_0-MTP.gguf")

        self.assertEqual(request.repo, "unsloth/gemma-4-12b-it-GGUF")
        self.assertEqual(request.file, "MTP/gemma-4-12b-it-Q8_0-MTP.gguf")

    def test_huggingface_url(self) -> None:
        url = huggingface_resolve_url(DownloadRequest(repo="owner/repo", file="dir/model.gguf"))

        self.assertEqual(url, "https://huggingface.co/owner/repo/resolve/main/dir/model.gguf")

    def test_existing_file_is_not_downloaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            path = models_dir / "owner--repo" / "model.gguf"
            path.parent.mkdir(parents=True)
            path.write_text("already here", encoding="utf-8")
            calls: list[str] = []

            result = download_model(
                "owner/repo/model.gguf",
                models_dir=models_dir,
                retrieve=lambda url, dest: calls.append(url),
            )

        self.assertEqual(result.path, path)
        self.assertFalse(result.downloaded)
        self.assertEqual(calls, [])

    def test_download_writes_to_models_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"

            def retrieve(url: str, dest: str) -> None:
                Path(dest).write_text(f"downloaded from {url}", encoding="utf-8")

            result = download_model("owner/repo/path/model.gguf", models_dir=models_dir, retrieve=retrieve)

            self.assertTrue(result.downloaded)
            self.assertEqual(result.path, models_dir / "owner--repo" / "path/model.gguf")
            self.assertIn("downloaded from https://huggingface.co/owner/repo/resolve/main/path/model.gguf", result.path.read_text(encoding="utf-8"))

    def test_download_mmproj_from_repo_uses_projector_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"

            def retrieve(url: str, dest: str) -> None:
                Path(dest).write_text(f"downloaded from {url}", encoding="utf-8")

            result = download_model(
                "ggml-org/gemma-4-12B-it-GGUF",
                models_dir=models_dir,
                prefer="mmproj",
                retrieve=retrieve,
            )

            self.assertTrue(result.downloaded)
            self.assertEqual(
                result.path,
                models_dir / "ggml-org--gemma-4-12B-it-GGUF" / "mmproj-gemma-4-12B-it-Q8_0.gguf",
            )

    def test_download_all_for_repo_downloads_target_mmproj_and_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            seen: list[str] = []

            def retrieve(url: str, dest: str) -> None:
                seen.append(url)
                Path(dest).write_text("ok", encoding="utf-8")

            batch = download_all_for_repo(
                "ggml-org/gemma-4-12B-it-GGUF",
                models_dir=models_dir,
                retrieve=retrieve,
            )

        self.assertEqual(len(batch.results), 3)
        self.assertIn("https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12B-it-Q4_K_M.gguf", seen)
        self.assertIn("https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/resolve/main/mmproj-gemma-4-12B-it-Q8_0.gguf", seen)
        self.assertIn("https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/MTP/gemma-4-12b-it-Q8_0-MTP.gguf", seen)


if __name__ == "__main__":
    unittest.main()
