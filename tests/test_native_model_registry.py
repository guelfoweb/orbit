from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from orbit.native_llama.model_registry import (
    default_models_dir,
    get_manifest,
    local_model_path,
    load_registry,
    resolve_model,
)


def _write_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "id": "test-gemma",
                        "backend": "native-llama",
                        "architecture": "gemma4",
                        "target": {
                            "repo": "target/repo",
                            "file": "target.gguf",
                            "cache_glob": "target/snapshots/*/target.gguf",
                        },
                        "mmproj": {
                            "repo": "target/repo",
                            "file": "mmproj.gguf",
                            "cache_glob": "target/snapshots/*/mmproj.gguf",
                        },
                        "mtp": {
                            "enabled_by_default": True,
                            "required": False,
                            "spec_type": "draft-mtp",
                            "repo": "draft/repo",
                            "file": "MTP/draft.gguf",
                            "cache_glob": "draft/snapshots/*/MTP/draft.gguf",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class NativeModelRegistryTests(unittest.TestCase):
    def test_loads_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            _write_registry(registry_path)

            manifests = load_registry(registry_path)

        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifests[0].id, "test-gemma")
        self.assertEqual(manifests[0].backend, "native-llama")
        self.assertEqual(manifests[0].architecture, "gemma4")
        self.assertIsNotNone(manifests[0].mtp)

    def test_falls_back_to_no_mtp_when_optional_draft_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            _write_registry(registry_path)
            models_dir = root / "models"
            manifest = get_manifest("test-gemma", registry_path=registry_path)
            target = local_model_path(manifest.target, models_dir=models_dir)
            mmproj = local_model_path(manifest.mmproj, models_dir=models_dir)
            target.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")
            resolved = resolve_model(manifest, models_dir=models_dir, hf_cache=root)

        self.assertEqual(resolved.target_path, target)
        self.assertEqual(resolved.mmproj_path, mmproj)
        self.assertTrue(resolved.multimodal_available)
        self.assertIsNone(resolved.draft_mtp_path)
        self.assertFalse(resolved.mtp_available)
        self.assertEqual(resolved.fallback_reason, "draft-mtp-missing")

    def test_mtp_available_when_target_and_draft_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            _write_registry(registry_path)
            models_dir = root / "models"
            manifest = get_manifest("test-gemma", registry_path=registry_path)
            target = local_model_path(manifest.target, models_dir=models_dir)
            mmproj = local_model_path(manifest.mmproj, models_dir=models_dir)
            draft = local_model_path(manifest.mtp, models_dir=models_dir)
            target.parent.mkdir(parents=True)
            draft.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")
            draft.write_text("draft", encoding="utf-8")
            resolved = resolve_model(manifest, models_dir=models_dir, hf_cache=root)

        self.assertEqual(resolved.target_path, target)
        self.assertEqual(resolved.mmproj_path, mmproj)
        self.assertEqual(resolved.draft_mtp_path, draft)
        self.assertTrue(resolved.multimodal_available)
        self.assertTrue(resolved.mtp_available)
        self.assertIsNone(resolved.fallback_reason)

    def test_default_models_dir_uses_project_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src/orbit").mkdir(parents=True)
            (root / "pyproject.toml").write_text("[project]\nname='orbit'\n", encoding="utf-8")

            self.assertEqual(default_models_dir(root / "src"), root / "models")

    def test_missing_target_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            _write_registry(registry_path)
            manifest = get_manifest("test-gemma", registry_path=registry_path)

            with self.assertRaises(FileNotFoundError):
                resolve_model(manifest, hf_cache=root)


if __name__ == "__main__":
    unittest.main()
