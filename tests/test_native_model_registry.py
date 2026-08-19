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


def _registry_data() -> dict:
    return {
        "version": 1,
        "models": [
            {
                "id": "test-gemma",
                "display_name": "Test Gemma",
                "profile_id": "orbit-gemma4-native-v1",
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


def _write_registry(path: Path, data: dict | None = None) -> None:
    path.write_text(json.dumps(data or _registry_data()), encoding="utf-8")


class NativeModelRegistryTests(unittest.TestCase):
    def test_packaged_registry_contains_only_current_verified_models(self) -> None:
        manifests = load_registry()

        self.assertEqual(
            [(item.display_name, item.profile_id) for item in manifests],
            [
                ("Gemma 4 26B-A4B", "orbit-gemma4-native-v1"),
                ("Qwen 3.6 35B-A3B", "orbit-qwen36-native-v1"),
                ("Qwen 3.8 27B", "orbit-qwen38-native-v1"),
                ("Qwen3-Coder 30B-A3B", "orbit-qwen3-coder-native-v1"),
            ],
        )
        self.assertEqual(
            [(item.profile_id, item.target.repo, item.target.file) for item in manifests],
            [
                (
                    "orbit-gemma4-native-v1",
                    "ggml-org/gemma-4-26B-A4B-it-GGUF",
                    "gemma-4-26B-A4B-it-Q4_0.gguf",
                ),
                (
                    "orbit-qwen36-native-v1",
                    "ggml-org/Qwen3.6-35B-A3B-GGUF",
                    "Qwen3.6-35B-A3B-Q4_K_M.gguf",
                ),
                (
                    "orbit-qwen38-native-v1",
                    "unsloth/Qwen3.8-27B-GGUF",
                    "Qwen3.8-27B-Q4_K_M.gguf",
                ),
                (
                    "orbit-qwen3-coder-native-v1",
                    "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
                    "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
                ),
            ],
        )

    def test_loads_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            _write_registry(registry_path)

            manifests = load_registry(registry_path)

        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifests[0].id, "test-gemma")
        self.assertEqual(manifests[0].display_name, "Test Gemma")
        self.assertEqual(manifests[0].profile_id, "orbit-gemma4-native-v1")
        self.assertEqual(manifests[0].backend, "native-llama")
        self.assertEqual(manifests[0].architecture, "gemma4")
        self.assertIsNotNone(manifests[0].mtp)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            registry_path.write_text('{"version":1,"version":1,"models":[]}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate model registry key"):
                load_registry(registry_path)

    def test_unknown_registry_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            data = _registry_data()
            data["models"][0]["unexpected"] = True
            _write_registry(registry_path, data)

            with self.assertRaisesRegex(ValueError, "unknown keys"):
                load_registry(registry_path)

    def test_bad_version_missing_fields_and_wrong_types_are_rejected(self) -> None:
        cases = []
        bad_version = _registry_data()
        bad_version["version"] = True
        cases.append(bad_version)
        missing_field = _registry_data()
        del missing_field["models"][0]["target"]
        cases.append(missing_field)
        wrong_type = _registry_data()
        wrong_type["models"] = {}
        cases.append(wrong_type)
        for data in cases:
            with self.subTest(data=data), tempfile.TemporaryDirectory() as tmp:
                registry_path = Path(tmp) / "registry.json"
                _write_registry(registry_path, data)

                with self.assertRaises(ValueError):
                    load_registry(registry_path)

    def test_duplicate_ids_names_profiles_and_downloads_are_rejected(self) -> None:
        mutations = {
            "model id": {},
            "model display name": {"id": "other"},
            "model profile": {"id": "other", "display_name": "Other"},
            "download": {
                "id": "other",
                "display_name": "Other",
                "profile_id": "orbit-qwen36-native-v1",
                "architecture": "qwen35moe",
            },
        }
        for reason, changes in mutations.items():
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                registry_path = Path(tmp) / "registry.json"
                data = _registry_data()
                duplicate = json.loads(json.dumps(data["models"][0]))
                duplicate.update(changes)
                data["models"].append(duplicate)
                _write_registry(registry_path, data)

                with self.assertRaisesRegex(ValueError, f"duplicate (model registry )?{reason}"):
                    load_registry(registry_path)

    def test_incorrect_profile_architecture_mapping_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            data = _registry_data()
            data["models"][0]["architecture"] = "qwen35moe"
            _write_registry(registry_path, data)

            with self.assertRaisesRegex(ValueError, "invalid profile/architecture mapping"):
                load_registry(registry_path)

    def test_unsafe_download_paths_and_globs_are_rejected(self) -> None:
        for key, value in (("file", "../model.gguf"), ("cache_glob", "../../**/model.gguf")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                registry_path = Path(tmp) / "registry.json"
                data = _registry_data()
                data["models"][0]["target"][key] = value
                _write_registry(registry_path, data)

                with self.assertRaises(ValueError):
                    load_registry(registry_path)

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


class Qwen38RegistryTests(unittest.TestCase):
    def test_qwen38_entry_resolves_with_isolated_profile(self) -> None:
        manifest = get_manifest("qwen38-27b-q4-k-m")
        self.assertEqual(manifest.profile_id, "orbit-qwen38-native-v1")
        self.assertEqual(manifest.architecture, "qwen35")

    def test_qwen38_entry_declares_no_mtp_or_mmproj(self) -> None:
        manifest = get_manifest("qwen38-27b-q4-k-m")
        self.assertIsNone(getattr(manifest, "mtp", None))
        self.assertIsNone(getattr(manifest, "mmproj", None))

    def test_existing_model_ids_still_resolve(self) -> None:
        for model_id, profile_id in (
            ("qwen36-35b-a3b-q4-k-m", "orbit-qwen36-native-v1"),
            ("qwen3-coder-30b-a3b-instruct-q4-k-m", "orbit-qwen3-coder-native-v1"),
            ("gemma4-26b-a4b-it-q40", "orbit-gemma4-native-v1"),
        ):
            with self.subTest(model_id=model_id):
                self.assertEqual(get_manifest(model_id).profile_id, profile_id)
