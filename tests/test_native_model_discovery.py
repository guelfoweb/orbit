from __future__ import annotations

from dataclasses import replace
import io
import os
import re
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from orbit.native_llama.model_discovery import (
    ModelDiscoveryResult,
    ModelDiscoveryRow,
    NativeProfileInspector,
    discover_models,
    format_model_discovery,
    paint_model_status,
)
from orbit.terminal.theme import GREEN, RED, RESET, supports_ansi
from orbit.native_llama.model_profiles import (
    GEMMA4_PROFILE_ID,
    QWEN36_PROFILE_ID,
    QWEN3_CODER_PROFILE_ID,
    detect_native_model_profile,
)
from orbit.native_llama.model_registry import load_registry, local_model_path


def _profile(profile_id: str, model_name: str):
    return SimpleNamespace(profile_id=profile_id, model_name=model_name, verified=True)


class NativeModelDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifests = tuple(load_registry())

    def test_no_local_models_lists_all_supported_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = discover_models(
                models_dir=root / "models",
                hf_cache=root / "hf",
                inspector=lambda _path: self.fail("no file should be inspected"),
            )

        self.assertEqual(len(result.rows), 5)
        self.assertTrue(all(row.local == "MISSING" for row in result.rows))
        self.assertTrue(all(row.support == "VERIFIED" for row in result.rows))
        self.assertEqual(
            {row.path_or_action for row in result.rows},
            {
                "orbit download ggml-org/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_0.gguf",
                "orbit download ggml-org/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf",
                "orbit download ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf",
                "orbit download unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf",
                "orbit download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
            },
        )

    def test_one_verified_local_model_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifests[1]
            model = local_model_path(manifest.target, models_dir=root / "models")
            model.parent.mkdir(parents=True)
            model.write_bytes(b"GGUF")
            result = discover_models(
                models_dir=root / "models",
                hf_cache=root / "hf",
                inspector=lambda _path: _profile(QWEN36_PROFILE_ID, "Qwen3.6-35B-A3B"),
            )

        row = next(row for row in result.rows if row.model == "Qwen 3.6 35B-A3B")
        self.assertEqual((row.local, row.support), ("AVAILABLE", "VERIFIED"))
        self.assertEqual(row.path_or_action, str(model.absolute()))
        self.assertEqual(row.model_id, manifest.id)
        self.assertFalse(row.low_memory_supported)
        self.assertEqual(result.metadata_inspections, 1)

    def test_multiple_verified_models_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles = {
                self.manifests[0].target.file: _profile(GEMMA4_PROFILE_ID, "gemma-4-26B-A4B-it"),
                self.manifests[2].target.file: _profile(
                    QWEN3_CODER_PROFILE_ID,
                    "Qwen3-Coder-30B-A3B-Instruct",
                ),
            }
            for manifest in (self.manifests[0], self.manifests[2]):
                path = local_model_path(manifest.target, models_dir=root / "models")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"GGUF")

            result = discover_models(
                models_dir=root / "models",
                hf_cache=root / "hf",
                inspector=lambda path: profiles[path.name],
            )

        available = [row.model for row in result.rows if row.local == "AVAILABLE"]
        self.assertEqual(available, ["Gemma 4 26B-A4B", "Qwen3-Coder 30B-A3B"])
        by_model = {row.model: row for row in result.rows if row.local == "AVAILABLE"}
        self.assertFalse(by_model["Gemma 4 26B-A4B"].low_memory_supported)
        self.assertTrue(by_model["Qwen3-Coder 30B-A3B"].low_memory_supported)

    def test_unsupported_gguf_is_never_presented_as_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models/other/foo.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"GGUF")
            unsupported = detect_native_model_profile(
                {"general.architecture": "unknown", "general.name": "foo"},
                "",
            )
            result = discover_models(
                models_dir=root / "models",
                hf_cache=root / "hf",
                inspector=lambda _path: unsupported,
            )

        row = next(row for row in result.rows if row.model == "foo.gguf")
        self.assertEqual((row.local, row.support), ("AVAILABLE", "UNSUPPORTED"))

    def test_filename_spoof_with_wrong_metadata_remains_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifests[2]
            model = local_model_path(manifest.target, models_dir=root / "models")
            model.parent.mkdir(parents=True)
            model.write_bytes(b"not the expected model")
            wrong = detect_native_model_profile(
                {
                    "general.architecture": "qwen3moe",
                    "general.name": "different-model",
                    "tokenizer.ggml.model": "gpt2",
                    "tokenizer.ggml.pre": "qwen2",
                    "general.file_type": "15",
                },
                "",
            )
            result = discover_models(
                models_dir=root / "models",
                hf_cache=root / "hf",
                inspector=lambda _path: wrong,
            )

        supported = next(row for row in result.rows if row.model == "Qwen3-Coder 30B-A3B")
        spoof = next(row for row in result.rows if row.model == manifest.target.file)
        self.assertEqual((supported.local, supported.support), ("MISSING", "VERIFIED"))
        self.assertEqual((spoof.local, spoof.support), ("AVAILABLE", "UNSUPPORTED"))
        self.assertFalse(supported.low_memory_supported)
        self.assertFalse(spoof.low_memory_supported)

    def test_verified_profile_with_different_model_identity_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models/other/gemma-4-12b.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"GGUF")
            result = discover_models(
                models_dir=root / "models",
                hf_cache=root / "hf",
                inspector=lambda _path: _profile(GEMMA4_PROFILE_ID, "gemma-4-12b-it"),
            )

        row = next(row for row in result.rows if row.model == "gemma-4-12b.gguf")
        self.assertEqual(row.support, "UNSUPPORTED")

    def test_missing_registry_entry_does_not_weaken_detected_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models/local/qwen.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"GGUF")
            manifests = tuple(item for item in self.manifests if item.profile_id != QWEN36_PROFILE_ID)
            result = discover_models(
                models_dir=root / "models",
                hf_cache=root / "hf",
                manifests=manifests,
                inspector=lambda _path: _profile(QWEN36_PROFILE_ID, "Qwen3.6-35B-A3B"),
            )

        row = next(row for row in result.rows if row.path_or_action == str(model.resolve()))
        self.assertEqual((row.local, row.support), ("AVAILABLE", "VERIFIED"))

    def test_incorrect_injected_registry_mapping_fails_closed(self) -> None:
        bad_manifest = replace(self.manifests[0], architecture="qwen35moe")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "profile mapping is not supported"):
                discover_models(
                    models_dir=root / "models",
                    hf_cache=root / "hf",
                    manifests=(bad_manifest,),
                    inspector=lambda _path: self.fail("no model"),
                )

    def test_malformed_or_unreadable_gguf_is_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models/broken.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"broken")

            def fail(_path: Path):
                raise RuntimeError("vocab-only GGUF inspection failed")

            result = discover_models(
                models_dir=root / "models",
                hf_cache=root / "hf",
                inspector=fail,
            )

        row = next(row for row in result.rows if row.model == "broken.gguf")
        self.assertEqual((row.local, row.support), ("AVAILABLE", "UNVERIFIED"))

    def test_hf_cache_uses_only_exact_registered_globs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifests[1]
            model = root / "hf" / manifest.target.cache_glob.replace("*", "snapshot-a")
            model.parent.mkdir(parents=True)
            model.write_bytes(b"GGUF")
            unrelated = root / "hf/models--other--repo/snapshots/a/other.gguf"
            unrelated.parent.mkdir(parents=True)
            unrelated.write_bytes(b"GGUF")
            result = discover_models(
                models_dir=root / "models",
                hf_cache=root / "hf",
                inspector=lambda _path: _profile(QWEN36_PROFILE_ID, "Qwen3.6-35B-A3B"),
            )

        self.assertEqual(result.metadata_inspections, 1)
        self.assertFalse(any(row.model == "other.gguf" for row in result.rows))
        self.assertEqual(result.filesystem_scans, 7)

    def test_symlink_escape_is_not_inspected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside.gguf"
            outside.write_bytes(b"GGUF")
            models = root / "models"
            models.mkdir()
            (models / "escape.gguf").symlink_to(outside)
            result = discover_models(
                models_dir=models,
                hf_cache=root / "hf",
                inspector=lambda _path: self.fail("escaping symlink must not be inspected"),
            )

        self.assertFalse(any(row.model == "escape.gguf" for row in result.rows))
        self.assertEqual(result.metadata_inspections, 0)

    def test_internal_symlink_alias_is_inspected_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            model = models / "model.gguf"
            models.mkdir()
            model.write_bytes(b"GGUF")
            (models / "alias.gguf").symlink_to(model)
            inspected: list[Path] = []
            unsupported = detect_native_model_profile({"general.architecture": "unknown"}, "")
            result = discover_models(
                models_dir=models,
                hf_cache=root / "hf",
                inspector=lambda path: inspected.append(path) or unsupported,
            )

        self.assertEqual(inspected, [model.resolve()])
        self.assertEqual(result.metadata_inspections, 1)

    def test_hard_link_alias_is_inspected_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            model = models / "model.gguf"
            models.mkdir()
            model.write_bytes(b"GGUF")
            os.link(model, models / "alias.gguf")
            inspected: list[Path] = []
            unsupported = detect_native_model_profile({"general.architecture": "unknown"}, "")
            result = discover_models(
                models_dir=models,
                hf_cache=root / "hf",
                inspector=lambda path: inspected.append(path) or unsupported,
            )

        self.assertEqual(len(inspected), 1)
        self.assertEqual(result.metadata_inspections, 1)

    def test_output_is_compact_and_includes_path_or_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = discover_models(
                models_dir=root / "models",
                hf_cache=root / "hf",
                inspector=lambda _path: self.fail("no file should be inspected"),
            )

        output = format_model_discovery(result)
        self.assertIn("Model", output)
        self.assertIn("Local", output)
        self.assertIn("Support", output)
        self.assertIn("Path / action", output)
        self.assertIn("MISSING", output)
        self.assertNotIn("wall_ms", output)

    def test_native_inspection_is_vocab_only_and_releases_model(self) -> None:
        params = SimpleNamespace(vocab_only=False, use_mmap=False, check_tensors=False)
        native_model = object()
        lib = mock.Mock()
        lib.llama_model_default_params.return_value = params
        lib.llama_model_load_from_file.return_value = native_model
        lib.llama_model_meta_count.return_value = 0
        lib.llama_model_chat_template.return_value = None
        binding = SimpleNamespace(lib=lib)

        with mock.patch("orbit.native_llama.model_discovery.LlamaLibrary", return_value=binding):
            inspector = NativeProfileInspector(Path("/native"))
            profile = inspector(Path("/models/model.gguf"))
            inspector.close()

        self.assertTrue(params.vocab_only)
        self.assertTrue(params.use_mmap)
        self.assertTrue(params.check_tensors)
        self.assertFalse(profile.verified)
        lib.ggml_backend_load_all.assert_called_once_with()
        lib.llama_model_free.assert_called_once_with(native_model)
        self.assertEqual(lib.llama_log_set.call_count, 2)

    def test_discovery_restores_native_logging_after_owned_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models/model.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"GGUF")
            inspector = mock.Mock(
                return_value=_profile(QWEN36_PROFILE_ID, "Qwen3.6-35B-A3B")
            )
            with mock.patch(
                "orbit.native_llama.model_discovery.NativeProfileInspector",
                return_value=inspector,
            ):
                discover_models(
                    models_dir=root / "models",
                    hf_cache=root / "hf",
                    build_bin=Path("/native"),
                )

        inspector.assert_called_once_with(model.resolve())
        inspector.close.assert_called_once_with()


class ModelStatusColorTests(unittest.TestCase):
    """SERVER-MODEL-STATUS-COLORS-1: colour only AVAILABLE (green) / MISSING
    (red), width-safe, and only when colour is enabled."""

    _ANSI = re.compile(r"\x1b\[[0-9;]*m")

    def _result(self) -> ModelDiscoveryResult:
        return ModelDiscoveryResult(
            rows=(
                ModelDiscoveryRow("Ornith 1.5 35B-A3B", "AVAILABLE", "VERIFIED", "/models/ornith.gguf"),
                ModelDiscoveryRow("Gemma 4 26B-A4B", "MISSING", "VERIFIED", "orbit download x/y.gguf"),
                ModelDiscoveryRow("mystery.gguf", "AVAILABLE", "UNVERIFIED", "/models/mystery.gguf"),
            ),
            wall_ms=0.0,
            filesystem_scans=0,
            metadata_inspections=0,
        )

    def test_available_is_green_and_missing_is_red(self) -> None:
        out = format_model_discovery(self._result(), color=True)
        self.assertIn(f"{GREEN}AVAILABLE{RESET}", out)
        self.assertIn(f"{RED}MISSING{RESET}", out)

    def test_plain_default_has_no_ansi_and_equals_explicit_plain(self) -> None:
        default = format_model_discovery(self._result())
        self.assertEqual(default, format_model_discovery(self._result(), color=False))
        self.assertNotIn("\x1b[", default)

    def test_colour_preserves_alignment_and_plain_text(self) -> None:
        colored = format_model_discovery(self._result(), color=True)
        plain = format_model_discovery(self._result(), color=False)
        # stripping the escapes must reproduce the exact plain layout byte-for-byte
        self.assertEqual(self._ANSI.sub("", colored), plain)

    def test_only_the_status_token_is_coloured(self) -> None:
        out = format_model_discovery(self._result(), color=True)
        self.assertNotIn(f"{GREEN}VERIFIED", out)
        self.assertNotIn(f"{RED}VERIFIED", out)
        self.assertNotIn(f"{GREEN}Ornith", out)
        coloured = re.findall(r"\x1b\[3\dm(.*?)\x1b\[0m", out)
        self.assertEqual(set(coloured), {"AVAILABLE", "MISSING"})

    def test_paint_model_status_token(self) -> None:
        self.assertEqual(paint_model_status("AVAILABLE", color=True), f"{GREEN}AVAILABLE{RESET}")
        self.assertEqual(paint_model_status("MISSING", color=True), f"{RED}MISSING{RESET}")
        self.assertEqual(paint_model_status("AVAILABLE", color=False), "AVAILABLE")
        self.assertEqual(paint_model_status("MISSING", color=False), "MISSING")
        # unknown / non-availability status is never coloured
        self.assertEqual(paint_model_status("UNVERIFIED", color=True), "UNVERIFIED")
        self.assertEqual(paint_model_status("VERIFIED", color=True), "VERIFIED")

    def test_non_tty_stream_forces_plain(self) -> None:
        color = supports_ansi(io.StringIO())  # a StringIO is not a TTY
        self.assertFalse(color)
        self.assertNotIn("\x1b[", format_model_discovery(self._result(), color=color))

    def test_numbered_list_surface_colours_via_shared_helper(self) -> None:
        # the interactive `Verified models:` list renders its bracketed status
        # through the same paint_model_status seam, gated on the stderr stream.
        src = (Path(__file__).resolve().parents[1] / "src/orbit/native_server/app.py").read_text()
        self.assertIn("paint_model_status(row.local, color=color)", src)
        self.assertIn("color = supports_ansi(sys.stderr)", src)
        self.assertIn("format_model_discovery(result, color=supports_ansi(sys.stderr))", src)


if __name__ == "__main__":
    unittest.main()
