from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from orbit.native_llama import bindings
from orbit.native_llama.bindings import MtmdLibrary, verify_core_abi_layouts
from orbit.native_llama.build_support import BUNDLED_SOURCE_ROOT, DEFAULT_VENDOR_BUILD_BIN
from orbit.native_llama.build_cli import _cmake_provenance_args
from orbit.native_llama.client import _resolve_mtmd_bridge_path
from orbit.native_llama.llama_provenance import LlamaProvenance, load_llama_provenance, source_tree_sha256
from orbit.native_llama.mtmd_bridge import _build_identity, validate_mtmd_bridge_artifact
from orbit.native_llama.native_names import platform_runtime_libs, runtime_library_filename


EXPECTED_COMMIT = "379ac6673b5cd75c7b4e07d1521c50f1e093878c"
EXPECTED_TAG = "b9551"


class MtmdBridgeAbiTests(unittest.TestCase):
    def test_python_does_not_expose_mtmd_struct_layouts(self) -> None:
        self.assertFalse(hasattr(bindings, "MtmdContextParams"))
        self.assertFalse(hasattr(bindings, "MtmdInputText"))
        self.assertFalse(hasattr(bindings, "MtmdCaps"))

    def test_vendored_provenance_matches_current_source_tree(self) -> None:
        provenance = load_llama_provenance(BUNDLED_SOURCE_ROOT)

        self.assertEqual(provenance.upstream_commit, EXPECTED_COMMIT)
        self.assertEqual(provenance.upstream_tag, EXPECTED_TAG)
        self.assertEqual(provenance.source_tree_sha256, source_tree_sha256(BUNDLED_SOURCE_ROOT))
        self.assertEqual(len(provenance.patchset_sha256), 64)
        self.assertEqual(len(provenance.patched_paths), 60)

    def test_cmake_build_metadata_uses_vendor_not_parent_git(self) -> None:
        arguments = _cmake_provenance_args(BUNDLED_SOURCE_ROOT)

        self.assertIn(f"-DLLAMA_BUILD_COMMIT={EXPECTED_COMMIT}", arguments)
        self.assertIn("-DLLAMA_BUILD_NUMBER=9551", arguments)

    def test_bridge_must_be_co_located_with_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_bin = Path(tmp)
            paths = SimpleNamespace(build_bin=build_bin)
            self.assertIsNone(_resolve_mtmd_bridge_path(paths))

            bridge = build_bin / runtime_library_filename("orbit-mtmd-bridge")
            bridge.write_bytes(b"bridge")
            self.assertEqual(_resolve_mtmd_bridge_path(paths), bridge)

    def test_identity_rejects_missing_sidecar_before_native_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_bin = Path(tmp)
            bridge = build_bin / runtime_library_filename("orbit-mtmd-bridge")
            bridge.write_bytes(b"not-a-library")

            with self.assertRaisesRegex(RuntimeError, "bridge identity"):
                MtmdLibrary(build_bin, bridge)

    def test_identity_rejects_mismatched_runtime_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_bin = Path(tmp)
            bridge = build_bin / runtime_library_filename("orbit-mtmd-bridge")
            bridge.write_bytes(b"bridge")
            libraries = {}
            for name in (*platform_runtime_libs(), runtime_library_filename("mtmd")):
                path = build_bin / name
                path.write_bytes(name.encode())
                libraries[name] = _sha256(path)
            libraries[runtime_library_filename("mtmd")] = "0" * 64
            identity = {
                "build": {"libraries": libraries},
                "artifact_sha256": _sha256(bridge),
            }
            bridge.with_name(f"{bridge.name}.identity.json").write_text(
                json.dumps(identity), encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "libmtmd"):
                validate_mtmd_bridge_artifact(build_bin, bridge)

    def test_build_identity_changes_with_every_revision_bound_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "llama"
            build_bin = Path(tmp) / "build" / "bin"
            build_bin.mkdir(parents=True)
            source = Path(tmp) / "bridge.cpp"
            inputs = (
                source,
                root / "include/llama.h",
                root / "tools/mtmd/mtmd.h",
                root / "tools/mtmd/mtmd-helper.h",
            )
            for path in inputs:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(path.name, encoding="utf-8")
            for name in (*platform_runtime_libs(), runtime_library_filename("mtmd")):
                (build_bin / name).write_text(name, encoding="utf-8")
            (build_bin.parent / "CMakeCache.txt").write_text(
                "\n".join(
                    (
                        "CMAKE_BUILD_TYPE:STRING=Release",
                        "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++",
                        "CMAKE_CXX_FLAGS:STRING=",
                        "CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG",
                        "GGML_NATIVE:BOOL=ON",
                        "GGML_OPENMP:BOOL=ON",
                    )
                ),
                encoding="utf-8",
            )
            provenance = LlamaProvenance("a" * 40, "b1", "b" * 64, "c" * 64, ("src/a.cpp",))
            with mock.patch("orbit.native_llama.mtmd_bridge.subprocess.run") as run:
                run.return_value = SimpleNamespace(stdout="c++ test\n", stderr="")
                baseline = _build_identity(
                    source=source,
                    llama_root=root,
                    build_bin=build_bin,
                    provenance=provenance,
                    defines=("-DTEST=1",),
                )
                (root / "include/llama.h").write_text("changed", encoding="utf-8")
                changed_header = _build_identity(
                    source=source,
                    llama_root=root,
                    build_bin=build_bin,
                    provenance=provenance,
                    defines=("-DTEST=1",),
                )
                (build_bin / runtime_library_filename("mtmd")).write_text("changed", encoding="utf-8")
                changed_library = _build_identity(
                    source=source,
                    llama_root=root,
                    build_bin=build_bin,
                    provenance=provenance,
                    defines=("-DTEST=1",),
                )
                changed_provenance = _build_identity(
                    source=source,
                    llama_root=root,
                    build_bin=build_bin,
                    provenance=LlamaProvenance("d" * 40, "b2", "e" * 64, "f" * 64),
                    defines=("-DTEST=1",),
                )
                source.write_text("changed bridge", encoding="utf-8")
                changed_source = _build_identity(
                    source=source,
                    llama_root=root,
                    build_bin=build_bin,
                    provenance=provenance,
                    defines=("-DTEST=1",),
                )
                cache = build_bin.parent / "CMakeCache.txt"
                cache.write_text(
                    cache.read_text(encoding="utf-8").replace("GGML_NATIVE:BOOL=ON", "GGML_NATIVE:BOOL=OFF"),
                    encoding="utf-8",
                )
                changed_cmake = _build_identity(
                    source=source,
                    llama_root=root,
                    build_bin=build_bin,
                    provenance=provenance,
                    defines=("-DTEST=1",),
                )
                changed_defines = _build_identity(
                    source=source,
                    llama_root=root,
                    build_bin=build_bin,
                    provenance=provenance,
                    defines=("-DTEST=2",),
                )
            with (
                mock.patch.dict("os.environ", {"CXX": "/opt/orbit-cxx"}),
                mock.patch("orbit.native_llama.mtmd_bridge.subprocess.run") as run,
            ):
                run.return_value = SimpleNamespace(stdout="alternate c++\n", stderr="")
                changed_compiler = _build_identity(
                    source=source,
                    llama_root=root,
                    build_bin=build_bin,
                    provenance=provenance,
                    defines=("-DTEST=1",),
                )

        self.assertNotEqual(baseline["source_inputs"], changed_header["source_inputs"])
        self.assertNotEqual(changed_header["libraries"], changed_library["libraries"])
        self.assertNotEqual(changed_library["upstream_commit"], changed_provenance["upstream_commit"])
        self.assertNotEqual(baseline["source_inputs"], changed_source["source_inputs"])
        self.assertNotEqual(changed_source["native_build"], changed_cmake["native_build"])
        self.assertNotEqual(changed_cmake["defines"], changed_defines["defines"])
        self.assertNotEqual(baseline["compiler"], changed_compiler["compiler"])
        self.assertNotEqual(baseline["compiler_version"], changed_compiler["compiler_version"])

    def test_core_layout_gate_rejects_artificial_mismatch(self) -> None:
        manifest = _python_core_manifest()
        verify_core_abi_layouts(manifest)
        manifest["llama_context_params"]["size"] += 8

        with self.assertRaisesRegex(RuntimeError, "llama_context_params"):
            verify_core_abi_layouts(manifest)

    @unittest.skipUnless(
        (DEFAULT_VENDOR_BUILD_BIN / runtime_library_filename("orbit-mtmd-bridge")).exists(),
        "native mtmd bridge is not built",
    )
    def test_current_native_bridge_reports_supported_v1_abi(self) -> None:
        library = MtmdLibrary(DEFAULT_VENDOR_BUILD_BIN)

        self.assertEqual(library.manifest["abi_profile"], "mtmd-context-v1")
        self.assertEqual(library.manifest["mtmd_context_params"]["size"], 56)
        self.assertEqual(library.manifest["mtmd_input_text"]["profile"], "mtmd-input-text-v1")
        self.assertEqual(library.manifest["mtmd_input_text"]["size"], 16)
        self.assertEqual(library.manifest["bitmap_result"], "pointer-v1")
        self.assertEqual(library.manifest["upstream_commit"], EXPECTED_COMMIT)


def _python_core_manifest() -> dict[str, dict[str, int]]:
    structures = {
        "llama_batch": (bindings.LlamaBatch, tuple(name for name, _ in bindings.LlamaBatch._fields_)),
        "llama_model_params": (
            bindings.LlamaModelParams,
            (
                "n_gpu_layers",
                "progress_callback",
                "progress_callback_user_data",
                "kv_overrides",
                "vocab_only",
                "no_alloc",
            ),
        ),
        "llama_context_params": (
            bindings.LlamaContextParams,
            (
                "n_ctx",
                "n_outputs_max",
                "n_threads",
                "flash_attn_type",
                "defrag_thold",
                "cb_eval",
                "type_k",
                "abort_callback",
                "embeddings",
                "samplers",
                "ctx_other",
            ),
        ),
        "llama_sampler_chain_params": (bindings.LlamaSamplerChainParams, ("no_perf",)),
        "llama_chat_message": (bindings.LlamaChatMessage, ("role", "content")),
    }
    return {
        name: {
            "size": ctypes.sizeof(structure),
            "align": ctypes.alignment(structure),
            **{field: getattr(structure, field).offset for field in fields},
        }
        for name, (structure, fields) in structures.items()
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
