from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from .build_support import compile_cpp_helper
from .llama_provenance import LlamaProvenance, load_llama_provenance
from .native_names import platform_optional_runtime_libs, platform_runtime_libs, runtime_library_filename


BRIDGE_BUILD_FLAGS = "-std=c++17;-shared;-fPIC;-fvisibility=hidden"


def build_mtmd_bridge(
    *,
    llama_root: Path,
    build_dir: Path,
    build_bin: Path,
    runner=subprocess.run,
) -> tuple[Path, LlamaProvenance]:
    provenance = load_llama_provenance(llama_root)
    source = Path(__file__).parent / "vendor" / "shim" / "orbit_mtmd_bridge.cpp"
    output = build_dir / runtime_library_filename("orbit-mtmd-bridge")
    defines = (
        _define("ORBIT_LLAMA_UPSTREAM_COMMIT", provenance.upstream_commit),
        _define("ORBIT_LLAMA_UPSTREAM_TAG", provenance.upstream_tag),
        _define("ORBIT_LLAMA_SOURCE_TREE_HASH", provenance.source_tree_sha256),
        _define("ORBIT_LLAMA_PATCHSET_HASH", provenance.patchset_sha256),
        _define(
            "ORBIT_LLAMA_BUILD_FLAGS",
            BRIDGE_BUILD_FLAGS,
        ),
        "-fvisibility=hidden",
    )
    identity = _build_identity(
        source=source,
        llama_root=llama_root,
        build_bin=build_bin,
        provenance=provenance,
        defines=defines,
    )
    identity_path = output.with_name(f"{output.name}.identity.json")
    identity_matches = False
    if output.exists() and identity_path.exists():
        try:
            stored = json.loads(identity_path.read_text(encoding="utf-8"))
            identity_matches = (
                stored.get("build") == identity
                and stored.get("artifact_sha256") == _sha256(output)
            )
        except (OSError, ValueError):
            identity_matches = False
    artifact = compile_cpp_helper(
        artifact_label="mtmd ABI bridge",
        source=source,
        output=output,
        llama_root=llama_root,
        build_bin=build_bin,
        runner=runner,
        shared=True,
        extra_compile_args=defines,
        extra_include_dirs=(llama_root / "tools" / "mtmd",),
        extra_link_args=(str(build_bin / runtime_library_filename("mtmd")), "-ldl"),
        force=not identity_matches,
    )
    stored_identity = {
        "build": identity,
        "artifact_sha256": _sha256(artifact),
    }
    identity_path.write_text(
        json.dumps(stored_identity, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return artifact, provenance


def _define(name: str, value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'-D{name}="{escaped}"'


def _build_identity(
    *,
    source: Path,
    llama_root: Path,
    build_bin: Path,
    provenance: LlamaProvenance,
    defines: tuple[str, ...],
) -> dict[str, object]:
    source_inputs = {
        "bridge_source": source,
        "provenance_manifest": llama_root.parents[1] / "LLAMA_PROVENANCE.json",
        "llama_header": llama_root / "include" / "llama.h",
        "mtmd_header": llama_root / "tools" / "mtmd" / "mtmd.h",
        "mtmd_helper_header": llama_root / "tools" / "mtmd" / "mtmd-helper.h",
    }
    libraries = {
        name: build_bin / name
        for name in (*platform_runtime_libs(), *platform_optional_runtime_libs())
    }
    compiler = os.environ.get("CXX", "c++")
    version = subprocess.run(
        [compiler, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "schema_version": 1,
        "compiler": compiler,
        "compiler_version": (version.stdout or version.stderr).splitlines()[0],
        "bridge_build_flags": BRIDGE_BUILD_FLAGS,
        "native_build": _native_build_configuration(build_bin),
        "defines": list(defines),
        "upstream_commit": provenance.upstream_commit,
        "upstream_tag": provenance.upstream_tag,
        "source_tree_sha256": provenance.source_tree_sha256,
        "patchset_sha256": provenance.patchset_sha256,
        "patched_paths": list(provenance.patched_paths),
        "source_inputs": {
            name: _sha256(path)
            for name, path in source_inputs.items()
            if path.exists()
        },
        "libraries": {
            name: _sha256(path)
            for name, path in libraries.items()
            if path.exists()
        },
    }


def _native_build_configuration(build_bin: Path) -> dict[str, str]:
    cache_path = build_bin.parent / "CMakeCache.txt"
    if not cache_path.exists():
        raise RuntimeError("missing CMakeCache.txt for native build provenance")
    keys = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
        "GGML_NATIVE",
        "GGML_OPENMP",
    }
    values: dict[str, str] = {"cmake_cache_sha256": _sha256(cache_path)}
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line or ":" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key, _separator, _value_type = key_and_type.partition(":")
        if key in keys:
            values[key] = value
    missing = keys.difference(values)
    if missing:
        raise RuntimeError("incomplete native build provenance: " + ", ".join(sorted(missing)))
    return values


def validate_mtmd_bridge_artifact(build_bin: Path, bridge_path: Path) -> dict[str, object]:
    identity_path = bridge_path.with_name(f"{bridge_path.name}.identity.json")
    try:
        stored = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("missing or invalid Orbit mtmd bridge identity") from exc
    build = stored.get("build")
    if not isinstance(build, dict):
        raise RuntimeError("invalid Orbit mtmd bridge build identity")
    expected_artifact = stored.get("artifact_sha256")
    if not isinstance(expected_artifact, str) or expected_artifact != _sha256(bridge_path):
        raise RuntimeError("Orbit mtmd bridge artifact identity mismatch")
    libraries = build.get("libraries")
    if not isinstance(libraries, dict):
        raise RuntimeError("missing Orbit mtmd bridge library identity")
    required = (*platform_runtime_libs(), runtime_library_filename("mtmd"))
    for name in required:
        expected = libraries.get(name)
        path = build_bin / name
        if not isinstance(expected, str) or not path.exists() or _sha256(path) != expected:
            raise RuntimeError(f"Orbit mtmd bridge library identity mismatch: {name}")
    return build


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
