from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .build_support import compile_cpp_helper
from .llama_provenance import LlamaProvenance, load_llama_provenance
from .native_names import platform_runtime_libs, runtime_library_filename


CHAT_BRIDGE_API_VERSION = 1
CHAT_BRIDGE_BUILD_FLAGS = "-std=c++17;-shared;-fPIC;-fvisibility=hidden"


def chat_bridge_filename() -> str:
    return runtime_library_filename("orbit-chat-bridge")


def build_chat_bridge(
    *,
    llama_root: Path,
    build_dir: Path,
    build_bin: Path,
    runner=subprocess.run,
) -> tuple[Path, LlamaProvenance]:
    provenance = load_llama_provenance(llama_root)
    source = Path(__file__).parent / "vendor" / "shim" / "orbit_chat_bridge.cpp"
    output = build_dir / chat_bridge_filename()
    identity = _build_identity(
        source=source,
        llama_root=llama_root,
        build_bin=build_bin,
        provenance=provenance,
    )
    identity_path = _identity_path(output)
    identity_matches = False
    if output.exists() and identity_path.exists():
        try:
            stored = json.loads(identity_path.read_text(encoding="utf-8"))
            identity_matches = stored.get("build") == identity and stored.get("artifact_sha256") == _sha256(output)
        except (OSError, ValueError):
            identity_matches = False

    artifact = compile_cpp_helper(
        artifact_label="chat compatibility bridge",
        source=source,
        output=output,
        llama_root=llama_root,
        build_bin=build_bin,
        runner=runner,
        shared=True,
        extra_compile_args=("-fvisibility=hidden",),
        extra_include_dirs=(llama_root / "vendor",),
        force=not identity_matches,
    )
    identity_path.write_text(
        json.dumps(
            {"build": identity, "artifact_sha256": _sha256(artifact)},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact, provenance


def install_chat_bridge(artifact: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, destination / artifact.name)
    identity = _identity_path(artifact)
    if identity.exists():
        shutil.copy2(identity, destination / identity.name)


def validate_chat_bridge_artifact(build_bin: Path, bridge_path: Path) -> dict[str, object]:
    try:
        stored = json.loads(_identity_path(bridge_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("missing or invalid Orbit chat bridge identity") from exc
    build = stored.get("build")
    if not isinstance(build, dict):
        raise RuntimeError("invalid Orbit chat bridge build identity")
    if stored.get("artifact_sha256") != _sha256(bridge_path):
        raise RuntimeError("Orbit chat bridge artifact identity mismatch")
    libraries = build.get("libraries")
    if not isinstance(libraries, dict):
        raise RuntimeError("missing Orbit chat bridge library identity")
    for name in platform_runtime_libs():
        expected = libraries.get(name)
        path = build_bin / name
        if not isinstance(expected, str) or not path.exists() or _sha256(path) != expected:
            raise RuntimeError(f"Orbit chat bridge library identity mismatch: {name}")
    return build


def _build_identity(
    *,
    source: Path,
    llama_root: Path,
    build_bin: Path,
    provenance: LlamaProvenance,
) -> dict[str, object]:
    compiler = os.environ.get("CXX", "c++")
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=False)
    inputs = {
        "bridge_source": source,
        "llama_header": llama_root / "include" / "llama.h",
        "chat_header": llama_root / "common" / "chat.h",
        "common_header": llama_root / "common" / "common.h",
        "json_header": llama_root / "vendor" / "nlohmann" / "json.hpp",
        "provenance_manifest": llama_root.parents[1] / "LLAMA_PROVENANCE.json",
    }
    return {
        "schema_version": 1,
        "api_version": CHAT_BRIDGE_API_VERSION,
        "compiler": compiler,
        "compiler_version": (version.stdout or version.stderr).splitlines()[0],
        "bridge_build_flags": CHAT_BRIDGE_BUILD_FLAGS,
        "upstream_commit": provenance.upstream_commit,
        "upstream_tag": provenance.upstream_tag,
        "source_tree_sha256": provenance.source_tree_sha256,
        "patchset_sha256": provenance.patchset_sha256,
        "source_inputs": {name: _sha256(path) for name, path in inputs.items() if path.exists()},
        "libraries": {
            name: _sha256(build_bin / name)
            for name in platform_runtime_libs()
            if (build_bin / name).exists()
        },
    }


def _identity_path(artifact: Path) -> Path:
    return artifact.with_name(f"{artifact.name}.identity.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
