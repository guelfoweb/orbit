from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .build_support import PACKAGE_NATIVE_ROOT
from .model_registry import ResolvedModel, get_manifest, resolve_model
from .native_names import runtime_library_filename


DEFAULT_VENDOR_LIB_DIR = PACKAGE_NATIVE_ROOT / "lib"
DEFAULT_VENDOR_SHIM_DIR = PACKAGE_NATIVE_ROOT / "shim"
DEFAULT_VENDOR_BUILD_BIN = PACKAGE_NATIVE_ROOT / "build" / "llama.cpp" / "bin"
BUNDLED_SOURCE_ROOT = PACKAGE_NATIVE_ROOT / "source" / "llama.cpp"
DEFAULT_LLAMA_LIB_DIR = Path(os.environ["ORBIT_LLAMA_LIB_DIR"]).expanduser().resolve() if os.environ.get("ORBIT_LLAMA_LIB_DIR") else None
BUNDLED_SOURCE_ROOT = PACKAGE_NATIVE_ROOT / "source" / "llama.cpp"
DEFAULT_LLAMA_ROOT = Path(os.environ["ORBIT_LLAMA_ROOT"]).expanduser().resolve() if os.environ.get("ORBIT_LLAMA_ROOT") else None
DEFAULT_MODEL_ID = "gemma4-12b-it-q4km"
LEGACY_MODEL_ID = "legacy-path"


@dataclass(frozen=True)
class NativeLlamaPaths:
    llama_root: Path | None
    build_bin: Path
    library: Path
    model: Path
    mmproj_model: Path | None = None
    draft_mtp_model: Path | None = None
    multimodal_available: bool = False
    multimodal_fallback_reason: str | None = None
    mtp_available: bool = False
    fallback_reason: str | None = None
    model_id: str = DEFAULT_MODEL_ID


def resolve_paths(
    *,
    llama_root: Path | None = DEFAULT_LLAMA_ROOT,
    model_id: str = DEFAULT_MODEL_ID,
    model: Path | None = None,
    mmproj: Path | None = None,
    models_dir: Path | None = None,
    hf_cache: Path | None = None,
) -> NativeLlamaPaths:
    source_root = _resolve_build_source_root(llama_root)
    _runtime_llama_root, build_bin, library = _resolve_native_runtime(llama_root)
    manifest = get_manifest(model_id)
    resolved = _resolve_model(
        manifest_id=model_id,
        model_override=model,
        mmproj_override=mmproj,
        models_dir=models_dir,
        hf_cache=hf_cache,
    )

    return NativeLlamaPaths(
        llama_root=source_root,
        build_bin=build_bin,
        library=library,
        model=resolved.target_path,
        mmproj_model=resolved.mmproj_path,
        draft_mtp_model=resolved.draft_mtp_path,
        multimodal_available=resolved.multimodal_available,
        multimodal_fallback_reason=resolved.multimodal_fallback_reason,
        mtp_available=resolved.mtp_available,
        fallback_reason=resolved.fallback_reason,
        model_id=manifest.id,
    )


def resolve_legacy_paths(
    *,
    llama_root: Path | None = DEFAULT_LLAMA_ROOT,
    model: Path,
    mmproj: Path | None = None,
) -> NativeLlamaPaths:
    source_root = _resolve_build_source_root(llama_root)
    _runtime_llama_root, build_bin, library = _resolve_native_runtime(llama_root)
    resolved_model = model.expanduser().resolve()

    if not resolved_model.exists():
        raise FileNotFoundError(f"model not found: {resolved_model}")
    resolved_mmproj = None if mmproj is None else mmproj.expanduser().resolve()
    if resolved_mmproj is not None and not resolved_mmproj.exists():
        raise FileNotFoundError(f"mmproj not found: {resolved_mmproj}")

    return NativeLlamaPaths(
        llama_root=source_root,
        build_bin=build_bin,
        library=library,
        model=resolved_model,
        mmproj_model=resolved_mmproj,
        draft_mtp_model=None,
        multimodal_available=resolved_mmproj is not None,
        multimodal_fallback_reason=None if resolved_mmproj is not None else "legacy-mmproj-missing",
        mtp_available=False,
        fallback_reason="legacy-model-path",
        model_id=LEGACY_MODEL_ID,
    )


def _resolve_model(
    *,
    manifest_id: str,
    model_override: Path | None,
    mmproj_override: Path | None,
    models_dir: Path | None,
    hf_cache: Path | None,
) -> ResolvedModel:
    manifest = get_manifest(manifest_id)
    return resolve_model(
        manifest,
        models_dir=models_dir,
        hf_cache=hf_cache,
        target_override=model_override,
        mmproj_override=mmproj_override,
    )


def _resolve_native_runtime(llama_root: Path | None) -> tuple[Path | None, Path, Path]:
    library_name = runtime_library_filename("llama")
    vendored_soname_library = DEFAULT_VENDOR_BUILD_BIN / library_name
    if _has_packaged_soname_runtime(DEFAULT_VENDOR_BUILD_BIN) and vendored_soname_library.exists():
        return None, DEFAULT_VENDOR_BUILD_BIN, vendored_soname_library
    vendored_library = DEFAULT_VENDOR_LIB_DIR / library_name
    if vendored_library.exists():
        return None, DEFAULT_VENDOR_LIB_DIR, vendored_library
    searched: list[Path] = [vendored_library]
    if DEFAULT_LLAMA_LIB_DIR is not None:
        env_library = DEFAULT_LLAMA_LIB_DIR / library_name
        if env_library.exists():
            return None, DEFAULT_LLAMA_LIB_DIR, env_library
        searched.append(env_library)
    if llama_root is not None:
        resolved_root = llama_root.expanduser().resolve()
        build_bin = resolved_root / "build/bin"
        library = build_bin / library_name
        if library.exists():
            return resolved_root, build_bin, library
        searched.append(library)
    searched_text = ", ".join(str(path) for path in searched)
    raise FileNotFoundError(
        f"{library_name} not found. "
        f"Searched: {searched_text}. "
        "Provide ORBIT_LLAMA_LIB_DIR, --llama-root, or ORBIT_LLAMA_ROOT, or package native libraries under orbit/native_llama/vendor/lib."
    )


def _has_packaged_soname_runtime(path: Path) -> bool:
    return (path / "libllama.so.0").exists() and (path / "libllama-common.so.0").exists()


def _resolve_build_source_root(llama_root: Path | None) -> Path | None:
    if llama_root is not None:
        resolved_root = llama_root.expanduser().resolve()
        if (resolved_root / "CMakeLists.txt").exists():
            return resolved_root
    if (BUNDLED_SOURCE_ROOT / "CMakeLists.txt").exists():
        return BUNDLED_SOURCE_ROOT
    return None
