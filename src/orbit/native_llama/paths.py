from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model_registry import ResolvedModel, get_manifest, resolve_model


DEFAULT_LLAMA_ROOT = Path("/home/guelfoweb/LAB/llama.cpp-gemma4-mtp-qualcomm")
DEFAULT_MODEL_ID = "gemma4-12b-it-q4km"
LEGACY_MODEL_ID = "legacy-path"


@dataclass(frozen=True)
class NativeLlamaPaths:
    llama_root: Path
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
    llama_root: Path = DEFAULT_LLAMA_ROOT,
    model_id: str = DEFAULT_MODEL_ID,
    model: Path | None = None,
    mmproj: Path | None = None,
    models_dir: Path | None = None,
    hf_cache: Path | None = None,
) -> NativeLlamaPaths:
    build_bin = llama_root / "build/bin"
    library = build_bin / "libllama.so"
    manifest = get_manifest(model_id)
    resolved = _resolve_model(
        manifest_id=model_id,
        model_override=model,
        mmproj_override=mmproj,
        models_dir=models_dir,
        hf_cache=hf_cache,
    )

    if not library.exists():
        raise FileNotFoundError(f"libllama.so not found: {library}")

    return NativeLlamaPaths(
        llama_root=llama_root,
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
    llama_root: Path = DEFAULT_LLAMA_ROOT,
    model: Path,
    mmproj: Path | None = None,
) -> NativeLlamaPaths:
    build_bin = llama_root / "build/bin"
    library = build_bin / "libllama.so"
    resolved_model = model.expanduser().resolve()

    if not library.exists():
        raise FileNotFoundError(f"libllama.so not found: {library}")
    if not resolved_model.exists():
        raise FileNotFoundError(f"model not found: {resolved_model}")
    resolved_mmproj = None if mmproj is None else mmproj.expanduser().resolve()
    if resolved_mmproj is not None and not resolved_mmproj.exists():
        raise FileNotFoundError(f"mmproj not found: {resolved_mmproj}")

    return NativeLlamaPaths(
        llama_root=llama_root,
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
