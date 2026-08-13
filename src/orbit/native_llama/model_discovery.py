from __future__ import annotations

from ctypes import create_string_buffer
from dataclasses import dataclass
from pathlib import Path
import glob
import time
from typing import Callable, Iterable

from orbit.native_llama.bindings import GgmlLogCallback, LlamaLibrary
from orbit.native_llama.model_profiles import (
    VERIFIED_NATIVE_MODEL_IDENTITIES,
    NativeModelProfile,
    detect_native_model_profile,
    supports_low_memory_mode,
    verified_native_model_identity,
)
from orbit.native_llama.model_registry import (
    ModelManifest,
    default_hf_cache,
    default_models_dir,
    load_registry,
)


ProfileInspector = Callable[[Path], NativeModelProfile]


def _discard_native_log(_level: int, _text: bytes, _data) -> None:
    return None


_QUIET_LOG_CALLBACK = GgmlLogCallback(_discard_native_log)
PROFILE_METADATA_KEYS = frozenset(
    {
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "qwen3moe.context_length",
        "qwen3moe.expert_count",
        "qwen3moe.expert_used_count",
    }
)


@dataclass(frozen=True)
class ModelDiscoveryRow:
    model: str
    local: str
    support: str
    path_or_action: str
    model_id: str | None = None
    low_memory_supported: bool = False


@dataclass(frozen=True)
class ModelDiscoveryResult:
    rows: tuple[ModelDiscoveryRow, ...]
    wall_ms: float
    filesystem_scans: int
    metadata_inspections: int


@dataclass(frozen=True)
class _Inspection:
    path: Path
    profile: NativeModelProfile | None


class NativeProfileInspector:
    """Inspect GGUF metadata and template without allocating model weights."""

    def __init__(self, build_bin: Path) -> None:
        self._binding = LlamaLibrary(build_bin)
        self._lib = self._binding.lib
        self._lib.llama_log_set(_QUIET_LOG_CALLBACK, None)
        self._lib.ggml_backend_load_all()

    def __call__(self, path: Path) -> NativeModelProfile:
        return inspect_native_model_profile(self._binding, path)

    def close(self) -> None:
        self._lib.llama_log_set(GgmlLogCallback(), None)



def inspect_native_model_profile(binding: LlamaLibrary, path: Path) -> NativeModelProfile:
    """Verify one GGUF profile without allocating its weight tensors."""
    lib = binding.lib
    params = lib.llama_model_default_params()
    params.vocab_only = True
    params.use_mmap = True
    params.check_tensors = True
    model = lib.llama_model_load_from_file(str(path).encode(), params)
    if not model:
        raise RuntimeError("vocab-only GGUF inspection failed")
    try:
        metadata = _read_profile_metadata(lib, model)
        template_ptr = lib.llama_model_chat_template(model, None)
        template = template_ptr.decode("utf-8", errors="replace") if template_ptr else ""
        return detect_native_model_profile(metadata, template)
    finally:
        lib.llama_model_free(model)


def _read_profile_metadata(lib, model) -> dict[str, str]:
    metadata: dict[str, str] = {}
    count = max(0, int(lib.llama_model_meta_count(model)))
    for index in range(count):
        key = _metadata_text(lib.llama_model_meta_key_by_index, model, index)
        if key in PROFILE_METADATA_KEYS:
            metadata[key] = _metadata_text(lib.llama_model_meta_val_str_by_index, model, index)
    return metadata


def _metadata_text(function, model, index: int) -> str:
    needed = int(function(model, index, None, 0))
    if needed < 0:
        return ""
    buffer = create_string_buffer(needed + 1)
    written = int(function(model, index, buffer, len(buffer)))
    if written < 0:
        return ""
    return bytes(buffer[:written]).decode("utf-8", errors="replace")


def discover_models(
    *,
    models_dir: Path | None = None,
    hf_cache: Path | None = None,
    explicit_model: Path | None = None,
    build_bin: Path | None = None,
    inspector: ProfileInspector | None = None,
    manifests: Iterable[ModelManifest] | None = None,
) -> ModelDiscoveryResult:
    started = time.monotonic()
    local_root = models_dir or default_models_dir()
    cache_root = hf_cache or default_hf_cache()
    supported = tuple(load_registry() if manifests is None else manifests)
    candidates, scan_count = _local_candidates(
        local_root,
        cache_root,
        supported,
        explicit_model=explicit_model,
    )

    owned_inspector: NativeProfileInspector | None = None
    if candidates and inspector is None and build_bin is not None:
        try:
            owned_inspector = NativeProfileInspector(build_bin)
            inspector = owned_inspector
        except (OSError, RuntimeError):
            inspector = None

    try:
        inspections = tuple(_inspect(path, inspector) for path in candidates)
    finally:
        if owned_inspector is not None:
            owned_inspector.close()
    rows = _rows(supported, inspections)
    return ModelDiscoveryResult(
        rows=rows,
        wall_ms=(time.monotonic() - started) * 1000.0,
        filesystem_scans=scan_count,
        metadata_inspections=len(inspections) if inspector is not None else 0,
    )


def format_model_discovery(result: ModelDiscoveryResult) -> str:
    headings = ("Model", "Local", "Support", "Path / action")
    widths = [len(value) for value in headings[:3]]
    for row in result.rows:
        widths[0] = max(widths[0], len(row.model))
        widths[1] = max(widths[1], len(row.local))
        widths[2] = max(widths[2], len(row.support))
    lines = ["Models:", _format_columns(headings, widths)]
    lines.extend(
        _format_columns((row.model, row.local, row.support, row.path_or_action), widths)
        for row in result.rows
    )
    return "\n".join(lines)


def _format_columns(values: tuple[str, str, str, str], widths: list[int]) -> str:
    return f"{values[0]:<{widths[0]}}  {values[1]:<{widths[1]}}  {values[2]:<{widths[2]}}  {values[3]}"


def _local_candidates(
    models_dir: Path,
    hf_cache: Path,
    manifests: tuple[ModelManifest, ...],
    *,
    explicit_model: Path | None,
) -> tuple[tuple[Path, ...], int]:
    auxiliary_names = {
        spec.file
        for manifest in manifests
        for spec in (manifest.mmproj, manifest.mtp)
        if spec is not None
    }
    paths: set[Path] = set()
    scan_count = 0
    for pattern in ("*.gguf", "*/*.gguf"):
        scan_count += 1
        for path in models_dir.glob(pattern):
            if path.name not in auxiliary_names and not path.name.startswith(("mmproj-", "mtp-")):
                confined = _confined_regular_path(path, models_dir)
                if confined is not None:
                    paths.add(confined)
    for manifest in manifests:
        scan_count += 1
        for value in glob.glob(str(hf_cache / manifest.target.cache_glob)):
            confined = _confined_regular_path(Path(value), hf_cache)
            if confined is not None:
                paths.add(confined)
    if explicit_model is not None and explicit_model.exists():
        try:
            resolved_explicit = explicit_model.expanduser().resolve(strict=True)
        except OSError:
            resolved_explicit = None
        if resolved_explicit is not None and resolved_explicit.is_file():
            paths.add(resolved_explicit)
    unique_files: dict[tuple[int, int], Path] = {}
    for path in sorted(paths, key=str):
        try:
            stat_result = path.stat()
        except OSError:
            continue
        unique_files.setdefault((stat_result.st_dev, stat_result.st_ino), path)
    return tuple(unique_files.values()), scan_count


def _confined_regular_path(path: Path, root: Path) -> Path | None:
    try:
        resolved_root = root.expanduser().resolve()
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _inspect(path: Path, inspector: ProfileInspector | None) -> _Inspection:
    if inspector is None:
        return _Inspection(path=path, profile=None)
    try:
        return _Inspection(path=path, profile=inspector(path))
    except (OSError, RuntimeError, ValueError):
        return _Inspection(path=path, profile=None)


def _rows(
    manifests: tuple[ModelManifest, ...],
    inspections: tuple[_Inspection, ...],
) -> tuple[ModelDiscoveryRow, ...]:
    rows: list[ModelDiscoveryRow] = []
    consumed: set[Path] = set()
    manifests_by_profile: dict[str, ModelManifest] = {}
    for manifest in manifests:
        identity = verified_native_model_identity(manifest.profile_id)
        if identity is None or identity.architecture != manifest.architecture:
            raise ValueError("model registry profile mapping is not supported")
        if manifest.profile_id in manifests_by_profile:
            raise ValueError(f"duplicate model registry profile: {manifest.profile_id}")
        manifests_by_profile[manifest.profile_id] = manifest

    for identity in VERIFIED_NATIVE_MODEL_IDENTITIES:
        manifest = manifests_by_profile.get(identity.profile_id)
        matches = [
            item
            for item in inspections
            if item.profile is not None
            and item.profile.verified
            and item.profile.profile_id == identity.profile_id
            and item.profile.model_name == identity.model_name
        ]
        if not matches:
            if manifest is None:
                continue
            rows.append(
                ModelDiscoveryRow(
                    model=manifest.display_name,
                    local="MISSING",
                    support="VERIFIED",
                    path_or_action=f"orbit download {manifest.target.repo}/{manifest.target.file}",
                    model_id=manifest.id,
                )
            )
            continue
        for item in matches:
            consumed.add(item.path)
            rows.append(
                ModelDiscoveryRow(
                    model=manifest.display_name if manifest is not None else identity.model_name,
                    local="AVAILABLE",
                    support="VERIFIED",
                    path_or_action=str(item.path),
                    model_id=None if manifest is None else manifest.id,
                    low_memory_supported=supports_low_memory_mode(item.profile),
                )
            )

    for item in inspections:
        if item.path in consumed:
            continue
        support = "UNVERIFIED" if item.profile is None else "UNSUPPORTED"
        rows.append(
            ModelDiscoveryRow(
                model=item.path.name,
                local="AVAILABLE",
                support=support,
                path_or_action=str(item.path),
            )
        )
    return tuple(rows)
