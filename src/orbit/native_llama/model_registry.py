from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
import glob
import json
from typing import Any

from orbit.native_llama.model_profiles import verified_native_model_identity


REGISTRY_RESOURCE = "model_registry.json"


@dataclass(frozen=True)
class ModelFileSpec:
    repo: str
    file: str
    cache_glob: str


@dataclass(frozen=True)
class MtpSpec(ModelFileSpec):
    enabled_by_default: bool
    required: bool
    spec_type: str


@dataclass(frozen=True)
class ModelManifest:
    id: str
    display_name: str
    profile_id: str
    backend: str
    architecture: str
    target: ModelFileSpec
    mmproj: ModelFileSpec | None
    mtp: MtpSpec | None


@dataclass(frozen=True)
class ResolvedModel:
    manifest: ModelManifest
    target_path: Path
    mmproj_path: Path | None
    draft_mtp_path: Path | None
    multimodal_available: bool
    multimodal_fallback_reason: str | None
    mtp_available: bool
    fallback_reason: str | None


def default_hf_cache() -> Path:
    return Path.home() / ".cache/huggingface/hub"


def default_orbit_model_cache() -> Path:
    return Path.home() / ".cache/orbit/models"


def find_project_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src/orbit").is_dir():
            return candidate
    return None


def default_models_dir(start: Path | None = None) -> Path:
    root = find_project_root(start)
    if root is not None:
        return root / "models"
    return default_orbit_model_cache()


def local_model_path(spec: ModelFileSpec, *, models_dir: Path) -> Path:
    return models_dir / spec.repo.replace("/", "--") / spec.file


def newest_match(pattern: Path) -> Path | None:
    matches = [Path(path) for path in glob.glob(str(pattern))]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _file_spec(data: dict[str, Any]) -> ModelFileSpec:
    return ModelFileSpec(
        repo=str(data["repo"]),
        file=str(data["file"]),
        cache_glob=str(data["cache_glob"]),
    )


def _mtp_spec(data: dict[str, Any]) -> MtpSpec:
    return MtpSpec(
        repo=str(data["repo"]),
        file=str(data["file"]),
        cache_glob=str(data["cache_glob"]),
        enabled_by_default=bool(data.get("enabled_by_default", False)),
        required=bool(data.get("required", False)),
        spec_type=str(data["spec_type"]),
    )


def _manifest(data: dict[str, Any]) -> ModelManifest:
    mmproj_data = data.get("mmproj")
    mtp_data = data.get("mtp")
    return ModelManifest(
        id=str(data["id"]),
        display_name=str(data["display_name"]),
        profile_id=str(data["profile_id"]),
        backend=str(data["backend"]),
        architecture=str(data["architecture"]),
        target=_file_spec(data["target"]),
        mmproj=_file_spec(mmproj_data) if isinstance(mmproj_data, dict) else None,
        mtp=_mtp_spec(mtp_data) if isinstance(mtp_data, dict) else None,
    )


def load_registry(path: Path | None = None) -> list[ModelManifest]:
    if path is None:
        text = resources.files(__package__).joinpath(REGISTRY_RESOURCE).read_text(encoding="utf-8")
        data = _load_json(text)
    else:
        data = _load_json(path.read_text(encoding="utf-8"))
    model_data = _validate_registry(data)
    return [_manifest(item) for item in model_data]


def _load_json(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate model registry key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite model registry value: {value}")

    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=reject_constant)


def _validate_registry(data: Any) -> list[dict[str, Any]]:
    root = _require_mapping(data, "registry")
    _require_keys(root, required={"version", "models"}, optional=set(), label="registry")
    if type(root["version"]) is not int or root["version"] != 1:
        raise ValueError("unsupported model registry version")
    if not isinstance(root["models"], list):
        raise ValueError("model registry models must be a list")

    models: list[dict[str, Any]] = []
    ids: set[str] = set()
    names: set[str] = set()
    profiles: set[str] = set()
    downloads: set[tuple[str, str]] = set()
    for index, value in enumerate(root["models"]):
        label = f"models[{index}]"
        model = _require_mapping(value, label)
        _require_keys(
            model,
            required={"id", "display_name", "profile_id", "backend", "architecture", "target"},
            optional={"mmproj", "mtp"},
            label=label,
        )
        model_id = _require_string(model, "id", label)
        display_name = _require_string(model, "display_name", label)
        profile_id = _require_string(model, "profile_id", label)
        backend = _require_string(model, "backend", label)
        architecture = _require_string(model, "architecture", label)
        if backend != "native-llama":
            raise ValueError(f"{label}.backend must be native-llama")
        identity = verified_native_model_identity(profile_id)
        if identity is None or identity.architecture != architecture:
            raise ValueError(f"{label} has an invalid profile/architecture mapping")
        _add_unique(ids, model_id, "model id")
        _add_unique(names, display_name, "model display name")
        _add_unique(profiles, profile_id, "model profile")

        specs = [("target", _validate_file_spec(model["target"], f"{label}.target"))]
        if "mmproj" in model:
            specs.append(("mmproj", _validate_file_spec(model["mmproj"], f"{label}.mmproj")))
        if "mtp" in model:
            specs.append(("mtp", _validate_mtp_spec(model["mtp"], f"{label}.mtp")))
        for kind, spec in specs:
            download = (str(spec["repo"]), str(spec["file"]))
            if download in downloads:
                raise ValueError(f"duplicate model registry download: {download[0]}/{download[1]}")
            downloads.add(download)
            if kind == "target" and not str(spec["file"]).lower().endswith(".gguf"):
                raise ValueError(f"{label}.target.file must be a GGUF")
        models.append(model)
    return models


def _validate_file_spec(value: Any, label: str) -> dict[str, Any]:
    spec = _require_mapping(value, label)
    _require_keys(spec, required={"repo", "file", "cache_glob"}, optional=set(), label=label)
    repo = _require_string(spec, "repo", label)
    file_name = _require_string(spec, "file", label)
    cache_glob = _require_string(spec, "cache_glob", label)
    repo_parts = repo.split("/")
    if len(repo_parts) != 2 or any(part in {"", ".", ".."} for part in repo_parts):
        raise ValueError(f"{label}.repo must be owner/repository")
    _validate_relative_path(file_name, f"{label}.file", allow_wildcard=False)
    _validate_relative_path(cache_glob, f"{label}.cache_glob", allow_wildcard=True)
    return spec


def _validate_mtp_spec(value: Any, label: str) -> dict[str, Any]:
    spec = _require_mapping(value, label)
    _require_keys(
        spec,
        required={"repo", "file", "cache_glob", "enabled_by_default", "required", "spec_type"},
        optional=set(),
        label=label,
    )
    _validate_file_spec(
        {key: spec[key] for key in ("repo", "file", "cache_glob")},
        label,
    )
    if type(spec["enabled_by_default"]) is not bool or type(spec["required"]) is not bool:
        raise ValueError(f"{label} enablement fields must be booleans")
    _require_string(spec, "spec_type", label)
    return spec


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_keys(data: dict[str, Any], *, required: set[str], optional: set[str], label: str) -> None:
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing:
        raise ValueError(f"{label} missing keys: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{label} unknown keys: {', '.join(sorted(unknown))}")


def _require_string(data: dict[str, Any], key: str, label: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _validate_relative_path(value: str, label: str, *, allow_wildcard: bool) -> None:
    if "\\" in value or PurePosixPath(value).is_absolute():
        raise ValueError(f"{label} must be a relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} contains an unsafe path component")
    wildcard_count = sum(part == "*" for part in parts)
    has_other_wildcard = any(any(char in part for char in "*?[") and part != "*" for part in parts)
    if has_other_wildcard or (wildcard_count and not allow_wildcard) or wildcard_count > 1:
        raise ValueError(f"{label} contains an unsupported wildcard")


def _add_unique(seen: set[str], value: str, label: str) -> None:
    if value in seen:
        raise ValueError(f"duplicate {label}: {value}")
    seen.add(value)


def get_manifest(model_id: str, *, registry_path: Path | None = None) -> ModelManifest:
    for manifest in load_registry(registry_path):
        if manifest.id == model_id:
            return manifest
    raise KeyError(f"unknown native model manifest: {model_id}")


def resolve_model(
    manifest: ModelManifest,
    *,
    models_dir: Path | None = None,
    hf_cache: Path | None = None,
    target_override: Path | None = None,
    mmproj_override: Path | None = None,
    draft_mtp_override: Path | None = None,
) -> ResolvedModel:
    local_root = models_dir or default_models_dir()
    cache_root = hf_cache or default_hf_cache()
    target_path = target_override or _resolve_file(manifest.target, models_dir=local_root, hf_cache=cache_root)
    if target_path is None or not target_path.exists():
        raise FileNotFoundError(f"target model not found: {manifest.target.repo}:{manifest.target.file}")

    mmproj_path: Path | None = None
    multimodal_fallback_reason: str | None = None
    if manifest.mmproj is None:
        multimodal_fallback_reason = "mmproj-not-declared"
    else:
        mmproj_path = mmproj_override or _resolve_file(manifest.mmproj, models_dir=local_root, hf_cache=cache_root)
        if mmproj_path is None or not mmproj_path.exists():
            mmproj_path = None
            multimodal_fallback_reason = "mmproj-missing"

    draft_path: Path | None = None
    fallback_reason: str | None = None
    mtp = manifest.mtp
    if mtp is None:
        fallback_reason = "mtp-not-declared"
    elif not mtp.enabled_by_default:
        fallback_reason = "mtp-disabled"
    else:
        draft_path = draft_mtp_override or _resolve_file(mtp, models_dir=local_root, hf_cache=cache_root)
        if draft_path is None or not draft_path.exists():
            if mtp.required:
                raise FileNotFoundError(f"draft MTP model not found: {mtp.repo}:{mtp.file}")
            draft_path = None
            fallback_reason = "draft-mtp-missing"

    return ResolvedModel(
        manifest=manifest,
        target_path=target_path,
        mmproj_path=mmproj_path,
        draft_mtp_path=draft_path,
        multimodal_available=mmproj_path is not None,
        multimodal_fallback_reason=multimodal_fallback_reason,
        mtp_available=draft_path is not None,
        fallback_reason=fallback_reason,
    )


def _resolve_file(spec: ModelFileSpec, *, models_dir: Path, hf_cache: Path) -> Path | None:
    local_path = local_model_path(spec, models_dir=models_dir)
    if local_path.exists():
        return local_path
    return newest_match(hf_cache / spec.cache_glob)
