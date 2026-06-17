from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import glob
import json
from typing import Any


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
    backend: str
    architecture: str
    target: ModelFileSpec
    mtp: MtpSpec | None


@dataclass(frozen=True)
class ResolvedModel:
    manifest: ModelManifest
    target_path: Path
    draft_mtp_path: Path | None
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
    mtp_data = data.get("mtp")
    return ModelManifest(
        id=str(data["id"]),
        backend=str(data["backend"]),
        architecture=str(data["architecture"]),
        target=_file_spec(data["target"]),
        mtp=_mtp_spec(mtp_data) if isinstance(mtp_data, dict) else None,
    )


def load_registry(path: Path | None = None) -> list[ModelManifest]:
    if path is None:
        text = resources.files(__package__).joinpath(REGISTRY_RESOURCE).read_text(encoding="utf-8")
        data = json.loads(text)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    return [_manifest(item) for item in data.get("models", [])]


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
    draft_mtp_override: Path | None = None,
) -> ResolvedModel:
    local_root = models_dir or default_models_dir()
    cache_root = hf_cache or default_hf_cache()
    target_path = target_override or _resolve_file(manifest.target, models_dir=local_root, hf_cache=cache_root)
    if target_path is None or not target_path.exists():
        raise FileNotFoundError(f"target model not found: {manifest.target.repo}:{manifest.target.file}")

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
        draft_mtp_path=draft_path,
        mtp_available=draft_path is not None,
        fallback_reason=fallback_reason,
    )


def _resolve_file(spec: ModelFileSpec, *, models_dir: Path, hf_cache: Path) -> Path | None:
    local_path = local_model_path(spec, models_dir=models_dir)
    if local_path.exists():
        return local_path
    return newest_match(hf_cache / spec.cache_glob)
