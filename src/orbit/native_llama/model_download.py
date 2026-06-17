from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
import os
import tempfile

from orbit.native_llama.model_registry import (
    ModelFileSpec,
    default_models_dir,
    load_registry,
    local_model_path,
)


HF_RESOLVE_BASE = "https://huggingface.co"


@dataclass(frozen=True)
class DownloadRequest:
    repo: str
    file: str


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    downloaded: bool
    url: str


def parse_huggingface_spec(spec: str) -> DownloadRequest:
    cleaned = spec.strip().strip("/")
    if not cleaned:
        raise ValueError("empty Hugging Face model spec")
    parts = cleaned.split("/")
    if len(parts) < 2:
        raise ValueError("expected Hugging Face repo or repo/file")

    repo = "/".join(parts[:2])
    file_part = "/".join(parts[2:])
    if file_part:
        if not file_part.endswith(".gguf"):
            raise ValueError("only explicit .gguf files are supported")
        return DownloadRequest(repo=repo, file=file_part)

    manifest_match = _find_manifest_file_for_repo(repo)
    if manifest_match is None:
        raise ValueError(f"repo requires an explicit .gguf file: {repo}")
    return DownloadRequest(repo=manifest_match.repo, file=manifest_match.file)


def download_model(
    spec: str,
    *,
    models_dir: Path | None = None,
    retrieve=urlretrieve,
) -> DownloadResult:
    request = parse_huggingface_spec(spec)
    destination = local_model_path(
        ModelFileSpec(repo=request.repo, file=request.file, cache_glob=""),
        models_dir=models_dir or default_models_dir(),
    )
    url = huggingface_resolve_url(request)
    if destination.exists():
        return DownloadResult(path=destination, downloaded=False, url=url)

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        retrieve(url, str(tmp_path))
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return DownloadResult(path=destination, downloaded=True, url=url)


def huggingface_resolve_url(request: DownloadRequest) -> str:
    return f"{HF_RESOLVE_BASE}/{request.repo}/resolve/main/{request.file}"


def _find_manifest_file_for_repo(repo: str) -> ModelFileSpec | None:
    for manifest in load_registry():
        if manifest.target.repo == repo:
            return manifest.target
        if manifest.mtp is not None and manifest.mtp.repo == repo:
            return manifest.mtp
    return None
