from __future__ import annotations

import json
import re
from pathlib import Path


SHA_ID_RE = re.compile(r"^(?:sha256-)?([0-9a-f]{64})$")


def resolve_model_display_name(
    server_id: str | None,
    *,
    model_path: str | None = None,
    manifest_roots: list[Path] | None = None,
) -> str | None:
    digest = _extract_digest(server_id) or _extract_digest(model_path)
    if not digest:
        return server_id
    name = _resolve_from_ollama_manifests(digest, manifest_roots=manifest_roots)
    return name or server_id


def default_manifest_roots() -> list[Path]:
    roots = [
        Path.home() / ".ollama" / "models" / "manifests",
        Path("/usr/share/ollama/.ollama/models/manifests"),
    ]
    return [root for root in roots if root.exists()]


def _resolve_from_ollama_manifests(digest: str, *, manifest_roots: list[Path] | None = None) -> str | None:
    roots = manifest_roots if manifest_roots is not None else default_manifest_roots()
    candidates: list[tuple[int, str]] = []
    for root in roots:
        for manifest in root.rglob("*"):
            if not manifest.is_file():
                continue
            priority = _manifest_priority_for_digest(manifest, digest)
            if priority is None:
                continue
            name = _manifest_path_to_name(root, manifest)
            if name:
                candidates.append((priority, name))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], len(item[1]), item[1]))
    return candidates[0][1]


def _manifest_priority_for_digest(manifest: Path, digest: str) -> int | None:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    layers = data.get("layers")
    if not isinstance(layers, list):
        return None
    expected = f"sha256:{digest}"
    for layer in layers:
        if not isinstance(layer, dict) or layer.get("digest") != expected:
            continue
        return 1 if isinstance(layer.get("from"), str) and layer["from"] else 0
    return None


def _manifest_path_to_name(root: Path, manifest: Path) -> str | None:
    try:
        parts = manifest.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) < 4:
        return None
    namespace = parts[-3]
    model = parts[-2]
    tag = parts[-1]
    if namespace == "library":
        return f"{model}:{tag}"
    return f"{namespace}/{model}:{tag}"


def _extract_digest(value: str | None) -> str | None:
    if not value:
        return None
    text = Path(value).name if "/" in value else value
    match = SHA_ID_RE.match(text)
    return match.group(1) if match else None
