from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess


EMPTY_PATCHSET_SHA256 = hashlib.sha256(b"[]").hexdigest()


@dataclass(frozen=True)
class LlamaProvenance:
    upstream_commit: str
    upstream_tag: str
    source_tree_sha256: str
    patchset_sha256: str
    patched_paths: tuple[str, ...] = ()


def load_llama_provenance(source_root: Path) -> LlamaProvenance:
    root = source_root.expanduser().resolve()
    manifest_path = root.parents[1] / "LLAMA_PROVENANCE.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = _from_payload(payload)
        actual_tree = source_tree_sha256(root)
        if actual_tree != provenance.source_tree_sha256:
            raise RuntimeError(
                "vendored llama.cpp tree does not match LLAMA_PROVENANCE.json"
            )
        return provenance
    return _from_clean_git_checkout(root)


def source_tree_sha256(root: Path) -> str:
    files: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            ".git" in relative.parts
            or "build" in relative.parts
            or "__pycache__" in relative.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _from_clean_git_checkout(root: Path) -> LlamaProvenance:
    status = _git(root, "status", "--porcelain")
    if status:
        raise RuntimeError("llama.cpp source override must be a clean Git checkout")
    commit = _git(root, "rev-parse", "HEAD")
    tags = [line for line in _git(root, "tag", "--points-at", "HEAD").splitlines() if line]
    tag = sorted(tags)[0] if tags else "untagged"
    return LlamaProvenance(
        upstream_commit=commit,
        upstream_tag=tag,
        source_tree_sha256=source_tree_sha256(root),
        patchset_sha256=EMPTY_PATCHSET_SHA256,
    )


def _from_payload(payload: object) -> LlamaProvenance:
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise RuntimeError("invalid llama.cpp provenance manifest")
    required = (
        "upstream_commit",
        "upstream_tag",
        "source_tree_sha256",
        "patchset_sha256",
    )
    values = {key: payload.get(key) for key in required}
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise RuntimeError("incomplete llama.cpp provenance manifest")
    if not re.fullmatch(r"[0-9a-f]{40}", values["upstream_commit"]):
        raise RuntimeError("invalid llama.cpp upstream commit")
    for key in ("source_tree_sha256", "patchset_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", values[key]):
            raise RuntimeError(f"invalid llama.cpp provenance hash: {key}")
    patched_paths = payload.get("patched_paths")
    if not isinstance(patched_paths, list) or any(
        not isinstance(path, str) or not path or path.startswith("/") or ".." in Path(path).parts
        for path in patched_paths
    ):
        raise RuntimeError("invalid llama.cpp patchset path list")
    if len(set(patched_paths)) != len(patched_paths):
        raise RuntimeError("duplicate llama.cpp patchset path")
    return LlamaProvenance(**values, patched_paths=tuple(patched_paths))


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"unable to read llama.cpp Git provenance: {detail}")
    return completed.stdout.strip()
