#!/usr/bin/env python3
"""Reproducible Orbit patchset attestation (orbit-patchset-v2).

The legacy `patchset_sha256` in LLAMA_PROVENANCE.json was produced by a packaging
pipeline that is not available here, and no candidate algorithm reproduces its
recorded values. It is preserved as an imported historical attestation: this tool
never recomputes, reinterprets, or validates it.

V2 is additive. It records `patchset_algorithm` and `patchset_v2_sha256`
alongside the legacy fields, so existing readers and the bridge identity chain
are unaffected.

The V2 hash binds the actual patch content: for every vendored file that differs
from the pinned upstream commit (or is absent upstream), both the upstream and
the vendored SHA-256 are recorded.

  canonical object : [{"path", "upstream_sha256"|null, "vendored_sha256"}]
  ordering         : sorted by path
  encoding         : json.dumps(..., separators=(",", ":")), UTF-8, no newline
  hashing          : sha256 over raw file bytes; no newline normalization
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ALGORITHM = "orbit-patchset-v2"

# Mirrors the exclusion filter used by source_tree_sha256 so the two views of the
# vendored tree cannot drift apart.
EXCLUDED_PARTS = {".git", "build", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _vendored_files(root: Path) -> list[str]:
    out: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        out.append(relative.as_posix())
    return sorted(out)


def _upstream_blob(upstream_root: Path, commit: str, relative: str) -> bytes | None:
    """Read a file from the pinned upstream commit.

    Reads through `git show <commit>:<path>` so a dirty upstream working tree
    cannot influence the attestation.
    """
    completed = subprocess.run(
        ["git", "-C", str(upstream_root), "show", f"{commit}:{relative}"],
        capture_output=True,
    )
    return completed.stdout if completed.returncode == 0 else None


def build_patchset(upstream_root: Path, vendored_root: Path, commit: str) -> list[dict]:
    """The canonical V2 object: every vendored file that diverges from upstream."""
    entries: list[dict] = []
    for relative in _vendored_files(vendored_root):
        vendored = (vendored_root / relative).read_bytes()
        upstream = _upstream_blob(upstream_root, commit, relative)
        if upstream is not None and upstream == vendored:
            continue
        entries.append(
            {
                "path": relative,
                "upstream_sha256": _sha256_bytes(upstream) if upstream is not None else None,
                "vendored_sha256": _sha256_bytes(vendored),
            }
        )
    entries.sort(key=lambda entry: entry["path"])
    return entries


def canonical_bytes(entries: list[dict]) -> bytes:
    return json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def patchset_v2_sha256(entries: list[dict]) -> str:
    return _sha256_bytes(canonical_bytes(entries))


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(args: argparse.Namespace) -> tuple[dict, Path, Path, str, list[dict]]:
    manifest_path = args.manifest
    manifest = _load_manifest(manifest_path)
    commit = args.upstream_commit or manifest["upstream_commit"]
    entries = build_patchset(args.upstream_root, args.vendored_root, commit)
    return manifest, manifest_path, args.vendored_root, commit, entries


def cmd_generate(args: argparse.Namespace) -> int:
    manifest, manifest_path, _, _, entries = _resolve(args)
    digest = patchset_v2_sha256(entries)
    declared = [entry["path"] for entry in entries]

    manifest["patched_paths"] = declared
    manifest["patchset_algorithm"] = ALGORITHM
    manifest["patchset_v2_sha256"] = digest
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"patchset_algorithm : {ALGORITHM}")
    print(f"patchset_v2_sha256 : {digest}")
    print(f"patched_paths      : {len(declared)}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    manifest, _, _, _, entries = _resolve(args)
    digest = patchset_v2_sha256(entries)
    declared = list(manifest.get("patched_paths") or [])
    computed = [entry["path"] for entry in entries]

    problems: list[str] = []
    if manifest.get("patchset_algorithm") != ALGORITHM:
        problems.append(
            f"patchset_algorithm is {manifest.get('patchset_algorithm')!r}, expected {ALGORITHM!r}"
        )
    if manifest.get("patchset_v2_sha256") != digest:
        problems.append(
            f"patchset_v2_sha256 mismatch:\n"
            f"    manifest {manifest.get('patchset_v2_sha256')}\n"
            f"    computed {digest}"
        )
    duplicates = sorted({path for path in declared if declared.count(path) > 1})
    if duplicates:
        # `_from_payload` also rejects these at load time; catching them here too
        # keeps `check` self-sufficient rather than relying on a second reader.
        problems.append(f"duplicate declared patched paths: {duplicates}")
    if sorted(declared) != computed:
        missing = sorted(set(computed) - set(declared))
        extra = sorted(set(declared) - set(computed))
        if missing:
            problems.append(f"vendored files differ from upstream but are undeclared: {missing}")
        if extra:
            problems.append(f"declared patched paths that do not differ from upstream: {extra}")
        if not missing and not extra:
            problems.append("declared patched paths do not match the derived set")

    if problems:
        for problem in problems:
            print(f"provenance v2 check FAILED: {problem}", file=sys.stderr)
        return 1
    print(f"provenance v2 check OK ({len(computed)} patched paths, {digest})")
    return 0


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    default_vendored = root / "src/orbit/native_llama/vendor/source/llama.cpp"
    default_manifest = root / "src/orbit/native_llama/vendor/LLAMA_PROVENANCE.json"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "check"))
    parser.add_argument("--upstream-root", type=Path, default=Path.home() / "LAB/llama.cpp")
    parser.add_argument("--vendored-root", type=Path, default=default_vendored)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--upstream-commit", default=None)
    args = parser.parse_args(argv)

    if not args.vendored_root.is_dir():
        print(f"vendored root not found: {args.vendored_root}", file=sys.stderr)
        return 2
    if not (args.upstream_root / ".git").exists():
        print(f"upstream git checkout not found: {args.upstream_root}", file=sys.stderr)
        return 2
    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    return cmd_generate(args) if args.command == "generate" else cmd_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
