from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "http://127.0.0.1:12120"
DEFAULT_WORKDIR = "workdir"
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_TOKENS = 512
METADATA_ENV_KEYS: tuple[str, ...] = (
    "ORBIT_KV_PREFIX_PREWARM",
    "ORBIT_MTP_TRACE",
    "ORBIT_KV_DIAG",
    "ORBIT_BASE_URL",
)
BACKEND_PROPS_KEYS: tuple[str, ...] = (
    "model",
    "ctx",
    "threads",
    "threads_batch",
    "batch",
    "ubatch",
    "mtp_enabled",
    "mtp_initialized",
    "multimodal_available",
    "gpu_layers",
)
PROMPTS: tuple[tuple[str, str], ...] = (
    ("chat", "hi, who are you? Answer in one short sentence."),
    ("list files", "list files and directories in this workdir"),
    ("small read", "read text/summary.txt and summarize it in one sentence"),
    ("long read", "read text/divina_commedia_inferno_canto1.txt and summarize it in Italian in 5 lines"),
    ("grep search", "search inside local text files for the word Virgilio and summarize the matches"),
    ("web url", "summarize this URL in one short paragraph: https://example.com"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbit bench-core")
    parser.add_argument("--base-url", default=os.environ.get("ORBIT_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--workdir", default=os.environ.get("WORKDIR", DEFAULT_WORKDIR))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("TIMEOUT", str(DEFAULT_TIMEOUT))))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", str(DEFAULT_MAX_TOKENS))))
    parser.add_argument("--orbit-bin", default=None, help="Override the orbit executable used for subprocess benchmark runs.")
    parser.add_argument("--no-metadata", action="store_true", help="Do not print benchmark metadata before running tasks.")
    return parser


def main(argv: list[str] | None = None, *, orbit_bin: str | None = None) -> int:
    args = build_parser().parse_args(argv)
    chosen_orbit = _resolve_orbit_bin(args.orbit_bin or orbit_bin)
    if not args.no_metadata:
        _print_metadata(args, chosen_orbit)
    for label, prompt in PROMPTS:
        print()
        print(f"## {label}")
        print(prompt)
        with tempfile.TemporaryDirectory() as home_dir:
            completed = subprocess.run(
                [
                    chosen_orbit,
                    "--base-url",
                    args.base_url,
                    "--workdir",
                    args.workdir,
                    "--timeout",
                    str(args.timeout),
                    "--max-tokens",
                    str(args.max_tokens),
                    prompt,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "HOME": home_dir},
            )
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode != 0:
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
            return completed.returncode
    return 0


def _resolve_orbit_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    discovered = shutil.which("orbit")
    if discovered:
        return discovered
    fallback = Path(sys.argv[0]).resolve()
    if fallback.exists():
        return str(fallback)
    raise SystemExit("error: orbit binary not found")


def _print_metadata(args: argparse.Namespace, chosen_orbit: str) -> None:
    print("# orbit bench-core metadata")
    print(f"orbit_commit: {_git_output(['rev-parse', 'HEAD'], fallback='unknown')}")
    print(f"orbit_tag: {_git_output(['describe', '--tags', '--exact-match', 'HEAD'], fallback='none')}")
    print(f"orbit_bin: {chosen_orbit}")
    print(f"base_url: {args.base_url}")
    print(f"workdir: {args.workdir}")
    print(f"timeout: {args.timeout}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print("env:")
    for key in METADATA_ENV_KEYS:
        print(f"  {key}: {os.environ.get(key, '<unset>')}")
    props = _fetch_backend_props(args.base_url)
    if props is None:
        print("backend_props: unavailable")
    else:
        print("backend:")
        for key in BACKEND_PROPS_KEYS:
            if key in props:
                print(f"  {key}: {props[key]}")


def _git_output(args: list[str], *, fallback: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return fallback
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        return fallback
    return value


def _fetch_backend_props(base_url: str) -> dict[str, object] | None:
    url = base_url.rstrip("/") + "/props"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError, TimeoutError):
        return None
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded
