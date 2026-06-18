from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


DEFAULT_BASE_URL = "http://127.0.0.1:11976"
DEFAULT_WORKDIR = "workdir"
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_TOKENS = 512
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
    return parser


def main(argv: list[str] | None = None, *, orbit_bin: str | None = None) -> int:
    args = build_parser().parse_args(argv)
    chosen_orbit = _resolve_orbit_bin(args.orbit_bin or orbit_bin)
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
