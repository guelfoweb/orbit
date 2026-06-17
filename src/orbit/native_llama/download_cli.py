from __future__ import annotations

import argparse
from pathlib import Path
import sys

from orbit.native_llama.model_download import download_model
from orbit.native_llama.model_registry import default_models_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbid")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download", help="Download a GGUF model into Orbit's local model cache.")
    download.add_argument("spec", help="Hugging Face repo or repo/path/to/model.gguf")
    download.add_argument("--models-dir", help="Override Orbit model cache directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        return _download(args)
    return 2


def _download(args: argparse.Namespace) -> int:
    models_dir = default_models_dir() if args.models_dir is None else Path(args.models_dir)
    try:
        result = download_model(args.spec, models_dir=models_dir)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    action = "downloaded" if result.downloaded else "already present"
    print(f"{action}: {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
