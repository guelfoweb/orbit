from __future__ import annotations

import argparse
from pathlib import Path
import sys

from orbit.native_llama.model_download import download_all_for_repo, download_model
from orbit.native_llama.model_registry import default_models_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbit download")
    parser.add_argument("spec", help="Hugging Face repo or repo/path/to/model.gguf")
    parser.add_argument("--all", action="store_true", help="Download all registry-declared artifacts for a known model repo: target GGUF, multimodal projector, and draft MTP when present.")
    parser.add_argument("--mmproj", action="store_true", help="When spec is a repo, download the registry-declared multimodal projector instead of the target GGUF.")
    parser.add_argument("--models-dir", help="Override Orbit model cache directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv[1:] if argv and argv[0] == "download" else argv)
    return _download(args)


def _download(args: argparse.Namespace) -> int:
    models_dir = default_models_dir() if args.models_dir is None else Path(args.models_dir)
    try:
        if args.all:
            if args.mmproj:
                print("error: --all cannot be combined with --mmproj", file=sys.stderr)
                return 1
            batch = download_all_for_repo(args.spec, models_dir=models_dir)
            for result in batch.results:
                action = "downloaded" if result.downloaded else "already present"
                print(f"{action}: {result.path}")
            return 0
        result = download_model(args.spec, models_dir=models_dir, prefer="mmproj" if args.mmproj else "target")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    action = "downloaded" if result.downloaded else "already present"
    print(f"{action}: {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
