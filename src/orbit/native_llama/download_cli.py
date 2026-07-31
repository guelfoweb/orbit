from __future__ import annotations

import argparse
from pathlib import Path
import sys

from orbit.native_llama.model_download import download_all_for_repo, download_model
from orbit.native_llama.model_registry import default_models_dir, get_manifest
from orbit.native_llama.paths import DEFAULT_MODEL_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbit download")
    parser.add_argument("spec", nargs="?", help="Hugging Face repo or repo/path/to/model.gguf")
    parser.add_argument("--all", action="store_true", help="Download all registry-declared artifacts for the default native model when no repo is provided: target GGUF, multimodal projector, and draft MTP when present.")
    parser.add_argument("--mmproj", action="store_true", help="When spec is a repo, download the registry-declared multimodal projector instead of the target GGUF.")
    parser.add_argument("--models-dir", help="Override Orbit model cache directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv[1:] if argv and argv[0] == "download" else argv)
    return _download(args)


def _download(args: argparse.Namespace) -> int:
    models_dir = default_models_dir() if args.models_dir is None else Path(args.models_dir)
    progress = _DownloadProgress()
    try:
        if args.all:
            if args.mmproj:
                print("error: --all cannot be combined with --mmproj", file=sys.stderr)
                return 1
            repo = args.spec or get_manifest(DEFAULT_MODEL_ID).target.repo
            batch = download_all_for_repo(repo, models_dir=models_dir, progress=progress)
            progress.finish()
            for result in batch.results:
                action = "downloaded" if result.downloaded else "already present"
                print(f"{action}: {result.path}")
            return 0
        if not args.spec:
            print("error: expected Hugging Face repo or repo/file", file=sys.stderr)
            return 1
        result = download_model(
            args.spec,
            models_dir=models_dir,
            prefer="mmproj" if args.mmproj else "target",
            progress=progress,
        )
        progress.finish()
    except Exception as exc:
        progress.finish()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    action = "downloaded" if result.downloaded else "already present"
    print(f"{action}: {result.path}")
    return 0


class _DownloadProgress:
    def __init__(self) -> None:
        self._active = False
        self._last_percent: int | None = None

    def __call__(self, downloaded: int, total: int) -> None:
        if total <= 0:
            return
        percent = min(100, max(0, int(downloaded * 100 / total)))
        if percent == self._last_percent:
            return
        if self._active and self._last_percent == 100 and percent < 100:
            print()
        self._active = True
        self._last_percent = percent
        print(f"\rdownload: {percent:3d}%", end="", flush=True)

    def finish(self) -> None:
        if self._active:
            print()
        self._active = False
        self._last_percent = None


if __name__ == "__main__":
    raise SystemExit(main())
