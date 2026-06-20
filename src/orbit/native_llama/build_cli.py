from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .build_support import BUNDLED_SOURCE_ROOT, DEFAULT_VENDOR_BUILD_BIN, DEFAULT_VENDOR_BUILD_ROOT, validate_llama_source_root
from .mtp_accept_probe import build_mtp_accept_probe_helper
from .mtp_completion import build_mtp_completion_helper
from .mtp_decode_probe import build_mtp_decode_probe_helper
from .mtp_dry_run import build_mtp_dry_run_helper
from .mtp_probe import build_mtp_probe_helper
from .native_names import platform_optional_runtime_libs, platform_runtime_libs
from .paths import DEFAULT_VENDOR_LIB_DIR, DEFAULT_VENDOR_SHIM_DIR
from .persistent_mtp import build_persistent_mtp_shim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbit build-native")
    parser.add_argument(
        "--llama-root",
        type=Path,
        help="Legacy alias for --source-dir. Use only as a developer override.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Optional developer override for the vendored llama.cpp source tree.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="Optional parallel build jobs passed to cmake --build.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing build directory before configuring.",
    )
    parser.add_argument(
        "--with-mtp-shim",
        action="store_true",
        help="Build packaged MTP helper binaries and shim artifacts under vendor/shim. This is included by default for a full native build.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root_or_error = _resolve_source_root(args.source_dir, args.llama_root)
    if isinstance(source_root_or_error, str):
        print(f"error: {source_root_or_error}", file=sys.stderr)
        return 1
    source_root = source_root_or_error
    cmake = shutil.which("cmake")
    if not cmake:
        print("error: cmake not found in PATH", file=sys.stderr)
        return 1

    build_dir = DEFAULT_VENDOR_BUILD_ROOT
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    DEFAULT_VENDOR_LIB_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_VENDOR_SHIM_DIR.mkdir(parents=True, exist_ok=True)

    configure_cmd = [
        cmake,
        "-S",
        str(source_root),
        "-B",
        str(build_dir),
        "-DBUILD_SHARED_LIBS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_COMMON=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_BUILD_UI=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_OPENSSL=OFF",
    ]
    build_cmd = [cmake, "--build", str(build_dir)]
    if args.jobs and args.jobs > 0:
        build_cmd.extend(["-j", str(args.jobs)])

    try:
        _run(configure_cmd)
        _run(build_cmd)
        _copy_runtime_libraries(DEFAULT_VENDOR_BUILD_BIN)
        _build_packaged_shims(source_root, DEFAULT_VENDOR_BUILD_BIN)
        missing = [name for name in platform_runtime_libs() if not (DEFAULT_VENDOR_LIB_DIR / name).exists()]
        if missing:
            print(
                "error: native build completed but packaged runtime libraries are missing: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 1
        print(f"native runtime built from {source_root}", flush=True)
        print(f"packaged runtime libraries: {DEFAULT_VENDOR_LIB_DIR}", flush=True)
        print(f"packaged MTP shims: {DEFAULT_VENDOR_SHIM_DIR}", flush=True)
        print("next: orbit server --port 11976", flush=True)
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _resolve_source_root(source_dir: Path | None, llama_root: Path | None) -> Path | str:
    explicit = source_dir or llama_root
    if explicit is not None:
        return validate_llama_source_root(explicit.expanduser().resolve())
    bundled = validate_llama_source_root(BUNDLED_SOURCE_ROOT)
    if isinstance(bundled, Path):
        return bundled
    return (
        "bundled llama.cpp sources are missing from Orbit: "
        f"{BUNDLED_SOURCE_ROOT}. "
        "Restore src/orbit/native_llama/vendor/source/llama.cpp "
        "or provide --source-dir as an explicit developer override."
    )


def _validate_llama_root(root: Path) -> Path | str:
    return validate_llama_source_root(root)


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(detail or f"command failed: {' '.join(command)}")


def _copy_runtime_libraries(build_bin: Path) -> None:
    for name in [*platform_runtime_libs(), *platform_optional_runtime_libs()]:
        source = build_bin / name
        if not source.exists():
            continue
        shutil.copy2(source, DEFAULT_VENDOR_LIB_DIR / name)


def _build_packaged_shims(source_root: Path, build_bin: Path) -> None:
    build_mtp_probe_helper(llama_root=source_root, build_dir=DEFAULT_VENDOR_SHIM_DIR, build_bin=build_bin)
    build_mtp_dry_run_helper(llama_root=source_root, build_dir=DEFAULT_VENDOR_SHIM_DIR, build_bin=build_bin)
    build_mtp_accept_probe_helper(llama_root=source_root, build_dir=DEFAULT_VENDOR_SHIM_DIR, build_bin=build_bin)
    build_mtp_decode_probe_helper(llama_root=source_root, build_dir=DEFAULT_VENDOR_SHIM_DIR, build_bin=build_bin)
    build_mtp_completion_helper(llama_root=source_root, build_dir=DEFAULT_VENDOR_SHIM_DIR, build_bin=build_bin)
    build_persistent_mtp_shim(llama_root=source_root, build_dir=DEFAULT_VENDOR_SHIM_DIR, build_bin=build_bin)
