from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .mtp_accept_probe import build_mtp_accept_probe_helper
from .mtp_completion import build_mtp_completion_helper
from .mtp_decode_probe import build_mtp_decode_probe_helper
from .mtp_dry_run import build_mtp_dry_run_helper
from .mtp_probe import build_mtp_probe_helper
from .native_artifacts import LINUX_RUNTIME_LIBS
from .paths import DEFAULT_VENDOR_LIB_DIR, DEFAULT_VENDOR_SHIM_DIR
from .persistent_mtp import build_persistent_mtp_shim

BUNDLED_SOURCE_ROOT = Path(__file__).resolve().parent / "vendor" / "source" / "llama.cpp"
DEFAULT_BUILD_ROOT = BUNDLED_SOURCE_ROOT / "build"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbit build-native")
    parser.add_argument(
        "--llama-root",
        type=Path,
        help="Optional override source tree. Defaults to Orbit's bundled llama.cpp sources.",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    llama_root = _resolve_llama_root(args.llama_root)
    if isinstance(llama_root, str):
        print(f"error: {llama_root}", file=sys.stderr)
        return 1
    cmake = shutil.which("cmake")
    if not cmake:
        print("error: cmake not found in PATH", file=sys.stderr)
        return 1

    build_dir = DEFAULT_BUILD_ROOT
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    DEFAULT_VENDOR_LIB_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_VENDOR_SHIM_DIR.mkdir(parents=True, exist_ok=True)

    configure_cmd = [
        cmake,
        "-S",
        str(llama_root),
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
        _copy_runtime_libraries(build_dir / "bin")
        _build_packaged_shims(llama_root)
        missing = [name for name in LINUX_RUNTIME_LIBS if not (DEFAULT_VENDOR_LIB_DIR / name).exists()]
        if missing:
            print(
                "error: native build completed but packaged runtime libraries are missing: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 1
        print(f"native runtime built from {llama_root}", flush=True)
        print(f"packaged runtime libraries: {DEFAULT_VENDOR_LIB_DIR}", flush=True)
        print(f"packaged MTP shims: {DEFAULT_VENDOR_SHIM_DIR}", flush=True)
        print("next: orbit server --port 11976 --mtp", flush=True)
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _resolve_llama_root(explicit: Path | None) -> Path | str:
    if explicit is not None:
        return _validate_llama_root(explicit.expanduser().resolve())
    bundled = _validate_llama_root(BUNDLED_SOURCE_ROOT)
    if isinstance(bundled, Path):
        return bundled
    return (
        "bundled llama.cpp sources are missing from Orbit: "
        f"{BUNDLED_SOURCE_ROOT}. "
        "Restore src/orbit/native_llama/vendor/source/llama.cpp "
        "or provide --llama-root only as an explicit rollback override."
    )


def _validate_llama_root(root: Path) -> Path | str:
    if not root.exists():
        return f"llama_root not found: {root}"
    if not root.is_dir():
        return f"llama_root is not a directory: {root}"
    if not (root / "CMakeLists.txt").exists():
        return f"llama_root does not look like a llama.cpp source tree: {root}"
    return root


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(detail or f"command failed: {' '.join(command)}")


def _copy_runtime_libraries(build_bin: Path) -> None:
    for name in [*LINUX_RUNTIME_LIBS, "libmtmd.so"]:
        source = build_bin / name
        if not source.exists():
            continue
        shutil.copy2(source, DEFAULT_VENDOR_LIB_DIR / name)


def _build_packaged_shims(llama_root: Path) -> None:
    build_mtp_probe_helper(llama_root=llama_root, build_dir=DEFAULT_VENDOR_SHIM_DIR)
    build_mtp_dry_run_helper(llama_root=llama_root, build_dir=DEFAULT_VENDOR_SHIM_DIR)
    build_mtp_accept_probe_helper(llama_root=llama_root, build_dir=DEFAULT_VENDOR_SHIM_DIR)
    build_mtp_decode_probe_helper(llama_root=llama_root, build_dir=DEFAULT_VENDOR_SHIM_DIR)
    build_mtp_completion_helper(llama_root=llama_root, build_dir=DEFAULT_VENDOR_SHIM_DIR)
    build_persistent_mtp_shim(llama_root=llama_root, build_dir=DEFAULT_VENDOR_SHIM_DIR)
