from __future__ import annotations

import argparse
import collections
import queue
import shutil
import subprocess
import sys
import threading
import time
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream full configure/build command output.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output to errors and final success summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose and args.quiet:
        print("error: --verbose and --quiet cannot be used together", file=sys.stderr)
        return 1
    source_root_or_error = _resolve_source_root(args.source_dir, args.llama_root)
    if isinstance(source_root_or_error, str):
        print(f"error: {source_root_or_error}", file=sys.stderr)
        return 1
    source_root = source_root_or_error
    reporter = BuildReporter(verbose=args.verbose, quiet=args.quiet)
    started_at = time.monotonic()
    reporter.phase("checking toolchain")
    cmake = shutil.which("cmake")
    if not cmake:
        print("error: cmake not found in PATH", file=sys.stderr)
        return 1

    build_dir = DEFAULT_VENDOR_BUILD_ROOT
    reporter.phase("preparing source/build dirs")
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
        reporter.phase("configuring CMake")
        _run(configure_cmd, reporter=reporter, heartbeat_label="configuring CMake")
        reporter.phase("building native libraries")
        _run(build_cmd, reporter=reporter, heartbeat_label="building native libraries")
        reporter.phase("copying/verifying libraries")
        _copy_runtime_libraries(DEFAULT_VENDOR_BUILD_BIN)
        _build_packaged_shims(source_root, DEFAULT_VENDOR_BUILD_BIN)
        missing = [name for name in platform_runtime_libs() if not (DEFAULT_VENDOR_LIB_DIR / name).exists()]
        if missing:
            print(
                "error: native build completed but packaged runtime libraries are missing: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 1
        duration = _format_elapsed(time.monotonic() - started_at)
        print(f"native runtime built from {source_root}", flush=True)
        print(f"packaged runtime libraries: {DEFAULT_VENDOR_LIB_DIR}", flush=True)
        print(f"packaged MTP shims: {DEFAULT_VENDOR_SHIM_DIR}", flush=True)
        print(f"verified runtime libraries: {', '.join(platform_runtime_libs())}", flush=True)
        print(f"completed in {duration}", flush=True)
        print("next: orbit server --port 12120", flush=True)
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


class BuildReporter:
    def __init__(self, *, verbose: bool, quiet: bool) -> None:
        self.verbose = verbose
        self.quiet = quiet

    def phase(self, name: str) -> None:
        if self.quiet:
            return
        print(f"==> {name}", flush=True)

    def heartbeat(self, label: str, elapsed_seconds: float) -> None:
        if self.quiet or self.verbose:
            return
        print(f"... still {label} ({int(elapsed_seconds)}s elapsed)", flush=True)

    def line(self, text: str) -> None:
        if self.quiet:
            return
        if self.verbose or _is_important_build_line(text):
            print(text, flush=True)


def _run(command: list[str], *, reporter: BuildReporter, heartbeat_label: str) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    output_queue: queue.Queue[str | None] = queue.Queue()
    output_tail: collections.deque[str] = collections.deque(maxlen=20)
    reader = threading.Thread(target=_enqueue_process_output, args=(process.stdout, output_queue), daemon=True)
    reader.start()
    start = time.monotonic()
    last_heartbeat_at = start

    while True:
        try:
            line = output_queue.get(timeout=1.0)
        except queue.Empty:
            line = None
            now = time.monotonic()
            if process.poll() is None and now - last_heartbeat_at >= 20:
                reporter.heartbeat(heartbeat_label, now - start)
                last_heartbeat_at = now
        if isinstance(line, str):
            text = line.rstrip()
            output_tail.append(text)
            reporter.line(text)
            continue
        if line is None and process.poll() is not None:
            break

    returncode = process.wait()
    reader.join(timeout=1.0)
    if returncode != 0:
        tail = "\n".join(entry for entry in output_tail if entry)
        detail = tail or f"command failed: {' '.join(command)}"
        raise RuntimeError(
            f"command failed with exit code {returncode}: {' '.join(command)}"
            + (f"\nlast output:\n{detail}" if detail else "")
        )


def _enqueue_process_output(stream, output_queue: queue.Queue[str | None]) -> None:
    try:
        for line in iter(stream.readline, ""):
            output_queue.put(line)
    finally:
        output_queue.put(None)


def _is_important_build_line(text: str) -> bool:
    lowered = text.lower()
    return (
        "built target" in lowered
        or lowered.startswith("-- configuring done")
        or lowered.startswith("-- generating done")
        or lowered.startswith("-- build files have been written")
        or "error:" in lowered
        or "warning:" in lowered
    )


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


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
