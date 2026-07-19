from __future__ import annotations

import sys


def shared_library_suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def runtime_library_filename(stem: str) -> str:
    return f"lib{stem}{shared_library_suffix()}"


def platform_runtime_libs() -> tuple[str, ...]:
    return tuple(runtime_library_filename(stem) for stem in ("llama", "llama-common", "ggml", "ggml-base", "ggml-cpu"))


def platform_optional_runtime_libs() -> tuple[str, ...]:
    return (runtime_library_filename("mtmd"),)


def persistent_mtp_shim_filename() -> str:
    return f"liborbit-persistent-mtp{shared_library_suffix()}"


def mtmd_bridge_filename() -> str:
    return runtime_library_filename("orbit-mtmd-bridge")
