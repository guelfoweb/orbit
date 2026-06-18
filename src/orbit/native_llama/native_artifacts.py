from __future__ import annotations

from pathlib import Path

from .paths import DEFAULT_VENDOR_LIB_DIR, DEFAULT_VENDOR_SHIM_DIR


LINUX_RUNTIME_LIBS = (
    "libllama.so",
    "libllama-common.so",
    "libggml.so",
    "libggml-base.so",
    "libggml-cpu.so",
)

OPTIONAL_RUNTIME_LIBS = (
    "libmtmd.so",
)

SHIM_ARTIFACTS = (
    "orbit-mtp-probe",
    "orbit-mtp-dry-run",
    "orbit-mtp-accept-probe",
    "orbit-mtp-decode-probe",
    "orbit-mtp-completion",
    "liborbit-persistent-mtp.so",
)


def packaged_runtime_lib_path(name: str) -> Path | None:
    candidate = DEFAULT_VENDOR_LIB_DIR / name
    if candidate.exists():
        return candidate
    return None


def packaged_shim_path(name: str) -> Path | None:
    candidate = DEFAULT_VENDOR_SHIM_DIR / name
    if candidate.exists():
        return candidate
    return None


def require_legacy_llama_root(llama_root: Path | None, *, artifact_name: str) -> Path:
    if llama_root is None:
        raise RuntimeError(
            f"missing native build inputs for {artifact_name}: "
            "no packaged shim artifact is available and no legacy llama_root was provided. "
            "Provide --llama-root or ORBIT_LLAMA_ROOT, or package the shim under orbit/native_llama/vendor/shim."
        )
    return llama_root
