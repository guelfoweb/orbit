from __future__ import annotations

from pathlib import Path
import os
import subprocess

from .native_names import platform_runtime_libs


PACKAGE_NATIVE_ROOT = Path(__file__).resolve().parent / "vendor"
BUNDLED_SOURCE_ROOT = PACKAGE_NATIVE_ROOT / "source" / "llama.cpp"
DEFAULT_VENDOR_BUILD_ROOT = PACKAGE_NATIVE_ROOT / "build" / "llama.cpp"
DEFAULT_VENDOR_BUILD_BIN = DEFAULT_VENDOR_BUILD_ROOT / "bin"


def validate_llama_source_root(root: Path) -> Path | str:
    if not root.exists():
        return f"llama source tree not found: {root}"
    if not root.is_dir():
        return f"llama source tree is not a directory: {root}"
    if not (root / "CMakeLists.txt").exists():
        return f"llama source tree does not look like a llama.cpp checkout: {root}"
    return root


def resolve_build_bin(*, llama_root: Path, build_bin: Path | None = None) -> Path:
    if build_bin is not None:
        return build_bin.expanduser().resolve()
    return llama_root.expanduser().resolve() / "build" / "bin"


def compile_cpp_helper(
    *,
    artifact_label: str,
    source: Path,
    output: Path,
    llama_root: Path,
    build_bin: Path | None = None,
    runner=subprocess.run,
    shared: bool = False,
    extra_compile_args: tuple[str, ...] = (),
    extra_include_dirs: tuple[Path, ...] = (),
    extra_link_args: tuple[str, ...] = (),
    force: bool = False,
) -> Path:
    resolved_root = llama_root.expanduser().resolve()
    resolved_bin = resolve_build_bin(llama_root=resolved_root, build_bin=build_bin)
    output.parent.mkdir(parents=True, exist_ok=True)
    dependency_paths = [
        source,
        resolved_root / "include" / "llama.h",
        resolved_root / "common" / "common.h",
        resolved_root / "common" / "speculative.h",
        resolved_root / "tools" / "mtmd" / "mtmd.h",
        resolved_root / "tools" / "mtmd" / "mtmd-helper.h",
        *(resolved_bin / name for name in platform_runtime_libs()),
        *(Path(arg) for arg in extra_link_args if Path(arg).is_absolute()),
    ]
    newest_input = max(
        (path.stat().st_mtime for path in dependency_paths if path.exists()),
        default=source.stat().st_mtime,
    )
    if not force and output.exists() and output.stat().st_mtime >= newest_input:
        return output

    command = [os.environ.get("CXX", "c++"), "-std=c++17"]
    if shared:
        command.extend(["-shared", "-fPIC"])
    command.extend(extra_compile_args)
    command.extend(
        [
            str(source),
            f"-I{resolved_root / 'include'}",
            f"-I{resolved_root / 'common'}",
            f"-I{resolved_root}",
            f"-I{resolved_root / 'ggml/include'}",
            f"-I{resolved_root / 'src'}",
            f"-Wl,-rpath,{resolved_bin}",
        ]
    )
    command.extend(f"-I{path}" for path in extra_include_dirs)
    command.extend(str(resolved_bin / name) for name in platform_runtime_libs())
    command.extend(extra_link_args)
    command.extend(["-o", str(output)])

    completed = runner(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"failed to build {artifact_label}: {detail or completed.returncode}")
    return output
