from __future__ import annotations

from pathlib import Path
import os
import subprocess

from .native_names import platform_runtime_libs


PACKAGE_NATIVE_ROOT = Path(__file__).resolve().parent / "vendor"
BUNDLED_SOURCE_ROOT = PACKAGE_NATIVE_ROOT / "source" / "llama.cpp"

# What a built artifact records as the place to find its sibling libraries.
#
# `$ORIGIN` is expanded by the dynamic loader to the directory of the object
# holding the entry, so a runtime family resolves within itself wherever it is
# copied. An absolute build path would instead follow the copy: a second Orbit
# checkout on the same machine would load its own libllama while resolving
# libggml back into the checkout it was built in, mapping two families of
# llama/ggml into one address space.
#
# Written as a literal `$ORIGIN`, never expanded by Python or the shell: the
# loader needs the token itself in DT_RUNPATH.
ORIGIN_RUNPATH = "$ORIGIN"


def origin_relative_runpath(artifact_dir: Path, library_dir: Path) -> str:
    """`$ORIGIN`-relative search path from an artifact to its libraries.

    Helpers are not all built beside the runtime: the MTP shims live in
    `vendor/shim` while the libraries they link against live in the build
    directory. A bare `$ORIGIN` would point those at their own directory,
    which holds no libraries, so the entry has to carry the hop between the
    two -- expressed relatively, so that copying the tree keeps it valid.
    """
    resolved_library = library_dir.resolve()
    resolved_artifact = artifact_dir.resolve()
    if not _share_a_tree(resolved_artifact, resolved_library):
        # The artifact is being built outside the tree that holds the
        # libraries -- a build directory under the user's home, say. A
        # relative entry would then climb out of the artifact's own tree and
        # re-enter the other one by traversal, which is neither relocatable
        # nor meaningful once either side moves. An absolute entry is honest
        # about that: it names one specific runtime rather than pretending
        # the two travel together.
        return str(resolved_library)
    relative = os.path.relpath(resolved_library, resolved_artifact)
    if relative == ".":
        return ORIGIN_RUNPATH
    return f"{ORIGIN_RUNPATH}/{relative}"


def _share_a_tree(artifact_dir: Path, library_dir: Path) -> bool:
    """Whether both live under the package root, so a copy moves them together.

    `vendor/shim` and `vendor/lib` are siblings of `vendor/build`, so a hop
    between them survives relocation. A directory outside `vendor` does not
    travel with it, and a relative entry there would only describe where the
    two happened to sit at build time.
    """
    root = PACKAGE_NATIVE_ROOT.resolve()
    return artifact_dir.is_relative_to(root) and library_dir.is_relative_to(root)

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
            # Relative to the artifact, so a copied tree keeps resolving.
            f"-Wl,-rpath,{origin_relative_runpath(output.parent, resolved_bin)}",
            # Keep DT_RUNPATH rather than the older DT_RPATH: RUNPATH is what
            # the CMake-built libraries already carry, and unlike RPATH it does
            # not apply to a dependency's own dependencies, so each library
            # states where its siblings are instead of inheriting a parent's
            # answer.
            "-Wl,--enable-new-dtags",
        ]
    )
    command.extend(f"-I{path}" for path in extra_include_dirs)
    command.extend(str(resolved_bin / name) for name in platform_runtime_libs())
    command.extend(extra_link_args)
    command.extend(["-o", str(output)])

    if force:
        # Remove the previous artifact before compiling. Otherwise a toolchain
        # that exits 0 without writing anything leaves the stale file in place,
        # and every check below -- exists, is_dir, size -- is satisfied by that
        # stale file, so it is returned as fresh output. Deleting first makes
        # "the artifact is present afterwards" mean the compiler produced it.
        # This also covers a truncated write, which a size check alone admits.
        #
        # Accepted trade: a forced build that then FAILS leaves no artifact,
        # where previously the stale one survived. That is the point -- keeping
        # it is the defect being fixed -- but it does mean a failed explicit
        # build leaves the tree without a loadable shim until the source is
        # fixed and rebuilt. Only `force` callers are affected, i.e. only the
        # explicit build path; the runtime never unlinks.
        output.unlink(missing_ok=True)

    completed = runner(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"failed to build {artifact_label}: {detail or completed.returncode}")
    # A zero return code is not proof the artifact was written. A compiler
    # wrapper (ccache/distcc misconfiguration, a shim script, an output path on
    # a full or read-only filesystem) can exit 0 having produced nothing, and
    # returning `output` regardless makes callers believe a stale or absent file
    # is a fresh build. Verify what the caller was promised.
    if not output.exists():
        raise RuntimeError(
            f"failed to build {artifact_label}: the compiler reported success "
            f"but produced no output at {output}"
        )
    if output.is_dir() or output.stat().st_size == 0:
        raise RuntimeError(
            f"failed to build {artifact_label}: {output} is not a usable artifact"
        )
    return output
