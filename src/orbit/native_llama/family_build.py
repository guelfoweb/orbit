"""Build an independently packaged CPU/text family without touching legacy."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

from .build_support import ORIGIN_RUNPATH
from .chat_bridge import build_chat_bridge, install_chat_bridge
from .mtmd_bridge import build_mtmd_bridge
from .native_families import native_family
from .native_names import platform_optional_runtime_libs, platform_runtime_libs


CPU_DISABLED_BACKENDS = ("CUDA", "HIP", "MUSA", "VULKAN", "METAL", "SYCL", "OPENCL",
                         "CANN", "RPC", "HEXAGON", "OPENVINO", "BLAS", "ZENDNN",
                         "ET", "ET_SYSEMU", "VIRTGPU_BACKEND", "WEBGPU", "BACKEND_DL")


def _verify_cpu_cache(path: Path) -> None:
    entries = {}
    for line in path.read_text().splitlines():
        if line.startswith(("#", "//")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        entries[key.split(":", 1)[0]] = value.upper()
    if any(entries.get("GGML_" + name) not in {"OFF", "FALSE", "NO", "0"}
           for name in CPU_DISABLED_BACKENDS):
        raise RuntimeError("isolated CPU family has unverified accelerator configuration")


def build_family(args) -> int:
    from .build_cli import _cmake_provenance_args, _copy_soname_aliases
    if args.source_dir or args.llama_root or args.with_mtp_shim:
        print("error: isolated CPU/text family does not accept source overrides or MTP", file=sys.stderr)
        return 1
    if args.verbose and args.quiet:
        print("error: --verbose and --quiet cannot be used together", file=sys.stderr)
        return 1
    family = native_family(args.family)
    vulkan = family.name == "mainline-vulkan"
    source = family.root / "source" / "llama.cpp"
    build = family.root / "build" / "llama.cpp"
    destination = family.root / "lib"
    try:
        if vulkan:
            from .vulkan_build import prepare_source, require_shader_compiler
            source = prepare_source(family.root, clean=args.clean)
            compiler = getattr(args, 'glslc', None) or shutil.which('glslc')
            if not compiler:
                raise RuntimeError('Intel GPU build requires glslc; select it with --glslc')
            require_shader_compiler(compiler, source)
        provenance = _cmake_provenance_args(source)
        if args.clean and build.exists():
            shutil.rmtree(build)
        destination.mkdir(parents=True, exist_ok=True)
        command = ["cmake", "-S", str(source), "-B", str(build),
                   "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
                   "-DLLAMA_BUILD_COMMON=ON", "-DLLAMA_BUILD_TESTS=OFF",
                   "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF",
                   "-DLLAMA_BUILD_APP=OFF", "-DLLAMA_BUILD_TOOLS=ON",
                   "-DLLAMA_OPENSSL=OFF", "-DGGML_CPU=ON",
                   *[f"-DGGML_{backend}={'ON' if vulkan and backend == 'VULKAN' else 'OFF'}" for backend in CPU_DISABLED_BACKENDS],
                   f"-DCMAKE_BUILD_RPATH={ORIGIN_RUNPATH}",
                   f"-DCMAKE_INSTALL_RPATH={ORIGIN_RUNPATH}",
                   "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", *provenance]
        if vulkan:
            command += [f"-DVulkan_GLSLC_EXECUTABLE={compiler}"]
        subprocess.run(command, check=True)
        if vulkan:
            cache = (build / "CMakeCache.txt").read_text()
            if "GGML_VULKAN:BOOL=ON" not in cache:
                raise RuntimeError("Vulkan backend was not enabled")
            # Verify every other accelerator remains disabled.
            checked = build / "cpu-only-check.txt"
            checked.write_text(cache.replace("GGML_VULKAN:BOOL=ON", "GGML_VULKAN:BOOL=OFF"))
            _verify_cpu_cache(checked)
        else:
            _verify_cpu_cache(build / "CMakeCache.txt")
        jobs = args.jobs if args.jobs is not None else min(4, os.cpu_count() or 1)
        if jobs <= 0:
            raise ValueError("build jobs must be positive")
        subprocess.run(["cmake", "--build", str(build), "--parallel", str(jobs)], check=True)
        binaries = build / "bin"
        for name in (*platform_runtime_libs(), *platform_optional_runtime_libs(), *(("libggml-vulkan.so",) if vulkan else ())):
            library = binaries / name
            if library.exists():
                shutil.copy2(library, destination / name)
                _copy_soname_aliases(library, destination)
        shim_root = native_family("mainline-support").root if vulkan else family.root
        for builder, filename in ((build_chat_bridge, "orbit_chat_bridge.cpp"),
                                  (build_mtmd_bridge, "orbit_mtmd_bridge.cpp")):
            artifact, _ = builder(llama_root=source, build_dir=binaries, build_bin=binaries,
                                  shim_source=shim_root / "shim" / filename)
            install_chat_bridge(artifact, destination)
        print(f"native CPU/text family ready: {family.name}")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
