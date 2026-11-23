"""Build the optional Intel Vulkan variant from a hash-bound source overlay."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

from .build_support import PACKAGE_NATIVE_ROOT
from .llama_provenance import load_llama_provenance, source_tree_sha256


def prepare_source(root: Path, *, clean: bool = False) -> Path:
    base = root.parent / 'mainline_support' / 'source' / 'llama.cpp'
    source = root / 'source' / 'llama.cpp'
    overlay = json.loads((root / 'OVERLAY.json').read_text())
    provenance = load_llama_provenance(base)
    if provenance.source_tree_sha256 != overlay['base_source_tree_sha256']:
        raise RuntimeError('Intel Vulkan overlay does not match the CPU source identity')
    for item in overlay['files']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('invalid Vulkan overlay path')
        for path, key in ((base / rel, 'base_sha256'), (root / 'overlay' / rel, 'overlay_sha256')):
            if hashlib.sha256(path.read_bytes()).hexdigest() != item[key]:
                raise RuntimeError(f'Intel Vulkan overlay identity mismatch: {rel}')
    if clean and source.exists():
        shutil.rmtree(source)
    if not source.exists():
        shutil.copytree(base, source)
        for item in overlay['files']:
            shutil.copy2(root / 'overlay' / item['path'], source / item['path'])
    try:
        load_llama_provenance(source)
    except RuntimeError as error:
        raise RuntimeError("Intel Vulkan generated source provenance is stale or modified; rebuild with --clean") from error
    return source


def require_shader_compiler(path: str, source: Path) -> None:
    test = source / 'ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/integer_dot.comp'
    result = subprocess.run([path, '-o', '/dev/null', '-fshader-stage=compute',
                             '--target-env=vulkan1.3', str(test)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Intel GPU requires glslc with GL_EXT_integer_dot_product support; '
                           'select a newer compiler with --glslc. ' + result.stderr.strip())
