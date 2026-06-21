# Native runtime libraries

This directory is reserved for packaged native runtime libraries shipped by Orbit.

The first self-contained native packaging target is Linux.
macOS may be addressed later with separate artifacts.
Windows is not part of this first packaged native backend cut.

Expected future contents include platform-specific builds such as:

- `libllama.so`
- `libllama-common.so`
- `libggml.so`
- `libggml-base.so`
- `libggml-cpu.so`
- `libmtmd.so` when multimodal support is packaged

Orbit now prefers this directory before any legacy external `llama_root` path.

These outputs must be treated as generated artifacts:

- do not commit `.so`, `.dylib`, `.a`, or build directories
- build them explicitly with `python scripts/build_native.py`
- keep `--llama-root` / `ORBIT_LLAMA_ROOT` only as developer compatibility fallbacks
