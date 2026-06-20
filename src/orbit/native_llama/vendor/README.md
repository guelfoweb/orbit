# Native backend vendor area

This directory is the packaging boundary for native Orbit backend artifacts.

Current status:

- Orbit model download/bootstrap already uses Orbit-owned Python code
- Orbit can build native runtime libraries from vendored `llama.cpp` sources with an explicit build step
- `vendor/lib/` is the first runtime lookup location for native libraries
- some optional MTP shim paths may still require an explicit shim build step

Expected future contents:

- `source/llama.cpp/`: vendored upstream native sources used for the local self-build path.
- `build/llama.cpp/`: local CMake build output, gitignored.
- `lib/`: locally built or packaged `libllama`, `libllama-common`, and `ggml` runtime libraries.
- `shim/`: Orbit-owned C/C++ shim source and optional built helper artifacts.

Orbit does not require an external `llama-server` process at runtime. It still
depends on native libraries derived from `llama.cpp`/`ggml`, either from an
explicit local build under this boundary or from documented fallback paths such
as `--llama-root` or `ORBIT_LLAMA_ROOT`.

The roadmap for turning this directory into the real product boundary lives in
[docs/NATIVE_PACKAGING_ROADMAP.md](../../../docs/NATIVE_PACKAGING_ROADMAP.md).
