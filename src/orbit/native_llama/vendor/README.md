# Native backend vendor area

This directory is the packaging boundary for native Orbit backend artifacts.

Current status:

- this boundary exists
- Orbit model download/bootstrap already uses Orbit-owned Python code
- native runtime libraries and shim binaries are not yet shipped here in the normal product path
- some native MTP paths still build helper artifacts locally

Expected future contents:

- `lib/`: packaged `libllama`, `libllama-common`, and `ggml` runtime libraries.
- `include/`: pinned public headers needed by the Orbit shim.
- `shim/`: Orbit-owned C/C++ shim source or built extension.

It is intentionally empty in this phase: no native libraries are committed and
no runtime code loads from this directory yet.

The roadmap for turning this directory into the real product boundary lives in
[docs/NATIVE_PACKAGING_ROADMAP.md](../../../docs/NATIVE_PACKAGING_ROADMAP.md).
