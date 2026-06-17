# Native backend vendor area

This directory is the packaging boundary for native Orbit backend artifacts.

Expected future contents:

- `lib/`: packaged `libllama`, `libllama-common`, and `ggml` runtime libraries.
- `include/`: pinned public headers needed by the Orbit shim.
- `shim/`: Orbit-owned C/C++ shim source or built extension.

It is intentionally empty in this phase: no native libraries are committed and
no runtime code loads from this directory yet.
