# Native mtmd ABI Boundary

Orbit isolates the unstable upstream mtmd C ABI behind a revision-bound native
bridge:

```text
Python runtime
  -> liborbit-mtmd-bridge (primitive values and opaque handles)
  -> matching libmtmd/libllama build
```

Python does not declare `mtmd_context_params`, `mtmd_input_text`, bitmap return
wrappers, or capability structures. The bridge constructs these values using
the headers from the native revision being built and explicitly initializes
every field used by Orbit. The bridge currently recognizes the reviewed mtmd
context/input/bitmap ABI profiles used by the bundled vendor and the separately
tested upstream candidate. Unknown layouts fail before mmproj initialization.

## Build Identity

The bridge is built into the same directory as its native libraries. Its
sidecar identity records:

- compiler and compiler version;
- bridge flags and relevant CMake configuration;
- hashes of the bridge source and relevant llama/mtmd headers;
- hashes of every co-located runtime library;
- upstream tag and commit;
- canonical source-tree and Orbit patchset hashes.

The bridge artifact and every required runtime library are hashed again before
the bridge is loaded. A missing sidecar, changed artifact, changed library, or
provenance mismatch is a controlled error. Build reuse is forbidden when any
identity input changes.

The bundled source provenance is stored in
`src/orbit/native_llama/vendor/LLAMA_PROVENANCE.json`. CMake receives the
manifest commit and build number explicitly; it must not derive
`LLAMA_COMMIT` from the enclosing Orbit repository.

## Current Vendor

- Upstream tag: `b9551`
- Upstream commit: `379ac6673b5cd75c7b4e07d1521c50f1e093878c`
- Canonical source-tree SHA-256:
  `4adb967e643363e7dc4d01d632b3a8471e0df2ec84ff304d364dc182f63e7ee1`
- Orbit patchset SHA-256:
  `dea2f205ed2a73d09ad203e08ba85545474742dbb0191f4f1a9b3a86beb4b435`

This hardening does not update the production llama.cpp revision. A future
vendor upgrade must rebuild the complete native runtime and bridge together,
then repeat tokenizer, renderer, mmproj, MTP, lifecycle, final-prefix, and
process-isolated conformance gates.
