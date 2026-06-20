# Native Packaging Roadmap

Orbit already owns:

- model download and local `models/` cache
- model registry and `model_id` resolution
- native `orbit-server`
- native MTP path
- optional multimodal projector support

Orbit does **not** yet own native binary distribution end to end.

Today, a fresh clone still depends on:

- a prepared `llama.cpp` build tree with `libllama.so`, `libllama-common.so`, and `ggml` libraries
- local native helper or shim compilation for some MTP paths

That means Orbit does not require an external `llama-server` runtime process, but it is not yet a packaged no-prerequisite product for a new user.

## Current bootstrap contract

Today, `orbit server` can start in three ways:

1. packaged native runtime libraries already exist under `src/orbit/native_llama/vendor/lib`
2. `ORBIT_LLAMA_ROOT` points to a prepared local `llama.cpp` tree
3. the user passes `--llama-root /path/to/llama.cpp`

For MTP paths, Orbit also needs either:

- a packaged shim under `src/orbit/native_llama/vendor/shim`
- or a buildable local `llama.cpp` tree so the shim can be rebuilt explicitly

If these prerequisites are missing, the release path should fail with a short actionable error, not a Python stacktrace.

## Target UX

The intended product path is:

```bash
pip install orbit
orbit download --all
orbit server --port 11976 --mtp
orbit
```

Current product default:

- `orbit server` starts the native backend with MTP off.
- `orbit server --mtp` enables the native MTP path explicitly.
- persistent multi-turn raw MTP chat reuse is not default and remains debug-only.

No external `llama.cpp` checkout.
No manual CMake build.
No `llama-server` install.
No hardcoded local paths.

## Current blockers

### 1. Native runtime libraries are not packaged by Orbit

The native loader still expects prebuilt shared libraries such as:

- `libllama.so`
- `libllama-common.so`
- `libggml.so`
- `libggml-base.so`
- `libggml-cpu.so`
- `libmtmd.so` when multimodal is enabled

These are currently resolved from an external build tree.

### 2. Orbit-owned C/C++ shims are not shipped as product artifacts

Persistent MTP and probe helpers still rely on local native compilation paths.

For product use, Orbit should ship:

- prebuilt shim binaries per supported platform
- or a tightly controlled build step during packaging, not ad hoc at runtime

### 3. The native loader still assumes an external build layout

The current path resolution still uses the concept of a `llama_root/build/bin` layout.

The product path should instead resolve:

- packaged Orbit native libs
- packaged Orbit shim artifacts
- downloaded model artifacts from the Orbit model cache

### 4. Release packaging policy is not finalized

Orbit still needs a concrete distribution policy for:

- Linux target builds
- wheel contents
- platform-specific native assets
- version pinning between Python package, native libs, and shim ABI

## Recommended implementation order

### Milestone 1. Internal native artifact layout

Goal:
- make Orbit load native assets from its own package boundary first

Deliverables:
- `src/orbit/native_llama/vendor/lib/` loader path support
- `src/orbit/native_llama/vendor/include/` pinned header boundary
- shim lookup from Orbit-owned paths before any legacy path
- clear runtime error if packaged native assets are missing

Acceptance:
- no runtime code requires a hardcoded developer-local `llama_root` path when packaged assets exist

### Milestone 2. Packaged shim artifacts

Goal:
- stop compiling MTP shims ad hoc on normal user startup

Deliverables:
- prebuilt packaged shim artifacts for supported Linux target
- explicit fallback behavior when a shim is unavailable
- runtime metadata showing whether packaged or legacy shim mode is active

Acceptance:
- normal `orbit server --mtp` startup does not invoke a local compiler on a prepared packaged install

### Milestone 3. Packaged native libs

Goal:
- make the native backend start from Orbit-owned packaged artifacts instead of requiring a separate local native build tree

Deliverables:
- packaged `libllama`, `libllama-common`, `ggml`, and `mtmd` runtime libraries
- stable loader path logic
- version compatibility check between Python package and native artifacts

Acceptance:
- a new user can install Orbit and start `orbit-server` without a manual `llama.cpp` build

### Artifact contract for the first Linux product cut

Expected packaged runtime libraries under `src/orbit/native_llama/vendor/lib/`:

- `libllama.so`
- `libllama-common.so`
- `libggml.so`
- `libggml-base.so`
- `libggml-cpu.so`
- optional: `libmtmd.so`

Expected packaged shim artifacts under `src/orbit/native_llama/vendor/shim/`:

- `orbit-mtp-probe`
- `orbit-mtp-dry-run`
- `orbit-mtp-accept-probe`
- `orbit-mtp-decode-probe`
- `orbit-mtp-completion`
- `liborbit-persistent-mtp.so`

### Naming and versioning policy

- Orbit Python package version and packaged native artifacts must be released as one tested set.
- Orbit should treat packaged native libs and packaged shim binaries as ABI-coupled artifacts.
- The first stable product cut should target Linux only.
- If packaged libs are present, Orbit should prefer them over any external `llama_root`.
- `--llama-root` and `ORBIT_LLAMA_ROOT` remain rollback compatibility, not the primary product path.

### Milestone 4. Product bootstrap

Goal:
- make `orbit download --all` + `orbit server` the default user path

Deliverables:
- documented default `model_id`
- clear target/draft/mmproj fallback reporting
- startup diagnostics that explain missing draft or missing multimodal projector without failing the base target model path

Acceptance:
- `orbit download --all`
- `orbit server --port 11976`
- `orbit server --port 11976 --mtp`

all work with no manual path entry on a supported packaged install.

## Non-goals for this packaging phase

- changing the runtime tool loop
- changing prompts or final answer policy
- speculative decoding redesign
- performance tuning by itself
- adding model-specific fast paths

## Risk notes

- Native ABI drift is the main packaging risk.
- Shipping mismatched `llama` and shim binaries is worse than keeping the current explicit dependency.
- Linux should remain the first supported packaging target.
- macOS should be treated as follow-up, not as part of the first fully packaged native product milestone.

## Release criterion for “Orbit is autonomous”

Orbit can be called autonomous from `llama-server` only when all of these are true:

1. `orbit-server` is the default documented backend path.
2. No external `llama-server` install is required.
3. No external `llama.cpp` checkout/build is required for normal usage.
4. Native libs and shims are shipped or installed by Orbit itself.
5. `orbit download --all` is sufficient to fetch the model-side artifacts.
