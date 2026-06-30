# orbit

Minimal local CLI for running Gemma 4 with the native `orbit server` backend.

Orbit is designed for local execution, streaming output, shell tools, and a simple terminal workflow. The normal setup uses Orbit's native backend and does not require an external `llama-server` process at runtime.

The native backend still depends on native libraries derived from `llama.cpp`/`ggml`, built either from Orbit's vendored sources or from a documented developer fallback such as `--llama-root`. Zero-build packaging remains future work. See [docs/NATIVE_PACKAGING_ROADMAP.md](docs/NATIVE_PACKAGING_ROADMAP.md).

Linux is the main target environment. macOS may work. Windows is not a target environment.

## What it does

- chats with a local model through a small terminal CLI
- streams answers and runtime metrics
- exposes unrestricted local shell tools by default, with explicit `/tools off`, `--tools off`, or `ORBIT_TOOLS=off` controls
- keeps the tool loop model-driven
- supports local image and audio input when the backend is started with multimodal support
- supports native MTP when target and draft models are available
- stores lightweight sessions and prompt history under `~/.orbit`

Orbit stays model-driven. The runtime enforces safety, size, timeout, and tool-contract boundaries, but it does not deterministically solve user tasks.

## Backend stance

- Primary backend path: native `orbit server`
- Compatibility path: `llama-server` or another OpenAI-compatible local backend
- CLI default base URL: `http://127.0.0.1:12120`

If your native server runs on another port, pass `--base-url`.

## CPU-first native build

Orbit is designed, tested, and supported primarily for CPU-only local execution.

The vendored native self-build path is CPU-only in this release:

```bash
python3 scripts/build_native.py
```

GPU acceleration is not part of Orbit's supported vendored self-build path yet. Advanced users who want CUDA, Metal, Vulkan, or ROCm should build `llama.cpp` natively on the target GPU machine and point Orbit to that build through the documented developer fallback:

```bash
orbit server --llama-root /path/to/gpu-enabled/llama.cpp
```

This keeps Orbit's default path portable and stable while still allowing advanced GPU experiments through upstream `llama.cpp` builds.

## Requirements

- Python 3.11 or newer
- Linux recommended
- a local Gemma 4 target GGUF model

Optional artifacts:

- MTP draft GGUF for native speculative decoding
- matching `mmproj` GGUF for multimodal image/audio input

## Quick start

### 1. Install Orbit

```bash
git clone https://github.com/guelfoweb/orbit.git
cd orbit
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

This installs the Python package and CLI.

### 2. Build the native backend libraries

If `vendor/lib/` does not already contain the required CPU native libraries, build them explicitly:

```bash
python3 scripts/build_native.py
```

This uses Orbit's vendored `llama.cpp` sources and writes local build output under `src/orbit/native_llama/vendor/`.

Developer fallback:

- `--llama-root /path/to/llama.cpp`
- `ORBIT_LLAMA_ROOT=/path/to/llama.cpp`

### 3. Download models

Download only the target model:

```bash
orbit download ggml-org/gemma-4-12B-it-GGUF
```

Download only the multimodal projector:

```bash
orbit download ggml-org/gemma-4-12B-it-GGUF/mmproj-gemma-4-12B-it-Q8_0.gguf
```

Download only the MTP draft:

```bash
orbit download unsloth/gemma-4-12b-it-GGUF/MTP/gemma-4-12b-it-Q8_0-MTP.gguf
```

### 4. Start the native backend

Stable default server, with MTP disabled:

```bash
orbit server
```

By default, the native server performs startup route-prefix prewarm for the
tools-on route prefix. This shifts the first tools-on route prefill cost to
startup. To disable only startup prewarm:

```bash
ORBIT_KV_PREFIX_PREWARM=off orbit server
```

To disable route prefix-anchor and prewarm:

```bash
ORBIT_KV_PREFIX_ANCHOR=off orbit server
```

The terminal client starts with tools enabled by default. To start without local
shell tools:

```bash
ORBIT_TOOLS=off orbit
orbit --tools off "hello"
```

To get a reasonable CPU/RAM starting profile, you can first run `scripts/suggest-server-profile.sh`; it checks local CPU and RAM and prints conservative environment-variable suggestions to review before export.

If native libraries are not packaged inside Orbit yet, use:

```bash
orbit server --llama-root /path/to/llama.cpp
```

Optional experimental MTP mode:

```bash
orbit server --mtp
```

With a multimodal projector:

```bash
  orbit server \
  --port 12120 \
  --mmproj models/ggml-org--gemma-4-12B-it-GGUF/mmproj-gemma-4-12B-it-Q8_0.gguf
```

You can combine MTP and multimodal flags when both artifacts are available:

```bash
  orbit server \
  --port 12120 \
  --mtp \
  --mmproj models/ggml-org--gemma-4-12B-it-GGUF/mmproj-gemma-4-12B-it-Q8_0.gguf
```

What this means:

- `orbit server` starts the stable native backend without MTP.
- `orbit server --mtp` enables the experimental MTP path explicitly.
- MTP can improve some workloads, but Orbit keeps it off by default because stability has priority.
- if native libs are missing, Orbit exits with a short error telling you to build them with `python3 scripts/build_native.py` or use `--llama-root` / `ORBIT_LLAMA_ROOT`
- native route KV prefix-anchor runs in safe auto mode by default for eligible
  tools-on route calls; disable it with `ORBIT_KV_PREFIX_ANCHOR=off` if you need
  the baseline prefill path
- experimental multi-turn raw MTP chat reuse remains debug-only behind:
  - `ORBIT_MTP_CHAT_REUSE_RAW=1`
  - `ORBIT_MTP_CHAT_REUSE_DEBUG=1`

### 5. Check the server

After startup, you can verify that the server is healthy:

```bash
orbit --health
```

Inside the interactive client, you can inspect backend state with:

```text
/health
/props
```

Expected `/props` values:

- default server:
  - `backend_mode=no-mtp`
  - `mtp_enabled=false`
- with `--mtp`:
  - `backend_mode=mtp-ready`
  - `mtp_enabled=true`

### 6. Start Orbit

```bash
orbit
```

Inside Orbit, tools are off by default:

```text
/tools on
```

## One-shot usage

```bash
orbit "Say who you are in one short sentence."
orbit --image workdir/media/image1.jpg "Describe this image."
orbit --audio workdir/media/audio1.wav "Transcribe or summarize this audio."
```

## Thinking mode

Orbit supports runtime thinking visibility:

```text
/think off
/think on
```

- `think off`: do not request visible reasoning
- `think on`: request visible reasoning first, then the final answer

The backend and model must actually support visible reasoning for this to appear correctly.

## Interactive commands

```text
/compact [tools]  Compact memory or old tool results.
/continue         Continue the last answer if it reached max_tokens.
/health           Check backend health.
/help             Show this help.
/max-tokens [n]   Show or set output token limit for following turns.
/think [off|on]   Show or set thinking visibility.
/reset            Clear current conversation and saved session.
/sessions clear   Delete all saved sessions for this workdir.
/status [ctx]     Show runtime status or estimated context usage.
/tools [off|on]   Show or set shell tool access.
/exit             Exit interactive mode.
```

`/max-tokens <n>` affects only the current runtime. It does not rewrite config or session files.

## Compatibility path

Orbit can still talk to compatible local HTTP backends through `--base-url`. Keep this as compatibility or comparison, not as the preferred product path.

## Troubleshooting

- backend unavailable: check `orbit --health --base-url ...`
- native libraries missing: run `python3 scripts/build_native.py`, or use `--llama-root` / `ORBIT_LLAMA_ROOT` as a developer fallback
- model not found: verify the Orbit models cache or explicit model paths
- multimodal unavailable: ensure the matching `mmproj` is present and the backend was started with it
- MTP unavailable: ensure both target and draft models are available and the backend was started with the experimental MTP path

## Benchmarks and prompts

- performance notes: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- native packaging roadmap: [docs/NATIVE_PACKAGING_ROADMAP.md](docs/NATIVE_PACKAGING_ROADMAP.md)
- runtime techniques: [docs/TECHNIQUES.md](docs/TECHNIQUES.md)
- manual regression prompts: [docs/PROMPTS.md](docs/PROMPTS.md)
- release confidence suite: [docs/RELEASE_CONFIDENCE.md](docs/RELEASE_CONFIDENCE.md)

## Maintenance commands

```bash
orbit bench-core --base-url http://127.0.0.1:12120
orbit release-confidence --base-url http://127.0.0.1:12120 --keep-failed
```
