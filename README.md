# orbit

Minimal local CLI for running Gemma 4 with the native `orbit-server`.

Orbit is designed for local execution, streaming output, optional shell tools, and a simple terminal workflow. The normal Orbit setup does not require an external `llama-server` process at runtime.

Important status note:
the native backend is the primary Orbit path, and Orbit does not require an external `llama-server` runtime process. The native backend still depends on native libraries derived from `llama.cpp`/`ggml`, built either from Orbit's vendored sources or from a documented developer fallback such as `--llama-root`. Zero-build packaging remains future work. See [docs/NATIVE_PACKAGING_ROADMAP.md](docs/NATIVE_PACKAGING_ROADMAP.md).

Linux is the main target environment. macOS may work. Windows is not a target environment.

## What it does

- chats with a local model through a small terminal CLI
- streams answers and runtime metrics
- exposes unrestricted local shell only when `/tools on` is enabled
- keeps the tool loop model-driven
- supports local image and audio input when the backend is started with multimodal support
- supports native MTP when target and draft models are available
- stores lightweight sessions and prompt history under `~/.orbit`

Orbit stays model-driven. The runtime enforces safety, size, timeout, and tool-contract boundaries, but it does not deterministically solve user tasks.

## Backend stance

- Primary backend path: native `orbit-server`
- Compatibility path: `llama-server` or another OpenAI-compatible local backend
- CLI default base URL: `http://127.0.0.1:11976`

If your native server runs on another port, pass `--base-url`.

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

If `vendor/lib/` does not already contain the required native libraries, build them explicitly:

```bash
python scripts/build_native.py
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

Download the full local stack declared by the registry:

```bash
orbit download --all
```

You can still override the repo explicitly:

```bash
orbit download --all ggml-org/gemma-4-12B-it-GGUF
```

### 4. Start the native backend

Stable default server, with MTP disabled:

```bash
orbit server --port 11976
```

If native libraries are not packaged inside Orbit yet, use:

```bash
orbit server --port 11976 --llama-root /path/to/llama.cpp
```

Optional MTP mode:

```bash
orbit server --port 11976 --mtp
```

With a multimodal projector:

```bash
orbit server \
  --port 11976 \
  --mmproj models/ggml-org--gemma-4-12B-it-GGUF/mmproj-gemma-4-12B-it-Q8_0.gguf
```

You can combine MTP and multimodal flags when both artifacts are available:

```bash
orbit server \
  --port 11976 \
  --mtp \
  --mmproj models/ggml-org--gemma-4-12B-it-GGUF/mmproj-gemma-4-12B-it-Q8_0.gguf
```

What this means:

- `orbit server` starts the stable native backend without MTP.
- `orbit server --mtp` enables the experimental MTP path explicitly.
- MTP can improve some workloads, but Orbit keeps it off by default because stability has priority.
- if native libs are missing, Orbit exits with a short error telling you to build them with `python scripts/build_native.py` or use `--llama-root` / `ORBIT_LLAMA_ROOT`
- experimental multi-turn raw MTP chat reuse remains debug-only behind:
  - `ORBIT_MTP_CHAT_REUSE_RAW=1`
  - `ORBIT_MTP_CHAT_REUSE_DEBUG=1`

### 4. Check the server

After startup, you can verify that the server is healthy:

```bash
orbit --base-url http://127.0.0.1:11976 --health
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
### 5. Start Orbit

```bash
orbit --base-url http://127.0.0.1:11976
```

Inside Orbit, tools are off by default:

```text
/tools on
```

## One-shot usage

```bash
orbit --base-url http://127.0.0.1:11976 "Say who you are in one short sentence."
orbit --base-url http://127.0.0.1:11976 --image workdir/media/image1.jpg "Describe this image."
orbit --base-url http://127.0.0.1:11976 --audio workdir/media/audio1.wav "Transcribe or summarize this audio."
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
- native libraries missing: run `python scripts/build_native.py`, or use `--llama-root` / `ORBIT_LLAMA_ROOT` as a developer fallback
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
orbit bench-core --base-url http://127.0.0.1:11976
orbit release-confidence --base-url http://127.0.0.1:11976 --keep-failed
```
