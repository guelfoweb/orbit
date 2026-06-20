# orbit

Minimal local agentic CLI centered on native `orbit-server`, with optional compatibility for other OpenAI-compatible local backends.

Orbit is designed around Gemma 4, local execution, streaming output, shell-tool opt-in, and CPU-only usability. The native path supports chat, tools, session reuse, MTP, and multimodal input without requiring `llama-server` as the primary backend.

Important status note:
the native backend is the primary Orbit path, but a fresh clone is not yet a zero-build product. Today, native Orbit still expects prepared native `llama`/`ggml` libraries, and some MTP paths still rely on local shim compilation. See [docs/NATIVE_PACKAGING_ROADMAP.md](docs/NATIVE_PACKAGING_ROADMAP.md).

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

This installs the Python package and CLI. It does not yet guarantee a fully self-contained native backend on a fresh machine.

### 2. Download models

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

### 3. Start the native backend

Basic native server, with MTP disabled by default:

```bash
orbit server --port 11976
```

With MTP enabled explicitly:

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

Notes:

- `orbit server` does not enable MTP unless `--mtp` is passed.
- experimental multi-turn raw MTP chat reuse remains debug-only behind:
  - `ORBIT_MTP_CHAT_REUSE_RAW=1`
  - `ORBIT_MTP_CHAT_REUSE_DEBUG=1`

### 4. Start Orbit

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
