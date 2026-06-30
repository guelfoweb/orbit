# orbit

Orbit is a small Python-first local runtime for Gemma 4 12B on CPU-only
machines. The primary path is the native `orbit server` backend, using
vendored `llama.cpp`/`ggml` libraries built and loaded by Orbit. It does not
require an external `llama-server` process for normal use.

Orbit stays model-driven. The runtime enforces safety, size, timeout, context,
and tool-contract boundaries, but the model decides whether to answer directly
or use exposed tools.

Linux is the main target environment. macOS may work. Windows is not a target.

## Current Scope

- local CLI and native HTTP server for Gemma 4 12B
- CPU-first native backend
- shell tools when tools mode is enabled
- streaming terminal output and compact progress phases
- route-prefix KV anchor and startup prewarm enabled by default
- optional multimodal image/audio support when the matching `mmproj` is loaded
- experimental native MTP with `orbit server --mtp`

MTP is supported for local testing, but it remains experimental. Do not treat it
as production-ready or as a guaranteed performance win.

## Requirements

- Python 3.11 or newer
- Linux recommended
- Gemma 4 12B target GGUF
- optional Gemma 4 `mmproj` GGUF for multimodal input
- optional MTP draft GGUF for `orbit server --mtp`

## Install

```bash
git clone https://github.com/guelfoweb/orbit.git
cd orbit
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Build the vendored native libraries if they are not already present:

```bash
python3 scripts/build_native.py
```

Download model artifacts as needed:

```bash
orbit download ggml-org/gemma-4-12B-it-GGUF
orbit download ggml-org/gemma-4-12B-it-GGUF/mmproj-gemma-4-12B-it-Q8_0.gguf
orbit download unsloth/gemma-4-12b-it-GGUF/MTP/gemma-4-12b-it-Q8_0-MTP.gguf
```

## Quick Start

Start the native server with MTP enabled:

```bash
PYTHONPATH=src .venv/bin/orbit server --mtp
```

The current release gate expects:

- native backend loaded
- MTP initialized when `--mtp` is used
- multimodal capability detected when `mmproj` is available
- route-prefix KV prewarm completed
- no duplicate `llama.cpp` runtime loaded
- clean shutdown without double-free, SIGABRT, or segfault

In another terminal:

```bash
.venv/bin/orbit --workdir workdir --tools on --think off "hi, how are you?"
```

For route/KV diagnostics:

```bash
ORBIT_KV_DIAG=1 .venv/bin/orbit --workdir workdir --tools on --think off "hi"
```

`ORBIT_KV_DIAG=1` is diagnostic only. It is not required for normal use.

## Tools

Tools are enabled by default in the current server/client flow.

Tools mode exposes unrestricted local shell access through the model-facing
shell tool. Use it only in an isolated lab or safe workdir.

Disable tools at server startup:

```bash
ORBIT_TOOLS=off .venv/bin/orbit server --mtp
```

Disable tools for a client/session:

```bash
.venv/bin/orbit --tools off "hello"
```

Interactive toggles:

```text
/tools off
/tools on
```

`--tools off` is client/session-side. If a server was already started with
tools enabled, it may already have performed startup route-prefix prewarm.

## KV Prefix Anchor and Prewarm

Route-prefix KV anchor is enabled by default in auto mode. Startup prewarm is
also enabled by default for the tools-on route prefix.

Disable only startup prewarm:

```bash
ORBIT_KV_PREFIX_PREWARM=off .venv/bin/orbit server --mtp
```

Disable route-prefix anchor and prewarm:

```bash
ORBIT_KV_PREFIX_ANCHOR=off .venv/bin/orbit server --mtp
```

The prewarm cost is paid at startup. It does not remove CPU work; it shifts part
of the first tools-on route cost before the first user request.

## Streaming and Progress

Orbit uses classic terminal UX, not a full-screen TUI. Progress phases
distinguish internal routing from final-answer generation:

- `tool decision`
- `final answer`
- `final retry`

Internal route prose is not accepted as a final answer. If the route stream
violates the route contract, Orbit can abort that internal route generation and
fall back to the existing final-answer retry path.

When the backend emits token deltas, final answers stream. If a backend returns
only final content without deltas, Orbit prints the returned content when the
call completes.

## Thinking Mode

```text
/think off
/think on
```

`think off` is the normal mode. `think on` requests visible reasoning when the
backend/model supports it. Think-on paths can be much slower on CPU.

## Multimodal Input

When the matching `mmproj` is available and detected by the native server:

```bash
.venv/bin/orbit --image workdir/media/image1.jpg "Describe this image."
.venv/bin/orbit --audio workdir/media/audio1.wav "Summarize this audio."
```

Multimodal capability should be visible through `/v1/models` and `/props`.

## Useful Commands

```text
/health           Check backend health.
/props            Show backend properties when available.
/status [ctx]     Show runtime status or estimated context usage.
/max-tokens [n]   Show or set output token limit for following turns.
/think [off|on]   Show or set thinking visibility.
/tools [off|on]   Show or set shell tool access.
/continue         Continue the last answer if it reached max_tokens.
/reset            Clear current conversation and saved session.
/sessions clear   Delete all saved sessions for this workdir.
/exit             Exit interactive mode.
```

## CPU Notes

Orbit targets local CPU-first operation. Some paths are expected to be slow:

- web-search final answers with large evidence
- `read` over large files
- visible thinking
- first requests after cold server startup
- experimental MTP paths

Do not interpret MTP as a general speed guarantee. Measure the actual workload.

## Compatibility

The preferred runtime is native `orbit server`. Orbit can still talk to a local
OpenAI-compatible HTTP backend through `--base-url`, but that is a compatibility
or comparison path, not the primary product path.

## Troubleshooting

- backend unavailable: run `.venv/bin/orbit --health --base-url ...`
- native libraries missing: run `python3 scripts/build_native.py`
- model not found: verify the Orbit model cache under `models/`
- multimodal unavailable: verify the matching `mmproj` is present
- MTP unavailable: verify both target and draft artifacts are present and start
  the server with `--mtp`
- slow web/read/think-on output: expected on CPU; inspect footer metrics and
  use `ORBIT_KV_DIAG=1` only when diagnosing cache behavior

## Regression Prompts and Checks

- manual prompts: [docs/PROMPTS.md](docs/PROMPTS.md)
- release confidence: [docs/RELEASE_CONFIDENCE.md](docs/RELEASE_CONFIDENCE.md)
- performance notes: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- native packaging roadmap: [docs/NATIVE_PACKAGING_ROADMAP.md](docs/NATIVE_PACKAGING_ROADMAP.md)

```bash
python3 -m unittest discover -s tests -q
python3 -m compileall -q src tests scripts
git diff --check
```
