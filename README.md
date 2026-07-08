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
- explicit shell tools when tools mode is enabled
- streaming terminal output and compact progress phases
- route-prefix KV anchor and startup prewarm enabled by default
- optional multimodal image/audio support when the matching `mmproj` is loaded
- optional native MTP support via `orbit server --mtp`, with diagnostics and recovery checks
- EvidenceStore-backed post-tool evidence handling

MTP is supported in the native server path as an explicit, experimental option.
It is not enabled by default, is not always-on for every internal completion,
and is not a guaranteed performance win.

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
```

## Quick Start

Start the native server:

```bash
PYTHONPATH=src .venv/bin/orbit server
```

The current release gate expects:

- native backend loaded
- multimodal capability detected when `mmproj` is available
- route-prefix KV prewarm completed
- no duplicate `llama.cpp` runtime loaded
- clean shutdown without double-free, SIGABRT, or segfault

In another terminal:

```bash
.venv/bin/orbit --workdir workdir --think off "hi, how are you?"
```

Enable tools only when you want to expose model-driven shell access:

```bash
.venv/bin/orbit --workdir workdir --tools on --think off
```

For route/KV diagnostics:

```bash
ORBIT_KV_DIAG=1 .venv/bin/orbit --workdir workdir --tools on --think off "hi"
```

`ORBIT_KV_DIAG=1` is diagnostic only. It is not required for normal use.

### Optional MTP

Native MTP is explicit:

Only download the MTP draft model if you intentionally want to test native MTP:

```bash
orbit download unsloth/gemma-4-12b-it-GGUF/MTP/gemma-4-12b-it-Q8_0-MTP.gguf
```

```bash
PYTHONPATH=src .venv/bin/orbit server --mtp
```

Use it for targeted validation or experiments, not as a default speed
assumption. Current diagnostics show MTP can be stable while still being slower
on some CPU-only workloads.

## Tools

Tools are off by default. Tools-on mode exposes unrestricted local shell access
through the model-facing shell tool. Use it only in an isolated lab or safe
workdir.

Keep tools disabled at server startup:

```bash
ORBIT_TOOLS=off .venv/bin/orbit server
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

## Tool Evidence

Orbit keeps tool results auditable without putting large raw outputs back into
the prompt.

After a tool runs:

- raw evidence is preserved in runtime memory and sidecar files
- prompt history stores bounded audit markers, not large raw tool output
- route/final/retry prompts receive compact EvidenceStore projections
- web, shell, grep/search, read, and unknown outputs use bounded evidence cards
- `/reset` clears in-memory evidence for the current session

This keeps post-tool follow-ups model-driven while reducing prompt bloat. The
model still decides whether to answer or use a tool; the runtime only enforces
size, safety, and tool-contract boundaries.

## KV Prefix Anchor and Prewarm

Route-prefix KV anchor is enabled by default in auto mode. Startup prewarm is
also enabled by default for the tools-on route prefix.

Disable only startup prewarm:

```bash
ORBIT_KV_PREFIX_PREWARM=off .venv/bin/orbit server
```

Disable route-prefix anchor and prewarm:

```bash
ORBIT_KV_PREFIX_ANCHOR=off .venv/bin/orbit server
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

- final/retry completions with low KV reuse
- web/read over large or noisy evidence
- visible thinking
- first requests after cold server startup
- MTP paths that do not benefit a specific completion

Do not interpret MTP as a general speed guarantee. Measure the actual workload.
For post-tool issues, first check whether raw evidence is leaking into the
prompt, whether a redundant tool call happened, and whether the final footer
shows a large prompt or simply slow CPU prefill.

Output budgets are per completion kind. `/max-tokens` is still the user-facing
budget, but Orbit may use smaller internal budgets for route, tool, final, and
repair phases to avoid excessive CPU work. If an answer is truncated, the footer
shows `stop: length` and `/continue` is available.

For CPU benchmarking, record the exact Orbit commit or tag, model artifact,
backend mode, context size, thread settings, MTP state, tools mode, and whether
startup prewarm was enabled. Single-run numbers are useful for triage, but not
release-quality performance evidence.

For local performance checks, use `orbit bench-core`. For a conservative
starting point on server thread and batch settings, see
`scripts/suggest-server-profile.sh`. Benchmark and tuning notes are in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Compatibility

The preferred runtime is native `orbit server`. Orbit can still talk to a local
OpenAI-compatible HTTP backend through `--base-url`, but that is a compatibility
or comparison path, not the primary product path.

Native Orbit is CPU-first and currently configures `gpu_layers=0`. GPU tests
should use an external OpenAI-compatible backend such as `llama-server` with
GPU offload enabled, then point Orbit at it with `--base-url`. Treat those
results as compatibility/backend comparisons, not native `orbit server`
performance.

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

Useful local smoke before a release candidate:

```text
/tools on
/think off
run pwd
what directory was that?
run command_that_does_not_exist_123
what happened?
search online for information about OpenAI
what did the search results say?
```
