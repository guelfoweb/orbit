# Orbit

Orbit is a small Python-first local AI runtime designed for CPU-only machines. It
supports a limited set of compatible and verified models, listed in
[Supported Models](#supported-models) below. The primary path is the native Orbit
server backend, using vendored llama.cpp/ggml libraries built and loaded by
Orbit. It does not require an external `llama-server` process for normal use.

Orbit stays model-driven: the runtime enforces safety, size, timeout, context and
tool-contract boundaries; the model decides whether to answer directly or reach
for a tool. Linux x86_64 CPU-only is the qualified platform.

<p align="center">
  <img src="docs/orbit-cli.png" alt="Orbit CLI" width="900">
</p>

## Requirements

- Python 3.11 or newer, on Linux
- CMake, to build the vendored native libraries
- Enough RAM for the model you choose (24 GB practical minimum)

## Install

```bash
git clone https://github.com/guelfoweb/orbit.git
cd orbit
sudo apt install cmake
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Build the vendored native libraries if they are not already present:

```bash
python3 scripts/build_native.py
```

To inspect the host and see a recommended configuration:

```bash
scripts/suggest-server-profile.sh
```

## First run

**1. Start the server** in one terminal. It loads the model and stays running:

```bash
orbit server --ctx 16384 --batch 256 --ubatch 128
```

`--ctx` sets the context window and the largest input that fits in one model
call. Larger contexts need more memory, mostly for the KV cache; if a document
does not fit, Orbit reports the minimum required.

**2. Start the CLI** in another terminal:

```bash
orbit
```

**3. Chat.** The default mode — ask a question, get an answer. The model may use
the exposed tools on its own.

**4. Analyse a file.** Orbit routes to ANALYSIS by itself when a request is
clearly about inspecting a local artifact:

```
analyze ~/samples/loader.js and tell me what it reads from disk
```

You can also enter it explicitly, and leave the same way:

```
/analysis path/to/artifact    # start an analysis
/report                       # answer from evidence already collected, running nothing
/chat                         # back to normal chat
```

**5. Choose how analysis advances.** By default each step returns to you. To let
a run continue on its own:

```
/autonomous on     # or: /autonomous off, or /autonomous to see the current setting
```

`/help` lists every command.

Other server flags: `--low-memory`, and `--model /path/to/model.gguf` to pick a
file explicitly. Without `--ctx` the default is 8192. For one shot without the
interactive session:

```bash
orbit --workdir workdir --think off "hi, how are you?"
```

## Chat and Analysis

**ANALYSIS** inspects one local artifact in an isolated workspace. The model
writes a short Python program, Orbit runs it in a sandbox with no network and a
read-only copy of the artifact, and the output is recorded with its provenance.
One model call performs at most one action, and everything derived stays in the
session workspace. Reports are plain text.

### Guided vs autonomous

ANALYSIS is **guided by default**: it advances one step at a time and control
returns to you after each, so you steer by typing the next instruction —
`continue` included.

Switch at any time, without restarting Orbit:

```
/autonomous          # show the current state
/autonomous on       # continue automatically
/autonomous off      # back to guided
```

The setting belongs to the running CLI process. It survives `/chat` ↔
`/analysis` transitions within that session, and is not persisted globally — a
new `orbit` starts guided again.

**Autonomous** continues by itself while each step produces verifiably new
evidence, stopping on completion, when progress stalls, or at a hard bound of 12
actions. Every ending but a cancellation produces one grounded report. Ctrl-C
stops a run immediately.

It is off by default, and opt-in for cost rather than correctness: a step that
measures something new counts as progress whether or not the measurement was
worth making, so an artifact with little to derive can still attract many
actions. Whether that trade is worthwhile is the operator's call.

## Supported Models

| Model | Prefill | Generation | Tools-on chat | Tool + final | Peak RAM |
|---|---:|---:|---:|---:|---:|
| [Gemma 4 26B-A4B](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) | ~24.2 tok/s | ~7.1 tok/s | ~3.0 s | ~20.9 s | ~29.4 GiB |
| [Qwen 3.6 35B-A3B](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) | ~31.8 tok/s | ~7.6 tok/s | ~6.1 s | ~23.8 s | ~36.4 GiB |
| [Qwen3-Coder 30B-A3B](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | ~26.0 tok/s | ~10.7 tok/s | ~3.0 s | ~18.4 s | ~31.3 GiB |
| [Ornith 1.5 35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) | — | — | — | — | — |

Measured on a NUC10 Intel Core i7-10710U (6 cores / 12 threads), 64 GB RAM, no
GPU; Linux, llama.cpp b9551, `ctx=8192`, 6 threads, batch/ubatch 256/128, Flash
Attention AUTO, thinking off, MTP off.

Chat and tool latencies are warm steady-state medians after one excluded
warm-up; prefill counts evaluated tokens only. **These are measurements on that
one system, not universal model performance** — results vary with hardware,
context, cache reuse and workload.

**Ornith 1.5 35B-A3B** is verified, and the profile the ANALYSIS workflow is
qualified on. It has not been run under the chat protocol above, so it has no
comparable row there. Measured on the same system under the ANALYSIS workload,
at `--ctx 16384`, quantization Q4_K_M:

| Metric | Measured |
|---|---|
| Prefill (cold, no cache reuse) | 21–33 tok/s (n=5, median 31.8) |
| Generation | ~5–10 tok/s (n=22 derived, median 5.9) |

Prefill counts only calls with no reuse, so it matches the definition above.
Generation is derived from wall clock minus prefill — a looser bound than a
dedicated decode benchmark. Peak RAM is not stated because it was never
measured. Methodology and per-call figures:
[docs/benchmarks/ornith-1.5-35b-a3b.md](docs/benchmarks/ornith-1.5-35b-a3b.md).

`--low-memory` is supported only for Qwen3-Coder 30B-A3B. On the system above it
cut peak RSS from ~31.3 to ~18.3 GiB, costing ~13.9% on prefill and ~1% on
decode.

## Limitations

- Linux x86_64 CPU-only is the qualified platform. macOS may work; Windows is
  not a target. There is no GPU path.
- Analysis handles one artifact per session, and the sandbox has no network.
- Autonomous analysis is opt-in and may do unnecessary work on trivial
  artifacts (see above).
- Analysis is qualified on the Ornith 1.5 profile. Other verified profiles fall
  back to ordinary cold behaviour rather than failing.
- The analysis prefix is captured lazily by the first analysis step, which costs
  that step and benefits every later one. Eager capture at startup is opt-in.

## Configuration

Behaviour is controlled by CLI flags and in-session commands. Two environment
variables are supported for startup configuration only:

- `ORBIT_ANALYSIS_AUTONOMOUS=1` — start with autonomous analysis already on.
  `/autonomous on` is the normal way; this only helps a scripted start.
- `ORBIT_ORNITH_ANALYSIS_PREFIX_PREWARM=1` — capture the analysis prefix eagerly
  at startup instead of lazily on first use.
