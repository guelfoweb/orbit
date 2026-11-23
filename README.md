# Orbit

Orbit is a Python local-AI runtime with CPU execution by default. It combines local
chat, model-selected tools, file workflows and static artifact analysis in a
terminal client. Linux x86_64 is the qualified platform.

The runtime manages tools, sessions, evidence, verification and bounded repair.
The integrated llama.cpp backend handles tokenization, inference, streaming and
model state. Orbit builds its own vendored backend; no separate llama.cpp
installation or external inference server is required.

## Install and quick start

You need Python 3.11 or newer, Git, CMake and a C/C++ build toolchain. ANALYSIS
also requires bubblewrap. Memory and storage requirements depend on the model
and workload.

On Debian/Ubuntu with Python 3.11 or newer:

```bash
sudo apt install git build-essential cmake python3-venv bubblewrap
git clone https://github.com/guelfoweb/orbit.git
cd orbit
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
python3 scripts/build_native.py
```

Start the local server:

```bash
orbit server
```

Choose a model from the menu. If it is missing, Orbit offers to download it.
To store models on another disk, [set the model directory](#configuration-and-model-store)
first. Orbit selects the startup profile; normal use needs no tuning flags.
Loading and any startup warm-up depend on the model. Wait for `listening on`
before connecting.

Leave the server terminal running. In a second terminal, open the same checkout:

```bash
. .venv/bin/activate
orbit
```

The client connects to the local server. Exiting the client leaves the server
running; stop the server with Ctrl-C when finished. See `orbit --help` and
`orbit server --help` for options.

`/status` shows the client and server Orbit commits. If both are known and
different, startup and `/status` display a non-blocking build-mismatch warning.
The native server also publishes its identity in `/props` and its startup log.
Identity is captured from the installed source checkout once per process, so
updating a checkout does not relabel an already-running server. Packages without
Git metadata and older/external servers report an unknown commit. This is a
revision diagnostic, not an integrity check of local edits or compatibility
certification; it never restarts or disconnects a server.

## Supported models and qualification

The [model registry](src/orbit/native_llama/model_registry.json) contains these
verified native profiles. Verification applies to the exact model,
quantization, template and tested workflows, not every related variant.
This list is not a performance ranking.

| Model | Verified quantization |
|---|---|
| Gemma 4 26B-A4B | Q4_0 |
| Ornith 1.5 35B-A3B | Q4_K_M |
| Qwen 3.6 35B-A3B | Q4_K_M |
| Qwen3-Coder 30B-A3B Instruct | Q4_K_M |
| Qwen3.8 27B | Q4_K_M |
| Qwen3.8 Flash Next | UD-IQ1_M, three shards |
| Inkling 2 Mini (Mainline Support) | UD-Q4_K_M |

**Backend/CHAT qualification is separate from ANALYSIS semantic qualification.**
A working model integration and valid tool calls do not establish that its
explanations are correct.

Ornith has bounded ANALYSIS qualification on specific retained tests, but
retained reports also contain factual errors. It is not generally qualified
for accurate explanations across the semantic corpus.

Qwen3.8 Flash Next UD-IQ1_M retains backend/CHAT qualification on the tested
Dell configuration. It did not pass the retained ANALYSIS semantic cases.
That result does not extend to every Qwen model or quantization.

See the [qualification scope](docs/ANALYSIS_QUALIFICATION.md) and
[semantic baseline](docs/ANALYSIS_SEMANTIC_BASELINE.md) for evidence and limits.
No model in this table is presented as generally semantically qualified for
ANALYSIS.

## Inkling 2 Mini

Inkling uses a separate CPU/text backend family. Build it alongside the default
backend; the existing models keep their original backend:

```bash
python3 scripts/build_native.py --family mainline-support --jobs 4
orbit download unsloth/Inkling-2-Mini-GGUF --models-dir /srv/orbit-models
orbit server --model-id inkling2-mini-ud-q4-k-m --models-dir /srv/orbit-models --ctx 8192 --threads 6 --threads-batch 6 --batch 256 --ubatch 128 --think off --repack off
```

Change `--models-dir` to your configured model store. The registered download is
pinned to the Mainline_Support artifact, saved locally with a `.gguf` extension.
The tested artifact is 16,610,544,608 bytes, SHA256
`78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae`.

Inkling accepts the named `reasoning_effort` values `none`, `minimal`, `low`,
`medium`, `high`, `xhigh` and `max` in native CHAT requests. `max` and `xhigh`
both map to 0.99 in this exact model's template. Explicit effort is passed to
the native template, including exact token counting; it does not insert custom
instructions into messages. The local GUI exposes all seven choices and starts
at **None**. None requests minimum effort, not a complete reasoning shutdown.
Higher effort does not guarantee more tokens or a slower response on every prompt.
Rebuild the mainline-support backend after updating to obtain this capability.
See [named effort controls](docs/MODEL_RESEARCH_QUALIFICATION.md#named-reasoning-effort).

Post-tool finals reserve 1024 internal reasoning tokens alongside their existing
bounded answer allowance. Native separated-reasoning CHAT uses incremental
visible output when streaming is requested; answers requiring evidence/citation
verification may still be delivered together after checking.

With the exact CPU configuration above, thinking off, None effort and no MTP, Inkling can
reuse one in-memory checkpoint of the first 1024 prompt tokens. The first long
request captures it; later requests must share those exact tokens and the same
model, template, native build, context and session identity. Short requests do
not use it. Reset, cancellation, completion errors or an ineligible configuration discard
it. Native calls carrying a tools payload are ineligible. A failed restore
clears native memory and falls back to ordinary prefill.
The observed checkpoint uses about 196 MiB plus a temporary copy; its hard limit
is 256 MiB. Set `ORBIT_ALIGNED_PREFIX_REUSE=0` before starting the server to
disable it. This reduces repeated prefill work; it does not accelerate token
generation or eliminate the first request's cold cost.
See [the checkpoint contract](docs/MODEL_RESEARCH_QUALIFICATION.md#inkling-aligned-prefill-checkpoint)
and [reasoning budgets](docs/MODEL_RESEARCH_QUALIFICATION.md#post-tool-finals-with-required-reasoning).

This profile does not use MTP, GPU offload or multimodal input. Its internal
reasoning remains active even with `--think off`; Orbit separates that reasoning
from visible answers and reserves the declared decision budget. The tested
configuration passed five manually reviewed CHAT observations and the frozen
24-case tool-intent gate, with the executor intercepted. This qualifies tool
intent and protocol handling, not generated-code correctness or ANALYSIS
semantics. No general performance claim follows from these observations.

## Chat and tools

Ask a question or request a task, for example:
`Use system_info to describe this computer.` Tools are on by default and can
run local shell commands; the model chooses when to use them. Use `/tools off`
for chat without tools and `/tools on` to restore access.

| Client command | Purpose |
|---|---|
| `/help` | List commands and their syntax. |
| `/status` | Inspect the active model and runtime settings. |
| `/read <source> [prompt]` | Read a local document or URL. |
| `/reset` | Clear the conversation and saved session. |
| `/exit` | Close the client. |

## ANALYSIS and its limits

Use `/analysis path/to/artifact` to investigate one local artifact, and `/chat`
to return to chat. ANALYSIS combines deterministic extraction and decoding with
model-guided investigation. Generated Python actions run in a resource-bounded
bubblewrap sandbox with a read-only input and network access denied.

The canonical Markdown report puts attested indicators and decoded values
before unverified model answers. It retains provenance, evidence coverage,
open questions and the reason the investigation stopped. A complete report
records the work performed; it does not mean the investigation answered every
question or established all of the artifact's behavior.

Source acquisition, delivery to the model and proof of a conclusion are distinct.
Review model interpretations against the evidence, especially claims about
remote content or behavior not observed. The
[semantic baseline](docs/ANALYSIS_SEMANTIC_BASELINE.md) records required facts,
known errors and unknowns; automatic integrity checks do not replace semantic
review.

## Configuration and model store

Downloads and the server use the same model directory: `models/` in a checkout,
or `~/.cache/orbit/models` outside one. To choose a writable location:

```bash
orbit config models-dir /mnt/data/orbit-models
orbit config models-dir
```

The first command creates the directory if needed and saves the setting; the
second shows the effective directory. Existing models are not moved. When
copying models, retain the `<owner>--<repo>/` subdirectories.

Precedence is an explicit command's `--models-dir`, then `ORBIT_MODELS_DIR`,
then the saved setting, then the default.

You can also download before starting the server, for example:

```bash
orbit download ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf
```

For a multi-shard GGUF, request its first shard. Orbit downloads the complete
set, reports per-shard progress, resumes interrupted transfers and reuses
complete shards. Keep the shards together; they appear as one model in the
menu. Rerun the same command to finish an interrupted set. Exact repository and
GGUF names are in the registry; options are in `orbit download --help`.

Single files and shards retain a `.part` beside the destination across
interruptions. Rerun the same command to resume; progress includes the bytes
already downloaded. Transfers use a 30-second socket inactivity timeout and
at most four attempts with bounded backoff. Only a complete transfer matching
the declared size is published by atomic rename. Existing final files are reused.
See [download recovery](docs/DOWNLOAD_RECOVERY.md) for server compatibility and
integrity limits.

## Development and tests

For the lightweight, artifact-free [core CI gate](docs/CORE_CI.md), run:

```bash
python3 scripts/check_core.py
```

The project uses `unittest`. From the checkout, run the non-live suite:

```bash
TMPDIR=/tmp PYTHONPATH=src python3 -m unittest discover -s tests -q
```

Some qualification tests require external, hash-pinned corpus files. Follow
[AGENTS.md](AGENTS.md) for provisioning and contribution gates. ANALYSIS changes
must preserve the [deterministic cross-sample gate](tests/test_analysis_cross_sample_gate.py);
claims of semantic improvement also require comparison against the
[semantic baseline](docs/ANALYSIS_SEMANTIC_BASELINE.md), with no regressions on
applicable criteria.

The [tool-intent qualification gate](docs/TOOL_INTENT_QUALIFICATION.md) checks
24 frozen bilingual intents with tool execution intercepted. Its retained-response
replay runs offline and emits JSON and Markdown capability results.

## Optional Intel GPU execution

CPU remains the default. Inkling's mainline text backend also has an optional,
experimental Intel integrated GPU variant using Vulkan. It is built separately;
selecting GPU never substitutes a CPU backend silently. Other model families
continue to use CPU. MTP and multimodal input are unavailable in this variant.

Build dependencies include Vulkan development files, SPIR-V headers, and a
`glslc` compiler supporting `GL_EXT_integer_dot_product`. Older distribution
compilers may not support that feature; select a newer one explicitly:

```bash
python3 scripts/build_native.py --family mainline-vulkan --glslc /path/to/glslc
orbit server --model-id inkling2-mini-ud-q4-k-m --accelerator intel-gpu
```

Use `--accelerator cpu` or omit the flag for CPU. The server reports the actual
device and assigned GPU layers in `/props` under `acceleration`. Startup fails
if no integrated Intel Vulkan device or no GPU-assigned layers are available.
GPU host defaults are one inference thread and one batch thread; the process
prefers two available efficient cores on hybrid CPUs. This profile does not
borrow CPU calibration or the CPU-only aligned-prefix checkpoint policy.

The local GUI offers **CPU (default)** and **Intel GPU (experimental)** beside
each available model. Changing the device restarts the GUI-owned server and
starts a new chat, preserving the old saved conversation. A failed startup
attempt restores the previous owned selection when possible. Externally
started servers remain under their operator's control.

The Vulkan source is a hash-bound overlay on the mainline source, with its own
build directory, libraries, source and patchset identities. CPU sources and
libraries are preserved. GPU speed depends on the device and workload; model
identity recognition does not extend CPU tool/ANALYSIS qualification to GPU.
