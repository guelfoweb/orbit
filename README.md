# Orbit

Orbit is a Python local-AI runtime for CPU-only machines. It combines local
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

## Development and tests

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
