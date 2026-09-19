# Orbit

Orbit is a Python local-AI runtime for CPU-only machines. Chat with a local
model, work with files, and let the model use tools when needed. Linux x86_64
is the qualified platform.

**Qwen3.8 Flash Next UD-IQ1_M** is the highlighted model. Its backend and
CHAT execution are qualified on a Dell Pro 5 14 with an Intel Core Ultra 7
366H, 30 GiB usable RAM and NVMe storage. Its 74.5 GB model is larger than RAM,
so storage speed matters. Other verified models are listed
[below](#supported-models).

Orbit includes its own vendored llama.cpp backend. The build below compiles it;
there is no separate llama.cpp installation or external inference server to set up.

## Install

You need Python 3.11 or newer, Git, CMake and a C/C++ build toolchain. RAM and
storage requirements depend on the model; the Qwen configuration above is a
measured setup, not a minimum requirement for every model or machine.

On Debian/Ubuntu with Python 3.11 or newer:

```bash
sudo apt install git build-essential cmake python3-venv
git clone https://github.com/guelfoweb/orbit.git
cd orbit
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
python3 scripts/build_native.py
```

## Choose a model directory

Downloads and the server use the same model directory. The default is `models/`
in the checkout, or `~/.cache/orbit/models` outside a checkout.

To use another disk, set a writable directory before downloading. Replace this
example path with yours:

```bash
orbit config models-dir /mnt/data/orbit-models
orbit config models-dir
```

The first command creates the directory if needed and saves the setting; the
second shows the effective directory. Existing models are not moved. If you
copy them, retain the `<owner>--<repo>/` subdirectories.

A command's `--models-dir` override takes precedence over `ORBIT_MODELS_DIR`,
then the saved setting, then the default. For a model larger than RAM, use fast
local storage; Orbit warns about known unsuitable filesystems such as eCryptfs.

## Run

Start the server from your activated environment:

```bash
orbit server
```

Choose **Qwen 3.8 Flash Next** from the model menu. If it is missing, confirm the
offered download, or choose another verified model. Orbit selects the startup
profile; normal use needs no tuning flags.

Leave this terminal running. Wait for the `listening on` message before chatting.
On the qualified Qwen setup, startup includes a synchronous warm-up that can take
a few minutes. It moves work to startup to shorten the first request; it does
not reduce total computation.

Open a second terminal in the checkout:

```bash
. .venv/bin/activate
orbit
```

Ask a question, or try `Use system_info to describe this computer.` Tools are
on by default; the model chooses when to use them.

Useful commands inside the chat:

| Command | Purpose |
|---|---|
| `/help` | List available commands. |
| `/status` | Inspect the active model and runtime settings. |
| `/reset` | Clear the conversation and saved session. |
| `/exit` | Close the chat client. |

The server stays open when the client exits. Stop it with Ctrl-C when finished.
For command-line options, use `orbit --help` or `orbit server --help`.

## Download models separately

You can download before starting the server:

```bash
orbit download unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf
```

This command downloads **all three shards**, about **74.5 GB total**. They form
one model: keep them together and choose its single entry in the model menu.
Orbit shows progress for each shard, resumes interrupted downloads, reuses
complete shards and validates the set. Rerun the same command to finish missing
or interrupted shards.

Other verified model repositories and exact GGUF filenames are in the
[model registry](src/orbit/native_llama/model_registry.json).
`orbit download --help` lists download options.

## Supported models

These are the verified native model/quantization combinations. Backend
verification is separate from ANALYSIS qualification and does not extend to
every quantization or similarly named model.

| Model | Verified quantization |
|---|---|
| **Qwen3.8 Flash Next** | **UD-IQ1_M**, three shards |
| Ornith 1.5 35B-A3B | Q4_K_M |
| Qwen3.8 27B | Q4_K_M |
| Qwen 3.6 35B-A3B | Q4_K_M |
| Qwen3-Coder 30B-A3B Instruct | Q4_K_M |
| Gemma 4 26B-A4B | Q4_0 |

Qwen Flash Next's automatic Dell profile uses ctx 4096, threads 10/10,
batch/ubatch 256/128, one slot, CPU repack off and MTP off. These settings and
its startup warm-up are qualified for that setup; other hardware and models
can resolve different defaults.

Static artifact analysis has bounded qualification with **Ornith 1.5 Q4_K_M**.
**Qwen3.8 Flash Next UD-IQ1_M**, in the configuration above, did not pass the
retained ANALYSIS FINISH semantic cases; its backend/CHAT qualification remains
valid. See [ANALYSIS qualification and limits](docs/ANALYSIS_QUALIFICATION.md).
In an Ornith session, use `/analysis path/to/artifact` to begin and `/chat` to
return to chat.
Analysis handles one local artifact at a time without executing the input or
fetching remote payloads. Advanced usage lives in [docs/](docs/).

## Observed performance

This is a production observation, not a comparison between models:

| Model and hardware | Workload | Prefill | Decode |
|---|---|---:|---:|
| Qwen3.8 Flash Next UD-IQ1_M; Dell Core Ultra 7 366H, 30 GiB RAM, NVMe, CPU-only | 119-token final prompt, 419-token answer after startup warm-up and two short turns | 10.2 tok/s | 2.8 tok/s |

Measured on 2026-09-17 at Orbit `de93c14`, with the Qwen profile above,
temperature 0 and thinking off. Rates are native per-call measurements; cached
input is not counted as evaluated prefill. This bounded conversation is not a
steady-state throughput benchmark.

In the same run, startup warm-up took 150.5 s, the server was ready after
175.4 s, and the first short turn took 12.5 s. Peak server RSS was 27.3 GiB.
Longer conversations can take substantially more time even with route caching,
because final answers still need their conversation context. Timing depends on
the prompt, reply length, model/page-cache warmth, storage and host load.
These figures do not describe Ornith, the other verified models, or other CPUs.
