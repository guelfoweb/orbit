# orbit

Minimal local agentic CLI for `llama.cpp` / `llama-server`.

Orbit is designed around `gemma4:12b-it`, local execution, prompt-cache awareness, streaming output, and CPU-only usability.

It is developed and tested primarily on Linux. macOS may work if `llama-server` and the model files are available. Windows is not a target environment.

## What it does

- Chat with a local `llama-server` model.
- Stream assistant responses in the terminal.
- Enable or disable unrestricted shell tools at runtime.
- Let the model use the local shell for files, web fetches, edits, system inspection, and automation when tools are enabled.
- Attach local images or audio files in one-shot mode when the server is started with multimodal support.
- Keep lightweight sessions and prompt history under `~/.orbit`.

Orbit stays model-driven: the model decides when tools are needed. The runtime only enforces safety, path, size, timeout, and tool-contract boundaries.

## Requirements

- Python 3.11 or newer.
- `llama-server` available in `PATH`.
- Gemma 4 12B instruction-tuned GGUF model.
- Linux recommended.

Optional multimodal support requires the matching `mmproj-gemma-4-12B-it-Q8_0.gguf` projector.

## Quick start

Follow these steps in order.

### 1. Install system packages

On Debian/Ubuntu-like systems, install the basic build tools first:

```bash
sudo apt update
sudo apt install -y git cmake build-essential python3 python3-venv
```

### 2. Build llama.cpp

Orbit is tested with a Gemma 4 compatible `llama.cpp` fork:

```text
https://github.com/qualcomm/llama.cpp
branch: gemma-4-support-smaller-assistants
```

Use this build for Orbit and MTP speculative decoding:

```bash
git clone https://github.com/qualcomm/llama.cpp.git llama.cpp-gemma4
cd llama.cpp-gemma4
git checkout gemma-4-support-smaller-assistants
cmake -B build -DGGML_NATIVE=ON -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
cmake --build build --config Release -j"$(nproc)"
export PATH="$PWD/build/bin:$PATH"
```

Verify:

```bash
llama-server --version
```

This step is required before any model download command, because Orbit's helper
uses `llama-server -hf` to fetch GGUF files into the Hugging Face cache.

### 3. Install Orbit

Clone this repository and install the CLI:

```bash
git clone https://github.com/guelfoweb/orbit.git
cd orbit
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

### 4. Download the model

Use the Orbit helper to download the main GGUF model and the MTP draft model
into the standard Hugging Face cache:

```bash
scripts/gemma4-12b-server.sh download --mtp
```

Internally this uses:

```bash
llama-server -hf ggml-org/gemma-4-12B-it-GGUF --hf-file gemma-4-12B-it-Q4_K_M.gguf
llama-server -hf unsloth/gemma-4-12b-it-GGUF --hf-file MTP/gemma-4-12b-it-Q8_0-MTP.gguf
```

If `llama-server` starts after the download, stop it with `Ctrl+C`.

### 5. Start Orbit

Optionally inspect a suggested CPU profile for your machine:

```bash
scripts/suggest-server-profile.sh
```

If you want to use the suggested profile, export the printed values before
starting the server.

Start the tuned local server:

```bash
scripts/gemma4-12b-server.sh start --mtp
```

Then start Orbit:

```bash
orbit
```

Inside Orbit, tools start disabled. Enable shell tools only when you need local or web operations:

```text
/tools on
```

If something fails, run:

```bash
scripts/gemma4-12b-server.sh status
tail -n 80 ~/.orbit/gemma4-12b-server.log
```

## Model paths

Orbit's server helper automatically searches the default Hugging Face cache
paths created by `llama-server -hf`:

```text
~/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/<snapshot>/gemma-4-12B-it-Q4_K_M.gguf
~/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/<snapshot>/MTP/gemma-4-12b-it-Q8_0-MTP.gguf
```

If the model files are elsewhere, set:

```bash
MODEL_PATH=/path/to/gemma-4-12B-it-Q4_K_M.gguf
MTP_DRAFT_PATH=/path/to/gemma-4-12b-it-Q8_0-MTP.gguf
```

The helper starts `llama-server` on `http://127.0.0.1:18080`.

Stop it with:

```bash
scripts/gemma4-12b-server.sh stop
```

Check status with:

```bash
scripts/gemma4-12b-server.sh status
```

If you need a custom model path at startup:

```bash
MODEL_PATH=/path/to/gemma-4-12B-it-Q4_K_M.gguf \
MTP_DRAFT_PATH=/path/to/gemma-4-12b-it-Q8_0-MTP.gguf \
scripts/gemma4-12b-server.sh start --mtp
```

Override the detected paths if needed:

```bash
MODEL_PATH=/path/to/gemma-4-12B-it-Q4_K_M.gguf \
MTP_LLAMA_SERVER_BIN=/path/to/compatible/llama-server \
MTP_DRAFT_PATH=/path/to/gemma-4-12b-it-Q8_0-MTP.gguf \
scripts/gemma4-12b-server.sh start --mtp
```

See [MTP speculative decoding](docs/PERFORMANCE.md#mtp-speculative-decoding)
for benchmark notes.

## Run orbit

Interactive mode:

```bash
orbit
```

One-shot mode:

```bash
orbit "Say who you are in one short sentence."
```

Health check:

```bash
orbit --health
```

Show tool modes:

```bash
orbit /tools
```

If your server uses a different URL:

```bash
orbit --base-url http://127.0.0.1:18080
```

## Interactive commands

```text
/compact [tools]  Compact conversation memory or old tool results.
/continue         Continue the last answer if it reached max_tokens.
/health           Check llama-server health.
/help             Show this help.
/max-tokens [n]   Show or set output token limit for following turns.
/reset            Clear current conversation and saved session.
/sessions clear   Delete all saved sessions for this workdir.
/status [ctx]     Show runtime status or estimated context usage.
/tools [spec]     Show or set tools: off or on.
/exit             Exit interactive mode.
```

`/max-tokens <n>` affects only the current runtime. It does not rewrite config or session files.

## Tool mode

Orbit starts chat-only unless configured otherwise.

This is intentional. With tools off, Orbit does not send tool schemas and does
not enter the tool loop, so ordinary chat turns are lighter and cheaper during
prefill.

This experimental branch has a single operational mode:

```text
/tools off = chat only
/tools on  = unrestricted local shell for files, web, edits, system, and automation
```

Examples:

```text
/tools off
/tools on
```

`off` keeps Orbit in chat-only mode. `on` exposes only one model-facing tool:
`exec_shell_full_command`.

When tools are on, the model can run arbitrary shell commands from the
configured workdir. Commands may read, write, delete, execute programs, access
the network, and access paths outside the workdir. Use this mode only in a
disposable lab environment.

## Common usage profiles

For ordinary chat, keep tools disabled:

```text
/tools off
```

For local files, URLs, shell automation, system inspection, edits, or isolated
analysis work:

```text
/tools on
```

If a session starts feeling slow, check the active context:

```text
/status ctx
```

When old tool results dominate the context, compact only those results:

```text
/compact tools
```

## Shell tool mode

`/tools on` is an explicit lab mode for tasks that need local commands.
It runs from the configured `--workdir`.

Recommended startup:

```bash
orbit --workdir workdir
```

```text
/tools on
Inspect samples/suspicious_dropper_demo.js without executing it. Return suspicious URLs, IPs, encoded payloads, or execution-related strings.
```

Do not use `/tools on` on a normal working directory unless you accept the risk.
Use a disposable lab directory for malware analysis, reverse engineering, or
commands with side effects.

## Safety boundaries

- Tools are off by default.
- `/tools on` exposes only `exec_shell_full_command`.
- `exec_shell_full_command` is unrestricted except for timeout/output limits.
- The runtime does not sandbox or validate shell commands in this mode.
- Shell commands run from the configured `--workdir`, but may access paths outside it.
- Web content is fetched by shell commands such as `curl` when the model decides it is needed.
- Long files and long web pages are currently handled by shell output limits; future work may add shell-callable Orbit helpers for chunked reads and extracted web text.

Current read limits:

```text
complete read: up to 256 KB
chunk mode: files up to 1 MB
chunk size: 6k chars by default, 12k chars max
chunk calls: max 3 per user turn
```

## Images and audio

Images and audio are optional. Skip this section for normal text usage.

Start the server with multimodal support:

```bash
scripts/gemma4-12b-server.sh start --multimodal
```

If the projector is not in the default cache path:

```bash
MMPROJ_PATH=/path/to/mmproj-gemma-4-12B-it-Q8_0.gguf \
scripts/gemma4-12b-server.sh start --multimodal
```

Image one-shot:

```bash
orbit --image path/to/image.jpg "Describe this image in one short sentence."
```

Audio one-shot:

```bash
orbit --audio path/to/audio.wav "Transcribe this audio."
```

Supported image types: JPEG, PNG, WebP.

Supported audio types: WAV, MP3. Audio support in `llama.cpp` is experimental and can be slow.

## Config

Optional config file:

```bash
mkdir -p ~/.orbit
cat > ~/.orbit/config.json <<'JSON'
{
  "base_url": "http://127.0.0.1:18080",
  "workdir": ".",
  "timeout": 300,
  "temperature": 0,
  "max_tokens": 512,
  "tools": "off"
}
JSON
```

CLI flags override config values.

## Sessions and history

- Sessions are stored under `~/.orbit/sessions`.
- Prompt history is stored under `~/.orbit/history` when `readline` is available.
- Slash commands are not stored in prompt history.
- Duplicate prompts are collapsed.
- Long pasted prompts are displayed compactly but preserved internally.

## Troubleshooting

- `llama-server not found in PATH`: build `llama.cpp` and export `PATH=/path/to/llama.cpp/build/bin:$PATH`.
- `gemma-4-12B-it-Q4_K_M.gguf not found`: run `scripts/gemma4-12b-server.sh download` or set `MODEL_PATH`.
- `multimodal projector not found`: set `MMPROJ_PATH`.
- `MTP draft not found`: run `scripts/gemma4-12b-server.sh download --mtp` or set `MTP_DRAFT_PATH`.
- `unknown model architecture: gemma4-assistant`: use a compatible MTP fork via `MTP_LLAMA_SERVER_BIN`.
- `existing llama-server is not multimodal`: stop it, then restart with `start --multimodal`.
- Another process owns the port: stop it or change `PORT` / `BASE_URL`.

## Tests

```bash
python3 -m unittest discover -s tests -q
```

Manual regression prompts are kept in [docs/PROMPTS.md](docs/PROMPTS.md).

The public regression benchmark is available as `scripts/bench-core.sh`.
It uses the repository `workdir/` fixture by default.

Performance design notes are kept in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
