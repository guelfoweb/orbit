# orbit

Minimal local agentic CLI for `llama.cpp` / `llama-server`.

Orbit is designed around `gemma4:12b-it`, local execution, prompt-cache awareness, streaming output, bounded tools, and CPU-only usability.

It is developed and tested primarily on Linux. macOS may work if `llama-server` and the model files are available. Windows is not a target environment.

## What it does

- Chat with a local `llama-server` model.
- Stream assistant responses in the terminal.
- Enable or disable tool groups at runtime.
- Read, inspect, edit, search, fetch URLs, and run bounded read-only shell commands.
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

Use this build for both normal mode and optional MTP mode:

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

Use the Orbit helper to download the expected GGUF file into the standard
Hugging Face cache:

```bash
scripts/gemma4-12b-server.sh download
```

Internally this uses:

```bash
llama-server -hf ggml-org/gemma-4-12B-it-GGUF --hf-file gemma-4-12B-it-Q4_K_M.gguf
```

If `llama-server` starts after the download, stop it with `Ctrl+C`.

### 5. Start Orbit

Start the tuned local server:

```bash
scripts/gemma4-12b-server.sh start
```

Then start Orbit:

```bash
orbit
```

Inside Orbit, tools start disabled. Enable only what you need:

```text
/tools files
```

If something fails, run:

```bash
scripts/gemma4-12b-server.sh status
tail -n 80 ~/.orbit/gemma4-12b-server.log
```

## Model paths

Orbit's server helper automatically searches the default Hugging Face cache path
created by `llama-server -hf`:

```text
~/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/<snapshot>/gemma-4-12B-it-Q4_K_M.gguf
```

If the model is elsewhere, set:

```bash
MODEL_PATH=/path/to/gemma-4-12B-it-Q4_K_M.gguf
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
MODEL_PATH=/path/to/gemma-4-12B-it-Q4_K_M.gguf scripts/gemma4-12b-server.sh start
```

## Optional MTP

MTP speculative decoding is optional. Skip it for the first run.

It requires:

- the Gemma 4 compatible `llama-server` built above;
- the main `gemma-4-12B-it-Q4_K_M.gguf` model;
- the separate draft `gemma-4-12b-it-Q8_0-MTP.gguf` model.

The draft model is not downloaded with the main `ggml-org` model.

Download the main model and the tested draft model:

```bash
scripts/gemma4-12b-server.sh download --mtp
```

The draft model comes from the Unsloth repository:

```bash
llama-server -hf unsloth/gemma-4-12b-it-GGUF --hf-file MTP/gemma-4-12b-it-Q8_0-MTP.gguf
```

If `llama-server` starts after the download, stop it with `Ctrl+C`.

The helper automatically searches the default Hugging Face cache path:

```text
~/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/<snapshot>/MTP/gemma-4-12b-it-Q8_0-MTP.gguf
```

Then start the server with MTP:

```bash
scripts/gemma4-12b-server.sh start --mtp
```

For MTP, the helper uses `MTP_LLAMA_SERVER_BIN` when provided. It also checks
common local build paths such as:

```text
~/LAB/llama.cpp-gemma4/build/bin/llama-server
~/LAB/llama.cpp-gemma4-mtp-qualcomm/build/bin/llama-server
```

See [MTP speculative decoding](docs/PERFORMANCE.md#mtp-speculative-decoding)
for the tested fork/branch and benchmark notes.

Override the detected paths if needed:

```bash
MTP_LLAMA_SERVER_BIN=/path/to/compatible/llama-server \
MTP_DRAFT_PATH=/path/to/gemma-4-12b-it-Q8_0-MTP.gguf \
scripts/gemma4-12b-server.sh start --mtp
```

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
orbit /health
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
/tools [spec]     Show or set tools: off, on, files, edit, web, shell, shell-full.
/exit             Exit interactive mode.
```

`/max-tokens <n>` affects only the current runtime. It does not rewrite config or session files.

## Tool modes

Orbit starts chat-only unless configured otherwise.

This is intentional. With tools off, Orbit does not send tool schemas and does
not enter the tool loop, so ordinary chat turns are lighter and cheaper during
prefill. Enable only the tool groups needed for the current task.

```text
/tools files = read/inspect local files
/tools edit  = create/modify/delete files or directories
/tools web   = search/fetch URLs
/tools shell = read-only local/system commands
/tools shell-full = DANGEROUS unrestricted local shell
```

Examples:

```text
/tools off
/tools files
/tools files,web
/tools on
/tools shell-full
```

`off` keeps Orbit in chat-only mode. `on` enables all standard safe groups.
`shell-full` is not included in `on`; it must be enabled explicitly.

`shell-full` gives the model unrestricted local shell access from the configured
workdir. It can run pipes, redirects, malware tooling, decompilers, network
commands, writes, deletes, and commands that access paths outside the workdir.
Use it only in a disposable lab environment.

## Shell-full mode

`shell-full` is an explicit lab mode for tasks that cannot be handled by the
bounded read-only shell tool.

Use it for isolated analysis workflows where the model may need arbitrary local
commands, for example static inspection with tools such as `strings`, `file`,
`readelf`, `objdump`, `jadx`, `apktool`, or other utilities available on the
host.

Enable it only when you want to give the model unrestricted shell access:

```text
/tools shell-full
```

Important boundaries:

- It is disabled by default.
- It is not included in `/tools on`.
- It runs from the configured `--workdir`.
- It may read, write, delete, execute programs, access the network, and access paths outside `--workdir`.
- Orbit still applies timeout and output-size limits.
- The runtime does not sandbox or validate commands in this mode.

Recommended usage:

```bash
orbit --workdir workdir
```

```text
/tools shell-full
Inspect samples/suspicious_dropper_demo.js without executing it. Return suspicious URLs, IPs, encoded payloads, or execution-related strings.
```

Do not use `shell-full` on a normal working directory unless you accept the
risk. Use a disposable lab directory for malware analysis, reverse engineering,
or commands with side effects.

## Safety boundaries

- Tools are opt-in by group.
- Local paths are confined to the configured `--workdir`.
- `read_file` accepts UTF-8 text/source files only.
- PDFs, images, audio, archives, and binary files are rejected by `read_file`.
- `write_file` creates new files only and refuses overwrites.
- `edit_file` and `apply_diff` are bounded edit paths.
- `exec_shell_command` is allowlisted and read-only oriented.
- `exec_shell_full_command` is available only through `/tools shell-full` and is unrestricted except for timeout/output limits.
- `fetch_url` accepts explicit `http`/`https` URLs and returns extracted text, not raw HTML.
- Long files and long fetched pages use explicit chunk reads.

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
