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

## Install llama.cpp

Build `llama.cpp` for CPU-only use:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
cmake --build build --config Release -j"$(nproc)"
```

Optionally add the binaries to `PATH`:

```bash
export PATH=/path/to/llama.cpp/build/bin:$PATH
```

Verify:

```bash
llama-server --version
```

## Download the model

Use `llama-server` once to download the expected GGUF file:

```bash
llama-server -hf ggml-org/gemma-4-12B-it-GGUF \
  --hf-file gemma-4-12B-it-Q4_K_M.gguf
```

Stop it with `Ctrl+C` after the download finishes.

Orbit's server helper automatically searches the default Hugging Face cache path:

```text
~/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/<snapshot>/gemma-4-12B-it-Q4_K_M.gguf
```

If the model is elsewhere, set:

```bash
MODEL_PATH=/path/to/gemma-4-12B-it-Q4_K_M.gguf
```

## Install orbit

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Check:

```bash
orbit --version
```

## Start the server

Use the helper script:

```bash
scripts/gemma4-12b-server.sh start
```

Then start Orbit:

```bash
orbit
```

The helper starts `llama-server` on:

```text
http://127.0.0.1:18080
```

Stop the server:

```bash
scripts/gemma4-12b-server.sh stop
```

Check status:

```bash
scripts/gemma4-12b-server.sh status
```

If you need a custom model path:

```bash
MODEL_PATH=/path/to/gemma-4-12B-it-Q4_K_M.gguf scripts/gemma4-12b-server.sh start
```

Optional MTP speculative decoding requires a compatible `llama-server` build
and a matching draft model:

```bash
LLAMA_SERVER_BIN=/path/to/compatible/llama-server \
MTP_DRAFT_PATH=/path/to/gemma-4-12B-it-MTP-Q8_0.gguf \
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
/health          Check llama-server health.
/help            Show commands.
/max-tokens [n]  Show or set output token limit for following turns.
/reset           Clear current conversation and saved session.
/status          Show runtime, session, and backend capabilities.
/tools [spec]    Show or set tools: off, on, files, edit, web, shell.
/exit            Exit interactive mode.
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
```

Examples:

```text
/tools off
/tools files
/tools files,web
/tools on
```

`off` keeps Orbit in chat-only mode. `on` enables all supported groups.

## Safety boundaries

- Tools are opt-in by group.
- Local paths are confined to the configured `--workdir`.
- `read_file` accepts UTF-8 text/source files only.
- PDFs, images, audio, archives, and binary files are rejected by `read_file`.
- `write_file` creates new files only and refuses overwrites.
- `edit_file` and `apply_diff` are bounded edit paths.
- `exec_shell_command` is allowlisted and read-only oriented.
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
- `gemma-4-12B-it-Q4_K_M.gguf not found`: download with `llama-server -hf ... --hf-file ...` or set `MODEL_PATH`.
- `multimodal projector not found`: set `MMPROJ_PATH`.
- `existing llama-server is not multimodal`: stop it, then restart with `start --multimodal`.
- Another process owns the port: stop it or change `PORT` / `BASE_URL`.

## Tests

```bash
python3 -m unittest discover -s tests -q
```

Manual regression prompts are kept in [PROMPTS.md](PROMPTS.md).

Benchmark helpers are available under `scripts/` for cache, tool-loop, continuation, memory-refresh, and chat timing checks.

Performance design notes are kept in [PERFORMANCE.md](PERFORMANCE.md).
