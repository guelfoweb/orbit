# orbit

Orbit is a minimal local CLI for Ollama tool calling.

It is an experimental project focused on small local Gemma4 runtimes, not a general-purpose agent framework.

It provides an interactive REPL, persistent sessions, a small set of bounded local tools, optional skills, and explicit local vision/audio paths. The runtime is designed to stay lean: low prompt overhead, narrow tool exposure, bounded loops, and compact context handling.

Orbit is developed primarily around `gemma4:e2b`, using a tuned Ollama profile called:

```text
gemma4:e2b-c8k
```

Other `gemma4` profiles may work, but they are best-effort and are not guaranteed to behave with the same reliability.

## Requirements

- Linux is the primary supported environment. macOS may work, but it is currently best-effort and not part of the tested target. Windows is not supported.
- Python 3.10+
- Git
- Ollama running locally
- A Gemma4 model with tool-call support
- `ffmpeg` and `ffprobe` only if you want audio support
- `pypdf` is installed automatically with Orbit and is used for bounded PDF text extraction

Install optional audio dependencies on Debian/Ubuntu/Linux Mint:

```bash
sudo apt install ffmpeg
```

## Install

Install Orbit CLI:

```bash
curl -fsSL https://raw.githubusercontent.com/guelfoweb/orbit/main/install.sh | sh
```

The installer:

- clones or updates the repository in `~/.local/share/orbit`
- creates `~/.local/share/orbit/.venv`
- installs the Python package
- links the executable to `~/.local/bin/orbit`
- creates `~/.orbit/config.json` only if it does not already exist
- does not create or download Ollama models

If `orbit` is not found after installation, make sure `~/.local/bin` is in your `PATH`.

## Set up gemma4:e2b

After installing Orbit, create the local model profiles from the local checkout:

```bash
~/.local/share/orbit/tune-e2b.sh
```

The script:

- checks that `ollama` is installed
- pulls `gemma4:e2b` if missing
- creates `gemma4:e2b-c8k` from `Modelfile.gemma4-e2b-c8k`
- creates `gemma4:e2b-c4k` from `Modelfile.gemma4-e2b-c4k`
- prints the final launch command

This creates two Ollama profiles that share the same base model, so they do not duplicate the full model storage:

| Profile | Target machine | When to use |
| --- | --- | --- |
| `gemma4:e2b-c8k` | Intel NUC 10 class CPU-only machine, Intel i7-10710U, 6 physical cores / 12 threads, 64 GB RAM | Default Orbit profile and benchmark target |
| `gemma4:e2b-c4k` | Intel Xeon E3-1275 v6 CPU-only machine, 4 physical cores / 8 threads, about 16 GB RAM | Conservative fallback for smaller machines or Ollama GGML scheduler crashes |

Manual setup is also possible:

```bash
ollama create gemma4:e2b-c8k -f Modelfile.gemma4-e2b-c8k
ollama create gemma4:e2b-c4k -f Modelfile.gemma4-e2b-c4k
```

Profile details:

`Modelfile.gemma4-e2b-c8k`

```text
FROM gemma4:e2b

# Intel i7-10710U, 6 physical cores / 12 threads, 64 GB RAM.
PARAMETER temperature 0
PARAMETER num_ctx 8192
PARAMETER num_thread 6
PARAMETER num_batch 96
```

`Modelfile.gemma4-e2b-c4k`

```text
FROM gemma4:e2b

# Intel Xeon E3-1275 v6, 4 physical cores / 8 threads, about 16 GB RAM.
PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER num_thread 4
PARAMETER num_batch 64
```

## Run

Start Orbit:

```bash
orbit --model gemma4:e2b-c8k
```

If `gemma4:e2b-c8k` fails at startup or on the first prompt with an Ollama error like:

```text
llama-server process has terminated: GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed
```

try the conservative profile:

```bash
orbit --model gemma4:e2b-c4k
```

That error comes from the Ollama/llama.cpp runner, not from Orbit. It usually means the current model profile is too aggressive for the machine or runner combination. The first values to reduce are `num_ctx` and `num_batch`.

If `~/.orbit/config.json` already defines the model, you can simply run:

```bash
orbit
```

Run inside a specific workspace:

```bash
orbit --workdir /path/to/workspace
```

Run a one-shot prompt:

```bash
orbit "list all files in the current workspace"
```

Use a skill:

```bash
orbit --workdir . --skill notes
```

A skill can be:

- a name resolved from `~/.orbit/skills` or `./skills`
- a directory containing `SKILL.md`
- a direct path to `SKILL.md`

## User config

Orbit can load default options from:

```text
~/.orbit/config.json
```

The file is optional. CLI flags override matching config values.

Recommended config:

```json
{
  "model": "gemma4:e2b-c8k",
  "host": "http://127.0.0.1:11434",
  "workdir": ".",
  "timeout": 300,
  "think": "off",
  "debug_timing": false,
  "ui": {
    "markdown": true,
    "collapse_long_input": true,
    "long_input_preview_chars": 50
  },
  "tools": {
    "max_loops": 10
  }
}
```

Keep project-specific behavior in skills, not in this global config. The bundled `orbit-default` skill is loaded automatically when no skill is selected.

## Ollama performance notes

Runtime environment variables must be set on the Ollama server process, not only before launching Orbit.

If Ollama runs as a systemd service, exporting these variables in the same shell used to start `orbit` has no effect on the already-running server. In that case, configure the service itself.

Recommended CPU-only settings:

```text
OLLAMA_NUM_PARALLEL=1
OLLAMA_KEEP_ALIVE=-1
OLLAMA_MAX_LOADED_MODELS=1
```

Check the active service configuration:

```bash
systemctl cat ollama
systemctl show ollama -p Environment
```

If the environment only shows `PATH=...`, the recommended Ollama settings are not active.

Create a systemd drop-in:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Verify that the override is loaded:

```bash
systemctl cat ollama
systemctl show ollama -p Environment
```

The output should include:

```text
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
```

Check the loaded model:

```bash
curl -s http://127.0.0.1:11434/api/ps | jq
```

Example CPU-only output:

```json
{
  "models": [
    {
      "name": "gemma4:e2b-c8k",
      "model": "gemma4:e2b-c8k",
      "size": 7679746272,
      "details": {
        "format": "gguf",
        "family": "gemma4",
        "families": ["gemma4"],
        "parameter_size": "5.1B",
        "quantization_level": "Q4_K_M"
      },
      "expires_at": "2318-09-09T17:43:43.936212989+02:00",
      "size_vram": 0,
      "context_length": 8192
    }
  ]
}
```

On laptops and NUC-class machines, CPU governor or power profile can affect throughput more than `num_ctx`, `num_thread`, or `num_batch`.

## Local tools

Orbit exposes only a narrow tool subset for each turn.

Available local tools:

- `read_file`
- `list_files`
- `stat_path`
- `replace_in_file`
- `write_file`
- `append_file`
- `bash`
- `search_web`
- `fetch_url`

## Safety

Orbit can read and modify files inside the configured `workdir` and can run bounded shell commands through guarded tool calls.

Use it only inside workspaces you control. Keep the `workdir` narrow, review file edits before trusting them, and do not point Orbit at directories containing secrets unless that is explicitly part of the task.

The guardrails reduce risk, but they do not make local agentic execution risk-free.

Tool access is bounded:

- filesystem access is confined to `--workdir`
- file reads enforce line and character limits
- writes are size-limited
- overwriting unread existing files is guarded
- `bash` uses `shell=False`
- shell operators, redirects, chaining, and common destructive commands are blocked
- web search and URL fetches return structured bounded text, not raw unbounded HTML

## Vision

Orbit can inspect explicit local image paths:

```text
describe the image images/cat.png
compare two images: image1.png and image2.jpg
```

Supported formats:

- `png`
- `jpg`
- `jpeg`
- `webp`
- `bmp`
- `gif`

Images are normalized before being sent to Ollama: RGB conversion, alpha flattening, and conservative resizing for large inputs.

## Audio

Orbit can inspect explicit local audio paths:

```text
transcribe audio/voice-sample.wav
summarize audio/voice-sample.wav in one sentence
```

Supported source formats:

- `wav`
- `mp3`
- `m4a`
- `flac`
- `ogg`

Audio requires `ffmpeg` and `ffprobe`. Files are normalized to WAV PCM 16-bit, mono, 16 kHz, and split into 5 second chunks before being sent to the model.

If `ffmpeg` or `ffprobe` is missing, Orbit fails clearly instead of sending unstable raw audio to Ollama.

## Sessions

Sessions are stored under:

```text
~/.orbit/sessions/
```

On interactive startup, Orbit looks for existing sessions matching the current `workdir`. If more than one exists, it lets you choose one or start a new session.

Useful commands:

- `/reset`: reset the current session
- `/compact`: compact older context into a structured memory summary
- `/sessions clear`: delete sessions for the current `workdir`

## Skills

Skills are optional `SKILL.md` instruction files.

Default lookup roots:

- `~/.orbit/skills`
- `./skills`

Useful commands:

- `/skill list`
- `/skill show`
- `/skill use <ref>`
- `/skill clear`

`/skill clear` restores the bundled `orbit-default` skill.

## Interactive commands

- `/help`
- `/tools`
- `/status`
- `/debug`
- `/skill clear | list | show | use <ref>`
- `/think on|off|auto`
- `/thinking on|off`
- `/compact`
- `/reset`
- `/sessions clear`
- `/exit`

Input history is persisted in:

```text
~/.orbit/history
```

Very long pasted prompts are visually collapsed in the terminal as `[text N chars]`, but the full original text is still sent to the model.

## Routing and intent gate

Orbit does not expose every tool on every turn.

The runtime first classifies the user request, then exposes only the smallest useful tool subset. Ambiguous high-risk routes can pass through a strict YES/NO intent gate before tools are shown to the model.

Turn flow:

```text
prompt -> intent routing -> optional intent gate -> minimal tool subset -> model
```

Examples:

- `show me how grep works` stays chat/knowledge and does not expose `bash`.
- `tell me about file systems` stays chat/knowledge and does not expose filesystem tools.
- `decode this string "Y2lhbw==" from base64` uses a bounded local transform.
- `search online for information about Dante Alighieri` exposes web tools only.
- `read README.md` exposes filesystem tools only.
- `compare image1.png and image2.jpg` uses the bounded vision path.

The goal is to reduce ambiguity for small local models: fewer exposed tools, fewer wrong tool calls, fewer loops, and lower token waste.

## Benchmark hardware

The current regression benchmarks were run on a CPU-only Intel NUC 10 class machine:

- Intel Core i7-10710U
- 6 cores / 12 threads
- 64 GB RAM
- no GPU acceleration
- Ollama serving the model from system RAM

See [BENCHMARK.md](BENCHMARK.md) for the current prompt suite and observed timings.

## Project layout

- `src/orbit/core/`: agent loop, Ollama client, runtime, sessions, routing, compaction, guardrails
- `src/orbit/tooling/`: local tool registry and tool implementations
- `src/orbit/terminal/`: CLI, config parsing, history, and terminal rendering

## License

MIT. See [LICENSE](LICENSE).

## Test

Run the unit test suite:

```bash
python3 -m unittest discover -s tests -q
```
