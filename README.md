# orbit

Version: `0.1.0`

Minimal interactive CLI that uses Ollama tool calling with a small set of local tools.
The default runtime stays small: low prompt overhead, bounded loops, compact tool-oriented context, and optional local vision/audio paths for explicit local files.
`orbit` is tuned primarily around `gemma4:e2b` and its local Ollama behavior; other models may work, but comparable reliability is not guaranteed.

Available local tools:

- `read_file`
- `list_files`
- `stat_path`
- `replace_in_file`
- `write_file`
- `append_file`
- `bash`
- `search_web`
- `fetch_url` for explicit known URLs
- optional `SKILL.md` import

The goal is to stay simple, predictable, and easy to debug.

## Layout

- `src/orbit/core/`: agent loop, Ollama client, compaction, runtime, sessions, turn policy, tool routing, context budget, loop guard
- `src/orbit/tooling/`: tool registry and local tools by domain
- `src/orbit/terminal/`: CLI, config parsing, history, and terminal rendering

## Requirements

- Python 3.10+
- Ollama running locally
- Python package `ollama` installed through the project dependency
- Python package `rich` installed through the project dependency for Markdown rendering of final model replies
- A model that supports tool calling; for local image/audio inspection, a model that also advertises `vision` and/or `audio`

Optional media dependencies:

- Vision requires the Python package `Pillow`, installed through the project dependency.
- Audio requires the system binaries `ffmpeg` and `ffprobe`; without them, audio prompts fail with a clear error instead of sending unstable raw audio to Ollama.

Install audio dependencies on Debian/Ubuntu/Linux Mint:

```bash
sudo apt install ffmpeg
```

## Install

```bash
cd orbit
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Run

```bash
.venv/bin/orbit --workdir .
```

With a skill:

```bash
.venv/bin/orbit --workdir . --skill malware-analysis-static
```

You can pass either:

- a skill name resolved from `~/.orbit/skills` or `./skills`
- a directory containing `SKILL.md`
- a direct path to `SKILL.md`

You can also point to a different Ollama endpoint:

```bash
.venv/bin/orbit --base-url http://127.0.0.1:11434
```

Long-running turns:

- use `--timeout <seconds>` if a model or task needs more time
- example: `orbit --timeout 600`
- use `--think on|off|auto` to control thinking-capable models
- use `--show-thinking` to render the reasoning trace in the terminal before the final answer
- the `gemma4:e2b` model-first profile defaults to `think=off` in `auto` mode
- the default loop budget is conservative; raise it with `--max-loops` only when the task genuinely needs more agentic depth

Performance notes for Ollama:

- Runtime env such as `OLLAMA_NUM_PARALLEL` and `OLLAMA_KEEP_ALIVE` must be set on the Ollama server process.
- If Ollama runs as `ollama.service`, exporting those variables before `orbit` has no effect on the server.
- Recommended CPU-only server env for this project: `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_KEEP_ALIVE=-1`, optionally `OLLAMA_MAX_LOADED_MODELS=1`.
- Keep the tuned model parameters in `Modelfile.gemma4-e2b-fast-t6-c8k`: `num_ctx 8192`, `num_thread 6`, `num_batch 96`.
- On laptops/NUCs, CPU governor or power profile can dominate token throughput; use `performance` when benchmarking.

Check the model currently loaded by Ollama:

```bash
curl -s http://127.0.0.1:11434/api/ps | jq
```

Example CPU-only output:

```json
{
  "models": [
    {
      "name": "gemma4:e2b-fast-t6-c8k",
      "model": "gemma4:e2b-fast-t6-c8k",
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

- `expires_at` far in the future means the model is effectively kept loaded.
- `size_vram: 0` means the model is running on system RAM/CPU, not GPU VRAM.
- `context_length` should match the tuned context window.

Vision:

- `orbit` can inspect explicit local image paths in prompts such as:
  - `describe the image workdir/cat.png`
  - `compare two images: images/vision-test-1.png and images/vision-test-2.jpg and tell me the differences`
- supported formats: `png`, `jpg`, `jpeg`, `webp`, `bmp`, `gif`
- images are normalized before being sent to Ollama:
  - converted to `RGB`
  - alpha flattened
  - large images resized conservatively
- this keeps the vision path bounded and avoids known runner instability on some raw image inputs

Audio:

- `orbit` can inspect explicit local audio paths in prompts such as:
  - `transcribe audio/voice-sample.wav`
  - `summarize audio/voice-sample.wav in one sentence`
- requires `ffmpeg` and `ffprobe` on `PATH`
- supported source formats: `wav`, `mp3`, `m4a`, `flac`, `ogg`
- audio is normalized with `ffmpeg` before being sent to Ollama:
  - WAV PCM 16-bit
  - mono
  - 16 kHz
  - 5 second chunks
- this keeps the audio path bounded; long single audio attachments are avoided because they can crash or timeout the local runner
- if `ffmpeg` or `ffprobe` is missing, Orbit does not attempt audio inspection

Model selection:

- if you pass `--model`, `orbit` uses that model
- if you omit `--model`, `orbit` uses the first model already running in Ollama
- if no model is running, `orbit` exits with an explicit error
- if the selected model does not advertise `tools` capability via Ollama model details, `orbit` warns and falls back to chat-only mode
- the main runtime and prompt strategy are optimized around `gemma4:e2b`; other models are best-effort only

Session storage:

- sessions are saved under `~/.orbit/sessions/`
- on interactive startup, `orbit` looks for existing sessions for the current `--workdir`
- if sessions exist, it shows their names plus the first prompt line and lets you choose one or start a new session
- new session names are derived from the current `--workdir`
- you can override it with `--session <name>`

## Interactive commands

- `/help`
- `/tools`
- `/status`
- `/debug`
- `/skill clear | list | show | use <name-or-path>`
- `/think on|off|auto`
- `/thinking on|off`
- `/compact`
- `/reset`
- `/exit`

Input history:

- arrow up/down scroll through previous prompts
- history is persisted in `~/.orbit/history`
- very long pasted prompts are collapsed visually in the terminal as `[text N chars]`
- the full original text is still sent to the model; only the REPL rendering is shortened

Orbit home:

- `~/.orbit/history` stores prompt history
- `~/.orbit/skills/` is the native skill directory
- `~/.orbit/sessions/` stores persistent chat sessions
- a bundled `orbit-default` skill is auto-loaded when no explicit skill is selected
- the bundled default skill is intentionally short so the default path stays lean

## Notes

- File access is restricted to `--workdir`.
- `read_file` is bounded by line and character limits.
- `stat_path` returns bounded filesystem metadata such as type, size, modified time, mode, and newest directory entries.
- `replace_in_file` is the preferred patch-first edit tool for focused updates to existing text files.
- first writes to existing unread files are guarded: `orbit` asks the model to `read_file` first before overwriting
- identical read-only tool calls can be served from a small session cache to reduce wasted latency
- tools that fail repeatedly in the same session are gradually de-prioritized or dropped from the exposed schema
- `bash` runs with `shell=False` and blocks shell operators and common destructive commands.
- `search_web` performs bounded generic web search and returns structured `title`/`url`/`snippet` results.
- `fetch_url` performs bounded HTTP fetches for explicit known URLs and returns structured text, title, final URL, and links. It is not a general web search tool.
- local image inspection is not a separate tool; it is a bounded multimodal path triggered only by explicit image paths in the prompt
- `orbit` uses a 2-stage tool routing strategy: it first narrows the tool category (`filesystem`, `write`, `shell`, `web`), then exposes only that subset to the model for the turn.
- auto-compaction uses an explicit context budget engine with one conservative internal profile tuned for the main `gemma4:e2b` path.
- Tool execution is handled client-side in a standard multi-turn loop through the Ollama Python SDK.
- When a skill is active, its `SKILL.md` content is appended to the system prompt.
- `/skill clear` restores the default `orbit-default` skill.
- `/compact` replaces older messages with a local structured memory summary and keeps the most recent raw messages intact.

## Test

```bash
cd orbit
python3 -m unittest discover -s tests -v
```
