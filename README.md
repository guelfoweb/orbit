# orbit

Minimal local CLI for `llama-server`, focused on prompt cache, continuation, and multimodal local inference.

The current baseline targets `gemma4:12b` through `llama.cpp`/`llama-server` on CPU-only hardware.

## Status

Early prototype:

- Text chat through `llama-server` OpenAI-compatible API.
- One-shot local image input through `--image`.
- One-shot local audio input through `--audio`.
- In-memory interactive conversation.
- Streaming assistant output in interactive mode.
- Local config file support.
- Minimal session persistence.
- Native `llama-server` tool-call loop, with bounded Orbit-only tools where the server has no equivalent.
- Model-driven session memory refresh for long interactive sessions.
- No dependency beyond the Python standard library.

## Start llama-server

Recommended quick start:

```bash
scripts/orbit-gemma4-12b.sh
```

The script pulls `gemma4:12b` with Ollama if needed, resolves the local GGUF blob downloaded by Ollama, starts `llama-server` on `127.0.0.1:18080`, and opens Orbit interactive chat.

For image/audio input, start the same flow with the Ollama projector blob:

```bash
scripts/orbit-gemma4-12b.sh --multimodal
```

Main text/tool profile for `gemma4:12b` reusing the GGUF blob already downloaded by Ollama:

```bash
llama-server \
  -m /usr/share/ollama/.ollama/models/blobs/sha256-1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606 \
  -c 8192 \
  -t 6 \
  -b 128 \
  -ub 128 \
  -np 1 \
  --reasoning off \
  --cache-ram 8192 \
  --host 127.0.0.1 \
  --port 18080
```

Or start only the server with the bundled helper:

```bash
scripts/start-gemma4-12b.sh
```

A conservative 4K context helper is available if 8K is unstable on smaller machines:

```bash
scripts/start-gemma4-12b-c4k.sh
```

The start helpers share the same implementation and accept conservative environment overrides for local experiments:

```bash
THREADS=6 BATCH_SIZE=128 UBATCH_SIZE=128 CACHE_RAM=8192 scripts/start-gemma4-12b-c8k.sh
```

Prompt caching is enabled in Orbit requests. `llama-server` also enables prompt caching by default. Keep `CACHE_REUSE` unset unless you are benchmarking it explicitly:

```bash
CACHE_REUSE=256 scripts/start-gemma4-12b-c8k.sh
```

Compare cache changes with `scripts/bench-kv-cache.py` before keeping them.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .

orbit --base-url http://127.0.0.1:18080 "Say who you are in one short sentence."
```

Interactive mode:

```bash
orbit --base-url http://127.0.0.1:18080
```

Interactive sessions are persisted under `~/.orbit/sessions` and keyed by `--workdir`.

Prompt history is persisted under `~/.orbit/history` when `readline` is available. Use arrow up/down to recall previous prompts. Slash commands are not stored and duplicate prompts are collapsed.

Long interactive sessions are shortened automatically when the estimated transcript approaches context pressure. Orbit asks the model, internally and without tools, to produce a durable session memory. The internal request is not saved as a normal user-visible turn. The rebuilt session keeps:

```text
system prompt
model-generated session memory
recent verbatim tail
```

If the model returns an empty memory, tries a tool call, or the rebuilt session would not be smaller, Orbit keeps the original session unchanged.

The default memory-refresh threshold is 85% of Orbit's configured context estimate. This keeps CPU-only sessions from compacting too aggressively.

When a refresh happens, Orbit prints a compact metric line:

```text
memory: 2090->320 est. tokens | saved 1770 (85%) | 238.7s | threshold 1360/1600
```

Interactive commands:

```text
/health          Check llama-server health.
/help            Show commands.
/max-tokens      Show current output token limit.
/max-tokens <n>  Set output token limit for following turns.
/reset           Clear current conversation and saved session.
/status          Show runtime, session, and backend capabilities.
/tools           Show available local tools.
/exit            Exit interactive mode.
```

`/max-tokens <n>` changes only the current interactive runtime. It does not rewrite the config file or session.

Tool surface:

```text
llama-server:
read_file
write_file
file_glob_search
grep_search
exec_shell_command
edit_file
apply_diff
get_datetime

orbit-only:
make_directory
delete_path
fetch_url
search_web
```

Orbit first asks the model for a compact route decision, then exposes only the tool subset for that route. There are no deterministic task fast paths: if the model answers normally instead of requesting tools, Orbit returns that answer.

Interactive assistant responses stream as tokens arrive. Tool-call turns remain compatible with streaming: Orbit may first receive a tool call, execute the bounded local tool, then stream the final assistant response.

While waiting for the first streamed token, the terminal shows a dim `Working` indicator with a spinner and elapsed time. Tool phases are shown as compact dim events, including tool result size. Tool results above 10k characters are marked as `large context`. The final metrics footer is also dimmed, includes total turn time, and is separated from the assistant response.

`read_file` is intentionally limited to bounded UTF-8 text and source-code files. PDFs, images, audio, archives, and binary files are rejected by the tool contract.

Metadata requests are normally handled through bounded `exec_shell_command` calls such as `stat` when the server tool is available. Orbit keeps compact local metadata helpers as fallback implementation detail.

`make_directory` creates one directory inside the workdir, including missing parents. It refuses paths outside the workdir and does not replace existing files.

`delete_path` deletes one file or directory inside the workdir. Non-empty directories require `recursive=true`, and the workdir root is always refused.

`fetch_url` accepts only explicit `http`/`https` URLs. It uses a browser-like user-agent, bounded downloads, content-type filtering, and conservative HTML-to-text extraction. It never returns raw HTML. Long fetched pages use explicit `chunk_index` reads, without saving files to the workdir.

`search_web` is for explicit web search requests. It returns a small structured result list with title, URL, and snippet. It does not expose raw search HTML. Optional `site` filters accept bare domains only, and optional `timelimit` accepts `d`, `w`, `m`, or `y`.

`write_file` creates new UTF-8 text/source files only. It requires an explicit target path, refuses to overwrite existing paths, does not create parent directories implicitly, and is not used for normal chat answers such as "write a poem" or "write code" unless the user asks to save a file.

Incremental edits are normally handled through native `edit_file` or `apply_diff` with Orbit guardrail schemas. Older local append/replace helpers remain bounded fallback implementation details.

If a UTF-8 text/source file is too large for a complete `read_file`, the model may call `read_file` again with `chunk_index` to read real bounded chunks. Orbit does not summarize chunks deterministically: the model receives the actual chunk text and decides how to continue.

Current read limits:

```text
complete read: up to 256 KB
chunk mode: files up to 1 MB
chunk size: 12k chars by default, 24k chars max
chunk calls: max 3 per user turn
```

Current fetch limits:

```text
download: up to 512 KB
extracted text: up to 256k chars
chunk size: 6k chars by default, 24k chars max
chunk calls: max 3 explicit chunk reads per user turn
```

Long web documents:

For long URLs, `fetch_url` returns `chunk_index`, `total_chunks`, and the current character range. A complete summary should be built progressively by reading additional chunks. Orbit does not save fetched pages into the workdir and does not pretend that the first chunk represents the whole document.

Tool-call smoke test:

```bash
scripts/smoke-tools.sh
```

Memory-refresh benchmark:

```bash
scripts/bench-memory-refresh.sh
```

The benchmark uses `--context-tokens` to lower Orbit's runtime context estimate without changing the server context window.

KV-cache probe:

```bash
scripts/bench-kv-cache.py
```

The probe sends consecutive chat turns with a stable prefix and reports prompt tokens, cached tokens, prefill speed, generation speed, wall time, and stop reason.

Example output shape:

| turn | prompt | cached | cache | pf/s | gen/s | wall | finish |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 1119 | 6 | 1% | 12.2 | 2.8 | 101.1s | stop |
| 2 | 1172 | 1115 | 95% | 8.8 | 2.9 | 18.4s | stop |
| 3 | 1228 | 1168 | 95% | 9.8 | 3.0 | 12.1s | stop |

Tool-loop cache probe:

```bash
scripts/bench-tool-cache.py
```

This reports cache metrics for each model call inside tool turns, separating tool-call selection from the final answer after tool results are injected. Use `--include-large` only when you intentionally want to measure slow chunked reads.

Continuation comparison probe:

```bash
scripts/bench-continuation-cache.py
```

This compares no-tool chat turns with tool-call turns using the same Orbit runtime metrics. Use it to distinguish generic chat-template continuation behavior from tool-specific cache misses.

Raw cache probe:

```bash
scripts/bench-raw-cache.py --mode all
```

This bypasses Orbit runtime and calls `llama-server` directly. It compares raw multi-message chat, raw tool replay, and a monolithic single-message transcript. Use it to distinguish Orbit behavior from server/template/cache-matching behavior.

Manual regression prompts are kept in [PROMPTS.md](PROMPTS.md).

Optional config:

```bash
mkdir -p ~/.orbit
cat > ~/.orbit/config.json <<'JSON'
{
  "base_url": "http://127.0.0.1:18080",
  "model": "gemma4:12b",
  "workdir": ".",
  "timeout": 300,
  "temperature": 0,
  "max_tokens": 512
}
JSON
```

Then run:

```bash
orbit "Say who you are in one short sentence."
```

CLI flags override config values.

## Vision

Attach one or more local images to a one-shot prompt:

```bash
orbit --image path/to/image.jpg "Describe this image in one short sentence."
```

Supported image types: JPEG, PNG, and WebP. The active `llama-server` must be started with a compatible multimodal projector.

Vision smoke test:

```bash
IMAGE=path/to/image.jpg scripts/smoke-vision.sh
```

## Audio

Attach one or more local audio files to a one-shot prompt:

```bash
orbit --audio path/to/audio.wav "Transcribe this audio."
```

Supported audio types: WAV and MP3. Audio support in `llama.cpp` is experimental and slower than text or image prompts.

Audio smoke test:

```bash
AUDIO=path/to/audio.wav scripts/smoke-audio.sh
```

## Benchmark

Run the same small prompt suite against the active `llama-server`:

```bash
scripts/bench-chat.sh
```

Run the slower long-prefix cache check:

```bash
CACHE_BENCH=1 scripts/bench-chat.sh
```

Run the persisted-session check:

```bash
scripts/bench-session.sh
```

## Design rules

- Start chat-only.
- Add one capability at a time.
- Measure prompt cache and continuation before adding tools.
- Keep the runtime small, explicit, and easy to test.
