# AGENTS.md

## Scope

This repository contains the new `orbit` CLI built around `llama.cpp` / `llama-server`.

The tool name remains `orbit`. The repository directory may be temporary.

## Hard rules

- Keep the runtime small, readable, and easy to debug.
- Do not turn Orbit into a generic agent framework.
- Do not add deterministic fast paths for user tasks.
- The model must decide whether to use available tools.
- The runtime may enforce safety, size, context, and tool-contract boundaries.
- Prefer standard-library Python unless a dependency has clear value.
- Keep one-shot, REPL, backend, runtime, tools, and terminal UI separated.
- Do not reintroduce Ollama-specific logic in this codebase.
- Do not expose broad shell, editing, web, PDF, or browser tools until explicitly designed.
- After runtime/tool/session changes, run unit tests.

## Backend

- Supported backend: local `llama-server`.
- API style: OpenAI-compatible chat completions.
- Default base URL: `http://127.0.0.1:18080`.
- Primary target model: `gemma4:12b`.
- Reasoning should be disabled at server startup for the current baseline.
- Context profiles are provided by helper scripts, not hidden runtime magic.

## Tools

Preferred model-facing tools:

- `read_file`
- `write_file`
- `file_glob_search`
- `grep_search`
- `exec_shell_command`
- `edit_file`
- `apply_diff`
- `get_datetime`

Orbit-only tools where `llama-server` has no equivalent:

- `make_directory`
- `delete_path`
- `fetch_url`
- `search_web`

Rules:

- Tools are exposed only in interactive text mode.
- Route selection must be model-driven before tool exposure.
- No deterministic task fast paths.
- `read_file` reads UTF-8 text/source files only.
- Metadata inspection should prefer bounded native shell commands when available.
- `make_directory` creates only directories confined to `workdir`.
- `delete_path` deletes only paths confined to `workdir`; non-empty directories require `recursive=true`.
- `delete_path` must refuse to delete the `workdir` root.
- `fetch_url` accepts only explicit http/https URLs and returns bounded extracted text, never raw HTML.
- `search_web` returns bounded structured results only: title, URL, and snippet.
- `search_web` optional `site` filters must be bare domains; optional `timelimit` must be one of d/w/m/y.
- `write_file` creates new UTF-8 text/source files only; it must not overwrite existing paths.
- `write_file` must not create parent directories implicitly.
- `write_file` is for explicit save/create-file requests, not ordinary chat requests to write prose or code.
- `edit_file` and `apply_diff` must use Orbit guardrail schemas, not broad raw server schemas.
- Long fetched pages use `fetch_url` with `chunk_index`; web content must not be silently saved into the workdir.
- Do not present a first fetched chunk as a complete summary of a long document.
- PDFs, images, audio, archives, and binary files must be rejected by `read_file`.
- Complete `read_file` is limited to 256 KB.
- Larger text/source files use `read_file` with `chunk_index`.
- Chunk mode is limited to 1 MB files, 12k chars default chunk size, 24k chars max, and 3 chunk reads per user turn.
- Unknown tools must fail clearly.

## Session memory

Orbit uses model-driven session memory refresh for long sessions.

Rules:

- The model generates the memory.
- The internal memory request must not be saved as a visible session turn.
- Memory refresh must run without tools.
- Rebuilt history keeps:
  - system prompt
  - model-generated session memory
  - recent verbatim tail
- If memory generation is empty, attempts tool calls, fails, or does not reduce context, keep the original history unchanged.
- Avoid back-to-back memory refreshes; a successful refresh must enter a short message-count cooldown.
- Memory refresh events must show before/after estimated tokens, saved ratio, elapsed seconds, and threshold/window.
- Default memory refresh threshold is 85% of the configured context estimate; lower it only with benchmark evidence.
- Do not summarize user prompts at ingestion time.
- Do not replace the current turn with a rewritten prompt.

## UX

- Classic terminal UX only.
- No full-screen TUI.
- Errors must be short and concrete.
- Prompt history should use `readline` when available, support arrow up/down, persist by workdir, and avoid duplicates.
- Slash commands must not be stored in prompt history.
- `/status` should expose useful runtime state without becoming noisy.
- `/tools` must show only currently exposed model tools.
- `/max-tokens` may adjust output token budget for following turns, runtime-only.
- Interactive final assistant responses should stream when supported by the backend.
- Streaming must not break tool-call parsing or duplicate the final response.
- Ctrl+C during streaming must interrupt the current turn, rollback partial messages, and return to the prompt without exiting.
- Tool phases should emit compact dim events, including tool name and result size.
- Tool event format is `<tool_name> <json_args>` followed by ` └ <tool_name> <result_chars> chars`.
- Show a dim elapsed-time indicator before the first streamed token.
- Keep one blank line between prompt and assistant response.
- Keep one blank line between assistant response and the dim metrics footer.
- Prefer resolved model names in status/footer; do not show SHA ids when reliable local metadata can map them.

## Tests

Minimum checks before considering a change safe:

```bash
python3 -m unittest discover -s tests -q
```

When `llama-server` is running, also run at least one real smoke test for the changed area.

Keep manual regression prompts in `PROMPTS.md`. The file should stay short and focused on currently supported behavior.

## Benchmark discipline

- Use `scripts/bench-kv-cache.py` before changing cache-related server flags.
- Use `scripts/bench-tool-cache.py` before changing tool-loop payloads, schemas, or cache behavior.
- Use `scripts/bench-continuation-cache.py` to compare no-tool continuation against tool-loop continuation.
- Use `scripts/bench-raw-cache.py --mode all` to distinguish Orbit runtime effects from raw `llama-server` behavior.
- Run `scripts/bench-memory-refresh.sh` on real sessions and record prompt tokens, cached tokens, and prefill speed before/after refresh.
- Keep explicit `llama-server` slot/cache management out of the core runtime unless benchmarks justify it.

## Todo

- Keep tool surface limited while native server tools and Orbit-only fallbacks are validated.
