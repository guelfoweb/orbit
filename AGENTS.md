# AGENTS.md

## Scope

This repository contains the `orbit` CLI built around local `llama.cpp`-based backends, with native `orbit-server` as the primary path.

The tool name remains `orbit`. The repository directory may be temporary.

## Hard rules

- Keep the runtime small, readable, and easy to debug.
- Do not turn Orbit into a generic agent framework.
- Do not add deterministic fast paths for user tasks.
- The model must decide whether to use available tools.
- The runtime may enforce safety, size, context, and tool-contract boundaries.
- `workdir/` is a public regression fixture and must stay safe to publish.
- Prefer standard-library Python unless a dependency has clear value.
- Keep one-shot, REPL, backend, runtime, tools, and terminal UI separated.
- Do not reintroduce Ollama-specific logic in this codebase.
- Do not expose browser tools until explicitly designed.
- Broad shell access is allowed only through explicit tools-on mode and must stay disabled by default.
- After runtime/tool/session changes, run unit tests.

## Backend

- Primary backend: native `orbit-server`.
- Compatibility backend: local OpenAI-compatible chat completions, including `llama-server`.
- Default base URL: `http://127.0.0.1:18080`.
- Primary target model: `gemma4:12b-it`.
- Reasoning should remain disabled by default at backend startup unless explicitly testing visible thinking.
- Context profiles are provided by helper scripts, not hidden runtime magic.

## Tools

Model-facing tools:

- `exec_shell_full_command`

Rules:

- Tools are exposed only when enabled by tool mode.
- Command selection must be model-driven before tool exposure.
- No deterministic task fast paths.
- `exec_shell_full_command` is unrestricted local shell access.
- The runtime enforces timeout/output-size limits around shell execution.
- For analysis prompts, metadata-only commands such as `ls`, `file`, or `stat` must trigger a model retry asking for direct content/source/string evidence.
- Failed shell commands may enter a model-driven repair loop using bounded exit code, stdout, and stderr evidence.
- Shell repair must stay generic: no command/utility whitelist; skip only clearly environmental failures such as permission, filesystem, memory, DNS, or timeout errors.
- Mutating shell commands that exit successfully with no output may trigger one model-driven verification command.
- Mutation verification must not validate domain-specific formats in runtime; the model must produce evidence of the changed value or state.
- Runtime guards must guide the model, not solve the task deterministically.
- Internal guard prompts may ask for content evidence, completion, minimal local patches, or semantic repair, but the model must choose the command.
- Tool-call output budgets may be dynamic by phase; do not raise global max tokens to hide internal control-flow bugs.
- Coding guards must stay language-agnostic unless benchmark evidence justifies a narrower rule.
- `cat` on large UTF-8 text/source files may be post-processed through the internal bounded reader.
- Commands that read or analyze local PDFs may be post-processed through text extraction.
- PDF text extraction must prefer `pdftotext`; if unavailable, fallback to filtered `strings`.
- PDF support is text-only: no OCR and no raw PDF reinjection.
- Generic web search should use `orbit-web-search "query"`; explicit URLs should use `curl`.
- HTML emitted by shell commands such as `curl` may be converted to readable text before reinjection.
- If the user asks for HTML/page source analysis, preserve source-like HTML instead of converting it to readable text.
- Web content must not be silently saved into the workdir.
- Do not present a first bounded shell result as a complete summary of a long document.
- Unknown tools must fail clearly.
- Tools remain off by default. `/tools on` exposes only `exec_shell_full_command`.

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
- `/status` should include shell repair/mutation verification counters when present.
- `/tools` must show only currently exposed model tools.
- `/max-tokens` may adjust output token budget for following turns, runtime-only.
- Interactive final assistant responses should stream when supported by the backend.
- Streaming must not break tool-call parsing or duplicate the final response.
- Ctrl+C during streaming must interrupt the current turn, rollback partial messages, and return to the prompt without exiting.
- Tool phases should emit compact dim events, including display tool name and result size.
- Tool event format is `exec <json_args>` followed by ` └ <result_chars> chars -> model`.
- Show a dim elapsed-time indicator before the first streamed token.
- Keep one blank line between prompt and assistant response.
- Keep one blank line between assistant response and the dim metrics footer.
- Prefer resolved model names in status/footer; do not show SHA ids when reliable local metadata can map them.

## Tests

Minimum checks before considering a change safe:

```bash
python3 -m unittest discover -s tests -q
```

When a local backend is running, also run at least one real smoke test for the changed area.

Keep manual regression prompts in `docs/PROMPTS.md`. The file should stay short and focused on currently supported behavior.

For release confidence, use `orbit release-confidence`. Its fixtures must stay isolated in `/tmp`, and its checkers must validate final behavior rather than specific shell commands.

## Benchmark discipline

- The software performance tuning line is closed.
- Keep the current benchmark set as the regression suite.
- Do not change routing, tool selection, final-answer policy, prompts, tool payloads, or cache behavior for performance unless a benchmark shows a strong, comparable benefit.
- In the absence of strong measurement evidence, do not touch observable behavior.
- Use `orbit bench-core` as the public regression benchmark.
- `orbit bench-core` uses the repository `workdir/` fixture by default.
- For deeper profiling, prefer temporary local scripts or manual measurements rather than adding permanent helper scripts.
- Keep explicit backend slot/cache management out of the core runtime unless benchmarks justify it.

## Todo

- Keep the tool surface limited and explicit.
