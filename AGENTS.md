# AGENTS.md

## Purpose

`orbit` is a minimal Python CLI for Ollama with an interactive REPL and local tool calling.

Current version: `0.1.0`

Internal notes must stay out of the public repository unless they are intentionally documented.

Permanent constraints:

- Keep the code easy to read.
- Keep the agent loop stable.
- Keep side effects predictable.
- Keep testing and debugging straightforward.

## Backend And Runtime

- Supported backend: local Ollama HTTP API.
- Main endpoint: `http://127.0.0.1:11434`.
- API endpoint: `/api/chat`.
- Preferred Python client: official `ollama` package.
- The main path must use Ollama native tool calling.
- A limited fallback is allowed for tool calls returned as JSON/text.
- `orbit` is built and tuned primarily around `gemma4:e2b`.
- Other Gemma4 profiles are best-effort and are not guaranteed to behave with the same reliability.
- If `--model` is omitted, use the first model already running from `/api/ps`.
- If no model is running and `--model` is omitted, fail with a clear error.
- Timeout must be configurable from the CLI and timeout errors must suggest increasing it.

## Operating Modes

- Interactive REPL is the default mode.
- One-shot mode is enabled by passing a positional prompt.
- Optional skills can be loaded with `--skill` or REPL slash commands.
- Sessions live under `~/.orbit/sessions`.
- CLI, runtime bootstrap, and terminal rendering must remain separated.

Minimum agent loop:

1. Append the user message to history.
2. Call Ollama with `messages` and `tools`.
3. Execute each tool call locally.
4. Reinject each result as `role: tool`.
5. Stop on a final assistant reply without tool calls.
6. Retry once on an empty reply.
7. Fail with a clear error after two empty replies in the same turn.

## Sessions And Skills

Sessions:

- If `--session` is omitted in interactive mode, look for sessions matching the current `workdir`.
- If multiple sessions exist, show session name, first user prompt line, and a new-session option.
- Session names must derive stably from `workdir`, with an incremental suffix when needed.
- Persist at least: `messages`, active skill, and `workdir`.
- Changing skill on an existing session must not load incompatible old messages.
- `/reset` must persist the empty session immediately.
- `/compact` must persist the compacted session immediately.
- There must be a way to clear all sessions for the current `workdir`.

Skills:

- Supported format: `SKILL.md`.
- Supported lookup: skill name, directory containing `SKILL.md`, or direct file path.
- Default roots: `~/.orbit/skills` and `./skills`.
- The active skill is appended to the system prompt.
- If no skill is selected, automatically load `orbit-default`.
- Changing skill must reset the current session.

## Supported Tools

Local tools:

- `read_file`
- `list_files`
- `stat_path`
- `replace_in_file`
- `write_file`
- `append_file`
- `bash`
- `search_web`
- `fetch_url`

New tools are allowed only when they are:

- small and bounded
- explicit in their parameters
- limited in risk surface
- covered by unit tests

## Guardrails

- All filesystem access must remain confined to `--workdir`.
- Paths outside the logical root must be rejected.
- `read_file` must enforce line and character limits.
- `replace_in_file` must operate only on existing UTF-8 text files.
- Ambiguous replacements must be rejected unless explicitly requested.
- `write_file` and `append_file` must enforce size limits.
- The first overwrite attempt on an unread existing file must be rejected with a `read_file` hint.
- `bash` must use `shell=False`.
- `bash` must not allow redirects, chaining, subshells, or equivalent operators.
- Minimal pipelines are allowed only to benign allowlisted filters.
- `bash` must block destructive commands or commands incoherent with the task.
- `rm` is allowed only for targets confined to `workdir`.
- `search_web` must return structured and bounded results, never raw HTML.
- `fetch_url` must accept only explicit `http`/`https` URLs and return structured bounded output.

## Architecture

Required structure:

- `orbit/core/`: agent loop, Ollama client, compaction, runtime, sessions, policy, routing, context budget, loop guard, parser, guardrails
- `orbit/tooling/`: registry and local tools by domain
- `orbit/terminal/`: CLI, config, history, and text rendering

Key responsibilities:

- `terminal/cli.py`: thin entrypoint
- `terminal/config.py`: config parsing and validation
- `core/runtime.py`: bootstrap, skill, session, and persistence
- `terminal/history.py`: REPL history
- `terminal/ui.py`: help, tools, status, and rendering
- `core/events.py`: typed events between loop, runtime, and UI
- `core/agent.py`: agent loop and turn metrics
- `core/tool_router.py`: tool subset exposed to the model
- `core/intent_gate.py`: model-assisted YES/NO confirmation for risky ambiguous tool routes
- `core/context_budget.py`: context pressure thresholds and profiles
- `core/loop_guard.py`: repeated tool-call history and matching
- `core/tool_call_parser.py`: fallback parsing for near-valid JSON/text tool calls
- `core/message_ops.py`: pure helpers over messages
- `core/tool_guardrails.py`: runtime constraints without bloating the main loop
- `core/tool_session_state.py` and `core/tool_execution_policy.py`: read-before-write, dedup, trust decay, and rehydration
- `core/compact.py`: session compaction
- `core/turn_policy.py`: explicit turn state classification
- `tooling/common.py`, `tooling/filesystem.py`, `tooling/shell.py`, `tooling/web.py`: domain tool implementations

Rules:

- If the selected model does not advertise `tools`, warn and degrade to chat-only.
- Do not recentralize logic in `terminal/cli.py`.
- `orbit` may expose `--think on|off|auto` and `--show-thinking`.
- `--debug-timing` may expose bounded timing diagnostics but must remain off by default.
- Do not expose all tools for vague prompts on the main Gemma runtime.
- Clear intents should keep fast direct routing; ambiguous, high-risk, or unsupported actions should pass through the YES/NO intent gate before exposing tools.
- The intent gate must answer only whether Orbit should proceed with available local tools; it must not perform the task itself.

## UX

- Use classic terminal UX, not a full-screen TUI.
- Persistent history should support arrow up/down when `readline` is available.
- Local artifacts live under `~/.orbit`.
- Errors must be short and readable.
- One `Ctrl+C` interrupts the current turn/input; two close interrupts exit.

Minimum slash commands:

- `/compact`
- `/debug`
- `/exit`
- `/help`
- `/reset`
- `/sessions clear`
- `/skill clear | list | show | use <ref>`
- `/status`
- `/think on|off|auto`
- `/thinking on|off`
- `/tools`

UX rules:

- Unknown slash commands must never be sent to the model.
- `/skill clear` restores the default skill.
- `/debug` shows the last-turn debug summary; `/debug last` may remain as a hidden compatibility alias but should not be advertised.
- `/status` shows current runtime state: model, capabilities, context usage, session, workdir, skill, tools, and thinking state.
- `/compact` must use hybrid compaction with both message-count and context-budget thresholds.
- The per-turn status line must show at least: model, context window with percentage, token flow with prefill/generation speeds, message count, response source, and media preprocessing when applicable.
- Very long user input may be visually collapsed as `[text N chars]` without changing the real content sent to the model.

## Compaction And Turn Policy

Compaction:

- Must be reliable even if the model fails.
- Must start from a deterministic and fast local structure.
- The model may only refine the local summary.
- If the model fails, returns empty content, or tries a tool call, use the local fallback immediately.
- Must replace old messages with one structured operational summary.
- Must keep recent messages verbatim.
- Must prioritize durable requests, findings, tool activity, previous compacted memory, touched paths, and next step.
- Before appending a large tool result, estimate projected pressure; if it would create hard pressure, compact first, then append the tool result.

Turn policy:

- Main cases must be classified explicitly, not with scattered branching.
- Loop, runtime, and renderer must communicate through typed events, not implicit event strings.
- Distinguish at least: final answer, tool phase, empty-reply retry, double-empty abort, repeated-tool abort, and max-loop stop.
- Context pressure must be preemptive with at least `soft` and `hard` levels.
- Initial thresholds must be conservative and adjustable by model family/size when needed.

## Dependencies And Testing

- Prefer the standard library when sufficient, except for the official Ollama client.
- Avoid extra dependencies without clear value.
- `Pillow` is allowed for bounded local image normalization in the vision path.
- `ffmpeg`/`ffprobe` are required for bounded local audio normalization and chunking; audio prompts must fail clearly if they are missing.
- Packaging must stay minimal.

Minimum test coverage:

- base agent loop
- JSON fallback tool-call parsing
- empty-reply retry
- double-empty abort
- session compaction
- skill resolution
- path confinement
- file read limits
- shell operator blocking
- simple `bash` execution
- context/model metadata parsing
- vision image normalization
- audio path detection and chunking
- tool-result compaction pressure

## Evolution Rules

- Do not turn the project into a general framework.
- Do not introduce browser automation or generic scraping outside bounded tools.
- If a model handles tool calls poorly, fix prompt, limits, and guardrails before complicating architecture.
- If a feature increases loop complexity, prefer a small bounded tool over implicit core logic.
- Never route or remediate based on one exact observed user prompt.
- Use reusable signals: tokens, request structure, intent classes, and behavior classes.
- Keep routing fixes semantic: distinguish explanation/opinion/learning prompts from operational tool requests.
- Discursive mentions of tools, commands, file systems, encoding, web search, or malware analysis must not expose tools unless the request is operational.
- Regression prompts must be checked in both Italian and English when the fix affects language behavior.
- Remediations must work coherently in both languages.
- After every relevant fix, run unit tests and at least one real test with the target model.
- Avoid fixes overly localized to language, dataset, path, filename, or a single development case.
- When a concrete defect appears, fix the class of problem, not just the observed example.

## Configuration

Recommended model profile:

```text
Modelfile.gemma4-e2b-fast-t6-c8k
```

```text
FROM gemma4:e2b

# Tuned for a 6-core / 12-thread CPU machine.
PARAMETER temperature 0
PARAMETER num_ctx 8192
PARAMETER num_thread 6
PARAMETER num_batch 96
```

Create it with:

```bash
ollama create gemma4:e2b-fast-t6-c8k -f Modelfile.gemma4-e2b-fast-t6-c8k
```

Run:

```bash
orbit --model gemma4:e2b-fast-t6-c8k
```

User config:

- Optional config path: `~/.orbit/config.json`.
- If the file is missing, Orbit must keep the same CLI defaults.
- CLI flags must override matching config values when a matching flag exists.
- Keep the config small and limited to runtime/UI defaults.
- Do not store prompt policy, project-specific behavior, session state, or skill content in the global config.
- `orbit-default` remains the automatic default skill and should not need to be duplicated in config.

Supported config keys:

```json
{
  "model": "gemma4:e2b-fast-t6-c8k",
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

Ollama server performance notes:

- If Ollama runs as a systemd service, environment variables exported before `orbit` do not configure the server.
- Put `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_KEEP_ALIVE=-1`, and optionally `OLLAMA_MAX_LOADED_MODELS=1` in the `ollama.service` drop-in, then run `daemon-reload` and restart Ollama.
- If launching `ollama serve` manually, export those variables before `ollama serve`, not before `orbit`.
- On CPU-only systems, power profile/governor can affect throughput more than `num_ctx`, `num_thread`, or `num_batch`.

## Regression Prompts

Keep the curated prompt suite in [PROMPTS.md](PROMPTS.md).

Run the benchmark instructions in [BENCHMARK.md](BENCHMARK.md) after runtime, routing, guardrail, compaction, or tool changes.
