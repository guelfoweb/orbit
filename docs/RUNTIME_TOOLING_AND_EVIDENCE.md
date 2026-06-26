# Runtime Tooling And Evidence

Orbit keeps runtime behavior small and explicit: the model decides when to use tools and how to write the final answer; the runtime enforces contracts, evidence boundaries, size limits, and diagnostics.

## Architecture Principles

- Runtime owns behavior: tool exposure, safety boundaries, evidence validation, retries, terminal events, and session state live in Orbit runtime code.
- Backend owns inference: tokenization, decoding, KV/session state, streaming transport, cancellation, and backend metrics stay in the backend/native layer.
- Tool use is model-guided: Orbit may expose capabilities, but it does not auto-route a user task to a command or tool.
- Deterministic guardrails are allowed only for safety, validation, evidence, bounded retry, diagnostics, and UX.
- The runtime must not construct, rewrite, or semantically repair final answers. The model writes the final answer.
- Dedicated tools can normalize evidence, but they are still selected by the model when tools are enabled.

## Evidence Policy

For requests that ask Orbit to read, explain, summarize, or analyze a source, the final answer must be grounded in real evidence from that source.

File requests:

- Real file content is required for `read`, `explain`, `summarize`, or `analyze` requests.
- Metadata-only evidence such as `ls`, `stat`, `file`, directory listings, and file existence is not enough for file-content tasks.
- Metadata can be sufficient when the user asks for metadata, for example whether a file exists or which files are present.
- If only metadata is available for a content task, the runtime can ask for one bounded model-guided recovery pass. The model must choose how to obtain content evidence.

URL requests:

- Explicit URL fetch/read/open/explain/summarize/analyze/extract tasks require real URL content or a real observed fetch failure.
- `fetch_url.status=ok` with extracted text is positive evidence.
- HTTP errors, DNS failures, TLS failures, timeouts, empty bodies, and unsupported content are valid negative evidence only when actually observed by a tool.
- Curl or wget progress meters, connection logs, transfer statistics, and empty output are not URL content evidence.
- The model must not speculate from the URL, title, date, or prior knowledge when the user asked to fetch the page.

Failure evidence:

- A negative final answer is acceptable only after a real failure has been observed.
- The answer should describe the observed failure, not invent a reason.
- Provider blocks, challenge pages, 403/404/500 statuses, DNS errors, TLS errors, and timeouts are evidence when surfaced by a tool result.

## Dedicated Tools

Tools are available only when `/tools on` is active. The model chooses among them; Orbit does not replace model choices with deterministic task routes.

### `exec_shell_full_command`

`exec_shell_full_command` remains the unrestricted local shell tool. It is useful for arbitrary local commands, but shell output can be noisy and expensive to reinject into the model context.

The runtime bounds shell execution by timeout and output size. It can post-process transport noise such as readable HTML conversion or bounded file/PDF extraction, but it does not write the final answer.

### `fetch_url`

`fetch_url` is the explicit URL retrieval capability for direct URL content tasks.

It returns structured evidence:

- `status`: `ok`, `http_error`, `network_error`, `timeout`, `unsupported_content`, or `empty_body`
- original URL and final URL after redirects
- HTTP status
- content type and encoding
- title when available
- cleaned text
- truncation flag
- error message when applicable

`fetch_url` follows redirects, uses bounded time and body size, rejects non-textual content, converts readable HTML to text, and does not execute JavaScript or bypass anti-bot challenges.

Shell fetch commands are still allowed. If the model uses shell and obtains real body text or a real failure, that can satisfy the URL evidence guard. Progress meters and technical logs do not.

### `list_directory`

`list_directory` provides compact, stable, bounded directory listings.

It is meant for directory structure and file-list questions where shell commands like `find .`, `ls -R`, or `tree` would produce large noisy outputs. It does not read file contents and therefore does not satisfy file-content evidence requirements.

The output includes path, recursion settings, entry counts, truncation status, and typed entries such as files, directories, symlinks, and other filesystem objects.

### `system_info`

`system_info` provides compact local machine specifications.

It reports OS, CPU, RAM, disk, and Python runtime information using standard-library logic plus conservative Linux `/proc` fallbacks. It avoids sensitive data such as username, hostname, IP addresses, MAC addresses, environment variables, process lists, serial numbers, and disk UUIDs.

It exists to avoid noisy shell reinjection from commands such as `lscpu`, `free`, `df`, `uname`, and `cat /proc/*` when the user asks for machine specs.

## Web Search

Generic web search and explicit URL fetch are separate workflows.

- Generic search uses `orbit-web-search "query"` through the shell path.
- Direct URL content tasks should use `fetch_url` when available.
- Search providers are best-effort and may return challenge pages, blocked pages, or no parseable results.
- Online search prompts are optional manual smoke tests, not blockers for the local CPU-first no-MTP release path.
- No-results search output must not cause Orbit or the model to invent facts.

## Capability Discovery

Orbit discovers local document-related capabilities at runtime startup with `shutil.which`.

Initial discovered commands:

- `pdftotext`
- `pandoc`
- `python3`
- `python`
- `file`
- `unzip`
- `libreoffice`
- `soffice`
- `antiword`

Discovery is environmental diagnostics only:

- It runs once at startup.
- It is stored in memory.
- It does not call the model.
- It does not choose a command.
- It does not install dependencies.
- It does not make `pdftotext`, `pandoc`, LibreOffice, or any extractor mandatory.

When tools are enabled, Orbit includes a compact capability summary in the tools-on prompt, for example:

```text
Local tools available: python3, file, unzip, pdftotext.
Unavailable: pandoc, libreoffice, antiword.
Use only tools that are available or verify availability before use.
```

The model can use this in the same inference pass to choose an available method.

Interactive commands:

- `/tools status` prints the current detected capabilities.
- `/tools refresh` reruns discovery and updates the cached capability summary.

Both commands are local runtime actions and do not invoke the model.

## Terminal UX

Current terminal rendering supports:

- live Markdown styling for final answers
- plain streaming via `--no-render-markdown` or `ORBIT_RENDER_MARKDOWN=plain|off|0|false`
- explicit live mode via `--render-markdown-live` or `ORBIT_RENDER_MARKDOWN=live`

The current default on `origin/main` is live Markdown rendering. Plain streaming remains available through the explicit opt-out above.

Markdown rendering is intentionally minimal and stream-oriented. It does not use a full-screen TUI, does not rewrite scrollback, and does not wait for complete Markdown blocks before showing text.

Rendering applies only to visible final-answer text. It does not apply to:

- thinking output
- tool events
- raw tool stdout/stderr
- progress lines
- debug logs
- metrics footers

The renderer may style headings, bullets, fenced code, and simple inline markers, but it must not change generated content.

## Performance Rationale

The main performance goal is reducing unnecessary prompt reinjection and prefill cost while preserving model-guided behavior.

Relevant techniques:

- dedicated `fetch_url` avoids reinjecting curl progress meters and huge raw HTML where possible
- `list_directory` avoids large recursive shell listings
- `system_info` avoids verbose platform command output
- capability discovery helps the model avoid absent document tools in the first tool decision
- evidence guards prevent premature ungrounded final answers that would otherwise require user correction
- bounded tool output keeps large content visible enough for grounding without flooding context

These techniques are especially important on CPU-only Gemma 4 12B runs, where long tool results and unstable prompt prefixes increase prefill time.

## Non-Goals

- No automatic task routing based on prompt keywords.
- No runtime-authored final answers.
- No hidden browser automation.
- No JavaScript execution for web pages.
- No anti-bot bypass.
- No backend/native/MTP behavior changes.
- No claim that generic web search is reliable without provider support.
