# Prompts Release Hardening Review

Commit reviewed: `7e810a5` (`origin/main`, PR #57 merged).

This is a strategic prompt review before a possible RC2. It does not publish
RC2 and does not change prompt text.

## Files Analyzed

- `docs/PROMPTS.md`
- `src/orbit/runtime/messages.py`
- `src/orbit/runtime/tools.py`
- `src/orbit/runtime/capabilities.py`
- `src/orbit/runtime/command_request.py`
- `src/orbit/native_llama/chat_template.py`
- related tests: `tests/test_messages.py`, `tests/test_config.py`,
  `tests/test_command_request.py`, `tests/test_native_chat_template.py`

## Findings

### Route Prompt

The route prompt remains model-guided. It asks the model to decide whether a
request needs local tools and explicitly avoids runtime-authored task answers.
No deterministic user-task mapping was found.

Current route constraints are consistent with the post-PR #51 behavior:

- direct route answers are allowed only for no-tool, no-evidence requests that
  fit in one short sentence
- normal/long no-tool answers should return `{"route":"CHAT"}`
- file read/explain/summarize/analyze requests require content-reading command
  decisions
- directory listing is separated from file content evidence
- web/search/latest/current/online and URL fetch/read/open/explain/summarize
  requests are tool tasks
- long prose in the route pass is explicitly forbidden

No route prompt rewrite is recommended before RC2.

### Tool Specs And Capability Summary

`src/orbit/runtime/tools.py` exposes tools in a stable tuple order:

```text
exec_shell_full_command, fetch_url, list_directory, system_info
```

`src/orbit/runtime/capabilities.py` uses a stable tuple of local capability
names and formats available/unavailable summaries deterministically from that
tuple. The summary can vary by machine, but its order is stable for a given
environment.

No timestamp, counter, session id, or dynamic path was found in the stable
route prefix. This preserves the route-prefix boundary needed by the native
route KV prefix-anchor auto path.

### Chat Final And Tool Final

`CHAT_SYSTEM_PROMPT` is intentionally short and separate from the route prompt.
`FINAL_FROM_TOOL_SYSTEM_PROMPT` keeps final answers model-generated from tool
results and does not allow tool calls in finalization.

No prompt-level cause for unnecessary repair was found in this review.

### Manual Regression Prompts

`docs/PROMPTS.md` covers:

- chat without tools
- visible thinking
- local tools
- file read/content evidence
- online optional smoke
- tools plus thinking

It already distinguishes optional online checks from required local release
checks. No broad rewrite is recommended before RC2.

## Risks

- The route prompt is necessarily dense because it encodes evidence boundaries,
  tool distinctions, and no-long-prose behavior. Compressing it before RC2 would
  risk regressing file/web/listing correctness.
- The tools-on prompt remains large. Route prefix-anchor auto mode helps
  repeated eligible route passes, but the first capture miss remains expensive.
- Generic web smoke can vary with provider/network behavior and should remain
  optional or interpreted conservatively.

## Patch Recommendations

Applied:

- no prompt patch
- import-only cleanup in native KV code
- documentation-only KV index and release reviews

Recommended but not applied:

- add a compact docs index for KV reports, now done as `KV_INDEX.md`

Rejected for this phase:

- route contract rewrite
- tool spec compression
- prompt-shape changes for cache behavior
- removal of historical no-go/reject documents

## Impact

Stability:

- No prompt behavior changed.
- Existing evidence and routing guarantees are preserved.

Performance:

- No prompt token reduction was attempted.
- Route prefix-anchor remains bounded to native route/tools-on and can be
  disabled with `ORBIT_KV_PREFIX_ANCHOR=off`.

KV:

- Stable route prefix assumptions remain intact.
- No timestamp, counter, or dynamic path was introduced into the stable prefix.

## Recommendation

Proceed to release validation without prompt changes. If later performance work
targets prompt size, require a dedicated benchmark branch with file-read,
listing, web/fetch, and route outcome regression checks.
