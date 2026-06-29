# Native Route Prefix Prewarm Design

## Problem

The first tools-on conversational turn still pays the cold route prefill cost.
The current route KV prefix-anchor improves compatible repeat route calls, but
the first eligible route must still build and capture the checkpoint.

The first-turn analysis showed that a short tools-on conversational request is:

- one `route` model call over the full route prompt
- one `chat_final` model call for the visible answer
- no route retry, repair loop, or tool-loop noise

Representative metadata from the analyzed setup:

| Mode | Phase | Prompt tokens | Cached | Evaluated | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| auto cold first | `route` | 711 | 0 | 711 | capture miss |
| auto cold first | `chat_final` | 39 | 4 | 35 | conversational final |
| auto repeat | `route` | 766 | 693 | 73 | restore hit |
| anchor off cold first | `route` | 711 | 0 | 711 | baseline route prefill |
| tools off | `chat_final` | 39 | 4 | 35 | no tools route |

The route prompt dominates first-turn tools-on latency. The stable route prefix
was 693 tokens and the captured checkpoint was 238,454,176 bytes in the tested
Gemma 4 12B setup.

## Objective

Native route prefix prewarm should move the stable route prefix capture before
the first real user request, without changing observable runtime behavior.

The accepted shape is:

- prefill only the stable route tools-on prefix
- capture a real native KV checkpoint
- produce no model output
- generate no content tokens
- use no fake user request
- use no prompt workaround
- leave the real route prompt semantically unchanged
- leave model-guided tool/chat routing unchanged
- leave file, web, fetch, and listing evidence policy unchanged

If prewarm fails or is unavailable, the next real request must use the existing
baseline path.

## Requirements

Prewarm must be constrained to the same safe surface as route prefix-anchor:

- native backend only
- tools-on only
- route stable prefix only
- `ORBIT_KV_PREFIX_ANCHOR=off` disables prewarm
- non-native and external `llama-server` backends do not prewarm
- no prewarm for `chat_final`, `final_from_tool`, `tool_call`, no-tools, or
  multimodal paths
- no deterministic routing or tool choice
- no prompt text or route contract change
- no user-visible error if prewarm fails
- at most one active route checkpoint per native client/session
- diagnostics are metadata-only

## Invalidation

A prewarmed checkpoint is valid only for the same compatibility identity used by
the runtime route anchor. It must be invalidated or bypassed when any of these
change:

- model path or model id
- tokenizer identity
- chat template identity
- route contract
- tool schema
- capability summary
- runtime policy inputs relevant to the route prefix
- backend/native library version
- tools mode
- backend session or native context
- `ORBIT_KV_PREFIX_ANCHOR=off`

The restore path must continue to validate prefix hash and token count before
using the checkpoint.

## Current Architecture

The current runtime path is:

1. The Python runtime marks route calls with phase metadata.
2. The HTTP backend sends `route_prefix_anchor=true` only for native
   route/tools-on calls.
3. The native server accepts normal chat requests on `/chat` and `/chat/stream`.
4. `NativeLlamaClient.complete_chat_text()` renders the normal route prompt.
5. `render_gemma4_route_prompt_segments()` derives the stable prefix and dynamic
   suffix while preserving byte-identical prompt text.
6. `_prepare_memory_with_route_anchor()` either restores an existing checkpoint
   or decodes the stable prefix and captures a checkpoint.
7. The same call then continues with the dynamic suffix and normal generation.

The important limitation is that step 6 is currently only reached from a real
chat completion path that also proceeds to generation. There is no public
native-server endpoint or runtime lifecycle hook that performs "prefill stable
route prefix and stop".

## Lifecycle Candidates

### Prewarm When Tools Are Enabled

Trigger a native prewarm when tools mode transitions to on.

Benefits:

- aligns cost with the user's explicit decision to expose tools
- avoids startup penalty when tools are never used
- avoids fake user turns

Risks:

- slash-command handling must not block the terminal unexpectedly without clear
  UX feedback
- failed or slow prewarm must not leave the session in an in-flight state
- the backend lock must prevent concurrent completion and prewarm races

This is the most plausible lifecycle for a first implementation.

### Explicit Manual Warmup Command

Add an explicit internal command that asks the native backend to prewarm route
tools-on prefix state.

Benefits:

- easiest UX and failure semantics to reason about
- useful for benchmarking and debugging
- no startup or tools-on surprise cost

Risks:

- less automatic benefit
- adds a command surface that must remain clearly non-semantic

This is a good stepping stone before automatic tools-on prewarm.

### Lazy Background Prewarm After Server Start

Start a background prewarm after model load.

Benefits:

- first user request may be fast if warmup completes in time

Risks:

- can increase startup CPU and memory pressure
- may compete with the first real request
- requires careful cancellation, locking, and observability
- tools may not be enabled or stable yet from the runtime perspective

This should not be the first implementation unless startup UX is explicitly
benchmarked.

### No Automatic Prewarm

Keep current behavior and document the first capture miss.

Benefits:

- zero new lifecycle risk
- current repeat behavior is already effective

Risks:

- first tools-on conversational request remains slow on CPU

This remains acceptable if prewarm lifecycle cost is not worth the complexity.

## API And Hook Candidates

### Native-Server Prefill-Only Endpoint

Add an internal endpoint such as `/kv/route-prefix/prewarm` that:

- accepts only metadata needed to identify the route prefix contract
- receives or derives the same stable route prefix text used by real routes
- validates token boundary and prefix hash
- decodes only stable prefix tokens
- captures the checkpoint
- clears or restores active memory as needed
- returns metadata-only status

This endpoint must not accept arbitrary user content and must not produce
assistant text. It should run under the same server lock used by normal
completion.

This is the cleanest server/API shape, but it is a real protocol addition and
needs unit and native smoke coverage.

### Python Backend Method

Add a method on the Python backend client that calls a native prewarm endpoint
or native-server method.

This is useful as the runtime-facing entry point, but it should not implement
prewarm by sending a normal chat request with a special max token budget. That
would still route through completion semantics and risks becoming a fake
request workaround.

### Native Client Internal Method

Refactor the existing native client route-anchor internals into a method such as
`prewarm_route_prefix_anchor(...)` that:

- renders or receives route prompt segments
- validates prompt segment identity
- tokenizes the stable prefix
- clears target memory
- decodes only the stable prefix
- captures checkpoint data
- marks continuation as not ready
- leaves cached prompt state consistent with the captured prefix
- returns metadata-only prewarm status

This is likely the core implementation needed by either endpoint or manual
command. It is not currently exposed as a lifecycle-safe method.

### Why Fake Requests Are Rejected

Fake chat requests are not acceptable because they would:

- introduce synthetic user content or prompt shape
- enter normal generation or completion control flow
- risk changing sampler, continuation, history, or diagnostics semantics
- hide a runtime optimization behind a semantic request
- make correctness dependent on a prompt workaround instead of native state

Prewarm must be explicit native prefill/checkpoint work, not a disguised route
call.

## Failure Handling

Failure must always degrade to the current baseline path.

Required fallback cases:

- prewarm disabled by configuration
- non-native backend
- tools off
- missing route prefix boundary
- token boundary mismatch
- prefix hash mismatch
- token count mismatch
- checkpoint capture failure
- checkpoint restore failure
- native memory/decode error
- concurrent in-flight request
- cancellation

The user should not see an error from prewarm failure. Diagnostics may record a
metadata-only reason.

## Metrics

Suggested metadata-only diagnostics:

- `prewarm_enabled`
- `prewarm_attempted`
- `prewarm_succeeded`
- `prewarm_failed_reason`
- `prewarm_prefix_hash`
- `prewarm_prefix_tokens`
- `prewarm_checkpoint_size_bytes`
- `prewarm_age_ms`
- `first_route_restore_hit`
- `route_anchor_hit`
- `route_anchor_miss`
- `restore_used`
- `cached_tokens`
- `evaluated_tokens`

Diagnostics must not include raw prompt text, token ids, user content, tool
output, file content, or web content.

## Test Plan

Unit tests:

- `ORBIT_KV_PREFIX_ANCHOR=off` prevents prewarm
- invalid env values remain safe
- non-native backend does not send prewarm
- tools-off mode does not prewarm
- prewarm success creates a valid checkpoint state
- prewarm failure leaves the next request on baseline
- mismatch invalidates or bypasses the checkpoint
- diagnostics contain metadata only

Integration tests:

- native-server endpoint parses only the allowed prewarm request shape
- server lock prevents concurrent prewarm/completion races
- prewarm returns no assistant text
- continuation is not marked ready after prewarm
- normal `/chat/stream` still works after failed prewarm

Smoke tests:

- current baseline cold tools-on first conversational request
- prewarm followed by first real tools-on conversational request
- `ORBIT_KV_PREFIX_ANCHOR=off` no-prewarm baseline
- file-read content evidence
- directory listing
- web search
- fetch URL
- listing followed by file read
- local evidence followed by fresh web request

Negative tests:

- non-native and external `llama-server` path
- prompt/template/tool schema changes
- cancellation during prewarm
- native decode/capture exception

## Feasibility Decision

Do not implement runtime prewarm in this PR.

The native client has the low-level ingredients needed for prewarm:

- stable route prefix rendering
- token boundary validation
- prefix decode via `llama_decode`
- native checkpoint capture
- one-checkpoint lifecycle

However, the safe lifecycle hook is not present yet:

- the native server exposes chat/continue endpoints, not prefill-only prewarm
- route-anchor capture is currently wired inside a real completion path
- there is no runtime-facing command or lifecycle event that can request
  prewarm without entering normal chat semantics
- concurrency, cancellation, continuation readiness, and stale memory behavior
  need explicit tests before this becomes runtime code

The next implementation PR should first add a bounded native prefill-only hook
or endpoint, then wire an explicit manual or tools-on lifecycle trigger to that
hook. It should not use fake user requests, max-token hacks, prompt changes, or
deterministic simple-chat routing.

## Verdict

Verdict: design-only, no runtime patch.

Native route prefix prewarm is technically plausible, but the repository needs a
small explicit prefill-only native hook before it is safe to enable. Until that
hook exists, the current route prefix-anchor behavior remains the correct
baseline: first eligible route captures, compatible repeats restore.
