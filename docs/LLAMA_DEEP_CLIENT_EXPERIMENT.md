# Llama Deep Client Experiment

This branch evaluates whether Orbit should move beyond the
OpenAI-compatible `llama-server` HTTP API and become a deeper client of
`llama.cpp`.

This is an experiment, not a release path.

## Goal

Determine whether deeper control of `llama.cpp` provides enough value to justify
the added complexity.

Target capabilities:

- real prefill progress;
- real backend cancellation;
- explicit KV/session ownership;
- reliable common-prefix/cache visibility;
- phase-level metrics that do not depend on post-hoc HTTP timings.

## Non-Goals

- Do not redesign Orbit as a generic agent framework.
- Do not change routing, prompts, tool-use, or final answer behavior.
- Do not replace model decisions with deterministic task fast paths.
- Do not merge native or FFI code unless the measured value is clear.
- Do not make this path the default during the experiment.

## Baseline Problem

Orbit currently uses:

```text
/v1/chat/completions
```

This keeps Orbit small and compatible, but it means Orbit only sees:

- request sent;
- first streamed delta;
- final usage/timings.

It does not truly own:

- prefill loop progress;
- cancellation;
- KV/session synchronization;
- common-prefix reuse.

The `/slots` endpoint is useful for observability, but in real use it often
updates too late or too sparsely to provide reliable prefill progress.

## Candidate Designs

### 1. Native library/FFI client

Orbit loads `llama.cpp` as a library through a Python binding or a small native
extension.

Benefits:

- full eval loop control;
- exact prefill progress;
- exact cancellation;
- direct KV/session management.

Costs:

- high implementation complexity;
- packaging complexity;
- platform-specific failures;
- tighter coupling to `llama.cpp` internals;
- higher maintenance burden.

### 2. Controlled local subprocess

Orbit controls a local `llama.cpp` binary/process with a protocol richer than
OpenAI chat completions.

Benefits:

- lower packaging risk than FFI;
- process isolation;
- easier rollback.

Costs:

- requires a stable protocol;
- may still lack KV/session ownership;
- cancellation depends on process/protocol semantics.

### 3. Minimal `llama-server` patch

Add server-sent prefill progress/cancel events to `llama-server`.

Benefits:

- keeps Orbit mostly unchanged;
- lower scope than full deep client;
- preserves current deployment model.

Costs:

- requires maintaining a fork/patch;
- still does not give full session ownership.

## Experiment Plan

### Phase 1: Capability Audit

Answer with code and measurements:

- Can a non-HTTP client observe per-token prefill progress?
- Can it cancel generation immediately and safely?
- Can it report common-prefix/KV reuse?
- Can it expose phase timings before final completion?
- Can this be done without destabilizing packaging?

### Phase 2: Prototype Adapter

Create an experimental backend adapter behind an explicit flag.

Requirements:

- no default behavior change;
- same `ChatBackend` contract where possible;
- no routing/tool/prompt changes;
- clear fallback to `llama-server`.

### Phase 3: Benchmark

Compare against the current `llama-server` backend:

- chat short;
- long prompt prefill;
- tool result final inference;
- Ctrl+C cancellation during prefill;
- Ctrl+C cancellation during generation;
- repeated prompt cache/common-prefix case.

Report:

- wall time;
- prefill progress fidelity;
- cancellation latency;
- prompt/cache metrics;
- implementation complexity.

## Acceptance Criteria

Do not merge unless the prototype demonstrates at least two substantial wins:

- reliable live prefill progress;
- reliable backend cancellation;
- better KV/session reuse control;
- measurable latency or robustness improvement.

If the only improvement is a nicer progress display, close the branch without
merge.

## First Decision Point

Before writing runtime code, inspect the local `llama.cpp` build and available
interfaces:

```bash
ls /home/guelfoweb/LAB/llama.cpp-gemma4-mtp-qualcomm/build/bin
find /home/guelfoweb/LAB/llama.cpp-gemma4-mtp-qualcomm/build -name '*llama*' -type f
```

Then decide whether the first prototype should use:

- native library/FFI;
- a controlled subprocess;
- or a minimal server patch.

## Local Capability Audit

Probe:

```bash
python3 scripts/probe-llama-deep-client.py
```

Observed local build:

```text
llama-server: yes
llama-cli: yes
libllama.so: yes
libllama-common.so: yes
llama.h: yes
llama-cpp.h: yes
```

Server capabilities:

```text
--slots: yes
--metrics: yes
--slot-save-path: yes
--spec-type: yes
--cache-ram: yes
--ctx-checkpoints: yes
```

CLI capabilities:

```text
--perf: yes
--show-timings: yes
--conversation: yes
--ctx-checkpoints: yes
--cache-ram: yes
--spec-type: yes
```

Initial recommendation:

- `native_ffi` is possible and gives the right control surface, but has high
  implementation and packaging complexity.
- `subprocess_cli` is possible but likely insufficient for exact KV/session
  ownership.
- `server_patch` is practical for progress/cancel events, but still keeps Orbit
  outside the eval loop.

## Pure Python Native Probe

This branch now tests the native path through Python `ctypes`, without C++
helper binaries and without touching the production `llama-server` backend.

Structure:

```text
src/orbit/native_llama/
  bindings.py   ctypes ABI for the small llama.cpp API surface used here
  paths.py      local llama.cpp/model discovery
  client.py     model/context/sampler lifecycle and decode loop
  events.py     progress/timing data objects
```

Probe:

```bash
PYTHONPATH=src python3 scripts/probe-llama-native-client.py \
  --prompt "hi, who are you?" \
  --max-tokens 12
```

Expected live progress:

```text
load: 100%
pf: 128/933 tk (13%)
gen: 4 tk
```

What this path can provide if it proves stable:

- real prefill progress from the actual `llama_decode()` loop;
- real abort through llama.cpp abort callbacks;
- explicit context ownership;
- eventual KV/session inspection and reuse experiments.

What is intentionally not implemented yet:

- MTP loading;
- production backend selection;
- packaging support.

## Measured Prototype Results

### Native Load

The Python native client can load the local Gemma 4 12B GGUF through
`libllama.so`.

The model load callback provides real loading progress, but it is noisy and
should be throttled in any user-facing UI.

### Real Prefill Progress

The native decode loop reports exact prompt-token progress because Orbit owns
the calls to `llama_decode()`:

```text
pf: 16/28 tk (57%)
pf: 28/28 tk (100%)
```

This is different from `/slots`: it does not depend on sparse backend polling
or guessed prompt sizes.

### Native Cancellation

Ctrl+C / SIGINT sets an abort callback consumed by llama.cpp.

Observed behavior:

```text
cancelled: True
```

No HTTP stream disconnect is involved. `llama_decode()` returns the native
aborted status and the client exits cleanly.

### Controlled KV Prefix Reuse

The prototype keeps the previously evaluated prompt tokens and trims llama.cpp
memory from the first divergent token before the next request.

Repeated prompt benchmark:

```text
turn 1:
prompt_tokens: 26
reused_prompt_tokens: 0
evaluated_prompt_tokens: 26
prefill_ms: 1865.8

turn 2:
prompt_tokens: 26
reused_prompt_tokens: 25
evaluated_prompt_tokens: 1
prefill_ms: 312.7
```

The last prompt token is intentionally re-evaluated to produce fresh logits for
generation. This still gives controlled reuse of the stable prefix.

## Architecture Direction

If this branch continues, the native backend should stay layered:

```text
native_llama/bindings.py  -> ctypes ABI only
native_llama/paths.py     -> local library/model discovery
native_llama/client.py    -> model/context/sampler lifecycle
native_llama/events.py    -> progress and timing events
```

Do not fold this into `backend/llama_server.py`.

The production integration should be an explicit backend adapter only after the
native path proves parity for:

- chat template formatting;
- tool-call prompting and parsing;
- MTP/speculative decoding;
- session memory and compaction behavior;
- release confidence prompts.

## Next Technical Steps

1. Implement chat-format parity.
   Current fallback uses the Gemma 4 control tokens directly because
   `llama_chat_apply_template()` does not support the full model Jinja template.

2. Measure long-prompt progress.
   Use a prompt large enough to show smooth prefill progress, not only two
   batches.

3. Add MTP support.
   Load the draft context/model only after the base native loop is stable.

4. Add structured benchmark comparison.
   Compare native vs `llama-server` for:
   - first-turn prefill;
   - repeated-prefix prefill;
   - Ctrl+C during prefill;
   - Ctrl+C during generation;
   - tool-result final inference.

5. Keep merge criteria strict.
   Do not merge unless native mode provides substantial wins beyond display
   polish.

## Orbit Server

This branch introduces an experimental `orbit-server` process implemented in
Python.

Goal:

- replace `llama-server` with an Orbit-owned local model process;
- keep the model loaded once;
- expose Orbit-native metrics and progress;
- preserve a compatibility bridge while the CLI is migrated.

Command:

```bash
PYTHONPATH=src python3 scripts/orbit-server.py --port 18082
```

Primary protocol:

```text
POST /chat
```

Compatibility protocol:

```text
GET  /health
GET  /v1/models
GET  /props
POST /v1/chat/completions
```

The OpenAI-like endpoint is a bridge only. The long-term direction is for Orbit
to use its own protocol and avoid depending on `llama-server` response shapes.

Smoke test:

```bash
PYTHONPATH=src python3 -m orbit.terminal.cli \
  --base-url http://127.0.0.1:18082 \
  "who developed you?"
```

Observed:

```text
I was developed by Google DeepMind.

model: gemma4:12b-it-native | tks: 38->8, cached 7 | pf 14.7/s | gen 3.2/s
```

Current `orbit-server` capabilities:

- loads Gemma 4 12B through Python `ctypes`;
- keeps model/context hot;
- supports streaming;
- exposes real native timings;
- exposes controlled prefix reuse as `cached_tokens`;
- allows current Orbit CLI to run through the compatibility endpoint.
- exposes an Orbit-native `/chat/stream` protocol with typed SSE events:
  - `progress.prefill`
  - `progress.generation`
  - `delta`
  - `metrics`
  - `done`
- supports explicit single-session identity through `session_id: "default"`;
- exposes `/cancel` for native backend cancellation;
- supports stop sequence trimming and early decode abort.

Current limitations:

- no native MTP yet;
- no production packaging;
- no dedicated Orbit protocol backend adapter yet;
- no release-confidence run against native mode yet.
- only one native session is supported in this experiment;
- OpenAI-compatible streaming is a bridge and does not expose typed native
  progress events.

Decision:

`orbit-server` is the correct direction for the deep-client branch. It gives the
benefits wanted from FFI/native mode while keeping Orbit as a Python project and
avoiding user-facing `llama-server` installation in the future.

## Phase 0: Protocol Parity Notes

Implemented in this branch:

- stable request parser for native chat payloads;
- native response builder with:
  - `finish_reason`;
  - `session_id`;
  - prompt/completion token counts;
  - reused/evaluated prompt token counts;
  - prefill/generation timings;
- OpenAI-like response bridge for current `LlamaServerBackend`;
- typed native SSE events for future Orbit-native backend adapter;
- `/sessions` endpoint exposing the currently supported single session;
- `/cancel` endpoint for backend-owned cancellation.

Differences from `llama-server`:

- `orbit-server` owns KV reuse directly instead of relying on
  `cache_prompt=true`.
- Native progress is emitted from the decode loop, not inferred through `/slots`.
- The native protocol can report `evaluated_tokens` separately from
  `reused_tokens`.
- The compatibility endpoint intentionally remains minimal.

Known incompatibilities:

- Tool schemas and llama-server built-in tools are not exposed by
  `orbit-server`.
- Multi-session scheduling is not implemented.
- Native MTP is not implemented.

Decision:

Phase 0 is good enough for continued iteration. It proves the protocol boundary
and backend ownership model, but it is not ready to replace `llama-server`.

## Phase 1: Chat Template Parity

Implemented in this branch:

- explicit Gemma 4 chat renderer for native mode;
- support for `system`, `user`, `assistant`, and `tool` messages;
- support for assistant `tool_calls` in Gemma 4 control-token format;
- support for tool responses in Gemma 4 control-token format;
- fallback to the explicit renderer when `llama_chat_apply_template()` cannot
  apply the model template;
- generation prompt aligned with the baseline `llama-server --reasoning off`
  profile;
- incremental UTF-8 detokenization to avoid corrupting multibyte characters;
- compatible finish reasons:
  - `stop`;
  - `length`;
  - `cancelled`;
  - `error`;
- stop-sequence trimming with early native cancellation.

The native template is intentionally kept in:

```text
src/orbit/native_llama/chat_template.py
```

It does not move tool execution, routing, memory, repair, verification, guards,
or final-answer policy into the backend.

### Native vs llama-server Benchmark

Same payloads, OpenAI-compatible endpoint, small output budgets:

```text
case                  backend       finish  prompt  output  cached  wall
chat breve            orbit-server  length      28      32       0  12.30s
chat breve            llama-server  length      27      32       8   7.70s
spiegazione tecnica   orbit-server  stop        30      33      13  11.52s
spiegazione tecnica   llama-server  stop        29      33       8   9.99s
prompt italiano       orbit-server  stop        34      11      13   5.48s
prompt italiano       llama-server  stop        33      12       8   3.97s
coding semplice       orbit-server  stop        38      18      13   8.07s
coding semplice       llama-server  stop        37      19       8   5.44s
prompt lungo          orbit-server  stop       588       7      13  48.54s
prompt lungo          llama-server  stop       587       8       8  49.12s
output con stop       orbit-server  stop        27       2      13   1.73s
output con stop       llama-server  stop        26       2      12   2.22s
max_tokens basso      orbit-server  length      30       3      13   2.31s
max_tokens basso      llama-server  length      29       3       8   2.43s
```

Observed qualitative results:

- no Gemma control-token leaks;
- no empty or corrupted responses;
- Italian UTF-8 output is preserved;
- stop sequence output is trimmed correctly;
- `max_tokens` low returns `finish_reason: length`;
- native streaming returns deltas and `[DONE]`;
- Orbit CLI can talk to `orbit-server` through the compatibility endpoint.

Tool-message smoke test:

```text
assistant tool_call + tool result -> README.md
finish: stop
special token leak: false
```

Differences from `llama-server`:

- prompt token counts differ by one token in simple cases because the native
  renderer applies the Gemma 4 template explicitly;
- `orbit-server` reports controlled prefix reuse from its own KV state;
- generation is currently greedy-only and does not use MTP;
- OpenAI-compatible streaming remains a bridge and does not expose native
  progress events.

Decision:

Phase 1 is good enough to proceed to Streaming Parity. Chat formatting is now
stable enough for comparative work, but `orbit-server` is still experimental
until streaming, cancellation, session lifecycle, and tool parity are measured
against the full Orbit regression suites.

## Phase 3: Streaming Parity

Implemented in this branch:

- stop-aware streaming filter for native generation;
- incremental UTF-8 streaming preserved from Phase 1;
- no duplicated final content in the OpenAI-compatible stream bridge;
- native cancellation through `/cancel`;
- typed native stream events remain available on `/chat/stream`:
  - `progress.prefill`;
  - `progress.generation`;
  - `delta`;
  - `metrics`;
  - `done`.

The main bug found in this phase was stop-sequence leakage in streaming mode.
The non-streamed final content was trimmed, but deltas could already have
emitted the stop sequence. The fix keeps a small suffix buffer equal to the
longest possible stop-sequence prefix and emits only text that can no longer be
part of a stop sequence.

The native backend now distinguishes:

- user/backend cancellation -> `finish_reason: cancelled`;
- internal stop-sequence abort -> `finish_reason: stop`.

### Streaming Benchmark

Same payloads, OpenAI-compatible streaming endpoint:

```text
case          backend       finish  TTFT   wall   content
stream_short  llama-server  stop    2.20s  2.69s  hello
stream_short  orbit-server  stop    2.29s  2.61s  hello
stream_stop   llama-server  stop    1.76s  2.43s  alpha
stream_stop   orbit-server  stop    2.50s  2.82s  alpha
stream_utf8   llama-server  stop    2.22s  3.32s  città è già qui
stream_utf8   orbit-server  stop    3.03s  4.87s  città è già qui
```

Observed:

- no duplicated deltas;
- no special-token leaks;
- UTF-8 output is intact;
- stop sequence is not emitted;
- final `finish_reason` matches `llama-server`;
- explicit backend cancellation returns `finish_reason: cancelled`.

Cancellation smoke:

```text
POST /chat/stream long generation
POST /cancel after 2s
finish_reason: cancelled
wall: 2.02s
```

Behavioral parity blocker:

`long_command_pressure` initially failed against `orbit-server` while passing
against `llama-server`.

Root cause:

- the runtime sent OpenAI-compatible `tools`;
- `llama-server` rendered those tools through the Gemma 4 template;
- `orbit-server` initially ignored `tools`, so the model missed part of the
  backend/tool contract during guard-sensitive coding turns.

Fix:

- preserve `tools` in the native protocol request;
- render Gemma 4 tool declarations inside the system turn using the model's
  native wrapper:
  - `<|tool>`;
  - `declaration:<name>{...}`;
  - `<tool|>`;
- keep tool execution, tool selection, repair, memory, guards, and policy in
  the Orbit runtime.

Additional streaming parser fix:

- native output can include empty Gemma channel blocks such as
  `<|channel>thought\n<channel|>`;
- `llama-server` strips these through its chat parser;
- `orbit-server` now strips generated channel blocks before emitting deltas or
  final content.

Confirmed blocker result:

```text
long_command_pressure against orbit-server: PASS

orbit-server:
  cat parser.py
  sed local return-value replacement
  PASS

llama-server:
  ls -F
  cat parser.py
  sed local return-value replacement
  PASS
```

### Promotion Run

Full release confidence against `orbit-server`:

```text
summary: 13/15 passed
json: /tmp/orbit-release-confidence-native-phase3-promotion.json
```

Baseline `llama-server` release confidence is documented as:

```text
summary: 12/15 passed
```

Residual differences:

```text
html_multiline_title:
  orbit-server: FAIL
  llama-server baseline: FAIL
  classification: non-blocking known limitation
  reason: fragile multiline HTML patch generation

css_regex_sensitive:
  orbit-server: FAIL
  llama-server baseline: FAIL
  classification: non-blocking known limitation
  reason: fragile command generation with regex-sensitive CSS

shell_script_hardening:
  orbit-server: PASS
  llama-server baseline: FAIL
  classification: positive variance / benchmark noise
  reason: model-driven patch strategy succeeded in this native run

long_command_pressure:
  orbit-server: PASS
  llama-server baseline: PASS
  classification: blocker closed
  reason: Gemma 4 tool declaration rendering restored model-facing contract
```

No new blocker was observed.

Decision:

PROMOTE Phase 3.

Phase 3 Streaming Parity is formally promoted. The native backend now matches
the measured streaming semantics required for continued iteration:

- no duplicated deltas;
- no control-token leaks;
- UTF-8 output preserved;
- stop sequences not emitted;
- `finish_reason` compatible with `llama-server`;
- cancellation reports `cancelled`;
- release confidence is at or above the `llama-server` baseline.

Next phase is Backend Reliability. Do not start it until this promotion state
is accepted.

## Phase 4: Reliability & Lifecycle Validation

Phase 4 validates one backend reliability risk at a time. It does not include
performance work, MTP, broad stress testing, or architectural changes.

### Iteration 1: Cancel During Prefill

Risk:

```text
cancel during prefill
```

Why this risk first:

- prefill is the longest silent phase on CPU-only systems;
- cancellation must interrupt native `llama_decode()`, not only disconnect the
  client;
- the backend owns cancellation and decode-loop lifecycle.

Minimal test:

- start `orbit-server`;
- send `/chat/stream` with a long prompt;
- wait for `progress.prefill` where `current < total`;
- call `/cancel`;
- expect `done` with `finish_reason: cancelled`;
- expect no generation progress before cancellation.

Observed result:

```text
status: PASS
first_prefill: 64/10828
prefill_events: 2
generation_events: 0
finish_reason: cancelled
wall: 3.9s
```

Decision:

PASS. No fix required.

Next recommended risk:

```text
client disconnect during stream
```

### Iteration 2: Client Disconnect During Stream

Risk:

```text
client disconnect during stream
```

Minimal test:

- start `orbit-server`;
- send `/chat/stream` with a long generation request;
- wait for the first SSE progress/generation event;
- close the client connection without calling `/cancel`;
- verify `/health`;
- immediately send a short `/chat` request to check whether the backend lock
  was released.

Observed before fix:

```text
stream_status: 200
events: progress.prefill, progress.generation
client_closed: yes
health: ok
follow-up /chat: timeout
```

The native server remained healthy, but the abandoned generation continued to
occupy the single backend context until it finished or reached its token limit.

Options evaluated:

```text
stdlib watchdog/lifecycle:
  complexity: low
  reliability: good for real TCP close/EOF, still dependent on OS socket state
  impact: backend HTTP layer only
  regression risk: low
  validation: raw streaming client closes TCP connection, then /chat must respond

minimal ASGI streaming layer:
  complexity: medium/high
  reliability: stronger request disconnect signal
  impact: new dependency/server lifecycle
  regression risk: higher
  validation: ASGI receive loop observes http.disconnect during SSE
```

Decision:

Use the stdlib lifecycle watcher first. It is the smallest backend-only change
and avoids migrating the experimental server before proving a real need.

Implemented:

- catch `BrokenPipeError` and `ConnectionResetError` while writing SSE events;
- cancel the native client when SSE writes fail;
- pass a backend-owned `should_cancel` predicate into the native decode loop;
- check socket hangup using `poll()`/`POLLRDHUP` plus a `select()`/`MSG_PEEK`
  fallback before progress and delta writes;
- add a request-scoped disconnect watcher for streaming requests;
- the watcher observes EOF/error on the HTTP socket in parallel with model
  generation and calls `client.cancel()` when the peer disconnects.

Observed with a high-level `HTTPConnection` close:

```text
stream_status: 200
events: progress.prefill, progress.generation
client_closed: yes
health: ok
follow-up /chat: timeout
```

That case did not produce an immediate observable TCP EOF from the server side
and is not a reliable disconnect validation.

Observed with a raw TCP streaming client that closes the connection:

```text
sent: yes
saw_event: true
raw_close: yes
health: ok
follow-up /chat: ok, 2.08s
```

Decision:

PASS for real client disconnect.

The fix is backend-only. It does not move runtime behavior, tool handling,
guards, repair, memory, or final-answer policy into `orbit-server`.

Remaining caveat:

The stdlib server still has no framework-level `request.is_disconnected()`.
If future clients expose cases where TCP EOF is delayed or hidden, the next
step should be a minimal ASGI streaming layer rather than more prompt/runtime
logic.

### Iteration 3: Internal Error During Stream

Risk:

```text
internal backend error during stream
```

Minimal test:

- run an in-process native server with a fake backend;
- emit `progress.prefill`;
- emit one partial `delta`;
- emit `progress.generation`;
- raise `RuntimeError("injected backend failure during stream")`;
- verify stream termination and follow-up request reuse.

Observed:

```text
events: progress.prefill, delta, progress.generation, error, done
error: injected backend failure during stream
finish_reason: error
follow-up /chat: ok, 0.03s
```

Decision:

PASS. No fix required.

The existing stream error path emits an explicit `error` event and `done` with
`finish_reason: error`, then releases the backend lock. The next request can
reuse the session normally.

### Iteration 4: Repeated Interrupted Streams

Risk:

```text
session cleanup after repeated interrupted streams
```

Minimal test:

- use the real backend;
- run 5 `/chat/stream` requests;
- wait for first generation events;
- call `/cancel`;
- verify each stream ends and a final `/chat` succeeds.

Observed:

```text
iterations: 5
each stream: finish_reason=cancelled
stream threads alive after cancel: false
follow-up /chat: ok, 2.57s
```

Decision:

PASS. No fix required.

Repeated cancellations did not leave visible lock, session, or thread state
behind.

### Iteration 5: Malformed Requests

Risk:

```text
malformed requests
```

Minimal test:

- invalid JSON;
- empty body;
- missing `messages`;
- wrong `messages` type;
- invalid `max_tokens`;
- then a valid `/chat`.

Observed before fix:

Some malformed payloads started real generation with an empty/default prompt.
This could occupy the native context and cause client timeouts.

Fix:

- request parser now rejects malformed `messages`;
- `messages` must be a non-empty list of message objects, unless legacy
  `prompt` is a string;
- `max_tokens`, when present, must be a positive integer;
- request parsing errors now return HTTP 400.

Observed after fix:

```text
invalid_json:        400 {"error": "invalid JSON"}
empty_body:          400 {"error": "messages must be a list"}
missing_messages:    400 {"error": "messages must be a list"}
messages_wrong_type: 400 {"error": "messages must be a list"}
max_tokens_invalid:  400 {"error": "max_tokens must be a positive integer"}
follow-up /chat:     ok, 3.13s
```

Decision:

PASS after parser fix.

### Iteration 6: Malformed Requests and Cancellation Mix

Risk:

```text
session reuse after malformed requests and cancellation mix
```

Minimal test:

```text
1. malformed request
2. valid request
3. long stream
4. cancel
5. malformed request
6. valid request
```

Observed:

```text
step1 malformed: 400 invalid JSON
step2 valid:     200 alpha., 3.55s
step4 cancel:    200 cancel_requested
step3 stream:    finish_reason=cancelled
step5 malformed: 400 messages must be a list
step6 valid:     200 omega., 3.65s
```

Decision:

PASS. No fix required beyond the malformed request parser fix.

Malformed requests and cancellation do not contaminate the shared native
session or the reused context in the measured sequence.

## Phase 4 Promotion Decision

Final status:

```text
PROMOTE Phase 4
```

Risks investigated:

```text
cancel during prefill: PASS
client disconnect during stream: PASS with caveat
internal error during stream: PASS
repeated interrupted streams: PASS
malformed requests: PASS after fix
malformed + cancellation mix: PASS
```

Lifecycle bugs fixed:

- abandoned streaming generation after real client disconnect;
- malformed requests starting generation with empty/default prompts;
- malformed request errors returning non-specific conflict status.

Risks validated without code changes:

- cancel during prefill;
- internal backend exception during stream;
- repeated interrupted streams;
- malformed/cancel mixed session reuse.

Open blockers:

```text
none
```

Non-blocking caveat:

- `ThreadingHTTPServer` has no framework-level `request.is_disconnected()`.
  The current watcher handles real TCP EOF/error. If a future client hides or
  delays EOF, evaluate a minimal ASGI streaming layer instead of adding runtime
  behavior.

Future hardening:

- validate session cleanup after backend load/model errors;
- validate malformed native `/v1/chat/completions` bridge payloads separately;
- add a focused regression test for malformed request parsing once the native
  server files are promoted out of experiment.

Decision rationale:

Phase 4 validated the main lifecycle risks for the current single-session
native backend without moving runtime behavior into `orbit-server`. The backend
now handles cancellation, stream errors, repeated interruptions, malformed
requests, and mixed malformed/cancel sequences well enough to continue to the
next roadmap phase.

Next phase:

```text
Phase 5: Performance Parity
```

Do not start Phase 5 until this promotion state is accepted.

## Current MTP Experimental State

The persistent MTP path has an additional experimental branch controlled by:

```text
ORBIT_MTP_BOUNDARY_SPLIT=0|1
```

Status:

- The boundary-split path is now the candidate default for persistent MTP.
- Default no-flag behavior enables the split path.
- `ORBIT_MTP_BOUNDARY_SPLIT=0` forces the rollback behavior for comparison and
  safe disablement.
- Runtime fallback and invariant checks remain active if the live boundary is
  not safe.

Boundary-split notes:

- The split path separates the logical frontier size from the validate batch
  starting position.
- The goal is to match the server-side partial/live boundary more closely while
  preserving the old path as an opt-out fallback.
- Because long-prompt behavior can diverge from the baseline while still being
  reference-like, correctness should be judged against the MTP invariants and
  reference path, not only against the old default output.

Draft helper policy:

- The MTP draft helper currently uses an intentional `top1-greedy` policy.
- `common_sampler_sample(...)` may return a token different from the highest
  probability candidate.
- That sampled token is not the committed draft token in the current helper.
- The committed draft token is the sorted top-1 candidate and must stay aligned
  with:
  - sampler accept;
  - `spec_draft` push;
  - next draft decode input.
- A naive switch to sampled-token draft policy regressed CPU-only latency and
  acceptance on real prompts, so no sampled-token draft policy is enabled.

Debugging aids kept intentionally gated:

- `LLAMA_MTP_DRAFT_MISMATCH_TRACE=1`

These traces are for boundary-level diagnosis only and should stay off during
normal benchmarking.

CPU-only benchmark discipline:

- Use fixed CPU affinity for comparable runs, for example `taskset -c 0-5`.
- Keep the same threads, batch, ubatch, prompt, model, and draft model between
  runs.
- Tail latency on CPU-only systems is sensitive to scheduler noise; do not
  promote or reject an experimental MTP change from one-off median measurements
  without controlled repeated runs.
