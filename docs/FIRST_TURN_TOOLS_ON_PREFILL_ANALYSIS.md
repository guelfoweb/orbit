# First-Turn Tools-On Prefill Analysis

## Summary

This report investigates the first tools-on conversational turn cost after the
route KV prefix-anchor work.

The observed slow first tools-on turn is not caused by a retry loop and is not a
tool-selection regression. On a cold native server, the first tools-on
conversational turn performs:

- one `route` model call over the full route prompt
- one `chat_final` model call for the conversational answer
- one route prefix-anchor capture miss

The repeat tools-on turn benefits from the route prefix-anchor and avoids most
of the stable route prefix prefill.

No runtime patch was applied in this pass. A safe prewarm needs a backend-native
prefill-only lifecycle hook or endpoint; the current production API captures the
anchor during a real route request.

## Environment

- Backend: native Orbit server
- Model: Gemma 4 12B Q4_K_M local GGUF
- Context: 8192
- Threads: 6
- Threads batch: 6
- Batch: 256
- UBatch: 128
- Route KV prefix-anchor default: `auto`
- Kill switch: `ORBIT_KV_PREFIX_ANCHOR=off`

The measurements below use metadata only. They do not include raw prompts, raw
tokens, user content, tool output, file content, or web content.

## Data

### Cold Server, Tools-On Auto, First Conversational Turn

| Phase | prompt_tokens | cached_tokens | evaluated_tokens | completion_tokens | finish | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| route | 711 | 0 | 711 | 5 | stop | route prefix capture miss |
| chat_final | 39 | 4 | 35 | 39 | stop | conversational final |

Wall time: about 69.7s

Model calls: 2

Route outcome: route selected chat continuation.

Retry/repair: none observed.

Route prefix-anchor event:

| Field | Value |
| --- | --- |
| route_anchor_enabled | true |
| route_anchor_attempted | true |
| route_anchor_miss | true |
| route_anchor_hit | false |
| capture_attempted | true |
| restore_attempted | false |
| restore_used | false |
| prefix_token_count | 693 |
| checkpoint_size_bytes | 238454176 |
| cached_tokens | 0 |
| evaluated_tokens | 711 |
| lcp_tokens | 693 |

### Cold Server, Tools-On Auto, Immediate Repeat

| Phase | prompt_tokens | cached_tokens | evaluated_tokens | completion_tokens | finish | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| route | 766 | 693 | 73 | 18 | stop | route prefix restore hit |

Wall time: about 12.5s

Model calls: 1

Retry/repair: none observed.

Route prefix-anchor event:

| Field | Value |
| --- | --- |
| route_anchor_enabled | true |
| route_anchor_attempted | true |
| route_anchor_miss | false |
| route_anchor_hit | true |
| capture_attempted | false |
| restore_attempted | true |
| restore_used | true |
| prefix_token_count | 693 |
| checkpoint_size_bytes | 238454176 |
| cached_tokens | 693 |
| evaluated_tokens | 73 |
| lcp_tokens | 693 |

### Cold Server, Tools-On With KV Prefix-Anchor Off

| Phase | prompt_tokens | cached_tokens | evaluated_tokens | completion_tokens | finish | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| route | 711 | 0 | 711 | 5 | stop | baseline route prefill |
| chat_final | 39 | 4 | 35 | 39 | stop | conversational final |

Wall time: about 69.9s

Model calls: 2

Retry/repair: none observed.

Diagnosis: the first-turn cost is effectively the same with the anchor disabled,
which confirms that the slow first turn is the cold route prefill plus
conversational final, not an anchor-induced regression.

### Tools-Off Conversational Turn

| Phase | prompt_tokens | cached_tokens | evaluated_tokens | completion_tokens | finish | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| chat_final | 39 | 4 | 35 | 39 | stop | no route/tools prompt |

Wall time: about 14.4s

Model calls: 1

Retry/repair: none observed.

## Diagnosis

The first tools-on conversational turn cost is route plus chat final:

- route: 711 prompt tokens evaluated on cold server
- chat_final: 35 prompt tokens evaluated
- total evaluated prompt tokens: about 746
- model calls: 2

The route prompt dominates first-turn prefill. The stable route prefix is 693
tokens in this setup.

The immediate repeat shows the expected prefix-anchor behavior:

- stable prefix restored: 693 cached tokens
- evaluated route suffix: 73 tokens
- route-only direct final in the repeat case

No `route_no_decision_length_retry`, repair loop, tool-call loop, or evidence
recovery loop was observed in the first conversational turn measurements.

The manual observation of a larger total token count is consistent with footer
or session-level accounting that can include more than one model call or prior
context. The phase-level measurement shows the backend route prefill itself is
711 tokens on this cold route call.

## Option Evaluation

### Option 1: Route Prefix-Anchor Prewarm

Prewarming is the most plausible way to improve first tools-on conversational
latency without changing prompt semantics or model-guided routing.

A safe design would:

- run only on native backend
- respect `ORBIT_KV_PREFIX_ANCHOR=off`
- build the same stable route prefix used by the real route prompt
- prefill and checkpoint that stable prefix without generating an answer
- validate the same prefix hash, token count, model, template, tools, and
  capability identity used by runtime route calls
- fall back silently if prewarm fails
- keep at most one checkpoint per native client/session

Current blocker:

- the production server API exposes chat completion and continuation, not a
  prefill-only route-prefix prewarm operation
- capturing the checkpoint today is wired into a real route request
- doing prewarm through a fake chat request would generate output or require a
  prompt-shaped workaround, which is not acceptable

Recommendation:

- do not implement prewarm by reusing a fake user request
- add a bounded native prewarm endpoint or internal lifecycle hook in a separate
  PR if first-turn latency is important enough to justify the protocol change

### Option 2: Route Direct Final For Simple Chat

The cold first conversational turn currently goes `route -> chat_final`.

The route contract already allows one-sentence direct answers when no tools or
external evidence are needed. However, changing prompt wording to make more
short chat requests close in route would alter model behavior and risks
reintroducing prompt-shape tuning problems.

Recommendation:

- do not change the route prompt in this phase
- do not hardcode greetings or simple-chat phrases
- keep this as a benchmark-only topic if future measurements show broad benefit

### Option 3: Mini-Route Tools-Needed

A smaller first classifier could reduce route prompt size, but it would add a
model call and creates pressure toward deterministic or duplicated routing
policy.

On CPU, an added call is likely to hurt latency unless it is extremely small and
highly reliable.

Recommendation:

- do not implement without strong comparative benchmark data
- treat as higher risk than prefix prewarm

## Recommendation

No runtime patch should be merged from this investigation.

The next safe technical step, if desired, is a dedicated native prewarm design:

1. Add a native-server endpoint or internal command that performs prefill-only
   capture of the stable route tools-on prefix.
2. Keep it behind the existing auto/off mode and native-only eligibility.
3. Trigger it explicitly from a safe lifecycle point, such as an optional
   tools-on warmup action, not silently during startup unless benchmarked.
4. Measure whether shifting the capture cost before the first user request is a
   better UX tradeoff than paying it on the first request.

Do not implement prompt changes, deterministic simple-chat routing, fake user
requests, or replay-token cache workarounds for this problem.

## PR Criteria For A Future Prewarm Patch

A future prewarm PR should prove:

- no prompt text or semantics change
- no route/tool/final/evidence policy change
- `ORBIT_KV_PREFIX_ANCHOR=off` prevents prewarm
- non-native backends do not prewarm
- failed prewarm does not affect the next request
- first real tools-on route uses restore hit after successful prewarm
- file-read, listing, web search, fetch URL, and stale-evidence behavior remain
  unchanged
- checkpoint memory remains bounded to the current one-anchor lifecycle
- diagnostics are metadata-only

## Verdict

Verdict: report-only, no patch.

The first tools-on conversational turn is expensive because it is the first cold
route prompt evaluation plus a conversational final call. The route
prefix-anchor already fixes compatible repeats. Reducing the first turn further
requires safe prewarm infrastructure, not prompt or routing changes.
