# KV Route Prefix Anchor Runtime Experiment

## Scope

This experiment adds an opt-in native KV prefix-anchor path for Orbit route calls.

Feature flag:

```text
ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1
```

Default is off. With the flag off, the payload, prompt rendering, routing, tool selection, final policy, and native cache behavior stay on the baseline path.

The only eligible production path is:

- tools on
- route phase
- no explicit native tool schema parameter
- no thinking mode
- no multimodal payload

The experiment does not apply to `chat_final`, `final_from_tool`, or `tool_call`.
File-read, web/fetch, and directory-listing requests may benefit only from the
shared route prefix before the model decides which tool to use. They are not
special-cased and no post-route tool/final path is anchored.

## Design

The runtime already marks model calls with phase metadata. When the feature flag is enabled, the HTTP backend adds a metadata-only `route_prefix_anchor=true` payload field only while the current model call context is `phase=route` and `tools_mode=on`.

The native client then performs conservative validation before attempting anchor use:

- Render the normal prompt exactly as before.
- Build `RoutePromptSegments` for the same messages.
- Require `segments.full_prompt_text == prompt`.
- Tokenize `segments.stable_prefix_text`.
- Require the stable-prefix tokens to be an exact prefix of the full prompt tokens.
- Compute a prefix-anchor key from model/template identity and stable-prefix hashes.
- Restore only if the stored checkpoint has the same key and token count.
- Fallback to baseline prefill on any mismatch or native error.
- Keep only one route anchor checkpoint per native client instance.

The model receives the same prompt text. The route decision remains model-guided.

## Memory And Invalidation

The implementation stores at most one checkpoint in each `NativeLlamaClient`.
There is no map of per-prompt anchors and no unbounded growth. Capturing a new
valid route prefix replaces the previous state.

For the tested Gemma 4 12B Q4_K_M model at `ctx=8192`, the route checkpoint was:

- prefix token count: 693
- checkpoint size: 238,454,176 bytes

This is a material memory cost and is the main reason the experiment remains
opt-in. The checkpoint is invalidated or bypassed if any compatibility input
changes:

- prefix hash
- token count
- model id
- template id
- tool schema hash
- capability summary hash
- runtime policy hash
- route contract hash
- backend/native version
- tools mode

If an existing checkpoint fails validation or restore, the current request falls
back to the full baseline prompt path. It does not expose an error to the user.
The next compatible request may capture a fresh checkpoint.

## Boundary

The route token-boundary probe previously verified the real native tokenizer:

- stable route prefix: 693 tokens
- token LCP with full route prompt: 693 tokens
- tested scenarios: short chat, trivial chat, medium no-tool, listing, file-read, web, fetch
- result: `route_boundary_token_prefix_ok=true`

This experiment uses that boundary but still validates every runtime call before restore.

## Fallback

Fallback reasons are metadata-only and include:

- `route_prompt_mismatch`
- `route_boundary_unavailable`
- `token_boundary_mismatch`
- `anchor_invalid`
- `checkpoint_restore_failed`
- `checkpoint_capture_failed`

Fallback never raises a visible user error. It rebuilds the full prompt through the baseline path.

## Diagnostics

When `ORBIT_KV_DIAG=1`, the native backend emits `kv_diag_route_prefix_anchor` events containing only metadata:

- `route_anchor_enabled`
- `route_anchor_attempted`
- `route_anchor_hit`
- `route_anchor_miss`
- `capture_attempted`
- `restore_attempted`
- `restore_used`
- `fallback_reason`
- `prefix_hash`
- `prefix_token_count`
- `checkpoint_size`
- `checkpoint_size_bytes`
- `checkpoint_age_ms`
- `anchor_invalidated`
- `invalidation_reason`
- `cached_tokens`
- `evaluated_tokens`
- `lcp_tokens`
- `phase`

The diagnostics do not log raw prompt text, raw token ids, user content, tool output, file content, or web content.

## Benchmark

Environment:

- model: `/home/guelfoweb/LAB/orbit/models/ggml-org--gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf`
- backend: native Orbit server, CPU, `ctx=8192`
- server port: local dedicated test server
- `ORBIT_KV_DIAG=1`
- workdir: repository root
- max tokens: 64 for all smoke prompts

Observed route-prefix checkpoint:

- prefix token count: 693
- checkpoint size: 238,454,176 bytes
- first eligible route: capture miss
- later eligible routes: restore hit
- server-side route anchor events: metadata-only

The run used one dedicated local native server. Other unrelated local Orbit
servers were stopped before the main smoke to avoid unnecessary memory pressure.

Representative results:

| Scenario | Flag | Phases | Route cached/evaluated | Request cached/evaluated | Wall | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| `hi` | off | `route -> chat_final` | `0 / 706` | `4 / 736` | ~60s | `CHAT` |
| `hi` first eligible | on | `route -> chat_final` | `0 / 706` | `4 / 736` | ~64s | capture miss |
| `hi` repeat | on | `route -> chat_final` | `693 / 13` | `697 / 43` | ~8.6s | restore hit |
| `what is 2+2?` | off | `route` | `4 / 708` | `4 / 708` | ~59s | direct route final |
| `what is 2+2?` | on | `route` | `693 / 19` | `693 / 19` | ~2.4s | direct route final |
| medium no-tool 1 | off warm | `route -> chat_final` | `693 / 30` | `697 / 77` | ~27s | warm baseline control |
| medium no-tool 1 | on | `route -> chat_final` | `693 / 30` | `697 / 77` | ~28s | restore hit |
| medium no-tool 2 | off | `route -> chat_final` | `4 / 714` | `8 / 756` | ~79s | cache miss |
| medium no-tool 2 | on | `route -> chat_final` | `693 / 25` | `697 / 67` | ~26s | restore hit |
| `list files in the workdir` | off | `route -> final_from_tool` | `4 / 707` | `711 / 874` | ~93s | `list_directory` preserved |
| `list files in the workdir` | on | `route -> final_from_tool` | `693 / 18` | `1386 / 199` | ~41s | `list_directory` preserved |
| `read README.md and explain it` | off warm | `route -> final_from_tool` | `696 / 16` | `1404 / 424` | ~64s | `Read` content evidence preserved |
| `read README.md and explain it` | on | `route -> final_from_tool` | `693 / 19` | `1386 / 442` | ~64s | `Read` content evidence preserved |
| web search prompt | off warm | `route -> final_from_tool` | `696 / 18` | `1406 / 447` | ~59s | `orbit-web-search` preserved |
| web search prompt | on | `route -> final_from_tool` | `693 / 21` | `1386 / 467` | ~62s | `orbit-web-search` preserved |
| `fetch https://example.com and summarize it` | off warm | `route -> final_from_tool` | `696 / 18` | `1406 / 149` | ~32s | fetch path preserved |
| `fetch https://example.com and summarize it` | on | `route -> final_from_tool` | `693 / 21` | `1386 / 169` | ~33s | fetch path preserved |

The medium no-tool scenarios used synthetic prompts:

- `Explain in two short paragraphs how a local AI runtime can reduce latency after the first turn.`
- `Explain the tradeoff between correctness and latency in a local agent runtime.`

Notes:

- The first `on` request must pay capture cost. The benefit appears after the checkpoint exists.
- One medium smoke used a small output budget and stopped final generation with `finish_reason=length`; the route phase still completed normally and did not enter repair/retry.
- File-read correctness was checked with `README.md`. The route selected a content-reading path, not `list_directory`.
- Web/fetch stayed at `route -> final_from_tool`.
- `route_no_decision_length_retry` was not observed in the tested scenarios.
- The A/B run was intentionally bounded for CPU runtime. It is sufficient to
  validate the opt-in experiment and known risks, but not sufficient to promote
  the feature to default.

## Current Verdict

Promote as opt-in experiment, not default.

The route-prefix anchor is technically effective and preserves the tested
model-guided control flow. It significantly reduces evaluated route tokens after
the first capture miss, including a measured cache-miss medium route changing
from `4 / 714` cached/evaluated to `693 / 25`.

It should remain behind `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1` because:

- the first capture miss is expensive;
- checkpoint memory is large for this model and context;
- more repeated-run data is needed before considering any default behavior;
- the benchmark should be repeated with a longer final budget for medium prompts.

Merge criteria for the opt-in experiment are satisfied:

- flag off preserves baseline payload and behavior;
- flag on is limited to route tools-on on the native backend;
- no deterministic routing, tool selection, final policy, or evidence policy changes;
- fallback returns to baseline on validation or restore failure;
- file-read, web/fetch, and listing paths remain model-guided and evidence-safe;
- diagnostics are metadata-only.
