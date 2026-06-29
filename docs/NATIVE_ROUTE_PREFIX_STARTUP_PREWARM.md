# Native Route Prefix Startup Prewarm

## Status

This document describes the first controlled runtime integration for native route
prefix prewarm.

The feature now runs by default when the native server starts with tools enabled.
It can still be disabled explicitly.

```bash
ORBIT_KV_PREFIX_PREWARM=startup  # default synchronous startup prewarm
ORBIT_KV_PREFIX_PREWARM=off      # disable startup prewarm
```

`ORBIT_KV_PREFIX_ANCHOR=off` disables startup prewarm even when
`ORBIT_KV_PREFIX_PREWARM=startup` is set or the prewarm variable is unset. The
legacy `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1` does not override `off`.

`ORBIT_TOOLS=off` disables startup route tools-on prewarm because there is no
tools-on route prefix to prepare for startup.

Invalid `ORBIT_KV_PREFIX_PREWARM` values fall back to `off`.

## What It Does

When enabled, startup prewarm uses the internal native prefill-only hook to
prepare the stable route tools-on prefix before the server starts accepting
requests.

The prewarmed prefix is the same route tools-on prefix used by real route calls:

- route system contract
- route tool schema
- stable route template boundary

The hook tokenizes the prefix with the native tokenizer, decodes only the prefix,
captures a real KV checkpoint, and marks the checkpoint restore-ready only after
full success.

No model response is generated during prewarm.

## What It Does Not Do

Startup prewarm does not introduce:

- background prewarm
- `/tools on` lifecycle integration
- `/chat` or `/chat/stream` integration
- system-prompt anchor
- chat-final anchor
- final-from-tool anchor
- tool-call anchor
- multiple anchors
- fake user request
- prompt workaround
- `max_tokens=0` through the normal generation path
- deterministic routing
- tool-selection changes
- final-answer policy changes
- file/web/fetch evidence-policy changes

## Lifecycle

`ORBIT_KV_PREFIX_PREWARM=startup` runs after the native model/client is loaded
and before the HTTP server starts serving requests.

This location is intentional:

- model, tokenizer, template, and native context are ready
- no user request is in flight
- no tool loop is active
- no session history is present
- the hook can use the native client's existing prefill lock

Because this is synchronous startup work, server readiness is delayed by the
prewarm duration when the feature is enabled. With the tested Gemma 4 12B CPU
setup, the prefill-only capture for the 693-token stable route prefix has been
observed around 49-59 seconds.

## Guardrails

Startup prewarm is eligible only when:

- `ORBIT_KV_PREFIX_PREWARM` is unset or `startup`
- tools are enabled for startup
- `ORBIT_KV_PREFIX_ANCHOR` is not `off`
- native backend/client is loaded
- route prefix-anchor support is available
- route tools-on stable prefix boundary is valid

It is skipped when:

- `ORBIT_KV_PREFIX_PREWARM=off`
- `ORBIT_KV_PREFIX_PREWARM` has an unrecognized value
- `ORBIT_TOOLS=off`
- `ORBIT_KV_PREFIX_ANCHOR=off`
- a native client request/context is already active
- the prefix-anchor hook reports an ineligible state

Failures are not user-facing. A failed prewarm leaves the server usable; the
first real route call falls back to the existing baseline/capture path.

## Locking, Cancellation, And Failure

The startup integration relies on the internal
`NativeLlamaClient.capture_route_prefix_prefill_only(...)` lock and validity
rules:

- concurrent capture on the same native client is rejected
- checkpoint is valid only after complete prefix decode and capture
- decode/capture failure clears target memory and invalidates the route anchor
- cancellation or incomplete prefill returns `restore_ready=false`
- sampler and session history are not touched

The startup lifecycle runs before serving requests, so there should be no
request/prewarm race in the normal startup path.

## Diagnostics

When `ORBIT_KV_DIAG=1` is enabled, startup prewarm emits metadata only:

- `prewarm_enabled`
- `prewarm_mode`
- `tools_default_enabled`
- `tools_startup_enabled`
- `prewarm_attempted`
- `prewarm_succeeded`
- `prewarm_skipped_reason`
- `prewarm_failed_reason`
- `prewarm_prefix_token_count`
- `prewarm_checkpoint_size_bytes`
- `prewarm_ms`
- `decode_calls`
- `sampled_tokens`
- `generated_tokens`
- `sampler_touched`
- `session_history_touched`
- `restore_ready`

Diagnostics must not include raw prompt text, raw token ids, tool specs, user
content, tool output, file content, or web content.

## Memory And Startup Tradeoff

The route prefix checkpoint is large. On the tested Gemma 4 12B CPU setup, the
checkpoint size is about 238 MB.

Startup prewarm trades startup readiness time for lower first-route latency on
eligible tools-on requests. It does not make tool execution faster and does not
cache tool results, file contents, web results, PDF contents, or user history.

## Validation Scope

Required validation for this integration:

- default/unset `ORBIT_KV_PREFIX_PREWARM` prewarms when tools are enabled
- `ORBIT_KV_PREFIX_PREWARM=off` does not prewarm
- `ORBIT_KV_PREFIX_PREWARM=startup` invokes the native prefill-only hook
- `ORBIT_KV_PREFIX_ANCHOR=off` skips startup prewarm
- `ORBIT_TOOLS=off` skips startup prewarm
- invalid prewarm values fall back to `off`
- hook success yields `restore_ready=true`
- hook failure leaves the server usable with `restore_ready=false`
- diagnostics stay metadata-only
- existing read/list/web/fetch behavior is unchanged
- stale-evidence and listing-to-read guardrails remain intact

## Next Steps

Possible future work requires separate benchmark evidence:

- background prewarm with explicit lock/readiness semantics
- explicit operator-triggered internal prewarm
- `/tools on` lifecycle integration

None of those are part of this startup-only opt-in integration.
