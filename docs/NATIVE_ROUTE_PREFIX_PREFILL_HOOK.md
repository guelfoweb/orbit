# Native Route Prefix Prefill Hook

## Scope

This document describes the first internal hook for native route prefix
prefill-only capture.

The hook is intentionally not a runtime prewarm feature. It is not called from
server startup, `/tools on`, `/reset`, `/chat`, `/chat/stream`, or the normal
route path. It is an internal primitive that can be invoked explicitly by probe
or test code to validate the lifecycle needed by a future prewarm integration.

## What It Introduces

- `NativeLlamaClient.capture_route_prefix_prefill_only(...)`
- Metadata-only result reporting through `NativeRoutePrefixPrefillResult`
- A manual probe script path that invokes the hook directly
- Mocked unit coverage for success, skip, failure, lock, and metadata behavior

The hook:

- uses the same stable route tools-on prefix structure used by the runtime route
  prefix anchor
- tokenizes with the native tokenizer
- decodes only the stable prefix
- captures a real KV checkpoint after the prefix decode
- returns explicit metadata
- sets `restore_ready=true` only after complete success

## What It Does Not Introduce

- no automatic prewarm
- no startup integration
- no `/tools on` integration
- no `/chat` or `/chat/stream` integration
- no fake user request
- no prompt workaround
- no `max_tokens=0` completion path
- no model output generation
- no prompt change
- no routing, tool selection, final-answer, or evidence policy change
- no public stable endpoint
- no release or tag change

## Guardrails

The hook is native-client-only. Non-native backends have no production path to
call it.

The hook skips without decode or checkpoint capture when:

- `ORBIT_KV_PREFIX_ANCHOR=off`
- tools mode is not `on`
- another native request is in flight
- a continuation or cached prompt context is active
- another prefill-only capture is already running on the same client

Legacy compatibility follows the existing prefix-anchor configuration:

- if `ORBIT_KV_PREFIX_ANCHOR` is unset, `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1`
  still enables auto behavior
- `ORBIT_KV_PREFIX_ANCHOR=off` wins over the legacy flag

## Locking And State Safety

The hook uses a non-blocking per-client lock. A concurrent call skips with
`prefill_in_flight` instead of waiting or sharing mutable decode state.

The hook is idle-only. It refuses to run when the native client has an active
request, continuation state, or cached prompt tokens. This avoids destroying an
active completion context and keeps the hook out of normal user-visible
interaction.

Checkpoint state is published only after the full prefix decode and checkpoint
capture succeed. Decode failure, cancellation, invalid prefix boundary, or
checkpoint failure clear partial decode state and return `restore_ready=false`.

## Cancellation And Failure

Cancellation during decode returns a failed result and does not mark the
checkpoint as valid.

Failure modes are reported as metadata-only reasons such as:

- `anchor_disabled`
- `tools_mode_ineligible`
- `native_client_not_loaded`
- `native_request_in_flight`
- `active_context_present`
- `prefill_in_flight`
- `route_boundary_unavailable`
- `route_anchor_plan_unavailable`
- `prefix_decode_failed:<error-type>`
- `cancelled`
- checkpoint capture failure reason from the prefix-anchor lifecycle

No user-facing runtime error is introduced because the hook is not called by
the normal runtime.

## Metadata

The result object reports:

- `attempted`
- `succeeded`
- `skipped`
- `skip_reason`
- `failed_reason`
- `prefix_hash`
- `prefix_token_count`
- `checkpoint_size_bytes`
- `prefill_ms`
- `decode_calls`
- `sampled_tokens`
- `generated_tokens`
- `sampler_touched`
- `session_history_touched`
- `restore_ready`

The hook does not print raw prompt text, raw tokens, user content, tool output,
file content, web content, or full tool specs.

## Manual Probe Path

`scripts/probe_native_route_prefix_prefill_only.py` now invokes the internal
hook directly after building the stable route tools-on prefix. It remains a
manual probe and is not part of the runtime lifecycle.

Expected successful probe properties:

- `sampled_tokens=0`
- `generated_tokens=0`
- `sampler_touched=false`
- `session_history_touched=false`
- `restore_ready=true`

The previously observed native probe values remain the reference baseline for
the tested setup:

- prefix token count: 693
- checkpoint size: 238454176 bytes
- sampled tokens: 0
- generated tokens: 0

## Tests

Unit tests cover:

- `ORBIT_KV_PREFIX_ANCHOR=off` skips without decode
- `ORBIT_KV_PREFIX_ANCHOR=off` wins over legacy experiment mode
- legacy experiment mode still enables the hook when the new variable is unset
- tools-off skips
- mocked success produces `restore_ready=true`
- decode failure returns `restore_ready=false`
- active context skips
- concurrent prefill skips
- metadata stays content-free

The standard test suite does not require a real model.

## Next Step

The next PR can decide whether to expose a controlled internal lifecycle trigger.
That future work must still prove:

- explicit lifecycle trigger semantics
- lock and cancellation behavior under real server concurrency
- invalidation on model, tokenizer, template, tools, capabilities, backend, or
  config changes
- fallback to baseline on every failure
- unchanged file/read/list/web/fetch behavior
- unchanged prompt, routing, tool selection, final-answer, and evidence policy

Runtime prewarm remains out of scope for this hook PR.
