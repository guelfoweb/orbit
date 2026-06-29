# Native Route Prefix Prefill-Only Feasibility

## Objective

This report verifies whether Orbit's native backend can prefill and checkpoint
the stable route tools-on prefix without:

- generating assistant output
- sampling a token
- passing through a fake user request
- using a prompt workaround
- touching the normal `/chat` or `/chat/stream` runtime path

The goal is to prove the backend-native primitive needed before any future
route-prefix prewarm integration. This is not runtime prewarm.

## Result

Result: feasible as an isolated backend-native probe.

The native client can:

1. construct the same stable route prefix used by route tools-on calls;
2. tokenize it with the real native tokenizer;
3. decode only the stable prefix tokens;
4. capture a native KV checkpoint;
5. stop without sampling or generating content.

No runtime behavior is changed by this probe. There is no startup hook, no
`/tools on` hook, no `/chat` integration, and no user-visible behavior change.

## Code Inspected

Relevant files:

- `docs/FIRST_TURN_TOOLS_ON_PREFILL_ANALYSIS.md`
- `docs/NATIVE_ROUTE_PREFIX_PREWARM_DESIGN.md`
- `docs/KV_ROUTE_PREFIX_ANCHOR_RUNTIME_EXPERIMENT.md`
- `src/orbit/native_llama/client.py`
- `src/orbit/native_llama/prefix_anchor.py`
- `src/orbit/native_llama/prefix_anchor_probe.py`
- `src/orbit/native_llama/chat_template.py`
- `src/orbit/native_server/app.py`
- `src/orbit/native_server/protocol.py`
- `src/orbit/backend/llama_server.py`
- `src/orbit/runtime/chat.py`
- `src/orbit/runtime/messages.py`

Current runtime capture path:

- `ChatRuntime._run_tool_loop()` marks route calls using
  `model_call_context(phase="route", tools_mode="on")`.
- `LlamaServerBackend` sends `route_prefix_anchor=true` only for eligible native
  route/tools-on calls.
- `NativeLlamaClient.complete_chat_text()` renders the route prompt.
- `NativeLlamaClient._route_anchor_segments_for_prompt()` rebuilds
  `RoutePromptSegments`.
- `NativeLlamaClient._prepare_memory_with_route_anchor()` restores an existing
  checkpoint or decodes the stable prefix and captures a new checkpoint.
- `NativeLlamaClient._complete_prompt_standard()` then continues with the
  dynamic suffix and calls `_generate_from_current_context()`.

The prefill/decode primitive is `llama_decode`, reached through
`_decode_prompt_range()` in the runtime and through the isolated probe helper in
`prefix_anchor_probe.py`.

Generation starts in `NativeLlamaClient._generate_from_current_context()`, where
the sampler is reset, `llama_sampler_sample()` is called, and sampled tokens are
decoded. The prefill-only probe does not call this function and does not receive
a sampler.

## Probe

Implemented files:

- `src/orbit/native_llama/prefix_anchor_probe.py`
- `scripts/probe_native_route_prefix_prefill_only.py`
- `tests/test_prefix_anchor_probe.py`

The script is manual-only and is not imported or called by production runtime
paths. It constructs route segments from the route system prompt only. The
stable prefix boundary is the system route block; no user turn is needed to
obtain that prefix.

The probe emits only metadata:

- `prefix_hash`
- `prefix_token_count`
- `checkpoint_size_bytes`
- `prefill_ms`
- `decode_calls`
- `sampled_tokens`
- `generated_tokens`
- `sampler_touched`
- `session_history_touched`
- `probe_ok`

It does not print raw prefix text, prompt text, tool specs, token ids, user
content, tool output, file content, or web content.

## Manual Probe Result

Command:

```bash
PYTHONPATH=src python3 scripts/probe_native_route_prefix_prefill_only.py \
  --model models/ggml-org--gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf \
  --ctx 8192 \
  --threads 6 \
  --threads-batch 6 \
  --batch 256 \
  --ubatch 128
```

Metadata output:

```json
{
  "checkpoint_size_bytes": 238454176,
  "decode_calls": 3,
  "generated_tokens": 0,
  "prefill_ms": 50211.951,
  "prefix_hash": "16216b5e9e8f642e170bba51c2b94f13",
  "prefix_token_count": 693,
  "probe_ok": true,
  "reason": null,
  "sampled_tokens": 0,
  "sampler_touched": false,
  "seq_id": 0,
  "session_history_touched": false
}
```

Interpretation:

- prefill-only capture is possible in an isolated native client;
- the real stable route prefix is 693 tokens in this setup;
- checkpoint size matches the runtime route-anchor checkpoint size observed in
  prior smoke runs;
- decode was chunked into three calls to respect the configured batch size;
- no sampled tokens or generated tokens were produced.

## Safety Constraints

This probe intentionally does not:

- register a server endpoint;
- modify `/chat`, `/chat/stream`, `/tools on`, startup, reset, or route runtime;
- use a fake user request;
- pass through normal completion with a token budget trick;
- change prompts;
- change routing, tool selection, final policy, or evidence policy;
- touch session history;
- call the sampler.

The probe creates a standalone native client process. Any KV memory it fills is
local to that probe process and is released when the process exits.

## Equivalence

This PR does not add a new equivalence test.

Reason: `KV_PREFIX_ANCHOR_EQUIVALENCE_PROBE.md` and the existing
`probe_prefix_anchor_equivalence()` already cover the important behavioral
property:

- baseline `P + S`
- checkpoint `P`, restore, then append `S`
- compare next token and logits hash

The new probe answers a narrower question: can Orbit perform only the `P`
prefill and checkpoint capture without generation. Repeating the equivalence
probe here would duplicate existing coverage without changing the runtime
decision.

## Blockers For Runtime Prewarm

The backend-native primitive is feasible, but runtime prewarm still needs a
separate implementation PR.

Remaining blockers:

- there is no native-server prefill-only endpoint or lifecycle hook;
- current production capture is still wired into the route completion path;
- server lock and concurrent request behavior need explicit design;
- cancellation and client disconnect behavior need explicit tests;
- continuation readiness must be forced false after prewarm;
- failure must fall back to baseline without user-visible errors;
- prewarm must respect `ORBIT_KV_PREFIX_ANCHOR=off`;
- non-native backends must remain unchanged.

## Next PR

Recommended next step: implement a bounded internal native prefill-only hook or
endpoint.

The next PR should:

- expose an internal prefill-only operation, not a normal chat request;
- keep runtime prewarm disabled unless explicitly wired in a later PR;
- preserve one-checkpoint memory bounds;
- validate prefix hash and token count before restore;
- return metadata-only diagnostics;
- test failure, cancellation, non-native, `off`, and mismatch paths;
- smoke file/read/list/web/fetch and stale-evidence scenarios.

Runtime prewarm remains out of scope until that hook is reviewed and measured.

## Verdict

Verdict: docs plus isolated probe.

Prefill-only route prefix capture is feasible in a standalone native client.
The probe validates the backend-native primitive without changing runtime
behavior. Production prewarm should not be enabled until a separate PR adds a
reviewed lifecycle hook or endpoint with concurrency, cancellation, invalidation,
and fallback tests.
