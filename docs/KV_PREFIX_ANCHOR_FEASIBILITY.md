## KV Prefix Anchor Feasibility

Commit analyzed: `3fa6713`

### Goal

Find a safe backend-native way to reuse KV for the stable tools-on prefix without changing prompt semantics, routing, tool selection, final policy, or evidence behavior.

### What is already true

- `cache_prompt=true` reaches the native backend.
- `slot_id` is stable in the observed repeat scenarios.
- `cached_tokens` tracks the backend-visible tokenized longest common prefix.
- KV reuse works when the tokenized prefix is already shared.
- The rejected prompt-shape experiment showed that changing prompt semantics is not a stable solution.

### Current standard native path

The standard chat path uses one active `llama_context` and one cached prompt lineage:

- one `NativeSessionState`
- one `ctx_tgt`
- one `cached_prompt_tokens`
- one `prompt_cache_mode`

Prompt reuse is implemented by:

1. tokenizing the full prompt
2. computing the longest common prefix against `cached_prompt_tokens`
3. mutating the same active memory with `llama_memory_clear()` or `llama_memory_seq_rm()`
4. decoding the remaining suffix

This is real KV reuse, but only for the single currently resident prompt lineage.

### What the vendor library exposes

The bundled `llama.cpp` sources do expose lower-level state primitives that are not currently used by the standard Orbit path:

- `llama_memory_seq_cp`
- `llama_memory_seq_keep`
- `llama_memory_seq_add`
- `llama_state_get_size`
- `llama_state_get_data`
- `llama_state_set_data`
- `llama_state_seq_get_size`
- `llama_state_seq_get_data`
- `llama_state_seq_set_data`

So a theoretical backend-native prefix-anchor implementation is possible in principle.

### Why there is still no safe small patch today

Those primitives are not enough, by themselves, to justify an immediate experiment in Orbit core.

The missing safe integration layer is:

1. **No standard-path bindings or wrapper contract**
   - Orbit binds only `llama_memory_clear()` and `llama_memory_seq_rm()` in the standard path.
   - There is no existing wrapper for standard-chat KV checkpoints or restore.

2. **No sequence-aware standard-chat flow**
   - The normal path uses `llama_batch_get_one()` and one active sequence lineage.
   - Orbit does not currently manage multiple live sequence identities or checkpoint namespaces in the standard path.

3. **No stable prefix checkpoint lifecycle**
   - There is no current mechanism for:
     - capturing a checkpoint exactly after the stable tools-on prefix
     - validating that the prefix tokens still match
     - restoring the checkpoint before decoding a dynamic suffix
     - invalidating checkpoints when tool schema, capability summary, or contract changes

4. **No proof yet that restore is behavior-preserving in this path**
   - A safe patch must preserve:
     - prompt semantics
     - model call count
     - route and chat control flow
     - file/web/fetch evidence behavior
   - None of that is implemented or benchmarked for standard chat restore yet.

### Why the obvious shortcuts are still invalid

These remain out of scope:

- replaying stable prefix tokens and calling it a cache
- changing prompt wording to increase LCP
- runtime-side semantic cache keyed by prefix hash
- forcing separate tool or chat routing
- fake cache lanes without real backend restore

### Technical verdict

Verdict: `no-go` for an immediate small core patch.

There is a plausible **future** backend-native path, but not a small safe patch ready to land now.

### Smallest credible next implementation

If this line continues, the next real experiment should be backend-native and feature-flagged off by default.

Minimum viable direction:

1. add standard-path bindings for sequence or full-context state save/restore
2. implement checkpoint capture only after a verified stable tools-on prefix
3. restore that checkpoint only for compatible tools-on no-tool chat phases
4. keep prompt text semantically identical
5. keep all visible behavior identical with the flag off

Prefer a dedicated native helper layer over ad-hoc runtime logic.

### Why this should not be implemented yet in this turn

This would be more than a small local patch:

- new bindings
- new native-client state lifecycle
- compatibility rules for checkpoint reuse
- invalidation rules
- A/B benchmarking across route, chat_final, file, listing, web, and fetch paths

That exceeds the threshold for a safe immediate patch under the current constraints.

### Recommended next lever

Proceed only with a separate backend-native experimental branch that does all of the following:

- introduces checkpoint/restore behind a dedicated flag such as `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1`
- limits scope initially to tools-on no-tool `chat_final`
- proves no regression in:
  - model calls
  - repair/retry
  - file-read content evidence
  - web/fetch two-call path
  - listing behavior
- compares OFF vs ON using the existing KV diagnostics

### Acceptance bar for any future experiment

Promote only if ON:

- reduces `evaluated_tokens`
- reduces wall time
- does not increase model calls
- does not increase retry or repair
- preserves file/web/fetch/listing correctness
- preserves model-guided behavior

Otherwise reject it and keep the diagnostics/documentation only.
