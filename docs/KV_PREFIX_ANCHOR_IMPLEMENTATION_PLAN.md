## KV Prefix Anchor Implementation Plan

Commit analyzed: `3fa6713`

### Goal

Prepare a real backend-native path for KV prefix-anchor or checkpoint reuse in the standard Orbit chat path without changing prompt semantics, routing, tool selection, final policy, or evidence behavior.

### What this patch does

This patch does **not** change runtime behavior.

It adds only the missing native bindings needed for a future standard-path experiment:

- `llama_memory_seq_cp`
- `llama_memory_seq_keep`
- `llama_state_get_size`
- `llama_state_get_data`
- `llama_state_set_data`
- `llama_state_seq_get_size`
- `llama_state_seq_get_data`
- `llama_state_seq_set_data`

These bindings are exposed so the next branch can prototype checkpoint or prefix-anchor restore using real native state, not token replay.

### Why no experiment is enabled yet

The standard Orbit path still lacks the higher-level integration required to use those primitives safely:

1. A stable checkpoint lifecycle for the tools-on prefix.
2. Compatibility rules for when a checkpoint is valid.
3. Restore logic that preserves the current prompt semantics exactly.
4. Invalidation when any stable component changes.
5. Benchmarks proving no regressions in:
   - model calls
   - repair or retry
   - file-read evidence
   - web or fetch two-call paths
   - listing behavior

Adding bindings alone is safe. Using them without that lifecycle is not.

### Smallest credible next experiment

The next branch, if pursued, should:

1. stay feature-flagged off by default, for example with `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1`
2. target only tools-on no-tool `chat_final`
3. capture a native checkpoint only after a verified stable prefix
4. restore that checkpoint only when:
   - model identity matches
   - template identity matches
   - runtime policy and route contract match
   - tool schema and capability summary match
   - backend-native checkpoint format matches
5. leave the final rendered prompt semantically unchanged

### Why runtime-only cache remains invalid

This plan does not revive runtime-only prefix caching.

That approach is still invalid because it would either:

- replay tokens instead of reusing KV,
- depend on semantic prompt reshaping,
- or create a fake cache keyed only by prefix hash.

### Safety boundary

Any future use of these bindings must remain:

- backend-native
- feature-flagged off by default
- invisible to the model
- free of prompt rewrites
- free of deterministic routing

### Validation required for the future experiment

OFF vs ON must be benchmarked on:

- repeated short tools-on chat
- repeated medium no-tool tools-on chat through `chat_final`
- listing
- file read
- valid web search
- valid `fetch_url`

Promotion requires:

- lower `evaluated_tokens`
- lower wall time
- no increase in model calls
- no increase in repair or retry
- no regressions in evidence behavior

### Current verdict

Verdict: `refine`

The missing primitive is no longer only conceptual. The necessary native state APIs are now bound locally, but the safe restore lifecycle still needs a dedicated experiment branch before any behavior change is acceptable.
