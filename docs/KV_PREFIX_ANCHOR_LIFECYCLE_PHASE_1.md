# KV Prefix Anchor Lifecycle Phase 1

## Scope

This phase adds only the first safe lifecycle layer for native KV prefix anchors:

- feature flag parsing via `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1`
- stable prefix-anchor key computation
- explicit invalidation checks
- isolated checkpoint capture and restore helpers
- metadata-only diagnostics payloads

It does **not** wire prefix-anchor restore into any production runtime path yet.

## What Was Implemented

The new native helper module defines:

- `PrefixAnchorState`
- `compute_prefix_anchor_key(...)`
- `can_use_prefix_anchor(...)`
- `capture_prefix_anchor(...)`
- `restore_prefix_anchor(...)`
- `invalidate_prefix_anchor(...)`
- `anchor_metadata(...)`

The state records:

- `prefix_hash`
- `token_count`
- `model_id`
- `template_id`
- `tool_schema_hash`
- `capability_summary_hash`
- `runtime_policy_hash`
- `route_contract_hash`
- `backend_version`
- `native_version`
- `tools_mode`
- `checkpoint_size`
- `valid`
- `invalidation_reason`

The helpers are isolated and explicit. They do not modify any runtime flow unless a caller chooses to invoke them.

## Why This Stops Here

The remaining unsolved part is not binding availability anymore. The remaining problem is lifecycle safety:

1. define the exact stable tools-on prefix boundary
2. capture the checkpoint after the correct prefill point
3. restore only into a compatible context/sequence
4. guarantee that restore does not alter logits, repair behavior, or control-flow
5. fall back cleanly whenever any compatibility check fails

Until that sequence is wired and benchmarked, the helpers remain infrastructure only.

## Safety Properties

- flag default remains off
- no runtime path uses anchor restore automatically
- no prompt content changes
- no routing changes
- no tool selection changes
- no final-policy changes
- no replay-token workaround
- no semantic cache of tool descriptions

## Metadata Only

The helper metadata contains only technical fields:

- `anchor_enabled`
- `anchor_key_hash`
- `anchor_valid`
- `anchor_hit`
- `anchor_miss`
- `capture_attempted`
- `restore_attempted`
- `restore_used`
- `fallback_reason`
- `checkpoint_size`
- `token_count`

No raw prompt, user content, tool output, file content, or web content is included.

## Remaining Step Before Runtime Wiring

The next exact step is a bounded backend-native integration experiment that:

1. computes a stable tools-on prefix key from the already existing prompt components
2. captures a checkpoint after that stable prefix is fully prefetched
3. restores it only for a single low-risk path
4. proves with A/B diagnostics that evaluated tokens decrease without increasing:
   - model calls
   - retries
   - repairs
   - evidence regressions

If those guarantees cannot be proven, runtime wiring should remain disabled.
