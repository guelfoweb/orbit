# KV Documentation Index

This index maps the KV/cache work after `v0.0.1-rc1`. It is a navigation aid
only. It does not replace the detailed reports and does not change any runtime
policy.

## Current Auto Path

- `KV_ROUTE_PREFIX_ANCHOR_RUNTIME_EXPERIMENT.md` documents the current
  native route-prefix anchor path.
- `ORBIT_KV_PREFIX_ANCHOR=auto` is the default when unset.
- `ORBIT_KV_PREFIX_ANCHOR=off` is the explicit kill switch.
- The legacy `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1` still enables auto mode when
  `ORBIT_KV_PREFIX_ANCHOR` is unset.
- `ORBIT_KV_PREFIX_ANCHOR=off` wins over the legacy experiment flag.
- Scope is native backend, tools-on route pass only.
- It must not be broadened to `chat_final`, `final_from_tool`, `tool_call`, or
  file/web/listing special paths without new benchmark evidence.

## Baseline And Analysis

- `KV_CACHE_REUSE_PLAN.md` defines the original phase plan and measurement
  questions.
- `KV_CACHE_PHASE_2_ANALYSIS.md` analyzes route pass count and multi-pass cost.
- `KV_PREFIX_REUSE_POST_ROUTE_BASELINE.md` records the post-route-fix baseline.
- `KV_LAYOUT_CACHE_ANALYSIS.md` compares logical prompt layout with backend
  cache behavior.
- `KV_BACKEND_CACHE_PATH_ANALYSIS.md` documents backend-visible cache path
  findings.

## Diagnostics

- `KV_PROMPT_LAYOUT_DIAGNOSTICS.md` describes prompt block layout metadata.
- `KV_BACKEND_ENVELOPE_DIAGNOSTICS.md` describes request-envelope metadata.
- `KV_BACKEND_NATIVE_CACHE_DIAGNOSTICS.md` describes native cache/LCP metadata.
- `ROUTE_OUTCOME_OBSERVABILITY.md` documents route outcome classification.

## Prefix Anchor Feasibility And Proofs

- `KV_PREFIX_ANCHOR_FEASIBILITY.md` explains why runtime-only prefix cache is a
  no-go.
- `KV_PREFIX_ANCHOR_IMPLEMENTATION_PLAN.md` records the native binding
  preparation plan.
- `KV_PREFIX_ANCHOR_LIFECYCLE_PHASE_1.md` documents isolated lifecycle
  scaffolding.
- `KV_PREFIX_ANCHOR_EQUIVALENCE_PROBE.md` documents checkpoint/restore
  equivalence in an isolated native probe.
- `KV_ROUTE_PREFIX_TOKEN_BOUNDARY.md` records the route prefix token-boundary
  validation used by the runtime experiment.

## Rejected Or Historical Branches

- `KV_PROMPT_SHAPE_EXPERIMENT.md` is a rejected prompt-shape experiment. It
  improved some short-chat cache metrics but introduced repair/retry risk.
- `KV_PREFIX_CACHE_FEASIBILITY.md` rejects fake runtime-only prefix cache.
- `KV_PREFIX_ANCHOR_RUNTIME_NO_GO.md` and
  `KV_ROUTE_PREFIX_ANCHOR_RUNTIME_NO_GO.md` are historical no-go reports. They
  should remain as context unless a future cleanup explicitly preserves their
  safety rationale elsewhere.

## Cleanup Review

- `KV_POST_MERGE_CLEANUP_REVIEW.md` lists safe cleanup candidates, risky
  cleanup candidates, and items that should not be touched before RC2.

## Release Guidance

For RC2 preparation, treat route prefix-anchor as a bounded auto feature with a
kill switch:

- keep `ORBIT_KV_PREFIX_ANCHOR=auto` as the default
- document `ORBIT_KV_PREFIX_ANCHOR=off` for disabling the feature
- keep historical no-go/reject reports available
- keep the isolated probe while the runtime path remains active
- require unit tests, compile checks, and native OFF/ON smoke before any release
