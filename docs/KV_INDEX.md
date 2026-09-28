# KV Documentation Index

This is a navigation aid, reconciled against
`ab6109743f47d391070a1ee374c5e17325db24be` on 2026-09-28. Current controls and
historical proofs are separate below; an old experiment or release checklist
does not authorize a new cache change.

## Current entry points

- [prefix_anchor.py](../src/orbit/native_llama/prefix_anchor.py) owns the
  general route-anchor mode controls listed below. The original evidence is in
  [KV_ROUTE_PREFIX_ANCHOR_RUNTIME_EXPERIMENT.md](KV_ROUTE_PREFIX_ANCHOR_RUNTIME_EXPERIMENT.md); it is not a description of all
  current model/phase-specific cache paths.
- `ORBIT_KV_PREFIX_ANCHOR=auto` is the default when unset.
- `ORBIT_KV_PREFIX_ANCHOR=off` is the explicit kill switch.
- The legacy `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1` still enables auto mode when
  `ORBIT_KV_PREFIX_ANCHOR` is unset.
- `ORBIT_KV_PREFIX_ANCHOR=off` wins over the legacy experiment flag.
- The original anchor's scope is native backend, tools-on route pass only,
  subject to its identity/eligibility checks.
- It must not be broadened to `chat_final`, `final_from_tool`, `tool_call`, or
  file/web/listing special paths without new benchmark evidence.

Other qualified paths have their own controls and identities:

| Path | Source of truth / retained explanation |
|---|---|
| Qwen, Qwen3-Coder and Ornith route prefixes | [qwen_route_prefix.py](../src/orbit/native_llama/qwen_route_prefix.py), [qwen3_coder_route_prefix.py](../src/orbit/native_llama/qwen3_coder_route_prefix.py), [ornith_route_prefix.py](../src/orbit/native_llama/ornith_route_prefix.py). Do not infer activation for these from the general anchor flag alone. |
| Startup prewarm | `ORBIT_KV_PREFIX_PREWARM` in [native_server/app.py](../src/orbit/native_server/app.py); [recorded design](NATIVE_ROUTE_PREFIX_STARTUP_PREWARM.md). Startup work must be counted separately from a served turn. |
| Eligible final-from-tool reuse | `ORBIT_FINAL_PREFIX_REUSE` in [final_prefix_config.py](../src/orbit/final_prefix_config.py); qualified aligned 64-token history in [AGENTS.md](../AGENTS.md#kv--cache--final-budget). This is separate from route reuse. |

Inspect `/props` for the running process's observed state and `/status` for
client/server build identity. Current source and qualified-profile evidence take
precedence over old line numbers or an RC2-era recommendation. No cache policy
or new optimization is introduced by this index.

## Historical baseline and analysis

- [KV_CACHE_REUSE_PLAN.md](KV_CACHE_REUSE_PLAN.md) defines the original phase plan and measurement
  questions.
- [KV_CACHE_PHASE_2_ANALYSIS.md](KV_CACHE_PHASE_2_ANALYSIS.md) analyzes route pass count and multi-pass cost.
- [KV_PREFIX_REUSE_POST_ROUTE_BASELINE.md](KV_PREFIX_REUSE_POST_ROUTE_BASELINE.md) records the post-route-fix baseline.

## Diagnostics

- [KV_PROMPT_LAYOUT_DIAGNOSTICS.md](KV_PROMPT_LAYOUT_DIAGNOSTICS.md) describes prompt block layout metadata.
- [KV_BACKEND_ENVELOPE_DIAGNOSTICS.md](KV_BACKEND_ENVELOPE_DIAGNOSTICS.md) describes request-envelope metadata.
- [KV_BACKEND_NATIVE_CACHE_DIAGNOSTICS.md](KV_BACKEND_NATIVE_CACHE_DIAGNOSTICS.md) describes native cache/LCP metadata.
- [ROUTE_OUTCOME_OBSERVABILITY.md](ROUTE_OUTCOME_OBSERVABILITY.md) documents route outcome classification.

## Prefix Anchor Feasibility And Proofs

- [KV_PREFIX_ANCHOR_FEASIBILITY.md](KV_PREFIX_ANCHOR_FEASIBILITY.md) explains why runtime-only prefix cache is a
  no-go.
- [KV_PREFIX_ANCHOR_IMPLEMENTATION_PLAN.md](KV_PREFIX_ANCHOR_IMPLEMENTATION_PLAN.md) records the native binding
  preparation plan.
- [KV_PREFIX_ANCHOR_LIFECYCLE_PHASE_1.md](KV_PREFIX_ANCHOR_LIFECYCLE_PHASE_1.md) documents isolated lifecycle
  scaffolding.
- [KV_PREFIX_ANCHOR_EQUIVALENCE_PROBE.md](KV_PREFIX_ANCHOR_EQUIVALENCE_PROBE.md) documents checkpoint/restore
  equivalence in an isolated native probe.
- [KV_ROUTE_PREFIX_TOKEN_BOUNDARY.md](KV_ROUTE_PREFIX_TOKEN_BOUNDARY.md) records the route prefix token-boundary
  validation used by the runtime experiment.

## Rejected Or Historical Branches

- [KV_PROMPT_SHAPE_EXPERIMENT.md](KV_PROMPT_SHAPE_EXPERIMENT.md) is a rejected prompt-shape experiment. It
  improved some short-chat cache metrics but introduced repair/retry risk.
- [KV_PREFIX_CACHE_FEASIBILITY.md](KV_PREFIX_CACHE_FEASIBILITY.md) rejects fake runtime-only prefix cache.
- [KV_PREFIX_ANCHOR_RUNTIME_NO_GO.md](KV_PREFIX_ANCHOR_RUNTIME_NO_GO.md) and
  [KV_ROUTE_PREFIX_ANCHOR_RUNTIME_NO_GO.md](KV_ROUTE_PREFIX_ANCHOR_RUNTIME_NO_GO.md) are historical no-go reports. They
  should remain as context unless a future cleanup explicitly preserves their
  safety rationale elsewhere.

## Historical cleanup review

- [KV_POST_MERGE_CLEANUP_REVIEW.md](KV_POST_MERGE_CLEANUP_REVIEW.md) records the pre-RC2 cleanup assessment.
  Its proposals are not a current deletion list; recheck source and dependencies
  before treating any entry as unfinished work.

## Historical RC2 release guidance

The following checklist was written for RC2 preparation and is preserved as
history, not the current release plan. It treated route prefix-anchor as a
bounded auto feature with a kill switch:

- keep `ORBIT_KV_PREFIX_ANCHOR=auto` as the default
- document `ORBIT_KV_PREFIX_ANCHOR=off` for disabling the feature
- keep historical no-go/reject reports available
- keep the isolated probe while the runtime path remains active
- require unit tests, compile checks, and native OFF/ON smoke before any release
