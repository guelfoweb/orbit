# REF-5: rolling route-anchor state extraction

Fifth step of the incremental, behaviour-preserving runtime refactor.

## Baseline

`fa805af3bc788fdf6a4b55316b4952c0963dd452`

## Seam audit — observed truth, not the estimate

The prior ranking estimated ~4 methods. The family actually holds **9**:

| method | ctx_tgt | lib | verdict |
|---|---|---|---|
| `_rolling_anchor_slot` | 0 | 0 | **extracted** |
| `_rolling_anchor_state_for` | 0 | 0 | **extracted** |
| `_store_rolling_anchor_state` | 0 | 0 | **extracted** |
| `_invalidate_rolling_route_anchor` | 0 | 0 | **extracted** |
| `_rolling_route_identity` | 0 | 0 | stays — needs 5+ client-wide facts |
| `_ornith_rolling_route_eligible` | 0 | 0 | stays — policy |
| `_ornith_rolling_analysis_eligible` | 0 | 0 | stays — policy |
| `_rolling_outranks_route_prefix` | 0 | 0 | stays — policy |
| `_prepare_memory_with_ornith_rolling_route_anchor` | **1** | **1** | stays — orchestration |

## Extraction gate: GO

The mission required STOP if extraction needed broad `ctx_tgt`/`lib`/KV access.
It does not. The single native dependency sits entirely in
`_prepare_memory_with_ornith_rolling_route_anchor`, which calls the native
restore, `_clear_target_memory`, `_invalidate_committed_sequence` and
`_prepare_memory_for_prompt`. Moving it inward would have required passing the
client or the lib into the collaborator — precisely what §2 forbids. It stays
byte-identical, and `RollingAnchorStore` has **zero** native access (verified by
AST: imports are `__future__` and the pure state module, nothing else).

## Ownership

Two checkpoint slots — CHAT route and ANALYSIS — with one owner. The client's
two fields became properties over the store, with an on-demand
`_rolling_anchor_store()` for `object.__new__` clients. Lineage separation
follows the identity's own `strategy_id`, so the backend still learns nothing
about CHAT vs ANALYSIS.

That separation is a **safety** property, not tidiness: restoring an analysis
checkpoint for a route prompt puts a different conversation's KV behind the
model, which then answers confidently from someone else's context. Both leak
directions are pinned by mutation.

## Deleter

A pre-existing test does `del client._rolling_analysis_anchor_state` to simulate
an absent slot. The property needed a deleter; it empties the slot rather than
removing the accessor. The contract that matters — a missing checkpoint falls
cold rather than raising — is identical. `hasattr` and double-`del` differ from
baseline, and the reviewer confirmed by exhaustive grep that nothing in `src/`
or `tests/` observes either.

## Qualification

* **10 mutants caught**: store no-op, invalidate no-op, analysis not
  invalidated, slot always route, slot always analysis, `state_for` reading the
  wrong slot, client property bypassing the store, deleter not emptying, and —
  added after review — the already-empty guard dropped on the analysis branch.
* Three surviving mutants analysed: two provably equivalent (an `or`→`and` that
  is unreachable after the first invalidation, and a dead `or
  RollingRouteAnchorState()` carried over from baseline), one redundant
  (`__init__` store creation, fully covered by the lazy path).
* Differential equivalence, verified by the reviewer: **16,661 checks over
  5,024 operation sequences, 0 divergences**.
* focused 221/221 plus 21 store tests
* full suite: 3796 collected, **3788 passed, 8 skipped, 0 failed, real exit 0**
  (baseline 3775; +21 tests, zero regressions)
* native calls unchanged: `restore_rolling_route_anchor` called exactly once
* `client.py` 4912 → 4933; store 103 lines

## Incident: an overwritten module, recovered

While creating the collaborator I wrote it to `rolling_route_anchor.py` without
checking the name — silently overwriting a **pre-existing tracked 216-line
module**. An ImportError surfaced it within seconds; it was restored with
`git checkout HEAD --` and the collaborator placed in `rolling_anchor_store.py`.

Verified byte-identical to baseline (`b45182c3…` both sides, zero diff), and
independently re-verified by the reviewer, who was told about the incident
rather than left to find it. Nothing was lost. The lesson is to check whether a
filename is taken *before* writing it.

## Review

One cycle: **BLOCKER 0 / MAJOR 0 / MINOR 3**. Two fixed, one recorded:

* **MINOR-1** — a text guard in `test_ornith_route_prefix.py` sliced 600 chars
  from `_rolling_anchor_state_for` to assert the rolling read never touches the
  prewarm slot. After extraction that window covered only a one-line delegate,
  so it pinned nothing. Re-pointed at the store, and confirmed it now catches a
  real injected prewarm-slot leak.
* **MINOR-3 / H2** — the already-empty guard was pinned on the route slot only;
  dropping it on the analysis branch survived. Two tests added, mutant caught.
* **MINOR-2** — the deleter's `hasattr`/double-`del` divergence, unreachable;
  recorded rather than fixed.

The reviewer also caught a weakness in my own test: the "store makes no native
calls" check was a raw-text search that failed on the module's own docstring,
which names `ctx_tgt` to say it does not use it. Rewritten over parsed AST.
