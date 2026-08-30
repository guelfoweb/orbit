# REF-3: complete MTP lifecycle delegation

Third step of the incremental, behaviour-preserving runtime refactor. Finishes
the REF-2 seam rather than opening a new one.

## Baseline

`3d529498c78c01144eec269c1089d77957f21755`

## What was left

REF-2 made `MtpSessionLifecycle` the sole owner of the MTP runtime handle and
its four derived session fields, but the completion hot path
(`_try_complete_with_mtp_experimental`) still hand-wrote those transitions at
**7 sites** — the request path, and so the one most exposed to the drift the
extraction exists to prevent.

## Site map

| site | baseline lines | transition | now |
|---|---|---|---|
| 1 | 2409-2411 | failure + disable | `record_failure(reason, disable=True)` |
| 2 | 2415-2417 | attach handle, verdict untouched | `attach(runtime)` |
| 3 | 2467-2469 | failure + disable (post-cancel) | `record_failure(reason, disable=True)` |
| 4 | 2471-2476 | publish + clear reason | `publish(runtime, clear_failure_reason=True)` |
| 5 | 2479-2481 | failure + disable | `record_failure(reason, disable=True)` |
| 6-7 | 2484-2486 | mark healthy, no handle | `mark_ready()` |

Direct writes to lifecycle-owned fields in the completion path: **32 → 0**. The
6 init-path writes (client.py ~870-935) are out of REF-3's scope and byte-
identical to baseline.

## Two methods added, and why

The mission bar is: one coherent transition, ≥2 duplicated sites, behaviourally
pinnable. Both qualify, and neither is expressible with existing methods:

* **`attach(runtime)`** records the handle and `ctx_dft`/`spec` but deliberately
  leaves `mtp_enabled`/`mtp_failed` alone. Verified by execution before adding
  it: from `(enabled=False, failed=True)`, `attach` preserves that, while
  `publish` flips to `(True, False)`. Using `publish` at site 2 would declare a
  failed session ready **before the completion has run**.
* **`mark_ready()`** sets the verdict with no handle changing hands. Expressing
  it as `publish(self._persistent_mtp_runtime, ...)` errors when nothing is
  attached — a genuinely different operation.

## Qualification

**9 mutants caught**: `attach`→`publish`, `mark_ready`→no-op, dropped
`disable=True`, dropped `clear_failure_reason=True`, `attach` not storing the
handle, `attach` setting enabled, `mark_ready` losing the reason, a restored
direct `_session.mtp_*` write, and a failed rebuild calling `free` instead of
recording failure (a double free).

* focused: 386/386 plus 23 lifecycle tests
* full suite: 3762 collected, **3754 passed, 8 skipped, 0 failed, real exit 0**
  (baseline 3752; +10 tests, zero regressions)
* `client.py` 4913 → **4909**; lifecycle module 135 → 160
* hot path: delegation only — no hashing, tokenization, native calls, resets,
  locks, context copies or per-token work
* `CommittedIdentity` zero diff; five guards unchanged; `use_mtp_experimental`
  still `False`; no `.cpp`/`.h` in the diff; self-MTP and external-draft still
  distinct

## The test-quality correction

The first version pinned the call sites with **AST inspection** — asserting
which method appeared in the source. That is the source-shape pattern three
earlier review cycles in this repo rejected, and it was wrong here for the same
reason: it proves shape, not behaviour.

Mid-review I replaced it. The failure block (14 lines, 7 stubbable
collaborators) and the two success statements are now extracted and **executed
verbatim** against a real lifecycle, asserting resulting session state. The
review confirmed the consequence: `mark_ready`→no-op and dropped `disable=True`
are caught **exclusively** by the executed tests. The suite would still catch
them with every AST test deleted.

Two AST tests remain as secondary pins, annotated to say so — cheap to keep,
useful for localising a failure, never the only thing between a mutation and a
green suite.

## Review

One cycle: **BLOCKER 0 / MAJOR 0 / MINOR 2**.

Equivalence was verified exhaustively rather than by inspection: 6 distinct
transitions × the full 64-precondition space of the five session fields plus the
prior handle → **0 divergences**.

Both minors addressed or recorded: the redundant AST tests are annotated, and
two surviving mutants in `mtp_fallback_reason` plumbing were confirmed
**pre-existing at baseline** — a client-owned field REF-3 deliberately does not
own. Recorded as a follow-up, not smuggled into this PR.
