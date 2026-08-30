# REF-2: MTP session lifecycle extraction

Second step of the incremental, behaviour-preserving runtime refactor.

## Baseline

`e9acbb28be5c22f9f36709043ca4a2dc2b850766`

## Seam audit

An MTP session is one `PersistentMtpSessionRuntime` plus four session fields
that project it: `ctx_dft`, `spec`, `mtp_enabled`, and the pair
`mtp_failed` / `mtp_failure_reason`. Those fields are not independent — they
describe whether a runtime exists and how its last construction went.

The audit found the cluster cohesive but **duplicated**: the same publish
epilogue at six sites and the same failure pattern at five, each with small
variations. Hand-written repetition is how the projection drifts away from the
handle.

| symbol | accesses in `client.py` before |
|---|---|
| `_persistent_mtp_runtime` | 19 |
| `_session.mtp_enabled` | 19 |
| `_session.mtp_failure_reason` | 16 |
| `_session.mtp_failed` | 15 |
| `_session.ctx_dft` / `spec` | 8 each |

## What moved

`MtpSessionLifecycle` (135 lines) owns the handle and the transitions:
`publish`, `record_failure`, `discard`, `free`, `clear_state`.

It owns **no policy**. Path selection, artifact qualification, whether MTP was
requested, and the SOFT/HARD reset decision all stay in the client. The
collaborator is told what happened and records it consistently.

Deliberately preserved rather than tidied:

* **self-MTP and external-draft stay distinct** — separate entry points, no
  branch on model names, no shared "construct MTP" abstraction.
* **`clear_failure_reason` asymmetry** — initialization and the in-completion
  republish leave a prior reason in place; `reset_session_state` and the
  post-cancel rebuild clear it. Collapsing these would change what `/props`
  reports after a recovered session.
* **`discard` never frees.** Baseline's failed-reset path (client.py:991-996 at
  `e9acbb28`) drops the runtime with **zero** `free_persistent_mtp_session`
  calls, because the native session has already torn itself down. Freeing there
  is a double free. Verified against baseline source, not assumed.

## Ownership

`client._persistent_mtp_runtime` is a property over the collaborator, so the 19
existing call sites keep working with one authoritative copy. The collaborator
receives a **session getter**, not the session: clients built via
`object.__new__` may not have `_session` yet, and writes must land on whichever
session is eventually assigned — as they did when these were plain attribute
writes.

Native ownership is unchanged: `free_persistent_mtp_session` is called with the
same three kwargs in the same order, `close()` is byte-identical to baseline,
`persistent_mtp.py` is untouched, and repeated close stays safe.

## Dependencies

* new module imports only `typing`; it never imports the client — no cycle
* `client.py` 4900 → **4913** lines. It GREW: the property and on-demand owner
  cost more than the inlined blocks saved. The win is single ownership and
  de-duplicated transitions, not line count.
* `CommittedIdentity` (REF-1) untouched — zero diff
* KV cluster, prompt cache, route/tool anchors and `ctx_tgt` ownership: not moved

## Qualification

* **10 mutants caught.** Four initially survived — publish not storing the
  runtime, `free` not clearing `mtp_enabled`, `discard` freeing, `clear_state`
  losing the reason — and characterization tests were added for each.
* After review, one further mutant with real safety consequence was closed: the
  client calling `free` instead of `discard` on the failed-reset path, i.e. a
  double free. The collaborator's `discard` was pinned, but nothing pinned that
  the *client* reaches for it. Now driven by executing the shipped block itself.
* focused: 378/378 plus 13 new lifecycle tests
* full suite: 3752 collected, **3744 passed, 8 skipped, 0 failed, real exit 0**
  (baseline 3739; +13 new tests, zero regressions)
* hot path: no new hashing, tokenization, native calls, locks, context-sized
  copies or per-token work; the collaborator imports only `typing`

## Review

One cycle: **BLOCKER 0 / MAJOR 0 / MINOR 3**. All three addressed or recorded:

* the extraction is partial — three publish sites and four failure sites in the
  `complete()` path remain hand-written. Verified behaviourally identical, and
  left for a later step rather than widening REF-2's blast radius.
* the double-free mutant (fixed above)
* one source-shape test (the no-client-import guard), kept as an architectural
  guard alongside 12 behavioural tests

## Process note

REF-1's lesson was applied: **full suite before review, not after**. It caught
nothing new here, but it is the ordering that would have caught REF-1's
duck-typed regressions before a reviewer saw them.

The REF-1 regression class did recur during development — the property setter
silently discarded assignments on `__new__`-built clients, breaking 7 tests —
and was fixed by creating the owner on demand.
