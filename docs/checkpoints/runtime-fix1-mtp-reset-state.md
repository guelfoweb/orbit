# RUNTIME-FIX-1: stale MTP status across session reset

A correctness fix, not a refactor. Structural refactoring is complete.

## Baseline

`a6e420efd9bc7b9bf18e2249c9fda8301c0d2cfb`

## Two independent root causes

`reset_session_state` destroys the conversation: KV memory, cached prompt
tokens, the committed sequence, every prefix and rolling anchor, and — through
`publish(runtime, clear_failure_reason=True)` — the session-level
`mtp_failure_reason`. Two **client-level** fields describing the *last request's*
MTP outcome were left standing, and both reach `/props`.

They mislead in **opposite directions**, which is why each needed its own case:

| field | `/props` | stale effect |
|---|---|---|
| `mtp_fallback_reason` | `app.py:448` | `orbit_smoke_harness.mtp_state_from_props` reads it as a fallback for `mtp_failure_reason`; a leftover reason turns a healthy reset session from `ready` into **`failed`** — under-reporting |
| `last_mtp_completion` | `app.py:442`, `_mtp_last_completion_payload` at `:1428` | a leftover success reports **`on`** for a session that has completed nothing, and republishes ~20 metric fields (acceptance ratios, token counts, timings) as if they belonged to the new session — over-reporting |

The mission premise held for the first field only. The second was verified to be
an over-report, not an under-report: its neutral `success` is already `False`,
so clearing it can never *restore* usability. Both are still defects; the second
is a data-attribution one.

## Reset contract

Derived from the current code, not from field names: everything describing the
destroyed conversation is cleared, while configuration and the persistent
runtime survive. `clear_failure_reason=True` at the end is direct evidence the
author intended a clean status slate here; the two client-level duplicates were
simply outside the lifecycle owner's reach.

## The fix

Twelve lines in `reset_session_state`, additions only:

```python
self.mtp_fallback_reason = None
self.last_mtp_completion = MtpCompletionResult(
    enabled=self.config.use_mtp_experimental, success=False, error=None
)
```

Neutral values derived from `__init__`, not guessed. `enabled` tracks config
rather than being hardcoded `False`, because `_mtp_last_completion_payload`
returns `None` when it is false — hardcoding would hide the payload for a
client that has MTP configured.

### Placement is load-bearing

After `_invalidate_committed_sequence()`, before every early return.

An earlier placement — right after `reset_cancel()` — was tried and **reverted**.
It broke `test_real_reset_session_state_invalidates_committed_sequence`, which
drives the real reset on a duck-typed client with no `self.config` inside
`try/except Exception: pass`; reading `self.config` above the invalidation makes
the `AttributeError` fire first, so the invalidation never runs. The new tests
alone cannot distinguish the two positions — the pre-existing test is what pins
it. The reviewer independently reproduced this.

Exits audited by AST: one `raise` (the `not ctx_tgt` guard, which precedes any
mutation — nothing has been reset, so nothing is stale) and two `return`s (the
absent-runtime and discard paths), both **after** the clears.

## Qualification

* **CASE A / B / C all fail on baseline**, pass with the fix. 6 of 8 original
  tests fail when the fix is reverted; the 2 that pass are deliberate *guard*
  tests whose job is to fail on over-reach.
* **Each clear independently necessary**: removing the fallback clear alone → 6
  failures; removing the completion clear alone → 5. Neither is masked by the
  other.
* **Mutants caught**: both clears removed; both removed together; fallback to a
  wrong sentinel; completion `enabled=False`; completion `success=True`; clears
  moved after the early returns; over-reach onto `mtp_failure_reason`,
  `mtp_failed`, and `_persistent_mtp_runtime`.
* Every mutation executed through QREL-1 (`scripts/qualify_fresh.py`), with
  `client.py` restored byte-identically after each.
* focused: 120 across the reset/lifecycle/persistent-MTP/strict-append suites,
  plus 180 across every stub-client caller of `reset_session_state`
* full suite: 3864 collected, **3856 passed, 8 skipped, 0 failed, real exit 0**
* `compileall` clean, `git diff --check` clean
* no model inference, no GGUF load

## Unchanged, deliberately

`/props` construction, the harness usability calculation, MTP admission and
fallback policy, session-level `mtp_enabled` / `mtp_failed` /
`mtp_failure_reason`, the persistent runtime, CommittedIdentity, and the
prefix/rolling anchor stores. The self-MTP builder's `return True` contract and
all write-only telemetry were explicitly out of scope.

## Review

One adversarial cycle: **BLOCKER 0 / MAJOR 0 / MINOR 4**.

* **M-1 — corrected.** I reported "10 mutants caught". Two over-reach mutants
  (`mtp_failure_reason`, `mtp_enabled`) in fact survive *this* suite: the clears
  run before `publish`, which repairs them on the happy path. They are caught,
  but by the pre-existing `test_the_client_never_assigns_lifecycle_state_directly`
  — my attribution was wrong, not the coverage. A discard-path guard test was
  added to pin the contract behaviourally.
* **M-2 — addressed in the comment.** `enabled` carries two meanings: config on
  reset, "available for this request" at `client.py:2361`. Verified no `/props`
  drift (the payload is `None` before and after), and the comment now says why.
* **M-3 — accepted.** A raise before the clears leaves the fields stale. Strictly
  no worse than baseline, which left them stale on *every* path, and fixing it
  requires the placement the pre-existing test forbids.
* **M-4** — untracked `workdir/` artifacts, unrelated to this change.

## SHA256SUMS

```
cbde5dbd5c0096f8a4ac7a88451c16cb6e54032cd18c48d93c3f1f9e67df1869  src/orbit/native_llama/client.py
faef07b6a623a12dce66c083259fe26ab22649ab748a9114aa619571ad256fe9  tests/test_mtp_reset_state.py
```
