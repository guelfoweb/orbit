# REF-4: complete MTP lifecycle ownership

Fourth step of the incremental, behaviour-preserving runtime refactor. Closes
the REF-2/REF-3 seam.

## Baseline

`b7308a71d2ad1ddc70181477ef4b4f66fb028ea3`

## Audit: observed truth, not the estimate

The mission estimated ~6 remaining direct writes. The current source held
**12 writes across 7 sites**, all in the initialization path.

| site | branch | transition |
|---|---|---|
| capability probe raised | self-MTP | `record_failure(reason)` |
| target context missing | self-MTP | `record_failure("target-context-missing")` |
| constructor raised | self-MTP | `record_failure(str(exc))` |
| profile does not support MTP | external-draft | `record_unavailable(...)` |
| no draft model / not available | external-draft | `record_unavailable(...)` |
| target context missing | external-draft | `record_failure("target-context-missing")` |
| constructor raised | external-draft | `record_failure(str(exc))` |

**Direct lifecycle writes in all of `client.py`: 12 → 0.** With REF-3 having
cleared the completion path, `MtpSessionLifecycle` is now genuinely the only
writer.

## One method added: `record_unavailable`

Two sites set **only** `mtp_failure_reason`, deliberately leaving `mtp_failed`
False. Verified by execution that `record_failure` flips it True — so using it
would turn *"MTP does not apply here"* into *"MTP broke"*.

Honest scope of that claim: `mtp_failed` has **no production reader** today and
is absent from `NativeSessionSnapshot`. The distinction is preserved because the
pre-extraction code drew it, not because a consumer depends on it — a latent
invariant, not a live one. The docstring says so. Collapsing it during a
behaviour-preserving extraction would still be an unforced change, and the field
is public on the session dataclass.

Meets the bar: 2 duplicated sites, not expressible by existing methods without
semantic distortion, behaviourally distinguished by tests.

## Policy stayed in the client

`record_unavailable` is a single-field write with no branching. Every decision
remains in `client.py`: eligibility, whether MTP was requested, profile support,
ABI/artifact availability, context presence, and the self-MTP → external-draft
ordering. The collaborator still takes no client reference.

## Preserved asymmetries

* an **ineligible** artifact returns False and records *nothing* — that is what
  lets external-draft be attempted
* a **capability error** records a failure and returns True, blocking fallback
* the two **unavailability** sites record only a reason
* a self-MTP failure is still not equivalent to an external-draft failure

## Qualification

* **9 mutants caught**: unavailable→failure (×2), `record_unavailable` setting
  `mtp_failed`, capability-error→unavailable, ordering skipped, restored direct
  write in init, restored direct write in the completion path, handle assigned
  outside its property setter, ineligible path recording a reason.
* **U6 proven equivalent** rather than waved through: adding `disable=True` at
  an init site is unobservable, because `_initialize_self_mtp_session` has one
  caller, reached only after `clear_state()` sets `mtp_enabled = False`. The
  reviewer independently confirmed the reachability and noted it is
  reachability-dependent, so a hypothetical direct caller would diverge.
* equivalence verified across **2304 exhaustive precondition combinations** by
  the reviewer, comparing return value, all six session fields, the runtime
  handle and the full call-order trace: **0 divergences**
* focused 363/363 plus 36 lifecycle tests
* full suite: 3775 collected, **3767 passed, 8 skipped, 0 failed, real exit 0**
  (baseline 3762; +13 tests, zero regressions)
* init call counts across 12 branch combinations: no new hashing, model loads,
  context creation, native constructors or ABI checks
* `client.py` 4909 → 4912; lifecycle module 160 → 171

## Out of scope, deliberately

`mtp_fallback_reason` was not touched. Its two pre-existing surviving mutants
were confirmed neither fixed nor worsened — one (`H3`, dropping the
`"draft-mtp-unavailable"` default) sits one line from code this PR edited and
was still left alone.

## Review

One cycle: **BLOCKER 0 / MAJOR 0 / MINOR 2**, both fixed.

MINOR-1 corrected a claim in my own docstring: I had written that the
unsupported/broken distinction is "visible in `/props`", and it is not. The
justification now rests on baseline preservation, which is the true reason.

The reviewer also strengthened my argument for the AST-based direct-write guard:
I had defended it as necessary because a restored direct write is "invisible
behaviourally", and that is false — the behavioural tests catch it, along with
three evasion variants. The AST test is defence-in-depth, not the sole guard.
