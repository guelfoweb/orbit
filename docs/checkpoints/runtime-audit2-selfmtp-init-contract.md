# RUNTIME-AUDIT-2 — self-MTP initialization boolean contract

Baseline: `4f012ec44c75d37cc772da2239a17b11752d8846`
Model inference: NONE. No GGUF loaded, no benchmark run.

## Question

`_initialize_self_mtp_session` documented its return as *"True when it was
actually built"*, yet three fail-closed paths also returned `True`. Two
readings were possible:

* **A** — `True` = self-MTP was successfully constructed.
* **B** — `True` = self-MTP handling is terminal; the caller must not fall
  through to external-draft.

## Verdict: CONTRACT B. Classification: DOCSTRING_DEFECT.

No behavioural bug. No production behaviour changed.

## Return-site matrix

Derived by executing the shipped method against deterministic stubs.

| # | Branch | Returns | Runtime built | `mtp_enabled` | `mtp_failed` | reason |
|---|--------|---------|---------------|---------------|--------------|--------|
| 1 | capability resolution raised | `True` | **no** | False | True | `self-mtp-capability-error: …` |
| 2 | not eligible | `False` | no | False | False | `None` (untouched) |
| 3 | target context missing | `True` | **no** | False | True | `target-context-missing` |
| 4 | constructor raised (incl. missing self-MTP ABI) | `True` | **no** | False | True | exception text |
| 5 | constructed | `True` | yes | True | False | `None` |

Four of the five `True` sites build nothing. `True` cannot mean "built".

## Caller

Exactly one production caller, `client.py:918`:

```python
if self._initialize_self_mtp_session():
    return
```

It is a pure control-flow guard. The value is never stored, never returned,
never compared to a success notion; `_initialize_persistent_mtp_session`
itself returns `None`. Success is read from lifecycle state
(`mtp_enabled` / `mtp_failed`), never from this boolean. No CONTRACT_SPLIT.

The decisive behavioural row: a self-MTP **failure** with external-draft fully
eligible (`profile.mtp_supported=True`, `paths.mtp_available=True`, a draft
GGUF declared) does **not** attempt external-draft. Only reading B explains it.

## Mutation evidence (all via QREL-1 source-fresh execution)

| Mutant | Pre-existing suite | New module |
|--------|--------------------|------------|
| capability-error `True`→`False` | caught (3) | caught |
| target-context-missing `True`→`False` | caught (2) | caught |
| constructor-failure `True`→`False` | caught (1) | caught |
| ineligible `False`→`True` | caught (4) | caught |
| **success `True`→`False`** | **SURVIVED the whole suite (3864, OK)** | **caught (3)** |

The asymmetry is the whole finding. The suite pinned "terminal ⇒ True"
thoroughly and "built ⇒ True" not at all — because the latter was never the
contract. That surviving mutant is the coverage gap this module closes.

## Historical intent

Both the code and the docstring arrived together in `dfd6e50` (#268). The
fail-closed `return True` sites were present from the first commit, carrying an
explicit comment:

> A capability resolution failure is not an MTP verdict. Fail closed and say
> why rather than silently trying another path.

So the behaviour was deliberate from birth and the docstring was **stale on
arrival**, not drifted into. The same commit added
`FailureDoesNotFallThroughTests`, whose docstring names the return value as the
mechanism outright:

> A qualified artifact that fails must not silently try external-draft.
> Asserted without a draft model configured too, so **the guard is the return
> value** rather than the absence of a draft path.

That is the strongest intent evidence available: the author states, in the
introducing commit, that the boolean *is* the fall-through guard.

One nuance: at birth the fail-closed sites wrote `mtp_failed` /
`mtp_failure_reason` directly; the `record_failure(...)` indirection arrived
with the later lifecycle extraction. The return-value semantics are unchanged
by that refactor.

Why fail-closed is the right behaviour: external-draft is a different
architecture requiring a separate draft GGUF. A qualified single-GGUF artifact
that fails to build self-MTP has a malfunction worth reporting, not a reason to
silently start a different topology on a model chosen for this one.

## Change

* `src/orbit/native_llama/client.py` — docstring only. **Zero executable lines
  changed.**
* `tests/test_selfmtp_initialization_contract.py` — new, 10 tests.

## Files

```
SHA256 (post-change)
```
dbadfeffe08e32562936702b346491f597d38fd7b567cc89ba5fbeab8a016860  src/orbit/native_llama/client.py
52c7d38d7942a88fbe96fdc528ac911c102fa474a88bf505159828bf7a2919a6  tests/test_selfmtp_initialization_contract.py
```
