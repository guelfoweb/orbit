# REF-6: final-prefix anchor state extraction

Sixth step of the incremental, behaviour-preserving runtime refactor.

## Baseline

`ce3444f49c43172ffccf29cff3f89d7e3bc2595c`

## Path safety (§0A)

REF-5 briefly overwrote a pre-existing tracked module before restoring it
byte-identically, so REF-6 required the candidate path to be checked as taken
or free *before* any write. Both new files were verified absent from the index
and the working tree first:

* `src/orbit/native_llama/final_prefix_store.py` — free
* `tests/test_final_prefix_store.py` — free

No tracked file was overwritten. `client.py` was modified in place by edit, never
rewritten wholesale.

## Seam audit — observed truth, not the estimate

| method / field | ctx_tgt | lib | verdict |
|---|---|---|---|
| `_final_prefix_anchor_state` (field) | 0 | 0 | **extracted** |
| `_final_prefix_status` (field) | 0 | 0 | **extracted** |
| `_record_final_prefix_fallback` | 0 | 0 | **extracted** |
| `_invalidate_final_prefix` | 0 | 0 | **extracted** |
| `_final_prefix_plan` | 0 | 0 | stays — policy/planning |
| `final_prefix_experiment_status` | 0 | 0 | stays — reporting projection |
| `_prepare_memory_with_final_prefix` | **1** | **1** | stays — orchestration |

## Extraction gate: GO

The mission required STOP (`REF-6 FINAL_PREFIX SEAM TOO COUPLED`) if extraction
needed broad `ctx_tgt`/`lib`/KV access. It does not. Every native dependency —
`restore_prefix_anchor`, `capture_prefix_anchor`, `_clear_target_memory`,
`_decode_prompt_range`, `self.lib.lib`, `self._session.ctx_tgt` — stays inside
`_prepare_memory_with_final_prefix`. The client performs capture and restore and
hands the *result* to the store, so physical KV ownership never moves.
`FinalPrefixStore` has **zero** native access, verified by AST rather than text
search.

## Ownership

The store owns the `PrefixAnchorState` checkpoint and the
`FinalPrefixExperimentStatus` counters together. The transitions that were
written out by hand at eight sites became five named ones: `mark_ready`,
`mark_not_ready`, `record_fallback`, `mark_unused`, `invalidate`. The client's
two fields became properties over the store, with an on-demand
`_final_prefix_store()` for `object.__new__` clients.

`captured` and `restored` are separate counters on purpose: a restore inflating
the capture total would misreport the experiment as prefilling work it never did.

## Duplicate status class — found and removed

`FinalPrefixExperimentStatus` was initially defined in **both** `client.py` and
the new store, so `isinstance(client._final_prefix_status, client.FinalPrefixExperimentStatus)`
became `False` where baseline gave `True`. The existing test missed it because it
*assigns* the status rather than comparing its type. The duplicate was deleted,
the store's class re-exported from `client.py`, a test added, and the remaining
four collaborator modules swept for the same defect (none found).

## Qualification

* **13 mutants caught**, including the two added after review: a `mark_unused`
  that also clears `failure_reason`, and the restore site delegating as
  `captured=True`.
* focused 24/24 (`test_prefix_anchor_probe`) plus 22 store tests
* full suite: 3818 collected, **3810 passed, 8 skipped, 0 failed, real exit 0**
  (baseline 3796; +22 tests, zero regressions)
* native calls unchanged: `capture_prefix_anchor` and `restore_prefix_anchor`
  each called exactly once per path
* `client.py` 4933 → 4947; store 121 lines
* collaborators from REF-1..REF-5 (`committed_identity`,
  `mtp_session_lifecycle`, `rolling_anchor_store`, `rolling_route_anchor`):
  **zero diff**

## Incident: a stale `.pyc` that faked a regression

The final full-suite run failed one test with `capture_count 2 != 1` — exactly
the signature of a mutant that makes the restore site count a capture. The source
was correct, and `inspect.getsource` on the *loaded* module also showed
`restored=True`, which made the failure look like a transient artifact of the
mutation run.

It was not. Disassembling the loaded function showed `KW_NAMES ('captured',)` at
**both** call sites: the imported module was stale bytecode. The `captured` →
`restored` edit preserved the file's byte length, and the write landed inside the
same filesystem-timestamp second, so the `(mtime, size)` pair CPython uses to
invalidate a cached `.pyc` was unchanged and the cache was trusted. `inspect`
reads the *file*, so it agreed with the source and hid the divergence.

Purging `__pycache__` recompiled it to `('restored',)` / `('captured',)` and the
suite went green. Nothing was wrong with the candidate — but the failure was real
while it lasted, and source inspection alone could not have cleared it. When a
same-length edit appears not to take effect, disassemble rather than re-read.
