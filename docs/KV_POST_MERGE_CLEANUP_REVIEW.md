# KV Post-Merge Cleanup Review

Commit reviewed: `7e810a5` (`origin/main`, PR #57 merged).

This review is intentionally non-destructive. It does not change runtime
behavior, prompts, tool selection, final policy, evidence policy, or KV/cache
behavior. It only records cleanup candidates found after the KV diagnostics and
prefix-anchor work landed on `main`.

## Summary

The KV line now contains three categories of material:

- production opt-in experiment code for `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1`
- diagnostics/probe code used to justify and validate that experiment
- historical analysis documents that explain rejected/no-go branches

No high-confidence dead production path was found. The most plausible cleanup is
small import cleanup plus possible documentation consolidation. Code removal is
riskier because several helpers are intentionally isolated from production but
still covered by tests and referenced by docs.

## Cleanup Safe

### Remove obviously unused imports after static confirmation

Candidate files:

- `src/orbit/native_llama/client.py`
- `src/orbit/native_llama/prefix_anchor_probe.py`

Observed candidates:

- `src/orbit/native_llama/client.py` imports `c_float` from `ctypes`, but local
  usage search only finds it on the import line in this file.
- `src/orbit/native_llama/prefix_anchor_probe.py` imports `LlamaBatch` and
  `PrefixAnchorState`, but local usage search only finds them on import lines.

Why this is safe:

- These are local imports only; removing them should not alter runtime behavior.
- The expected validation is fast and direct.

Required validation before removal:

```bash
PYTHONPATH=src python3 -m unittest tests.test_native_bindings tests.test_prefix_anchor tests.test_prefix_anchor_probe tests.test_kv_diag tests.test_native_chat_template -q
PYTHONPATH=src python3 -m unittest tests.test_payloads tests.test_native_server_protocol tests.test_native_server_think tests.test_llama_server_backend -q
python3 -m unittest discover -s tests -q
python3 -m compileall -q src tests scripts
git diff --check
```

Extra validation:

- run `rg "c_float" src/orbit/native_llama/client.py`
- run `rg "LlamaBatch|PrefixAnchorState" src/orbit/native_llama/prefix_anchor_probe.py`

### Add a docs index instead of deleting historical documents

Candidate:

- add or update a single index section in `docs/KV_CACHE_REUSE_PLAN.md` or a new
  `docs/KV_DOCS_INDEX.md`

Why this is safe:

- The docs set is large and hard to navigate after the phase-by-phase work.
- An index can label documents as current, historical, rejected, no-go, probe,
  or production opt-in without deleting technical evidence.

Required validation:

```bash
git diff --check
```

## Cleanup Risky

### Remove historical no-go/reject reports

Potentially redundant documents:

- `docs/KV_PROMPT_SHAPE_EXPERIMENT.md`
- `docs/KV_PREFIX_CACHE_FEASIBILITY.md`
- `docs/KV_PREFIX_ANCHOR_RUNTIME_NO_GO.md`
- `docs/KV_ROUTE_PREFIX_ANCHOR_RUNTIME_NO_GO.md`
- `docs/KV_ROUTE_PREFIX_TOKEN_BOUNDARY.md`

Why risky:

- These documents explain why prompt-shape hacks, runtime-only fake prefix cache,
  and early route-anchor attempts were rejected.
- They are useful guardrails against reintroducing unsafe performance shortcuts.
- Some documents are superseded technically, but still preserve benchmark and
  safety rationale.

Recommendation:

- Do not delete these now.
- If cleanup is desired, archive them under a clearly named docs section or add
  a phase index that marks them as historical.

Required validation if moving/renaming:

```bash
rg "KV_PROMPT_SHAPE_EXPERIMENT|KV_PREFIX_CACHE_FEASIBILITY|KV_PREFIX_ANCHOR_RUNTIME_NO_GO|KV_ROUTE_PREFIX_ANCHOR_RUNTIME_NO_GO|KV_ROUTE_PREFIX_TOKEN_BOUNDARY" docs README.md
python3 -m unittest discover -s tests -q
python3 -m compileall -q src tests scripts
git diff --check
```

### Remove probe code because production does not call it

Candidate files:

- `src/orbit/native_llama/prefix_anchor_probe.py`
- `tests/test_prefix_anchor_probe.py`

Why risky:

- The probe is intentionally isolated from production, so lack of production
  imports is not proof that it is dead.
- It documents and tests the checkpoint/restore equivalence property used to
  justify the route-prefix-anchor experiment.
- Removing it would weaken regression coverage for the native state APIs.

Recommendation:

- Do not remove.
- Keep as a tested diagnostic/proof harness until the prefix-anchor experiment is
  either promoted, redesigned, or retired.

### Remove boundary diagnostics

Candidate files/areas:

- `src/orbit/native_llama/chat_template.py` route prompt segment helpers
- `tests/test_native_chat_template.py` boundary tests
- `src/orbit/runtime/kv_diag.py` layout/boundary metadata

Why risky:

- The route-prefix-anchor experiment depends on the invariant that the stable
  route prefix boundary is byte- and token-prefix compatible.
- These helpers are part of the safety argument that the prompt semantics are
  unchanged.

Recommendation:

- Do not remove while `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT` exists.

## Things To Not Touch

### Feature flag default

Do not change `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT` default behavior.

Reason:

- The experiment is explicitly opt-in.
- Checkpoint memory cost is material, about 238 MB in the local benchmark.
- The first capture miss remains expensive.

### Route/tools gating

Do not broaden route-prefix-anchor usage beyond:

- native backend
- `phase=route`
- `tools_mode=on`
- explicit `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1`

Reason:

- The merge-ready evidence only covers route tools-on.
- `chat_final`, `final_from_tool`, `tool_call`, file-read, web/fetch, and listing
  must stay behaviorally unchanged.

### Rejected prompt-shape experiment

Do not resurrect or partially reapply `ORBIT_KV_PROMPT_SHAPE_EXPERIMENT`.

Reason:

- It improved short chat cache behavior but introduced repair/retry risk on
  medium no-tool prompts.
- It changed prompt shape semantically enough to be rejected.

### Runtime-only fake prefix cache

Do not introduce replay-token or runtime-only cache masquerading as KV reuse.

Reason:

- The accepted path relies on backend-native checkpoint/restore only.

## Candidate Removal List

No file is recommended for immediate removal.

Low-risk edit candidates, not removals:

- remove unused import `c_float` from `src/orbit/native_llama/client.py` if a
  targeted diff confirms no usage
- remove unused imports `LlamaBatch` and `PrefixAnchorState` from
  `src/orbit/native_llama/prefix_anchor_probe.py` if a targeted diff confirms no
  usage
- add a KV docs index to reduce navigation cost without deleting evidence

## Test Matrix For Any Future Cleanup

For import-only cleanup:

```bash
PYTHONPATH=src python3 -m unittest tests.test_native_bindings tests.test_prefix_anchor tests.test_prefix_anchor_probe tests.test_kv_diag tests.test_native_chat_template -q
PYTHONPATH=src python3 -m unittest tests.test_payloads tests.test_native_server_protocol tests.test_native_server_think tests.test_llama_server_backend -q
python3 -m unittest discover -s tests -q
python3 -m compileall -q src tests scripts
git diff --check
```

For documentation-only cleanup:

```bash
git diff --check
```

For any removal touching prefix-anchor, boundary, or probe code:

```bash
PYTHONPATH=src python3 -m unittest tests.test_native_bindings tests.test_prefix_anchor tests.test_prefix_anchor_probe tests.test_kv_diag tests.test_native_chat_template -q
PYTHONPATH=src python3 -m unittest tests.test_payloads tests.test_native_server_protocol tests.test_native_server_think tests.test_llama_server_backend -q
python3 -m unittest discover -s tests -q
python3 -m compileall -q src tests scripts
git diff --check
```

Also run a native smoke if the removal touches runtime-facing prefix-anchor code:

- OFF tools-on `hi`: no anchor event
- ON tools-on first `hi`: capture miss
- ON tools-on repeat `hi`: restore hit
- file-read: content evidence preserved
- listing: `list_directory` preserved
- web/fetch: `route -> final_from_tool` preserved

## Recommendation

Do not perform code removal yet. The safest immediate cleanup is limited to
unused imports and a documentation index. Keep the historical KV reports and the
isolated probe code until the opt-in prefix-anchor experiment is either promoted
or explicitly retired.
