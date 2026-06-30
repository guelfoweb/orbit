# MTP Unresolved Issues

Status: consolidated MTP correctness/stability analysis. No prompt, routing,
tool-selection, final-policy, evidence-policy, KV, or streaming behavior changes
are included here.

## Baseline Problem Statement

The current MTP line is correctness-first and experimental. The target symptom
to explain is:

- reference `llama-server` MTP on the `response_medium` workload: about
  `6.26s generation_ms`, `16` validates;
- Orbit MTP on the same workload: about `10.16s generation_ms`, `17` validates;
- output is equivalent;
- Orbit has one extra small validate, reportedly `n_tok=2`;
- the extra validate alone does not explain the full remaining latency;
- boundary split and live partial propagation looked promising but need
  revalidation against current `main`.

## Issue Index

| ID | Issue | Evidence | Risk | Candidate Fix Point |
| --- | --- | --- | --- | --- |
| MTP-001 | Orbit MTP is slower than reference `llama-server` despite equivalent output. | Observed `10.16s` vs `6.26s` generation time. | A correctness-preserving path may still be unusable on CPU if overhead exceeds draft savings. | Add comparable phase trace and benchmark matrix before changing behavior. |
| MTP-002 | Orbit performs one extra validate. | Observed `17` validates vs reference `16`; one small validate has `n_tok=2`. | Removing it blindly can break equivalence at the partial/replay boundary. | Expose and compare per-step validate traces before deciding whether it is redundant. |
| MTP-003 | Residual latency is not explained by the extra validate alone. | Extra `n_tok=2` should not account for roughly four seconds by itself. | Optimizing only validate count may leave most slowdown untouched. | Attribute time across target validate, draft generation, checkpoint/restore, `seq_rm`, sampler clone/restore, replay, detokenization, and callback overhead. |
| MTP-004 | Rich step-level C++ traces are intentionally not exposed in the consolidated patch. | Retrieving the heavy step/validate/target-decode JSON was associated with process-exit `double free or corruption (!prev)` in the metadata harness. Stable timing, output-token-hash, and first-sample metadata remain exposed behind `ORBIT_MTP_TRACE=1`. | Re-enabling heavy trace getters can reintroduce teardown instability. | Keep retained diagnostics minimal; redesign heavy trace ownership/lifetime separately if needed. |
| MTP-005 | Boundary split correctness/performance needs current validation. | `ORBIT_MTP_BOUNDARY_SPLIT` defaults enabled; known promising behavior must be rechecked after later changes. | Boundary split can alter partial accept state, `n_past`, target/draft KV shape, and replay needs. | A/B default vs `ORBIT_MTP_BOUNDARY_SPLIT=0` with output equivalence and phase timing. |
| MTP-006 | Live partial propagation may change the next-step validate shape. | `live_partial` clears draft and avoids replay; previous trace fields track `next_validate_n_tok`. | A live partial win could hide an extra follow-up validate or corrupt frontier state if wrong. | Trace `live_partial`, `restored_partial`, and `replay_fallback` resolutions per step. |
| MTP-007 | `seq_rm` and checkpoint/restore costs are unknown in the observed slowdown. | Shim times `seq_rm`, target/draft checkpoint, target/draft restore, but Python does not expose them. | CPU-only `seq_rm`/state restore may dominate when validates are few. | Surface phase timings and compare against `llama-server` speculative state handling. |
| MTP-008 | Request-boundary checkpoint restore may interact with multi-turn reuse. | Shim stores `request_boundary_ckpt` and `request_boundary_prompt_tgt`; README says raw multi-turn MTP reuse remains debug-only. | Prefix restore can improve speed but risks stale frontier reuse if prefix compatibility is incomplete. | Keep raw reuse debug-only; benchmark single-turn and repeat-turn separately. |
| MTP-009 | MTP generation cap is clamped to 32 tokens. | Python calls shim with `max(1, min(max_tokens, 32))`. | Longer answers may silently use shorter MTP segments and continuation behavior differs from standard generation. | Document/benchmark cap behavior; do not change until equivalence and continuation tests exist. |
| MTP-010 | Tool-call rounds are intentionally non-MTP. | `complete_chat` passes `allow_mtp_experimental=not tools`. | Generation benchmarks with tools enabled may not exercise MTP at all during tool-call phases. | Separate plain chat/final-from-tool MTP tests from tool-call route tests. |
| MTP-011 | Thinking mode is intentionally non-MTP. | Client sets fallback reason `thinking-mode`. | Visible-thinking measurements may compare different paths accidentally. | Keep thinking off in MTP generation benchmarks unless a separate design exists. |
| MTP-012 | Sampler state equivalence after partial accept is not deeply tested. | Unit tests mock wrapper results; C++ sampler clone/restore is only real-runtime tested manually. | A partial accept bug can preserve output in one prompt but fail on another. | Add real probe cases that hit full accept, live partial, restored partial, and replay fallback. |
| MTP-013 | Cancellation behavior inside the C++ shim is under-specified. | Python clears stale cancel before MTP and skips MTP if cancel is already true; the C++ loop itself has no visible `should_cancel` callback. | Long MTP validation may not be interruptible at the same granularity as standard generation. | Design cancellation propagation into the shim before treating MTP as stable. |
| MTP-014 | Streaming correctness depends on callback and post-strip behavior. | Python avoids double emit when callbacks fire and strips control channels after completion. | If C++ streams text later stripped by Python, visible stream and final content can diverge. | Add real streaming MTP smoke with control-channel and stop-token boundaries. |
| MTP-015 | Validation rows request logits for every validate token. | `fill_batch` sets `logits[i] = 1` for every row. | This may differ from `llama-server` and increase target validate cost. | Compare with reference speculative implementation before changing logits row selection. |
| MTP-016 | Draft max is fixed at three. | `ORBIT_MTP_DRAFT_N_MAX = 3`. | Draft size may be suboptimal for Gemma 4 12B CPU-only. | Tune only after phase attribution and comparable benchmarks. |
| MTP-017 | Prompt/cache accounting for MTP is approximate from Python tokenization. | MTP reports `prefill_ms=0.0` and `generation_ms=result.elapsed_ms`; prompt reuse is computed after shim completion. | Metrics may mix prefill/replay/generation costs and make comparison misleading. | Split prompt prefill, replay, draft, validate, and output phases in user-visible diagnostics. |
| MTP-018 | Existing tests are wrapper-heavy and do not assert real output equivalence against a target-only decode. | Unit tests mock C/Python boundaries and parse synthetic JSON payloads. | The most important invariant is not protected by CI. | Add optional real-model smoke outside default CI; keep standard suite model-free. |
| MTP-019 | Reference comparison workflow is not codified. | Known measurements were manual. | Future changes can regress or appear to improve due to CPU noise. | Add a repeatable benchmark note or script that records environment, affinity, prompt, and exact artifacts. |
| MTP-020 | Upstream `llama.cpp` now has merged Gemma 4 MTP support, but Orbit's shim has not been re-diffed against current upstream. | PR #23398, `llama : add Gemma4 MTP`, is merged in upstream `master`; upstream docs list `draft-mtp`. | Orbit may carry old assumptions or extra local shim work that current upstream no longer needs. | Compare Orbit's persistent shim against current upstream `common/speculative` and server MTP paths before changing behavior. |
| MTP-021 | Upstream speculative branches exist but are diverged from `master`. | `gg/spec-mtp-experiments`, `gg/spec-ckpt-test`, `gg/server-fix-spec`, and `gg/spec-refactor-ctx` are visible upstream comparison candidates. | Treating an experimental branch as a fix can import stale or incompatible changes. | Use branches as targeted diff sources only; require local equivalence and benchmark proof before porting. |
| MTP-022 | The final `n_tok=2` validate is not proven to be a bug. | Current trace validates `[id_last] + draft`; at the generation cap tail only one draft token remains, so the final validate naturally has two tokens. | Removing it can drop the final fallback/sample row and break output equivalence. | Do not patch validate count until an upstream-equivalent trace proves redundancy. |
| MTP-023 | Target validate dominates latency. | `response_medium_tradeoff`: `target_validate=8101.59 ms / 14 calls`, while `draft_generation=884.893 ms`, `seq_rm=0.284881 ms`, checkpoint/restore and sampler are small. | Optimizing rollback/checkpoint paths will not materially improve this prompt. | Focus diagnosis on target validate batch shape, logits rows, acceptance ratio, and metrics attribution. |
| MTP-024 | Validate batches request logits for every `[id_last] + draft` row. | Orbit `fill_batch(...)` sets `batch.logits[i] = 1` for every validate token; trace reports `validate_batch_logits_count == validate_batch_n_tokens`. Upstream server also adds sampled token and speculative draft tokens with `logits=true` before calling `common_sampler_sample_and_accept_n(...)`. | These rows are likely required by the current sample-and-accept algorithm; reducing them blindly can make acceptance impossible or change fallback sampling. | Treat as equivalent to upstream until a narrower logits-row algorithm is designed and proven. |
| MTP-025 | Acceptance ratio is low enough that target validation erodes draft savings. | `response_medium_tradeoff`: `accepted_tokens_total=18`, `rejected_tokens_total=21`, `acceptance_ratio=0.4615`. | Low acceptance can be normal for this target/draft/prompt, or it can indicate position, suffix, sampler, BOS, cache, or prompt-frontier divergence. | Add metadata-only alignment trace before changing sampler, prompt, positions, or cache behavior. |
| MTP-026 | MTP `generation_ms` is not directly comparable with target-only `generation_ms`. | Target-only reports `prefill_ms=2132.187` and `generation_ms=10179.768`; MTP reports `prefill_ms=0.0` while `generation_ms=10738.387` includes `suffix_target_prefill=1580.72`. | Comparing only `generation_ms` can understate target-only cost or overstate MTP cost depending on which prefill phases are included. | Split MTP suffix prefill from speculative loop in diagnostics before making performance claims. |
| MTP-027 | First-request MTP target suffix prefill is required in the current persistent path. | Without a compatible request-boundary checkpoint, Orbit clears/replays target and draft memory, then decodes the prompt suffix to place target KV at the prompt frontier before validation. | Treating this as avoidable without a valid prefix/request-boundary checkpoint can leave target KV missing or stale. | Investigate reuse/persistent prompt frontier separately; do not remove suffix prefill from the MTP path. |
| MTP-028 | MTP output is not yet proven equivalent to target-only under a prompt-matched control. | Phase 4 target-only with the MTP-prepared prompt used `23` prompt tokens and produced a stable hash different from MTP first-run and second-run hashes. | Any performance optimization before equivalence is explained can preserve a fast but wrong path. | Build a metadata-only equivalence harness around the exact prompt text hash/length, sampler config hash, first-sample state, and per-step frontier. |
| MTP-029 | Phase 4 alignment trace did not show target/draft KV frontier mismatch. | Across three first-run and three second-run MTP traces, `ctx_tgt` and `ctx_dft` frontier min/max matched before each validate; draft origin was consistently `fresh`; replay-before appeared once at request start. | Acceptance may still be low due to sampler/config or draft quality, but not due to the basic frontier mismatch fields collected here. | Do not patch frontier/KV rollback logic based on current data. |
| MTP-030 | Mixed target-only/MTP benchmark still aborts at process teardown. | The first post-PR #72 validation benchmark completed 18 rows, wrote metadata JSON, then exited with `double free or corruption (!prev)` and exit code 134. | MTP cannot be considered stable for broader use until repeated mixed-client/session lifetimes exit cleanly. | Reproduce the 18-row harness with lifetime counters; isolate whether target-only client, MTP reset, mmproj context, CDLL cache, or process-global llama.cpp cleanup still owns memory twice. |

## Candidate Fix Points, Without Patching Runtime

1. Diagnostics bridge:
   - keep only stable env-gated metadata in Python results:
     `timing_json`, output-token hashes, and first-sample metadata;
   - do not expose heavy step/validate/target-decode JSON until its native
     lifetime/teardown behavior is redesigned;
   - do not change decode, sampler, prompt, or runtime policy.
2. Validate sequence comparison:
   - collect Orbit per-step `validate_batch_n_tokens`;
   - collect reference `llama-server` validate count/shape if available;
   - identify the step that creates the extra `n_tok=2` validate.
3. Boundary split A/B:
   - run current default boundary split;
   - run `ORBIT_MTP_BOUNDARY_SPLIT=0`;
   - compare output equivalence, validate count, phase timing, and resolutions.
4. Live partial audit:
   - confirm each `live_partial` has matching target/draft KV max positions after
     commit;
   - confirm no replay follows unless explicitly required;
   - verify the next validate's token count is expected.
5. Phase attribution:
   - inspect phase totals for `target_validate`, `draft_generation`, `seq_rm`,
     `ctx_tgt_checkpoint`, `ctx_tgt_restore`, `ctx_dft_checkpoint`,
     `ctx_dft_restore`, `sampler_clone`, `sampler_restore`, `sampler_ops`,
     `rollback_replay`, and `detokenize_output_bridge`.
6. Reference implementation comparison:
   - compare Orbit's `fill_batch`/logits-row strategy and partial rollback logic
     to upstream `common/speculative` and the `llama-server` path used in the
     baseline.
7. Upstream compatibility diff:
   - start from upstream `master` after Gemma 4 MTP support;
   - diff MTP-specific code against `gg/spec-mtp-experiments`;
   - inspect checkpoint/rollback changes in `gg/spec-ckpt-test` and
     `gg/spec-refactor-ctx`;
   - do not port branch code without output-equivalence and phase-timing proof.

## Phase 2 Trace Evidence

Captured on local CPU-only Gemma 4 12B using repository models under
`models/`, `ctx=8192`, `threads=6`, `threads-batch=6`, `batch=256`,
`ubatch=128`, `taskset -c 0-5`, `max_tokens=32`, thinking off, tools off for
generation-only comparison.

Instrumentation status:

- The consolidated patch retains only stable env-gated metadata exposed through
  Python when `ORBIT_MTP_TRACE=1`:
  - phase timing JSON;
  - generated output token hashes;
  - first-sample prompt/frontier/logits/sample hashes.
- Heavy step/validate/target-decode trace getters are not exposed by the Python
  result object in the consolidated patch. Earlier experiments showed that
  retrieving those large native JSON strings could reproduce process-exit heap
  corruption in the lifecycle harness.
- No prompt, routing, tool, final, evidence, decode, sampler, KV, or MTP state
  machine behavior was changed by the diagnostics bridge.

### Orbit Target-Only Control

`response_medium_tradeoff` control, target-only:

| Field | Value |
| --- | ---: |
| output tokens | 32 |
| prompt tokens | 30 |
| evaluated prompt tokens | 30 |
| prefill ms | 2132.187 |
| generation ms | 10179.768 |
| wall ms | 12312.637 |

The target-only control is not expected to expose validate sequence because it
does not use speculative decoding.

### Orbit MTP Trace: `response_medium_tradeoff`

| Field | Value |
| --- | ---: |
| output tokens | 32 |
| target decode calls | 15 |
| draft decode calls | 14 |
| validate count | 14 |
| draft tokens total | 39 |
| accepted tokens total | 18 |
| rejected tokens total | 21 |
| acceptance ratio | 0.4615 |
| full accept steps | 6 |
| partial accept steps | 8 |
| replay fallback steps | 0 |
| generation ms | 10738.387 |
| wall ms | 10740.206 |

Validate `n_tok` sequence:

```text
4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2
```

The final `n_tok=2` is visible in this run. It is not an unexplained duplicate
by itself: Orbit validates `[id_last] + draft`, and at the generation cap tail
the remaining draft budget is one token, so the validate batch naturally becomes
two tokens.

Phase timing totals:

| Phase | Calls | Total ms |
| --- | ---: | ---: |
| target validate | 14 | 8101.59 |
| draft generation | 14 | 884.893 |
| suffix target prefill | 1 | 1580.72 |
| target checkpoint | 15 | 34.0878 |
| draft restore | 14 | 0.105214 |
| sampler ops | 15 | 25.2337 |
| seq_rm | 28 | 0.284881 |
| speculative loop total | 14 | 10738.4 |

Primary latency finding:

- Residual latency is dominated by target validation, not `seq_rm`,
  checkpoint/restore, sampler operations, or draft generation.
- `seq_rm` is effectively negligible in this trace (`0.284881 ms` total).
- Target validate averages roughly `578.7 ms` per call for this prompt.

### Additional Orbit MTP Sweep

| Label | Validates | `n_tok` sequence | Generation ms | Target validate ms | Draft ms | Acceptance |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `response_medium_latency` | 13 | `4 x 13` | 7605.400 | 5180.27 | 730.55 | 0.4872 |
| `response_medium_tradeoff` | 14 | `4 x 12, 3, 2` | 10738.387 | 8101.59 | 884.893 | 0.4615 |
| `response_medium_cpu` | 11 | `4 x 11` | 9277.915 | 6749.24 | 762.029 | 0.5714 |

The historical `17` validate count was not reproduced by these three prompt
variants on current `main`, but the small tail validate (`n_tok=2`) was
reproduced.

### Reference `llama.cpp` Check

Attempted local `llama-speculative` reference using the same target/draft GGUFs,
`draft-mtp`, `n_predict=32`, and matching CPU/thread/batch settings. The local
binary exited with `SIGSEGV` (`returncode=-11`) before exposing usable timing or
validate metadata. The stock reference path therefore remains unavailable in
this worktree for a direct per-validate sequence comparison.

The known external reference remains:

- `llama-server` MTP: about `6.26s generation_ms`, `16` validates.
- Orbit historical MTP: about `10.16s generation_ms`, `17` validates.

Current local evidence is enough to attribute Orbit's measured slowdown for the
captured prompts to target validate cost, but not enough to prove the historical
Orbit-vs-`llama-server` validate-count delta.

## Phase 3 Bottleneck Diagnosis

Phase 3 changes the working hypothesis:

- the tail validate with `n_tok=2` is not demonstrated to be a bug;
- `seq_rm`, target/draft checkpoint/restore, sampler operations, and draft
  decode are not the dominant costs in the captured run;
- the dominant cost is target-side validation;
- the current metric split makes target-only and MTP `generation_ms` only
  partially comparable.

### Validate Batch Construction

Orbit builds each target validate batch in
`src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp`:

- `orbit_mtp_session_complete(...)` constructs:
  - `validate_tokens = [id_last] + draft`
  - `validate_pos0 = n_past`
- `resolve_validate_accept_restore(...)` creates a `llama_batch` from those
  tokens and calls `llama_decode(ctx_tgt, validate)`.
- `fill_batch(...)` sets `batch.logits[i] = 1` for every validate row.
- After target decode, Orbit passes row indices `0..validate_tokens.size()-1`
  to `common_sampler_sample_and_accept_n(...)`.

The current validate rows are therefore:

```text
row 0: logits after id_last
row 1..N: logits after each draft token
```

That shape matches the current sample-and-accept algorithm. The sampler compares
the target sample from each row to the corresponding draft token; if all draft
tokens match, it also samples the final row to produce the next token. With a
draft of three tokens, four rows are needed. With a final draft of one token,
two rows are needed. This explains why the observed tail sequence can end with
`n_tok=2` without implying an extra duplicate validate.

### Upstream `llama.cpp` Comparison

Checked against upstream `llama.cpp` `master` after Gemma 4 MTP support.

Relevant upstream server/speculative behavior:

- `server_n_outputs_max(...)` reserves `1 + common_speculative_n_max(...)`
  output rows per sequence.
- `server_slot::handle_last_sampled_token(...)` adds the sampled token and each
  speculative draft token to the target batch with `logits=true`.
- The server stores the batch row indices in `spec_i_batch`.
- After `llama_decode`, the server calls
  `common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt,
  slot.spec_i_batch, slot.spec_draft)`.
- `common_sampler_sample_and_accept_n(...)` requires `idxs.size() ==
  draft.size() + 1`, samples one row per draft token until mismatch, and samples
  the final row on full accept.

Conceptual diff:

| Area | Orbit persistent MTP | Upstream server MTP |
| --- | --- | --- |
| Validate token sequence | standalone `[id_last] + draft` batch | sampled token plus draft tokens in server batch |
| Logits rows | all validate rows set `logits=1` | sampled and draft rows added with `logits=true` |
| Sampler call | `common_sampler_sample_and_accept_n(..., rows, draft)` | same function with server batch row indices |
| Final small validate | expected if only one draft token remains | expected under same `draft.size()+1` rule |
| Scheduling | one standalone target validate decode per MTP step | validate rows integrated into server slot batch |

Conclusion: Orbit is equivalent to upstream at the level that matters for
validate rows and sample-and-accept semantics. The current evidence does not
show Orbit validating extra logits rows beyond the rows needed by this upstream
algorithm.

### Why Target Validate Costs So Much

The likely bottleneck is the combination of:

1. every validate step asks the 12B target model to decode a small multi-row
   batch with logits on every row;
2. acceptance is low (`0.4615` on `response_medium_tradeoff`), so many expensive
   target rows do not turn into accepted output;
3. CPU-only target validation is much more expensive than draft decode, so the
   acceptance ratio must be high enough to amortize target validate cost.

For the captured `response_medium_tradeoff` run:

```text
target_validate       8101.59 ms / 14 calls
draft_generation       884.893 ms / 14 calls
suffix_target_prefill 1580.72 ms / 1 call
ctx_tgt_checkpoint      34.0878 ms / 15 calls
sampler_ops             25.2337 ms / 15 calls
seq_rm                   0.2849 ms / 28 calls
ctx_dft_restore          0.1052 ms / 14 calls
```

This rules out `seq_rm`, checkpoint/restore, and sampler operations as the main
explanation for the measured latency.

### Acceptance Ratio Diagnosis

The observed `acceptance_ratio=0.4615` is not enough by itself to prove a bug.
It can be normal for a weak or mismatched draft, but it can also indicate an
alignment issue. The next trace should verify, without printing raw tokens:

- target prompt token count versus MTP prompt token count;
- `id_last` position and prompt-frontier position at each step;
- draft `n_past` and target `n_past`;
- target/draft KV max positions before draft, before validate, and after
  accept;
- sampler state hash before draft, before validate, and after accept;
- draft `p_min`, temperature, top-k, top-p, and greedy/grammar-relevant config;
- whether the prompt suffix used by target and draft is identical by hash and
  length.

No sampler, prompt, BOS, suffix, or position fix is justified until this
alignment trace exists.

### Suffix Target Prefill

`suffix_target_prefill=1580.72 ms` is expected in the current first-request MTP
path. The persistent session starts the request with `need_replay=true`. If no
compatible request-boundary checkpoint is available, the shim clears target and
draft memory, decodes the rendered prompt into the target context with
`fill_target_prefill_batch(...)`, and only then enters speculative validation.

This is necessary to put target KV at the correct prompt frontier. It may become
avoidable only when a valid request-boundary checkpoint or persistent prompt
frontier can be reused safely. Removing it from the first-request path would be
a correctness risk.

### Metric Comparability

Target-only and MTP timing fields are not directly comparable:

- target-only reports prompt prefill separately as `prefill_ms`;
- MTP currently reports `prefill_ms=0.0`;
- MTP `generation_ms` includes the suffix target prefill and the speculative
  loop.

For the captured run:

```text
target-only wall ~= prefill_ms 2132.187 + generation_ms 10179.768
MTP wall         ~= generation_ms 10738.387
MTP generation  includes suffix_target_prefill 1580.72
```

If `suffix_target_prefill` is subtracted only for diagnostic comparison, the
MTP speculative loop is roughly `9157.7 ms`, which is modestly lower than the
target-only generation phase. That does not prove a runtime speedup because the
public metric currently includes different phase boundaries.

## Phase 3 Fix Candidates, Without Patching

These are candidate fix points only. None should be patched until the next
targeted trace confirms the condition they depend on.

### Candidate 1. Split MTP Timing Attribution

- File/function:
  - `src/orbit/native_llama/persistent_mtp.py::run_persistent_mtp_completion`
  - `src/orbit/native_llama/client.py` MTP timing mapping
  - C++ phase timing JSON in `orbit_persistent_mtp.cpp`
- Minimal change:
  - expose `suffix_target_prefill_ms` separately from speculative loop time in
    user-visible diagnostics or `NativeTimings`;
  - do not change decode behavior.
- Risk:
  - low runtime risk, but medium reporting risk because external benchmarks may
    rely on current `generation_ms` semantics.
- Correctness test:
  - unit test that total elapsed still matches phase sum and output is
    unchanged.
- Expected benchmark effect:
  - no speed change; makes target-only versus MTP comparisons valid.

### Candidate 2. Draft/Target Alignment Trace

- File/function:
  - `src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp`
  - trace construction around draft generation and
    `resolve_validate_accept_restore(...)`
- Minimal change:
  - add env-gated metadata for prompt hash/length, suffix hash/length,
    target/draft positions, sampler config hash, and KV frontier positions;
  - keep token ids, prompt text, and token pieces out of the trace.
- Risk:
  - low if metadata-only; privacy risk if hashes are replaced by raw content.
- Correctness test:
  - tests assert raw token/prompt fields are absent and trace is disabled by
    default.
- Expected benchmark effect:
  - no speed change; determines whether `0.4615` acceptance is normal or caused
    by divergence.

### Candidate 3. Validate Row Reduction Experiment

- File/function:
  - `src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp::fill_batch`
  - `resolve_validate_accept_restore(...)`
- Minimal change:
  - experimental branch only: attempt narrower logits row selection and replace
    the acceptance algorithm if an upstream-equivalent method supports it.
- Risk:
  - high. The current upstream `common_sampler_sample_and_accept_n(...)`
    requires `draft.size()+1` logits rows. Reducing rows without changing the
    algorithm breaks acceptance or fallback sampling.
- Correctness test:
  - real-model target-only equivalence across full accept, partial accept,
    reject, and generation cap tail cases.
- Expected benchmark effect:
  - possible validate speedup only if a mathematically equivalent lower-output
    acceptance path exists; current upstream comparison does not show one.

## Phase 4 Clean Metrics And Acceptance Diagnosis

Phase 4 added metadata-only trace fields behind `ORBIT_MTP_TRACE=1`:

- timing summary:
  - `suffix_target_prefill_ms`
  - `speculative_loop_ms`
  - `speculative_loop_including_suffix_ms`
  - `target_validate_ms`
  - `draft_generation_ms`
  - `checkpoint_restore_ms`
  - `sampler_ms`
  - `seq_rm_ms`
  - `non_loop_overhead_ms`
- per-step alignment metadata:
  - `draft_count`, `accepted_draft`, `rejected_draft`
  - `draft_origin`, `draft_is_fresh`
  - `need_replay_before`, `post_step_need_replay`
  - `validate_n_tok`, `validate_pos0`
  - `old_n_past`, `new_n_past`
  - `prompt_tgt_len`, `prompt_dft_len`
  - target/draft KV frontier min/max and frontier hashes
  - sampler state hashes before/after
  - `remaining_generation_cap`

The trace remains sanitized on the Python side. Token id arrays, token pieces,
raw sampler summaries, prompt text, user content, file content, tool output, and
web output are not exposed through the Python result object.

### Phase 4 Benchmark Setup

Local CPU-only Gemma 4 12B, repository models under `models/`, `ctx=8192`,
`threads=6`, `threads-batch=6`, `batch=256`, `ubatch=128`, `taskset -c 0-5`,
`max_tokens=32`, thinking off, tools off for generation-only comparison.

Three runs were collected for each primary case:

- target-only through normal `complete_chat_text`;
- MTP first request through the persistent MTP session;
- MTP second identical request on the same persistent MTP client.

An additional target-only diagnostic used the same MTP-prepared prompt length
(`23` tokens) so prompt accounting can be compared to MTP. That diagnostic is
not a runtime behavior change.

### Metric Summary

| Case | Runs | Prompt tokens | Output tokens | Wall ms avg | Prefill ms avg | Generation ms avg | Suffix target prefill ms avg | Speculative loop ms avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target-only normal prompt | 3 | 30 | 32 | 13188.973 | 2665.622 | 10522.632 | n/a | n/a |
| target-only MTP-prepared prompt | 3 | 23 | 32 | 12263.795 | 2088.044 | 10175.415 | n/a | n/a |
| MTP first request | 3 | 23 | 32 | 10881.980 | 0.000 | 10878.310 | 2107.363 | 8770.927 |
| MTP second request | 3 | 23 | 32 | 7617.320 | 0.000 | 7614.183 | 0.000 | 7614.167 |

Interpretation:

- MTP `generation_ms` is now split into suffix prefill and speculative loop.
- Compared to target-only on the MTP-prepared prompt, first-run MTP's loop-only
  time is lower (`8770.927 ms` vs `10175.415 ms`), but first-run MTP still pays
  `2107.363 ms` of suffix target prefill.
- The second identical MTP request eliminates suffix target prefill in this
  trace and improves wall time materially (`7617.320 ms` average).
- This confirms that the request-boundary/persistent checkpoint reuse path is
  valuable for repeated prompts.

Important correctness caveat:

- target-only normal prompt, target-only MTP-prepared prompt, MTP first request,
  and MTP second request each produced stable hashes across their three runs;
- the hashes differ across those cases;
- therefore Phase 4 does not prove target-only/MTP output equivalence.

No performance fix should be applied until that equivalence gap is understood.

### MTP First Request Acceptance Trace

Across all three runs the first-request trace was identical:

```text
validate_n_tok:
4,4,4,4,4,4,4,4,4,4,4,4,4,3,2

accepted_draft:
3,3,3,1,2,0,0,0,0,1,0,2,1,0,1

rejected_draft:
0,0,0,2,1,3,3,3,3,2,3,1,2,2,0
```

Aggregate:

| Metric | Value |
| --- | ---: |
| acceptance ratio | 0.4047619 |
| target decode calls | 16 |
| draft decode calls | 15 |
| target validate ms avg | 7650.517 |
| draft generation ms avg | 929.830 |
| checkpoint/restore ms avg | 46.917 |
| sampler ms avg | 31.341 |
| seq_rm ms avg | 0.299 |

Alignment observations:

- `ctx_tgt` and `ctx_dft` frontier min/max matched before every validate.
- `draft_origin` was consistently `fresh`.
- `need_replay_before` appeared only on the request-start step.
- No replay fallback occurred in the summarized trace.
- The tail validates are explained by residual generation cap:
  - step 13: remaining cap `4`, validate `4`, draft `3`;
  - step 14: remaining cap `2`, validate `3`, draft `2`;
  - step 15: remaining cap `1`, validate `2`, draft `1`.

Conclusion: the final `n_tok=2` is explained by the generation cap tail in this
trace.

### MTP Second Request Acceptance Trace

Across all three second-request runs the trace was also identical:

```text
validate_n_tok:
4,4,4,4,4,4,4,4,4,4,4,4,3

accepted_draft:
0,0,0,3,3,3,3,2,0,0,0,3,2

rejected_draft:
3,3,3,0,0,0,0,1,3,3,3,0,0
```

Aggregate:

| Metric | Value |
| --- | ---: |
| acceptance ratio | 0.5000000 |
| target decode calls | 13 |
| draft decode calls | 13 |
| target validate ms avg | 6539.393 |
| draft generation ms avg | 860.911 |
| suffix target prefill ms avg | 0.000 |
| checkpoint/restore ms avg | 34.397 |
| sampler ms avg | 31.659 |
| seq_rm ms avg | 0.275 |

The second request is faster because the suffix target prefill cost is removed
and the validate sequence is shorter. The trace does not show target/draft KV
frontier mismatch.

### Acceptance Diagnosis

Current classification: acceptance is suspicious / data insufficient.

Reasons:

- basic target/draft frontier alignment looks coherent in the collected fields;
- the low first-run acceptance ratio (`0.4047619`) could be draft weakness;
- however, target-only with the same MTP-prepared prompt produced a stable
  output hash different from MTP;
- this means the next correctness question is equivalence of target-only and
  MTP sampling/frontier, not raw speed.

Known prompt accounting detail:

- normal target-only chat path reports `30` prompt tokens;
- MTP path uses `_prepare_mtp_prompt(...)` and reports `23` prompt tokens;
- target-only on the MTP-prepared prompt also reports `23` prompt tokens.

The prompt-length difference explains why normal target-only and MTP metrics are
not directly comparable, but it does not explain why target-only on the
MTP-prepared prompt and MTP produce different output hashes.

## Phase 4 Fix Candidates, Without Patching

No semantic or performance fix is justified yet. The data supports at most these
two next steps:

### Candidate 1. Output-Equivalence Harness

- File/function:
  - `src/orbit/native_llama/client.py::_try_complete_with_mtp_experimental`
  - `src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp`
- Minimal change:
  - add a debug-only harness that runs target-only and MTP on the exact same
    prompt hash/length and reports only output hash, output token count, timing,
    and first divergence index if a token-hash stream can be exposed safely.
- Risk:
  - medium privacy/correctness risk if raw token ids or content leak; keep it
    env-gated and metadata-only.
- Correctness test:
  - assert no raw prompt/token/content fields appear in the harness output.
- Expected benchmark effect:
  - no speed change; determines whether MTP is a correct optimization path.

### Candidate 2. Sampler/First-Sample Alignment Trace

- File/function:
  - `src/orbit/native_llama/client.py::_generate_from_current_context`
  - `src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp`
- Minimal change:
  - expose metadata-only sampler config hash and first-sample state hash for
    target-only prepared-prompt control versus MTP before the first generated
    token.
- Risk:
  - low if no raw sampler summaries or token ids are emitted.
- Correctness test:
  - model-free sanitizer tests plus one optional real-model smoke.
- Expected benchmark effect:
  - no speed change; explains whether output hash divergence comes from sampler
    setup, prompt boundary, or MTP state machine.

## Risks

- Removing the extra validate without a state proof can break output
  equivalence.
- Optimizing boundary split can introduce missing/duplicated streamed tokens.
- Reducing `seq_rm`/restore work can leave target or draft KV tails valid when
  they should be invalid.
- MTP can appear faster or slower because of CPU scheduler noise; single runs
  are not strong evidence.
- Making MTP useful for generation must not change prompt, route, tool,
  evidence, or final-answer behavior.
- Tool-call and thinking paths are deliberately not the same as plain MTP chat;
  benchmark prompts must not mix these modes accidentally.
- Any runtime fix that depends on a specific prompt such as `response_medium`
  would be an invalid semantic hardcode.

## Phase 5 Output-Equivalence Blocker

Phase 5 added a metadata-only output-equivalence harness for target-only
prepared-prompt generation versus persistent MTP. The harness reports prompt
hashes, prompt token counts, canonical sampler config hashes, output hashes,
per-output token hash digests, and first divergence indexes. It does not report
raw prompt text, token ids, token pieces, user content, or model output.

### Controlled Setup

- Same prepared prompt was used for target-only and MTP.
- Same max token cap was used: `32`.
- Same canonical sampler config hash was reported for both paths:
  `df8522360e5dfcea`.
- `sampler_initial_state_hash` is not available yet.
- `first_sample_state_hash` is a hash of the first generated token only, not a
  raw token id.

### Three-Run Result

All three runs produced the same prepared prompt metadata:

- `prompt_token_count`: `27`
- `prompt_hash`: `589ef9d2af97baae`

Output hashes were stable within each path but differed across paths:

| Path | Output hash sequence | Token-hash digest sequence | First sample hash |
| --- | --- | --- | --- |
| target-only prompt-matched | `a465892fd4563b81` x3 | `b9e505e22dac86b1` x3 | `4194370396891896960` |
| MTP first request | `db3387031e7c3d53` x3 | `42aa0d61a075f302` x3 | `11503514907415216418` |
| MTP second identical request | `b75fc404db1d052c` x3 | `8c58843c4536586b` x3 | `1225426879707157548` |

The first divergence index was stable:

- MTP first request versus target-only: `0`
- MTP second identical request versus target-only: `0`
- MTP second identical request versus MTP first request: `0`

### Timing Context

The prompt-matched target-only generation and MTP loop are now separable:

| Path | Wall avg | Generation avg | Suffix target prefill avg | Speculative loop avg | Target validate avg | Draft generation avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| target-only prompt-matched | `11999.770 ms` | `9806.816 ms` | n/a | n/a | n/a | n/a |
| MTP first request | `6418.855 ms` | `6415.958 ms` | `1549.840 ms` | `4866.110 ms` | `4170.290 ms` | `534.852 ms` |
| MTP second identical request | `7623.555 ms` | `7620.295 ms` | `0 ms` | `7620.277 ms` | `6601.570 ms` | `833.411 ms` |

The MTP first-request loop can be faster than the target-only prompt-matched
generation path, but this is not actionable until output equivalence is
established.

### Diagnosis

The divergence happens before any validate/accept correctness question can be
answered:

- prompt hash and prompt token count are identical;
- canonical sampler config hash is identical;
- first generated token hash differs at index `0`;
- MTP first and MTP second also diverge at index `0`.

This points to one of the following unresolved causes:

- the target-only harness is still not equivalent to the target path used inside
  MTP despite matching prompt tokens;
- the MTP path samples the first token from a different target-context state;
- the MTP path's `common_sampler` setup is not bit-equivalent to Orbit's
  standard greedy sampler chain;
- a sampler initial state or target frontier detail is missing from the current
  trace.

Acceptance ratio and target validate cost must not be optimized until this
first-sample equivalence blocker is resolved. The final `n_tok=2` validate tail
remains a likely generation-cap effect, not a demonstrated bug.

### Current Decision

Do not patch performance, acceptance, validate rows, suffix prefill, or boundary
split yet. The next safe step is narrower correctness instrumentation that
proves whether the first-sample mismatch is caused by sampler implementation,
target context state, or a non-equivalent target-only harness.

## Phase 6 First-Sample Root Cause

Phase 6 added metadata-only first-sample traces immediately before and after the
first sample in both the target-only prepared-prompt path and the persistent MTP
path. The trace includes prompt hash/count, KV frontier, last logits hash,
sampler implementation id, sampler state hash when available, first sample hash,
and whether request-boundary restore was used. It does not expose token ids,
token pieces, prompt text, logits values, or model output.

### Finding 1. The Phase 5 Target Harness Was Not Equivalent

The earlier target-only "prompt-matched" comparison used a 27-token prompt, but
the actual MTP C shim sampled from a 20-token prompt. The Python MTP entrypoint
applies `_prepare_mtp_prompt(...)` again before calling the C shim, which strips
the thought-channel suffix from an already-rendered prompt. Therefore the Phase
5 target-only prompt was not the same prompt consumed by MTP.

First divergent field in the invalid Phase 5 comparison:

- effective internal prompt count:
  - target-only harness: `27`
  - MTP first request: `20`

This makes the original target-only-vs-MTP-first output divergence a harness
error, not an MTP validate/acceptance bug.

### Corrected Harness Result

The corrected harness uses the effective prompt that reaches the C shim. With
that prompt, target-only and the first MTP request match at the first-sample
boundary and in final output hash.

| Path | Prompt count | Prompt hash | Frontier max | Last logits hash | First sample hash | Output hash |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| target-only effective prompt | `20` | `10014497124828543528` | `19` | `4957177649268112296` | `11503514907415216418` | `db3387031e7c3d53` |
| MTP first request | `20` | `10014497124828543528` | `19` | `4957177649268112296` | `11503514907415216418` | `db3387031e7c3d53` |

This falsifies the sampler-implementation mismatch hypothesis for the first
request: despite `llama_sampler_chain+greedy` versus `common_sampler(top_k,temp)`
metadata, the first request samples the same token from the same logits.

### Finding 2. MTP Second Request Diverges At Restored Logits

The second identical MTP request still diverges at index `0`, but the first
divergent field is not prompt hash, prompt count, frontier, or sampler state
hash before sample. It is `last_logits_hash`.

| Path | Prompt count | Prompt hash | Frontier max | Request-boundary restore | Last logits hash | First sample hash | Output hash |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| MTP first request | `20` | `10014497124828543528` | `19` | `false` | `4957177649268112296` | `11503514907415216418` | `db3387031e7c3d53` |
| MTP second identical request | `20` | `10014497124828543528` | `19` | `true` | `9250577076402425798` | `1225426879707157548` | `b75fc404db1d052c` |

The KV frontier hash is the same (`7521935930782995232`) and the sampler state
hash before the first sample is the same (`1469598103934665603`), but the logits
row used for sampling differs after loading the request-boundary checkpoint.

Most likely root cause:

- `common_prompt_checkpoint::load_tgt(...)` restores KV/state enough for
  frontier accounting, but the logits row used by `common_sampler_sample(...)`
  is not restored to the same value as a fresh target prefill.
- The second request therefore samples from stale or otherwise non-equivalent
  target logits even though the restored KV frontier looks correct.

Relevant code point:

- `src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp`
  - `orbit_mtp_session_complete(...)`
  - request-boundary restore branch using
    `session->request_boundary_ckpt.load_tgt(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)`
  - first sample immediately after restore through
    `common_sampler_sample(smpl, ctx_tgt, -1)`

### Current Classification

- Target-only versus MTP first: previous divergence was a non-equivalent harness.
- MTP first versus MTP second: checkpoint/restore logits mismatch.
- Sampler implementation mismatch: falsified for the first request.
- Validate/acceptance mismatch: not implicated in the first-sample divergence.
- Performance patching remains blocked until second-request restore equivalence
  is fixed or request-boundary restore is excluded from equivalence benchmarks.

## Phase 7: Restore/Logits Equivalence And Teardown Stability

### Restore/Logits Root Cause

The restored request-boundary path had the same effective prompt and KV frontier
as fresh MTP, but the final logits row used by `common_sampler_sample(...)`
differed:

```text
MTP first:
  prompt_count=20
  prompt_hash=10014497124828543528
  frontier=0..19
  last_logits_hash=4957177649268112296
  first_sample_hash=11503514907415216418
  output_hash=db3387031e7c3d53

MTP second before fix:
  prompt_count=20
  prompt_hash=10014497124828543528
  frontier=0..19
  restore=true
  last_logits_hash=9250577076402425798
  first_sample_hash=1225426879707157548
  output_hash=b75fc404db1d052c
```

Confirmed:

- H1/H3: the request-boundary checkpoint restored enough KV/frontier state for
  accounting, but did not restore a final logits row equivalent to fresh target
  prefill.
- H6: sampling after restore read a non-equivalent logits row even though the
  frontier was correct.

Rejected or narrowed:

- H4 as a one-token refresh: removing and decoding only the final prompt token
  after restore did not recover equivalence. The resulting logits still differed.
- H5 as the only cause: the checkpoint can still be useful for draft/spec state,
  but the target logits row cannot be trusted after partial restore.

### Correctness Patch Applied

The request-boundary restore branch now refreshes the target side by clearing
target memory and decoding the effective target prompt again before the first
sample. Draft/spec state is still restored from the request-boundary checkpoint.

Relevant file/function:

- `src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp`
  - `orbit_mtp_session_complete(...)`
  - request-boundary restore branch before first `common_sampler_sample(...)`

Post-fix result:

```text
target-only effective:
  prompt_hash=10014497124828543528
  frontier=0..19
  last_logits_hash=4957177649268112296
  first_sample_hash=11503514907415216418
  output_hash=db3387031e7c3d53

MTP first:
  prompt_hash=10014497124828543528
  frontier=0..19
  last_logits_hash=4957177649268112296
  first_sample_hash=11503514907415216418
  output_hash=db3387031e7c3d53

MTP second:
  prompt_hash=10014497124828543528
  frontier=0..19
  restore=true
  request_boundary_logits_refreshed=true
  last_logits_hash=4957177649268112296
  first_sample_hash=11503514907415216418
  output_hash=db3387031e7c3d53
```

Performance impact:

- The fix is correctness-first and not a performance optimization.
- MTP second now pays a target prompt refresh cost before sampling.
- In the measured harness, MTP second had roughly `suffix_target_prefill_ms` in
  the 1.7-2.4s range and a loop around 5.3-5.6s for the tested prompt/cap.
- Future performance work must preserve the restored-logits equivalence above.

### Teardown Double-Free Status

The teardown abort was reproducible after repeated mixed target-only/MTP client
lifetimes:

```text
target-only client -> MTP client -> repeated 3 times
all completions and client.close() calls completed
process aborted at exit with: double free or corruption (!prev)
```

The minimal single-client MTP path exited cleanly. A gdb run showed the abort
inside libc exit handlers after all Orbit cleanup had completed, with a
`common_log` worker thread still present from `libllama-common.so`. The issue was
not a direct `orbit_mtp_session_free(...)` failure.

Confirmed so far:

- The crash happens after all completions and explicit `client.close()` calls
  complete.
- The crash is detected by libc during process-exit cleanup.
- A single MTP client with completion and close exits cleanly.
- Multiple MTP-only clients can exit cleanly.
- Repeated mixed target-only/MTP clients without trace metadata can exit cleanly
  after avoiding per-client backend global free.
- The full metadata equivalence harness originally exited with `double free or
  corruption (!prev)` after producing all expected rows.
- For the consolidated patch, the retained `ORBIT_MTP_TRACE=1` surface is limited
  to stable metadata (`timing_json`, output-token hashes, and first-sample
  hashes). The same mixed target/MTP lifecycle reproducer exits cleanly with
  that retained surface.
- Heavy step/validate/target-decode JSON retrieval remains excluded from the
  retained diagnostics because it can still reproduce the exit-134 teardown
  failure.

Fixes applied:

- `NativeLlamaClient.close()` still frees the MTP session, sampler, contexts, and
  model, but no longer frees llama.cpp process-global backend state per client.
- Native `CDLL` handles are cached by absolute path so repeated client/shim use
  does not repeatedly unload/reload the same runtime libraries.
- `PersistentMtpSessionRuntime` retains the MTP shim library object for the
  lifetime of the C++ session handle.
- The persistent MTP shim build now links against the packaged vendor build bin
  when the runtime lib directory does not expose `libllama.so.0` and
  `libllama-common.so.0`. This avoids building or loading the shim against a
  stale source-tree llama.cpp build while the Python runtime uses packaged
  vendor libraries.

Validation and residual blocker:

```text
single MTP client, one completion, close: exit 0
target + MTP pair, close: exit 0
3 MTP clients, two completions each: exit 0
3 mixed target/MTP cycles before backend-free fix: exit 134
3 mixed target/MTP cycles after backend-free fix: exit 0
3 target/MTP cycles without mmproj: exit 0
5 load/close cycles with mmproj only: exit 0
3 target-only cycles with mmproj: exit 0
3 MTP clients with mmproj: exit 0
3 target/MTP cycles with mmproj after shim link-bin fix: exit 0
2 target/MTP cycles with retained stable ORBIT_MTP_TRACE metadata: exit 0
2 target/MTP cycles with heavy step/target-decode metadata retrieval: exit 134
```

The runtime-path double-free is resolved in the local repro matrix for normal
MTP operation and for the retained stable diagnostics. One confirmed root cause
was a runtime/linkage mismatch: the persistent MTP shim could be built against a
different llama.cpp build than the runtime libraries loaded by `NativeLlamaClient`.
That allowed incompatible llama.cpp/common runtime state to coexist in one Python
process and abort during process-exit cleanup.

The fix does not intentionally leak runtime objects: client-owned MTP sessions,
samplers, contexts, models, and multimodal contexts are still released. The
process-global llama.cpp backend is not freed per client, and native CDLL handles
are cached with `RTLD_NODELETE` so the dynamic loader does not unload shared
llama.cpp globals while other loaded components can still reference them.

Residual risks:

- Heavy step/validate/target-decode trace retrieval is not retained. It needs a
  separate native lifetime/ownership audit before re-exposure.
- The vendored llama.cpp source is not a Git submodule, and the local vendor tree
  is not identical to current upstream `master`. It matches PR 23398 for several
  MTP/sampler files, while `common/speculative.cpp` carries additional
  Orbit/vendor changes. Future vendor refreshes should record the upstream commit
  explicitly and rerun the lifecycle harness.

## Post-PR 72 Validation Benchmark

After PR #72 was merged, a short validation benchmark was run against `main`
`b6ad0481b6302b1c79f04f1403aaa1b101df0a35`.

The temporary harness:

- used `ORBIT_MTP_TRACE=1`;
- loaded target-only and MTP-enabled native clients in one Python process;
- used repository models under `models/`;
- used `ctx=8192`, `threads=6`, `threads-batch=6`, `batch=256`,
  `ubatch=128`;
- ran three repetitions each for `response_short` and `response_medium`;
- collected target-only baseline, MTP first request, and MTP second identical
  request rows;
- wrote only metadata hashes, counts, and timings.

The harness produced a complete JSON result with 18 completion rows, then
aborted during teardown:

```text
double free or corruption (!prev)
exit code: 134
```

This reopens the lifetime/teardown blocker for mixed target-only/MTP benchmark
processes. The previous PR #72 linkage/lifetime changes remain necessary, but
they are not sufficient for this longer mixed-client benchmark.

Aggregated results before teardown:

| Prompt | Scenario | Runs | Wall ms avg | Refill ms avg | Spec loop ms avg | Target validate ms avg | Draft ms avg | Generation ms avg | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| response_short | target-only baseline | 3 | 9139.895 | n/a | n/a | n/a | n/a | 6588.005 | n/a |
| response_short | MTP first request | 3 | 7381.530 | 1394.750 | 5985.537 | 5193.330 | 595.129 | 7380.301 | 0.381 |
| response_short | MTP second identical | 3 | 7814.371 | 1588.383 | 6224.740 | 5253.133 | 608.663 | 7813.138 | 0.381 |
| response_medium | target-only baseline | 3 | 16906.423 | n/a | n/a | n/a | n/a | 13701.029 | n/a |
| response_medium | MTP first request | 3 | 12749.564 | 2256.660 | 10491.033 | 9153.290 | 1092.203 | 12747.715 | 0.462 |
| response_medium | MTP second identical | 3 | 12691.774 | 2162.367 | 10527.433 | 9043.203 | 1083.620 | 12689.795 | 0.462 |

Correctness observations from the completed rows:

- each scenario had stable output hashes across its three runs;
- MTP first and second identical requests had matching output-token-hash
  sequence hashes for each prompt;
- MTP first and second identical requests had matching first-sample hashes for
  each prompt;
- no per-completion MTP fallback occurred.

Stability verdict:

- FAIL for this pass because process teardown aborted after the benchmark.

Performance verdict:

- Inconclusive for defaults. MTP showed lower end-to-end wall time than
  target-only for the two prompts in this short run, but stability failure blocks
  any default-enablement conclusion.
- Target validate remains the dominant loop cost.
- The request-boundary target refill cost is material and should remain tracked.

Recommendation:

- keep MTP explicit/experimental;
- do not optimize acceptance, validate rows, or refill until the teardown abort
  is resolved;
- next patch candidate is a narrow lifetime reproducer and fix for the 18-row
  mixed target-only/MTP benchmark.

## Minimal Test And Benchmark Plan

### Local Unit Suite

Run the MTP-specific model-free tests first:

```bash
PYTHONPATH=src python3 -m unittest tests.test_native_mtp_experimental tests.test_native_persistent_mtp -q
PYTHONPATH=src python3 -m unittest tests.test_native_mtp_probe tests.test_native_mtp_dry_run tests.test_native_mtp_accept_probe tests.test_native_mtp_decode_probe -q
```

Then run the full safety net:

```bash
python3 -m unittest discover -s tests -q
python3 -m compileall -q src tests scripts
git diff --check
```

### Real Model Benchmark

Use the same target GGUF, MTP draft GGUF, prompt, and generation cap for all
runs.

Recommended fixed setup:

```text
ctx=8192
threads=6
threads-batch=6
batch=256
ubatch=128
CPU affinity: taskset -c 0-5
thinking=off
tools disabled for the generation-only comparison
```

Compare:

- Orbit native target-only;
- Orbit native `--mtp`;
- reference `llama-server` MTP.

Collect:

- normalized output equivalence;
- total wall time;
- `generation_ms`;
- output token count;
- validate count;
- validate `n_tok` sequence;
- target decode calls;
- draft decode calls;
- full/live/restored/replay resolution counts;
- accepted/rejected draft totals;
- phase timing totals;
- checkpoint/restore counts;
- `seq_rm` counts.

### Focused A/B Checks

- Default boundary split vs `ORBIT_MTP_BOUNDARY_SPLIT=0`.
- `ORBIT_MTP_PARTIAL_DEBUG=1` for frontier transitions.
- `ORBIT_MTP_VALIDATE_DEBUG=1` for validate shapes.
- `ORBIT_MTP_DRAFT_TRACE=1` for draft positions and token counts.

### Correctness Gates

- Output must remain equivalent to target-only greedy reference.
- Streamed content must match final content after filtering.
- Cancel/interruption must not corrupt a later request.
- Fallback after MTP failure must still use standard generation.
- Tools, routing, final policy, evidence policy, and prompts must remain
  untouched.

## MTP-030 Follow-Up: Mixed Teardown Root Cause And Fix

The mixed target-only/MTP teardown abort was reduced with a temporary
process-per-scenario harness.

Minimal reproducer:

```text
target-only completion -> MTP completion -> explicit client cleanup
ORBIT_MTP_TRACE=0
mmproj disabled
```

Reduction matrix:

| Scenario | Runtime path | Trace | Result |
| --- | --- | --- | --- |
| target-only only, repeated | split runtime | off | PASS |
| MTP first only, repeated | split runtime | off | PASS |
| MTP first+second, repeated | split runtime | off | PASS |
| target-only then MTP | split runtime | off | FAIL, exit 134 |
| target-only then MTP | split runtime | on | FAIL, exit 134 |
| target-only then MTP | unified runtime bin | off | PASS, 3/3 |

Root cause:

- `resolve_paths()` selected `src/orbit/native_llama/vendor/lib` for the
  target-only Python client runtime.
- The persistent MTP shim linked against
  `src/orbit/native_llama/vendor/build/llama.cpp/bin` because that directory
  contains the required SONAME entries (`libllama.so.0`,
  `libllama-common.so.0`, and ggml SONAMEs).
- A mixed process could therefore load two llama.cpp runtime instances with
  separate process-wide backend/global state.
- Teardown after using both paths could double-free or corrupt shared global
  state.

Fix:

- Native runtime resolution now prefers the packaged vendor build bin when it
  exposes the SONAME runtime needed by the MTP shim.
- `vendor/lib` remains the fallback when no packaged SONAME runtime is
  available.
- This keeps target-only and MTP shim usage on the same llama.cpp runtime
  path in the local packaged build.

Post-fix reproduction:

```text
target-only completion -> MTP completion -> explicit client cleanup
ORBIT_MTP_TRACE=0
mmproj disabled
3/3 runs exited 0
```

Post-fix 18-row benchmark:

| Prompt | Scenario | Runs | Exit codes | Wall ms avg | Refill ms avg | Spec loop ms avg | Target validate ms avg | Draft ms avg | Generation ms avg | Acceptance |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| response_short | target-only baseline | 3 | 0 | 6778.707 | n/a | n/a | n/a | n/a | 5044.464 | n/a |
| response_short | MTP first request | 3 | 0 | 5215.916 | 1073.164 | 4141.823 | 3566.127 | 446.265 | 5214.993 | 0.381 |
| response_short | MTP second identical | 3 | 0 | 5255.835 | 1061.765 | 4193.210 | 3510.967 | 446.878 | 5254.986 | 0.381 |
| response_medium | target-only baseline | 3 | 0 | 12038.292 | n/a | n/a | n/a | n/a | 9767.841 | n/a |
| response_medium | MTP first request | 3 | 0 | 8914.028 | 1502.527 | 7409.923 | 6435.117 | 807.487 | 8912.465 | 0.462 |
| response_medium | MTP second identical | 3 | 0 | 9032.386 | 1536.167 | 7495.057 | 6421.367 | 802.408 | 9031.243 | 0.462 |

Correctness/stability observations after the fix:

- MTP first and second identical requests kept matching output-token-hash
  sequence hashes for each prompt.
- MTP first and second identical requests kept matching first-sample hashes for
  each prompt.
- No per-completion MTP fallback occurred.
- The full mixed 18-row benchmark exited 0.

Status:

- Stability blocker resolved for the current text-generation mixed benchmark.
- Performance remains scenario-dependent and should not be generalized from
  this short run.
