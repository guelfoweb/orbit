# MTP State Machine Analysis

Status: analysis-only. No runtime, prompt, routing, tool-selection, final-policy,
evidence-policy, KV, streaming, or MTP code changes are included here.

## Repository State

- Branch inspected: `main`
- HEAD inspected: `a10413cab2f26e32acf21068e5345bcdf82a70e6`
- `origin/main`: `a10413cab2f26e32acf21068e5345bcdf82a70e6`
- Release tag present on HEAD: `v0.0.1-rc4`
- Working tree before this analysis: no tracked diffs; only `workdir/.miktex/`
  was untracked and was not touched.

## Relevant File Map

### Runtime Entrypoints

- `src/orbit/native_server/app.py`
  - exposes `--mtp` / `--enable-mtp-experimental`
  - constructs `NativeClientConfig(use_mtp_experimental=...)`
  - exposes MTP state in `/props`
  - routes `/chat`, `/chat/stream`, continuation, cancel, and health requests
- `src/orbit/native_server/protocol.py`
  - packages native usage and timing fields, including `generation_ms`

### Native Python Client

- `src/orbit/native_llama/client.py`
  - owns the target model/context/session state
  - initializes probe helpers and the persistent MTP runtime
  - decides whether a completion may attempt MTP
  - falls back to standard generation when MTP is disabled, unavailable, failed,
    cancelled, or blocked by thinking/tool-call conditions
  - maps successful MTP completion into `NativeTimings`
- `src/orbit/native_llama/session_state.py`
  - exposes session snapshot fields for `mtp_enabled`, `mtp_initialized`,
    `mtp_failure_reason`, cached token count, in-flight state, and cancellation

### Python MTP Wrappers

- `src/orbit/native_llama/persistent_mtp.py`
  - loads or builds `liborbit-persistent-mtp.so`
  - creates, resets, frees, and calls a persistent target/draft MTP session
  - maps C shim counters into `MtpCompletionResult`
- `src/orbit/native_llama/mtp_completion.py`
  - one-shot helper wrapper around `orbit-mtp-completion`
- `src/orbit/native_llama/mtp_probe.py`
  - load/init probe wrapper; no generation
- `src/orbit/native_llama/mtp_dry_run.py`
  - draft-generation dry-run wrapper; no accept loop
- `src/orbit/native_llama/mtp_accept_probe.py`
  - single accept-loop probe wrapper
- `src/orbit/native_llama/mtp_decode_probe.py`
  - decode-loop probe wrapper over a fixed prompt set

### Native Shim Sources

- `src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp`
  - persistent target/draft speculative decoding implementation
  - owns draft generation, target validation, acceptance, partial restore,
    live partial commit, replay fallback, checkpointing, `seq_rm`, and tracing
- `src/orbit/native_llama/vendor/shim/orbit_mtp_completion.cpp`
  - one-shot MTP completion helper
- `src/orbit/native_llama/vendor/shim/orbit_mtp_probe.cpp`
  - target/draft load/init probe
- `src/orbit/native_llama/vendor/shim/orbit_mtp_dry_run.cpp`
  - draft generation dry-run
- `src/orbit/native_llama/vendor/shim/orbit_mtp_accept_probe.cpp`
  - accept-loop probe
- `src/orbit/native_llama/vendor/shim/orbit_mtp_decode_probe.cpp`
  - decode-loop probe
- `src/orbit/native_llama/vendor/shim/orbit_mtp_step_trace.cpp`
  - trace-oriented helper

### Tests

- `tests/test_native_mtp_experimental.py`
  - parser flag, fallback paths, thinking-mode skip, streaming callback behavior,
    cached prompt token accounting, tool-call-round MTP suppression
- `tests/test_native_persistent_mtp.py`
  - persistent shim discovery, stale shim rebuild, init/reset/free behavior, and
    wrapper result extraction
- `tests/test_native_mtp_probe.py`
  - load/init probe wrapper
- `tests/test_native_mtp_dry_run.py`
  - draft dry-run wrapper
- `tests/test_native_mtp_accept_probe.py`
  - accept probe wrapper
- `tests/test_native_mtp_decode_probe.py`
  - decode probe wrapper
- `tests/test_native_session_state.py`
  - MTP state visibility in session snapshots
- `tests/test_native_thinking.py`
  - MTP interaction with visible-thinking and continuation paths

### Docs And Scripts

- `README.md`
  - documents `orbit server --mtp` as explicit experimental MTP mode
- `docs/PERFORMANCE.md`
  - describes MTP as generation-oriented and benchmark-sensitive on CPU
- `docs/NATIVE_PACKAGING_ROADMAP.md`
  - lists MTP shims as native packaging artifacts
- `scripts/suggest-server-profile.sh`
  - mentions `orbit server --mtp`

## Upstream llama.cpp Check

Checked on 2026-06-30 against `ggml-org/llama.cpp`.

Relevant upstream facts:

- Upstream `master` has merged Gemma 4 MTP support through PR #23398,
  `llama : add Gemma4 MTP`, merged on 2026-06-07.
- Upstream `docs/speculative.md` documents `draft-mtp` as a supported
  speculative decoding type.
- The following upstream branches exist and are relevant comparison candidates:
  - `gg/spec-mtp-experiments`
  - `gg/spec-ckpt-test`
  - `gg/server-fix-spec`
  - `gg/server-fix-spec-ctx-shift`
  - `gg/server-reenable-swa-spec`
  - `gg/spec-refactor-ctx`
- GitHub compare metadata shows those `gg/spec-*` branches are currently
  diverged from `master`, not cleanly newer replacements for `master`.

Implication for Orbit:

- MTP compatibility itself is no longer only an out-of-tree idea upstream;
  `draft-mtp` exists on `llama.cpp` `master`.
- There is not yet evidence from this pass that upstream has a merged fix for
  Orbit's specific extra small validate (`n_tok=2`) or the residual latency.
- The next safe step is source comparison, not blind porting:
  - compare Orbit's `orbit_persistent_mtp.cpp` against upstream
    `common/speculative` / server speculative paths;
  - inspect `gg/spec-mtp-experiments` first for MTP-specific changes;
  - inspect `gg/spec-ckpt-test` and `gg/spec-refactor-ctx` for checkpoint,
    rollback, and context-lifecycle changes;
  - inspect `gg/server-fix-spec*` only for server integration behavior.

Sources:

- https://github.com/ggml-org/llama.cpp/pull/23398
- https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- https://github.com/ggml-org/llama.cpp/tree/gg/spec-mtp-experiments
- https://github.com/ggml-org/llama.cpp/tree/gg/spec-ckpt-test
- https://github.com/ggml-org/llama.cpp/tree/gg/server-fix-spec
- https://github.com/ggml-org/llama.cpp/tree/gg/spec-refactor-ctx

## Activation And Top-Level Flow

1. `orbit server --mtp` sets `NativeClientConfig.use_mtp_experimental=True`.
2. `NativeLlamaClient.load()` initializes the target model/context and then calls
   `_initialize_persistent_mtp_session()`.
3. `_initialize_persistent_mtp_session()`:
   - exits if MTP is not enabled;
   - exits with a fallback reason if the draft MTP model is unavailable;
   - exits with failure if the target context is missing;
   - otherwise creates a persistent MTP session with target context, draft
     context, context size, batch size, ubatch size, and thread settings.
4. During `complete_prompt(...)`, Orbit attempts MTP only when:
   - `allow_mtp_experimental=True`;
   - thinking mode is off;
   - no immediate cancellation is requested;
   - `use_mtp_experimental=True`;
   - draft artifacts are available;
   - the persistent runtime is initialized and still marked enabled.
5. If MTP returns success, the result becomes the request's `NativeTimings`.
6. If MTP returns failure or is not eligible, Orbit falls back to standard target
   generation.

Important guardrail: `complete_chat(...)` disables MTP for tool-call rounds by
passing `allow_mtp_experimental=not tools`. Final-from-tool history can still use
MTP when no new tool call is being requested.

## Persistent MTP Session State

The C++ shim stores a persistent `orbit_mtp_session` with:

- draft model and draft context;
- shared speculative state (`common_speculative`);
- target/draft request-boundary checkpoints;
- target prompt tokens for boundary restore eligibility;
- cached prompt tokens;
- last completion counters;
- acceptance, replay, restore, checkpoint, and `seq_rm` counters;
- phase timing aggregates;
- optional debug trace JSON strings.

The original Python result object read the main counters and aggregate timings,
but did not expose metadata needed to compare prompt/logits/sample equivalence.

The consolidated patch keeps a narrow env-gated metadata bridge for
`ORBIT_MTP_TRACE=1`: phase timing JSON, generated output token hashes, and
first-sample prompt/frontier/logits/sample hashes. Heavy step, validate, and
target-decode JSON strings are not exposed because retrieving them reproduced a
process-exit teardown crash in the lifecycle harness. This bridge is
diagnostics-only. It does not change prompt rendering, routing, tool selection,
final policy, evidence policy, decode, sampling, KV mutation, boundary split,
replay, or acceptance behavior.

## Completion State Machine

### State 0. Reset Per-Request State

`orbit_mtp_session_complete(...)` clears per-request counters, traces, timing
aggregates, output content, acceptance totals, and replay state. It then
tokenizes the rendered prompt and initializes a deterministic reference sampler.

Generation is capped in the Python wrapper to `max(1, min(max_tokens, 32))`.
The C++ shim also uses that effective cap in the loop.

### State 1. Replay Or Restore Prompt Frontier

The loop starts with `need_replay=true`.

On replay:

- target and draft memories are cleared;
- if an existing request-boundary checkpoint is compatible with the current
  prompt prefix, it is restored into both target and draft contexts;
- otherwise the prompt is decoded on the target in chunks;
- the same prompt chunk stream is processed through `common_speculative_process`
  so the speculative state sees the target prompt;
- on the first generated token of a request, a request-boundary checkpoint is
  captured for target and draft contexts.

If this is the first output token, the target sampler samples one token from the
prefilled target context, accepts it into sampler state, emits it, and then the
MTP loop proceeds.

### State 2. Draft Generation

When no residual draft is available, the shim:

1. checkpoints the current target/draft frontier;
2. sets `common_speculative_get_draft_params(...)` with:
   - enabled `true`;
   - `n_max=min(3, remaining_generation_cap)`;
   - `n_past`;
   - `id_last`;
   - pointers to the target prompt frontier and draft buffer;
3. calls `common_speculative_draft(...)`;
4. records one draft decode call;
5. records the number of fresh draft tokens;
6. restores and trims the draft context back to the checkpoint frontier using
   `llama_memory_seq_rm(...)`;
7. marks the checkpoint as available for partial handling.

The hard-coded native draft maximum is currently `ORBIT_MTP_DRAFT_N_MAX = 3`.

### State 3. Build Validate Batch

The validate batch is always:

```text
[id_last] + draft_tokens
```

with `validate_pos0 = n_past`.

The validate batch requests logits for every row. The shim records batch size,
logits row count, target KV min/max before and after decode, and decode timing.

### State 4. Optional Boundary Split / Live Logical Commit

If `ORBIT_MTP_BOUNDARY_SPLIT` is unset or non-zero, and the checkpoint/frontier
preconditions are satisfied, the shim tentatively appends `id_last + draft` to
`prompt_tgt` before validation. This is the `boundary_committed_live` path.

The intended effect is to let a partial accept commit the accepted frontier
without replaying the whole target/draft prompt.

Preconditions include:

- checkpoint exists;
- draft is non-empty;
- `n_past == prompt_tgt.size()`;
- target and draft KV max positions match `n_past - 1`.

### State 5. Target Validate

`resolve_validate_accept_restore(...)`:

1. builds a target validate batch for `[id_last] + draft`;
2. runs `llama_decode(ctx_tgt, validate)`;
3. increments target decode counters;
4. runs `common_speculative_process(...)` against the validate batch;
5. optionally clones the sampler if a checkpoint is available;
6. calls `common_sampler_sample_and_accept_n(...)` over all validate rows and the
   draft vector.

The resulting `ids` represent accepted output ids. The number of accepted draft
tokens is `ids.size() - 1`.

### State 6. Resolution

There are four non-error resolutions:

- `full_accept`
  - all draft tokens are accepted;
  - `common_speculative_accept(...)` commits the accepted draft span;
  - accepted ids are emitted;
  - target and draft KV tails beyond the new frontier are removed with
    `llama_memory_seq_rm(...)`;
  - draft/checkpoint state is cleared.
- `live_partial`
  - boundary split is active and partial commit is safe;
  - accepted ids are emitted;
  - target and draft KV are trimmed to the accepted frontier;
  - no replay is required on the next iteration;
  - residual draft is cleared and the next draft is marked fresh.
- `restored_partial`
  - sampler and target/draft contexts are restored to the checkpoint;
  - `draft = ids`, so the accepted ids become a residual draft for the next
    iteration;
  - no tokens are emitted in that iteration;
  - the next loop reuses the residual draft.
- `replay_fallback`
  - used when partial rollback/restore cannot be safely completed;
  - any tentative boundary split is reverted;
  - accepted ids are emitted;
  - next iteration enters prompt replay.

An error resolution stops MTP and returns failure to Python; the Python client
then disables MTP for that session and falls back to standard generation.

### State 7. Emission

For all emitting resolutions, each accepted id is:

1. checked for EOG;
2. appended to the generated token vector;
3. converted to text with `llama_token_to_piece(...)`;
4. appended to `last_content`;
5. passed to the token callback when non-empty.

The Python MTP path strips control-channel output after the C++ completion when
thinking is off. If the C++ path already streamed tokens, Python avoids
double-emitting the final string.

### State 8. Finish

The loop exits when:

- max generation cap is reached;
- EOG is sampled;
- an error occurs.

At finish, the shim stores output content, token counts, acceptance ratios,
decode counts, elapsed time, tokens/sec, output token hashes, first-sample
metadata, and phase timing JSON. The heavy step/validate/target-decode trace
strings are not part of the retained Python-facing diagnostics.

## Diagnostics And Trace Surfaces

Available C++ environment-gated diagnostics:

- `ORBIT_MTP_PARTIAL_DEBUG`
  - frontier and partial state traces
- `ORBIT_MTP_VALIDATE_DEBUG`
  - validate pre/decode/result/replay traces
- `ORBIT_MTP_DRAFT_TRACE`
  - draft token and draft-context traces
- `ORBIT_MTP_BOUNDARY_SPLIT=0`
  - disables boundary split for comparison

Available counters returned through Python today:

- `draft_tokens_total`
- `accepted_tokens_total`
- `rejected_tokens_total`
- reused draft/accepted/rejected totals
- `acceptance_ratio`
- `fresh_acceptance_ratio`
- `consumed_acceptance_ratio`
- `target_decode_calls`
- `draft_decode_calls`
- `elapsed_ms`
- `tokens_per_second`
- `full_accept_steps`
- `replay_steps`
- `partial_accept_steps`
- `partial_no_replay_steps`
- `replay_fallback_steps`
- `seq_rm_supported`
- `rollback_tokens_total`
- `checkpoint_count`
- `restore_count`

Python-facing diagnostics now available when `ORBIT_MTP_TRACE=1`:

- phase timing JSON;
- generated output token hashes;
- first-sample prompt hash/count;
- target frontier hash/range;
- final prompt logits hash;
- first-sample hash;
- request-boundary restore/refill markers.

These fields remain unset by default.

## Known Observations To Revalidate

The investigation starts from these external observations:

- Orbit MTP on `response_medium` produced equivalent output but was slower than
  the reference `llama-server` MTP run.
- Reference `llama-server` MTP: about `6.26s generation_ms`, `16` validate
  calls.
- Orbit MTP: about `10.16s generation_ms`, `17` validate calls.
- Orbit appears to perform one extra small validate with `n_tok=2`.
- The residual latency is not explained by that extra validate alone.
- Boundary split and live partial propagation looked promising but need
  revalidation with current `main`.

## State-Machine Invariants

Any future MTP patch should preserve:

- output equivalence with the target-only greedy reference for the same rendered
  prompt, sampler, stop conditions, and max tokens;
- no changes to user prompts, route prompts, tool policy, final policy, or
  evidence policy;
- no model-facing deterministic semantic fast paths;
- no unvalidated draft token emitted as final user-visible content;
- no duplicated or missing streamed token fragments;
- cancellation must not leave target/draft KV or sampler state marked valid;
- fallback must return to standard target generation when MTP fails;
- tool-call rounds remain non-MTP unless explicitly redesigned and benchmarked;
- thinking mode remains non-MTP unless explicitly redesigned and tested.

## Candidate Fix Points For Later Patches

No fix is applied in this analysis. Candidate investigation points are:

1. Redesign heavy step/validate/target-decode diagnostics so ownership and
   teardown are stable before exposing them through Python.
2. Compare the Orbit validate sequence with `llama-server` for the same prompt:
   per-step `n_tok`, accepted count, rejected count, full/partial/replay
   resolution, target decode time, draft decode time, `seq_rm` count, and
   checkpoint/restore count.
3. Determine whether the extra `n_tok=2` validate is:
   - required for correctness;
   - caused by boundary split/live partial residual propagation;
   - caused by residual draft reuse after partial restore;
   - caused by final cap/EOG handling;
   - an avoidable duplicate validate.
4. Attribute the unexplained latency to target validate, draft generation,
   checkpoint/restore, `seq_rm`, sampler clone/restore, replay, detokenization,
   or Python/C callback overhead.
5. Revalidate `ORBIT_MTP_BOUNDARY_SPLIT=0` vs default boundary split on the same
   fixed CPU setup.
6. Revalidate live partial against restored partial and replay fallback paths,
   especially after partial accept.
7. Add a benchmark harness that compares Orbit no-MTP, Orbit MTP, and
   `llama-server` MTP with identical prompt, model artifacts, CPU affinity, and
   thread/batch settings.

## Minimal Test And Benchmark Plan

Static/unit checks:

```bash
PYTHONPATH=src python3 -m unittest tests.test_native_mtp_experimental tests.test_native_persistent_mtp -q
PYTHONPATH=src python3 -m unittest tests.test_native_mtp_probe tests.test_native_mtp_dry_run tests.test_native_mtp_accept_probe tests.test_native_mtp_decode_probe -q
python3 -m unittest discover -s tests -q
python3 -m compileall -q src tests scripts
git diff --check
```

Real MTP benchmark matrix:

- fixed setup: Gemma 4 12B target, matching MTP draft, CPU-only, fixed affinity
  such as `taskset -c 0-5`;
- same `ctx=8192`, `threads=6`, `threads-batch=6`, `batch=256`,
  `ubatch=128`;
- same rendered prompt and max token cap;
- compare:
  - Orbit target-only native generation;
  - Orbit `--mtp`;
  - reference `llama-server` MTP;
- record:
  - normalized output equivalence;
  - `generation_ms`;
  - output tokens;
  - target decode calls;
  - draft decode calls;
  - validate count and per-validate `n_tok`;
  - accepted/rejected draft counts;
  - full/live/restored/replay resolutions;
  - checkpoint/restore count;
  - `seq_rm` count;
  - phase timing totals.

Streaming/cancel checks:

- stream an MTP response and confirm no duplicated or missing fragments;
- interrupt/cancel during draft, validate, and after partial accept if possible;
- verify a later non-MTP request still works and does not reuse corrupt state.
