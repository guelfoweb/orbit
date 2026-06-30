# MTP Validation And Benchmark Plan

Status: first validation pass after PR #72. This document is a test and
benchmark matrix only; it does not change runtime behavior.

## Scope

The goal is to prove native MTP correctness and stability before treating it as
a performance feature.

Do not use this plan to justify prompt, routing, tool, final, evidence, vendor,
or speculative optimization changes. If a benchmark exposes a correctness or
stability failure, stop and diagnose before patching.

## Metrics

Separate every run into three classes of metrics.

Correctness metrics:

- exit code
- fallback count and reason
- generated token count
- output hash or output-token-hash sequence hash
- first-sample hash when `ORBIT_MTP_TRACE=1`
- stop/cancel status
- streaming final-content equivalence when streaming is used

Stability metrics:

- process exit status
- native aborts or teardown failures
- MTP fallback reason
- restore used
- session reuse state
- repeated-run consistency across at least three runs where practical

Performance metrics:

- wall ms
- prefill ms
- request-boundary target refill ms
- speculative loop ms
- target validate ms
- draft generation ms
- generation ms
- tokens/sec
- accepted total
- draft total
- acceptance ratio

## Scenario Matrix

| Scenario | Purpose | Required setup | Minimum checks |
| --- | --- | --- | --- |
| target-only baseline | Compare correctness and end-to-end latency without MTP. | `use_mtp_experimental=False`, same prompt and cap. | output hash, token count, wall/prefill/generation ms. |
| MTP first request | Validate cold MTP request behavior. | `use_mtp_experimental=True`, reset session before run. | fallback=0, output hash, first sample hash, phase timing. |
| MTP second identical request | Validate request-boundary restore and refill path. | Same MTP client, same prompt immediately repeated. | `restore_used=true`, first/output hashes stable, refill cost measured. |
| MTP + KV/session reuse | Verify MTP does not corrupt session cache or later requests. | Repeat same prompt and then a different prompt on one client. | no fallback, no stale output, stable process exit. |
| MTP + tools on default | Verify runtime integration when tools are available. | Native server default tools-on mode. | tool-call rounds must not use MTP; final generation may use MTP only when eligible. |
| MTP + `ORBIT_KV_PREFIX_PREWARM=startup` | Verify startup prewarm does not change MTP correctness. | Native server with default/explicit startup prewarm. | prewarm succeeds or skips safely; MTP output remains stable. |
| MTP + `ORBIT_KV_PREFIX_PREWARM=off` | Verify prewarm-off baseline. | Native server with prewarm disabled. | no startup prewarm; MTP behavior unchanged after startup. |
| MTP + `ORBIT_TOOLS=off` | Verify no-tools server path. | Native server started with tools disabled. | no route tools-on prewarm; generation MTP behavior unchanged. |
| MTP + streaming | Verify token callback and final content alignment. | `/chat/stream` or native callback path. | streamed visible content hash matches final content hash. |
| MTP + cancel during generation | Verify cancellation does not corrupt later requests. | Start generation, cancel mid-run, then run a normal request. | cancel status, no crash, next request succeeds. |
| MTP + stop sequence | Verify stop filter does not diverge from final text. | Completion with explicit stop sequence. | stopped status, final output hash, no hidden extra content. |
| MTP + mmproj/multimodal | Verify linkage/lifetime with multimodal context loaded. | Target model plus mmproj present. | load/complete/close exits 0; MTP fallback or success is explicit. |

## First Short Benchmark

Command used: a temporary metadata-only benchmark harness outside the repository
with `ORBIT_MTP_TRACE=1`.

The temporary harness:

- loads the target model once;
- loads a separate MTP-enabled client once;
- uses `ctx=8192`, `threads=6`, `threads-batch=6`, `batch=256`,
  `ubatch=128`;
- runs three repetitions for `response_short` and `response_medium`;
- resets session state before each target-only and MTP-first repetition;
- runs an identical second MTP request immediately after each MTP-first request;
- records metadata only.

The harness wrote a complete JSON result file with 18 completion rows, then the
process aborted during teardown:

```text
double free or corruption (!prev)
exit code: 134
```

Because the process did not exit cleanly, the stability verdict for this pass is
FAIL even though per-completion metadata was written.

## First Short Benchmark Results

All numbers are averages across three runs.

| Prompt | Scenario | Wall ms | Prefill ms | Refill ms | Spec loop ms | Target validate ms | Draft ms | Generation ms | Tokens/sec | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| response_short | target-only baseline | 9139.895 | 2550.368 | n/a | n/a | n/a | n/a | 6588.005 | 2.432 | n/a |
| response_short | MTP first request | 7381.530 | 0.000 | 1394.750 | 5985.537 | 5193.330 | 595.129 | 7380.301 | 2.169 | 0.381 |
| response_short | MTP second identical | 7814.371 | 0.000 | 1588.383 | 6224.740 | 5253.133 | 608.663 | 7813.138 | 2.050 | 0.381 |
| response_medium | target-only baseline | 16906.423 | 3204.657 | n/a | n/a | n/a | n/a | 13701.029 | 2.336 | n/a |
| response_medium | MTP first request | 12749.564 | 0.000 | 2256.660 | 10491.033 | 9153.290 | 1092.203 | 12747.715 | 2.511 | 0.462 |
| response_medium | MTP second identical | 12691.774 | 0.000 | 2162.367 | 10527.433 | 9043.203 | 1083.620 | 12689.795 | 2.522 | 0.462 |

Correctness observations:

- Each scenario produced stable output hashes within its own three-run group.
- MTP first and MTP second produced matching output-token-hash sequence hashes
  for both prompts.
- MTP first and MTP second produced matching first-sample hashes for both
  prompts.
- No per-completion MTP fallback occurred.

Stability observation:

- The process aborted at teardown after writing all benchmark rows.
- This is a blocker for enabling MTP beyond explicit experimental use.

Performance observations:

- For `response_short`, MTP end-to-end wall time improved over target-only
  baseline, but MTP generation tokens/sec was lower and stability failed.
- For `response_medium`, MTP end-to-end wall time improved over target-only
  baseline in this run.
- Target validate remains the dominant MTP loop cost.
- The request-boundary refill cost was material: about `1.4-1.6s` for
  `response_short` and `2.1-2.3s` for `response_medium`.

## Verdict For This Pass

Correctness verdict: PASS for the limited text-generation rows collected.

Stability verdict: FAIL due to teardown abort after the repeated mixed
target-only/MTP benchmark.

Performance verdict: inconclusive for product defaults. MTP improved end-to-end
wall time in this short run, but the process abort prevents treating the result
as production-ready performance evidence.

## Recommendation

Keep MTP env/flag-gated and experimental.

Do not enable MTP by default.

Next patch candidate: isolate and fix the teardown double-free reproduced by the
18-row mixed target-only/MTP benchmark. Do not optimize refill, acceptance, or
validate rows until this stability blocker is resolved.

## Follow-Up After Runtime Path Unification

The teardown crash was traced to split native runtime loading:

- target-only Python client resolved llama.cpp from `vendor/lib`;
- the persistent MTP shim linked to the SONAME runtime in
  `vendor/build/llama.cpp/bin`;
- mixed target-only/MTP use in one process could load two llama.cpp runtime
  instances and corrupt process-wide teardown state.

After changing native runtime resolution to prefer the packaged SONAME build
bin when available, the minimal reducer passed:

```text
target-only completion -> MTP completion -> explicit client cleanup
ORBIT_MTP_TRACE=0
mmproj disabled
3/3 runs exited 0
```

The same 18-row short benchmark was repeated. The process exited 0.

| Prompt | Scenario | Wall ms | Prefill ms | Refill ms | Spec loop ms | Target validate ms | Draft ms | Generation ms | Tokens/sec | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| response_short | target-only baseline | 6778.707 | 1733.729 | n/a | n/a | n/a | n/a | 5044.464 | 3.172 | n/a |
| response_short | MTP first request | 5215.916 | 0.000 | 1073.164 | 4141.823 | 3566.127 | 446.265 | 5214.993 | 3.079 | 0.381 |
| response_short | MTP second identical | 5255.835 | 0.000 | 1061.765 | 4193.210 | 3510.967 | 446.878 | 5254.986 | 3.057 | 0.381 |
| response_medium | target-only baseline | 12038.292 | 2269.837 | n/a | n/a | n/a | n/a | 9767.841 | 3.276 | n/a |
| response_medium | MTP first request | 8914.028 | 0.000 | 1502.527 | 7409.923 | 6435.117 | 807.487 | 8912.465 | 3.591 | 0.462 |
| response_medium | MTP second identical | 9032.386 | 0.000 | 1536.167 | 7495.057 | 6421.367 | 802.408 | 9031.243 | 3.543 | 0.462 |

Post-fix verdict for this short text-generation matrix:

- Correctness: PASS for hash-stable MTP first/second rows.
- Stability: PASS for the 18-row mixed benchmark, exit 0.
- Performance: positive signal for these two prompts, still insufficient for a
  default-enablement decision.

Remaining validation before broader MTP use:

- streaming;
- cancel;
- stop sequence;
- session reuse with tools and route-prefix prewarm;
- multimodal/mmproj generation path.
