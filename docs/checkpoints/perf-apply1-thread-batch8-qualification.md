# PERF-APPLY-1 — `threads_batch=8` cross-workload qualification

Qualification only. **No production source, test or default changed.**

## Result: REJECTED

`threads_batch = 8` **fails qualification**. It delivers the promised cold-prefill
gain but costs materially more on generation and short prompts — the workloads a
user meets most often.

| Property | Value |
|---|---|
| Repository baseline | `7d2e2c6c4181c6655c1a7bced02081f52182f7e8` |
| **Executable/source baseline** | **`aaf544013a1f398e5587c710316a56ccd4d64e7d`** |
| Classification | **C — REJECTED_CROSS_WORKLOAD_REGRESSION** |

`git diff aaf5440..HEAD -- src/ tests/` = 0 files: every commit since is
docs-only, so the measured runtime is byte-identical to PERF-OPT-2's.

## Protocol

| Property | Value |
|---|---|
| Model | `Ornith-1.5-35B-Q4_K_M.gguf`, 21,713,462,848 bytes |
| SHA-256 | `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f` |
| Backend | llama.cpp b9551 (`379ac6673b5cd75c7b4e07d1521c50f1e093878c`) |
| Control | `threads=6`, `threads_batch=6` |
| Candidate | `threads=6`, `threads_batch=8` |
| Frozen | ctx 8192, batch 256, ubatch 128, temp 0, think off, 1 slot, MTP off |
| Long prompt | 18,680 bytes, SHA-256 `3ceb97abe7c54764bbc5bfca86c8c89e017e550c619f7871649f5b305b7bd056` |

Workload order was frozen before inference and identical in every load: warm-up
→ short → generation → **long** → 3-turn reuse → 8-turn session → final capture,
with a session reset before each cold workload.

Every run asserted `threads`, `threads_batch`, ctx/batch/ubatch/slots, thinking
mode, all three MTP flags and `cached_tokens == 0` from `/props` before
proceeding. **No parameter drift**; all three runs printed the identical prompt
SHA and `prompt_tokens = 3716` with `reused = 0`.

## Model loads

Three: control, candidate, and one candidate re-test. The third is the §3
allowance for a run suspected of host contention — see "The contention
hypothesis was wrong" below.

## Gate results

| Gate | Requirement | Measured | Result |
|---|---|---|---|
| **A** long cold prefill | ≥ +5 % | **+6.79 %** (mean of 2) | **PASS** |
| **B** short prompt | < 3 % regression | **+18.77 % slower** (best of 2) | **FAIL** |
| **C** generation | < 3 % regression | **−12.24 %** (best of 2) | **FAIL** |
| **D** KV reuse | identical semantics | identical, all turns | PASS |
| **E** bounded session | no instability | reuse identical, RSS flat | PASS |
| **F** peak RSS | < 3 % regression | +0.0068 % | PASS |
| **G** thermals | no pathology | control 92–97 °C, candidates 89–97 °C (limit 100) | PASS |
| **H** reliability | no errors/drift | 0 errors | PASS |

Thresholds were fixed before the runs and are applied exactly as written.

## Gate A — the gain is real

| Run | Prefill tok/s | `prompt_ms` | TTFT ms | Cores | Gain |
|---|---:|---:|---:|---:|---:|
| control (tb=6) | 26.109 | 142,326.774 | 142,384.6 | 5.982 | — |
| candidate run 1 | 27.826 | 133,542.297 | 133,600.9 | 6.773 | +6.58 % |
| candidate run 2 | 27.936 | 133,015.988 | 133,071.8 | 6.763 | +7.00 % |

Repeatable to 0.396 %. The candidate genuinely receives its extra batch threads
(6.77 cores vs 5.98) and converts them, at +6.08 % CPU (mean of the two runs). This reproduces
PERF-OPT-2's finding.

Absolute rates are lower than PERF-OPT-2's (26.1 vs 29.1 control) because this
suite runs sustained work and the CPU sat at 92–97 °C throughout, versus 66→81 °C
for PERF-OPT-2's single request. That affects both arms equally and the A/B
denominator is the same-mission control, so the *gain* remains valid.

## Gates B and C — the cost is also real

| Workload | Control | Cand run 1 | Cand run 2 |
|---|---:|---:|---:|
| short prompt wall | 3.692 s | 4.385 s (+18.8 %) | 5.065 s (+37.2 %) |
| generation fixture tok/s | 10.448 | 8.215 (−21.4 %) | 9.169 (−12.2 %) |
| mean gen delta, 13 workloads | — | −5.11 % | −8.98 % |

Both candidate runs regress on both gates. Even the most favourable candidate
number, −12.24 %, is four times the 3 % allowance.

### Gates B and C are one effect, not two

Decomposing the short prompt shows the regression is dominated by generation
time, not prefill:

| Run | `prompt_ms` | `predicted_ms` |
|---|---:|---:|
| cand 1 | 662 → 737 (+11.2 %) | 2,993 → 3,610 (**+20.6 %**) |
| cand 2 | 662 → 843 (+27.2 %) | 2,993 → 4,182 (**+39.7 %**) |

So B and C are the same underlying slowdown observed through two fixtures. This
document does **not** treat them as independent corroborating failures; the
rejection rests on one reproduced effect, measured two ways.

### The spread is wide, and that is stated plainly

The per-workload generation deltas swing in both directions, and the two
candidate runs disagree with each other:

| Statistic | Cand run 1 | Cand run 2 |
|---|---:|---:|
| dedicated generation fixture | −21.37 % | −12.24 % |
| mean across 13 workloads | −5.11 % | −8.98 % |
| **8-turn session only** | **+0.34 %** | **−6.41 %** |

Run 1's session mean is essentially neutral. Taken alone it would not fail the
gate — and that is reported here rather than omitted. What survives every
framing is the direction and the core count: no statistic in either run shows
the candidate *faster* on generation overall, and both runs independently land
at ~5.41 utilized cores against the control's ~5.91.

The short-prompt and dedicated-generation fixtures are also each n=1 per arm.
That is a genuine limitation of this protocol, which spent its load budget on
breadth across workloads rather than repetition within one. It is why the
rejection leans on the reproduced core-utilization signal rather than on any
single latency number.

### The contention hypothesis was wrong

The first candidate run showed ~5.41 utilized cores on generation-dominated
workloads against the control's ~5.91 — **at identical `threads = 6`**. Since
`threads_batch` cannot reduce core usage on non-batched decode, I attributed
this to external CPU load and used the §3 allowance to re-test. (That
attribution was itself uninstrumented: run 1's recorded pre-run load was 0.92,
the *lowest* of the three runs, which argues against run 1 having been the
contended one.)

**The re-test refuted that explanation.** The instrumented pre-run state was
`temp=77 load=1.24 mem_gb=55`, against the control's `temp=68 load=1.39
mem_gb=56` — comparable or quieter by the one contention proxy actually
recorded. (An earlier draft of this document also cited a "top process 7 %" and
a load of 1.35. Neither is instrumented: `run_q.sh` records only temp, load and
free memory. Those figures were uninstrumented recollection and are withdrawn.)
The candidate reproduced the effect almost exactly:

| | Control | Cand run 1 | Cand run 2 |
|---|---:|---:|---:|
| mean cores, 13 gen-dominated workloads | **5.910** | 5.408 | **5.404** |

5.408 vs 5.404 across independent runs is reproduction, not noise. The control's
*minimum* (5.872) exceeds both candidates' *maxima* (5.477 / 5.469): all 13
workloads separate in the same direction with **zero overlap**.

The decisive evidence is internal to each run. In the *same server process,
minutes apart*, the candidate's long-prompt row shows **6.773 and 6.763
utilized cores** — 13 % *more* than the control's 5.982. A host contended enough
to deny the decode phase half a core could not simultaneously grant the prefill
phase more cores than the control received. The CPU accounting agrees: over the
13 workloads candidate run 1 did **less** CPU work (−1.9 %) in **more** wall time
(+7.4 %). Starvation shows as the same CPU spread over longer wall; this process
is not being denied CPU, it is not asking for it.

The generation regression is therefore a **real, reproducible property of the
configuration**, not a measurement artifact — so §9's escape clause does not
apply and the gate stands as failed.

What the audit predicted was that `threads_batch` would not affect *generation
throughput*, because `llama-context.cpp:2556` selects
`batched ? n_threads_batch : n_threads` and `threads` was held at 6. The
prediction is not borne out end-to-end. The mechanism is **not established
here**: measuring it would need the sampling profiler that
`perf_event_paranoid = 4` blocks. Recording the effect without explaining it is
the honest stopping point; a plausible-sounding mechanism would be invention.

## Gates D, E, F — no regression

**KV reuse is byte-identical.** Reused/evaluated counts match exactly in every
turn of both arms, measured from `usage.prompt_tokens_details` (never
`/props.cached_tokens`, which reports resident cache size after the turn and
previously produced a false 100 % cold-reuse reading).

| Turn | Prompt | Reused | Evaluated | Control | Cand |
|---:|---:|---:|---:|---|---|
| 1 | 21 | 0 | 21 | ✓ | ✓ |
| 2 | 74 | 53 | 21 | ✓ | ✓ |
| 3 | 125 | 106 | 19 | ✓ | ✓ |

The 8-turn session likewise matches on all eight turns (reuse 0 → 93.3 %,
evaluated flat at 17–21). Session RSS growth: control 0.1032 %, candidate
0.0001 %. Peak RSS 35.697 → 35.700 GiB (+0.0068 %).

## Decision

**C — REJECTED_CROSS_WORKLOAD_REGRESSION.**
**`threads_batch = 8` is NOT a qualified production candidate.**

The causal reason: the +6.79 % cold-prefill gain is real and repeatable, but it
is not free. Generation regresses by at least 12 % and short-prompt latency by
at least 19 %, both reproduced on independent runs under matched thermal and
host conditions. Cold prefill is one workload; generation and short prompts are
the common case. Trading a 12–21 % loss on every short interaction for a 7 %
gain on a cache-miss long prompt is a bad trade for a default.

This closes the `threads_batch` axis for this model and hardware. PERF-OPT-2's
narrow finding stands as measured — it simply did not measure the cost, which is
exactly why qualification exists.

## Budget

3 model loads, ~7.5 minutes of measured inference, **1,173 generated tokens**
(391 per suite × 3): 1 warm-up + 32 short + 64 generation + 8 long + 94 reuse +
192 session.
