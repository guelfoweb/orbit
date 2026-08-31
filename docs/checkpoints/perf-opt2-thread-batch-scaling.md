# PERF-OPT-2 — cold-prefill `threads_batch` scaling

Benchmark and decision only. **No production source, test or default changed.**
One variable moved: `threads_batch` ∈ {6, 8, 12}, with `threads = 6` held fixed
in every run.

| Property | Value |
|---|---|
| Repository baseline | `8dc9fa6ab84be56c1ea58dd0b28fb7a3274e6544` |
| **Executable/source baseline** | **`aaf544013a1f398e5587c710316a56ccd4d64e7d`** |
| Model inference | YES — 4 model loads, 4 cold-prefill requests |

`git diff aaf5440..HEAD -- src/ tests/` returns 0 files: every commit since the
qualified source baseline has been docs-only, so the measured runtime is
byte-identical to the PERF-BASELINE-1 / PERF-OPT-1 runtime.

## Independent parameter control (§1 gate — PASSED)

Proven from source before inference:

* `--threads` → `args.threads` → `NativeClientConfig.threads` →
  `ctx_params.n_threads` (`client.py:676`)
* `--threads-batch` → `args.threads_batch` → `NativeClientConfig.threads_batch`
  → `ctx_params.n_threads_batch` (`client.py:677`)

Separate flags, separate config fields, separate llama.cpp context parameters.
The backend honours the split at
`vendor/source/llama.cpp/src/llama-context.cpp:2556`:

```c
int n_threads = batched ? cparams.n_threads_batch : cparams.n_threads;
```

Prefill is batched and therefore uses `n_threads_batch`; single-token decode
uses `n_threads`. They are independently configurable, so the experiment is
**not confounded** — and this is also why generation rate is excluded as a
target here.

Every run additionally *asserted* at runtime, from `/props`, that
`threads == 6`, `threads_batch == <target>`, `mtp_enabled == false`,
`mtp_experimental_enabled == false` and `cached_tokens == 0`. A drift would have
aborted the run rather than producing a quiet result.

## Frozen protocol

| Property | Value |
|---|---|
| Model | `Ornith-1.5-35B-Q4_K_M.gguf` |
| SHA-256 | `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f` |
| Backend | llama.cpp b9551 (`379ac6673b5cd75c7b4e07d1521c50f1e093878c`) |
| Fixed | `ctx=8192`, `threads=6`, `batch=256`, `ubatch=128`, temp 0, think off, 1 slot, MTP off |
| Prompt | the exact PERF-OPT-1 prompt — 18,680 bytes, SHA-256 `3ceb97abe7c54764bbc5bfca86c8c89e017e550c619f7871649f5b305b7bd056` |
| Output cap | 8 tokens |
| Cold proof | `cached_tokens = 0` before, `reused_tokens = 0` / `evaluated_tokens = 3716` after |

The identical prompt SHA was printed by every run; all four report
`prompt_tokens = 3716`.

## Raw measurements

| Run | `threads_batch` | Prefill tok/s | `prompt_ms` | TTFT ms | Wall s | CPU-s | Cores | `VmHWM` kB | Temp start→end | Load start |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| control | 6 | 29.135 | 127,542.199 | 127,602.440 | 128.570 | 770.28 | 5.991 | 37,433,776 | 66→81 °C | 1.12 |
| challenger | 8 | 30.771 | 120,764.470 | 120,821.372 | 122.072 | 817.65 | 6.698 | 37,434,156 | 75→79 °C | 1.03 |
| challenger | 12 | 29.440 | 126,221.777 | 126,275.643 | 127.254 | 1,256.51 | 9.874 | 37,435,188 | 67→83 °C | 2.32 |
| confirm | 8 | 30.849 | 120,457.771 | 120,511.529 | 121.481 | 813.99 | 6.701 | 37,434,084 | 67→83 °C | 1.76 |

The confirmation run's CPU cost is +5.67 % over control — so the ~6 % CPU
premium for `tb = 8` is itself repeatable, not just its throughput.

Generation rate is reported but not a target of this mission; `threads = 6`
governs it and was never varied. The `tb=8` first run's 6.40 tok/s generation is
an 8-token sample and is not treated as signal.

## Derived comparison (denominator = the same-mission control)

| `threads_batch` | Gain | `prompt_ms` reduction | CPU-s | Cores | Peak RSS | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 6 | — | — | 770.3 | 5.991 | — | control |
| 8 | **+5.61 %** | +5.31 % | +6.15 % | 6.698 | +0.0010 % | **CANDIDATE** |
| 12 | **+1.05 %** | +1.04 % | **+63.12 %** | 9.874 | +0.0038 % | **REJECT** |

Thresholds were fixed before the runs (§9): `<3 %` reject, `3–5 %` marginal
(reject), `5–10 %` candidate, `≥10 %` strong. They were applied exactly as
written and not revised after seeing results.

## Confirmation (§10)

`tb = 8` reached ≥5 %, so one additional load repeated it.

| Criterion | Requirement | Measured | Result |
|---|---|---|---|
| Gain vs control | ≥5 % | **+5.88 %** | PASS |
| Repeatability | within 3 % of first | **0.255 %** | PASS |

Mean of the two `tb = 8` runs: **30.810 tok/s, +5.75 %** over control.
`prompt_ms` spread 0.254 %; utilized cores 6.698 vs 6.701 — effectively
identical.

## Control validity (§7)

The control measured 29.135 tok/s against PERF-OPT-1's 33.05 — **−11.84 %**,
which exceeds the 8 % gate and required investigation before continuing.

| Run | Prefill tok/s | Conditions |
|---|---:|---|
| PERF-BASELINE-1 medium_1 | 29.29 | 77 °C start → 96 °C end (series) |
| PERF-BASELINE-1 medium_2 | 30.32 | same series, ended ~96 °C |
| PERF-OPT-1 | 33.05 | 60 °C start, **load 0.48** |
| PERF-OPT-2 control | 29.14 | 66 °C start, **load 1.12** |

A causal explanation exists: this control ran under roughly twice the
background load of PERF-OPT-1, from unrelated user processes that must not be
killed. The control agrees within **3.9 %** of the three other historical
measurements; **PERF-OPT-1's 33.05 is the outlier**, exactly as that checkpoint
itself cautioned when it recorded "n=1 cannot support a rate comparison" and
"must not be used as a new baseline".

The experiment is therefore not invalidated: the A/B denominator is the
**same-mission control**, measured on the same machine within ~14 minutes of the
challengers, not the historical figure. A uniformly depressed absolute rate
cancels in a ratio.

One limit this does *not* remove: if the whole mission ran under contention,
+5.75 % is the gain **under those conditions** and may not hold on an idle
machine. That is one more reason the value is a candidate rather than a new
default. All four runs of this mission shared
comparable conditions (start 66–75 °C, end 79–83 °C).

## What the CPU numbers show

Extra logical threads are genuinely *used* — utilized cores rise 5.991 → 6.698
→ 9.874, tracking the request. But they do not convert into throughput:

* `tb = 8`: +6.15 % CPU buys +5.61 % throughput — roughly break-even, 0.91 %
  gain per 1 % extra CPU.
* `tb = 12`: **+63.12 % CPU buys +1.05 %** — 0.02 % gain per 1 % extra CPU.

### Is the `tb = 12` rejection confounded by its higher start load (2.32)?

No, and the run's own counters rule it out. If background contention had starved
it, it would show *fewer* utilized cores and a *longer* wall time. It shows the
opposite on both:

* **Cores utilized 9.874** — the highest of any run, 64.8 % above control. It
  received the threads it asked for and used them.
* **Wall 127.254 s vs the control's 128.570 s** — marginally *faster* in wall
  terms despite roughly double the background load.

So `tb = 12` performed 63.1 % more CPU work and converted almost none of it into
throughput. That is an efficiency collapse intrinsic to the thread count, not a
symptom of being denied CPU.

The measured conclusion is that **additional logical threads beyond ~8 do not
improve this workload**. This document deliberately does *not* claim memory
bandwidth is saturated: that mechanism was not measured, `perf` remains blocked
by `perf_event_paranoid = 4`, and the PERF-OPT-1 attribution limits still apply.

## Veto checks (§9) — none triggered

| Check | Result |
|---|---|
| Peak RSS regression > 3 % | NO — +0.0010 % |
| Thermal instability | NO — `tb=8` runs 75→79 °C and 67→83 °C vs control 66→81 °C. The **hotter** `tb=8` run produced the **better** number (30.849 vs 30.771), which argues against throttling. |
| Not repeatable | NO — 0.255 % between runs |
| Token semantics differ | NO — 3,716 / 0 reused / 3,716 evaluated in all runs |
| Other backend parameter differs | NO — asserted at runtime |
| Errors / reliability regress | NO — all `finish_reason=length`, clean shutdown |

## Decision

**B — THREADS_BATCH_8_CANDIDATE.**
**THREAD_BATCH SCALING: CANDIDATE 8.**

`threads_batch = 12` is rejected on evidence: it costs **63 % more CPU for a
1.05 % gain** — far below the 3 % reject band. Note 1.05 % is about 4× the only
noise estimate this mission actually has (the 0.25 % spread between the two
`tb = 8` runs), so it is most likely a real but negligible gain rather than
noise. The CPU cost carries the rejection either way. End temperature is not
cited as a cost: `tb = 8`'s confirmation run also ended at 83 °C at a fraction
of the CPU, so end temperature tracks cumulative session heat, not thread
count. `threads_batch = 8` delivers a
confirmed, repeatable +5.75 % mean gain at ~6 % extra CPU with flat memory.

**No default is changed here.** +5.75 % is a real but modest gain measured on a
single prompt shape, on one machine, with n=2. Qualifying it against short
prompts, generation, KV reuse and bounded-session stability is a separate
mission's job.

## Budget

4 model loads (3 mandatory + 1 confirmation, the §3 normal maximum), 4
cold-prefill requests, 32 generated tokens total, ~8.3 minutes of measured
inference plus passive cooldowns.
