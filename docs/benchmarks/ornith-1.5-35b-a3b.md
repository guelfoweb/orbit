# Ornith 1.5 35B-A3B — measurements on the reference system

What has actually been measured for this profile, where each number came from,
and — just as importantly — what has **not** been measured.

Everything below is a measurement on one machine. It is not a claim about the
model in general: results move with CPU, memory bandwidth, quantization,
backend build, and workload.

## Why this profile is measured separately

The other rows in the README table are warm steady-state medians from a
controlled *chat* benchmark: one excluded warm-up, three measured repetitions, a
fixed long-prefill fixture, at `ctx=8192` with 6 threads.

**No Ornith run has been made under that protocol.** Every measurement here
comes from the *analysis* workflow at `ctx=16384` with 12 threads. Reporting
those numbers inside the chat table would imply a comparability that does not
exist, so the README gives this profile its own table with its own stated basis
instead.

Within that basis the figures are sound. The prefill numbers below count only
calls with **no cache reuse**, which makes them comparable to the README's
"evaluated tokens only, excludes cache benefit" definition; the reuse-affected
numbers are excluded and explained under
[Prefill on reuse calls](#prefill-on-reuse-calls-not-comparable).

## Model specification

Properties of the model file, not of Orbit.

| Property | Value |
|---|---|
| Repository | [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) |
| File | `Ornith-1.5-35B-Q4_K_M.gguf` |
| Quantization | Q4_K_M (4.89 BPW) |
| File size | 20.21 GiB |
| Parameters | 35.51 B (35B-A3B) |
| Architecture | `qwen35moe`, 256 experts, 8 active |
| Training context | 262144 |

Source: `print_info` lines 54–136 of the run log listed under
[Evidence sources](#evidence-sources).

## Reference hardware

| Property | Value |
|---|---|
| CPU | Intel Core i7-10710U (NUC10), 6 cores / 12 threads |
| RAM | 64 GB |
| GPU | none (`gpu_layers = 0`) |
| OS | Linux x86_64 |
| llama.cpp build | b9551 |

## Orbit configuration used for these measurements

| Setting | Value |
|---|---|
| Context | 16384 |
| Batch / ubatch | 256 / 128 |
| Threads | 12 (generation and batch) |
| Flash Attention | `auto` |
| Thinking | off |

```bash
orbit server --ctx 16384 --batch 256 --ubatch 128
```

Note this differs from the README table's benchmark configuration
(`ctx=8192`, 6 threads), which is one reason the numbers are not interchangeable.

## Measured values

### Prefill, cold calls only — the published range

Calls that evaluated their prompt with **zero** reused tokens, so no restore work
sits inside the timing window. These are the samples behind the README's
`21–33 tok/s (n=5, median 31.8)`:

| Source | Evaluated tokens | Prefill |
|---|---:|---:|
| `/report` measurement | 5526 | 32.9 tok/s |
| autonomous run A, call 1 | 604 | 31.8 tok/s |
| autonomous run A, final call | 9613 | 21.0 tok/s |
| autonomous run B, call 1 | 604 | 31.9 tok/s |
| autonomous run B, final call | 8419 | 23.2 tok/s |

n = 5, min 21.0, max 32.9, median 31.8. The two lower values are the largest
prompts, which is the expected shape rather than noise.

### Generation — the published range

Derived as output tokens ÷ (wall clock − prefill) over every call with at least
20 output tokens: **n = 22, median 5.9 tok/s**, spanning 1.8–9.9 with a single
low outlier at 1.8; excluding it, 4.8–9.9. The README states `~5–10 tok/s`.

This is a **looser bound than a dedicated decode benchmark**: the denominator
includes any non-decode overhead in the call. The one directly measured decode
figure, 8.1 tok/s from the `/report` call below, falls inside the derived range,
which is the cross-check that makes the range publishable.

### Single `/report` call

One model call, measured end to end. **n = 1** — recorded here because it is
real and directly measured, not because one sample is a benchmark.

| Metric | Value |
|---|---|
| Prompt tokens | 5526 evaluated, 0 cached |
| Output tokens | 886 |
| Prefill | 32.9 tok/s |
| Decode | 8.1 tok/s |
| Wall clock | 277.1 s (168.1 s to first token) |

Measured 2026-08-24 against served commit `98049a9`.

### Prefix cache reuse

The clearest result on this profile. Across one 13-call autonomous analysis run,
per-call KV reuse rises as the run accumulates context:

| Call | Reused | Evaluated | Reuse |
|---:|---:|---:|---:|
| 1 | 0 | 604 | 0.0% |
| 2 | 604 | 2,337 | 20.5% |
| 3 | 2,941 | 2,957 | 49.9% |
| 4 | 5,898 | 1,285 | 82.1% |
| 6 | 8,616 | 824 | 91.3% |
| 10 | 12,227 | 443 | 96.5% |
| 12 | 14,736 | 594 | 96.1% |
| 13 | 0 | 9,613 | 0.0% |

Run totals: **95,799 reused / 24,943 evaluated = 79.3% overall**, with per-call
reuse peaking near 96%. Call 13 opens a fresh report context and so starts cold
again — the reset is expected, not a regression. Token accounting was exact on
every call.

Read the aggregate carefully: 79.3% is the whole run including its cold start;
~96% is the *steady-state peak*, not an average.

### ANALYSIS prefix prewarm

Descriptive timings, **n = 1** each, for the opt-in
`ORBIT_ORNITH_ANALYSIS_PREFIX_PREWARM=1` startup capture:

- capture cost: 11.0 s of startup prefill, 73,736,684 bytes resident
- with it on, a first analysis step restored 384 of 588 prompt tokens and its
  prefill fell from 18.2 s to 6.0 s

Reuse itself is on by default and free — the first analysis step captures the
checkpoint on its way past. The environment variable only decides whether that
capture is paid eagerly at startup instead.

## Not measured

Stated explicitly so nobody infers a number that does not exist:

- **Peak RAM / RSS.** No resident-memory measurement exists for this profile.
  The 20.21 GiB file size and the mapped model buffer are not RSS, and the
  other 35B-A3B profile's figure must not be carried over.
- **Warm steady-state chat throughput**, tools-on chat latency, and tool+final
  latency — the chat-protocol columns. No data of any kind; the throughput
  figures above are the analysis workload, not that protocol.
- **Flash Attention resolved value.** The log records `auto`, never what it
  resolved to.
- **MTP state.** Absent from the evidence; absent is not the same as off.

### Prefill on reuse calls, not comparable

Per-call `prefill_ms` on calls that **did** reuse cache is not usable as a
prefill-throughput figure. The timer starts before the prefix reuse and restore
work, while the token count excludes reused tokens, so restore overhead lands in
the denominator but not the numerator. That systematically depresses the rate:
medians of 14.6 tok/s (n=13) and 18.3 tok/s (n=9) across two runs, against
21–33 tok/s on cold calls — a spread that reflects the metric's definition, not
the hardware.

This is why the published range is drawn from zero-reuse calls only.

## Evidence sources

All paths relative to the qualification checkpoint archive, which is kept
outside the repository.

| What | Source |
|---|---|
| `/report` call | `report-performance-baseline-20260824-022942/artifacts/measurement.json` |
| Served commit | `report-performance-baseline-20260824-022942/artifacts/served_commit.txt` |
| Cache reuse, per call | `analysis-autonomous-final-20260824-202440/evidence/fullrun-qualification.json` |
| Model spec, runtime config | `analysis-autonomous-progress-20260824-190424/evidence/fullrun.ABORTED-infra-kill.log` |
| Prewarm timings | `AGENTS.md` |
| Qualification status | `docs/releases/v0.0.1-rc34.md`, `docs/releases/v0.0.1-rc35.md` |

Two log files in the archive — `fullrun-budget.log` and
`fullrun-qualification.log` — are byte-identical. They are one run, and counting
them twice would double the sample.

## Reproducing a publishable row

To fill the README row properly, run the protocol the other rows use — one
excluded warm-up and three measured repetitions against the fixed long-prefill
fixture, at `ctx=8192` with 6 threads — and capture peak RSS alongside it. See
`docs/QWEN3_CODER_QUALIFICATION.md` for the procedure.
