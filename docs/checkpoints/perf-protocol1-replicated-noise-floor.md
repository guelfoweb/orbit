# PERF-PROTOCOL-1 — replicated benchmark protocol and measured noise floor

Methodology only. **No production source, test, default or host setting
changed.** Both benchmark arms were deliberately **identical** — this is an A/A
placebo, not an optimization test.

| Property | Value |
|---|---|
| Repository baseline | `6aeeaa0f57d0a58c73a6d05967a196bbb2aa72a7` |
| **Executable/source baseline** | **`aaf544013a1f398e5587c710316a56ccd4d64e7d`** |
| Model inference | YES — 6 model loads |

## Headline result

**Two identical configurations differed by up to 21.9 % (prefill) and 12.3 %
(generation) on paired runs.** The measured noise floor at n=3/arm is **~17 %
for generation and ~16 % for prefill**. Every performance "win" this project has
measured — including PERF-OPT-2's +5.75 % — sits **inside** that floor.

## Protocol

| Property | Value |
|---|---|
| Model | `Ornith-1.5-35B-Q4_K_M.gguf`, 21,713,462,848 bytes |
| SHA-256 | `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f` |
| Backend | llama.cpp b9551 (`379ac6673b5cd75c7b4e07d1521c50f1e093878c`) |
| Config (**both arms**) | `ctx=8192`, `threads=6`, `threads_batch=6`, `batch=256`, `ubatch=128`, temp 0, think off, 1 slot, MTP off |
| SHORT fixture | 79 bytes, SHA-256 `0268bbac816f3a9b5604094f811c059582f49daf393607082d8b020292615f7b`, 64 output tokens |
| LONG fixture | 18,680 bytes, SHA-256 `3ceb97abe7c54764bbc5bfca86c8c89e017e550c619f7871649f5b305b7bd056`, 8 output tokens |
| Run order (frozen before inference) | **A1 · B1 · B2 · A2 · A3 · B3** |

Each load ran the identical sequence: config assertion → warm-up (excluded) →
reset → SHORT → reset → LONG → final capture. Every run asserted `threads`,
`threads_batch`, ctx/batch/ubatch/slots, thinking mode, all three MTP flags and
`cached_tokens == 0` before proceeding.

**Parameter drift: NONE.** All 11 compared config fields identical across all
six runs; `threads_batch` was 6 in every run.

## Raw per-run results

### SHORT — generation tok/s (primary)

| Run | gen tok/s | `predicted_ms` | TTFT ms | wall s | CPU s | cores | VmRSS kB |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 9.208 | 6,950.7 | 672.0 | 7.623 | 45.50 | 5.968 | 37,279,384 |
| B1 | 10.337 | 6,191.3 | 800.3 | 6.992 | 41.70 | 5.964 | 37,279,440 |
| B2 | 10.452 | 6,123.2 | 636.4 | 6.760 | 40.33 | 5.966 | 37,279,272 |
| A2 | 10.487 | 6,102.9 | 694.9 | 6.799 | 40.53 | 5.962 | 37,279,276 |
| A3 | 9.001 | 7,109.9 | 678.2 | 7.789 | 46.49 | 5.969 | 37,279,272 |
| B3 | 8.022 | 7,977.8 | 710.0 | 8.689 | 51.57 | 5.935 | 37,279,460 |

### LONG — prefill tok/s (primary)

| Run | prefill tok/s | `prompt_ms` | TTFT ms | wall s | CPU s | cores | VmHWM kB |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 24.245 | 153,266.8 | 153,322.0 | 154.471 | 924.74 | 5.986 | 37,433,840 |
| B1 | 29.549 | 125,757.3 | 125,815.1 | 126.766 | 759.45 | 5.991 | 37,433,440 |
| B2 | 30.569 | 121,562.9 | 121,624.0 | 122.545 | 734.15 | 5.991 | 37,433,884 |
| A2 | 29.561 | 125,704.1 | 125,761.7 | 126.708 | 759.17 | 5.992 | 37,433,748 |
| A3 | 26.627 | 139,555.1 | 139,612.3 | 140.756 | 843.14 | 5.990 | 37,434,024 |
| B3 | 25.136 | 147,836.2 | 147,897.9 | 148.915 | 891.69 | 5.988 | 37,434,124 |

Every LONG run: `prompt_tokens = 3716`, `reused_tokens = 0`,
`evaluated_tokens = 3716`, `output_tokens = 8`. **All six runs VALID**; none
excluded.

### Host / thermal per run

| Run | Time | load1 | busy ≥5 % | mem avail | T start | SHORT T | LONG T |
|---|---|---:|---:|---:|---:|---|---|
| A1 | 15:59 | 3.55 | 5 | 39.2 GB | 94 °C | 94→93 | 93→96 |
| B1 | 16:09 | 3.61 | 6 | 38.8 GB | 95 °C | 95→94 | 94→91 |
| B2 | 16:18 | 1.80 | 5 | 39.2 GB | 93 °C | 93→94 | 94→92 |
| A2 | 16:22 | 4.15 | 5 | 39.2 GB | 91 °C | 91→96 | 96→94 |
| A3 | 16:25 | 3.94 | 5 | 39.3 GB | 93 °C | 93→94 | 94→94 |
| B3 | 16:30 | 4.29 | 5 | 39.1 GB | 95 °C | 95→95 | 95→93 |

The machine ran at 91–96 °C throughout with an unrelated desktop workload
(browser, chat clients) present. That is deliberate: this measures **real-world
noise on the actual machine**, not laboratory conditions.

## Statistics

### SHORT generation tok/s

| Set | n | mean | median | min | max | SD | CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Arm A | 3 | 9.565 | 9.208 | 9.001 | 10.487 | 0.805 | **8.4 %** |
| Arm B | 3 | 9.604 | 10.337 | 8.022 | 10.452 | 1.371 | **14.3 %** |
| **Pooled** | 6 | 9.585 | 9.772 | 8.022 | 10.487 | 1.006 | **10.5 %** |

* max pairwise delta (identical configs): **30.7 %**
* max deviation from pooled median: **17.9 %**
* **false A/B mean delta: +0.40 %; false A/B median delta: +12.27 %**

### LONG prefill tok/s

| Set | n | mean | median | min | max | SD | CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Arm A | 3 | 26.811 | 26.627 | 24.245 | 29.561 | 2.663 | **9.9 %** |
| Arm B | 3 | 28.418 | 29.549 | 25.136 | 30.569 | 2.888 | **10.2 %** |
| **Pooled** | 6 | 27.615 | 28.088 | 24.245 | 30.569 | 2.635 | **9.5 %** |

* max pairwise delta (identical configs): **21.6 %**
* max deviation from pooled median: **13.7 %**
* **false A/B mean delta: +5.99 %; false A/B median delta: +10.97 %**

The +5.99 % false mean delta on LONG is, on its own, larger than PERF-OPT-2's
entire +5.75 % `threads_batch=8` "gain" — produced here by **two identical
configurations**.

### Paired A_i vs B_i (identical configurations)

| Pair | SHORT | LONG |
|---|---:|---:|
| A1 vs B1 | **+12.27 %** | **+21.88 %** |
| A2 vs B2 | −0.33 % | +3.41 % |
| A3 vs B3 | −10.88 % | −5.60 % |

The sign flips across pairs and the magnitude spans 23 points (SHORT) and 27
points (LONG). This is the cleanest statement of the problem: the same
configuration, measured three times against itself, disagrees by more than any
optimization this project has proposed.

### Drift and correlation (n=6 — descriptive only)

* **Chronological:** SHORT first-half mean 9.999 → second-half 9.170 = **−8.3 %**;
  LONG 28.121 → 27.108 = **−3.6 %**. Neither is monotonic.
* r(start loadavg) = **−0.51** for both metrics.
* r(fixture start temp) = −0.50 (SHORT), +0.35 (LONG).
* r(fixture end temp) = +0.12 (SHORT), **−0.69** (LONG).
* r(busy processes) = +0.37 / +0.36.

With n=6 these are **not** significance tests and no causal claim is made from
them. The consistent negative loadavg correlation and the second-half decline
are *consistent with* the host being the dominant noise source, which is what the
A/A design was built to expose.

## Noise floor

Derived conservatively — the maximum of several independent estimators, not the
smallest:

| Estimator | SHORT | LONG |
|---|---:|---:|
| pooled CV | 10.5 % | 9.5 % |
| max false A/B delta | 12.3 % | 11.0 % |
| max paired identical-config delta | 12.3 % | 21.9 % |
| 95 % band on a difference of two n=3 means | ±16.8 % | ±15.3 % |
| **Adopted noise floor** | **17 %** | **16 %** |

The single worst pairwise value (30.7 % SHORT) is excluded from the adopted
floor as an extreme of the pooled distribution rather than a typical A/B
comparison; the 95 % band, which is the quantity an A/B test actually competes
against, drives the result instead.

## Decision rule for future A/B experiments

```
minimum_meaningful_gain = max(practical_floor, noise_derived_threshold)
```

* **`practical_floor` = 5 %.** Derived, not assumed: below this a change is not
  distinguishable by a user on interactive workloads, and it is smaller than the
  session-to-session drift this project has repeatedly observed (~13 % between
  sessions on the same configuration).
* **`noise_derived_threshold`** = the 95 % band at the chosen replication count,
  from the CVs measured here (10.5 % SHORT / 9.5 % LONG).

| n per arm | SHORT detectable | LONG detectable | loads | ≈ wall time |
|---:|---:|---:|---:|---:|
| 1 | 29.1 % | 26.5 % | 2 | 0.1 h |
| 3 | 16.8 % | 15.3 % | 6 | 0.4 h |
| 5 | 13.0 % | 11.8 % | 10 | 0.7 h |
| 8 | 10.3 % | 9.4 % | 16 | 1.1 h |
| 20 | 6.5 % | 5.9 % | 40 | 2.9 h |
| 30 | 5.3 % | 4.8 % | 60 | 4.3 h |

**To resolve a 5 % effect at 95 % confidence requires n ≥ 34 per arm (SHORT) or
n ≥ 28 (LONG) — 56–68 model loads, 4–5 hours of inference.**

### Rules adopted

1. **n ≥ 3 per arm minimum**, and n=1 is never sufficient — a single pair here
   produced a 21.9 % false delta.
2. **Interleave arms**: `A/B/A/B/A/B`, never `AAA/BBB`. Blocked ordering would
   have converted this run's −8.3 % chronological drift into a fabricated
   arm effect.
3. **Record loadavg, busy-process count and temperature per run**, and publish
   them with the result.
4. **Report the spread, not just the mean.** A delta without its pooled CV is
   uninterpretable on this machine.
5. **A gain below the noise floor for its replication count is "indistinguishable
   from noise"** — not a small win.

## Environment classification

**C — BENCHMARK_ENVIRONMENT_TOO_NOISY.**

At n=3/arm this machine cannot discriminate below ~16–17 %. No optimization axis
under consideration offers a gain of that size — the best measured candidate to
date was +5.75 %, which this A/A experiment reproduced as pure noise between
identical configurations. Reaching a 5 % resolution needs 4–5 hours per
experiment, which is not a proportionate cost for the expected benefit.

This is a property of the measurement environment — a thermally saturated
15 W laptop CPU running at 91–96 °C with a live desktop workload — not of Orbit.

## Consequence

**Further performance tuning on this NUC is not justified.** Any future A/B
result under ~16 % would be indistinguishable from the noise measured here, and
the axes that remain (batch/ubatch) have no mechanism suggesting gains of that
magnitude.

This does not close performance work permanently. It closes it *on this
machine, at this replication cost*. The protocol above is reusable the moment
either changes — a quieter or better-cooled host, or a willingness to spend
4–5 hours per comparison.
