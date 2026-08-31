# PERF-BASELINE-1 — post-refactor CPU-only production baseline

Measurement only. No production source, test or configuration was changed;
nothing was tuned. Numbers are one machine's, not a claim about the model.

| Property | Value |
|---|---|
| Orbit baseline | `aaf544013a1f398e5587c710316a56ccd4d64e7d` |
| Historical comparison | `892cfad` (`docs/benchmarks/ornith-1.5-35b-a3b.md`) |
| Model inference | YES — one model load |

## Reference system

| Property | Value |
|---|---|
| CPU | Intel Core i7-10710U, 6 cores / 12 threads |
| RAM | 62 GiB total, 56 GiB available at start |
| GPU | none |
| OS | Linux 7.0.0-30-generic x86_64 |
| Load average before | 1.51 |
| Package temperature | 77 °C before, 96 °C after (limit 100 °C) |

## Model

Resolved from repository evidence, not assumed. The mission brief named Gemma 4
12B, but no such artifact exists locally and the README "Supported Models" table
does not list it; the only Gemma 4 12B GGUF present is a 444 MB **MTP draft**
model, not a base model. Of the four README rows, **only Ornith has a
reproducible benchmark record**, so it is the sole model with a
protocol-comparable historical baseline.

| Property | Value |
|---|---|
| File | `Ornith-1.5-35B-Q4_K_M.gguf`, 21,713,462,848 bytes |
| SHA-256 | `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f` |
| Quantization | Q4_K_M |
| Profile | `orbit-ornith15-native-v1` |
| Backend | llama.cpp b9551 (`379ac6673b5cd75c7b4e07d1521c50f1e093878c`) |

The SHA-256 and the backend revision are **identical** to the historical record,
on the same machine — which is what makes the comparison below valid.

## Configuration

```bash
orbit server --model-id ornith15-35b-a3b-q4-k-m \
  --ctx 8192 --threads 6 --threads-batch 6 --batch 256 --ubatch 128 --think off
```

`temperature=0`, thinking off, tools off, one parallel slot. **MTP off and
auto-MTP off**, verified from `/props` before and after every phase:
`mtp_enabled=False`, `mtp_experimental_enabled=False`, `mtp_initialized=False`,
`self_mtp_active=False`.

## Startup and memory

| Metric | Value |
|---|---|
| Process start → `/props` ready | **41.21 s** |
| RSS after load (idle) | 37,355,268 kB = **34.79 GiB** |
| Peak RSS (`VmHWM`, whole lifetime) | 37,433,472 kB = **34.86 GiB** |

## Warm-up (excluded from medians)

20 prompt tokens, 1 output token, 0.659 s.

## Short single-turn — 24 input, 32 output, n=3

| Run | Wall | Prefill tok/s | Generation tok/s |
|---:|---:|---:|---:|
| 1 | 4.426 s | 35.66 | 8.66 |
| 2 | 4.369 s | 35.97 | 8.73 |
| 3 | 4.463 s | 35.37 | 8.54 |
| **Median** | **4.426 s** | **35.66** | **8.66** |

## Medium prompt — 3,716 input, 32 output, n=2

| Run | Wall | Prefill ms | Prefill tok/s | Generation tok/s |
|---:|---:|---:|---:|---:|
| 1 | 130.556 s | 126,877 | 29.29 | 8.89 |
| 2 | 126.275 s | 122,561 | 30.32 | 8.75 |
| **Median** | **128.4 s** | **124,719** | **29.80** | **8.82** |

The prompt overshot the 1k–2k target (3,716 tokens) because the filler
tokenized denser than estimated. It is reported as measured rather than
re-run, since re-running would spend a further ~4 minutes of CPU to move a
number that is already consistent with the historical prefill rate.

## Strict prompt/KV reuse — one session, exact-prefix continuation

Canonical per-request accounting from `usage.prompt_tokens_details`
(`reused_tokens` / `evaluated_tokens`), **not** `/props.cached_tokens` — the
latter is `len(cached_prompt_tokens)`, the resident cache size *after* the turn,
and reads as 100 % even on a cold first turn.

| Turn | Prompt | Reused | Evaluated | Reuse rate | Wall |
|---:|---:|---:|---:|---:|---:|
| 1 | 21 | 0 | 21 | **0.0 %** | 3.75 s |
| 2 | 74 | 53 | 21 | **71.6 %** | 3.60 s |
| 3 | 125 | 106 | 19 | **84.8 %** | 3.33 s |

Proof reuse is real: turn 1 evaluates every token with zero reuse, and from
turn 2 on only the new user text is evaluated (~19–21 tokens) while the resident
prefix is reused. Strict prefix reuse is intact after the refactor line.

## Bounded session stability — 8 turns, 24 output tokens each

| Turn | Prompt | Reused | Evaluated | Reuse | Wall | RSS (kB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 0 | 17 | 0.0 % | 2.898 s | 37,280,396 |
| 2 | 59 | 41 | 18 | 69.5 % | 2.669 s | 37,280,396 |
| 3 | 101 | 83 | 18 | 82.2 % | 2.699 s | 37,280,396 |
| 4 | 143 | 125 | 18 | 87.4 % | 2.700 s | 37,280,396 |
| 5 | 186 | 167 | 19 | 89.8 % | 2.899 s | 37,280,396 |
| 6 | 228 | 210 | 18 | 92.1 % | 2.733 s | 37,280,396 |
| 7 | 270 | 252 | 18 | 93.3 % | 2.785 s | 37,280,396 |
| 8 | 315 | 294 | 21 | 93.3 % | 2.903 s | 37,280,396 |

* **RSS: byte-identical across all 8 turns.** No growth.
* **Evaluated tokens flat at 17–21** while the prompt grows 17 → 315.
* **Latency flat** (2.669–2.903 s), no degradation.
* No unexpected reset or compaction.

This is BOUNDED SESSION STABILITY, not a long-run soak claim.

## Historical comparison — VALID

Same artifact SHA-256, same machine, same backend build, same ctx/threads/
batch/ubatch, thinking off, MTP off. The only material variable is the Orbit
revision (`892cfad` → `aaf5440`), which is what the refactor line changed.

| Metric | `892cfad` | `aaf5440` | Delta |
|---|---:|---:|---:|
| Prefill | 29.0 tok/s | 29.8 tok/s | **+2.8 %** |
| Generation | 8.3 tok/s | 8.66 tok/s | **+4.4 %** |
| Peak RSS | 35.78 GiB | 35.70 GiB | **−0.2 %** |

Both deltas are small and positive, and both are within the run-to-run spread
the historical record itself shows (its own three repetitions ranged
26.55–29.98 prefill, 7.04–8.30 generation). The honest reading is **no
regression**, not a speed-up: the refactor line cost nothing measurable.

Workload wall times are **not** compared — this mission ran short synthetic
prompts, not the five qualification fixtures, so those columns are not
protocol-identical.

## Largest measured bottleneck

**Prompt evaluation (prefill).** On the 3,716-token prompt, prefill was
124.7 s of a 128.4 s request — **97 % of wall time** — at 29.8 tok/s against
8.7 tok/s generation. Startup (41 s, once), memory (flat), and multi-turn
degradation (none) are all non-issues by comparison.

The measured mitigation already works: strict prefix reuse drives evaluated
tokens to ~19 per turn regardless of context size. So the exposure is
specifically the **cold, cache-miss prefill** — the first turn of a session and
any turn whose prefix does not match.

## Validity

| Gate | Result |
|---|---|
| Correct artifact (SHA verified) | PASS |
| Correct configuration | PASS |
| MTP off throughout | PASS |
| Reuse actually occurred (not assumed) | PASS |
| No unexpected restart | PASS — one load |
| Output caps honoured | PASS — ≤32 tokens |
| Metrics parser consistent | PASS |
| Thermal/contention | ACCEPTABLE — see note |
| Stale benchmark process | PASS — none |

Thermal note: package temperature reached 96 °C (limit 100 °C) and cumulative
throttle counters are non-zero, but those counters are since-boot, not per-run.
Within-run evidence argues against material throttling: generation held
8.54–8.73 tok/s across repetitions, RSS was byte-identical, latency was flat,
and cores were still boosting to 4.58 GHz at the end. Numbers are reported as
measured; a cooler machine could plausibly do slightly better.

## Budget

One model load. Roughly 8 minutes of measured inference, ~700 generated tokens
across all phases. The five-fixture qualification protocol was **skipped**: its
own record shows ~172 s and ~322 s for two single fixtures, so three repetitions
would have cost well over an hour, exceeding this mission's explicit budget.
