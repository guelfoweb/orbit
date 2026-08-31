# PERF-OPT-1 — cold-prefill causal profile

Profiling only. **No production source, test or configuration changed**, and
nothing was tuned. This mission answers *where* the cold-prefill time goes; it
does not optimize it.

| Property | Value |
|---|---|
| Repository baseline (profiling) | `0f2560e24eefce749cc73eec045a0b52ca79dbb2` |
| **Executable/source baseline** | **`aaf544013a1f398e5587c710316a56ccd4d64e7d`** |
| Model inference | YES — one model load, one long request |

The two differ because PERF-BASELINE-1 and the RSS correction were both
docs-only: the measured code is byte-identical to the PERF-BASELINE-1 runtime.

## PERF-BASELINE-1 RSS audit (corrected before profiling)

`/proc/<pid>/status` labels these fields `kB` but the values are KiB, so the
conversion is `value / 1024 / 1024`. The original checkpoint divided by
~1,073,823 instead of 1,048,576, understating each figure by ~2.4 %.

| Raw | Was | Corrected |
|---|---:|---:|
| 37,355,268 kB (idle) | 34.79 GiB | **35.62 GiB** |
| 37,433,472 kB (peak `VmHWM`) | 34.86 GiB | **35.70 GiB** |
| 37,280,396 kB (session) | not stated | **35.55 GiB** |

These raw values are **PERF-BASELINE-1's process**, not this run's. This
profiling run's own server measured `VmRSS 37,355,308` / `VmHWM 37,433,924` kB
(35.62 / 35.70 GiB) — a separate model load, hence the small deltas.

The error was confined to two summary lines; the comparison table already used
the correct 35.70 GiB for the same raw value, so the document contradicted
itself — which is how the bug surfaced. Peak RSS remains **35.70 GiB current vs
35.78 GiB historical = −0.23 %**, still effectively neutral, so the
PERF-BASELINE-1 performance verdict is unchanged. Corrected in PR #290
(`0f2560e2`), docs-only, no rerun.

## Protocol

Identical artifact, backend and server configuration to PERF-BASELINE-1. The
**output cap differs by design** — 8 tokens here versus 32 — because this
mission profiles prefill, not generation. That cannot affect `prompt_ms`: `pf_ms`
closes at `client.py:2658`, before generation starts at `:2681`.

| Property | Value |
|---|---|
| Model | `Ornith-1.5-35B-Q4_K_M.gguf`, 21,713,462,848 bytes |
| SHA-256 | `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f` |
| Backend | llama.cpp b9551 (`379ac6673b5cd75c7b4e07d1521c50f1e093878c`) |
| Config | `ctx=8192`, threads 6, threads-batch 6, batch 256, ubatch 128, temp 0, think off |
| MTP / auto-MTP | **OFF** — verified from `/props` before and after |
| Workload | the exact PERF-BASELINE-1 `medium_1` prompt |
| Output cap | 8 tokens |
| Cache | cold — `/props.cached_tokens = 0` before, `reused_tokens = 0` in the response |

## Timing-boundary semantics (resolved before inference)

Established by reading `client.py`, not assumed:

* **`prompt_ms` / `prefill_ms`** starts at `pf_start = lib.llama_time_us()`
  (`client.py:2577`), which is **after** `self.tokenize(prompt)`. It therefore
  **excludes** tokenization and **includes** route-anchor planning, KV
  preparation, batching and native `llama_decode`. It is *not* pure matmul.
* **`backend_ttft_ms`** comes from `_RequestTiming`, started unconditionally at
  **`client.py:1428`** in `complete_chat`, and marked at the first generated
  token (`client.py:4107-4108`). The occurrences at `:2261`/`:2566` are
  `or`-defaulted fallbacks that never fire on this path — the object is created
  once at `:1428` and threaded through.
* Therefore `backend_ttft_ms − prompt_ms` is an **upper bound** on all
  Orbit-side pre-prefill work. That band is wider than tokenization alone: it
  also covers `mark_unused` (`:1429`), multimodal preparation (`:1431`),
  prompt-cache mode (`:1437`), chat-template rendering (`:1463`), four
  anchor/segment planning calls (`:1472-1500`), MTP admission and rolling-route
  eligibility (`:1517-1547`), and tokenization (`:2571`). Measuring it as one
  band needs no instrumentation, and a bound is sufficient here because the
  number turns out to be negligible.

## Measured decomposition

3,716 prompt tokens, 0 reused, 3,716 evaluated, 8 generated.

| Component | Time | Share | Basis |
|---|---:|---:|---|
| HTTP wall | 113.391 s | 100.00 % | MEASURED |
| `backend_ttft_ms` | 112.497 s | 99.21 % | MEASURED |
| — native prefill (`prompt_ms`) | **112.447 s** | **99.17 %** | MEASURED |
| — tokenize + template + orchestration | **0.050 s** | **0.04 %** | DERIVED (`ttft − prompt_ms`) |
| generation (8 tokens) | 0.895 s | 0.79 % | MEASURED |
| HTTP/server residual | −0.0008 s | ~0.00 % | DERIVED |

The residual is slightly **negative** because three clocks are involved: HTTP
wall uses the client's `time.monotonic`, `backend_ttft_ms` uses
`time.monotonic_ns()` in the server, and `prompt_ms`/`predicted_ms` use
`llama_time_us()`. −0.8 ms is below their cross-clock resolution.

Prefill rate 33.05 tok/s; generation 8.94 tok/s.

### CPU saturation

| Metric | Value |
|---|---|
| CPU user | 679.37 s |
| CPU system | 0.08 s (**0.012 %**) |
| CPU total | 679.45 s over 113.391 s wall |
| **Cores utilized** | **5.99** of 6 configured threads |
| Thread efficiency | 99.9 % of the configured 6 |
| Machine capacity | 12 logical cores — this run used **49.9 %** of it |

`--threads 6` is the PERF-BASELINE-1 protocol constant, not a tuning finding.
"5.99 of 6" is saturation of a **self-imposed limit**, not of the hardware: six
hyperthreads sat idle by configuration. Whether raising the thread count would
help is untested and depends on the unresolved question above — a
memory-bandwidth-bound workload would not benefit.

The prefill is **CPU-busy for its whole duration**, using essentially exactly
the 6 threads it was given, with negligible system time. It is not waiting on
I/O, syscalls or a scheduler. What those cycles are spent *on* is not resolved
here — see the attribution limits below.

### Orbit-side overhead

Non-model time (`wall − prompt_ms − predicted_ms` = 48.7 ms, the same band as
the 49.5 ms row above minus generation-side overhead) is
**0.049 s = 0.043 %** of the request — well inside the "negligible" (<2 %)
category. Orbit's Python layer is not a measurable cost on this path.

## Attribution limits

`perf` is installed but `perf_event_paranoid = 4` blocks unprivileged sampling,
and altering host security settings is out of scope. `py-spy` is not installed
and installing it is not authorized. So sampling-level attribution **inside**
the native prefill was not collected.

Consequently the 112.447 s is reported as **NATIVE BACKEND COMPUTE/SCHEDULING**
and is *not* subdivided into ggml matmul vs attention vs MoE expert routing vs
batching. That distinction is **UNRESOLVED**. The CPU-saturation evidence
(5.99 cores, 0.012 % sys) makes compute-bound work by far the most likely
dominant term, but "most likely" is not measured, and no breakdown is invented
here.

## Validity

| Gate | Result |
|---|---|
| Model SHA correct | PASS |
| Configuration correct | PASS |
| MTP off | PASS |
| Cache genuinely cold (`reused_tokens = 0`) | PASS |
| Output cap honoured (8 tokens) | PASS |
| Prompt token count matches baseline (3,716) | PASS |
| No unexpected restart | PASS — one load |
| Profiler perturbation | PASS — none used |
| Thermal | PASS — 60 °C before, 81 °C after (limit 100 °C), cores at ~4.0 GHz |
| Host contention | PASS — load 0.48 before |

Thermal conditions were **better** than PERF-BASELINE-1 (60 °C before / 81 °C
after, vs 96 °C), and this run measured 33.05 tok/s prefill against the
baseline's 29.80 tok/s median — **+9.8 % against the median, +8.3 % against the
baseline's own fastest run**, which exceeds that baseline's internal two-run
spread of 3.5 %.

The output cap is **not** a confound: `prompt_ms` is measured entirely before
the first token is generated, so 8 vs 32 output tokens cannot causally affect
it. The plausible explanation is the thermal difference, but with n=1 here that
is **not established** — this mission ran one request by budget, which is enough
to attribute *where* time goes and not enough to compare rates.

**No improvement is claimed, and this number must not be used as a new
baseline.** PERF-BASELINE-1's 29.80 tok/s median remains the reference figure;
any future optimization must be measured against it under its own protocol.

## Causal classification

**D — NATIVE_BACKEND_DOMINATED** (native prefill).

The classification deliberately says *native backend*, not *native compute*.
The CPU evidence rules out I/O-wait, syscall-bound and scheduler-starved
execution, but ~100 % busy user time looks identical for useful arithmetic,
memory-bandwidth stall, cache thrash and spin-waiting in a thread barrier. On a
35.6 GiB Q4_K_M model on a 6-core CPU with no GPU, memory-bandwidth-bound is a
live hypothesis, and `perf` — blocked here — is exactly the instrument that
would separate it. Naming compute in the headline would assert the mechanism
this document elsewhere marks UNRESOLVED.

Native prefill is 99.17 % of wall time, Orbit-side work is 0.043 %, and the
process holds 5.99 of 6 threads with 0.012 % system time. Since the native
backend consumes ≥90 % of the relevant time, **no Orbit Python refactor is
justified by this evidence** — optimizing the entire Python layer to zero would
recover at most 0.04 % of the request.

## Budget

One model load, one long cold-prefill request, 8 generated tokens, ~2 minutes of
measured inference. The second model load authorized under §16 was **not**
performed: it is gated on Orbit overhead appearing ≥5 %, and the measured figure
is 0.043 %, so a direct-backend control would answer a question that is already
settled.
