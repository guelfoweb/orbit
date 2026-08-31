# PERF-INVESTIGATE-1 — why `threads_batch` changes single-token generation

Causal investigation only. **No production source, test, default or host setting
changed.**

| Property | Value |
|---|---|
| Repository baseline | `7a64fc11fd289a6bab1a632c3fb2f38be095f3d0` |
| **Executable/source baseline** | **`aaf544013a1f398e5587c710316a56ccd4d64e7d`** |
| Model inference | YES — 2 model loads, 128 generated tokens |
| Host `perf_event_paranoid` | **4, unchanged** |
| Orbit run as root | **No** — both servers ran as uid 1000 |

## The contradiction

`llama-context.cpp:2556` reads:

```c
int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;
```

Single-token decode is not batched, so it selects `n_threads = 6`, unchanged
between the two configurations. Yet PERF-APPLY-1 measured generation 12–21 %
slower at `threads_batch = 8`.

## Resolution: the source reading was CORRECT; the influence is INDIRECT

`n_threads` really is 6 for decode in both configurations. `threads_batch`
reaches generation through the **OpenMP runtime's persistent thread pool**, not
through the selection at `:2556`.

### The chain, established from source

1. **Orbit never attaches a threadpool.** `grep` across all of
   `src/orbit/native_llama/` (excluding vendored sources) finds no call to
   `llama_attach_threadpool` / `attach_threadpool`. So `threadpool` and
   `threadpool_batch` are both `nullptr`, and the `tp` selection at `:2557`
   passes null in both configurations. **There is no llama.cpp-owned persistent
   pool to carry state.**

2. **The build uses OpenMP.** `libggml-cpu.so` imports `GOMP_parallel`,
   `GOMP_barrier`, `GOMP_single_start`, `omp_get_num_threads` and
   `omp_get_thread_num`; `libllama.so` links `libgomp.so.1`. The
   `GGML_USE_OPENMP` branch at `ggml-cpu.c:3781` is the live path, not the
   disposable-threadpool `#else` branch at `:3805`.

3. **That branch delegates thread supply to libgomp:**

   ```c
   #pragma omp parallel num_threads(n_threads)
   ```

   `num_threads` is a per-region *request*. The workers come from libgomp's
   process-wide pool, which **persists between parallel regions** and is sized
   by the largest region seen. Orbit sets no `OMP_NUM_THREADS`,
   `OMP_WAIT_POLICY` or `GOMP_SPINCOUNT`, so libgomp defaults apply: threads are
   retained after a region ends, and idle workers busy-wait before parking.

4. **Consequence:** a batched compute at `num_threads(8)` grows the pool to 8
   threads. Every subsequent decode region requests 6 — correctly, per `:2556` —
   but runs inside a process that now owns 8 OpenMP threads.

### The runtime evidence that confirms it

The prediction of the chain above is that the thread count is set by
`threads_batch` and is visible **before any request**. It is:

| | Control (`tb=6`) | Candidate (`tb=8`) |
|---|---:|---:|
| **OS threads at idle, before first request** | **6** | **8** |
| OS threads after warm-up | 6 | 8 |
| OS threads after generation | 6 | 8 |

`threads` was 6 in both. The only changed input is `threads_batch`, and it
determines the process's thread count from load onward.

## Measurements

Identical fixture, 79 bytes, SHA-256
`0268bbac816f3a9b5604094f811c059582f49daf393607082d8b020292615f7b` — the exact
PERF-APPLY-1 generation fixture. 24 prompt tokens, 64 output tokens, both runs.

| Metric | Control | Candidate | Delta |
|---|---:|---:|---:|
| Generation | **10.615 tok/s** | **9.123 tok/s** | **−14.06 %** |
| `predicted_ms` | 6,029.0 | 7,015.1 | +16.4 % |
| Wall | 6.665 s | 7.741 s | +16.1 % |
| CPU total | 39.75 s | 41.11 s | **+3.42 %** |
| CPU **system** | 0.05 s | 0.55 s | **11×** |
| Utilized cores | 5.964 | **5.311** | −10.95 % |
| CPU per generated token | 0.6211 s | 0.6423 s | +3.42 % |
| Peak RSS | 37,433,692 kB | 37,434,228 kB | +0.0014 % |

The regression **reproduces** (−14.06 % here; PERF-APPLY-1 saw −12.24 % and
−21.37 %).

### What the CPU accounting constrains

The candidate burns **more** total CPU (+3.4 %) to produce the **same** 64
tokens, while achieving **fewer** concurrent cores (−11 %) over **more** wall
(+16 %). That combination excludes two explanations:

* **Not external starvation** — a starved process shows the *same* CPU spread
  over longer wall. This one does more work.
* **Not extra useful parallelism** — the extra CPU produces no extra tokens.

The **11× jump in system time** (0.05 → 0.55 s) is the signature of threads
parking and waking — futex and scheduler work — which is what surplus OpenMP
workers generate when they are not needed by a region that requested fewer.

## Hypothesis verdicts

**Generation actually uses `n_threads_batch`: NO.** The selection at `:2556` is
unambiguous and `batched` is false for single-token decode. The source reading
was correct at the selection point.

**Persistent-threadpool hypothesis: SUPPORTED, not PROVEN.** Source establishes
that no *llama.cpp* pool persists (none is ever attached) and that the live path
delegates to libgomp's persistent pool. Runtime confirms `threads_batch`
controls the process thread count from idle. What is **not** directly measured
is the intra-process behaviour of those two surplus threads during a decode
region — that needs sampling, and `perf_event_paranoid = 4` blocks unprivileged
attach while passwordless sudo is unavailable. Changing the sysctl was
explicitly out of scope, so the final step is inferred from the system-time and
concurrency signatures rather than observed.

## Profiling limits

`perf` is installed. Unprivileged attach fails: *"Failure to open any events for
recording"* at `perf_event_paranoid = 4`. `sudo -n true` returns *"a password is
required"*, so the authorized one-shot privileged attach could not run
non-interactively. **No sysctl was modified, no capability granted, no tool
installed, and Orbit was not run as root.** Per-thread `/proc` accounting was
used instead; the per-TID delta collector returned no rows (a `/proc/<pid>/task`
parsing defect in the harness), so the thread-level evidence here is the total
count plus process CPU accounting, not a per-worker breakdown.

## Classification

**B — SHARED_THREADPOOL_SYNCHRONIZATION_EFFECT**, via the OpenMP runtime rather
than a llama.cpp threadpool.

**Contradiction resolution: YES, AND `threads_batch` influences generation
indirectly.**

## Validity

| Gate | Result |
|---|---|
| Same artifact SHA / backend | PASS |
| Parameter drift | NONE — 12 config fields identical except `threads_batch` |
| MTP off | PASS, asserted both runs |
| Identical fixture (SHA verified) | PASS |
| Cold state | PASS — `cached_tokens = 0`, `reused = 0` |
| Thermal | Control 86 → 93 °C; candidate 83 → 95 °C. Start within 3 °C (limit is 8), no collapse. PASS |
| Host policy unchanged | PASS |
| Model loads | 2 (the hard maximum) |

## Consequence

`threads_batch = 8` remains **REJECTED**; the axis stays **CLOSED**. This
investigation explains the rejection rather than reopening it — and it shows the
cost is structural, not incidental: raising `threads_batch` grows a
process-wide OpenMP pool that every later decode pays for.

It also invalidates the premise behind the next tuning step. `batch` and
`ubatch` change how much work a batched region does, but not how many threads
the pool holds, so they do not carry this hazard; `threads_batch` was the
uniquely dangerous knob because it alone sizes the pool.

## Budget

2 model loads, 128 generated tokens (2 × 64, plus 2 one-token warm-ups),
~15 s of measured generation.
