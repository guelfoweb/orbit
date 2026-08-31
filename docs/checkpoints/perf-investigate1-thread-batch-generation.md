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
   disposable-threadpool `#else` branch at `:3805`. The build cache settles it
   outright: `src/orbit/native_llama/vendor/build/llama.cpp/CMakeCache.txt` carries
   `GGML_OPENMP:BOOL=ON` and `OpenMP_C_FLAGS:STRING=-fopenmp`.

3. **That branch delegates thread supply to libgomp:**

   ```c
   #pragma omp parallel num_threads(n_threads)
   ```

   `num_threads` is a per-region *request*. The workers come from libgomp's
   process-wide pool, which **persists between parallel regions** and is sized
   by the largest region seen. Orbit sets no `OMP_NUM_THREADS`,
   `OMP_WAIT_POLICY` or `GOMP_SPINCOUNT`, so libgomp's untuned defaults apply and
   threads are retained after a region ends. What those idle workers do between
   regions — spin, park, or both — is **not established here**: llama.cpp's own
   spin tuning is inert on this build, since `ggml-cpu.c:4254-4257` leaves
   `OMP_WAIT_POLICY` commented out and `:4259-4266` sets `KMP_BLOCKTIME`, an
   Intel/LLVM-OpenMP variable that libgomp does not read (verified: zero `KMP_`
   strings in `libgomp.so.1`).

4. **A batched compute runs at startup, before the socket binds.**
   `run_server` calls `prewarm_startup_route_prefix(client)` at `app.py:867` /
   `:879`, while `ThreadingHTTPServer` is not constructed until `app.py:899`.
   That path reaches `_decode_prompt_range` (`client.py:2731`), which issues
   `llama_batch_get_one` / `llama_decode` in chunks of up to 64 tokens over a
   768-token route prefix — genuinely batched work, since `batched` is
   `ubatch.n_tokens > 1` (`llama-context.cpp:1487`).

5. **Consequence:** that startup region runs at `num_threads(8)` and grows the
   pool to 8 threads. Every subsequent decode region requests 6 — correctly, per
   `:2556` — but runs inside a process that now owns 8 OpenMP threads. This is
   why the count is observable before any client request.

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

### Ruling out the alternative sources of those two threads

"Two extra OS threads" alone would not identify libgomp — they could in
principle be ggml workers, an unfreed disposable pool, or Python/HTTP threads.
Source excludes the ggml explanation directly: in
`ggml_threadpool_new_impl` (`ggml-cpu.c:3674`), the only call to
`ggml_thread_create` sits inside the **`#else // GGML_USE_OPENMP`** branch.
Under the OpenMP build ggml creates **no pthreads of its own** — it fills
`workers[]` and lets `#pragma omp parallel` supply the threads.

The Python/HTTP explanation is excluded by the delta: both servers run the same
Orbit code, the same one-slot HTTP server and the same interpreter, and differ
only in `threads_batch`. A constant runtime cannot produce a count that tracks
that flag.

That leaves the OpenMP runtime as the only source consistent with both the
source and the observed 6-vs-8 split.

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

**Each configuration was measured once** — the mission's hard cap is two model
loads — so there is no dispersion estimate behind the −14.06 %. What reproduces
is the *direction and rough magnitude* across three independent single-shot
measurements: −12.24 %, −21.37 % (PERF-APPLY-1) and −14.06 % (here). That 9-point
spread is itself evidence of high run-to-run variance, so the four-significant-
figure precision should not be read as accuracy.

### What the CPU accounting constrains

The candidate burns **more** total CPU (+3.4 %) to produce the **same** 64
tokens, while achieving **fewer** concurrent cores (−11 %) over **more** wall
(+16 %). That combination excludes two explanations:

* **Not external starvation** — a starved process shows the *same* CPU spread
  over longer wall. This one does more work.
* **Not extra useful parallelism** — the extra CPU produces no extra tokens.

The **11× jump in system time** (0.05 → 0.55 s) is *consistent with* threads
parking and waking — futex and scheduler work — which is what surplus OpenMP
workers generate. It is not proof: no futex counts or scheduler statistics were
collected, and that 0.50 s is only 37 % of the 1.36 s total CPU increase.

### A second explanation this design cannot exclude

The machine has **6 physical cores and 12 logical** (`Thread(s) per core: 2`,
one NUMA node). At `threads_batch = 8` two OpenMP workers must therefore land on
SMT siblings, sharing a physical core's execution ports and L1/L2. That produces
the *same* signature observed here — more CPU seconds retired, fewer effective
concurrent cores, longer wall, identical output — with no synchronization story
required. Memory-bandwidth contention is a related candidate: a 35B Q4_K_M model
at ~35.7 GiB streams heavily on a dual-channel mobile part.

So the evidence narrows the mechanism to **at least two** contributors, not one.
Pool growth and SMT oversubscription are not rival hypotheses so much as two
consequences of the same 6 → 8 change, and both are plausibly active. The
classification below is chosen accordingly.

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
used instead; the per-TID delta collector returned no rows, so the thread-level
evidence here is the total count plus process CPU accounting, not a per-worker
breakdown. **The cause of that failure is not established.** The field offsets
are correct and the identical parse works for the process totals, so it is not a
parsing defect — an earlier draft said it was, and that guess is withdrawn. No
replacement cause is asserted either: the collector's bare
`except Exception: pass` meant the real error was never surfaced, and it was not
pursued because the load-bearing datum (`THREADS_AT_IDLE`) comes from a
separate, working read.

## Classification

**E — MIXED_NATIVE_EFFECT.**

The pool-growth mechanism is established: `threads_batch` sizes a process-wide
OpenMP pool, via the startup prewarm, and that is confirmed by source and by the
6-vs-8 idle thread count. What is *not* established is that pool
synchronization is the whole story. On a 6-physical-core SMT part, growing to 8
threads necessarily oversubscribes physical cores, and that alone reproduces the
observed CPU signature. Both effects follow from the same 6 → 8 change and this
design cannot separate them.

An earlier draft classified this **B (SHARED_THREADPOOL_SYNCHRONIZATION_EFFECT,
via the OpenMP runtime rather than a llama.cpp threadpool)**. That label names a
real and correctly identified mechanism, but it asserts an exclusivity the
evidence does not support, so **E** is the honest classification. The
distinction does not change any decision below.

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
| Thermal | Control 86 → 93 °C; candidate 83 → 95 °C — starts differ by exactly 3 °C (gate is 8), no collapse. PASS, with the caveat below |
| Host load at start | Control 1.17, candidate **2.13** — disclosed, not equalised |
| Host policy unchanged | PASS |
| Model loads | 2 (the hard maximum) |

### Confounders this design does not exclude

Two conditions differed between the runs beyond the treatment, and honesty
requires naming them rather than burying them in a PASS:

* **Thermal trajectory.** The candidate ended 2 °C hotter (95 vs 93 °C) and
  heated faster (1.55 vs 1.05 °C/s). On a 15 W i7-10710U, 95 °C is at the
  ceiling where clocks throttle, so part of the candidate's longer wall may be
  reduced sustained frequency rather than threading. The "no collapse" gate
  rules out a catastrophic drop, not gradual throttling.
* **Background load.** The candidate started at loadavg 2.13 against the
  control's 1.17. The runs were sequential (control first), so run-order and
  thermal-soak effects are confounded with the treatment.

With n=1 per arm these cannot be separated from the measured effect. They are
one more reason the classification is E rather than B, and a reason the −14.06 %
figure should be read as directional.

## Consequence

`threads_batch = 8` remains **REJECTED**; the axis stays **CLOSED**. This
investigation explains the rejection rather than reopening it — and it shows the
cost is structural, not incidental: raising `threads_batch` grows a
process-wide OpenMP pool that every later decode pays for.

It also constrains the next tuning step. `batch` and `ubatch` reach
`graph_compute` only through the `batched` boolean
(`llama-context.cpp:1487`) — grepping `n_batch|n_ubatch` against `thread` in
that file yields nothing but profile-logging lines — so they cannot resize the
pool, and `threads_batch` was the uniquely dangerous knob because it alone does.
They are not hazard-*free*, since they change the working-set size per region;
they are free of *this* hazard.

## Budget

2 model loads, 128 generated tokens (2 × 64, plus 2 one-token warm-ups),
~15 s of measured generation.
