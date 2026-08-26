# Ornith 1.5 35B-A3B — Supported Models benchmark

Auditable source for the `Ornith 1.5 35B-A3B` row in the README
[Supported Models](../../README.md#supported-models) table.

Every number here is a measurement on one machine. It is not a claim about the
model in general: results move with CPU, memory bandwidth, quantization,
backend build, cache state, and workload.

Raw machine-readable results: [`ornith-1.5-35b-a3b.json`](ornith-1.5-35b-a3b.json).

## Reference system

| Property | Value |
|---|---|
| CPU | Intel Core i7-10710U (NUC10i7FNH), 6 cores / 12 threads |
| RAM | 63 GiB |
| GPU | none |
| OS | Linux 7.0.0-30-generic x86_64 |
| llama.cpp build | b9551 (`379ac66`) |
| Orbit revision | `892cfad` |

## Model

| Property | Value |
|---|---|
| Repository | [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) |
| File | `Ornith-1.5-35B-Q4_K_M.gguf`, 21,713,462,848 bytes (20.22 GiB) |
| SHA-256 | `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f` |
| Quantization | Q4_K_M |
| Profile | `orbit-ornith15-native-v1`, architecture `qwen35moe` |

## Configuration

```bash
orbit server --ctx 8192 --threads 6 --threads-batch 6 --batch 256 --ubatch 128 --think off
```

`ctx=8192`, 6 generation and batch threads, batch/ubatch 256/128, one parallel
slot, temperature 0, thinking off, MTP off.

## Protocol

Harness: `scripts/orbit_qualify.py` against a separately managed server, over
the five qualification fixtures used for this table:

| Workload | Fixture |
|---|---|
| tools-on chat | `chat_route_first` (`optimizations-v1`) |
| pwd route | `pwd_route` (`core-v1`) |
| tool + final | `pwd_final` (`core-v1`) |
| artifact + verify | `json_artifact` (`core-v1`) |
| existing-file modification | `existing_file_modification` (`workflows-v1`) |

One excluded warm-up, then three measured repetitions. All 15 fixture
executions passed, with identity, lifecycle and protocol gates clean.

Column definitions, taken from the harness rather than the column names:

- **Prefill** — `sum(evaluated) / sum(per-call evaluated seconds)` across every
  call of every fixture, from the server's `timings.prompt_per_second`.
  Evaluated tokens only, so cached tokens do not inflate it.
- **Generation** — the same weighting on output tokens, from
  `timings.predicted_per_second`.
- **Tools-on chat** — wall time of the tools-on fixture that routes to CHAT and
  executes no tool.
- **Tool + final** — wall time of the fixture that calls one tool and then
  produces a final answer.
- **Peak RAM** — Linux `VmHWM` of the server process, whole lifetime including
  model load.

## Measured values

### Aggregate rates, per repetition

| Run | Model calls | Prefill | Generation |
|---:|---:|---:|---:|
| 1 | 14 | 29.98 tok/s | 8.30 tok/s |
| 2 | 15 | 26.55 tok/s | 7.04 tok/s |
| 3 | 15 | 28.95 tok/s | 8.27 tok/s |

Published: **~29.0 tok/s** prefill, **~8.3 tok/s** generation.

Every published figure is the median of its measured values, rounded
half-up to one decimal: prefill 28.95 → 29.0, generation 8.27 → 8.3.

### Per-workload wall time

| Workload | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| tools-on chat | 6.79 s | 32.53 s | 37.82 s | see below |
| pwd route | 5.42 s | 7.14 s | 8.19 s | 7.14 s |
| tool + final | 41.41 s | 49.01 s | 53.72 s | **49.01 s** |
| artifact + verify | 161.05 s | 189.80 s | 171.90 s | 171.90 s |
| existing-file modification | 272.53 s | 328.84 s | 321.57 s | 321.57 s |

Published: **~49.0 s** for tool + final.

### Tools-on chat depends on route-prefix cache state

This workload is bimodal, so it was repeated five times and the two states are
reported separately rather than averaged together:

| Run | Fixture wall | Route call | Cached tokens | State |
|---:|---:|---:|---:|---|
| 1 | 6.79 s | 5.56 s | 768 | warm |
| 2 | 32.53 s | 31.03 s | 0 | cold |
| 3 | 37.82 s | 35.81 s | 0 | cold |
| 4 | 36.84 s | 35.05 s | 0 | cold |
| 5 | 10.01 s | 8.08 s | 768 | warm |

The workload is identical in every run — 940 input tokens, same output. Only
the route-prefix checkpoint differs: restoring 768 tokens cuts the route call
from a median of 35.05 s (cold, n=3) to 6.82 s (warm, n=2).

Published: **~36.8 s cold** (median, n=3) **/ ~8.4 s warm** (median, n=2). Both
states are given because a single number here says more about the cache than
about the model.

### Peak RAM

`VmHWM` was **38,422,634,496 bytes = 35.78 GiB**, identical across all runs.
Published: **~35.8 GiB**.

## Comparability with the other rows

The Ornith row was measured with the qualification harness in this repository,
which is the only benchmark protocol the repository actually contains.

The other three rows were published in `ba5c580` and **could not be reproduced
from any data in this repository**. Their values do not appear in any surviving
harness output, log, or result file — only in README revisions. They also differ
substantially from the earlier `ff74ee3` values, most sharply on tools-on chat
(Gemma 31.7 s → 3.0 s, Qwen 3.6 25.1 s → 6.1 s), while their Peak RAM figures
are byte-identical across both tables — consistent with having been carried over
rather than re-measured. The `ff74ee3` set is itself only partly traceable: 12 of
its 15 cells match a recovered baseline document, but Qwen 3.6's tools-on chat
(25.1 s vs 44.63 s) and tool + final (30.5 s vs 43.16 s) do not.

The measurements above show why that column moves so much: it is dominated by
route-prefix cache state, not by model speed. Combining the two sets silently
would misrepresent both, so the README footnote marks rows 1–3 as unverified
against the current harness.

## Reproducing

```bash
orbit server --ctx 8192 --threads 6 --threads-batch 6 --batch 256 --ubatch 128 --think off &
SRV=$!   # server PID, required for VmHWM

python3 scripts/orbit_qualify.py --base-url http://127.0.0.1:12120 \
  --profile orbit-ornith15-native-v1 --server-pid $SRV \
  --fixtures qualification/fixtures/optimizations-v1.json --fixture chat_route_first \
  --output run-chat.json
python3 scripts/orbit_qualify.py --base-url http://127.0.0.1:12120 \
  --profile orbit-ornith15-native-v1 --server-pid $SRV \
  --fixtures qualification/fixtures/core-v1.json \
  --fixture pwd_route --fixture pwd_final --fixture json_artifact \
  --output run-core.json
python3 scripts/orbit_qualify.py --base-url http://127.0.0.1:12120 \
  --profile orbit-ornith15-native-v1 --server-pid $SRV \
  --fixtures qualification/fixtures/workflows-v1.json \
  --fixture existing_file_modification --output run-mod.json
```

Discard the first pass as a warm-up and repeat three times.

## Not measured

- **Warm-only steady state.** Runs are a mix of cold and warm cache; only the
  tools-on chat workload was repeated enough to separate the two.
- **Flash Attention resolved value.** The configuration records `auto`, never
  what it resolved to.
- **`--low-memory`.** Not supported for this profile.
- **TTFT.** The harness reports it as unavailable.
