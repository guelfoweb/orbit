# Performance notes

Orbit is tuned for a practical local-agent constraint: Gemma 4 on CPU-only hardware, low operational complexity, and model-driven behavior.

The goal is not maximum synthetic throughput. The goal is predictable end-to-end latency without turning Orbit into a deterministic workflow engine.

## Baseline assumptions

- primary backend: native `orbit server`
- compatibility backend: `llama-server` or another OpenAI-compatible local backend
- model family: Gemma 4 GGUF
- context target: 8192 tokens
- tools: off by default
- runtime style: model-driven routing, tool loop, and final answer

## Native backend profile

The stable CPU-oriented native profile is:

```text
threads=6
threads_batch=6
ctx_size=8192
batch_size=256
ubatch_size=128
parallel_slots=1
```

These defaults favor stability on CPU-only systems. Change them only with comparable benchmarks.

On Apple Silicon, thread count is not automatically "all cores". Local
measurements have shown that using only performance cores can beat using
performance plus efficiency cores for token-by-token generation. On thermally
constrained Intel laptops, extra threads can also be erased by throttling.
Treat thread changes as workload- and machine-specific.

## GPU and compatibility backends

Native `orbit server` is CPU-first and currently configures `gpu_layers=0`.
It does not expose native GPU offload as a normal server option.

GPU measurements should use a separate OpenAI-compatible backend, for example
`llama-server` built with Metal/CUDA and started with GPU offload enabled. Point
Orbit at that backend with:

```bash
orbit bench-core --base-url http://127.0.0.1:8080
```

Do not include `/v1` in the base URL; Orbit appends the OpenAI-compatible paths.

Interpret GPU results as backend comparisons. They are useful, but they are not
native `orbit server` results. MLX/Ollama results are even less directly
comparable when quantization, context length, and tool-call behavior differ.

## MTP

Native MTP is supported through:

- target model GGUF
- draft MTP GGUF
- persistent session state
- conservative fallback to standard generation

Practical points:

- MTP primarily helps generation, not prompt prefill
- long or tool-heavy prompts are still dominated by prefill and reinjection costs
- CPU-only latency is sensitive to scheduler noise
- comparative MTP benchmarks should use fixed CPU affinity, for example `taskset -c 0-5`

## Multimodal

Multimodal support adds a third artifact:

- target model
- draft MTP model, if MTP is enabled
- `mmproj` model for image/audio support

This affects startup and memory footprint more than token generation speed.

## Runtime techniques that matter most

The main latency wins in Orbit have come from:

- tools off by default
- keeping chat and tool prompts separate
- stable prompt prefixes for reuse
- compact dedicated tools for common noisy tasks
- local capability discovery in the tools-on prompt
- bounded tool-result reinjection
- content-evidence guards that prevent ungrounded retries from becoming user-visible answers
- short final answers by policy
- explicit memory refresh instead of uncontrolled context growth

Implementation notes are in [TECHNIQUES.md](TECHNIQUES.md) and [RUNTIME_TOOLING_AND_EVIDENCE.md](RUNTIME_TOOLING_AND_EVIDENCE.md).

## Runtime tooling for lower prefill

Large shell outputs are a major CPU-only bottleneck because they are reinjected into the next model pass.

Orbit uses dedicated model-guided tools for common high-noise cases:

- `fetch_url` normalizes direct URL fetch evidence instead of reinjecting curl progress meters or raw transport noise.
- `list_directory` returns bounded deterministic listings instead of large `find`, `ls -R`, or `tree` output.
- `system_info` returns compact machine specs instead of verbose `lscpu`, `free`, `df`, `uname`, or `/proc` dumps.
- startup capability discovery tells the model which local document utilities are available before it chooses a command.

These tools reduce context noise without adding deterministic task routing. The model still chooses the tool.

## Benchmark discipline

Public regression benchmark:

```bash
orbit bench-core --base-url http://127.0.0.1:12120
```

Rules:

- compare like with like
- keep prompt, config, model, and backend mode identical
- use fixed CPU affinity for serious CPU-only measurements
- treat single-run differences as noise unless repeated
- record the exact Orbit commit or tag with `git rev-parse HEAD`
- record the model artifact, quantization, context size, threads,
  `threads-batch`, MTP state, tools mode, and startup prewarm state
- keep raw benchmark output or logs outside the repository workdir unless they
  are intentionally sanitized fixtures

Minimum metadata to attach to any benchmark note:

```text
orbit_commit=<git rev-parse HEAD>
orbit_tag=<tag or none>
backend=native|llama-server|other
model=<repo/file or local path>
quant=<for example Q4_K_M>
ctx=8192
threads=<n>
threads_batch=<n>
mtp=on|off
tools=on|off
prewarm=on|off
platform=<OS/CPU/GPU>
```

## Exploratory hardware observations

Local exploratory tests on Gemma 4 12B Q4_K_M are consistent with common
llama.cpp behavior:

- generation is often memory-bandwidth-bound, not pure compute-bound
- prefill benefits more from GPU/offload and larger batch compute
- laptop Intel CPUs can be dominated by thermal throttling
- Apple Silicon CPU results depend strongly on performance-core thread counts
- external GPU backends can greatly improve prefill, but they are separate from
  native Orbit CPU behavior

These observations are useful for choosing hardware and benchmark matrices.
They should not be treated as release gates unless the exact commit, backend,
model, configuration, and repeated raw results are recorded.

## Main bottlenecks observed

The dominant costs have usually been:

- prefill on large prompts, tool schemas, and tool results
- long final generations
- reinjection of large command output
- scheduler noise on CPU-only runs

Micro-optimizing one function is less useful than reducing unnecessary inference work.

## KV cache reuse planning

KV cache reuse work must start from measurement, not implementation.

The current planning document is [KV_CACHE_REUSE_PLAN.md](KV_CACHE_REUSE_PLAN.md). It defines stable prompt-prefix candidates, invalidation risks, and the benchmark matrix to run before any cache reuse patch.

## What Orbit intentionally does not optimize

Orbit avoids optimizations that make behavior brittle or hide correctness issues:

- deterministic task-specific fast paths
- hidden route rewrites
- prompt surgery to fake speedups
- broad architectural changes without benchmark evidence
