# Performance notes

Orbit is tuned for a practical local-agent constraint: Gemma 4 on CPU-only hardware, low operational complexity, and model-driven behavior.

The goal is not maximum synthetic throughput. The goal is predictable end-to-end latency without turning Orbit into a deterministic workflow engine.

## Baseline assumptions

- primary backend: native `orbit-server`
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
- bounded tool-result reinjection
- short final answers by policy
- explicit memory refresh instead of uncontrolled context growth

Implementation notes are in [TECHNIQUES.md](TECHNIQUES.md).

## Benchmark discipline

Public regression benchmark:

```bash
orbit bench-core --base-url http://127.0.0.1:11976
```

Rules:

- compare like with like
- keep prompt, config, model, and backend mode identical
- use fixed CPU affinity for serious CPU-only measurements
- treat single-run differences as noise unless repeated

## Main bottlenecks observed

The dominant costs have usually been:

- prefill on large prompts, tool schemas, and tool results
- long final generations
- reinjection of large command output
- scheduler noise on CPU-only runs

Micro-optimizing one function is less useful than reducing unnecessary inference work.

## What Orbit intentionally does not optimize

Orbit avoids optimizations that make behavior brittle or hide correctness issues:

- deterministic task-specific fast paths
- hidden route rewrites
- prompt surgery to fake speedups
- broad architectural changes without benchmark evidence
