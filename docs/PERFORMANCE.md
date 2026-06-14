# Performance notes

Orbit is optimized for a specific constraint: running a useful local agentic CLI with Gemma 4 12B on CPU-only, mid-tier hardware.

The goal is not maximum theoretical throughput. The goal is practical end-to-end latency while keeping the runtime model-driven, safe, inspectable, and predictable.

## Baseline assumptions

- Backend: local `llama-server`.
- Model: Gemma 4 12B instruction-tuned GGUF.
- Target profile: CPU-only machine, no dedicated GPU.
- Context window: 8192 tokens.
- Runtime style: model-driven routing and final answers.
- Tools: disabled by default; one explicit shell tool mode when enabled.
- Recommended `llama.cpp` build: Gemma 4 compatible fork/branch documented in
  the README.

Orbit does not replace the model's reasoning with deterministic task answers. The runtime optimizes what the model sees, when tools are exposed, how much context is injected, and when boundaries are enforced.

## Reference hardware

The default profile was tuned on this CPU-only machine:

```text
Machine class: Intel NUC 10 class system
CPU: Intel Core i7-10710U @ 1.10 GHz
Physical cores: 6
Logical CPUs: 12
Architecture: x86_64
L3 cache: 12 MiB
RAM: 62 GiB visible
Swap: 2 GiB
GPU acceleration: none
```

This is a general-purpose workstation class machine, not an AI workstation. The selected defaults favor stability and acceptable latency over aggressive throughput.

## Server profile

The default helper script starts `llama-server` with conservative CPU-oriented settings:

```bash
THREADS=6
BATCH_SIZE=256
UBATCH_SIZE=128
CACHE_RAM=8192
CTX_SIZE=8192
PARALLEL_SLOTS=1
```

The main server flags are:

```bash
llama-server \
  -c 8192 \
  -t 6 \
  -b 256 \
  -ub 128 \
  -np 1 \
  --reasoning off \
  --cache-ram 8192
```

These defaults were chosen for stability on CPU-only systems. Faster machines can raise `THREADS`, `BATCH_SIZE`, `UBATCH_SIZE`, and `CACHE_RAM`, but changes should be benchmarked instead of assumed.

## MTP speculative decoding

MTP speculative decoding is the recommended default startup profile for Orbit.

The recommended `llama.cpp` build is the same Gemma 4 compatible fork used for
standard mode. MTP adds a draft model and speculative decoding flags; it does
not change Orbit's runtime philosophy.

In practical terms, MTP is speculative decoding:

1. The main Gemma 4 12B model remains the authority for the final output.
2. A smaller/specialized draft model proposes upcoming tokens in advance.
3. The main model verifies the proposed tokens.
4. Correct draft tokens can be accepted in batches.
5. Incorrect draft tokens are discarded and generation continues normally.

This can increase generation throughput because the main model does not always
need to generate one token at a time. When the draft model predicts well, the
main model validates multiple tokens more efficiently than producing each one
sequentially.

MTP mostly helps the generation phase (`gen/s`). It does not significantly
reduce prefill cost (`pf/s`), because prefill is dominated by processing the
input prompt, tool schemas, conversation context, and tool results.

Operational tradeoffs:

- It requires a compatible `llama-server` build.
- It requires an additional draft model file.
- It uses more memory than the baseline profile.
- It is most useful when the final answer generates enough tokens to amortize
  the extra draft-model work.

The tested implementation came from:

```text
Repository: https://github.com/qualcomm/llama.cpp
Branch: gemma-4-support-smaller-assistants
```

The tested server profile kept the same CPU-oriented settings used by the
baseline:

```bash
THREADS=6
BATCH_SIZE=256
UBATCH_SIZE=128
CACHE_RAM=8192
CTX_SIZE=8192
PARALLEL_SLOTS=1
```

The only relevant difference was enabling the draft MTP model:

```bash
llama-server \
  -m <gemma-4-12B-it-Q4_K_M.gguf> \
  --spec-type draft-mtp \
  --model-draft <gemma-4-12b-it-Q8_0-MTP.gguf> \
  -c 8192 \
  -t 6 \
  -b 256 \
  -ub 128 \
  -np 1 \
  --reasoning off \
  --cache-ram 8192
```

Observed CPU-only benchmark results:

| Prompt | No MTP | MTP | Wall-time delta |
| --- | ---: | ---: | ---: |
| `hi, who are you?` | `pf 12.5/s`, `gen 3.5/s`, `11s` | `pf 16.5/s`, `gen 7.1/s`, `6s` | ~45% faster |
| `tell me who designed you` | `pf 17.0/s`, `gen 4.2/s`, `6s` | `pf 15.5/s`, `gen 6.7/s`, `5s` | ~17% faster |
| `search online for information about Agenzia per l'Italia Digitale` | `pf 10.7/s`, `gen 3.0/s`, `1m56s` | `pf 11.3/s`, `gen 4.1/s`, `1m28s` | ~24% faster |
| `what configuration does this computer have?` | `pf 10.4/s`, `gen 3.2/s`, `3m00s` | `pf 11.8/s`, `gen 4.6/s`, `2m26s` | ~19% faster |

The main gain was higher generation throughput. Prefill did not change as much,
which is expected: MTP helps most when the final answer has enough generated
tokens to amortize the extra draft model work.

Because this depends on a fork/branch rather than the default upstream
`llama.cpp` baseline, the compatible build and MTP draft model should be treated
as part of the tested Orbit profile.

## Runtime integration methods

Orbit uses a set of runtime techniques to reduce latency without
turning user tasks into deterministic fast paths.

The detailed implementation notes are kept in [Techniques](TECHNIQUES.md).

The main techniques are:

- tools off by default
- separate chat and tools prompts
- stable prompt prefixes for cache reuse
- complete command decisions when obvious
- adaptive prefill estimation
- bounded tool-result reinjection
- chunked long-file handling
- HTML cleanup before reinjection
- explicit URL fetch through `curl`
- generic search through `orbit-web-search`
- model-driven memory refresh
- manual tool-result compaction
- compact final-answer policy

## Terminal UX

The terminal UI is intentionally simple.

Performance-related UX choices:

- streamed final answers
- elapsed-time indicator before first token
- compact tool events
- dim metrics footer
- compact preview for long pasted text
- prompt history without duplicate prompts

The user sees progress immediately, even when CPU-only inference is slow.

## Benchmark discipline

Orbit keeps one public regression benchmark helper:

```bash
scripts/bench-core.sh
```

It exercises chat, file listing, short reads, longer reads, grep, and URL fetch
through the normal CLI path. Deeper profiling should be done with temporary
local scripts or manual measurements, not permanent project scripts.

New performance changes should show measurable benefit before they are kept.

## Benchmark findings

Across the Gemma 4 12B CPU-only benchmark runs, the main bottleneck was not route classification itself.

The highest costs usually came from:

- prefill on large or unstable prompt inputs
- reinjecting large tool results into the final inference
- final answers that were longer than the task required

The strongest improvements came from reducing unnecessary inference steps, keeping tool results bounded, and making final-answer policies more concise.

Backend tuning still matters, especially threads, batch size, micro-batch size, context size, and cache behavior. But after a stable server profile is found, removing avoidable model calls and avoidable output often has a larger practical impact than micro-optimizing backend flags.

## What was intentionally not optimized

Orbit avoids optimizations that would make behavior brittle:

- no deterministic answers for normal user tasks
- no hidden web-to-file save path
- no automatic continuation after `finish=length`
- no broad shell access by default
- no generic browser tool
- no silent local summarization of long text

Broad shell access exists only after `/tools on` and should be used in isolated
lab workdirs.

The runtime can enforce boundaries, but the model remains responsible for reasoning, tool selection, and final answers.

## Practical takeaway

On CPU-only hardware, the biggest wins came from:

1. Keeping tools off by default.
2. Keeping one explicit shell tool surface.
3. Making route/tool-call turns short.
4. Keeping prompt prefixes stable for cache reuse.
5. Bounding and structuring tool results.
6. Chunking long text instead of flooding context.
7. Reducing unnecessarily long final answers.
8. Measuring every change before keeping it.

The result is not a generic agent framework. It is a small, controlled local CLI tuned for a specific model and hardware class.

For CPU-only local agents, reducing unnecessary inferences and unnecessary output often produces larger gains than increasing raw model throughput.
