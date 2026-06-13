# Performance notes

Orbit is optimized for a specific constraint: running a useful local agentic CLI with Gemma 4 12B on CPU-only, mid-tier hardware.

The goal is not maximum theoretical throughput. The goal is practical end-to-end latency while keeping the runtime model-driven, safe, inspectable, and predictable.

## Baseline assumptions

- Backend: local `llama-server`.
- Model: Gemma 4 12B instruction-tuned GGUF.
- Target profile: CPU-only machine, no dedicated GPU.
- Context window: 8192 tokens.
- Runtime style: model-driven routing and final answers.
- Tools: enabled only by explicit user-selected groups.
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

MTP speculative decoding is optional.

The recommended `llama.cpp` build is the same Gemma 4 compatible fork used for
normal mode. MTP only adds a draft model and speculative decoding flags; it does
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
`llama.cpp` baseline, the build should be treated as part of the tested Orbit
profile. MTP itself remains optional and experimental.

## Tool exposure

Orbit starts with tools disabled.

This is a performance choice as much as a safety choice. When tools are off,
Orbit avoids sending tool schemas and skips the tool loop, reducing new tokens
introduced during prefill. The user explicitly enables only the groups needed
for the current task.

Users enable tools by group:

```text
/tools files
/tools edit
/tools web
/tools shell
/tools files,web
/tools on
```

This keeps the model from seeing unnecessary tools on ordinary chat turns.

The main benefit is not just safety. It also reduces ambiguity and token cost.
A smaller tool surface makes it less likely that the model chooses the wrong
tool, enters a tool loop, or spends extra tokens deciding among irrelevant
capabilities.

In practical terms, tools off means no tool schemas are sent to the model for
ordinary chat. This directly reduces prompt size and prefill work.

`shell-full` is intentionally excluded from normal `on` mode. It is a dangerous
lab mode for unrestricted shell workflows such as reverse engineering,
malware-style static analysis, and custom local tooling. It can be powerful, but
it is also higher-risk and usually more expensive because it may need multiple
tool rounds.

## Model-driven routing

Orbit keeps routing model-driven.

The model receives a compact route prompt and chooses among:

```text
CHAT
FILESYSTEM
FILE_EDIT
WEB
MEDIA
```

When the model can infer the route, tool, and arguments from the user prompt, it is encouraged to return them in the first route response. This avoids an additional tool-selection inference in common cases such as:

```text
read agent.py
list all files in this workdir
summarize https://example.com
```

Orbit still validates the selected route and only exposes tools allowed by the current `/tools` mode.

## Route completeness

A useful route is not only the route label. The best case is:

```text
route + tool + arguments
```

For example:

```text
{"_route":"FILESYSTEM","tool":"list_files","path":"."}
```

This is the case that produced the measured improvement:

- complete route
- one fewer inference
- lower wall time

When the model can return all three in the first inference, Orbit can execute the tool directly and skip a separate tool-selection turn.

The benchmark evidence is simple: removing one inference often saves more wall time than small token-level prompt optimizations. This is especially true on CPU-only systems, where each additional model call adds prefill, generation, and scheduling overhead.

Orbit therefore encourages complete route responses when the user request is clear, while still keeping the decision model-driven and validated by the runtime.

## Chat-only mode

With tools off, Orbit uses a chat-only path.

This avoids sending tool schemas and avoids the tool loop entirely:

```text
prompt -> chat final answer
```

This is the cheapest path for normal questions, writing, explanation, and discussion.

## Tool turn structure

When tools are enabled, Orbit separates the task into bounded phases:

```text
route inference
tool-call inference if needed
tool execution
final inference from tool result
```

The final answer is always generated by the model.

Intermediate route/tool-call turns use lower output budgets than final answers. This prevents expensive `finish=length` behavior in internal control phases and keeps the model from generating prose when only a route or tool call is needed.

## Token budget strategy

Orbit uses different token budgets for different phases:

- Route turns: small budget, enough for compact JSON.
- Tool-call turns: small budget, enough for one tool call.
- Final answers: normal user-facing budget.
- Continuation: explicit `/continue`, never automatic.

This avoids spending generation time on intermediate text that the user will never read.

## Output truncation

Orbit treats `finish_reason=length` as a recoverable UX event, not as a reason to automatically continue.

When the model reaches the output token limit:

- the partial answer remains visible
- Orbit reports that the output stopped because `max_tokens` was reached
- `/continue` can continue the previous answer
- `/max-tokens <n>` can increase the output budget for following turns
- Orbit never auto-continues

This avoids hidden extra inference cost and keeps control with the user.

## Prompt cache awareness

On CPU-only inference, prefill can dominate latency.

Orbit keeps the static prompt structure as stable as possible:

- Stable system prompts.
- Stable phase prompts.
- Stable tool schemas.
- Stable message ordering.
- No prompt rewriting at ingestion time.

This improves the chance that `llama-server` can reuse cached prefix tokens across consecutive turns.

The runtime reports cache-related footer metrics when available:

```text
tks: prompt->completion, cached N | cache: X% | pf .../s | gen .../s
```

## Context discipline

Orbit avoids silently injecting large data into the model.

Important boundaries:

- `read_file` reads UTF-8 text/source files only.
- Complete file reads are bounded.
- Larger files use explicit chunks.
- Fetched web pages are extracted to text and chunked.
- Binary files, PDFs, archives, images, and audio are rejected by `read_file`.

Long inputs are handled through real model-visible chunks, not local deterministic summaries.

Context pressure can affect latency before the context window is close to full.
On CPU-only systems, a session around 50% of the context window can already feel
slower if the active context is dominated by tool results or long assistant
answers. This is why Orbit exposes context status and provides explicit
compaction commands instead of hiding context growth.

## File chunking

For medium and large text/source files, `read_file` returns chunk metadata:

```text
path: chat.py
chunk_index: 0
total_chunks: 5
chars: 0-6000 of 28134
content:
...
```

This gives the model enough real content to analyze while making the remaining available context explicit.

Chunk reads are bounded per turn. This prevents accidental long loops while still letting the model request more content when needed.

## Web content handling

`fetch_url` does not return raw HTML.

It applies:

- explicit `http`/`https` only
- browser-like user-agent
- content-type checks
- bounded download size
- conservative HTML-to-text extraction
- chunk metadata for long pages

Long web documents are processed progressively. Orbit does not save fetched pages into the workdir and does not pretend that the first chunk represents the full document.

## Tool result compactness

Tool results are kept compact and structured.

Examples:

- `search_web` returns bounded title, URL, and snippet.
- `list_files` returns bounded directory entries.
- `exec_shell_command` output is bounded.
- `read_file` provides chunk metadata and real text.

Orbit does not synthesize tool results locally. It only bounds and structures them before passing them to the model.

## Final answer policy

Some categories tend to produce longer answers than needed.

Orbit uses category-specific final-answer instructions after tool results:

- Web search: concise bullets, expand only if asked.
- Shell output: compact findings, preserve numbers and names.
- Shell-full output: evidence-based findings, function/file and exploit impact
  when available, no generic methodology unless asked.
- File reads: respect requested length, otherwise answer concisely.
- Lists: return listed names compactly.

The final response remains model-generated.

## Loop control

Orbit protects against pathological loops without deciding the answer itself.

Controls include:

- maximum tool rounds per task
- repeated tool-call detection
- chunk-read budget
- fetch chunk budget
- timeout enforcement
- unsupported route handling

When a loop limit is reached, Orbit moves to the final inference using the content already available.

Most tool groups use tight round limits. `shell-full` has a larger explicit
budget because lab analysis may legitimately require sequential commands such
as inspecting source, grepping indicators, checking tool availability, or
running local analyzers. Repeated equivalent calls are still detected.

## Session memory

For long sessions, Orbit can refresh memory using the model.

The memory refresh is internal and not saved as a visible user turn.

It keeps:

- system prompt
- model-generated durable session memory
- recent verbatim tail

The refresh is discarded if it is empty, attempts tools, fails, or does not reduce context.

This keeps long interactive sessions usable without replacing the current user prompt or rewriting the latest turn.

## Memory observability

Session memory refresh is model-generated and observable.

`/status` exposes memory state such as:

- refresh threshold
- refresh count
- last before/after token estimate
- last saved token estimate
- total tokens saved
- cooldown state
- last refresh outcome

These fields make it possible to understand whether memory refresh is actually helping instead of treating compaction as invisible runtime magic.

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

Broad shell access exists only as explicit `shell-full` mode and should be used
in isolated lab workdirs.

The runtime can enforce boundaries, but the model remains responsible for reasoning, tool selection, and final answers.

## Practical takeaway

On CPU-only hardware, the biggest wins came from:

1. Keeping tools off by default.
2. Exposing only selected tool groups.
3. Making route/tool-call turns short.
4. Keeping prompt prefixes stable for cache reuse.
5. Bounding and structuring tool results.
6. Chunking long text instead of flooding context.
7. Reducing unnecessarily long final answers.
8. Measuring every change before keeping it.

The result is not a generic agent framework. It is a small, controlled local CLI tuned for a specific model and hardware class.

For CPU-only local agents, reducing unnecessary inferences and unnecessary output often produces larger gains than increasing raw model throughput.
