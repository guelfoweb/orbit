# Tool-Call Early-Stop Shadow

## Scope

This completed probe asked whether native Gemma 4 tool-mode generation
continues after a single complete, canonical-valid tool call. It was
observational only. It did not cancel generation, execute a tool, start
post-tool finalization, alter the sampler, or change the production backend.

The one-time benchmark scanner consumed native SSE token deltas and exact
generation progress counts. It recognized only the production Gemma tool
envelope, reused the existing raw tool-call normalizer, and then applied the
same canonical contract used by active execution. Reports contained only
bounded categories and hashes; raw prompts, output, arguments, paths, URLs,
and evidence were not written. The scanner, harness branch, and focused tests
were removed after the technical stop was established; no early-stop module or
flag remains in production or benchmark code.

## Measured Result

The production-like CPU probe used Gemma 4 12B Q4_K_M, context 8192, six fixed
threads, temperature zero, and a 96-token tool-call budget. All 11 scenarios
were evaluable and cleanup was healthy:

- seven expected calls were canonical-valid and selected the expected tool;
- one adversarial multiple-tool request produced one canonical-valid but
  semantically unwanted call;
- the JSON-example and incomplete-markup negatives did not create a completion
  point;
- the prose-before/after request did not satisfy the strict completion gate;
- no tool was executed and no finalization was started.

All eight observed canonical completion points occurred at the final generated
token. Tokens after the closed envelope were `0` in 8/8 cases; median and p95
were both `0`. The theoretical avoidable decode time was therefore `0 ms`.
There were no false complete detections and no observed trailing prose, markup,
whitespace token, or second candidate after a valid completion point.

This follows the production decode loop: the closed `<tool_call|>` envelope is
the final non-EOG token for these outputs. The next sampled EOG token terminates
the loop before token decoding and before the generated-token counter is
incremented. An application-level stop after the envelope would not avoid a
target decode token.

## Early-Stop Technical Stop

### Conclusion

Active early stopping is not justified. The measured opportunity rate is zero,
while implementation would add streaming/cancellation and lifecycle complexity
to a path that already terminates at the earliest useful boundary.

### Reopening criteria

Reopen only if a production model or template produces a
repeatable population of canonical-valid calls with meaningful trailing
tokens. A new investigation must show all of the following:

- at least 20% of representative canonical-valid tool calls have two or more
  trailing tokens;
- the avoided decode time is outside process and thermal variability;
- zero false completion on adversarial prose, examples, incomplete markup, and
  multiple candidates;
- byte-identical tool name and arguments;
- safe streaming cleanup across cancel, timeout, reset, and restart;
- no interaction with MTP, canonical validation, healing, or prefix reuse.

## ngram-mod Technical Stop

### Measured facts

The current vendor exposes useful ngram drafting in upstream server/common
code, but Orbit production uses a custom one-token Python decode loop and does
not have the batch validation, sampler cloning, KV rollback, and checkpoint
mechanics required to integrate it safely. A staging probe improved a highly
repetitive copy workload, while non-repetitive medium/long controls proposed no
drafts and remained within timing variability. The draft path also retained
additional memory.

### Conclusion

Observed usefulness was limited to a highly repetitive sequence and did not
generalize to the representative non-repetitive controls. No ngram-mod runtime
module, flag, or decode path is active.

### Reopening criteria

Reopen the investigation only if either upstream provides a stable C API for
the required speculative validation and rollback operations, or a new
production-like corpus shows that highly repetitive workloads occur frequently
and receive a repeatable, material benefit. Any later promotion would still
require revision-matched integration, exact greedy output equivalence,
process-isolated ABBA validation, bounded memory, and passing streaming, cancel,
timeout, reset, restart, MTP, and prefix-reuse lifecycle gates.
