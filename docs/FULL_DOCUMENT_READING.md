# Long-File Reading

## Contract

Orbit never presents a prefix or a search window as a complete file read.
Every long-file result reports the relative path, total bytes and lines,
SHA-256, exact returned ranges, and complete, partial, or no coverage.

For non-integral requests, the model chooses every tool, path, search
expression, synonym, translation, and final answer. For an explicit integral
request naming exactly one local file, runtime performs deterministic admission
after the existing route decision and before tool selection. Runtime otherwise
performs only path confinement, exact extraction, integrity checks, tokenizer
admission, output bounds, and diagnostics.

## Three Paths

### Exact display or export

The internal file-reading primitive returns an exact UTF-8 page without asking
the model to reproduce the content. It is not registered in the production
model tool schema. Pages are bounded to 200 lines and 64 KiB and include a
SHA-bound continuation cursor. A changed file invalidates the cursor instead
of mixing pages from different snapshots. Binary, non-UTF-8, unsafe, oversized,
symlinked, and non-regular files fail closed. Reads use descriptor-relative
no-follow traversal and verify the inode and file version before exposing
content.

The model may select an exact `cat` or `sed -n X,Yp` command. For long files,
Orbit satisfies those exact read operations with the internal bounded page
primitive instead of placing an unbounded shell result in context. This does
not select a command for the model or turn a page into a full analysis.

### Targeted question or search

For a targeted fact, the model selects one bounded `grep` or `rg` search. It
may put exact terms, synonyms, stems, or translations in the same expression.
Orbit does not generate terms, rank matches, or silently run fallback queries.
For a recognized single-file search, Orbit attaches the immutable file
identity and exact returned line ranges without rewriting the command.

The final model verifies the retrieved passage semantically. A positive answer
may stop on clear source evidence. Search evidence always reports partial
semantic coverage, even when the literal search examined the complete file.
A negative or exhaustive answer is valid only after complete semantic coverage.
If a model-selected search returns no passages, Orbit returns a bounded partial-
coverage notice directly and does not ask a final model call to turn lexical
absence into semantic absence. Recognized searches attest the source before and
after execution and discard results if the source changed.

### Full-document analysis

After the existing route identifies a filesystem request, a narrow preflight
recognizes an explicit full or integral analysis naming exactly one local UTF-8
file, up to 1 MiB. This happens before the model can choose a file-reading shell
command. The native backend tokenizes the exact rendered analysis prompt
without decoding or mutating session state. Orbit admits one full-document
inference only when the rendered input, bounded output reserve, and 256-token
safety margin fit the active context.

Admission performs two identical exact tokenizer/context attestations around
the document-token count, then reattests path, device, inode, version, size and
SHA-256 immediately before inference. A tokenizer, template, context or source
change fails closed. The source is reattested once more before any buffered
model output is returned, so a concurrent replacement discards the conclusion.
The final inference registers no tools, so the exact rendered count contains
the complete system prompt, user request, document wrapper and template, with a
zero-length tool-schema section; output reserve and safety margin are added
separately.

If the exact prompt does not fit, Orbit fails closed. It reports no semantic
coverage, exact file metadata, the required context, and a rounded `orbit
server --ctx N` recommendation. It never sends a prefix to the model or calls
that prefix a complete read. `/max-tokens` changes response length; it does not
increase server context.

External OpenAI-compatible backends that do not expose Orbit's exact tokenizer
endpoint also fail closed rather than estimating admission from bytes or
characters.

## Chunk-Coverage Technical Stop

Deterministic chunk accounting was mechanically correct, but the model record
protocol failed the real-model gate and is not active. On a 29,124-byte,
540-line fixture, seven non-overlapping chunks covered bytes 0-29,124 and lines
1-540. All seven calls stopped normally, but 0/7 structured records passed the
source-range contract. The model selected an oversized 77-line range, omitted
required boundary facts, and in a simplified line-number probe confused a
technical delimiter with source evidence and missed the final-line fact.

The failed run used 10,665 cumulative chunk prompt tokens, 363 chunk output
tokens, and 413.6 seconds including final synthesis. The synthesis correctly
reported that no supported conclusion could be made. This proves fail-closed
handling, not useful full-document analysis.

Reopen chunk coverage only for a model/template that repeatedly produces valid
bounded records and passes beginning, middle, end, and negative-fact tests with
complete coverage and no invented evidence. Do not reintroduce the removed
runtime chunk loop from unit tests alone.

A second, materially different probe used one tool-mode record containing only
a bounded summary and model-selected absolute evidence line numbers. The model
identified lines 1, 270, and 540 correctly. The first four-chunk pass emitted
valid structured calls for the three relevant chunks but returned prose for the
irrelevant chunk; a stronger instruction made that isolated empty record valid.
This improved format reliability but did not pass the production performance
gate.

The integrated tokenizer-maximized smoke was cancelled after 158 seconds while
the first chunk was still at native prefill progress 1,844/8,385. It had not
completed a single chunk record or started synthesis. Maximizing each chunk to
the context boundary reduced call count but made latency unacceptable on the
CPU target even for the 29,124-byte fixture. Smaller chunks completed, but the
earlier seven-chunk run still required 413.6 seconds and failed its record
contract. SSD sidecars cannot reduce this model-attention cost.

Conclusion: no active chunk-analysis path meets correctness, stability, and
latency together. Reopening requires both repeated structured-record success
and an end-to-end CPU result whose cumulative prefill and wall time are useful
for the target file sizes. A successful unit protocol alone is insufficient.

## SSD Sidecar

The existing evidence sidecar temporarily stores the immutable raw snapshot.
Large raw evidence is not duplicated in the in-memory evidence cache after a
successful write. Full-document sidecars are marked ephemeral and removed at
the end of the request, including failure and cancellation paths; reset removes
them, and a later process removes a crash-abandoned snapshot while loading its
index. Missing or corrupt sidecar data is never considered complete evidence.

The sidecar bounds memory and preserves integrity; it does not accelerate model
attention. Generated summaries, tokenizer state, and KV data are not persisted.
Disk read and hash cost is small compared with CPU prefill.

## Measured Paths

- Exact display returned complete content directly, without a final model call.
- Exact full analysis of an 11,690-byte, 220-line fixture fit safely at context
  8,192. It reported complete coverage and the file SHA, found facts at lines 1,
  110, and 220, and correctly reported the absent sentinel. The final natural
  run used two model calls, 3,086 evaluated prompt tokens, 38 output tokens, and
  109 seconds. No reading tool was selected; the preflight followed the normal
  route decision directly. CPU timing is descriptive.
- A beginning/middle/end positive lookup naturally selected one bounded regex,
  returned lines 1, 270, and 540 with full file identity, and stopped correctly
  in two model calls. Coverage remained explicitly partial semantic retrieval.
- A multilingual lookup used model-selected English alternatives and matched
  the relevant Italian passage, then stopped on exact line evidence.
- Natural negative probes did not select complete analysis reliably. A zero-
  match targeted result now terminates with an explicit partial-coverage
  refusal, so it cannot become a definitive model-authored negative. No
  automatic semantic scan is active, and no production-readiness claim is made
  for long-file negative questions at an insufficient context.

The supported common case is targeted positive retrieval. Exact full analysis
is supported only when tokenizer admission proves that the complete document
fits safely in one context.

| Scenario | Result | Model calls | Evaluated prompt tokens | Output tokens | Wall |
| --- | --- | ---: | ---: | ---: | ---: |
| Three-position targeted positive | Correct lines 1, 270, and 540; partial semantic coverage | 2 | 1,943 | 99 | 69 s |
| Cross-language targeted positive | Correct Italian passage from model-selected alternatives | 2 | 1,281 | 50 | 48 s |
| Exact full fit, beginning/middle/end plus negative | Correct; complete coverage | 2 | 3,086 | 38 | 109 s |
| Long negative, exact prompt does not fit | Safe refusal; coverage none; `--ctx 18432` | 1 | 46 | 8 | 2 s |
| Rejected seven-chunk protocol plus synthesis | Incorrect records; fail-closed synthesis | 8 | 10,686 | 410 | 413.6 s |

The targeted and exact-fit rows are successful answers. The long negative row
is a correct refusal, not a semantic answer. The rejected chunk row is retained
only as technical-stop evidence. Timings came from one CPU-only machine and are
not deterministic performance claims.

The internal reader is absent from the production model schema. The normal
tool-mode prompt therefore retains the pre-change tool surface. The schema and
tool-call prompt sources are byte-identical to the baseline, so the measured
708-token production baseline has a zero-token delta. An exact tokenizer probe
of the edit fixture rendered 943 tokens before and after removal of the visible
reader. The preflight adds no model call to direct chat, targeted reads,
searches, or other existing shell paths.

The final production corpus completed 12/12 scenarios with `finish_reason=stop`:
27 model calls, 15,647 input tokens, 656 output tokens, 4,863 evaluated prefill
tokens, 10,784 cached tokens, and 220 seconds observed wall time. CPU wall time
is descriptive. The previously failing edit workflow executed and verified its
mutation in three independent final-patch runs; no prose-to-command repair was
needed.
