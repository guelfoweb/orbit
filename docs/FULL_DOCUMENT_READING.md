# Long-File Reading

## Contract

Orbit never presents a prefix or a search window as a complete file read.
Every long-file result reports the relative path, total bytes and lines,
SHA-256, exact returned ranges, and complete, partial, or no coverage.

For clear targeted-search requests naming exactly one local file, runtime
selects the internal literal or conceptual document-search path before normal
tool selection. Literal terms come directly from the request. For conceptual
search, one bounded model call selects translations and synonyms from
stratified language samples, and one bounded model call verifies positive
source windows. Ambiguous requests retain the normal model-driven route. For an
explicit integral request naming exactly one local file, runtime performs
deterministic admission after the existing route decision and before tool
selection.

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

Clear literal and conceptual searches naming exactly one local file use an
internal `document_search` primitive. It is not registered in the production
tool schema and never executes shell. Search runs over one immutable UTF-8
snapshot using NFKC normalization, Unicode case folding, exact phrases, and
controlled whole-word boundaries. It counts all occurrences, returns bounded
deduplicated windows in source order, and preserves exact line ranges and the
snapshot SHA-256.

Literal search extracts one exact word or phrase from the request and does not
add a model call. A complete literal scan may support a definitive statement
about whether that exact string occurs. Conceptual search uses one bounded
model call to identify up to three document languages and at most twelve
validated translations or synonyms. The planner sees only stratified samples
from the beginning, middle, and end for language and term selection; those
samples are not answer evidence. A second bounded call checks whether returned
exact windows are semantically relevant and must cite existing line ranges.

`scan_coverage=complete` means every line was searched, while
`semantic_coverage=partial` means that lexical terms cannot exhaust every way a
concept may be expressed. Therefore a zero-match conceptual search cannot
support a definitive semantic negative. If the exact full-document admission
check proves that the complete document fits, Orbit escalates once to the
existing full-document path and reports `semantic_coverage=complete`.
Otherwise it reports the searched terms and a bounded non-definitive result.
No chunk inference, map-reduce pass, semantic ranker, or hidden retry exists.

Source identity is attested before and after planning, scanning, and model
verification. Concurrent replacement or mutation discards the result. An
ambiguous file or request fails safely to the normal route rather than choosing
a path in runtime.

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
- A complete literal scan found a sentinel at line 219 without a route call or
  model-selected `cat`; the internal search took approximately 0.689 ms.
- Italian-question/English-document, English-question/Italian-document, and
  mixed Italian/English concept cases each passed three clean repetitions. The
  model selected document-language terms and the verifier cited exact evidence
  at line 160, line 150, and lines 110 and 210 respectively.
- The English-question/Italian-document plans included terms such as `inferno`,
  `abisso`, `dannazione`, and `luogo di tormento`. The mixed-document plans
  included both `cybersecurity` and `sicurezza informatica` families. Runtime
  did not add or translate those terms.
- An incidental `what the hell` match was not accepted as evidence that the
  document discussed the religious concept. A 12,000-line zero-match concept
  case scanned the full snapshot but retained partial semantic coverage and a
  non-definitive answer because the document did not fit in one context.

The supported common case is targeted positive retrieval. Exact full analysis
is supported only when tokenizer admission proves that the complete document
fits safely in one context.

| Scenario | Result | Model calls | Evaluated prompt tokens | Output tokens | Wall |
| --- | --- | ---: | ---: | ---: | ---: |
| Literal line-219 positive | Exact complete scan; no model inference | 0 | 0 | 0 | 0.689 ms search |
| Italian question, English document, 3/3 | Correct line 160; partial semantic coverage | 2 median | 1,286 median | 86 median | 49.286 s median |
| English question, Italian document, 3/3 | Correct line 150; partial semantic coverage | 2 median | 1,597 median | 102 median | 63.383 s median |
| Mixed-language document, 3/3 | Correct lines 110 and 210; partial semantic coverage | 2 median | 1,724 median | 136 median | 72.840 s median |
| Incidental `what the hell` | Lexical-only match; no unsupported concept claim | 2 | 1,554 | 114 | 70.690 s |
| 12,000-line concept negative outside context | Full lexical scan; prudent semantic result | 1 | 928 | 63 | 41.180 s |
| Exact full fit, beginning/middle/end plus negative | Correct; complete coverage | 2 | 3,086 | 38 | 109 s |
| Long negative, exact prompt does not fit | Safe refusal; coverage none; `--ctx 18432` | 1 | 46 | 8 | 2 s |
| Rejected seven-chunk protocol plus synthesis | Incorrect records; fail-closed synthesis | 8 | 10,686 | 410 | 413.6 s |

The literal, concept-positive, and exact-fit rows are successful answers. The
concept-negative row is deliberately non-definitive. The long full-document
negative row is a correct refusal, not a semantic answer. The rejected chunk
row is retained only as technical-stop evidence. Timings came from one
CPU-only machine and are descriptive, not deterministic performance claims.

The internal reader is absent from the production model schema. The normal
tool-mode prompt therefore retains the pre-change tool surface. The schema and
tool-call prompt sources are byte-identical to the baseline, so the measured
708-token production baseline has a zero-token delta. An exact tokenizer probe
of the edit fixture rendered 943 tokens before and after removal of the visible
reader. The preflight adds no model call to direct chat, targeted reads,
searches, or other existing shell paths.

The pre-search full-document baseline production corpus completed 12/12
scenarios with 27 model calls, 4,863 evaluated prefill tokens, 656 output
tokens, and 220 seconds observed wall time. The deterministic-search branch
also completed 12/12 with `finish_reason=stop`: 31 model calls, 19,343 input
tokens, 11,642 cached tokens, 7,701 evaluated prefill tokens, 739 output tokens,
and 406.686 seconds observed wall time. None of those corpus prompts selected
the document-search route, the tool schema and prompt source were unchanged,
and the variation is treated as model/run and CPU variability rather than a
search-path performance change. The previously failing edit workflow still
executed and verified its mutation. No performance improvement is claimed.
