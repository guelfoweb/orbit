# Deterministic Document Search

## Scope

Orbit intercepts only clear literal or conceptual questions that name exactly
one confined local text file. This prevents a long-document question from
degrading into an accidental `cat` or `sed` prefix read. Ambiguous requests use
the existing route and tool loop. The internal `document_search` primitive is
not a registered model tool, so the production tool schema and its tool-mode
prefill are unchanged.

## Literal Search

A literal request supplies its own exact word or phrase. Runtime validates that
term and scans the complete stable snapshot with no model call. Search uses
strict UTF-8, NFKC normalization, Unicode case folding unless case sensitivity
is explicit, and controlled whole-word boundaries for single words. All
occurrences are counted even when only a bounded number of windows is returned.

A zero-match literal result has `scan_coverage=complete` and
`semantic_coverage=exact`. It proves only that the requested string does not
occur; it does not prove that a related concept is absent.

## Concept Search

One bounded model call receives the original question and stratified samples
from the beginning, middle, and end of the same snapshot. It returns a strict
JSON plan with lowercase ISO 639 language tags and natural-language terms.
Runtime validates only structure and bounds:

- at most three languages;
- at most twelve terms in total;
- at most four words and 96 characters per term;
- no regex, glob, path, command, shell syntax, duplicate key, or duplicate
  normalized term.

The model selects languages, translations, and synonyms. Runtime does not add,
rank, rewrite, or repair semantic terms. Invalid or truncated plans fail closed
without falling back to shell discovery.

The internal scanner examines the complete snapshot. If matches exist, one
bounded model call decides whether the exact windows support the requested
concept or are merely lexical coincidences. A supported result must reference
an existing returned line range; Orbit appends the corresponding exact bounded
window.

If no term matches, `scan_coverage=complete` but
`semantic_coverage=partial`. The result explicitly states that indirect
formulations remain possible. Orbit performs one full-document inference only
when the production tokenizer proves that the entire rendered document prompt,
output reserve, and safety margin fit the active context. Otherwise no semantic
chunking or additional inference is attempted.

## Coverage Fields

- `file_coverage` reports how much document text was visible to the answering
  model or returned directly: `complete`, `partial`, or `none`.
- `scan_coverage` reports whether the lexical scanner covered the entire stable
  snapshot: `complete` or `none`.
- `semantic_coverage` is `exact` for literal presence, `partial` for bounded
  conceptual terms and windows, or `complete` only for admitted full-document
  analysis.
- `search_mode` distinguishes `literal`, `concept`, and `full_document`.

The result also carries searched terms, detected document languages, complete
match count, returned-window count, truncation, exact line ranges, and the file
SHA-256. These fields are intentionally separate: a complete lexical scan is
not complete semantic understanding.

## Safety And Lifecycle

Search reuses the full-document snapshot contract: descriptor-relative opens,
`O_NOFOLLOW`, regular files only, strict UTF-8, bounded size, coherent size and
SHA-256, and device/inode/version attestation. Language samples, matches, and
windows always belong to the same snapshot. Source changes fail closed before
output becomes visible.

The snapshot sidecar is ephemeral. Success, parser failure, model failure,
cancellation, timeout, reset, and later process startup use the existing
cleanup paths. Diagnostics expose only route, mode, language tags, counts,
truncation, escalation, bounded failure reason, and timings; terms and document
content are not logged.

## Non-Goals

This path does not add a production tool, semantic ranking, embeddings, RAG,
regex supplied by the model, shell execution, semantic chunking, map-reduce,
or a definitive conceptual negative from partial semantic coverage.

## Measured Validation

Process-isolated CPU-only Gemma 4 26B-A4B Q4_0 runs used clean workdirs and
MTP disabled. Italian-question/English-document, English-question/Italian-
document, and mixed-language concept cases each passed three repetitions with
exact supporting ranges. Median wall times were 49.286, 63.383, and 72.840
seconds respectively; these values are descriptive and are not deterministic
performance claims.

A literal match after line 100 required no model call and scanned the complete
snapshot in approximately 0.689 ms. A 12,000-line concept-negative fixture used
one planning call and approximately 24.485 ms of internal search time. Because
the full document did not fit, the result remained explicitly non-definitive.
An incidental `what the hell` match was classified as lexical-only rather than
supporting the requested religious concept.

The production tool definitions and tool-call prompt sources are unchanged.
The accepted 708-token tool-mode baseline therefore has a zero-token schema
delta. The existing twelve-scenario production corpus remained 12/12 correct;
its CPU timing variation is not attributed to this route because no corpus
prompt selected document search.
