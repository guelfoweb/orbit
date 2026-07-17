# Native Backend Capability Manifest

Orbit exposes a bounded, observational Gemma 4 compatibility manifest under
`native_backend_capabilities` in the native server `/props` response.

The manifest records:

- the `llama.cpp` build number, commit, target, compiler, and runtime-library hash;
- the Orbit Gemma 4 renderer profile and a versioned fixture suite covering
  tool declarations, tool generation, structured argument rendering, and tool
  responses;
- the exact 64-token `final_from_tool` prefix hash and next dynamic token;
- whether the current build, renderer, and tokenizer match the validated profile.

The initial profile is `orbit-gemma4-native-v1`. A `verified` status requires
the checked-in renderer fixtures, the production tokenizer probe, and an
explicitly reviewed `llama.cpp` commit to match. New backend revisions remain
usable but report `backend_unverified` until their conformance corpus is
reviewed.

`verified` is deliberately scoped to backend identity and prompt/tokenizer
conformance. It is not a claim that every inference output is equivalent or
that a backend revision is performance-positive.

This manifest does not alter startup, inference, routing, tool execution,
healing, MTP, or final-prefix eligibility. It is intended to make backend and
template drift visible before a behavioral compatibility decision is made.

The manifest contains hashes and bounded build metadata only. It does not
include prompts, model output, evidence, tool arguments, environment values, or
model paths.

Renderer fixtures are golden hashes, not self-derived expectations. A drifted
fixture identifies the affected rendering family while the aggregate suite
hash provides one bounded compatibility identity. Fixture text and synthetic
arguments are never exposed through `/props` or benchmark JSONL.

The `--tool-call-generation-only` smoke-harness mode records this redacted
identity together with a versioned corpus hash. This makes results comparable
across backend and template revisions without copying corpus prompts into the
JSONL output. The manifest is stored once in the environment and summary rows,
not repeated for every generated sample.

Use `scripts/compare_tool_call_generation.py BASELINE.jsonl CANDIDATE.jsonl`
for an offline comparison. The comparison refuses different corpus or sample
sets, tool-mode protocol fingerprints, or comparable runtime configurations.
It reports backend/template identity changes and fails on new semantic
tool-selection failures, markup leakage, multiple candidates, tool execution,
finalization, or additional model calls. Timing deltas are reported but are not
treated as deterministic correctness evidence.

A new, unreviewed backend commit remains comparable: the behavioral corpus is
the evidence needed to review it. A missing manifest or renderer/tokenizer
mismatch is instead non-comparable because the generation-only corpus does not
prove final-prefix or template conformance.
