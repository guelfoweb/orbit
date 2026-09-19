# ANALYSIS model qualification

Backend compatibility, CHAT qualification and evidence-grounded ANALYSIS are
separate claims. A verified native model profile identifies a supported model,
template and backend integration; it does not certify a generated conclusion.

## Qwen3.8 Flash Next UD-IQ1_M

Backend and CHAT qualification remain valid for the tested Dell configuration:
CPU-only, ctx 4096, threads 10/10, batch/ubatch 256/128, one slot, CPU repack and
MTP off, with the qualified lazy loading. The retained FINISH comparisons used
temperature 0 and thinking off.

This exact model/quantization/configuration **did not pass the tested ANALYSIS
FINISH semantic cases**:

- Given a bounded Fattura source observation and deterministic transformation
  references, it operationally closed a question asking for the complete source
  and structure without establishing that scope. A valid control call did not
  make the conclusion adequately supported.
- Given sufficient evidence of a 40-byte acquisition, it correctly identified
  the acquired bytes, archive and hash, but incorrectly explained the complete
  37-character UTF-8 text as a truncation artifact. The supported main answer
  does not validate that additional explanation.

The omission of the original `missing_fact` field was already corrected in the
tested candidate. Preserving that requirement is necessary information, not a
proof that the model will satisfy it. These observations do not establish that
quantization caused the errors, and do not apply automatically to other Qwen
models, quantizations or configurations.

A local UD-IQ3_XXS diagnostic attempt reached its declared time limit during
prefill, without generating output. Its positive case was not executed.
That comparison provides **no semantic quality result for IQ3** and no evidence
that changing quantization fixes the problem.

## Runtime guarantees and remaining limits

Accepted model answers can end operational work while remaining
`answered_unverified`. Their original questions, requirements, proposed answers
and citations are retained separately from runtime-attested deterministic facts.
This distinction does not make an incorrect or incomplete answer correct.

The runtime composes the canonical Markdown investigation report independently
of optional model narrative. It records evidence and provenance, original
questions, proposed answers, coverage, unknowns and the stop reason, including
when questions remain OPEN or BLOCKED. Missing or incompatible evidence makes
document incompleteness explicit. A complete investigation record is not a claim
that every question was answered or that the artifact's full behavior is known.

Context and work limits can still stop an investigation. FINISH admission and
generation safety are runtime contracts, not guarantees of model reasoning.
See [answers, reports and compatibility](ANALYSIS_COMPLETION.md).

Ornith 1.5 35B-A3B Q4_K_M retains its existing bounded ANALYSIS qualification;
its model-authored answers are subject to the same verification distinction.
This note neither extends those oracles to every investigation nor changes
model availability, selection, defaults or fallback behavior.

Qualification evidence was retained on 2026-09-19 at candidate
`8084e49f1782192e849c75b31630ec82de608309`. The scope above does not claim that
ANALYSIS is completely resolved.
