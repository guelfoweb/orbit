# REPORT accounting: complete request before optional excerpts

The optional model narrative could fail admission because its excerpt allocator
priced the REPORT instruction and deterministic sections, but not the complete
question/dossier or chat framing. The final admission correctly refused the
oversized request. The canonical runtime document remained available.

## Isolated change

Only `_report_messages` and `_evidence_cards` change. Before an optional excerpt
replaces its provenance card at additional cost, the allocator renders the
complete candidate request and calls the existing exact admission path with
the same effective output cap, no tools, no next-action reserve and no evidence
rehydration. References in this request remain provenance, not read requests.

The original dossier, instructions, deterministic sections, indicators and
existing bounded record-selection policy are unchanged. Excerpts are offered
in the existing order. A short complete record that costs less than its
provenance card still contributes its saving; final admission remains mandatory.
If required context cannot fit, no model call is dispatched. The runtime
continues to publish its canonical Markdown document with the optional narrative
explicitly unavailable. No controller decision is changed.

This extracts only the REPORT accounting change from the preserved Ornith
recovery candidate. It does not include raw archive access, duplicate
suppression, FINISH reference ownership, or changes to PLAN/STEP/FINISH.

## Retained replay

The frozen main Fattura request was reconstructed from its retained dossier and
re-attested EvidenceStore using the integrated Ornith chat renderer and tokenizer
in vocab-only mode. No model generation or sample execution ran.

| Same retained request | Before | With accounting fix |
|---|---:|---:|
| Complete input tokens | 7131 | 5278 |
| Input allowance | 5888 | 5888 |
| Output / safety reserve | 2048 / 256 | 2048 / 256 |
| Exact admission | Refused | Admitted |

Context remains 8192. The dossier, required grounding, four selected evidence
records, all five transform records and canonical report are byte-identical.
The current reconstruction matches the historical token hashes, not just its
counts. The only changed prompt section is the optional evidence-card body.

The omitted WMI excerpt remains archived and in the canonical document; it is
no longer delivered in this optional model request. **This is not lossless
prompt compression or a claim of improved model interpretation.** No full
Ornith semantic non-regression or Qwen ANALYSIS qualification follows from it.

Artifacts: `workdir/diag/analysis_report_accounting_1/`, including original and
candidate messages, rendered prompts/token IDs, replay identities, unchanged
canonical documents, pre-fix failures and applied mutation results. The sample
and historical diagnostics remain local, not new repository fixtures.

## Deterministic qualification

`tests.test_analysis_report_dossier_admission` exercises real recording,
rendering, admission and dispatch seams with safe synthetic observations:

- complete dossier and provenance priority; small and oversized dossiers;
- exact boundary and one-token overflow, including excerpt-free requests;
- large optional bodies and non-character token costs;
- complete template cost, explicit lower context/output limits;
- cheaper complete records, unknown token counts and no rehydration;
- no history/evidence mutation and canonical document recovery on refusal.

Applied mutations cover ignoring full-request cost, dropping the dossier or
provenance, accepting/rejecting all quotes, discarding savings, ignoring the
output reserve and rehydrating citations. Each is caught by the tests.

Run the focused report tests, mandatory `tests.test_analysis_cross_sample_gate`
and full unittest discovery according to `AGENTS.md`. The canonical semantic
baseline is applied to source/stage identity and provenance preservation across
Fattura, IBAN and mine.hta. Its model-narrative criteria remain `MANUAL_CHECK`;
no narrative is generated or graded by the accounting tests.

## Cost and limits

The change adds one exact complete-request probe per positive-cost excerpt
considered (at most the existing twelve-record limit), plus final admission.
It adds no model calls, actions or retries. Tokenizer probes can add local or
endpoint round trips; no end-to-end performance claim is made.

Existing evidence selection, per-card bounds and non-exact-backend policy are
unchanged. An unavailable tokenizer cannot justify exact admission. A required
minimum exceeding the allowance still fails closed rather than cutting the
dossier, increasing context or reducing output space. Accounting does not
establish the truth of a model's optional narrative.
