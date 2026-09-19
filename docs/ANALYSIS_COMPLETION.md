# ANALYSIS answers and verification

A FINISH control call is a model proposal. `status=resolved` remains accepted
on the model-facing wire, but a non-empty answer with reattestable citations
is recorded as `answered_unverified`. It ends automatic work on that question;
it does not attest the truth of its summary or narrow the original question.
The original question, missing fact, summary and citations remain in the ledger.
`open` and `blocked` retain their existing action and repair limits.

The runtime's deterministic results have their own provenance and validation.
A valid evidence identifier establishes that a record exists, not that every
claim beside it follows from that record. Complete source acquisition is not
proof of an arbitrary conclusion, and is not by itself full delivery to a model.

## Public results and compatibility

- `answered_unverified_questions` names questions where a model answer ended
  operational work. It does not certify those answers.
- `open_questions` retains its previous operational meaning: both still-open
  and blocked questions that did not receive an accepted answer.
- `unverified_questions` includes every free-form question, including answers.
  Deterministic facts remain available independently of these question states.
- `resolved_questions` and the controller's `resolved` count are deprecated and
  empty/zero for new runs. Do not use them as the operational completion count.
- Loading a legacy `QuestionState(status="resolved")` maps it to
  `answered_unverified` and records `legacy_status="resolved"`. Loading an
  `AutonomousRunResult` with nonempty `resolved_questions` preserves those IDs
  in `legacy_resolved_questions`, adds them to the unverified answer fields,
  and empties the deprecated field. This is not a verification upgrade.
- Existing diagnostic JSON and saved Markdown are historical artifacts and are
  not rewritten or retroactively certified. Raw historical `resolved` labels
  must be interpreted as model decisions under the old contract.

The dossier includes answered questions even when no question remains open.
It labels summaries as unverified claims and citations as references made by
those claims. Progress events use `answered_unverified`; the terminal renders
legacy `resolved` events as unverified legacy answers.

This separation alone does not validate arbitrary model narrative or guarantee
report generation. It does not import an objective catalogue or infer evidence
obligations from question text.


## The runtime document and optional narrative

`report.text` is canonical Markdown shared by the terminal, API and saved
session. It contains the original questions and proposed answers, stop reason,
source identity and coverage, re-attested deterministic results, observations,
raw action records, provenance and committed non-system history. These remain
available after the temporary workspace closes; artifact handles themselves
have session lifetime and are not durable download links.

`document_complete=true` means the retained mandatory records passed their
identity and re-attestation checks at composition. It does not mean every
question was verified, or that the sample's complete behaviour is known.
Missing, withdrawn or incompatible evidence is named and makes the document
incomplete. Historical prose-only reports have unknown completeness (`null`).

The existing single report generation is optional. Admission refusal, empty,
truncated, cancelled or failed narrative leaves the runtime document available.
Only a stopped complete narrative is published, explicitly unverified. Raw
incomplete output remains in `model_text` for diagnostics. `evidence_ids`
names the document's retained evidence; `narrative_evidence_ids` identifies the
bounded selection actually sent to the optional model call. They are distinct.

Saved sessions retain documents outside CHAT messages. Saving later CHAT turns
preserves this archive; unreadable existing archives are not overwritten.
The live recorder's RC=0 gates document completeness and investigation
lifecycle, not truth of free-form model conclusions.

FINISH uses its existing prompt admission and compaction policy first. If only
capacity blocks that request, it freezes the complete resulting view, including
rehydrated evidence, and measures it again with the production tokenizer. Its
output cap is the smaller of the qualified phase limit (2048), any lower explicit
limit, and the effective context minus safety (256) and exact input tokens.
Admission and generation use that same cap; a repair measures its own request.
PLAN, STEP, REPORT, context size and action limits are unchanged.

This cap is a limit, not a promise that every schema-valid answer can fit.
FINISH accepts proposals only from completed generation. A parseable control in
a length-limited, cancelled or failed response commits no model decision.
Cancellation and outages retain the existing stop handlers; insufficient space
causes a controlled stop with the runtime report. Parser and schema repairs
share the existing two-dispatch ceiling, including when both kinds of failure
occur in one FINISH phase.

A completed but unusable `empty_response` is not a truncated generation. It
may use the existing missing-control repair, but none of its tool arguments
can commit a proposal. Exhausting that repair preserves prior OPEN summaries
and citations, just as an interrupted FINISH does.

The FINISH tool description exposes the same cross-field completion rule as
the parser: `resolved` requires a non-empty `answer_summary` and no
`child_question`. A remaining dependency is not an operationally finished
answer. Rejection never converts the status or discards the child to accept
the call. The existing repair carries the parser's exact rejection when a
call remains; exhaustion blocks the question without committing the proposal.
