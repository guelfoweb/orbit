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
