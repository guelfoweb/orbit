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

## Explicit FINISH constrained decoding

FINISH constrained decoding defaults to **off**. With an Orbit server running,
start the interactive client with:

```sh
orbit --constrain-finish
```

Then use `/analysis <path>` as usual. `/status` shows `FINISH constraint on`
or `off`: the active analysis runtime's setting, or the client setting before
an analysis is opened. This reports the requested policy, not a successful
capability check or verified conclusions. The server checks support when the
request is submitted.

The existing client JSON configuration (`--config <path>`) also accepts
`"constrain_finish": true`. Use `orbit --no-constrain-finish` to override it for
one client. The choice applies to new analysis sessions in that client,
including analysis entered through routing, and survives `/reset` within that
client. It is not saved as a conversation setting, does not modify server state,
and is not enabled automatically for any model. Other clients keep their own
configuration. PLAN, STEP and CHAT retain their defaults.

Python embeddings can still set `AnalysisRuntime.constrain_finish` explicitly
on an initialized runtime:

```python
runtime.constrain_finish = True
```

The equivalent constructor option is
`AnalysisRuntime(..., constrain_finish=True)`, with the usual backend, source,
evidence store and workspace. Both factories default to `False`; the terminal
passes only the client's explicit choice. No environment variable enables it.

For an opted-in runtime, requests whose sole tool is
`finish_analysis_question` use `tool_choice="required"` in exact token counting,
admission and generation, including the existing FINISH repair. The integrated
native server and rebuilt Orbit chat bridge must support the contract; an old
server, unavailable grammar or incompatible configuration produces an error,
never a silent fallback to unconstrained generation. The supported path requires
a native chat bridge profile, text tools, thinking off and no custom stop
override. Use the bundled native build process (`python3 scripts/build_native.py`)
after a source upgrade; no external llama.cpp installation is needed.

The bridge derives the grammar from the offered schema and actual chat format,
including the control name. It uses the native XML format for the qualified
Qwen and Ornith profiles. A sampler belongs to one request and is released on
success, error or cancellation; the next request does not inherit its state.
Requests using the default `auto` choice retain their normal behavior.

Constrained decoding does not select a completion outcome: `resolved`,
`still_open` and `blocked` remain representable. The XML grammar does not enforce
every schema constraint, including the status enum and duplicate optional
parameters. Existing parsing, schema and semantic validation remain mandatory;
contradictory or incomplete output still cannot commit a decision. Budget,
action and repair limits are unchanged. Grammar acceptance does not establish
the truth of the proposed answer or its citations.

The retained native qualification used one FINISH call per model, without
repair, on identical prompt token IDs and unchanged output caps. Qwen3.8 Flash
Next UD-IQ1_M produced an accepted `still_open` control but invented a numeric
decoding detail in its summary. Its ANALYSIS semantic qualification remains
limited as described in [model qualification](ANALYSIS_QUALIFICATION.md).
Ornith Q4_K_M produced the exact previously retained, evidence-grounded
`still_open` response; this covers that case, not every investigation.

On the Dell, one offline replay of captured token sequences measured additional
grammar-plus-greedy **apply** time over greedy of about 21.2 ms/token for Qwen and
22.2 ms/token for Ornith. This used controlled logits and the full vocabulary,
without model inference; it excludes sampler accept time. It is neither an
end-to-end latency delta nor a portable performance guarantee. Request-local
sampler creation during the native calls took about 1.01 ms and 0.74 ms,
respectively. Source, raw requests/responses, identities and gate results are
retained locally in `workdir/diag/finish_constrained_decoding/`.
