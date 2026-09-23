# FINISH reference ownership

FINISH receives an already bounded observation inside a user-role message.
That observation can contain `evidence:<id>` in a traceback, stdout, stderr or
quoted model output. These are data, not instructions to retrieve another
archive. The same applies to FINISH repair errors.

The runtime selects explicit references in the original question and its
`missing_fact` only when an existing structural association also exists:

- the current action result handed from the autonomous loop to its active
  question;
- the question's validated `caused_by` reference;
- evidence IDs already associated with that question's controller state.

An association permits a read; it does not prove the model's conclusion. The
controller validates the active question before building the transient request.
There is no inference of ownership from prose, and no reconstruction of an
older action's question from text. An otherwise unowned mention is ignored.

Selection is transient request metadata, bound to the runtime, EvidenceStore,
snapshot and analyst turn. It is not persisted in canonical history or rendered
into the model prompt. Both existing repair paths retain the original selection;
their additional text cannot authorize more reads. A missing selection permits
no FINISH retrieval; ambiguous or stale metadata fails closed.

Before each retrieval, including repair, the runtime checks the snapshot bytes
and reuses the current store's exact reattestation and canonical tool-reference
checks (tool name, call ID, user turn and persisted reference). The current
action must match the latest committed tool result and current analyst turn.
Registered transforms and Office modules retain their existing producer
identities. An explicitly associated raw sibling additionally requires its
committed bounded record and matching action, turn, code and source provenance.
There is no automatic raw-sibling discovery or new sandbox access.

Unavailable, altered, revoked or incompatible selected records refuse admission.
The existing exact tokenizer budget and bounded Office-source delivery remain
authoritative. Observation text, controller state, stored evidence, completion
semantics, prompts and repair allowances are unchanged. PLAN/STEP keep their
existing explicit-request path.

## Retained offline replay

On the canonical 7706-byte Fattura snapshot
(`b7cfd5fdeb16d7b5ecea1063419bdad6ad280ed9b73c636707874c3f4001dc0c`),
the current production Ornith tokenizer reproduces the retained Q4 failure:

| View | Input tokens |
|---|---:|
| Main, compacted request including accidental raw retrieval | 8817 |
| Same compacted request without that raw block | 2344 |
| Corrected admission, retaining history and five transform bodies | 4342 |

The raw block costs 6473 tokens. Removing it lets unchanged admission retain
1998 more tokens of legitimate material. The resulting request fits the 5888
input allowance with the unchanged 2048 output cap and 256 safety reserve.
Q1 remains byte/token-identical at 4391 tokens. This is admission evidence,
not a native latency or model semantic-quality claim.

Machine-local replay artifacts are preserved under
`workdir/diag/analysis_finish_reference_ownership_1/`.
The focused regression suite is `tests.test_analysis_finish_reference_ownership`.
The canonical semantic baseline remains applicable; this change does not
certify model narratives or resolve the broader ANALYSIS semantic limitations.
