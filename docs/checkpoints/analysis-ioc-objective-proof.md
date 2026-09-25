# Source-bound network destinations

The runtime records statically provable network destination operands before
PLAN. A local destination candidate that lacks a closed proof receives a
runtime-owned `IOC` question, ahead of model questions. This is independent of
whether PLAN notices it. Model answers cannot resolve that question or substitute
a different question for it. Existing per-question and global action, repair,
and model-call limits apply; cancellation, errors and exhausted limits preserve
an explicit BLOCKED outcome and the reason in the canonical report.

## Proof and scope

The bounded producer consumes a closed JavaScript syntax subset. It checks
bindings, initialization, supported effects, DOM origins and script boundaries
before attesting a destination operand. Exact literals, string concatenations,
pure zero-argument return helpers and a complete split/numeric-XOR/character loop
are supported. Every dependency is bound to original UTF-8 byte intervals and
hashes, including the whole effect scope and the sink. No JavaScript is executed.

An unknown construct interrupts the proof. There is no semantic matching,
sample-name exception, dynamic evaluation, or interpretation of model prose.
Recognized destination candidates stay OPEN for bounded local investigation;
known runtime input cannot be recovered from a static snapshot and is BLOCKED.
Unsupported local syntax can ultimately remain BLOCKED because the verifier
cannot establish a closed proof: an action's claimed result does not extend
the verifier. The recorded reason must distinguish that limit from proof that
the destination is intrinsically impossible to recover.

Discovery currently covers direct `.href`, `.host`, and `.hostname` assignments.
It is not a complete JavaScript or browser analyzer. Unrecognized/computed sink
syntax may not be discovered. Oversized sources have a separate incomplete-scan
limitation; an unscanned region does not establish absence of destinations and
does not invent a destination objective. Source acquisition is not delivery of
all source bytes to the model.

The proof establishes an exact **static operand linked to a URL-bearing sink**.
It does not establish that an event happened, a browser setter succeeded, a
network request occurred, or a remote payload behaved in any particular way.
An anchor host setter, for example, can be ineffective without an existing URL.

## Evidence and reporting

Proofs use existing deterministic transform records and authorized evidence
delivery. They carry snapshot, session store, phase, history ownership and
source-range identities. Publication re-attests the snapshot, reruns the closed
proof and verifies the evidence bytes and metadata. Revoked, changed or foreign
evidence cannot retain exact status. JSON persistence preserves proof metadata.
An ordinary decoder producing identical bytes does not replace sink provenance.

URL/host/IP records and objective limits precede unverified narrative in the
canonical Markdown report. The existing indicator presentation limit stays 32;
complete proof records remain in the dossier. Existing literal URI extraction
continues to mean literal presence, not a network-behavior attestation.

## Offline gates

`tests.test_analysis_ioc_proof` covers closed positive chains and adversarial
scope, evaluation, numeric spelling, mutation, script-order and byte-range cases.
`tests.test_analysis_ioc_objectives` exercises the real runtime: empty PLAN,
priority, unverified FINISH, bounded stopping, repair, cancellation, incomplete
generation, evidence withdrawal, session/snapshot binding and JSON persistence.
The canonical semantic baseline and cross-sample gate remain required. A native
HTML run qualifies integration only; it cannot qualify arbitrary model prose.

The earlier rejected recognizer's five false proofs were caused by two missing
scope bindings (destructured and method parameters), two unclosed dynamic
evaluation effects (aliases of `eval` and `Function`), and treating legacy octal
`021` as decimal 21. All five are rejected by the closed subset, rather than by
a blacklist of sample-specific spellings.
