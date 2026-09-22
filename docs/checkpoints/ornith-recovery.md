# Ornith recovery: bounded runtime fixes, semantic qualification incomplete

This candidate is based on main `10f9438e71438ee7e691a60041211348cc201810`.
The native/tested production diff is frozen at `9266ddd16e1045ff4b0d81d0ac3e8615359c0810`.
Keep the PR draft: the Fattura investigation did not meet its semantic or
end-to-end recovery criteria. Passing code tests does not establish model quality.

## Reproduced runtime defects and corrections

- Source reacquisition suppression advertised an archived raw output that
  `read_evidence` could not read. Reuse the existing bounded action input map
  for the one committed, re-attested complete-source acquisition. Check source,
  parent/raw lineage, session history and withdrawal; no EvidenceStore mount.
  Accept the exact `evidence:` spelling through that same authorization map.
- A real sandbox execution subsequently suppressed as a source copy returned
  before recording its fingerprint. Record it under exact code/source/workspace/
  authorized-input identity, so the identical next attempt stops before execution.
  Removing the committed evidence removes that reuse authorization.
- FINISH's user-role controller observation let a traceback reference become
  an implicit raw-evidence retrieval request. Preserve question-owned request
  metadata through FINISH and repair; keep the observation and rendered content.
- REPORT allocated optional excerpts without pricing the full required question
  dossier. Test excerpt upgrades with the existing exact admission on complete
  rendered messages. The existing record selection remains bounded; the retained
  replay keeps all its evidence IDs, deterministic results and original questions.
  Required content that cannot fit still refuses narrative generation.

Acquisition, action inputs, model delivery and conclusions remain distinct.
No sampler/backend/cache/profile, context/reserve, action or repair limit changes.
The helper description describes the newly available capability; no FINISH prompt
variant or narrative correction is introduced.

## Causal evidence

Retained manual session: four identical source-listing sandbox executions were
suppressed only after execution; its last generated `read_evidence("evidence:…")`
selected the raw reference advertised by suppression, which was unauthorized.
The reference then appeared in FINISH inside the error observation and caused
unrequested rehydration. Exact production-tokenizer replay changes that FINISH
from 8817 tokens (rejected) to 4342 (cap2048); the other retained Q1 remains4391.

Fresh main Fattura REPORT:7131/5888 tokens. Replaying its unchanged required
question dossier with the corrected excerpt allocator gives5278/5888. This
removes an optional head/tail quote, not question/provenance/stage state. It is
bounded delivery, not lossless preservation of model-visible source. The fresh
candidate native REPORT admitted5683/5888 and completed1360 output tokens.

## Historical and native limits

No inspected full historical Fattura report establishes rc39 or4b5e1ae as a clean
last-good state. The two-call IBAN/mine smokes used empty plans. Other historical
runs used different hardware, backend/repack or source variant and already contain
semantic errors. No historical rollback configuration is qualified here.

Dell, local Ornith-1.5-35B-A3B Q4_K_M, SHA256
`42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`;
integrated41abbfd backend;ctx8192;threads6/6;batch256/ubatch128;parallel1;
CPU;repack/MTP/thinking off;constrain_finish OFF;fresh native session each run.
One main and one candidate run per sample, sequential. Fattura ceiling2700s,
IBAN1200s; existing call/action/repair limits. No retries until a favorable answer.
Candidate captures pre-parser output with observation-only diagnostics; main did
not. Cache warmth, generated trajectories and thermal conditions are not paired.
Wall times are observations, not isolated performance effects.

The Fattura candidate still confuses the already decoded URI with an undecoded
inner dropper, confuses irm/iex and fails to establish the actual WMI linkage.
Three identical action attempts are skipped before sandbox execution, but the
whole investigation uses more calls and time. All five exact stages and the C2
remain in the canonical document; none of the narrative is independently certified.
No complete-source acquisition occurred in that fresh candidate Fattura trajectory,
so it does not by itself qualify a native selection of the newly readable archive.

No complete Ornith recovery or general ANALYSIS qualification is claimed. The
first remaining semantic failure is interpretation of already delivered exact
stage bodies, distinct from the repaired access/admission defects. A source
excerpt omitted by admission also cannot establish a whole-source claim.

## Artifacts and validation

Machine-local retained reports, requests/token IDs, raw responses, model controls,
authorizations, sandbox results, evidence, timing and memory traces are under
`workdir/diag/ornith_recovery/`. Before/after reports are mandatory gate artifacts,
not replaced by test counters. Frozen malware data is not committed.

Production main stays unchanged; the FINISH projection bundle stays unchanged.
During initial isolated-worktree provisioning, build tests followed mission-owned
symlinks and rebuilt two production bridge outputs. The incident was preserved;
all production binary/sidecar hashes were restored byte-for-byte, the candidate
now has physically isolated build outputs, and no native result was qualified
with those accidental binaries. The unrelated user server was never signalled.

The initial failed full-suite log is retained. Focused final ANALYSIS suite:
1962 tests,1skip,RC0, including cross-sample. Applied causal mutations:19/19
caught. Independent code review:BLOCKER0/MAJOR0; native semantic acceptance is
separate and remains failed. The final full-suite log and real child return code are retained in the mission diagnostics.


## Final native comparison (observational)

| Sample / checkout | Total calls | Sandbox executions | Pre-execution duplicate skips | Evaluated / cached tokens | Generated tokens | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fattura main | 9 | 4 | 0 | 27015 / 6279 | 3410 | 840.9 |
| Fattura candidate | 17 | 5 | 3 | 40940 / 32392 | 5518 | 1304.7 |
| IBAN main | 11 | 4 | 0 | 20594 / 7660 | 3536 | 662.6 |
| IBAN candidate | 6 | 2 | 0 | 17724 / 1074 | 2082 | 532.6 |

All totals include any dispatched REPORT: none for main Fattura (admission
refusal), one for the other three. Fattura candidate has two actual bounded
FINISH protocol repairs (calls3→4 and10→11); the existing run counter reports0,
so it must not be used to infer no retries. These are not new action budgets.

IBAN candidate final narrative correctly describes synchronous GET, TEMP output
and launch, while earlier FINISH claims System32 and calls the file-writing
payload fileless. The final narrative corrects those errors and respects unknown
remote-payload behavior. Its GetSpecialFolder(2) mapping IS supplied by the existing
runtime platform-constant renderer; it is not a new COM query or an invented
attestation. The final IBAN synthesis is substantially supported, while the earlier
FINISH answers remain mixed. Different trajectories do not prove a general semantic
non-regression or end-to-end recovery, and Fattura still fails the frozen gate.

Peak server RSS across the four runs is22292632KiB, process swap0. Host swap
was already present and decreased (about2.64→2.56GiB); no OOM or uncontrolled
swap growth was observed in these runs. Startup14.2s route prewarm on the
candidate is separate from analysis wall time; ANALYSIS prewarm was not requested.

## Retained native action, offline execution replay

`manual_action_main.json` and `manual_action_candidate.json` feed the exact
recorded generated action through real STEP and sandbox execution, replacing
only inference. Original committed history and source/evidence identities are
restored; no obligation or manual authorization is added. The program only reads
an archive and prints it: no sample code executes and the sandbox denies network.
Main fails with unavailable evidence. Candidate prints all7780 archived UTF-8
bytes unchanged plus print's newline. Committed history prefix is unchanged.
This qualifies consumption of that retained native action; it does not prove a
fresh model will select the archive, or that the resulting conclusion is correct.

## Decision

Retain the focused runtime candidate for review as a draft. Do not merge it as
an Ornith recovery: the frozen Fattura semantic gate failed and measured total
latency did not recover. The minimum next decision is whether to accept these
independently demonstrated runtime corrections on their technical contracts,
not another speculative prompt/model/backend campaign. Broader investigation
quality, the original Q1 echo/truncation and legacy summary clipping remain
unresolved. No automatic release, tuning, Qwen work or merge follows this gate.
