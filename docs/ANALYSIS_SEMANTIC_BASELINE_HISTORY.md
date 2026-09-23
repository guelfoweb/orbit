# Retained semantic baseline and draft assessment

This is a selected reproduction index, not a ranking or new qualification.
Original records live under `workdir/diag/`; missing local records must be
reported, not reconstructed from summary PASS labels. The full frozen index
is `analysis_semantic_baseline_1/HISTORY.md` (SHA256
`be4b52980f953d6015f21c809e0fd9ca2771bae4eccbd7f3e464c56a71fde9e3`).
Its corresponding `runs.json` is SHA256
`ecb9f540b1e8fc388cec696e5449034a1007963654ec44689ccf9345c0bceff7`.

## Useful retained comparisons

| Records relative to workdir/diag | What the artifacts support | Comparability / limit |
|---|---|---|
| `ornith_recovery/native_main_fattura`, `native_candidate_fattura` | Main: wrong VBScript claim; candidate report exists but no complete clean semantic PASS established | COMPARABLE_CONFIG_ONLY: same canonical 7706-byte source, Ornith profile and mapped library hashes; different generated PLAN/history. Runtime main `10f9438`, measured candidate `9266ddd`, not later PR head `ba97fd4` |
| `ornith_recovery/native_main_iban`, `native_candidate_iban` | Main proposed answers wrong on folder/language. Only candidate final optional `model_text` passes the small oracle; preceding answers remain wrong | COMPARABLE_CONFIG_ONLY with same limits. No PASS of the whole candidate report or #374 |
| `dell_runtime_iban_closure/iban_run_B.json` | Useful JScript/synchronous GET/TEMP/write/run explanation, but IOC markup and counter discrepancy remain | NON_COMPARABLE: incomplete build/config identity; not a clean historical PASS |
| `ornith_direct_synthesis_16k/fattura/response.json` | Full source and stages supplied; wrong VBScript and irm/iex claims | NON_COMPARABLE to controller at 8k: context, input and mode differ; valid semantic counterexample, no causal latency comparison |
| `ornith_pre_action_finish_probe/mine_present/response.json` | Wrong attribution of the .bat to rundll32 | NON_COMPARABLE to full analysis: FINISH-only |
| `ornith_pre_action_finish_probe/iban_missing/response.json` | Correctly preserves missing wrapper/source boundary | Positive coverage-boundary case only, not a complete report |
| `analysis_evidence_delivery_report/native_final_ornith/run.json` | Complete 564-byte body delivered; incorrect asynchronous GET and VBScript interpretation | NON_COMPARABLE: delivery qualification, different history and constrain_finish enabled |
| `analysis_evidence_delivery_report/native_final_qwen/run.json` | 1008-byte delivery works; answer misinterprets decoded URL/alias | Qwen IQ1_M ctx4096: technical delivery PASS does not qualify semantics |
| `finish_model_resolution/native_iq1/insufficient/result.json` | Unsupported whole-source answer from partial evidence | FINISH-only, stack `8084e49`, not full corpus qualification |
| `finish_model_resolution/native_iq1/sufficient/result.json` | Acquisition recognized, spurious truncation explanation | Safe 40-byte fixture, NOT a corpus sample |
| `finish_model_resolution/measurements.json` | IQ3 timed out with no output | NOT_EVALUABLE; no semantic verdict or quantization-causality claim |

There is no demonstrated clean complete historical Fattura or mine report in
the inspected records. The old mine ~32-second/two-call/zero-action smoke and
similar empty PLAN runs do not prove successful investigation. Historical
NUC/repack-on/backend-b9551 measurements are not comparable to the recorded
Dell/repack-off/backend-41abbfd runs. A client banner is not a server SHA.

Legacy IBAN outer SHA
`f74ee18642a0c6466884d202be86ab26c26e7d9cb8354e27d028d732056eabbe`
shares the canonical decoded stage but is **not** the canonical source.
Fattura origin19948 is not Fattura7706. Never apply these oracles to those
different sources or transfer report reviews across them.

These observations establish known failures and limited positive components
for both models; they do not establish a model ranking or general Qwen/Ornith
ANALYSIS qualification. Newly evaluated reports must use exact source, report
and versioned oracle hashes, with actual delivery scope identified.

## #375: recommend closing without merge

Keep the diagnostic evidence, but do not promote the current PLAN guard.
At `9a3efe04347960f614e710a4b0cbb0d5ac95cc24`, `data_request=null` makes no
structured absence claim, so a prose claim that supplied data is missing can
bypass the guard. This is observed, not a hypothetical parser defect.
The guard correctly rejects covered structured ranges, but cannot validate
the question-to-range meaning. Its schema costs about 142 tokens in the
retained Ornith requests. Production-contract IBAN and Fattura exhaust bounded
repair without an adopted PLAN; reduced-contract Fattura still adopts redundant
requests through null. Cost and repair do not recover reliable PLAN behavior.

Sources: `plan_delivery_guard_1/RESULT.md` and
`ornith_plan_reduced_with_guard/RESULT.md`. No prose classifier, new instruction,
or alternative implementation is proposed here. **Recommendation only: this
promotion does not close or modify #375.**

## #374: split only independently justified fixes; no monolithic merge

Assessment of `ba97fd499f0a1add094a08eeeeaf1b29cf604e1e`, using
`ornith_finish_recovery/RESULT.md` and the subsequent behavioral-bisect decision:

| Change | Autonomous causal evidence | Recommendation and required boundary |
|---|---|---|
| REPORT token accounting | Optional admission omitted dossier cost. Retained request 7131 → 5278 tokens against 5888 allowance; dedicated tests and generated candidate report | Eligible for a separate future PR with its own frozen-content gates. Independent of FINISH Q1; reduced optional quotes do not imply semantic recovery |
| FINISH reference ownership | Incidental `evidence:` text in traceback/observation caused unrequested rehydration; Q4 8817 → 4342 tokens | Eligible for a separate future PR. Keep question ownership, selection and repair propagation together; explicit legitimate evidence requests must survive. Does not change frozen Q1 |
| Pre-sandbox duplicate suppression | Deterministic memoization defect reproduced, but native skips do not isolate benefit from the existing fast path or other changes | **Deferred.** Exact code/source/workspace/real-input equivalence and continued committed, reattestable evidence must be proved on a standalone diff. No semantic suppression |
| Raw archive access | Retained read fails on main and succeeds through candidate helper; not exercised in fresh native Fattura | **Deferred.** Depends on pre-action authorization, session/snapshot/producer/action linkage, reattestation/revocation, bounds and input fingerprint; changes STEP interface. No generic EvidenceStore/filesystem access |

The first two have enough independent causal evidence to justify future
extraction, **not automatic merge acceptance**. The latter two do not have
sufficient independent recovery/integration evidence for extraction now.
The earlier four-fix extraction proposal is superseded by this defer decision.
No fix is extracted, implemented or requalified by this documentation PR.
#374 remains draft and unchanged.
