# Canonical ANALYSIS semantic baseline

The versioned [corpus](../scripts/evaluation/analysis_semantic_baseline/corpus.json)
promotes `ANALYSIS-SEMANTIC-BASELINE-1`, frozen on
`10f9438e71438ee7e691a60041211348cc201810`. It changes no runtime behavior.
The original three oracles retain all **53 criteria** and all **12 transform hashes**.
The HTML extension adds **20 criteria** from the retained source/run audit at
`c6eba6de676eb3cddc9f0c661e638ffa2c3c81db`: four samples, 73 criteria, and no
new transform or model run. Other samples remain covered by the separate
six-sample deterministic gate.

## Acceptance rule for future changes

An ANALYSIS change **must not be accepted as a semantic improvement** without
a baseline/candidate comparison against this corpus and evidence of no
regression on the applicable criteria. Record applicability, delivery scope,
individual verdicts and any missing comparison explicitly. A pending
`MANUAL_CHECK`, missing report, or non-comparable run is not a semantic PASS.
Improving one criterion cannot hide a regression in another.

This supplements the mandatory deterministic cross-sample gate. Neither a
green unit suite nor successful extraction certifies a model's explanation.
It does not mandate new inference for documentation changes or replace the
repository's bounded live-qualification policy. Reuse retained, correctly
identified runs when they cover the changed behavior.

## Corpus and meaning of the criteria

Samples are external, inert qualification data under `workdir/samples/`.
Do not commit, execute, download, or silently replace them. A filename or a
matching decoded stage does not establish source identity.

| Oracle | Source | Bytes | SHA256 |
|---|---|---:|---|
| [Fattura](../scripts/evaluation/analysis_semantic_baseline/oracles/fattura.json) | Fattura981033956.js | 7706 | `b7cfd5fdeb16d7b5ecea1063419bdad6ad280ed9b73c636707874c3f4001dc0c` |
| [IBAN](../scripts/evaluation/analysis_semantic_baseline/oracles/iban.json) | IBAN.js | 7963 | `86e23fa673271308578daf61e783a00662351bab66d74f5f16e16302ad40d8b8` |
| [mine](../scripts/evaluation/analysis_semantic_baseline/oracles/mine.json) | mine.hta | 50114 | `6840b6d84f7c7190424fd465e466e2477e7c8a781457e2c6dcd523df498cea3d` |
| [HTML](../scripts/evaluation/analysis_semantic_baseline/oracles/html.json) | IT5440738233991.html | 30668 | `fdf79cf05d09968868ac2ac739a7f82693e9d47e9acb0be6effdcfab469c626e` |

- **MUST:** explain the specified supported fact.
- **MUST_NOT:** do not assert the specified error. Quoting or denying that error
  is not itself a violation.
- **MAY:** optional supported detail; omission is allowed, a false assertion is
  not. A failed MAY also fails a reviewed report.
- **UNKNOWN:** preserve the stated evidence boundary. Do not invent remote
  payload behavior or observed victim execution from static source alone.

Each criterion has a stable ID, claim, evidence references and
`evaluation: MANUAL_CHECK`. Correct appendix text or literal matches cannot
cancel contradictory model prose elsewhere in the document. Review the whole
document, including proposed answers, synthesis and assertions outside the
small checklist. Preserve uncertainty rather than requiring invented details.

## Reproduce without the historical diagnostics

From the repository root, with the pinned samples provisioned:

```sh
TMPDIR=/tmp PYTHONPATH=src python3 scripts/evaluation/analysis_semantic_baseline/gate.py --verify
TMPDIR=/tmp PYTHONPATH=src python3 -m unittest tests.test_analysis_semantic_baseline tests.test_analysis_semantic_baseline_html -q
TMPDIR=/tmp PYTHONPATH=src python3 scripts/evaluation/analysis_semantic_baseline/mutations.py --output /tmp/analysis-semantic-mutations.json
TMPDIR=/tmp PYTHONPATH=src python3 -m unittest tests.test_analysis_cross_sample_gate -q
```

`--verify` checks source hashes/sizes, evidence hashes, decoded UTF-8 body
hashes/sizes, complete byte ranges, source binding and the ordered stage set. HTML source locators additionally
check half-open byte bounds, exact slice hashes and Unicode-to-UTF-8 offsets;
they do not certify model delivery or semantic correctness.
It reuses the existing bounded `analysis_deobfuscate` producer on inert text;
it runs no sample, action, model, backend or network request. The expected
container is canonical JSON with ordered `body`, `kind`, `key` records. Its
hash was derived from the retained qualified outputs, not updated to follow
whatever the current decoder produces. A decoder mismatch is a regression
until explained; never blindly regenerate the oracle.

The small versioned `grounding.json` records the already-qualified platform
and invocation contracts and their original source/test identity. It is a
reference for review, not evidence of actual execution. The separate runtime
cross-sample gate tests those current production contracts.

No untracked `workdir/diag` file is required to run this gate. The original
twenty harness tests use benign temporary fixtures; a separate integration
test verifies all four real samples without any retained diagnostics.
On a sample-less checkout, `ORBIT_ALLOW_MISSING_CORPUS=1` explicitly waives
that integration test, as for the existing cross-sample gate. Such a skip is
not corpus qualification. The CLI never silently waives missing samples.

The mutation command applies each of the **13 original causal mutations** in
an isolated copy and requires its named assertion to fail. Missing fixtures,
import errors and other test errors do not count as caught mutants. Both
accept-all and reject-all must fail. One additional mutation covers the review
finding that an explicitly pending MAY decision must prevent overall PASS.
Nine HTML mutations cover source ranges/UTF-8 mapping, dropping explicit
negative review decisions for DOM/meta/absence/delivery errors, and HTML
accept-all/reject-all. All 23 mutants must be caught by named assertions.
The adversarial reports are authored fixtures with manual annotations, not
model results. Their meaning is not inferred by a keyword classifier; tests
prove that the gate preserves those review decisions.

## Review a report

```sh
PYTHONPATH=src python3 scripts/evaluation/analysis_semantic_baseline/gate.py --sample fattura --report /path/to/report.md
PYTHONPATH=src python3 scripts/evaluation/analysis_semantic_baseline/gate.py --sample fattura --report /path/to/report.md --review /path/to/review.json
```

Without a review the result is `MANUAL_CHECK`. `literal_witnesses` produce only
byte locators to help inspection; there is no keyword score, semantic matching
or LLM judge. The program checks integrity and review bookkeeping, not the
truth of arbitrary reasoning or the identity of the reviewer.

A review JSON contains:

```json
{
  "reviewer": "named reviewer",
  "method": "inspection against pinned source and deterministic outputs",
  "oracle_sha256": "hash of this versioned oracle file",
  "sample_sha256": "canonical source hash",
  "report_sha256": "hash of the exact report bytes",
  "whole_report_inspected": true,
  "additional_claims_checked": true,
  "criteria": [
    {
      "id": "fattura.language",
      "verdict": "MANUAL_CHECK",
      "rationale": "record the actual evidence and reasoning here",
      "whole_report_inspected": true
    }
  ]
}
```

This is an incomplete example, never a passing qualification. Add a decision
for every MUST/MUST_NOT/UNKNOWN criterion, and for MAY claims actually made.
Allowed verdicts are PASS, FAIL, MANUAL_CHECK. Each decision needs a rationale
and exact UTF-8 quotes (`quotes: [{"byte_range": [start, end], "text": "…"}]`)
or an explicit whole-document inspection. Ranges are half-open byte offsets.

CLI exit codes: **0** reviewed PASS (or integrity-only `--verify`), **1**
reviewed FAIL, **2** MANUAL_CHECK, **3** invalid/missing artifact. Any reviewed
FAIL wins; missing required decisions or incomplete inspection prevent PASS.
Omitted MAY decisions remain optional; explicitly submitted MANUAL_CHECK
decisions, including MAY, prevent PASS.
Review declarations are not signatures or an automated proof of semantics.

The original promotion changed only portable evidence bindings, not its 53 facts.
Each of those original oracles records its original oracle SHA; the corpus records the original
manifest and corpus hashes. Session-local evidence IDs are replaced by stable
source/stage identities. Byte ranges describe the archived/reconstructed
body, **not a receipt of delivery to a model**. Historical reviews bound to
old oracle hashes must not be silently rebound or treated as new PASS results.

## Historical comparisons and open draft decisions

The compact [history and draft assessment](ANALYSIS_SEMANTIC_BASELINE_HISTORY.md)
records reusable limits of the retained runs and recommendations for #374 and
#375. Historical reports and full diagnostics stay unchanged outside git.
No historical timing is a new benchmark or a causal model ranking.

## HTML bounded-evidence oracle

The [retained-view metadata](../scripts/evaluation/analysis_semantic_baseline/html_delivery.json)
is a compact, hash-bound extraction from `raw_archive_need_audit_1`. It records
raw/bounded evidence identities, question/action links, source ranges, original
artifact hashes and the limits of reconstruction. No malware source, full raw
body, sandbox code or session is committed. Only the pinned external sample
and tracked metadata are needed for portable integrity/review checks.

The source establishes inline JavaScript, a `navigator.userAgent` read,
Windows/Android/other branches and DOM `style.display = 'block'` assignments.
Function definition is distinct from invocation; the final top-level inline
script is evaluated during parsing, not merely defined. This is a static
code statement, not evidence of browser execution. Meta attributes contain
long word sequences including English words. Neither a non-English language
classification nor an SEO/decoy purpose is established by those strings.

The retained investigation acquired all **30668 bytes / 30644 characters**.
Its bounded observations did not deliver all those bytes. The reconstructed
last REPORT card omits **15902 characters**, corresponding to source bytes
**[12552, 28478)**, **15926 bytes**. Some of that region was in earlier Q2
observations. The exact admitted requests and per-call receipts are absent:
never interpret the card gap as the union of all bytes the model never saw.
The oracle records this uncertainty instead of fabricating delivery receipts.

`read_file(SOURCE_PATH, offset, limit)` already addresses these literal source
bytes; offsets/limits are bytes and UTF-8 boundaries matter. Smaller reads
remain subject to action/admission limits. Printing the entire missing region
would hit the observation bound again. This case does **not** justify raw
archive access or reopen any deferred TODO. The capability criterion applies
when reviewing this retained delivery failure, not as mandatory API advice in
every future sample report.

UNKNOWN describes the supplied evidence scope. It is not a permanent ban on
establishing additional facts after new, attested coverage. Conversely,
available archives, a correct appendix or a complete report document do not
prove that earlier model claims were supported. Review proposed answers and
narrative together; no clean semantic PASS is claimed for the retained HTML
run. Model/build/profile comparability and exact wire coverage are incomplete.
