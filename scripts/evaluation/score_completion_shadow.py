#!/usr/bin/env python3
"""Deterministic offline scorer for the completion-shadow corpus.

Evaluation only. Reads a preserved corpus read-only, scores each persisted
checkpoint against an evaluator-only oracle, and reports what the two verifiers
got right and wrong. It performs no inference: there is no backend, no client,
and no network here, and a test asserts as much.

Two views of "what was established" are scored separately, because they answer
different questions:

  snapshot   -- exactly what the verifiers were shown at that checkpoint. This
                is what A and B can fairly be judged against.
  cumulative -- everything the analysis had established by then, from the
                EvidenceStore. This is what "was the analysis actually
                complete" means, and it is not what the verifiers saw.

The two views diverge for two independent reasons, and they call for different
fixes, so they are worth keeping apart:

  * per-record truncation -- each record enters the snapshot as a 600-character
    excerpt (MAX_SNAPSHOT_CHARS_PER_RECORD). This is the dominant effect and it
    bites from the very first checkpoint: in this corpus, at action 4 of the
    PowerShell run, six of eight records carry the hidden-launch evidence in
    their full text and none carry it inside the excerpt.
  * window eviction -- at most MAX_SNAPSHOT_RECORDS (12) records are carried,
    so older evidence drops out later in a run. Real, but it cannot explain a
    checkpoint that holds only eight records.

An oracle finding that cannot be decided from persisted text is UNSCORABLE, and
a sample with an unscorable required finding is never declared complete: it is
INDETERMINATE, which is the conservative reading.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCHEDULE = (4, 6, 8, 10, 12)


@dataclass(frozen=True)
class FindingResult:
    id: str
    satisfied: bool
    unscorable: bool
    detail: str = ""


# What a predicate is allowed to inspect.
#
#   SOURCE_FACT           a positive/negative fact about the immutable artifact
#   DERIVED_FACT          established only after deterministic analysis
#   ARTIFACT_FACT         a fact about an artifact analysis produced
#   NEGATIVE_SOURCE_FACT  an absence property of the source
#
# The distinction is not decorative. A NEGATIVE_SOURCE_FACT decided by searching
# cumulative narrative is unsound: an analysis step that *explains* the artifact
# makes no network call has to name the calls it does not make, and the moment
# it does, the absence predicate fails against its own explanation. Under
# append-only state that makes COMPLETE -> INCOMPLETE reachable from commentary
# alone, which is exactly what happened to `no_dangerous_capability`.
#
# So absence is evaluated against the source bytes, never the narrative.
PREDICATE_SCOPES = {
    "literal_all": "DERIVED_FACT",
    "literal_any": "DERIVED_FACT",
    "regex": "DERIVED_FACT",
    "absent_all": "NEGATIVE_SOURCE_FACT",
    "unscorable": "UNSCORABLE",
}


def predicate_scope(spec: dict) -> str:
    """Scope of one predicate, per-spec first and `kind` as the fallback.

    Keying on `kind` alone would force every absence predicate onto source
    bytes, including a legitimate absence-of-DERIVED-fact ("analysis never
    reported an unresolved gap"), where the token by construction never appears
    and the finding would silently self-satisfy. No such predicate exists in the
    oracle today, so this is a latent hazard rather than a live bug -- but the
    override is one line and the failure mode is invisible.
    """
    declared = spec.get("scope")
    if declared:
        return declared if declared in set(PREDICATE_SCOPES.values()) else "UNKNOWN"
    return PREDICATE_SCOPES.get(spec["kind"], "UNKNOWN")


def evaluate_finding(
    spec: dict, text: str, *, source_text: str | None = None
) -> FindingResult:
    """One deterministic predicate. No interpretation, no similarity.

    `text` is the cumulative evidence narrative. `source_text` is the immutable
    artifact. A NEGATIVE_SOURCE_FACT is decided against `source_text` and
    refuses to be scored without it, rather than silently falling back to the
    narrative and reintroducing the anti-monotonicity this exists to prevent.
    """
    kind = spec["kind"]
    fid = spec["id"]
    if kind == "unscorable":
        return FindingResult(fid, False, True, spec.get("comment", ""))

    if predicate_scope(spec) == "NEGATIVE_SOURCE_FACT":
        if source_text is None:
            return FindingResult(
                fid, False, True,
                "negative source predicate requires the artifact source",
            )
        text = source_text

    # A forbidden value can never satisfy a finding, even if it looks close.
    # Note for whoever first writes an `absent_all` carrying a `forbidden` list:
    # this reads whichever text won the scope routing above, so on a negative
    # source fact it inspects source bytes rather than the narrative. No
    # predicate combines the two today.
    # This is what keeps a superseded value from passing as authoritative.
    forbidden = [v for v in spec.get("forbidden", []) if v.lower() in text.lower()]

    if kind == "literal_all":
        hit = all(v.lower() in text.lower() for v in spec["values"])
    elif kind == "literal_any":
        hit = any(v.lower() in text.lower() for v in spec["values"])
    elif kind == "absent_all":
        hit = not any(v.lower() in text.lower() for v in spec["values"])
    elif kind == "regex":
        hit = re.search(spec["pattern"], text) is not None
    else:
        return FindingResult(fid, False, True, f"unknown predicate kind {kind!r}")

    detail = f"forbidden present: {forbidden}" if forbidden else ""
    return FindingResult(fid, bool(hit), False, detail)


@dataclass
class CheckpointScore:
    action: int
    required_total: int
    satisfied: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    unscorable: list = field(default_factory=list)

    @property
    def oracle_complete(self) -> bool:
        # Never complete while anything required is unscorable: an undecided
        # requirement is treated as unmet, not waved through.
        return not self.missing and not self.unscorable

    @property
    def state(self) -> str:
        if self.unscorable:
            return "INDETERMINATE"
        return "COMPLETE" if not self.missing else "INCOMPLETE"


def score_text(
    oracle: dict, text: str, action: int, *, source_text: str | None = None
) -> CheckpointScore:
    """Score one state. `source_text` is required for negative source facts.

    Without it a NEGATIVE_SOURCE_FACT scores UNSCORABLE rather than being
    decided against the narrative -- refusing to answer beats answering from
    the wrong evidence.
    """
    score = CheckpointScore(action=action, required_total=len(oracle["required_findings"]))
    for spec in oracle["required_findings"]:
        result = evaluate_finding(spec, text, source_text=source_text)
        if result.unscorable:
            score.unscorable.append(result.id)
        elif result.satisfied:
            score.satisfied.append(result.id)
        else:
            score.missing.append(result.id)
    return score


def classify_a(
    verdict: str | None,
    oracle_complete: bool,
    indeterminate: bool,
    *,
    skipped: bool = False,
) -> str:
    # A checkpoint whose verifier never ran has no verdict to grade. Reading a
    # missing answer as CONTINUE would invent an observation -- and would score
    # the budget's own skip as if the verifier had been wrong.
    #
    # But a missing verdict has two causes, and they are not the same result.
    # The budget declining to spend a call is the mechanism working; a verifier
    # that raised leaves `verification_skipped` false, and reporting that as a
    # skip would hide a broken backend inside a healthy-looking matrix.
    if verdict is None:
        return "A_NOT_CALLED" if skipped else "A_ERRORED"
    if indeterminate:
        return "A_INDETERMINATE"
    if verdict == "COMPLETE":
        return "A_TRUE_COMPLETE" if oracle_complete else "A_FALSE_COMPLETE"
    return "A_TRUE_CONTINUE" if not oracle_complete else "A_FALSE_CONTINUE"


# Reasons B is deliberately never asked. Each is the gate working: A did not
# propose a completion, or the completion was disqualified before the expensive
# half was worth spending. None of them is a verifier failure.
B_NOT_ASKED_REASONS = frozenset({
    "snapshot_too_large",
    "verifier_a_unparsed",
    "verifier_a_continue",
    "verifier_a_cited_no_evidence",
    "referenced_evidence_not_active",
    "referenced_evidence_reattest_failed",
})


def classify_b(
    verdict: str | None,
    oracle_complete: bool,
    indeterminate: bool,
    *,
    blocked_by: str | None = None,
) -> str:
    # B's absence is read from why the checkpoint stopped, not from A's skip
    # flag. `verification_skipped` is set only by the budget, so sharing it
    # would report the ordinary "A said CONTINUE, so B was never asked" path
    # -- the cost control working exactly as designed -- as a dead backend.
    if verdict is None:
        return "B_NOT_CALLED" if blocked_by in B_NOT_ASKED_REASONS else "B_ERRORED"
    if indeterminate:
        return "B_INDETERMINATE"
    if verdict == "GAP":
        return "B_TRUE_GAP" if not oracle_complete else "B_FALSE_GAP"
    return "B_TRUE_NO_GAP" if oracle_complete else "B_FALSE_NO_GAP"


def classify_gap(detail: str, score: CheckpointScore) -> str:
    """Is B's stated omission a real material one, or already satisfied?

    Decided against the oracle's own findings, never against B's prose meaning:
    the prose records what B claimed, and treating it as ground truth would let
    the thing under test grade itself.

    The consequence is worth stating plainly, because the label is easy to
    over-read. `NON_MATERIAL_OR_ALREADY_SATISFIED` means only "the oracle's
    required findings were all present when B objected". It does not mean B was
    incoherent. B may have raised a real process objection -- an artifact whose
    contents were never displayed, a path that does not obviously correspond to
    the requested one -- that this oracle simply does not encode, because the
    oracle scores material findings about the artifact rather than the shape of
    the report. Whether such objections *should* be material is a policy
    question this scorer deliberately does not answer.
    """
    if not detail:
        return "UNSCORABLE"
    if score.unscorable:
        return "UNSCORABLE"
    if score.missing:
        return "MATERIAL_AND_REAL"
    return "NON_MATERIAL_OR_ALREADY_SATISFIED"


def load_run(corpus: Path, label: str) -> dict:
    """Read one preserved run. Corpus is opened read-only and never written."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from orbit.runtime.completion_shadow_ledger import read_ledger, verify_snapshot_hashes
    from orbit.runtime.evidence import EvidenceStore

    d = corpus / "runs" / label
    run = json.loads((d / "run.json").read_text())
    ledger = read_ledger(d / "completion-shadow.jsonl")
    if ledger.malformed_lines or ledger.unsupported_versions:
        raise SystemExit(f"[{label}] corrupt ledger; refusing to score")
    mismatched = verify_snapshot_hashes(ledger)
    if mismatched:
        raise SystemExit(f"[{label}] snapshot hash mismatch at actions {mismatched}; refusing to score")

    store = EvidenceStore(root=d / "evidence")
    store.load_index()
    cumulative = "\n".join(
        (store.load_raw(r.evidence_id) or "") for r in store.records.values()
    )
    # The immutable artifact, for NEGATIVE_SOURCE_FACT predicates. Absence is a
    # property of the source, and scoring it against the narrative is the defect
    # the predicate scopes exist to prevent. A corpus that preserved no artifact
    # yields None, and such findings then score UNSCORABLE rather than guessed.
    #
    # Bound to its recorded hash, exactly as the ledger is. The artifact is now
    # load-bearing for absence, so accepting unverified bytes here would let a
    # swapped or rewritten file score confidently against the wrong source --
    # and a `final_report.txt` picked up as "the artifact" would reintroduce
    # narrative scoring through a different door, which is the whole defect.
    artifacts = sorted(d.glob("artifact.*"))
    if len(artifacts) > 1:
        raise SystemExit(
            f"[{label}] {len(artifacts)} artifact.* files; cannot choose a source"
        )
    source_text = None
    if artifacts:
        # A directory named artifact.* is a refusal like any other, not an
        # uncaught IsADirectoryError from three frames down.
        if not artifacts[0].is_file():
            raise SystemExit(f"[{label}] {artifacts[0].name} is not a regular file")
        raw = artifacts[0].read_bytes()
        expected = str(run.get("artifact_sha256_before") or "")
        # An absent or empty recorded hash is a refusal, not a waiver. Skipping
        # verification when the corpus records nothing would make the guard's
        # strength depend on the data it is guarding: a hash-less run would
        # accept arbitrary bytes as "the immutable artifact", and absence
        # scored against a narrative-shaped file is the original defect.
        if not expected:
            raise SystemExit(
                f"[{label}] artifact present but run.json records no "
                "artifact_sha256_before; cannot vouch for the source bytes"
            )
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise SystemExit(
                f"[{label}] artifact hash mismatch: {actual} != {expected}; "
                "refusing to score absence against unverified bytes"
            )
        # Absence over nothing is vacuously true, so an empty artifact would
        # satisfy every absent_all. Authentic bytes or not, that is not a
        # judgement worth making silently.
        if not raw.strip():
            raise SystemExit(
                f"[{label}] artifact is empty; absence predicates would be "
                "vacuously satisfied"
            )
        source_text = raw.decode(errors="replace")
    return {
        "label": label, "run": run, "ledger": ledger,
        "cumulative": cumulative, "source_text": source_text,
    }


def snapshot_text(checkpoint: dict) -> str:
    return "\n".join(e.get("text", "") for e in checkpoint.get("snapshot_evidence", []))


def score_corpus(corpus: Path, oracles: dict) -> dict:
    # The configured schedule, kept for reference. Per-run observed schedules
    # are recorded alongside each run: a run that ended early was never
    # checkpointed at the later actions, and reporting the constant as though
    # it had been reads as "checkpointed and skipped".
    report = {"runs": {}, "schedule": list(SCHEDULE)}
    # Discovered rather than hardcoded: a corpus need not contain every
    # workload class, and a missing one is an absent run, not a failure.
    #
    # A run directory without run.json is a third thing, though, and silence
    # would be wrong for it: a run killed after checkpointing but before its
    # final write leaves exactly that shape. It cannot be scored -- there is no
    # final state to score against -- but it is reported rather than dropped,
    # so a lost workload class cannot pass for a corpus that never had one.
    present = sorted(d.name for d in (corpus / "runs").iterdir() if d.is_dir())
    labels = [name for name in present if (corpus / "runs" / name / "run.json").exists()]
    report["incomplete_runs"] = [name for name in present if name not in labels]
    for label in labels:
        data = load_run(corpus, label)
        oracle = oracles["samples"][label]
        rows = []
        for cp in data["ledger"].checkpoints:
            snap = score_text(oracle, snapshot_text(cp), cp["action"],
                              source_text=data["source_text"])
            cum = score_text(oracle, data["cumulative"], cp["action"],
                             source_text=data["source_text"])
            skipped = bool(cp.get("verification_skipped"))
            a_class = classify_a(
                cp["verifier_a"], snap.oracle_complete, bool(snap.unscorable),
                skipped=skipped,
            )
            b_class = classify_b(
                cp["verifier_b"], snap.oracle_complete, bool(snap.unscorable),
                blocked_by=cp.get("blocked_by"),
            )
            rows.append({
                "action": cp["action"],
                "required_total": snap.required_total,
                "snapshot_satisfied": snap.satisfied,
                "snapshot_missing": snap.missing,
                "unscorable": snap.unscorable,
                "snapshot_state": snap.state,
                "cumulative_state": cum.state,
                "cumulative_missing": cum.missing,
                "verifier_a": cp["verifier_a"],
                "verifier_b": cp["verifier_b"],
                "would_stop": cp["would_stop"],
                "a_class": a_class,
                "b_class": b_class,
                "b_gap_materiality": classify_gap(cp.get("verifier_b_detail", ""), snap),
                "b_detail": cp.get("verifier_b_detail", "")[:160],
                "verifier_calls": cp["verifier_calls"],
                "verifier_wall_seconds": cp["verifier_wall_seconds"],
                "verifier_prompt_tokens": cp["verifier_prompt_tokens"],
                "verifier_output_tokens": cp["verifier_output_tokens"],
            })
        report["runs"][label] = {
            "artifact": data["run"]["artifact"],
            "artifact_sha256": data["run"]["artifact_sha256_before"],
            "stop_reason": data["run"]["stop_reason"],
            "actions_executed": data["run"]["actions_executed"],
            "model_calls": data["run"]["model_calls"],
            "wall_seconds": data["run"]["wall_seconds"],
            "shadow": data["run"]["shadow"],
            # Observed, not configured: a run that ended at action 9 was never
            # checkpointed at 10 or 12, and must not read as having skipped them.
            "observed_schedule": [row["action"] for row in rows],
            "checkpoints": rows,
        }
    return report


def main(argv: list[str]) -> int:
    corpus = Path(argv[1])
    oracles = json.loads(Path(argv[2]).read_text())
    report = score_corpus(corpus, oracles)
    out = Path(argv[3]) if len(argv) > 3 else None
    if out:
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for label, r in report["runs"].items():
        print(f"\n=== RUN {label}  {r['artifact']}  actions={r['actions_executed']} "
              f"stop={r['stop_reason']}")
        print(f"{'act':>4} {'snapshot':>13} {'cumulative':>12} {'A':>9} {'B':>5} "
              f"{'WOULD':>6} {'a_class':>18} {'b_class':>14} {'gap':>34}")
        for c in r["checkpoints"]:
            print(f"{c['action']:>4} {c['snapshot_state']:>13} {c['cumulative_state']:>12} "
                  f"{str(c['verifier_a']):>9} {str(c['verifier_b']):>5} {str(c['would_stop']):>6} "
                  f"{c['a_class']:>18} {c['b_class']:>14} {c['b_gap_materiality']:>34}")
            if c["snapshot_missing"]:
                print(f"       snapshot missing: {c['snapshot_missing']}")
            if c["cumulative_missing"]:
                print(f"       cumulative missing: {c['cumulative_missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
