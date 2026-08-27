#!/usr/bin/env python3
"""Reconstruct snapshot fidelity for an already-collected corpus. No inference.

The corpus preserves both the bounded snapshot the verifiers saw and the
EvidenceStore it was drawn from, so fidelity is recoverable after the fact by
comparing them. That is the whole point of having persisted both.

Read-only: the corpus is never written to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from orbit.runtime.completion_shadow import (  # noqa: E402
    MAX_SNAPSHOT_ARTIFACTS,
    MAX_SNAPSHOT_CHARS_PER_RECORD,
    MAX_SNAPSHOT_RECORDS,
    MAX_SNAPSHOT_REQUEST_CHARS,
)
from orbit.runtime.completion_shadow_ledger import read_ledger  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402


def reconstruct(checkpoint: dict, store: EvidenceStore) -> dict:
    """Fidelity of one persisted checkpoint, from the corpus alone."""
    reasons: list[str] = []
    truncated = []
    for entry in checkpoint.get("snapshot_evidence", []):
        evidence_id = entry.get("evidence_id", "")
        included = entry.get("text", "")
        authoritative = store.load_raw(evidence_id) or ""
        if len(included) < len(authoritative):
            truncated.append({
                "evidence_id": evidence_id,
                "authoritative_chars": len(authoritative),
                "included_chars": len(included),
            })
    if truncated:
        reasons.append("record_content_truncated")

    included_records = len(checkpoint.get("snapshot_evidence", []))
    if included_records >= MAX_SNAPSHOT_RECORDS:
        # At the cap the snapshot cannot show whether more existed; the run's
        # own record count is the only evidence, and it is not per-checkpoint.
        reasons.append("record_count_cap_reached")

    included_artifacts = len(checkpoint.get("snapshot_artifacts", []))
    if included_artifacts >= MAX_SNAPSHOT_ARTIFACTS:
        reasons.append("artifact_count_cap_reached")
    if len(checkpoint.get("request", "")) >= MAX_SNAPSHOT_REQUEST_CHARS:
        reasons.append("request_cap_reached")

    return {
        "action": checkpoint["action"],
        "lossless": not reasons,
        "reasons": reasons,
        "active_records_included": included_records,
        "truncated_record_count": len(truncated),
        "truncated_records": truncated[:8],
        "artifacts_included": included_artifacts,
    }


def main(argv: list[str]) -> int:
    corpus = Path(argv[1])
    out: dict = {"corpus": str(corpus), "runs": {}}
    for label in ("A", "B", "C"):
        d = corpus / "runs" / label
        if not (d / "completion-shadow.jsonl").exists():
            continue
        ledger = read_ledger(d / "completion-shadow.jsonl")
        store = EvidenceStore(root=d / "evidence")
        store.load_index()
        rows = [reconstruct(cp, store) for cp in ledger.checkpoints]
        out["runs"][label] = rows
        print(f"\n[{label}]  {len(store.records)} records in store")
        print(f"{'act':>4} {'fidelity':>9} {'incl':>5} {'trunc':>6} {'arts':>5}  reasons")
        for r in rows:
            print(f"{r['action']:>4} {('LOSSLESS' if r['lossless'] else 'LOSSY'):>9} "
                  f"{r['active_records_included']:>5} {r['truncated_record_count']:>6} "
                  f"{r['artifacts_included']:>5}  {','.join(r['reasons'])}")
    if len(argv) > 2:
        Path(argv[2]).write_text(json.dumps(out, indent=2))
    total = sum(len(v) for v in out["runs"].values())
    lossy = sum(1 for v in out["runs"].values() for r in v if not r["lossless"])
    print(f"\nTOTAL: {total} checkpoints, {lossy} LOSSY, {total-lossy} LOSSLESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
