"""Offline integrity checks and explicit manual review of the ANALYSIS corpus.

No inference, sample execution, network, or semantic keyword scoring. Decoding
uses Orbit's existing bounded, deterministic transforms on inert source text.
"""

import argparse
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encode_evidence(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def deterministic_evidence(raw):
    # No backend, EvidenceStore session, sandbox or sample interpreter is used.
    from orbit.runtime.analysis_deobfuscate import deobfuscate_with_status

    result = deobfuscate_with_status(raw.decode("utf-8"))
    if result.status != "complete":
        raise ValueError("deterministic extraction incomplete")
    return encode_evidence({"stages": [
        {"body": s.output, "kind": s.kind, "key": s.key} for s in result.stages
    ]})


def deref(value, pointer):
    for part in pointer:
        value = value[part]
    return value


def verify_oracle(oracle, root=ROOT):
    sample = oracle["sample"]
    raw = (root / sample["path"]).read_bytes()
    if sha(raw) != sample["sha256"] or len(raw) != sample["bytes"]:
        raise ValueError("sample identity mismatch")
    # Static locators / retained-view ranges, never model-delivery receipts.
    for name, span in oracle.get("source_ranges", {}).items():
        a, b = span["byte_range"]
        if (type(a) is not int or type(b) is not int
                or not 0 <= a < b <= len(raw)):
            raise ValueError("source range bounds: " + name)
        if sha(raw[a:b]) != span["sha256"]:
            raise ValueError("source range identity: " + name)
        if span["char_range"] != [len(raw[:a].decode("utf-8")),
                                  len(raw[:b].decode("utf-8"))]:
            raise ValueError("source range UTF-8 mapping: " + name)
    ids = [f["id"] for f in oracle["facts"]]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("duplicate/empty criteria")
    for fact in oracle["facts"]:
        if (fact["category"] not in ("MUST", "MUST_NOT", "MAY", "UNKNOWN")
                or fact["evaluation"] != "MANUAL_CHECK"):
            raise ValueError("criterion contract")
        if not fact["evidence"] or any(e not in oracle["evidence"] for e in fact["evidence"]):
            raise ValueError("missing proof reference")
    stages = []
    derived = None
    for name, evidence in oracle["evidence"].items():
        if "producer" in evidence:
            if evidence["producer"] != "analysis_deobfuscate" or "pointer" not in evidence:
                raise ValueError("unknown evidence producer")
            if derived is None:
                derived = deterministic_evidence(raw)
            data = derived
        else:
            data = (root / evidence["path"]).read_bytes()
        if sha(data) != evidence["sha256"]:
            raise ValueError("evidence artifact identity mismatch: " + name)
        if name == "source" and (
            evidence["path"] != sample["path"]
            or evidence["sha256"] != sample["sha256"]
            or evidence["bytes"] != len(raw)
            or evidence["byte_range"] != [0, len(raw)]
        ):
            raise ValueError("source scope mismatch")
        if "pointer" in evidence:
            body = deref(json.loads(data), evidence["pointer"]).encode("utf-8")
            if sha(body) != evidence["body_sha256"] or len(body) != evidence["body_bytes"]:
                raise ValueError("body mismatch: " + name)
            if evidence["byte_range"] != [0, len(body)] or evidence["source_sha256"] != sample["sha256"]:
                raise ValueError("scope mismatch: " + name)
            stages.append(sha(body))
    if stages != oracle["qualified_stage_sha256"]:
        raise ValueError("stage set/order mismatch")
    return {"source_bytes": len(raw), "stages": len(stages), "provenance": "PASS"}


def evaluate(oracle, report, review=None, root=ROOT, oracle_hash=None):
    integrity = verify_oracle(oracle, root)
    report.decode("utf-8")
    rows = []
    for fact in oracle["facts"]:
        locators = []
        for term in fact["literal_witnesses"]:
            needle = term.encode("utf-8")
            if not needle:
                raise ValueError("empty literal locator")
            start, spans = 0, []
            while (pos := report.find(needle, start)) >= 0:
                spans.append([pos, pos + len(needle)])
                start = pos + len(needle)
            locators.append({"literal": term, "byte_ranges": spans, "presence_only": bool(spans)})
        rows.append({"id": fact["id"], "category": fact["category"],
                     "semantic": "MANUAL_CHECK", "locators": locators})
    verdict = "MANUAL_CHECK"
    manual = False
    if review is not None:
        if (not oracle_hash or review["oracle_sha256"] != oracle_hash
                or review["report_sha256"] != sha(report)
                or review["sample_sha256"] != oracle["sample"]["sha256"]):
            raise ValueError("review binding mismatch")
        if not review.get("reviewer") or not review.get("method"):
            raise ValueError("unowned review")
        decisions = review["criteria"]
        if len({x["id"] for x in decisions}) != len(decisions):
            raise ValueError("duplicate review criterion")
        by_id = {r["id"]: r for r in rows}
        for item in decisions:
            if (item["id"] not in by_id or item["verdict"] not in ("PASS", "FAIL", "MANUAL_CHECK")
                    or not item.get("rationale")):
                raise ValueError("invalid review criterion")
            if not item.get("quotes") and item.get("whole_report_inspected") is not True:
                raise ValueError("review lacks excerpt or explicit whole-document review")
            for quote in item.get("quotes", []):
                a, b = quote["byte_range"]
                if (type(a) is not int or type(b) is not int
                        or a < 0 or b <= a or b > len(report)
                        or report[a:b] != quote["text"].encode("utf-8")):
                    raise ValueError("review quote mismatch")
            by_id[item["id"]]["semantic"] = item["verdict"]
            by_id[item["id"]]["review"] = item
        required = [r for r in rows if r["category"] != "MAY"]
        if any(r["semantic"] == "FAIL" for r in rows):
            verdict = "FAIL"
        elif (all(r["semantic"] == "PASS" for r in required)
              and not any(d["verdict"] == "MANUAL_CHECK" for d in decisions)
              and review.get("whole_report_inspected") is True
              and review.get("additional_claims_checked") is True):
            verdict = "PASS"
        manual = True
    return {"sample_id": oracle["sample_id"], "report_sha256": sha(report),
            "integrity": integrity, "semantic_gate": verdict,
            "manual_review_used": manual, "criteria": rows,
            "warning": "Presence is not truth. PASS requires explicit complete manual/engineering review; "
                       "reviewer identity and reasoning are not independently authenticated by this program."}


def load_oracle(sample_id):
    entry = next(s for s in read_json(D / "corpus.json")["samples"] if s["id"] == sample_id)
    data = (D / entry["oracle"]).read_bytes()
    if sha(data) != entry["oracle_sha256"]:
        raise ValueError("frozen oracle changed")
    oracle = json.loads(data)
    if (oracle["sample"] != {k: entry[k] for k in ("path", "sha256", "bytes")}
            or oracle["sample_id"] != entry["id"]
            or len(oracle["qualified_stage_sha256"]) != entry["stages"]):
        raise ValueError("corpus identity mismatch")
    return oracle, sha(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=[s["id"] for s in read_json(D / "corpus.json")["samples"]])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repository root containing externally provisioned workdir/samples")
    parser.add_argument("--verify", action="store_true", help="integrity only, never semantic certification")
    args = parser.parse_args(argv)
    if args.verify:
        result = [{"sample": s["id"], **verify_oracle(load_oracle(s["id"])[0], args.root)}
                  for s in read_json(D / "corpus.json")["samples"]]
        print(json.dumps(result, indent=2))
        return 0
    if not args.sample or not args.report:
        parser.error("--sample and --report required unless --verify")
    oracle, oracle_hash = load_oracle(args.sample)
    result = evaluate(oracle, args.report.read_bytes(), read_json(args.review) if args.review else None,
                      args.root, oracle_hash)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return {"PASS": 0, "FAIL": 1, "MANUAL_CHECK": 2}[result["semantic_gate"]]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        print(json.dumps({"semantic_gate": "INVALID_ARTIFACT", "error": str(error)}))
        raise SystemExit(3)
