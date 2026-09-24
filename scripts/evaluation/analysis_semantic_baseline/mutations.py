"""Run canonical semantic-gate mutations in isolated copies, without inference."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

D = Path(__file__).resolve().parent
ROOT = D.parents[2]
GATE = D / "gate.py"
TEST = Path("tests/test_analysis_semantic_baseline.py")

# A caught mutant must fail the designated assertion. Import errors, missing
# samples and other infrastructure failures never count as a successful kill.
MUTANTS = [
    ("skip_source_scope", 'if name == "source" and (', 'if False and (', 'test_source_range'),
    ("accept_everything", 'verdict = "MANUAL_CHECK"', 'verdict = "PASS"', 'test_no_review_is_manual'),
    ("reject_everything", '            verdict = "PASS"', '            verdict = "FAIL"', 'test_complete_review_positive'),
    ("skip_sample_identity", 'if sha(raw) != sample["sha256"] or len(raw) != sample["bytes"]:',
     'if False:', 'test_qualified_sample_hash'),
    ("skip_evidence_identity", 'if sha(data) != evidence["sha256"]:', 'if False:', 'test_artifact_hash'),
    ("skip_body_identity", 'if sha(body) != evidence["body_sha256"] or len(body) != evidence["body_bytes"]:',
     'if False:', 'test_body_hash'),
    ("skip_delivery_scope", 'if evidence["byte_range"] != [0, len(body)] or evidence["source_sha256"] != sample["sha256"]:',
     'if False:', 'test_range'),
    ("skip_stage_set", 'if stages != oracle["qualified_stage_sha256"]:', 'if False:', 'test_all_stages'),
    ("skip_review_binding", '(not oracle_hash or review["oracle_sha256"] != oracle_hash\n'
     '                or review["report_sha256"] != sha(report)\n'
     '                or review["sample_sha256"] != oracle["sample"]["sha256"])',
     'False', 'test_report_binding'),
    ("ignore_negative_review", 'if any(r["semantic"] == "FAIL" for r in rows):',
     'if False:', 'test_negative_review'),
    ("accept_incomplete_review", '(all(r["semantic"] == "PASS" for r in required)\n'
     '              and not any(d["verdict"] == "MANUAL_CHECK" for d in decisions)\n'
     '              and review.get("whole_report_inspected") is True\n'
     '              and review.get("additional_claims_checked") is True)',
     'True', 'test_missing_review_is_not_pass'),
    ("skip_quote_bytes", '(type(a) is not int or type(b) is not int\n'
     '                        or a < 0 or b <= a or b > len(report)\n'
     '                        or report[a:b] != quote["text"].encode("utf-8"))',
     'False', 'test_quote_exact_byte_range'),
    ("ignore_unknown", 'required = [r for r in rows if r["category"] != "MAY"]',
     'required = [r for r in rows if r["category"] not in ("MAY", "UNKNOWN")]', 'test_unknown_cannot_be_skipped'),
    ("ignore_pending_may", 'and not any(d["verdict"] == "MANUAL_CHECK" for d in decisions)',
     'and True', 'test_explicit_pending_may_prevents_pass'),
]

# These annotations are human-supplied verdicts on fixed adversarial texts.
# A mutation discards one FAIL category; it must not turn that review into PASS.
for suffix, target in (
    ("no_dom", "test_no_dom_contradiction_rejected"),
    ("meta_purpose_as_fact", "test_seo_purpose_as_fact_rejected"),
    ("absence_from_excerpt", "test_false_whole_source_absence_rejected"),
    ("archive_is_delivery", "test_acquisition_is_not_delivery"),
):
    MUTANTS.append((
        "ignore_html_" + suffix,
        'by_id[item["id"]]["semantic"] = item["verdict"]',
        'by_id[item["id"]]["semantic"] = "PASS" if item["id"] == "html.' + suffix + '" else item["verdict"]',
        target,
    ))
MUTANTS.extend([
    ("skip_html_range_bounds", 'or not 0 <= a < b <= len(raw)', 'or False',
     "test_source_ranges_reject_out_of_bounds"),
    ("skip_html_range_digest", 'if sha(raw[a:b]) != span["sha256"]:', 'if False:',
     "test_source_ranges_reject_changed_bytes"),
    ("skip_html_utf8_mapping", 'if span["char_range"] != [len(raw[:a].decode("utf-8")),\n'
     '                                  len(raw[:b].decode("utf-8"))]:',
     'if False:', "test_source_ranges_reject_character_byte_confusion"),
    ("html_accept_all", 'verdict = "MANUAL_CHECK"', 'verdict = "PASS"',
     "test_all_unreviewed_texts_remain_manual"),
    ("html_reject_all", '            verdict = "PASS"', '            verdict = "FAIL"',
     "test_supported_limited_report_passes"),
])

RUNNER = """
import io,json,unittest
from tests.test_analysis_semantic_baseline import GateTests, GatePortabilityTests
from tests.test_analysis_semantic_baseline_html import HtmlGateTests
stream=io.StringIO()
suite=unittest.defaultTestLoader.loadTestsFromTestCase(GateTests)
suite.addTest(GatePortabilityTests('test_explicit_pending_may_prevents_pass'))
suite.addTest(GatePortabilityTests('test_omitted_may_remains_optional'))
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(HtmlGateTests))
result=unittest.TextTestRunner(stream=stream).run(suite)
print(json.dumps({'run':result.testsRun,'failures':[t._testMethodName for t,_ in result.failures],
                  'errors':[t._testMethodName for t,_ in result.errors],'log':stream.getvalue()}))
"""


def run_copy(source):
    with tempfile.TemporaryDirectory(prefix="orbit-semantic-mutant-") as temp:
        root = Path(temp)
        dest = root / D.relative_to(ROOT)
        shutil.copytree(D, dest, ignore=shutil.ignore_patterns("__pycache__"))
        (dest / "gate.py").write_text(source, encoding="utf-8")
        (root / "tests").mkdir()
        (root / "tests/__init__.py").touch()
        shutil.copyfile(ROOT / TEST, root / TEST)
        shutil.copyfile(ROOT / "tests/test_analysis_semantic_baseline_html.py",
                        root / "tests/test_analysis_semantic_baseline_html.py")
        env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
        child = subprocess.run([sys.executable, "-c", RUNNER], cwd=root, env=env,
                               text=True, capture_output=True, timeout=60, check=True)
        return json.loads(child.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = GATE.read_text(encoding="utf-8")
    baseline = run_copy(source)
    if baseline["run"] != 34 or baseline["failures"] or baseline["errors"]:
        raise RuntimeError("unmutated gate is not green: " + baseline["log"])
    results = []
    for name, old, new, target in MUTANTS:
        if source.count(old) != 1:
            raise RuntimeError("mutation anchor drift: " + name)
        result = run_copy(source.replace(old, new))
        caught = result["run"] == 34 and not result["errors"] and target in result["failures"]
        results.append({"mutant": name, "target": target, "caught": caught, **result})
    report = {"baseline": baseline, "caught": sum(r["caught"] for r in results),
              "total": len(results), "mutations": results}
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["caught"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
