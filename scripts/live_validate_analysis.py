#!/usr/bin/env python3
"""Run one autonomous analysis against a live server and record what happened.

This exists to answer a narrow question honestly: on a real model, does the
structured controller actually engage, does every action belong to a question,
and does the run reach a finite report? It is a recorder, not a judge -- it
asserts nothing about the artifact and never decides that an analysis was
"good". The analyst reads the JSON.

One run per invocation, one fresh session per run. The server is not managed
here: it is started and verified outside, so the model, its SHA and its
profile are the operator's statement rather than this script's guess.

On the model SHA specifically, and deliberately: this harness does NOT verify
which weights answered, because it cannot. The server's `/props` payload
carries several SHA-256 values -- the chat template, the llama.cpp library,
tokenizer and prefix identities -- and none of them is the model file. Hashing
the file per request is ruled out upstream on purpose, a 20 GiB read being no
part of a props response. So a check here would have both failure modes at
once: it could never match the real model hash, and it could match one of the
several unrelated hashes that ARE present, reporting "validated against SHA X"
for a run that confirmed only which chat template was loaded. Rather than ship
a check that reads as verification and is not, `--operator-model-sha` is
metadata: the value is recorded verbatim as `operator_model_sha256`, beside
`operator_model_sha256_verified: false`, and nothing here reads it back or
gates on it. The authoritative precondition stays where it can actually be
performed -- `sha256sum <GGUF>` on the host, once, before validation.

Static analysis only. No network is available to the analysed program, and
nothing here fetches, executes or rewrites the artifact beyond the snapshot
the runtime already takes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.llama_server import LlamaServerBackend  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    NO_EVIDENCE_REPORT,
    NO_USABLE_REPORT_TEXT,
    REPORT_NOT_COMPOSED_PREFIX,
    SOURCE_TOO_LARGE_REPORT,
    STOP_BACKEND_ERROR,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

#: Every way the runtime can return a report whose prose is an admission
#: that no answer was produced. Each is matched as a PREFIX, because the
#: deterministic appendix is appended after it whenever the artifact carries
#: an indicator -- which is the common case on a real sample, and the reason
#: an equality test failed open on exactly the artifacts that matter.
#:
#: Imported, never copied. A copy drifts silently: rewording the opening
#: clause in the runtime left every suite green while both composed-failure
#: reports began exiting 0, which is the false pass this gate exists to
#: prevent. Importing makes the harness follow a rename automatically.
#: The oversized-source closing report is a truthful result but NOT an answer:
#: it states that coverage was impossible and asserts nothing about the
#: artifact. The harness must still fail on it, so its stable opening (the part
#: before the `{size}`/`{sha256}` fields) joins the non-answer set. Derived from
#: the imported template by cutting at its first format field, so a reworded
#: template is followed automatically rather than copied and left to drift.
_SOURCE_TOO_LARGE_PREFIX = SOURCE_TOO_LARGE_REPORT.split("{", 1)[0].rstrip()

NON_ANSWER_REPORT_PREFIXES = (
    NO_USABLE_REPORT_TEXT,
    REPORT_NOT_COMPOSED_PREFIX,
    NO_EVIDENCE_REPORT,
    _SOURCE_TOO_LARGE_PREFIX,
)



def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--request", default="Analyse this artifact.",
        help="The analyst message the run starts from.",
    )
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--operator-model-sha",
        help=(
            "Metadata only. The model SHA-256 the operator verified out of "
            "band with `sha256sum <GGUF>`, recorded verbatim in the output "
            "beside `operator_model_sha256_verified: false`. This harness "
            "does not and cannot check it, and never gates on it -- see the "
            "note in the module docstring."
        ),
    )
    return parser


def _report_of(run) -> dict:
    """What the run concluded, read from the authoritative field.

    `run.final_report` is the report. The last step's text is not: a run whose
    last step executed an action carries an empty `text` there while holding a
    perfectly good report, so reading the step would call a valid run a
    failure. Presence and content are recorded separately from any step text
    so that a reader can never confuse the two, and neither is ever inferred
    from a model-call count -- a call that was spent is not a report that
    exists.
    """
    # Attributes are read directly, never through `getattr` with a default.
    # A default turns a renamed or misspelled field into a plausible-looking
    # zero: this recorder read `step.text` when the field is `assistant_text`
    # and reported 0 characters of prose for steps that had hundreds -- the
    # exact reading it exists to stop anyone making. A rename must raise.
    report = run.final_report
    last = run.last_step
    return {
        "final_report_present": report is not None,
        "final_report": (report.text or "") if report is not None else "",
        "document_complete": report.document_complete if report else None,
        "report_limitations": list(report.limitations) if report else [],
        "narrative_status": report.narrative_status if report else None,
        "final_report_evidence_ids": (
            list(report.evidence_ids or ()) if report is not None else []
        ),
        # Diagnostic only, and deliberately adjacent: an empty last-step text
        # beside a present report is the normal shape, not a defect.
        "last_step_text_len": len(last.assistant_text or "") if last else 0,
    }


def _exit_code(error, run) -> int:
    """Zero requires a complete runtime document and no interrupted investigation.

    This is a document/lifecycle gate, not certification of proposed answers.
    OPEN, BLOCKED and ANSWERED_UNVERIFIED questions may all be faithfully
    reported. Optional narrative refusal or truncation does not discard that
    record; absent or unreattestable mandatory evidence still fails the gate.
    """
    if error or run is None or run.cancelled or run.final_report is None:
        return 1
    # The one failure a report does NOT reveal. A backend that dies mid-run
    # still lets the closing report be written from the evidence collected
    # so far, so a report exists while the investigation was cut short. The
    # stop reason is the only place that is recorded, `error` being None for
    # a failure the runtime contained.
    if run.final_report.document_complete is not True:
        return 1
    if str(run.stop_reason or "").startswith(STOP_BACKEND_ERROR):
        return 1
    # Legacy prose-only placeholders never qualify as a complete document.
    if (run.final_report.text or "").lstrip().startswith(
        NON_ANSWER_REPORT_PREFIXES
    ):
        return 1
    return 0


def main() -> int:
    args = _parser().parse_args()

    artifact = args.artifact.resolve()
    data = artifact.read_bytes()
    digest = hashlib.sha256(data).hexdigest()

    backend = LlamaServerBackend(base_url=args.base_url, timeout=args.request_timeout)
    if not backend.health():
        raise SystemExit(f"Orbit server at {args.base_url} is not healthy")
    props = backend.backend_props()

    workspace = AnalysisWorkspace.create()
    try:
        snapshot = workspace.source_root / artifact.name
        snapshot.write_bytes(data)
        runtime = AnalysisRuntime(
            backend=backend,
            source=AnalysisSource(
                snapshot_path=snapshot, sha256=digest,
                size_bytes=len(data), original_path=str(artifact),
            ),
            evidence_store=EvidenceStore(root=workspace.root / "evidence"),
            workspace=workspace,
        )
        started = time.time()
        error = None
        run = None
        try:
            run = runtime.run_autonomous(args.request)
        except BaseException as exc:  # noqa: BLE001 - recorded, not re-raised
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - started

        record = {
            "artifact": str(artifact),
            "artifact_sha256": digest,
            "artifact_bytes": len(data),
            "request": args.request,
            "base_url": args.base_url,
            "backend_props": props,
            # The workspace directory this run owned. Recorded so cleanup is
            # checkable from outside: after `main()` returns, the path must
            # not exist. An internal "closed" flag would only restate what the
            # implementation believes about itself.
            "workspace_root": str(workspace.root),
            # Metadata, never a gate. The value is whatever the operator
            # passed, recorded verbatim; the flag beside it states plainly
            # that nothing here checked it, so a reader cannot mistake the
            # record for an attestation. It is deliberately never compared
            # against the chat-template, tokenizer, library or prefix hashes
            # that DO appear in `backend_props` -- none of them identifies
            # the weights, and matching one would report a verification that
            # did not happen.
            "operator_model_sha256": args.operator_model_sha,
            "operator_model_sha256_verified": False,
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
        }
        # Present on every record, including the failure path: a consumer
        # reading `final_report_present` must not get a KeyError just because
        # the run raised, and `None` is not the same answer as `false`.
        record.update(_report_of(run) if run is not None else {
            "final_report_present": False,
            "final_report": "",
            "final_report_evidence_ids": [],
            "last_step_text_len": 0,
        })
        if run is not None:
            record.update({
                # Did the controller engage at all, and on what.
                "source_covered": run.source_covered,
                "cover_calls": run.cover_calls,
                "plan_calls": run.plan_calls,
                "control_attempts": run.control_attempts,
                "control_repairs": run.control_repairs,
                "initial_questions": run.initial_questions,
                "child_questions": run.child_questions,
                "resolved_questions": list(run.resolved_questions),
                "answered_unverified_questions": list(run.answered_unverified_questions),
                "unverified_questions": list(run.unverified_questions),
                "legacy_resolved_questions": list(run.legacy_resolved_questions),
                "open_questions": list(run.open_questions),
                # Named for free actions, but the runtime fills it from the
                # controller's `rejected_children`: child questions the
                # controller refused. Recorded under its shipped name so the
                # JSON matches the field, with the meaning stated here rather
                # than guessed at by a reader.
                "rejected_child_questions": run.rejected_free_actions,
                "actions_executed": run.actions_executed,
                "model_calls": run.model_calls,
                "suppressed_duplicates": run.suppressed_duplicates,
                "repairs": run.repairs,
                "replans": run.replans,
                "stop_reason": run.stop_reason,
                "cancelled": run.cancelled,
                "steps": [
                    {
                        "action_executed": step.action_executed,
                        "text_len": len(step.assistant_text or ""),
                        "evidence_id": (
                            step.evidence.evidence_id if step.evidence else None
                        ),
                    }
                    for step in run.steps
                ],
                "progress": [r.classification for r in run.progress],
            })

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, default=str))
        print(json.dumps({k: v for k, v in record.items()
                          if k not in ("final_report", "backend_props")},
                         indent=2, default=str))
        return _exit_code(error, run)
    finally:
        workspace.close()


if __name__ == "__main__":
    raise SystemExit(main())
