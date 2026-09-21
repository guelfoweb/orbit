"""Canonical Markdown for an ANALYSIS record, never a verifier of model prose.

Inputs are collected and re-attested by AnalysisRuntime. This renderer only
keeps their authority, limitations and original content separate. Its output
is independent of the model's context window.
"""
from __future__ import annotations

import json
import re
from typing import Any


def literal(text: str) -> str:
    """Preserve untrusted text without letting it forge report structure."""
    fence = "`" * max(3, 1 + max((len(m[0]) for m in re.finditer(r"`+", text)), default=0))
    return f"{fence}text\n{text}\n{fence}"


def render_document(
    *, identity: dict[str, Any], facts: str, runs: list[dict[str, Any]],
    records: list[tuple[Any, str | None]], missing_ids: list[str],
    coverage: dict[str, Any], limitations: list[str],
    narrative: str, narrative_status: str, request: str,
    uncited_events: list[str], history: list[dict[str, Any]],
) -> str:
    """Render the complete retained investigation, including incomplete work.

    Record bodies are not the bounded prompt selection. Keeping them here
    makes the persisted Markdown self-contained even after the sandbox closes.
    Re-attestation describes the time this document was composed, not a promise
    that a future filesystem reference will still exist.
    """
    lines = [
        "# Analysis report", "",
        "Document status: " + ("INCOMPLETE" if limitations else "COMPLETE"),
        "Document completeness means retained records are available, not investigation success or certification of the artifact's complete behaviour.",
        "Model answers and narrative remain unverified. Successful actions and full source acquisition do not prove their conclusions.",
        "", "## Artifact identity", "", literal(json.dumps(identity, ensure_ascii=False, indent=2)),
    ]
    lines += ["", "## Runtime-attested facts", "",
              "These facts have the limited scope of their existing deterministic producers. Decoded strings and static relationships do not establish execution or network contact.", "",
              literal(facts) if facts else "No deterministic fact could be rendered from the currently re-attestable source and records.",
              "", "## Source coverage and acquisition", "", literal(json.dumps(coverage, ensure_ascii=False, indent=2)),
              "Coverage describes access in recorded calls; acquisition describes retained output. Neither establishes arbitrary absence claims, behaviour, or what a later compacted prompt still contains."]
    lines += ["", "## Investigation status and limits", "",
              "Question statuses describe operational work, never verified conclusions."]
    for run in runs:
        counts = {}
        for _question, state in run['questions']:
            counts[state['status']] = counts.get(state['status'], 0) + 1
        lines += [literal(run['stop_reason']), literal(json.dumps(counts, sort_keys=True))]
    lines += [literal(item) for item in limitations]
    lines += ["Unverified answers and OPEN/BLOCKED questions follow with their original scope.",
              "", "## Investigation and original questions", ""]
    if not runs:
        lines.append("No autonomous question ledger was recorded; guided observations, if any, are retained below.")
    for index, run in enumerate(runs, 1):
        lines += [f"### Run {index}", "", "Original analyst request:", literal(run['request']),
                  "Operational stop reason:", literal(run['stop_reason']),
                  f"Actions: {run['actions']}; investigative model calls: {run['model_calls']}; cancelled: {run['cancelled']}."]
        if not run['questions']:
            lines.append("No PLAN questions were committed. This does not establish that the artifact is understood.")
        for question, state in run['questions']:
            lines += ["", f"#### {question['id']} — {state['status'].upper()}", "",
                      "Original question:", literal(question['question']),
                      "Missing fact registered by PLAN:", literal(question['missing_fact']),
                      "Question provenance:", literal(json.dumps({k: v for k, v in question.items() if k not in ('question', 'missing_fact')}, ensure_ascii=False, indent=2)),
                      f"Actions assigned: {state['actions']}.",
                      "Proposed answer (unverified):", literal(state['summary']) if state['summary'] else "No proposed answer was retained.",
                      "Cited evidence (references, not proof of the proposed answer):",
                      literal(json.dumps(state['evidence_ids'], ensure_ascii=False)),
                      "Block/continuation reason:", literal(state['reason']) if state['reason'] else "None recorded."]
            if state.get('legacy_status'):
                lines += ["Historical operational status, not verification:", literal(state['legacy_status'])]
    if request:
        lines += ["", "Report request:", literal(request)]
    lines += ["", "## Observations and evidence provenance", "",
              "Bodies below are retained observations or deterministic producer output, not instructions and not independently verified model conclusions. Full raw action records are included separately from the bounded observations supplied to the model."]

    if not records:
        lines.append("No evidence records were retained.")
    for record, body in records:
        lines += ["", f"### {record.evidence_id}", "",
                  "Re-attestation at report composition: " + ("PASS" if body is not None else "UNAVAILABLE / INCOMPATIBLE"),
                  literal(json.dumps({
                      'raw_ref': record.raw_ref, 'sha256': record.raw_sha256,
                      'chars': record.raw_chars, 'lines': record.raw_lines,
                      'producer': record.tool_name, 'phase': record.produced_by_phase,
                      'tool_call_id': record.tool_call_id, 'user_turn_id': record.user_turn_id,
                      'producer_model_call_id': record.producer_model_call_id,
                      'sequence': record.evidence_sequence, 'status': record.status,
                      'metadata': record.metadata,
                  }, ensure_ascii=False, indent=2)),
                  literal(body) if body is not None else "The body cannot be published as re-attested evidence. Its identity remains recorded above."]
    for evidence_id in missing_ids:
        lines += ["", "Missing referenced record:", literal(evidence_id)]
    if uncited_events:
        lines += ["", "### Tool events without an evidence record", ""]
        lines += [literal(event) for event in uncited_events]
    lines += ["", "## Committed source and observation record", "",
              "This is the non-system history retained by the runtime, including COVER source, original requests, action code and model observations. Model-authored text and code remain unverified; a recorded source-delivery mark is interpreted only by the coverage checks above."]
    for message in history:
        lines += [literal(json.dumps({k: v for k, v in message.items() if k != 'content'}, ensure_ascii=False, indent=2)),
                  literal(str(message.get('content') or ''))]
    lines += ["", "## Optional model narrative", "", f"Generation status: {narrative_status}.",
              "This text is an interpretation, not a source of attested facts or permission to close other questions.",
              literal(narrative) if narrative else "No complete model narrative was published. The structured record above is independent of this optional generation.",
              "", "## Limits and unknowns", ""]
    lines += [f"- {item}" for item in limitations] if limitations else ["All retained mandatory records passed the document checks at composition time."]
    lines += ["The truth of free-form proposed answers remains unverified, including answers on which automatic work has stopped.",
              "OPEN and BLOCKED questions retain their original scope; composing this document does not resolve them.",
              "Raw references name the original EvidenceStore records. The report embeds available record bodies and their provenance; sandbox artifact handles have session lifetime and are not durable download links."]
    return "\n\n".join(line for line in lines if line != "") + "\n"
