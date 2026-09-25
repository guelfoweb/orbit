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


def render_summary(*, identity, runs, coverage, limitations, records, narrative_status):
    """Short human view. The complete document remains in report.dossier_text.

    Render producer facts, not model prose. An extracted literal and a proven
    network operand have different authority even when their values coincide.
    """
    objectives = coverage.get('network_destination_objectives', [])
    exact = [item for item in objectives if item['state'] == 'RESOLVED_EXACT']
    counts = {state: sum(item['state'] == state for item in objectives)
              for state in ('RESOLVED_EXACT', 'OPEN', 'BLOCKED')}
    lines = ['# Analysis report', '## Summary',
             literal(json.dumps(identity, ensure_ascii=False)),
             (f"Known network destinations: {counts['RESOLVED_EXACT']} exact, "
              f"{counts['OPEN']} open, {counts['BLOCKED']} blocked."
              if objectives else 'No network destination objective was identified by the supported static producers; this does not prove absence.'),
             'Document status: ' + ('INCOMPLETE' if limitations else 'COMPLETE') +
             '. Record completeness is not investigation success or proof of complete behaviour.']
    if runs:
        lines.extend(['Operational stop:', literal(runs[-1]['stop_reason'])])
    lines += ['## Technical behaviour']
    if exact:
        lines.append('Static destination operands were established for ' + str(len(exact)) +
                     ' source assignments. These prove values, not effective browser mutation, observed execution or network contact.')
        lines.extend(['Sink properties:', literal(', '.join(sorted({item['property'] for item in exact})))])
    relationships = coverage.get('static_relationships', [])
    lines.extend(literal(text) for text in relationships)
    if not exact and not relationships:
        lines.append('No additional behavioural relationship was attested by the available producers.')
    lines += ['## Limits',
              'Only known destinations are accounted for; unsupported source constructs and undiscovered destinations remain outside this completion decision. No sample execution or destination contact is established by static proofs.',
              'Acquisition and archived output do not establish complete delivery to the model. Free-form answers remain unverified. The complete original questions, answers, observations, proof chains and provenance are retained in dossier_text and the saved session.']
    if runs and any(state['status'] in ('open', 'blocked') for _q, state in runs[-1]['questions']):
        lines.append('Other questions remain OPEN/BLOCKED; they were not resolved by composing this report.')
    lines.extend(literal(item) for item in limitations)
    for item in objectives:
        if item['state'] != 'RESOLVED_EXACT':
            lines.append(literal(f"{item['id']}: {item['state']}: {item['reason']}"))
    lines += ['## IoC / Evidence']
    by_id = {r.evidence_id: r for r, body in records if body is not None}
    from urllib.parse import urlsplit
    import ipaddress

    def kind(value):
        if '://' in value:
            return 'URL'
        try:
            ipaddress.ip_address(value)
            return 'IP'
        except ValueError:
            return 'domain'

    for item in objectives:
        record = by_id.get(item['evidence_id'])
        value = item['value']
        entry = {'objective': item['id'], 'type': kind(value) if value else 'unresolved network destination',
                 'value': value, 'state': item['state'], 'sink': item['property'],
                 'evidence_id': item['evidence_id'], 'producer': record.metadata.get('transform_kind', record.tool_name) if record else None,
                 'source_sha256': item['source_sha256'], 'source_byte_range': item['source_byte_range']}
        if value and kind(value) == 'URL':
            try:
                entry['hostname_from_URL'] = urlsplit(value).hostname
            except ValueError:
                # An exact source operand is not a guarantee of URL syntax.
                # Preserve it; a derived display field must not lose a report.
                entry['hostname_from_URL'] = None
        lines.append(literal(json.dumps(entry, ensure_ascii=False, indent=2)))
    exact_values = {item['value'] for item in exact}
    for item in coverage.get('literal_indicators', []):
        if item['value'] not in exact_values:
            # The existing producer verifies literal occurrence, not a sink.
            lines.append(literal(json.dumps({'state': 'ATTESTED_LITERAL_NOT_SINK_PROOF',
                                            'source_sha256': identity['sha256'], **item}, ensure_ascii=False)))
    transforms = [r for r, body in records if body is not None and r.produced_by_phase == 'analysis_transform']
    for record in transforms:
        if any(item['evidence_id'] == record.evidence_id for item in exact):
            continue
        lines.append(literal(json.dumps({'evidence_id': record.evidence_id, 'producer': record.tool_name,
            'output_sha256': record.raw_sha256, 'source_sha256': identity['sha256'],
            'state': 'ATTESTED_TRANSFORM_OUTPUT', 'body': 'retained in dossier_text'}, ensure_ascii=False)))
    if not objectives and not coverage.get('literal_indicators') and not transforms:
        lines.append('No currently re-attested network indicator or deterministic output.')
    return '\n\n'.join(lines) + '\n'


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
