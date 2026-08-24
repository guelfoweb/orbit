"""Generic progress classification for bounded autonomous analysis.

Runtime-side orchestration only. This module reasons about verifiable state
changes -- evidence content hashes, artifact digests, action identity -- and
never about what an artifact means. There is no malware, encoding, protocol or
indicator knowledge here, and nothing in it can be made to depend on any.

Recovered from the preserved research harness
(`research/autonomous-malware-analysis/harnesses/ornith-native-ptc/progress_controller.py`)
which carried the same disclaimer. What is reused is the shape: an append-only
ledger, action fingerprints, and a three-way classification driving a bounded
loop. What is deliberately not reused is its `evidence_atoms` heuristic -- a
regex over runs of >=24 non-whitespace characters, normalised for quoting.
That predicate existed because the harness had no evidence store to ask; it
guessed at novelty from the text of an observation. Production has
`EvidenceStore`, which already content-addresses every action's full output as
`raw_sha256` and records a sha256 per artifact, so novelty here is read from
attested state rather than inferred from prose.

Classification of one executed action:

  NEW_CONTENT   the action produced an evidence content hash not seen before,
                or an artifact digest that is new or changed for its handle.
  NO_PROGRESS   the action executed and re-derived only what is already known.
  ERROR         the action was attempted but not executed, or the model
                produced a structurally invalid call.
  COMPLETE      the model returned prose and attempted no action. Control
                belongs to the analyst; the runtime does not judge whether the
                analysis is finished.

Model prose is never evidence of progress. A step whose only output is text
classifies as COMPLETE, never NEW_CONTENT, however much the text asserts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NEW_CONTENT = "NEW_CONTENT"
NO_PROGRESS = "NO_PROGRESS"
ERROR = "ERROR"
COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class ProgressRecord:
    """What one step contributed, judged against everything before it."""

    index: int
    classification: str
    evidence_sha256: str | None = None
    code_sha256: str | None = None
    new_artifacts: tuple[str, ...] = ()
    changed_artifacts: tuple[str, ...] = ()
    repeated_action: bool = False
    detail: str | None = None


@dataclass
class ProgressLedger:
    """Append-only record of what an autonomous run has already established.

    Holds hashes, never content: it decides whether something is new, and has
    no way to decide whether something is interesting.
    """

    known_evidence: set[str] = field(default_factory=set)
    artifacts: dict[str, str] = field(default_factory=dict)
    action_fingerprints: set[tuple[str, str]] = field(default_factory=set)
    history: list[ProgressRecord] = field(default_factory=list)

    def classify(self, index: int, step: object) -> ProgressRecord:
        """Classify one `AnalysisStepResult` against the ledger, then absorb it.

        Reads only fields the production runtime already fills in. A step is
        passed whole rather than destructured by the caller so that the rule
        for what counts as progress lives in exactly one place.
        """
        attempted = bool(getattr(step, "action_attempted", False))
        executed = bool(getattr(step, "action_executed", False))
        rejection = getattr(step, "rejection", None)

        if not attempted:
            # Prose with no action. Not a failure and not progress: the model
            # has nothing further it wants to run, so control returns.
            return self._append(
                ProgressRecord(index=index, classification=COMPLETE)
            )

        if not executed:
            # Attempted but refused: malformed call, capacity, sandbox refusal.
            # The refusal text is carried so a bound can be reported truthfully.
            return self._append(
                ProgressRecord(
                    index=index,
                    classification=ERROR,
                    detail=str(rejection) if rejection else "action not executed",
                )
            )

        evidence = getattr(step, "evidence", None)
        evidence_sha = getattr(evidence, "raw_sha256", None)
        code_sha = self._code_sha256(step, evidence)

        new_artifacts, changed_artifacts = self._artifact_delta(evidence)
        new_evidence = bool(evidence_sha) and evidence_sha not in self.known_evidence

        if new_evidence or new_artifacts or changed_artifacts:
            classification = NEW_CONTENT
        else:
            classification = NO_PROGRESS

        fingerprint = (code_sha or "", evidence_sha or "")
        repeated = fingerprint in self.action_fingerprints

        record = ProgressRecord(
            index=index,
            classification=classification,
            evidence_sha256=evidence_sha,
            code_sha256=code_sha,
            new_artifacts=new_artifacts,
            changed_artifacts=changed_artifacts,
            repeated_action=repeated,
        )
        if evidence_sha:
            self.known_evidence.add(evidence_sha)
        self.action_fingerprints.add(fingerprint)
        return self._append(record)

    def _append(self, record: ProgressRecord) -> ProgressRecord:
        self.history.append(record)
        return record

    @staticmethod
    def _code_sha256(step: object, evidence: object) -> str | None:
        result = getattr(step, "result", None)
        code_sha = getattr(result, "code_sha256", None)
        if code_sha:
            return str(code_sha)
        metadata = getattr(evidence, "metadata", None) or {}
        value = metadata.get("code_sha256") if isinstance(metadata, dict) else None
        return str(value) if value else None

    def _artifact_delta(self, evidence: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Handles whose digest is new, and handles whose digest changed.

        Both count as progress. A handle appearing for the first time is new
        transport; the same handle carrying different bytes is a genuine state
        transition, which is exactly what a multi-stage transformation looks
        like from outside. Rewriting a file with identical bytes is neither.
        """
        metadata = getattr(evidence, "metadata", None) or {}
        entries = metadata.get("artifacts") if isinstance(metadata, dict) else None
        if not isinstance(entries, list):
            return (), ()
        new: list[str] = []
        changed: list[str] = []
        seen: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            handle = entry.get("handle") or entry.get("name")
            digest = entry.get("sha256")
            if not handle or not digest:
                continue
            handle, digest = str(handle), str(digest)
            seen[handle] = digest
            previous = self.artifacts.get(handle)
            if previous is None:
                new.append(handle)
            elif previous != digest:
                changed.append(handle)
        self.artifacts.update(seen)
        return tuple(new), tuple(changed)


__all__ = [
    "COMPLETE",
    "ERROR",
    "NEW_CONTENT",
    "NO_PROGRESS",
    "ProgressLedger",
    "ProgressRecord",
]
