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

Two things are asked of every executed action, and both must pass for it to
count as progress:

  1. Is this a strategy that has not already been tried against this state?
     The fingerprint is (code sha, source identity, pre-action workspace
     state). Re-running the same program against the same inputs is the same
     experiment, and an experiment already run cannot produce new knowledge --
     however different its stdout looks. That is what makes a program printing
     a timestamp, a pid or a random value stop counting as discovery.
  2. Did it actually add attested state -- an evidence content hash not seen
     before, or an artifact digest new or changed for its handle?

Classification of one executed action:

  NEW_CONTENT   an unseen strategy that also added attested state.
  NO_PROGRESS   a repeated strategy, or a strategy that added nothing.
  ERROR         the action was attempted but not executed, or the model
                produced a structurally invalid call.
  COMPLETE      the model returned prose and attempted no action. Control
                belongs to the analyst; the runtime does not judge whether the
                analysis is finished.

Model prose is never evidence of progress. A step whose only output is text
classifies as COMPLETE, never NEW_CONTENT, however much the text asserts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

NEW_CONTENT = "NEW_CONTENT"
NO_PROGRESS = "NO_PROGRESS"
ERROR = "ERROR"
COMPLETE = "COMPLETE"


def observation_fingerprint(
    code_sha256: str | None,
    source_sha256: str | None,
    workspace_digests: "dict[str, str] | None" = None,
) -> str:
    """Identity of one deterministic experiment, before or after it runs.

    The same program, over the same input, against the same workspace, is the
    same experiment -- so running it again cannot establish anything the
    session does not already hold. Three hashes the runtime already has; no
    normalisation of code or output, and nothing that inspects what any of it
    means.

    Pure and side-effect free precisely so the loop can ask the question in
    both directions: `ProgressLedger` asks it after an action to classify what
    happened, and the runtime asks it before an action to decide whether
    executing it could tell anyone anything. Both must agree on what "the same
    experiment" is, which is why there is one definition rather than two.
    """
    workspace = ";".join(f"{h}={d}" for h, d in sorted((workspace_digests or {}).items()))
    components = "\x00".join([code_sha256 or "", str(source_sha256 or ""), workspace])
    # `surrogatepass`, because a scratch filename is whatever the filesystem
    # holds: a name with undecodable bytes reaches Python as lone surrogates,
    # which plain UTF-8 encoding refuses. This is a hash of state, so any
    # total, injective encoding will do -- and refusing to compute one would
    # turn an unreadable filename into a failed analysis step, which is the
    # recovery path this runtime already handles by design.
    return hashlib.sha256(components.encode("utf-8", "surrogatepass")).hexdigest()


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
    repeated_strategy: bool = False
    detail: str | None = None

    @property
    def is_new_content(self) -> bool:
        """Whether this step added verifiably new state.

        Read by the loop to decide whether a run may continue past its action
        budget: only a step that actually advanced the analysis earns that.
        """
        return self.classification == NEW_CONTENT


@dataclass
class ProgressLedger:
    """Append-only record of what an autonomous run has already established.

    Holds hashes, never content: it decides whether something is new, and has
    no way to decide whether something is interesting.
    """

    known_evidence: set[str] = field(default_factory=set)
    artifacts: dict[str, str] = field(default_factory=dict)
    action_fingerprints: set[tuple[str, str]] = field(default_factory=set)
    strategy_fingerprints: set[str] = field(default_factory=set)
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

        suppressed = getattr(step, "suppressed_duplicate_of", None)
        if suppressed:
            # The runtime declined to re-run an experiment already run against
            # this exact state. Nothing failed, so this is not an ERROR -- it
            # must not spend the consecutive-error budget on a model that is
            # behaving reasonably. It is the clearest possible NO_PROGRESS: not
            # "this added nothing" inferred after the fact, but "this could not
            # have added anything", known before it ran.
            return self._append(
                ProgressRecord(
                    index=index,
                    classification=NO_PROGRESS,
                    repeated_action=True,
                    repeated_strategy=True,
                    detail=f"duplicate observation suppressed: {suppressed}",
                )
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

        # Computed before the delta is absorbed: the state a strategy is
        # judged against is the state it actually ran against, not the one it
        # produced.
        strategy = self._strategy_fingerprint(step, evidence, code_sha)
        repeated_strategy = strategy in self.strategy_fingerprints

        new_artifacts, changed_artifacts = self._artifact_delta(evidence)
        new_evidence = bool(evidence_sha) and evidence_sha not in self.known_evidence
        added_state = bool(new_evidence or new_artifacts or changed_artifacts)

        # A repeated strategy is not discovery -- unless it materialised or
        # changed an artifact. Writing a durable handle is transport, not
        # inference: the bytes become addressable and re-attestable, which
        # changes what the session can do next rather than restating what it
        # already knew. Re-writing a handle with identical bytes is neither,
        # and the delta is empty there.
        #
        # This exception is deliberately not bounded further. A program that
        # writes on every run may be carving the next stage out of a packed
        # artifact -- exactly the work autonomous analysis exists for -- or it
        # may be rewriting random bytes and going nowhere. No deterministic
        # signal available here separates them: both reuse one program, both
        # write every run, and both produce a new evidence hash because the
        # observation names the digest they just wrote. Every rule strong
        # enough to stop the second was measured to halt the first mid-unpack,
        # which is the worse failure and the one this mechanism exists to
        # prevent. The action bound ends the spinning case, and ends it safely.
        if added_state and (not repeated_strategy or new_artifacts or changed_artifacts):
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
            repeated_strategy=repeated_strategy,
        )
        if evidence_sha:
            self.known_evidence.add(evidence_sha)
        self.action_fingerprints.add(fingerprint)
        self.strategy_fingerprints.add(strategy)
        return self._append(record)

    def _strategy_fingerprint(
        self, step: object, evidence: object, code_sha: str | None
    ) -> str:
        """Identity of the experiment, independent of what it printed.

        The same program, over the same input, against the same workspace, is
        the same experiment. Its stdout may still differ -- a timestamp, a pid,
        a random value, a dict that iterates in a new order -- and none of that
        is a discovery about the artifact.

        The workspace component is the set of artifact handles and digests the
        ledger had already absorbed when this action ran, so a program that is
        genuinely re-run after the workspace changed is a different experiment
        and is judged on its own merits. Nothing here inspects or normalises
        code or output: it is three hashes of state the runtime already had.
        """
        metadata = getattr(evidence, "metadata", None) or {}
        result = getattr(step, "result", None)
        source = getattr(result, "input_sha256", None)
        if not source and isinstance(metadata, dict):
            source = metadata.get("input_sha256") or metadata.get("analysis_source_sha256")
        return observation_fingerprint(code_sha, source, self.artifacts)

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
    "observation_fingerprint",
]
