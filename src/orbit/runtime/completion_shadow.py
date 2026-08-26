"""Observational completion verification for autonomous ANALYSIS.

Shadow mode only. Nothing here may stop, shorten, or otherwise steer a run:
the loop asks this module what it *would* do and then ignores the answer. That
is the whole contract, and it is why the module owns no state the loop reads
back and returns a value the loop only records.

The mechanism is two asymmetric tool-free checks over one content-addressed
snapshot of the ACTIVE evidence. Verifier A asks whether the request is already
satisfied; only if A says COMPLETE does verifier B, which never sees A's
verdict, look for a reason stopping would be premature. `WOULD_STOP` requires
both to agree, both to parse strictly, and every referenced evidence id to
still exist, be ACTIVE, and re-attest.

Every ambiguity resolves to "continue". A malformed verdict, a stale reference,
a snapshot mismatch, or any exception is not a weak yes -- it is a no.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# Distinct from `analysis_step` and `analysis_report` for the same reason those
# two are distinct from each other: the backend keys its rolling checkpoints by
# the phase the caller declares. A verifier that borrowed the step phase would
# ask for the analysis anchor and could replace it; declaring its own phase
# means it requests no anchor at all and the analysis lineage is untouched.
ANALYSIS_COMPLETION_SHADOW_PHASE = "analysis_completion_shadow"

SHADOW_ENV = "ORBIT_ANALYSIS_COMPLETION_SHADOW"

# Bounded so a verifier prompt cannot grow with the session. The snapshot is a
# projection for a yes/no question, not a report context: it carries what was
# asked and what is currently established, and nothing about how the analysis
# got there.
MAX_SNAPSHOT_RECORDS = 12
MAX_SNAPSHOT_CHARS_PER_RECORD = 600
MAX_SNAPSHOT_ARTIFACTS = 16
# The request is bounded like everything else. It is operator-supplied rather
# than hostile, but it is echoed into every checkpoint and the final record, so
# leaving it unbounded would make "bounded by construction" untrue of the one
# term an analyst can make arbitrarily long.
MAX_SNAPSHOT_REQUEST_CHARS = 2000

# Short by construction. Both verifiers answer in one or two lines, and a long
# generation here is a cost paid against a run that is not going to stop.
VERIFIER_MAX_TOKENS = 64

# What a persisted verifier answer may occupy. Enough to audit a parse verdict
# after the fact; far too little to become a transcript. The verifiers are
# instructed to answer in one line, so anything past this is already a protocol
# violation and the truncation is itself the finding.
MAX_PERSISTED_RAW_CHARS = 400

_EVIDENCE_ID_RE = re.compile(r"\bev_[0-9a-f]{12}_[0-9a-f]{16}\b")


def shadow_enabled(env: dict[str, str] | None = None) -> bool:
    """Whether shadow verification runs at all. Off unless explicitly enabled."""
    source = os.environ if env is None else env
    return source.get(SHADOW_ENV, "").strip() == "1"


def scheduled_actions(schedule: str = "after4every2") -> Callable[[int], bool]:
    """Deterministic checkpoint schedules. No adaptive behaviour by design."""
    table = {
        "after3every2": (3, 2),
        "after4every2": (4, 2),
        "after4every3": (4, 3),
    }
    if schedule not in table:
        raise ValueError(f"unknown shadow schedule: {schedule}")
    start, stride = table[schedule]

    def due(actions: int) -> bool:
        return actions >= start and (actions - start) % stride == 0

    return due


@dataclass(frozen=True)
class CompletionSnapshot:
    """What both verifiers see, and the hash that proves they saw the same thing."""

    request: str
    evidence: tuple[tuple[str, str], ...]
    artifacts: tuple[str, ...]
    digest: str

    def render(self) -> str:
        lines = [f"REQUEST: {self.request}", "", "ESTABLISHED EVIDENCE:"]
        for evidence_id, text in self.evidence:
            lines.append(f"[{evidence_id}] {text}")
        if self.artifacts:
            lines.extend(("", "ARTIFACTS:"))
            lines.extend(f"- {handle}" for handle in self.artifacts)
        return "\n".join(lines)


def build_snapshot(
    *,
    request: str,
    records: Sequence[Any],
    load_raw: Callable[[str], str | None],
) -> CompletionSnapshot:
    """The smallest bounded projection of ACTIVE evidence sufficient to judge.

    `records` must already be the ACTIVE view: this does not decide standing,
    it reads it. Feeding superseded records here would let a corrected value
    and the value it replaced argue with each other inside one prompt.
    """
    evidence: list[tuple[str, str]] = []
    artifacts: list[str] = []
    for record in records[-MAX_SNAPSHOT_RECORDS:]:
        evidence_id = str(getattr(record, "evidence_id", "") or "")
        if not evidence_id:
            continue
        raw = load_raw(evidence_id) or ""
        evidence.append((evidence_id, raw[:MAX_SNAPSHOT_CHARS_PER_RECORD]))
        metadata = getattr(record, "metadata", None) or {}
        for artifact in metadata.get("artifacts") or []:
            handle = str(artifact.get("handle") or "")
            if handle and handle not in artifacts:
                artifacts.append(handle)
    artifacts = artifacts[:MAX_SNAPSHOT_ARTIFACTS]
    request = (request or "")[:MAX_SNAPSHOT_REQUEST_CHARS]
    payload = json.dumps(
        {"request": request, "evidence": evidence, "artifacts": artifacts},
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return CompletionSnapshot(
        request=request,
        evidence=tuple(evidence),
        artifacts=tuple(artifacts),
        digest=digest,
    )


VERIFIER_A_INSTRUCTION = (
    "You judge whether an analysis request has already been materially "
    "satisfied by the evidence below.\n"
    "Answer on one line, exactly one of:\n"
    "COMPLETE evidence: <evidence ids that establish it>\n"
    "CONTINUE missing: <one concrete material unresolved requirement>\n"
    "Answer with nothing else."
)

VERIFIER_B_INSTRUCTION = (
    "You look for one material requirement of the request that the evidence "
    "below does NOT yet establish, and that would make it premature to stop.\n"
    "Answer on one line, exactly one of:\n"
    "GAP missing: <one concrete unresolved requirement>\n"
    "NO_GAP\n"
    "Answer with nothing else."
)


@dataclass(frozen=True)
class VerifierVerdict:
    """A parsed verifier answer. `ok` is false for anything not strictly parsed."""

    ok: bool
    decision: str
    detail: str = ""
    evidence_ids: tuple[str, ...] = ()
    raw: str = ""


def parse_verifier_a(text: str) -> VerifierVerdict:
    """Strict. Anything that is not an unambiguous COMPLETE reads as CONTINUE.

    Two properties do the work, and both were holes once. The verdict word is
    matched on a word boundary, so `COMPLETELY unsure` is not a completion.
    And the contradiction check reads the WHOLE answer, not its first line: the
    evidence ids are harvested from the whole answer, so a later line saying
    `CONTINUE` has to be able to disqualify it too.
    """
    stripped = (text or "").strip()
    if not stripped:
        return VerifierVerdict(False, "CONTINUE", "empty verifier output", raw=stripped)
    whole = stripped.upper()
    says_continue = re.search(r"\bCONTINUE\b", whole) is not None
    head = stripped.splitlines()[0].strip()
    if re.match(r"\s*CONTINUE\b", head, re.IGNORECASE):
        return VerifierVerdict(True, "CONTINUE", head[len("CONTINUE"):].strip(), raw=stripped)
    # An answer naming both verdicts is ambiguous however it is arranged, and
    # the conservative reading of an ambiguous answer is the one that keeps going.
    if re.match(r"\s*COMPLETE\b", head, re.IGNORECASE) and not says_continue:
        return VerifierVerdict(
            True,
            "COMPLETE",
            head[len("COMPLETE"):].strip(),
            evidence_ids=tuple(dict.fromkeys(_EVIDENCE_ID_RE.findall(stripped))),
            raw=stripped,
        )
    return VerifierVerdict(False, "CONTINUE", "unparsed verifier output", raw=stripped)


def parse_verifier_b(text: str) -> VerifierVerdict:
    """Strict, and asymmetric: only a clean NO_GAP clears the way.

    `NO_GAP` must be the entire answer. A challenger that clears the way and
    then describes a gap anyway -- on the same line or a later one, in any
    case -- has found a gap, and the trailing prose is the finding rather than
    decoration. `NO_GAPS` is not the token either.
    """
    stripped = (text or "").strip()
    if not stripped:
        return VerifierVerdict(False, "GAP", "empty verifier output", raw=stripped)
    if re.fullmatch(r"\s*NO_GAP\s*\.?\s*", stripped, re.IGNORECASE):
        return VerifierVerdict(True, "NO_GAP", raw=stripped)
    head = stripped.splitlines()[0].strip()
    if re.match(r"\s*GAP\b", head, re.IGNORECASE):
        return VerifierVerdict(True, "GAP", head[len("GAP"):].strip(), raw=stripped)
    # Anything else -- including a NO_GAP that kept talking -- is not a clean
    # clearance, so it reads as a gap.
    return VerifierVerdict(False, "GAP", "unparsed verifier output", raw=stripped)


@dataclass
class ShadowObservation:
    """One checkpoint's result. Diagnostics only; the loop never acts on it."""

    action: int
    snapshot_digest: str
    verifier_a: str | None = None
    verifier_b: str | None = None
    would_stop: bool = False
    blocked_by: str | None = None
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    # Recorded so a decision can be audited offline without rerunning
    # inference. `snapshot` is the exact bounded projection both verifiers saw,
    # which is what the historical corpus lacked; the raw answers are bounded
    # excerpts kept for parser audit, never a transcript.
    snapshot: "CompletionSnapshot | None" = None
    verifier_a_evidence_ids: tuple[str, ...] = ()
    verifier_a_detail: str = ""
    verifier_b_detail: str = ""
    verifier_a_raw: str = ""
    verifier_b_raw: str = ""

    def as_log_fields(self) -> dict[str, object]:
        return {
            "action": self.action,
            "snapshot": self.snapshot_digest[:16],
            "verifier_a": self.verifier_a,
            "verifier_b": self.verifier_b,
            "would_stop": self.would_stop,
            "blocked_by": self.blocked_by,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "wall_seconds": round(self.wall_seconds, 3),
        }


@dataclass
class ShadowLedger:
    """Verifier accounting, deliberately separate from the loop's own budget."""

    observations: list[ShadowObservation] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return sum(item.calls for item in self.observations)

    @property
    def prompt_tokens(self) -> int:
        return sum(item.prompt_tokens for item in self.observations)

    @property
    def output_tokens(self) -> int:
        return sum(item.output_tokens for item in self.observations)

    @property
    def wall_seconds(self) -> float:
        return sum(item.wall_seconds for item in self.observations)

    @property
    def would_stop_actions(self) -> list[int]:
        return [item.action for item in self.observations if item.would_stop]


def evaluate_completion_shadow(
    *,
    action: int,
    snapshot: CompletionSnapshot,
    ask: Callable[[str, str], Any],
    active_evidence_ids: set[str],
    reattest: Callable[[str], object | None],
) -> ShadowObservation:
    """Run the two-stage check and report what it *would* have done.

    `ask(instruction, snapshot_text)` performs one tool-free model call and
    returns an object exposing `content` and, optionally, token counts. It is
    injected so the loop owns the call and this module owns only the policy.
    """
    observation = ShadowObservation(
        action=action, snapshot_digest=snapshot.digest, snapshot=snapshot
    )
    started = time.monotonic()

    def account(response: Any) -> str:
        observation.calls += 1
        observation.prompt_tokens += int(getattr(response, "prompt_tokens", 0) or 0)
        observation.output_tokens += int(getattr(response, "completion_tokens", 0) or 0)
        return str(getattr(response, "content", "") or "")

    try:
        # Rendered inside the guard, not before it: the outer caller absorbs a
        # failure here anyway, but a diagnostic should contain its own.
        rendered = snapshot.render()
        verdict_a = parse_verifier_a(account(ask(VERIFIER_A_INSTRUCTION, rendered)))
        observation.verifier_a = verdict_a.decision
        observation.verifier_a_evidence_ids = verdict_a.evidence_ids
        observation.verifier_a_detail = verdict_a.detail[:MAX_PERSISTED_RAW_CHARS]
        observation.verifier_a_raw = verdict_a.raw[:MAX_PERSISTED_RAW_CHARS]
        if not verdict_a.ok:
            observation.blocked_by = "verifier_a_unparsed"
            return observation
        if verdict_a.decision != "COMPLETE":
            # B is the expensive half and only ever asked about a proposed
            # completion. A run that is still going never pays for it.
            observation.blocked_by = "verifier_a_continue"
            return observation

        # A completion that cites nothing is not checkable, and an uncheckable
        # claim is exactly what the re-attestation gate exists to refuse. This
        # is also what keeps that gate on the passing path rather than only on
        # the failing ones.
        if not verdict_a.evidence_ids:
            observation.blocked_by = "verifier_a_cited_no_evidence"
            return observation

        # Checked before B is asked: a COMPLETE resting on evidence that no
        # longer stands is already disqualified, and asking B would spend a
        # call to arrive at the same answer.
        for evidence_id in verdict_a.evidence_ids:
            if evidence_id not in active_evidence_ids:
                observation.blocked_by = "referenced_evidence_not_active"
                return observation
            if reattest(evidence_id) is None:
                observation.blocked_by = "referenced_evidence_reattest_failed"
                return observation

        # B is handed the same snapshot text and nothing else. It cannot see
        # A's verdict, A's cited ids, or that A ran at all -- otherwise the
        # challenge collapses into agreement with the thing it exists to test.
        verdict_b = parse_verifier_b(account(ask(VERIFIER_B_INSTRUCTION, rendered)))
        observation.verifier_b = verdict_b.decision
        observation.verifier_b_detail = verdict_b.detail[:MAX_PERSISTED_RAW_CHARS]
        observation.verifier_b_raw = verdict_b.raw[:MAX_PERSISTED_RAW_CHARS]
        if not verdict_b.ok:
            observation.blocked_by = "verifier_b_unparsed"
            return observation
        if verdict_b.decision != "NO_GAP":
            observation.blocked_by = "verifier_b_gap"
            return observation

        observation.would_stop = True
        return observation
    except BaseException as exc:  # noqa: BLE001 - a failed verifier must never end a run
        # BaseException, not Exception, and specifically for KeyboardInterrupt:
        # the loop guards Ctrl-C around its own step only, so an interrupt
        # raised here would unwind past `run_autonomous` to a caller holding a
        # pre-run checkpoint and delete the history and provenance of every
        # completed step. A diagnostic must not be able to destroy the run it
        # is observing.
        observation.blocked_by = f"verifier_error: {type(exc).__name__}"
        observation.would_stop = False
        return observation
    finally:
        observation.wall_seconds = time.monotonic() - started


__all__ = [
    "ANALYSIS_COMPLETION_SHADOW_PHASE",
    "MAX_PERSISTED_RAW_CHARS",
    "MAX_SNAPSHOT_REQUEST_CHARS",
    "SHADOW_ENV",
    "VERIFIER_MAX_TOKENS",
    "CompletionSnapshot",
    "ShadowLedger",
    "ShadowObservation",
    "VerifierVerdict",
    "build_snapshot",
    "evaluate_completion_shadow",
    "parse_verifier_a",
    "parse_verifier_b",
    "scheduled_actions",
    "shadow_enabled",
]
