"""Durable record of what the completion shadow saw and decided.

Diagnostics, and nothing else. Every write is best-effort: a ledger that cannot
be created or written must leave the analysis exactly as it would have been,
because the alternative is a diagnostic that decides outcomes.

The format is JSONL, one record per line, appended and flushed as it happens.
That choice is about crash behaviour rather than speed: an atomic whole-file
replace would keep the file consistent but would put every earlier observation
at risk of the last write, and the observations are the expensive part -- each
one cost a model call that will not be repeated. Append-only means a run killed
mid-flight keeps everything it had already committed, and the missing final
record is itself how a reader detects that it was killed.

What is persisted is the exact bounded snapshot the verifiers were shown. The
historical corpus failed precisely there: it kept evidence ids and hashes but
not the content, so no later question could be answered without rerunning the
analysis. Hashes prove integrity; they do not reconstruct what was judged.

No raw sandbox output and no second copy of EvidenceStore. Verifier answers are
kept as 400-character excerpts for parser audit, so if a verifier emits
reasoning markup that markup can appear inside an excerpt -- bounded, never a
transcript. The claim here is that nothing unbounded is persisted, not that a
verifier's own words are filtered.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

# Bumped when a reader could otherwise misread an older file. Readers must
# refuse a version they do not know rather than guess at its shape.
LEDGER_SCHEMA_VERSION = 1

LEDGER_FILENAME = "completion-shadow.jsonl"

RECORD_RUN_START = "run_start"
RECORD_CHECKPOINT = "checkpoint"
RECORD_RUN_FINAL = "run_final"


def ledger_path_for_evidence_root(root: Path) -> Path:
    """Beside the evidence it describes, in Orbit's own session directory."""
    return Path(root).parent / f"{Path(root).name}.{LEDGER_FILENAME}"


def _canonical(payload: dict[str, Any]) -> str:
    # Deterministic: sorted keys, no incidental whitespace, UTF-8 preserved.
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class ShadowLedgerWriter:
    """Append-only writer. Never raises into the caller.

    `failures` is kept rather than logged away so a run can report that its
    diagnostics were incomplete instead of a reader inferring completeness from
    a file that simply stopped early.

    Every record carries a `run_id`, because one analysis session can hold more
    than one run and they share this file. Without it a finished run's final
    record would make the next run -- killed halfway -- read as complete, which
    is the one conclusion this ledger must never support.
    """

    path: Path
    enabled: bool = True
    run_id: str = ""
    failures: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []
        if not self.run_id:
            self.run_id = uuid4().hex

    def _append(self, payload: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = _canonical(payload) + "\n"
            # Flushed per record: an observation already paid for must not be
            # lost to a buffer if the process dies during the next model call.
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            return True
        except Exception as exc:  # noqa: BLE001 - diagnostics must not end a run
            # Deliberately not BaseException: a KeyboardInterrupt here is the
            # operator stopping the CLI, and swallowing it to save a diagnostic
            # would break ordinary interrupt semantics.
            self.failures.append(f"{type(exc).__name__}: {exc}")
            return False

    def write_run_start(self, **fields: Any) -> bool:
        return self._append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "record": RECORD_RUN_START,
                "run_id": self.run_id,
                **fields,
            }
        )

    def write_checkpoint(self, observation: Any, **fields: Any) -> bool:
        snapshot = getattr(observation, "snapshot", None)
        payload = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "record": RECORD_CHECKPOINT,
            "run_id": self.run_id,
            "action": observation.action,
            "snapshot_sha256": observation.snapshot_digest,
            "request": getattr(snapshot, "request", ""),
            # The exact bounded projection the verifiers judged, not a
            # reference to evidence that may be unreadable later.
            "snapshot_evidence": [
                {"evidence_id": evidence_id, "text": text}
                for evidence_id, text in (getattr(snapshot, "evidence", ()) or ())
            ],
            "snapshot_artifacts": list(getattr(snapshot, "artifacts", ()) or ()),
            "verifier_a": observation.verifier_a,
            "verifier_a_evidence_ids": list(observation.verifier_a_evidence_ids),
            "verifier_a_detail": observation.verifier_a_detail,
            "verifier_a_raw": observation.verifier_a_raw,
            "verifier_b": observation.verifier_b,
            "verifier_b_detail": observation.verifier_b_detail,
            "verifier_b_raw": observation.verifier_b_raw,
            "would_stop": bool(observation.would_stop),
            "blocked_by": observation.blocked_by,
            "verifier_calls": observation.calls,
            "verifier_prompt_tokens": observation.prompt_tokens,
            "verifier_output_tokens": observation.output_tokens,
            "verifier_wall_seconds": round(observation.wall_seconds, 3),
            # Additive: a v1 reader ignores it and a v1 ledger simply has no
            # fidelity, which reads as unknown rather than lossless. Bumping
            # the schema would have invalidated an expensive corpus that is
            # still perfectly valid for everything it does record.
            "snapshot_fidelity": (
                observation.snapshot_fidelity.as_log_fields()
                if getattr(observation, "snapshot_fidelity", None) is not None
                else None
            ),
            **fields,
        }
        return self._append(payload)

    def write_run_final(self, **fields: Any) -> bool:
        return self._append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "record": RECORD_RUN_FINAL,
                "run_id": self.run_id,
                **fields,
            }
        )


@dataclass(frozen=True)
class LedgerReadResult:
    """What a ledger file contains, and whether it can be trusted."""

    run_start: dict[str, Any] | None
    checkpoints: tuple[dict[str, Any], ...]
    run_final: dict[str, Any] | None
    malformed_lines: tuple[int, ...]
    unsupported_versions: tuple[int, ...]

    @property
    def complete(self) -> bool:
        """A run that never wrote its final record did not finish normally."""
        return self.run_final is not None

    @property
    def lossy_actions(self) -> tuple[int, ...]:
        """Checkpoints whose snapshot was not the whole of its input.

        A checkpoint with no fidelity record is not counted: an older ledger
        that predates the field is unknown, not lossless.
        """
        return tuple(
            int(item["action"])
            for item in self.checkpoints
            if isinstance(item.get("snapshot_fidelity"), dict)
            and item["snapshot_fidelity"].get("lossless") is False
        )

    @property
    def fidelity_unknown_actions(self) -> tuple[int, ...]:
        return tuple(
            int(item["action"])
            for item in self.checkpoints
            if not isinstance(item.get("snapshot_fidelity"), dict)
        )

    @property
    def would_stop_actions(self) -> tuple[int, ...]:
        return tuple(
            int(item["action"]) for item in self.checkpoints if item.get("would_stop")
        )

    @property
    def verifier_calls(self) -> int:
        return sum(int(item.get("verifier_calls") or 0) for item in self.checkpoints)

    @property
    def verifier_wall_seconds(self) -> float:
        return sum(float(item.get("verifier_wall_seconds") or 0.0) for item in self.checkpoints)

    @property
    def verifier_tokens(self) -> tuple[int, int]:
        return (
            sum(int(item.get("verifier_prompt_tokens") or 0) for item in self.checkpoints),
            sum(int(item.get("verifier_output_tokens") or 0) for item in self.checkpoints),
        )


def _snapshot_digest(record: dict[str, Any]) -> str:
    """Recompute the snapshot hash from the persisted projection.

    Mirrors `build_snapshot` exactly. If the two ever drift, a stored decision
    can no longer be tied to the content it was made from, so the reader
    reports a mismatch rather than trusting the stored hash.
    """
    import hashlib

    payload = json.dumps(
        {
            "request": record.get("request", ""),
            "evidence": [
                [item.get("evidence_id", ""), item.get("text", "")]
                for item in record.get("snapshot_evidence") or []
            ],
            "artifacts": list(record.get("snapshot_artifacts") or []),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_snapshot_hashes(result: LedgerReadResult) -> tuple[int, ...]:
    """Actions whose persisted snapshot does not re-hash to its stored digest."""
    return tuple(
        int(record["action"])
        for record in result.checkpoints
        if _snapshot_digest(record) != record.get("snapshot_sha256")
    )


def read_ledger(path: Path) -> LedgerReadResult:
    """Parse a ledger, tolerating truncation and refusing unknown versions.

    A partially written last line is expected rather than exceptional: the
    process may have been killed mid-write. It is reported, never interpreted.
    """
    run_start: dict[str, Any] | None = None
    checkpoints: list[dict[str, Any]] = []
    run_final: dict[str, Any] | None = None
    malformed: list[int] = []
    versions: list[int] = []
    records_in_order: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            malformed.append(number)
            continue
        if not isinstance(payload, dict):
            malformed.append(number)
            continue
        version = payload.get("schema_version")
        if version != LEDGER_SCHEMA_VERSION:
            if isinstance(version, int):
                versions.append(version)
            else:
                malformed.append(number)
            continue
        kind = payload.get("record")
        if kind in (RECORD_RUN_START, RECORD_CHECKPOINT, RECORD_RUN_FINAL):
            records_in_order.append(payload)
        if kind == RECORD_RUN_START:
            run_start = payload
        elif kind == RECORD_CHECKPOINT:
            if "action" in payload:
                checkpoints.append(payload)
            else:
                malformed.append(number)
        elif kind == RECORD_RUN_FINAL:
            run_final = payload
        else:
            malformed.append(number)
    # One session can hold several runs in one file, so the result describes
    # the most recent run rather than a blend of all of them. "Most recent"
    # means the run whose record appears last: two genuinely concurrent runs
    # sharing one file would leave the older one invisible here. That is the
    # conservative direction -- it never reports a killed run as complete --
    # and the sequential single-session case is the one this serves.
    #
    # The current run is identified by the newest id seen on ANY record, not by
    # the newest `run_start`. A `run_start` can be lost on its own -- each
    # record opens the file independently, so a transient ENOSPC can drop it
    # while later writes succeed -- and keying off it would then credit the
    # previous run's final record to this one, which is exactly the "killed run
    # looks complete" conclusion this ledger must never support.
    current = None
    for item in reversed(records_in_order):
        candidate = item.get("run_id")
        if isinstance(candidate, str) and candidate:
            current = candidate
            break
    if current is None:
        # No record carries an id: the file predates them or is unattributable,
        # so nothing may be credited to a specific run.
        checkpoints = []
        run_final = None
        run_start = None
    else:
        checkpoints = [item for item in checkpoints if item.get("run_id") == current]
        if run_final is not None and run_final.get("run_id") != current:
            run_final = None
        if run_start is not None and run_start.get("run_id") != current:
            run_start = None
    return LedgerReadResult(
        run_start=run_start,
        checkpoints=tuple(checkpoints),
        run_final=run_final,
        malformed_lines=tuple(malformed),
        unsupported_versions=tuple(sorted(set(versions))),
    )


__all__ = [
    "LEDGER_FILENAME",
    "LEDGER_SCHEMA_VERSION",
    "LedgerReadResult",
    "RECORD_CHECKPOINT",
    "RECORD_RUN_FINAL",
    "RECORD_RUN_START",
    "ShadowLedgerWriter",
    "ledger_path_for_evidence_root",
    "read_ledger",
    "verify_snapshot_hashes",
]
