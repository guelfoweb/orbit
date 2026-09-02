"""A finite ledger of questions that actually require running something.

Source acquisition stopped being the problem. The live run that motivated this
supplied the whole artifact, suppressed the one re-read, and still spent seven
actions -- four of them re-deriving facts the source already showed (twice, in
near-identical pairs), and three genuinely needing execution. None of the seven
was caused by an earlier result. The pattern is not bad reasoning; it is that
every valid new action counts as progress, so an investigation grows until a
ceiling stops it.

The ledger bounds one thing: which TOOL ACTIONS may run. After coverage the
model declares the concrete questions it cannot answer from the source it was
given, and each action must name one of them. When none is open, the run
reports.

What it deliberately does not bound is what the model may *know*. A finding
visible in the source needs no question, cannot be suppressed by the ledger,
and belongs in the report whether or not anyone declared it. An empty ledger
means "nothing left that needs a tool" -- never "the analysis is complete", and
never "there is nothing more to say".

The other half of that promise is that an unresolved question stays visibly
unresolved. A question the model could not answer is reported as open, with
whatever evidence it did produce. Nothing here marks a question resolved: only
the model does that, and only about the question its own action just ran.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# States a question can be in. `OPEN` is the only one an action may target.
OPEN = "open"
RESOLVED = "resolved"

# What bounds expansion, on top of the global action ceiling. Depth 1 means a
# question raised by a result may itself raise nothing: the live evidence shows
# no causal chains at all, so anything deeper would be building for a case the
# artifact has never produced.
MAX_CHILD_DEPTH = 1
# The most questions a plan may declare, sized so the whole ledger fits the
# existing model-call budget: coverage and planning cost two calls, each
# question costs one to run and one to classify, so six questions is fourteen
# of the fifteen available. A plan that could not be worked through would be
# one the run cannot honour, which is worse than a shorter one.
#
# Children can push past that, and deliberately are not separately capped: the
# global ceiling is what stops them, and a run that hits it reports with its
# open questions still shown as open. Nothing is lost when that happens -- the
# ledger simply stops being what bounded the run.
MAX_INITIAL_QUESTIONS = 6
# The most questions a ledger may ever hold, children included. Without it the
# only bound on growth is the global action ceiling -- one distinct child per
# action, indefinitely -- which makes termination a property of the ceiling
# rather than of the ledger, and lets the rendered ledger grow into every later
# prompt until admission refuses it. Twice the initial cap is exactly "each
# declared question may raise one child, and no more".
MAX_TOTAL_QUESTIONS = MAX_INITIAL_QUESTIONS * 2

# Identifiers are model-authored text that ends up in prompts and provenance.
MAX_ID_CHARS = 16
MAX_TEXT_CHARS = 300
# A plan is a short JSON document. Anything vastly larger is not one.
MAX_PLAN_CHARS = 20_000

_ID = re.compile(r"\A[A-Za-z][A-Za-z0-9_.-]{0,15}\Z")


class LedgerError(ValueError):
    """A plan or classification that cannot be trusted. Always fails closed."""


@dataclass(frozen=True)
class Question:
    """One thing the model says it cannot answer without running something."""

    id: str
    question: str
    why: str
    depth: int = 0
    parent: str | None = None
    caused_by: str | None = None

    def as_line(self) -> str:
        return f"{self.id} | {self.question} | {self.why}"


@dataclass
class QuestionLedger:
    """The open/resolved state of one analysis, and the rules that move it.

    Deliberately a plain record with explicit transitions rather than anything
    that decides on the model's behalf. It refuses; it never resolves.
    """

    questions: "dict[str, Question]" = field(default_factory=dict)
    state: "dict[str, str]" = field(default_factory=dict)
    evidence: "dict[str, str]" = field(default_factory=dict)
    order: "list[str]" = field(default_factory=list)
    rejected_free_actions: int = 0
    rejected_children: int = 0
    reopen_attempts: int = 0

    @property
    def open_ids(self) -> "list[str]":
        return [qid for qid in self.order if self.state.get(qid) == OPEN]

    @property
    def resolved_ids(self) -> "list[str]":
        return [qid for qid in self.order if self.state.get(qid) == RESOLVED]

    @property
    def exhausted(self) -> bool:
        """No question is open. Never means the analysis is complete."""
        return not self.open_ids

    def add(self, question: Question) -> None:
        if question.id in self.questions:
            raise LedgerError(f"duplicate question id: {question.id}")
        self.questions[question.id] = question
        self.state[question.id] = OPEN
        self.order.append(question.id)

    def is_open(self, qid: str) -> bool:
        return self.state.get(qid) == OPEN

    def resolve(self, qid: str, evidence_id: str) -> None:
        """Mark a question answered, on the model's word plus a reference.

        The evidence id is required and recorded so a reader can check the
        claim. The runtime does not verify that the evidence answers the
        question -- that judgement is the analysis itself -- but it does insist
        that something was named.
        """
        if not self.is_open(qid):
            raise LedgerError(f"not an open question: {qid}")
        if not evidence_id:
            raise LedgerError(f"no evidence named for {qid}")
        self.state[qid] = RESOLVED
        self.evidence[qid] = evidence_id

    def reopen(self, qid: str, caused_by: str) -> None:
        """A resolved question may return only on new causal evidence.

        Without that rule a run can cycle: resolve, reopen, resolve again,
        spending the ceiling on a question it has already answered.
        """
        if self.state.get(qid) != RESOLVED:
            raise LedgerError(f"not a resolved question: {qid}")
        if not caused_by or caused_by == self.evidence.get(qid):
            self.reopen_attempts += 1
            raise LedgerError(
                f"reopening {qid} requires new evidence it was not resolved by"
            )
        self.state[qid] = OPEN

    def accept_child(self, child: Question) -> None:
        """A question raised by a result, admitted only if it is really caused.

        Four conditions, and each rules out a different way a run grows without
        bound: the parent must exist and have just run, the evidence that
        caused it must be named, the depth cap must hold, and it must not
        restate something already asked.
        """
        parent = self.questions.get(child.parent or "")
        if parent is None:
            self.rejected_children += 1
            raise LedgerError(f"child {child.id} names no known parent")
        if not child.caused_by:
            self.rejected_children += 1
            raise LedgerError(f"child {child.id} names no causing evidence")
        if child.depth > MAX_CHILD_DEPTH:
            self.rejected_children += 1
            raise LedgerError(
                f"child {child.id} is deeper than the permitted {MAX_CHILD_DEPTH}"
            )
        if child.id in self.questions:
            self.rejected_children += 1
            raise LedgerError(f"duplicate question id: {child.id}")
        if len(self.questions) >= MAX_TOTAL_QUESTIONS:
            self.rejected_children += 1
            raise LedgerError(
                f"ledger already holds {MAX_TOTAL_QUESTIONS} questions"
            )
        if _restates(child, self.questions.values()):
            self.rejected_children += 1
            raise LedgerError(f"child {child.id} restates an existing question")
        self.add(child)

    def render(self) -> str:
        """The ledger as the model sees it, open questions first."""
        lines = []
        for qid in self.order:
            mark = "OPEN" if self.state[qid] == OPEN else "RESOLVED"
            suffix = ""
            if self.state[qid] == RESOLVED:
                suffix = f" (evidence:{self.evidence.get(qid, '?')})"
            lines.append(f"{qid} [{mark}]{suffix}: {self.questions[qid].question}")
        return "\n".join(lines)


def _normalise(text: str) -> str:
    """Lowercased alphanumeric words, for exact-duplicate detection only.

    This is not similarity matching and must not become it: it catches a child
    that is literally the same question reworded in case or spacing. Anything
    that requires judgement about meaning is the model's to make, and a child
    that is genuinely new is admitted even if it reads like an existing one.
    """
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _restates(child: Question, existing) -> bool:
    target = _normalise(child.question)
    return any(_normalise(other.question) == target for other in existing)


def parse_plan(text: str) -> "list[Question]":
    """Read a planning response into questions, or raise.

    The wire format is a JSON object with a `questions` array. It is parsed
    strictly: unknown shapes, missing fields, bad ids and over-long text all
    raise rather than being coerced, because a plan that has to be guessed at
    is not a plan the run can be held to.

    An empty list is valid and meaningful -- it says the source answered
    everything -- so it returns `[]` rather than raising.
    """
    if not text or len(text) > MAX_PLAN_CHARS:
        raise LedgerError("plan is empty or too large")
    payload = _extract_json_object(text)
    if not isinstance(payload, dict):
        raise LedgerError("plan is not a JSON object")
    raw = payload.get("questions")
    if raw is None:
        raise LedgerError("plan has no `questions` field")
    if not isinstance(raw, list):
        raise LedgerError("`questions` is not a list")
    if len(raw) > MAX_INITIAL_QUESTIONS:
        raise LedgerError(
            f"plan declares {len(raw)} questions, more than {MAX_INITIAL_QUESTIONS}"
        )
    questions: list[Question] = []
    seen: set[str] = set()
    for entry in raw:
        question = _question_from(entry)
        if question.id in seen:
            raise LedgerError(f"duplicate question id: {question.id}")
        seen.add(question.id)
        questions.append(question)
    return questions


def _question_from(entry: object) -> Question:
    if not isinstance(entry, dict):
        raise LedgerError("question entry is not an object")
    qid = entry.get("id")
    text = entry.get("question")
    why = entry.get("why")
    for value, name in ((qid, "id"), (text, "question"), (why, "why")):
        if not isinstance(value, str) or not value.strip():
            raise LedgerError(f"question is missing a usable `{name}`")
    if not _ID.match(qid):
        raise LedgerError(f"question id is not a plain identifier: {qid!r}")
    if len(text) > MAX_TEXT_CHARS or len(why) > MAX_TEXT_CHARS:
        raise LedgerError(f"question {qid} is longer than {MAX_TEXT_CHARS} characters")
    return Question(id=qid, question=text.strip(), why=why.strip())


def _extract_json_object(text: str) -> object:
    """The JSON object in a model response, tolerating prose around it.

    Models put a sentence before the JSON, or fence it. Only the object is
    read; everything else is ignored. This is not tolerant parsing of the plan
    itself -- the object's contents are still checked strictly -- it is
    tolerance about where the object sits in the reply.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except ValueError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise LedgerError("no JSON object found in the plan")
    try:
        return json.loads(stripped[start : end + 1])
    except ValueError as exc:
        raise LedgerError(f"plan is not valid JSON: {exc}") from exc


def parse_resolution(text: str, ledger: QuestionLedger) -> "tuple[str, str, str]":
    """Read a `question / state / evidence` classification, or raise.

    Returns `(question_id, state, evidence_id)`. The state must be one the
    ledger recognises and the id must name a question that is currently open --
    a classification about anything else is not something the run can act on.
    """
    payload = _extract_json_object(text)
    if not isinstance(payload, dict):
        raise LedgerError("resolution is not a JSON object")
    qid = payload.get("question")
    state = payload.get("state")
    evidence = payload.get("evidence") or ""
    if not isinstance(qid, str) or not ledger.is_open(qid):
        raise LedgerError(f"resolution names no open question: {qid!r}")
    if state not in (RESOLVED, "still_open"):
        raise LedgerError(f"unknown state: {state!r}")
    if not isinstance(evidence, str):
        raise LedgerError("evidence reference is not a string")
    return qid, state, evidence
