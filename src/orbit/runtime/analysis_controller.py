"""Runtime-owned control state for an autonomous analysis.

The defect this replaces is architectural rather than behavioural. The previous
control plane carried its state in natural language: the model was asked to
begin a reply with `question: Q1` so the runtime could tell which question an
action belonged to. Live evidence killed it -- native tool calls often carry
little or no assistant prose, so the field had nowhere to travel, every action
was refused, and the run ended having done nothing. Prose is not a transport.

Here the model contributes judgement and the runtime owns bookkeeping, and the
split is deliberate about which is which:

    the model decides   which questions need a tool, which action to run,
                        what the evidence means, whether it answered
    the runtime decides question identity, which question is active, what an
                        action belongs to, when to stop

Identity is the clearest case. The model never names a question; Orbit assigns
`Q1`, `Q2`, `Q3` after validating the plan, so duplicate, missing and malformed
ids cannot exist -- three failure modes deleted rather than validated.

Association is the load-bearing one. Exactly one question is active at a time,
and an action issued while it is active belongs to it. Nothing is parsed out of
the reply and nothing is asked of the model: with one question open there is
only one thing an action could be for. An empty assistant message is fine.

Nothing here interprets an artifact. There is no file-type policy, no language
parser, no similarity measure, and no judgement about whether a finding
matters. The controller counts, validates shape, and refuses -- and when it
refuses it stops, because falling back to an unbounded loop is what produced
the context exhaustion this exists to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Question states. `OPEN` is the only one that can be made active.
OPEN = "open"
RESOLVED = "resolved"
BLOCKED = "blocked"

# Controller phases, in the order they occur.
PHASE_PLAN = "plan"
PHASE_RESOLVE = "resolve"
PHASE_REPORT = "report"

# How deep a causally-raised question may go. One: a result may raise a
# question, and that question's result may not raise another. The live
# evidence shows no causal chains at all, so anything deeper would be built
# for a case that has never occurred.
MAX_CHILD_DEPTH = 1

# What one question may spend before it is declared blocked. Two attempts is
# what the observed runs needed -- a first action, and one correction after
# seeing its output. A question that cannot be settled in two has not failed
# for want of another try, and letting it keep going is how one question
# consumes a whole run.
MAX_ACTIONS_PER_QUESTION = 2

# Bounds on model-authored text entering the ledger. The question is rendered
# into later prompts, so its size compounds; the missing-fact appears once.
# Sized from what a live model actually wrote: questions of 101-163 characters
# and rationales of 295-357.
MAX_QUESTION_CHARS = 300
MAX_MISSING_FACT_CHARS = 512
MAX_SUMMARY_CHARS = 1000
# The most questions a plan may declare, and the most the ledger may ever hold.
# Six leaves the global model-call ceiling room to work every one of them; the
# total is twice that, which is exactly one child per declared question.
MAX_PLAN_QUESTIONS = 6
MAX_TOTAL_QUESTIONS = MAX_PLAN_QUESTIONS * 2


class ControlError(ValueError):
    """A control message that cannot be trusted. Always fails closed."""


@dataclass(frozen=True)
class Question:
    """One thing the model said it cannot answer without running something."""

    id: str
    question: str
    missing_fact: str
    depth: int = 0
    parent: str | None = None
    caused_by: str | None = None


@dataclass
class QuestionState:
    """What became of a question, recorded as the model reported it."""

    status: str = OPEN
    evidence_ids: "tuple[str, ...]" = ()
    summary: str = ""
    reason: str = ""
    actions: int = 0


@dataclass
class AnalysisController:
    """The finite state of one autonomous analysis.

    Deliberately a plain record with explicit transitions. It refuses; it never
    decides on the model's behalf, and in particular it never marks a question
    resolved -- only an explicit control call from the model does that.
    """

    questions: "dict[str, Question]" = field(default_factory=dict)
    states: "dict[str, QuestionState]" = field(default_factory=dict)
    order: "list[str]" = field(default_factory=list)
    active: str | None = None
    phase: str = PHASE_PLAN
    # Counters the analyst sees. Each names a refusal that really happened,
    # so a run that went wrong can be explained rather than guessed at.
    rejected_children: int = 0
    repairs: int = 0
    unsupported: bool = False

    # -- identity ---------------------------------------------------------
    def _next_id(self, parent: str | None = None) -> str:
        """Q1, Q2, ... for plan questions; Q1.1 for a child of Q1.

        Assigned here and nowhere else. The model never supplies an id, so a
        duplicate, missing or malformed one is not a case to validate -- it is
        a case that cannot arise.
        """
        if parent is None:
            return f"Q{sum(1 for q in self.questions.values() if q.depth == 0) + 1}"
        existing = sum(1 for q in self.questions.values() if q.parent == parent)
        return f"{parent}.{existing + 1}"

    # -- plan -------------------------------------------------------------
    def adopt_plan(self, entries: "list[dict]") -> "list[Question]":
        """Validate a plan and take it, assigning ids. Raises on anything else.

        An empty plan is valid and meaningful: it says the source answered
        everything, and the run goes straight to the report.
        """
        if self.phase != PHASE_PLAN:
            raise ControlError("a plan has already been adopted")
        if not isinstance(entries, list):
            raise ControlError("questions is not a list")
        if len(entries) > MAX_PLAN_QUESTIONS:
            raise ControlError(
                f"plan declares {len(entries)} questions, more than "
                f"{MAX_PLAN_QUESTIONS}"
            )
        adopted: list[Question] = []
        seen: set[str] = set()
        for entry in entries:
            text, missing = _question_fields(entry)
            key = " ".join(text.lower().split())
            if key in seen:
                raise ControlError("plan repeats a question")
            seen.add(key)
            adopted.append(
                Question(id=self._next_id(), question=text, missing_fact=missing)
            )
            self.questions[adopted[-1].id] = adopted[-1]
            self.states[adopted[-1].id] = QuestionState()
            self.order.append(adopted[-1].id)
        self.phase = PHASE_RESOLVE if adopted else PHASE_REPORT
        return adopted

    # -- activation -------------------------------------------------------
    @property
    def open_ids(self) -> "list[str]":
        return [qid for qid in self.order if self.states[qid].status == OPEN]

    @property
    def exhausted(self) -> bool:
        """Nothing is open. Never a claim that the analysis is complete."""
        return not self.open_ids

    def activate_next(self) -> "Question | None":
        """Make the first open question active, or None when none is left.

        Selection is the runtime's, in declaration order. A model that could
        choose would be choosing what to work on next, which is planning, and
        planning already happened.
        """
        for qid in self.order:
            if self.states[qid].status == OPEN:
                self.active = qid
                return self.questions[qid]
        self.active = None
        self.phase = PHASE_REPORT
        return None

    def may_act(self) -> bool:
        """Whether the active question has budget for another action."""
        if self.active is None:
            return False
        return self.states[self.active].actions < MAX_ACTIONS_PER_QUESTION

    def record_action(self) -> None:
        """Count an execution against the active question.

        The whole association mechanism: an action is issued while exactly one
        question is active, so it belongs to that question. Nothing is parsed
        and nothing is asked of the model.
        """
        if self.active is None:
            raise ControlError("no active question to associate an action with")
        self.states[self.active].actions += 1

    def exhaust_active(self, reason: str) -> None:
        """Block the active question because it ran out of attempts."""
        if self.active is None:
            return
        state = self.states[self.active]
        if state.status == OPEN:
            state.status = BLOCKED
            state.reason = reason

    # -- completion -------------------------------------------------------
    def close_active(
        self,
        status: str,
        *,
        evidence_ids: "tuple[str, ...]" = (),
        summary: str = "",
        reason: str = "",
    ) -> None:
        """Record what the model said became of the active question."""
        if self.active is None:
            raise ControlError("no active question to close")
        if status not in (RESOLVED, OPEN, BLOCKED):
            raise ControlError(f"unknown status: {status!r}")
        state = self.states[self.active]
        state.evidence_ids = tuple(evidence_ids)
        state.summary = summary[:MAX_SUMMARY_CHARS]
        if status == RESOLVED:
            state.status = RESOLVED
        elif status == BLOCKED:
            state.status = BLOCKED
            state.reason = reason or "the model reported it blocked"
        # OPEN leaves it open: the model said the action did not settle it, and
        # `may_act` decides whether another attempt is available.

    # -- children ---------------------------------------------------------
    def accept_child(
        self, question: str, missing_fact: str, caused_by: str, known_evidence
    ) -> Question:
        """Take a question the current result forced, or raise.

        Every condition rules out a different way a run grows without bound:
        there must be an active parent, the causing evidence must exist and
        re-attest, the depth cap must hold, the ledger must have room, and it
        must not restate something already asked.
        """
        if self.active is None:
            self.rejected_children += 1
            raise ControlError("no active question to attach a child to")
        parent = self.questions[self.active]
        if parent.depth + 1 > MAX_CHILD_DEPTH:
            self.rejected_children += 1
            raise ControlError(
                f"a child of {parent.id} would exceed depth {MAX_CHILD_DEPTH}"
            )
        if len(self.questions) >= MAX_TOTAL_QUESTIONS:
            self.rejected_children += 1
            raise ControlError(f"already holding {MAX_TOTAL_QUESTIONS} questions")
        if not caused_by or caused_by not in known_evidence:
            self.rejected_children += 1
            raise ControlError("child names no evidence that re-attests")
        text, missing = _question_fields(
            {"question": question, "missing_fact": missing_fact}
        )
        key = " ".join(text.lower().split())
        if any(
            " ".join(q.question.lower().split()) == key for q in self.questions.values()
        ):
            self.rejected_children += 1
            raise ControlError("child restates an existing question")
        child = Question(
            id=self._next_id(parent.id),
            question=text,
            missing_fact=missing,
            depth=parent.depth + 1,
            parent=parent.id,
            caused_by=caused_by,
        )
        self.questions[child.id] = child
        self.states[child.id] = QuestionState()
        self.order.append(child.id)
        return child

    # -- rendering --------------------------------------------------------
    def dossier(self) -> str:
        """Every question and what became of it, for the closing report.

        Blocked and still-open questions are rendered as plainly as resolved
        ones. A control plane that could quietly drop what it failed to answer
        would be worse than the exploration it replaced.
        """
        if not self.order:
            return ""
        lines = ["Questions raised during this analysis:"]
        for qid in self.order:
            question = self.questions[qid]
            state = self.states[qid]
            # One row per question, always. The text is model-authored and the
            # model reads the artifact, so a newline in it could otherwise
            # render a second row -- a forged `Q9 [RESOLVED]` line beneath the
            # real ones. The authoritative id list is generated from state and
            # cannot be forged, but a confusable row is still worth not
            # printing.
            head = f"{qid} [{state.status.upper()}]: {_one_line(question.question)}"
            lines.append(head)
            if state.summary:
                lines.append(f"    {_one_line(state.summary)}")
            if state.evidence_ids:
                lines.append(f"    evidence: {', '.join(state.evidence_ids)}")
            if state.status != RESOLVED:
                lines.append(
                    "    unresolved: "
                    f"{_one_line(state.reason) or 'no answer was established'}"
                )
        return "\n".join(lines)

    def counts(self) -> "dict[str, int]":
        return {
            "questions": len(self.questions),
            "resolved": sum(1 for s in self.states.values() if s.status == RESOLVED),
            "blocked": sum(1 for s in self.states.values() if s.status == BLOCKED),
            "open": len(self.open_ids),
            "actions": sum(s.actions for s in self.states.values()),
        }


def _one_line(text: str) -> str:
    """Collapse whitespace so one field cannot render as several rows."""
    return " ".join(text.split())


def _question_fields(entry: object) -> "tuple[str, str]":
    """The two bounded strings a question is made of, or raise."""
    if not isinstance(entry, dict):
        raise ControlError("question entry is not an object")
    unexpected = set(entry) - {"question", "missing_fact"}
    if unexpected:
        raise ControlError(f"unexpected fields: {sorted(unexpected)}")
    text = entry.get("question")
    missing = entry.get("missing_fact")
    if not isinstance(text, str) or not text.strip():
        raise ControlError("question text is missing or empty")
    if not isinstance(missing, str) or not missing.strip():
        raise ControlError("missing_fact is missing or empty")
    if len(text) > MAX_QUESTION_CHARS:
        raise ControlError(f"question is longer than {MAX_QUESTION_CHARS} characters")
    if len(missing) > MAX_MISSING_FACT_CHARS:
        raise ControlError(
            f"missing_fact is longer than {MAX_MISSING_FACT_CHARS} characters"
        )
    return text.strip(), missing.strip()


def parse_plan_call(arguments: object) -> "list[dict]":
    """The `questions` list from a submit_analysis_plan call, or raise."""
    if not isinstance(arguments, dict):
        raise ControlError("plan arguments are not an object")
    unexpected = set(arguments) - {"questions"}
    if unexpected:
        raise ControlError(f"unexpected fields: {sorted(unexpected)}")
    questions = arguments.get("questions")
    if questions is None:
        raise ControlError("plan has no questions field")
    if not isinstance(questions, list):
        raise ControlError("questions is not a list")
    return questions


def parse_finish_call(arguments: object) -> "dict":
    """A finish_analysis_question call, validated into plain fields."""
    if not isinstance(arguments, dict):
        raise ControlError("completion arguments are not an object")
    unexpected = set(arguments) - {
        "status", "evidence_ids", "answer_summary", "child_question"
    }
    if unexpected:
        raise ControlError(f"unexpected fields: {sorted(unexpected)}")
    status = arguments.get("status")
    if status not in (RESOLVED, "still_open", BLOCKED):
        raise ControlError(f"unknown status: {status!r}")
    raw_ids = arguments.get("evidence_ids") or []
    if not isinstance(raw_ids, list) or not all(
        isinstance(value, str) for value in raw_ids
    ):
        raise ControlError("evidence_ids is not a list of strings")
    summary = arguments.get("answer_summary") or ""
    if not isinstance(summary, str):
        raise ControlError("answer_summary is not a string")
    child = arguments.get("child_question")
    if child is not None and not isinstance(child, dict):
        raise ControlError("child_question is not an object")
    return {
        "status": OPEN if status == "still_open" else status,
        "evidence_ids": tuple(raw_ids[:16]),
        "answer_summary": summary[:MAX_SUMMARY_CHARS],
        "child_question": child,
    }
