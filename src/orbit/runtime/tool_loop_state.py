from __future__ import annotations

from dataclasses import dataclass, field

from orbit.runtime.tool_calls import tool_call_signature


SHELL_FULL_ROUND_LIMIT = 8
DEFAULT_TOOL_ROUND_LIMIT = 1

EVIDENCE_UNRESOLVED = "unresolved"
EVIDENCE_METADATA_ONLY = "metadata_only"
EVIDENCE_DIRECT_READ_FAILED = "direct_read_failed"
EVIDENCE_CANDIDATE_PATHS_FOUND = "candidate_paths_found"
EVIDENCE_DIRECT_CONTENT_READ = "direct_content_read"
EVIDENCE_FINALIZABLE = "finalizable"
EVIDENCE_EXHAUSTED = "exhausted"

RECONSIDER_CONTENT_EVIDENCE = "content_evidence"
RECONSIDER_ANALYSIS_COMPLETION = "analysis_completion"
RECONSIDER_FILE_RECOVERY = "file_recovery"
RECONSIDER_URL_RECOVERY = "url_recovery"
RECONSIDER_COMPLETION = "completion"
RECONSIDER_MINIMAL_PATCH = "minimal_patch"


@dataclass
class ToolRepairState:
    """Bounded internal retries for a single tool turn.

    This state is local to one tool-driven user turn. It tracks only runtime
    repair/verification requests and must not carry semantic task decisions.
    """

    contract_retry_pending: bool = False
    read_only_mutation_retry_pending: bool = False
    shell_empty_result_check_pending: bool = False
    shell_empty_result_check_used: bool = False
    shell_error_final_pending: bool = False
    shell_repair_prompt_pending: str | None = None
    shell_repair_retries: int = 0
    mutation_verification_pending: bool = False
    mutation_verification_repair_pending: bool = False
    mutation_verification_repair_used: bool = False
    mutation_semantic_repair_pending: bool = False
    mutation_semantic_repair_used: bool = False
    file_content_retry_used: bool = False
    url_content_retry_used: bool = False

    def has_pending(self) -> bool:
        return (
            self.contract_retry_pending
            or self.read_only_mutation_retry_pending
            or self.shell_empty_result_check_pending
            or self.shell_repair_prompt_pending is not None
            or self.mutation_verification_pending
            or self.mutation_verification_repair_pending
            or self.mutation_semantic_repair_pending
        )


@dataclass
class ToolTurnState:
    """Semantic state for one tool turn, not for the whole session.

    Invariants:
    - no UI/REPL state
    - no global/session memory
    - no semantic override of the model's next step
    - at most one reconsider per reconsider kind
    """

    requested_user_path: str | None = None
    requested_user_url: str | None = None
    round_count: int = 0
    evidence_state: str = EVIDENCE_UNRESOLVED
    candidate_paths: list[str] = field(default_factory=list)
    direct_read_failed: bool = False
    last_error: str | None = None
    tool_result_kind: str | None = None
    reconsider_counts: dict[str, int] = field(default_factory=dict)
    repair_state: ToolRepairState = field(default_factory=ToolRepairState)
    content_evidence_satisfied: bool = False
    finalizable: bool = False
    pending_content_evidence_guard: bool = False
    pending_analysis_completion_guard: bool = False
    pending_file_recovery_guard: bool = False
    pending_file_recovery_guard_prompt: str | None = None
    pending_url_recovery_guard: bool = False
    pending_url_recovery_guard_prompt: str | None = None
    pending_completion_guard: bool = False
    pending_minimal_patch_guard: bool = False
    metadata_only_rejections: int = 0
    shell_commands_seen: int = 0
    shell_mutation_attempted: bool = False
    shell_mutation_succeeded: bool = False
    url_fetch_attempted: bool = False
    url_content_evidence_satisfied: bool = False
    url_failure_evidence_satisfied: bool = False

    def increment_round(self) -> None:
        self.round_count += 1

    def can_reconsider(self, kind: str) -> bool:
        return self.reconsider_counts.get(kind, 0) < 1

    def mark_reconsider(self, kind: str) -> None:
        self.reconsider_counts[kind] = self.reconsider_counts.get(kind, 0) + 1

    def has_pending_internal_request(self) -> bool:
        return (
            self.repair_state.has_pending()
            or self.pending_content_evidence_guard
            or self.pending_analysis_completion_guard
            or self.pending_file_recovery_guard
            or self.pending_url_recovery_guard
            or self.pending_completion_guard
            or self.pending_minimal_patch_guard
        )

    def set_tool_result_kind(self, kind: str) -> None:
        self.tool_result_kind = kind
        if kind == "metadata_listing" and self.evidence_state == EVIDENCE_UNRESOLVED:
            self.evidence_state = EVIDENCE_METADATA_ONLY

    def mark_direct_read_failed(self, error: str | None) -> None:
        self.direct_read_failed = True
        self.last_error = error
        self.evidence_state = EVIDENCE_DIRECT_READ_FAILED
        self.finalizable = False

    def mark_candidate_paths_found(self, paths: list[str]) -> None:
        for path in paths:
            if path not in self.candidate_paths:
                self.candidate_paths.append(path)
        if self.direct_read_failed and not self.content_evidence_satisfied and self.candidate_paths:
            self.evidence_state = EVIDENCE_CANDIDATE_PATHS_FOUND

    def mark_direct_content_read(self) -> None:
        self.content_evidence_satisfied = True
        self.finalizable = True
        self.evidence_state = EVIDENCE_DIRECT_CONTENT_READ

    def mark_url_content_evidence(self) -> None:
        self.url_content_evidence_satisfied = True
        self.finalizable = True

    def mark_url_failure_evidence(self, error: str | None) -> None:
        self.url_fetch_attempted = True
        self.url_failure_evidence_satisfied = True
        self.last_error = error
        self.finalizable = True

    def mark_finalizable(self) -> None:
        self.finalizable = True
        if self.content_evidence_satisfied:
            self.evidence_state = EVIDENCE_FINALIZABLE

    def mark_exhausted(self) -> None:
        self.finalizable = True
        self.evidence_state = EVIDENCE_EXHAUSTED

    def clear_pending_after_model_call(self) -> None:
        self.repair_state.shell_repair_prompt_pending = None
        self.repair_state.read_only_mutation_retry_pending = False
        self.repair_state.shell_empty_result_check_pending = False
        self.repair_state.mutation_verification_pending = False
        self.repair_state.mutation_verification_repair_pending = False
        self.repair_state.mutation_semantic_repair_pending = False
        self.pending_content_evidence_guard = False
        self.pending_analysis_completion_guard = False
        self.pending_file_recovery_guard = False
        self.pending_file_recovery_guard_prompt = None
        self.pending_url_recovery_guard = False
        self.pending_url_recovery_guard_prompt = None
        self.pending_completion_guard = False
        self.pending_minimal_patch_guard = False


@dataclass
class ToolLoopState:
    """Loop-global mechanics for the current request only.

    This object owns round limits, seen tool-call signatures, and chunk budgets.
    It intentionally does not track file evidence or semantic recovery state.
    """

    allowed_tool_names: tuple[str, ...]
    chunk_budget: dict[str, int] = field(default_factory=dict)
    seen_tool_calls: set[tuple[str, str]] = field(default_factory=set)
    tool_rounds: int = 0
    used_tool_call_prompt: bool = False

    @property
    def round_limit(self) -> int:
        if "exec_shell_full_command" in self.allowed_tool_names:
            return SHELL_FULL_ROUND_LIMIT
        return DEFAULT_TOOL_ROUND_LIMIT

    def increment_round(self) -> None:
        self.tool_rounds += 1

    def round_limit_reached(self) -> bool:
        return self.tool_rounds >= self.round_limit

    def mark_tool_call(self, tool_call: dict[str, object]) -> tuple[str, str]:
        signature = tool_call_signature(tool_call)
        self.seen_tool_calls.add(signature)
        return signature

    def has_seen_tool_call(self, tool_call: dict[str, object]) -> bool:
        return tool_call_signature(tool_call) in self.seen_tool_calls

    def tool_call_signature(self, tool_call: dict[str, object]) -> tuple[str, str]:
        return tool_call_signature(tool_call)
