from __future__ import annotations

from dataclasses import dataclass, field

from orbit.runtime.tool_calls import tool_call_signature


@dataclass
class ToolLoopState:
    allowed_tool_names: tuple[str, ...]
    chunk_budget: dict[str, int] = field(default_factory=lambda: {"read_file_chunks": 0, "fetch_url_chunks": 0})
    seen_tool_calls: set[tuple[str, str]] = field(default_factory=set)
    tool_rounds: int = 0
    used_tool_call_prompt: bool = False

    @property
    def round_limit(self) -> int:
        edit_tools = {"write_file", "edit_file", "apply_diff", "make_directory", "delete_path"}
        return 2 if edit_tools.intersection(self.allowed_tool_names) else 1

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
