from __future__ import annotations

from dataclasses import dataclass

from ..skills import Skill
from .intent_router import (
    INTENT_CLASS_AMBIGUOUS,
    INTENT_CLASS_BINARY_ANALYSIS,
    INTENT_CLASS_BINARY_OR_PDF_ANALYSIS,
    INTENT_CLASS_CHAT_GENERAL,
    INTENT_CLASS_CODEBASE_INSPECTION,
    INTENT_CLASS_FILE_EDITING,
    INTENT_CLASS_FILE_READING,
    INTENT_CLASS_KNOWLEDGE_QUESTION,
    INTENT_CLASS_MACHINE_INSPECTION,
    INTENT_CLASS_PDF_ANALYSIS,
    INTENT_CLASS_SHELL_TASK,
    INTENT_CLASS_URL_INSPECTION,
    INTENT_CLASS_WEB_LOOKUP,
    INTENT_CLASS_WORKSPACE_DISCOVERY,
    route_intent,
)
from .skill_hints import extra_categories_for_skill


TOOL_CATEGORY_FILESYSTEM = "filesystem"
TOOL_CATEGORY_WRITE = "write"
TOOL_CATEGORY_SHELL = "shell"
TOOL_CATEGORY_WEB = "web"

ALL_TOOL_CATEGORIES = (TOOL_CATEGORY_FILESYSTEM, TOOL_CATEGORY_WRITE, TOOL_CATEGORY_SHELL, TOOL_CATEGORY_WEB)


@dataclass(frozen=True)
class ToolRoute:
    intent: str
    intent_class: str
    categories: tuple[str, ...]
    reason: str


def route_tool_categories(user_input: str, *, skill: Skill | None = None) -> ToolRoute:
    intent_route = route_intent(user_input)
    categories = categories_for_intent_class(intent_route.intent_class)
    if categories and _looks_like_pattern_extraction_request(user_input):
        categories = _merge_categories(categories, (TOOL_CATEGORY_SHELL,))
    categories = _merge_categories(categories, extra_categories_for_skill(skill, intent_route.intent))
    return ToolRoute(intent=intent_route.intent, intent_class=intent_route.intent_class, categories=categories, reason=intent_route.reason)


def categories_for_intent_class(intent_class: str) -> tuple[str, ...]:
    if intent_class in {INTENT_CLASS_WEB_LOOKUP, INTENT_CLASS_URL_INSPECTION}:
        return (TOOL_CATEGORY_WEB,)
    if intent_class in {INTENT_CLASS_KNOWLEDGE_QUESTION, INTENT_CLASS_CHAT_GENERAL}:
        return ()
    if intent_class == INTENT_CLASS_FILE_EDITING:
        return (TOOL_CATEGORY_WRITE, TOOL_CATEGORY_FILESYSTEM)
    if intent_class == INTENT_CLASS_CODEBASE_INSPECTION:
        return (TOOL_CATEGORY_FILESYSTEM,)
    if intent_class in {INTENT_CLASS_WORKSPACE_DISCOVERY, INTENT_CLASS_FILE_READING}:
        return (TOOL_CATEGORY_FILESYSTEM,)
    if intent_class in {INTENT_CLASS_BINARY_ANALYSIS, INTENT_CLASS_PDF_ANALYSIS, INTENT_CLASS_BINARY_OR_PDF_ANALYSIS}:
        return (TOOL_CATEGORY_SHELL, TOOL_CATEGORY_FILESYSTEM)
    if intent_class in {INTENT_CLASS_MACHINE_INSPECTION, INTENT_CLASS_SHELL_TASK}:
        return (TOOL_CATEGORY_SHELL,)
    if intent_class == INTENT_CLASS_AMBIGUOUS:
        return ALL_TOOL_CATEGORIES
    return ALL_TOOL_CATEGORIES


def _merge_categories(base: tuple[str, ...], extra: tuple[str, ...]) -> tuple[str, ...]:
    if not extra:
        return base
    merged = list(base)
    for category in extra:
        if category not in merged:
            merged.append(category)
    return tuple(merged)


def _looks_like_pattern_extraction_request(user_input: str) -> bool:
    lowered = user_input.lower()
    pattern_hints = (
        "- [ ]",
        "todo",
        "fixme",
        "cve",
        "url",
        "urls",
        "ip",
        "hash",
        "error",
        "errors",
        "warning",
        "warnings",
        "tag",
        "tags",
        "open tasks",
        "task aperti",
    )
    action_hints = ("extract", "estrai", "find", "show", "return", "list", "cerca", "trova")
    return any(hint in lowered for hint in pattern_hints) and any(hint in lowered for hint in action_hints)
