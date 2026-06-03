from __future__ import annotations

from pathlib import Path

from ..skills import Skill


ANALYSIS_INTENTS = {
    "binary_analysis",
    "pdf_analysis",
    "text_document_analysis",
    "codebase_inspection",
}


def skill_requires_workspace_docs(skill: Skill | None) -> bool:
    if skill is None:
        return False
    content = skill.content
    return "AGENTS.md" in content and "REPORT.md" in content


def skill_requires_case_setup(skill: Skill | None) -> bool:
    if skill is None:
        return False
    lowered = skill.content.lower()
    return "case directory" in lowered or "case layout" in lowered


def extra_categories_for_skill(skill: Skill | None, intent: str) -> tuple[str, ...]:
    if skill is None or intent not in ANALYSIS_INTENTS:
        return ()
    if skill_requires_workspace_docs(skill) or skill_requires_case_setup(skill):
        return ("write",)
    return ()


def startup_prompt_for_skill(skill: Skill | None, intent: str, messages: list[dict[str, object]]) -> str | None:
    if skill is None or intent not in ANALYSIS_INTENTS:
        return None
    if _has_recent_skill_startup_prompt(messages):
        return None
    parts: list[str] = []
    if skill_requires_case_setup(skill):
        parts.append("Active skill startup is mandatory: create or reuse the case/work directory.")
    if skill_requires_workspace_docs(skill):
        parts.append("Create or read AGENTS.md and REPORT.md, then keep them updated.")
    if intent == "binary_analysis":
        parts.append("Then identify the real sample, hash it, identify its type/container, and inspect embedded members.")
    if not parts:
        return None
    parts.append("Follow the active skill workflow before deeper inspection unless the user asked to skip it.")
    return " ".join(parts)


def workspace_doc_bootstrap_actions(skill: Skill | None, workdir: Path) -> list[tuple[str, dict[str, object]]]:
    if not skill_requires_workspace_docs(skill):
        return []
    actions: list[tuple[str, dict[str, object]]] = []
    agents = workdir / "AGENTS.md"
    report = workdir / "REPORT.md"
    if not agents.exists():
        actions.append(
            (
                "write_file",
                {
                    "path": "AGENTS.md",
                    "content": (
                        "# AGENTS.md\n\n"
                        "## Current Summary\n\n- Initialized by orbit bootstrap.\n\n"
                        "## Sample Inventory\n\n- Pending triage.\n\n"
                        "## Next Actions\n\n- Identify the sample, hashes, type, and analysis path.\n"
                    ),
                },
            )
        )
    if not report.exists():
        actions.append(
            (
                "write_file",
                {
                    "path": "REPORT.md",
                    "content": (
                        "# REPORT.md\n\n"
                        "## Status\n\n- Initialized by orbit bootstrap.\n\n"
                        "## Findings\n\n- Pending triage.\n"
                    ),
                },
            )
        )
    return actions


def should_bootstrap_workspace_docs(skill: Skill | None, intent: str) -> bool:
    return intent in ANALYSIS_INTENTS and skill_requires_workspace_docs(skill)


def _has_recent_skill_startup_prompt(messages: list[dict[str, object]]) -> bool:
    marker = "Active skill startup is mandatory:"
    for message in reversed(messages):
        if message.get("role") == "user":
            return False
        if message.get("role") == "system" and marker in str(message.get("content", "")):
            return True
    return False
