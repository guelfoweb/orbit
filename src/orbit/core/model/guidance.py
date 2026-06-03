from __future__ import annotations

from ..guardrails.review_signals import (
    CODE_REVIEW_REQUEST_HINTS,
    has_code_file_extension,
    has_code_language_hint,
)

MODEL_FIRST_INTENT_GUIDANCE = {
    "chat_general": "Intent: chat_general. Prefer a direct answer.",
    "knowledge_question": "Intent: knowledge_question. Prefer a direct answer unless fresh evidence is needed.",
    "workspace_discovery": "Intent: workspace_discovery. Prefer list_files and answer in one line only as a short top-level comma-separated list.",
    "file_reading": "Intent: file_reading. Prefer read_file for contents and stat_path for size, mtime, permissions, or existence.",
    "file_editing": "Intent: file_editing. Prefer the smallest matching file edit tool.",
    "machine_inspection": "Intent: machine_inspection. Prefer one or two small bash calls, then answer from results only. Use df for filesystem free space at the requested path or mount point, including /. Use du only for directory size.",
    "shell_task": "Intent: shell_task. Prefer one bounded bash call at a time.",
    "web_lookup": "Intent: web_lookup. Prefer search_web, then synthesize briefly from results.",
    "url_inspection": "Intent: url_inspection. Prefer fetch_url for the explicit URL, then read it chunk by chunk if needed. If the user asks about a specific entity, answer that point first.",
    "codebase_inspection": "Intent: codebase_inspection. Prefer list_files first, then read only relevant files. If the user asked for a fixed number of files, stop after you have that many relevant files. Answer in one line only unless the user explicitly asks for a multi-part breakdown.",
    "binary_analysis": "Intent: binary_analysis. Prefer bounded static inspection of a real discovered path.",
    "pdf_analysis": "Intent: pdf_analysis. Prefer bounded text extraction from the explicit PDF path.",
}

MODEL_FIRST_POST_TOOL_GUIDANCE = {
    "chat_general": "Reply briefly.",
    "knowledge_question": "Reply briefly and use tool evidence only if needed.",
    "workspace_discovery": "Answer briefly from list_files/stat_path evidence. Use stat_path for newest, size, mtime, permissions, or existence.",
    "file_reading": "Answer only from the file evidence. Keep it short.",
    "file_editing": "Confirm only the actual change. Do not repeat file contents, code blocks, or long summaries.",
    "machine_inspection": "Report only the requested system facts.",
    "shell_task": "Answer only from the shell output. Keep it short.",
    "web_lookup": "Synthesize only the key result evidence. Keep it short.",
    "url_inspection": "Summarize the fetched chunk only. If the user asked about a specific entity, answer that point first. If more text is needed, fetch the same URL again with start_char=next_start_char.",
    "codebase_inspection": "Answer only the requested inspection task. Keep it short and preferably in one line.",
    "binary_analysis": "Report only the relevant static-analysis findings.",
    "pdf_analysis": "Answer only from extracted PDF evidence. Keep it short.",
}


def model_first_post_tool_prompt(intent_class: str | None, latest_user: str) -> str | None:
    prompt = MODEL_FIRST_POST_TOOL_GUIDANCE.get(intent_class)
    if prompt is None:
        return None
    if intent_class == "workspace_discovery":
        if _asks_for_listing_style_answer(latest_user):
            return (
                "Return up to 8 top-level names only, comma-separated. "
                "No bullets, headings, or interpretation."
            )
        return (
            "Answer briefly from the workspace listing only. "
            "Do not add interpretation or broad follow-up questions."
        )
    if intent_class == "file_reading":
        if _asks_for_one_line_answer(latest_user):
            return (
                "You have file content or metadata. Answer in one short line only from that evidence. "
                "Do not quote the whole file and do not add extra explanation."
            )
        return (
            "You have file content or metadata. Answer briefly from that evidence only. "
            "Include each requested fact if the evidence contains it. Do not quote long passages unless the user explicitly asked for full content."
        )
    if intent_class == "machine_inspection":
        return (
            "You have machine-inspection results. Answer only the requested system facts briefly. "
            "Do not add broad explanation or follow-up questions."
        )
    if intent_class == "codebase_inspection":
        if _asks_for_explicit_file_bug_review(latest_user):
            return (
                "You already have one concrete source file. Review that file directly and report up to 3 concrete bug or risk findings visible in the file itself. "
                "Do not stop at generic uncertainty about surrounding modules. If no clear bug is visible in this file, say that explicitly and give at most 1 precise remaining uncertainty."
            )
        return (
            "You have codebase evidence. Answer only from that evidence in at most 3 short bullets or file paths. "
            "Do not restate the whole listing or quote full file contents."
        )
    return prompt


def _asks_for_one_line_answer(user_input: str) -> bool:
    lowered = user_input.lower()
    return any(hint in lowered for hint in ("in one line", "one short line", "in una riga", "una riga"))


def _asks_for_listing_style_answer(user_input: str) -> bool:
    lowered = user_input.lower()
    listing_hints = ("cosa contiene", "what does", "contains", "contain", "contents", "content", "quali file", "which files")
    return any(hint in lowered for hint in listing_hints)


def _asks_for_explicit_file_bug_review(user_input: str) -> bool:
    lowered = user_input.lower()
    if not any(hint in lowered for hint in CODE_REVIEW_REQUEST_HINTS):
        return False
    if not (has_code_file_extension(lowered) or "/" in lowered or "\\" in lowered):
        return False
    return has_code_file_extension(lowered) or has_code_language_hint(lowered) or "this file" in lowered or "questo file" in lowered
