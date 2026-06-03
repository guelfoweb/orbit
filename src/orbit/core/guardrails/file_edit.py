from __future__ import annotations

import json
import re
import shlex
from typing import Any, Callable

from ..text_utils import word_tokens


REPLACE_KEEP_HINTS = ("and keep", "and leave", "otherwise unchanged", "file otherwise unchanged", "lascia", "invariat")
APPEND_SECTION_TITLE_RE = re.compile(r"(?:section titled|sezione(?: intitolata)?|section)\s+(?P<title>[A-Za-z0-9 _-]+?)\s+(?:to|in|su)\s+", re.IGNORECASE)
APPEND_BULLET_RE = re.compile(r"(?:one bullet|un bullet|with bullet|con bullet)\s*:\s*(?P<bullet>.+)$", re.IGNORECASE)
WRITE_BULLETS_RE = re.compile(r"(?:one|two|three|1|2|3)\s+bullets?\s*:\s*(?P<body>.+)$", re.IGNORECASE)
APPEND_ANY_TITLE_RE = re.compile(r"(?:section titled|sezione(?: intitolata)?|section)\s+(?P<title>[A-Za-z0-9 _-]+?)(?:\s+with|\s+con|\s*$)", re.IGNORECASE)
REPLACE_WITH_RE = re.compile(r"\breplace\s+(?P<old>.+?)\s+with\s+(?P<new>.+)$", re.IGNORECASE)
REPLACE_WITH_IT_RE = re.compile(r"\bsostituisci\s+(?P<old>.+?)\s+con\s+(?P<new>.+)$", re.IGNORECASE)
PLACEHOLDER_WRITE_HINTS = (
    "<tool_response.content>",
    "<tool_response",
    "<readme_content>",
    "_content>",
    '"+',
    "' +",
    "tool_response.content",
)
POST_WRITE_FINALIZE_HINTS = (
    "open_file",
    "created successfully",
    "creato con successo",
    "visualizzarlo",
    "view it",
    "update it as needed",
    "here is the content",
    "here's the content",
    "the content of the file",
    "file content is",
    "i have written",
    "i wrote",
    "saved it to a file",
    "saved to a file",
)
WRITE_PATH_RE = (
    r'"path"\s*:\s*"(?P<path>[^"]+)"',
    r"'path'\s*:\s*'(?P<path>[^']+)'",
)
SECTION_TITLE_PATTERNS = (
    re.compile(r"(?:adding|add)\s+(?:a|an)?\s*(?P<title>[A-Za-z0-9 _-]+?)\s+section", re.IGNORECASE),
    re.compile(r"(?:sezione)\s+(?P<title>[A-Za-z0-9 _-]+)", re.IGNORECASE),
    re.compile(r"aggiungendo\s+(?:una|un)?\s*sezione\s+(?P<title>[A-Za-z0-9 _-]+)", re.IGNORECASE),
)
CREATE_ACTION_TOKENS = {"create", "make", "mkdir", "crea", "creare"}
DELETE_ACTION_TOKENS = {"delete", "remove", "rm", "rmdir", "cancella", "rimuovi", "elimina"}
DIRECTORY_TARGET_NOUNS = {"directory", "folder", "dir", "cartella"}
FILE_TARGET_NOUNS = {"file", "document", "documento"}
FOLLOWUP_EDIT_TOKENS = {"poi", "then", "after", "dopo", "append", "aggiungi", "update", "aggiorna", "replace", "sostituisci", "reopen", "riapri"}
SKIP_TARGET_TOKENS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "la",
    "il",
    "lo",
    "i",
    "gli",
    "le",
    "un",
    "una",
    "uno",
    "questa",
    "questo",
    "quella",
    "quello",
    "named",
    "called",
    "chiamata",
    "chiamato",
    "denominata",
    "denominato",
    "nome",
    "di",
    "poi",
    "then",
}


def apply_deterministic_file_edit(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    summary_hints: tuple[str, ...],
    run_guardrail_tool: Callable[..., dict[str, Any]],
    last_read_file_result: Callable[[list[dict[str, Any]], str], dict[str, Any] | None],
    summarize_text_content: Callable[[str], str | None],
    successful_read_results_in_current_turn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    file_edit_completion_message: Callable[[set[str]], str],
    extract_explicit_text_path: Callable[[str], str | None],
    extract_explicit_pdf_path: Callable[[str], str | None],
    text_path_re: re.Pattern[str],
    normalize_relative_path: Callable[[str], str],
) -> str | None:
    if route.intent != "file_edit":
        return None
    plan = _infer_direct_directory_create_edit(user_input, normalize_relative_path)
    if plan is None:
        plan = _infer_direct_delete_path_edit(user_input, extract_explicit_text_path, extract_explicit_pdf_path, normalize_relative_path)
    if plan is None:
        plan = _infer_summary_then_append_edit(
            user_input,
            summary_hints=summary_hints,
            text_path_re=text_path_re,
            normalize_relative_path=normalize_relative_path,
        )
    if plan is None:
        plan = _infer_direct_append_section_edit(user_input, extract_explicit_text_path)
    if plan is None:
        plan = _infer_direct_replace_edit(user_input, extract_explicit_text_path)
    if plan is None:
        plan = _infer_direct_write_file_edit(
            user_input,
            summary_hints=summary_hints,
            text_path_re=text_path_re,
            normalize_relative_path=normalize_relative_path,
        )
    if plan is None:
        plan = _infer_direct_create_empty_file_edit(
            user_input,
            extract_explicit_text_path=extract_explicit_text_path,
            extract_explicit_pdf_path=extract_explicit_pdf_path,
            normalize_relative_path=normalize_relative_path,
        )
    if plan is None:
        return None
    path = plan["path"]
    if plan["name"] == "make_directory":
        create_result = run_guardrail_tool(
            name="make_directory",
            arguments={"path": path},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=True,
        )
        if not create_result.get("ok"):
            return None
        return f"Created directory `{path}`."
    if plan["name"] == "delete_path":
        delete_arguments = {"path": path, "recursive": bool(plan.get("recursive", False))}
        delete_result = run_guardrail_tool(
            name="delete_path",
            arguments=delete_arguments,
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=True,
        )
        if not delete_result.get("ok"):
            return None
        return f"Removed `{path}`."
    if plan["name"] == "write_then_append":
        read_result = run_guardrail_tool(
            name="read_file",
            arguments={"path": plan["read_path"]},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=True,
        )
        if not read_result.get("ok"):
            return None
        source_result = last_read_file_result(messages, plan["read_path"])
        if source_result is None:
            return None
        source_content = source_result.get("content")
        if not isinstance(source_content, str) or not source_content:
            return None
        summary = summarize_text_content(source_content)
        if summary is None:
            return None
        write_result = run_guardrail_tool(
            name="write_file",
            arguments={"path": path, "content": summary + "\n"},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=False,
        )
        if not write_result.get("ok"):
            return None
        append_result = run_guardrail_tool(
            name="append_file",
            arguments={"path": path, "content": plan["append_content"]},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=False,
        )
        if not append_result.get("ok"):
            return None
        return file_edit_completion_message({path})
    read_path = plan.get("read_path", path)
    read_arguments = {"path": read_path}
    should_read_first = plan["name"] != "write_file" or read_path != path or "content" not in plan
    if should_read_first and not any(item.get("path") == read_path for item in successful_read_results_in_current_turn(messages)):
        read_result = run_guardrail_tool(
            name="read_file",
            arguments=read_arguments,
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=True,
        )
        if not read_result.get("ok"):
            error = read_result.get("error")
            if not (
                plan["name"] == "append_file"
                and isinstance(error, str)
                and "file not found" in error.lower()
            ):
                return None
    if plan["name"] == "write_file" and "content" not in plan:
        source_result = last_read_file_result(messages, read_path)
        if source_result is None:
            return None
        source_content = source_result.get("content")
        if not isinstance(source_content, str) or not source_content:
            return None
        summary = summarize_text_content(source_content)
        if summary is None:
            return None
        plan = dict(plan)
        plan["content"] = summary + "\n"
    if plan["name"] == "append_file":
        write_arguments = {"path": path, "content": plan["content"]}
    elif plan["name"] == "write_file":
        write_arguments = {"path": path, "content": plan["content"]}
    else:
        write_arguments = {
            "path": path,
            "old": plan["old"],
            "new": plan["new"],
        }
    append_result = run_guardrail_tool(
        name=plan["name"],
        arguments=write_arguments,
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        emit_route=False,
    )
    if not append_result.get("ok"):
        return None
    if plan["name"] == "replace_in_file":
        return f"Updated `{path}`."
    if plan["name"] == "write_file":
        return f"Created `{path}`."
    return file_edit_completion_message({path})


def file_edit_placeholder_handling(
    *,
    intent: str | None,
    content: str,
    messages: list[dict[str, Any]],
    policy_state: Any,
    successful_write_results_in_current_turn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    successful_read_results_in_current_turn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    file_edit_completion_message: Callable[[set[str]], str],
) -> tuple[str, str] | None:
    if intent != "file_edit":
        return None
    lowered = content.lower()
    if not any(hint in lowered for hint in PLACEHOLDER_WRITE_HINTS):
        return None
    write_results = successful_write_results_in_current_turn(messages)
    if write_results:
        paths = {
            str(item.get("path")).strip()
            for item in write_results
            if isinstance(item.get("path"), str) and str(item.get("path")).strip()
        }
        if paths:
            return ("final", file_edit_completion_message(paths))
    if not successful_read_results_in_current_turn(messages):
        return None
    if policy_state.synthesis_retries >= 1:
        return (
            "final",
            "The model kept proposing a file edit with placeholders instead of real content derived from the previous read result. "
            "Retry with a more specific edit request or ask for a smaller extracted section first.",
        )
    policy_state.synthesis_retries += 1
    return (
        "retry",
        "Do not emit placeholders, pseudo-code, string concatenation, or tokens like <tool_response.content> inside write_file, append_file, or replace_in_file. "
        "Materialize the actual text from the previous read_file result and pass it as the real content argument.",
    )


def file_edit_post_write_reply_handling(
    *,
    intent: str | None,
    content: str,
    messages: list[dict[str, Any]],
    successful_write_results_in_current_turn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    file_edit_completion_message: Callable[[set[str]], str],
) -> tuple[str, str] | None:
    lowered = content.lower()
    if not any(hint in lowered for hint in POST_WRITE_FINALIZE_HINTS):
        return None
    latest_write = _latest_successful_write_record(messages)
    if latest_write is None:
        return None
    tool_name, path = latest_write
    if not path:
        return None
    return ("final", _file_edit_confirmation_message(tool_name, path))


def _latest_successful_write_record(messages: list[dict[str, Any]]) -> tuple[str, str] | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            break
        if message.get("role") != "tool":
            continue
        tool_name = message.get("tool_name")
        if tool_name not in {"write_file", "append_file", "replace_in_file", "make_directory", "delete_path"}:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            continue
        path = payload.get("path")
        if isinstance(path, str) and path.strip():
            return str(tool_name), path.strip()
    return None


def _file_edit_confirmation_message(tool_name: str, path: str) -> str:
    if tool_name == "write_file":
        return f"Created `{path}`."
    if tool_name == "append_file" or tool_name == "replace_in_file":
        return f"Updated `{path}`."
    if tool_name == "make_directory":
        return f"Created directory `{path}`."
    if tool_name == "delete_path":
        return f"Removed `{path}`."
    return file_edit_completion_message({path})


def placeholder_write_replacement_text(
    messages: list[dict[str, Any]],
    content: str,
    *,
    successful_read_results_in_current_turn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    latest_successful_read_result_in_current_turn: Callable[[list[dict[str, Any]]], dict[str, Any] | None],
    normalize_relative_path: Callable[[str], str],
) -> str | None:
    read_results = successful_read_results_in_current_turn(messages)
    if not read_results:
        return None
    target_path = _extract_write_target_path(content)
    if target_path is not None:
        normalized_target = normalize_relative_path(target_path)
        for item in reversed(read_results):
            path = item.get("path")
            body = item.get("content")
            if not isinstance(path, str) or not isinstance(body, str) or not body:
                continue
            if normalize_relative_path(path) != normalized_target:
                return body
    latest = latest_successful_read_result_in_current_turn(messages)
    if latest is None:
        return None
    latest_content = latest.get("content")
    if not isinstance(latest_content, str) or not latest_content:
        return None
    return latest_content


def infer_file_edit_section_append(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
    successful_write_results_in_current_turn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    successful_read_results_in_current_turn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    normalize_relative_path: Callable[[str], str],
) -> dict[str, Any] | None:
    if intent != "file_edit":
        return None
    if "section" not in user_input.lower() and "sezione" not in user_input.lower():
        return None
    if successful_write_results_in_current_turn(messages):
        return None
    read_results = successful_read_results_in_current_turn(messages)
    if len(read_results) < 2:
        return None
    target = read_results[-1]
    source = None
    target_path = target.get("path")
    if not isinstance(target_path, str) or not target_path.strip():
        return None
    normalized_target = normalize_relative_path(target_path)
    for item in reversed(read_results[:-1]):
        path = item.get("path")
        content = item.get("content")
        if not isinstance(path, str) or not isinstance(content, str) or not content.strip():
            continue
        if normalize_relative_path(path) != normalized_target:
            source = item
            break
    if source is None:
        return None
    source_content = source.get("content")
    if not isinstance(source_content, str) or not source_content.strip():
        return None
    title = _extract_section_title(user_input)
    if title is None:
        return None
    append_content = f"\n\n## {title}\n\n{source_content.strip()}"
    return {"name": "append_file", "arguments": {"path": target_path, "content": append_content}}


def extract_all_text_paths(
    user_input: str,
    *,
    text_path_re: re.Pattern[str],
    normalize_relative_path: Callable[[str], str],
) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for match in text_path_re.finditer(user_input):
        normalized = normalize_relative_path(match.group("path"))
        if normalized in seen:
            continue
        seen.add(normalized)
        paths.append(normalized)
    return paths


def _infer_direct_append_section_edit(
    user_input: str,
    extract_explicit_text_path: Callable[[str], str | None],
) -> dict[str, str] | None:
    lowered = user_input.lower()
    if "append" not in lowered and "aggiungi" not in lowered:
        return None
    path = extract_explicit_text_path(user_input)
    if path is None:
        return None
    title_match = APPEND_SECTION_TITLE_RE.search(user_input)
    bullet_match = APPEND_BULLET_RE.search(user_input)
    if title_match is None or bullet_match is None:
        return None
    title = title_match.group("title").strip().strip("`'\"")
    bullet = bullet_match.group("bullet").strip().strip("`")
    if not title or not bullet:
        return None
    return {"name": "append_file", "path": path, "content": f"\n\n## {title}\n\n- {bullet}\n"}


def _infer_summary_then_append_edit(
    user_input: str,
    *,
    summary_hints: tuple[str, ...],
    text_path_re: re.Pattern[str],
    normalize_relative_path: Callable[[str], str],
) -> dict[str, str] | None:
    lowered = user_input.lower()
    if not any(hint in lowered for hint in summary_hints):
        return None
    if not any(token in lowered for token in ("append", "aggiungi", "section", "sezione")):
        return None
    paths = extract_all_text_paths(user_input, text_path_re=text_path_re, normalize_relative_path=normalize_relative_path)
    if len(paths) < 2:
        return None
    source_path = paths[0]
    target_path = paths[-1]
    if source_path == target_path:
        return None
    title, bullet = _extract_append_section_parts(user_input)
    if not title or not bullet:
        return None
    return {
        "name": "write_then_append",
        "path": target_path,
        "read_path": source_path,
        "append_content": f"\n\n## {title}\n\n- {bullet}\n",
    }


def _infer_direct_directory_create_edit(
    user_input: str,
    normalize_relative_path: Callable[[str], str],
) -> dict[str, Any] | None:
    token_set = set(word_tokens(user_input.lower()))
    if not token_set & CREATE_ACTION_TOKENS:
        return None
    path = None
    if token_set & DIRECTORY_TARGET_NOUNS:
        path = _extract_target_from_action_noun_pattern(user_input, CREATE_ACTION_TOKENS, DIRECTORY_TARGET_NOUNS, normalize_relative_path)
    if path is None and "mkdir" in token_set:
        path = _extract_target_after_action(user_input, {"mkdir"}, normalize_relative_path)
    if path is None:
        return None
    return {"name": "make_directory", "path": path}


def _infer_direct_delete_path_edit(
    user_input: str,
    extract_explicit_text_path: Callable[[str], str | None],
    extract_explicit_pdf_path: Callable[[str], str | None],
    normalize_relative_path: Callable[[str], str],
) -> dict[str, Any] | None:
    token_set = set(word_tokens(user_input.lower()))
    if not token_set & DELETE_ACTION_TOKENS:
        return None
    if token_set & DIRECTORY_TARGET_NOUNS:
        path = _extract_target_from_action_noun_pattern(user_input, DELETE_ACTION_TOKENS, DIRECTORY_TARGET_NOUNS, normalize_relative_path)
        if path is None:
            return None
        return {"name": "delete_path", "path": path, "recursive": True}
    if token_set & FILE_TARGET_NOUNS:
        path = _extract_target_from_action_noun_pattern(user_input, DELETE_ACTION_TOKENS, FILE_TARGET_NOUNS, normalize_relative_path)
        if path is None:
            path = extract_explicit_text_path(user_input) or extract_explicit_pdf_path(user_input)
        if path is None:
            return None
        return {"name": "delete_path", "path": path, "recursive": False}
    path = extract_explicit_text_path(user_input) or extract_explicit_pdf_path(user_input)
    if path is None:
        action_tokens = token_set & {"rm", "rmdir", "delete", "remove", "cancella", "rimuovi", "elimina"}
        path = _extract_target_after_action(user_input, action_tokens, normalize_relative_path)
        if path is None:
            return None
    return {"name": "delete_path", "path": path, "recursive": False}


def _infer_direct_replace_edit(
    user_input: str,
    extract_explicit_text_path: Callable[[str], str | None],
) -> dict[str, str] | None:
    path = extract_explicit_text_path(user_input)
    if path is None:
        return None
    match = REPLACE_WITH_RE.search(user_input) or REPLACE_WITH_IT_RE.search(user_input)
    if match is None:
        return None
    old = _clean_replace_fragment(match.group("old"))
    new = _clean_replace_fragment(match.group("new"))
    if not old or not new or old == new:
        return None
    lowered = user_input.lower()
    for hint in REPLACE_KEEP_HINTS:
        marker = lowered.find(hint)
        if marker >= 0:
            original_marker = len(user_input[:marker])
            new = _clean_replace_fragment(user_input[match.start("new"):original_marker])
            break
    if not old or not new or old == new:
        return None
    return {"name": "replace_in_file", "path": path, "old": old, "new": new}


def _infer_direct_write_file_edit(
    user_input: str,
    *,
    summary_hints: tuple[str, ...],
    text_path_re: re.Pattern[str],
    normalize_relative_path: Callable[[str], str],
) -> dict[str, str] | None:
    lowered = user_input.lower()
    if not any(hint in lowered for hint in ("create", "write", "crea", "scrivi")):
        return None
    paths = extract_all_text_paths(user_input, text_path_re=text_path_re, normalize_relative_path=normalize_relative_path)
    if not paths:
        return None
    bullet_match = WRITE_BULLETS_RE.search(user_input)
    if bullet_match is not None:
        items = [item.strip().strip(".") for item in bullet_match.group("body").split(";")]
        bullets = [item for item in items if item]
        if bullets:
            content = "".join(f"- {item}\n" for item in bullets)
            return {"name": "write_file", "path": paths[-1], "content": content}
    if any(hint in lowered for hint in summary_hints) and len(paths) >= 2:
        source_path = paths[0]
        target_path = paths[-1]
        if source_path != target_path:
            return {"name": "write_file", "path": target_path, "read_path": source_path}
    return None


def _infer_direct_create_empty_file_edit(
    user_input: str,
    *,
    extract_explicit_text_path: Callable[[str], str | None],
    extract_explicit_pdf_path: Callable[[str], str | None],
    normalize_relative_path: Callable[[str], str],
) -> dict[str, str] | None:
    token_set = set(word_tokens(user_input.lower()))
    if not (token_set & CREATE_ACTION_TOKENS and token_set & FILE_TARGET_NOUNS):
        return None
    if token_set & FOLLOWUP_EDIT_TOKENS:
        return None
    path = _extract_target_from_action_noun_pattern(user_input, CREATE_ACTION_TOKENS, FILE_TARGET_NOUNS, normalize_relative_path)
    if path is None:
        path = extract_explicit_text_path(user_input) or extract_explicit_pdf_path(user_input)
    if path is None:
        return None
    return {"name": "write_file", "path": path, "content": ""}


def _clean_replace_fragment(value: str) -> str:
    cleaned = value.strip().strip(".")
    cleaned = cleaned.strip("`")
    cleaned = cleaned.strip()
    for quote in ("'", '"'):
        if cleaned.startswith(quote) and cleaned.endswith(quote) and len(cleaned) >= 2:
            cleaned = cleaned[1:-1].strip()
    return cleaned


def _extract_append_section_parts(user_input: str) -> tuple[str | None, str | None]:
    bullet_match = APPEND_BULLET_RE.search(user_input)
    if bullet_match is None:
        return None, None
    bullet = bullet_match.group("bullet").strip().strip("`")
    if not bullet:
        return None, None
    title_match = APPEND_SECTION_TITLE_RE.search(user_input) or APPEND_ANY_TITLE_RE.search(user_input)
    title: str | None = None
    if title_match is not None:
        title = title_match.group("title").strip().strip("`'\"")
    lowered = user_input.lower()
    if not title:
        if "next steps" in lowered:
            title = "Next Steps"
        elif "sezione finale" in lowered or "section finale" in lowered:
            title = "Finale"
        elif "final section" in lowered:
            title = "Final Section"
    elif title.islower():
        title = " ".join(part.capitalize() for part in title.split())
    return title, bullet


def _extract_target_from_action_noun_pattern(
    user_input: str,
    action_tokens: set[str],
    noun_tokens: set[str],
    normalize_relative_path: Callable[[str], str],
    *,
    max_gap: int = 4,
) -> str | None:
    raw_text = user_input.replace("`", '"')
    try:
        tokens = shlex.split(raw_text)
    except ValueError:
        tokens = raw_text.split()
    lowered_tokens = [token.lower().strip(".,:;!?") for token in tokens]
    for action_index, lowered_action in enumerate(lowered_tokens):
        if lowered_action not in action_tokens:
            continue
        noun_index = None
        upper_bound = min(len(tokens), action_index + 1 + max_gap)
        for index in range(action_index + 1, upper_bound):
            if lowered_tokens[index] in noun_tokens:
                noun_index = index
                break
        if noun_index is None:
            continue
        for candidate in tokens[noun_index + 1 :]:
            normalized = _normalize_target_token(candidate, normalize_relative_path)
            if normalized is None:
                continue
            if normalized.lower() in noun_tokens or normalized.lower() in SKIP_TARGET_TOKENS:
                continue
            return normalized
    return None


def _extract_target_after_action(
    user_input: str,
    action_tokens: set[str],
    normalize_relative_path: Callable[[str], str],
) -> str | None:
    if not action_tokens:
        return None
    raw_text = user_input.replace("`", '"')
    try:
        tokens = shlex.split(raw_text)
    except ValueError:
        tokens = raw_text.split()
    lowered_tokens = [token.lower().strip(".,:;!?") for token in tokens]
    for index, lowered_token in enumerate(lowered_tokens):
        if lowered_token not in action_tokens:
            continue
        for candidate in tokens[index + 1 :]:
            normalized = _normalize_target_token(candidate, normalize_relative_path)
            if normalized is None:
                continue
            if normalized.lower() in SKIP_TARGET_TOKENS | DIRECTORY_TARGET_NOUNS | FILE_TARGET_NOUNS:
                continue
            return normalized
    return None


def _normalize_target_token(value: str, normalize_relative_path: Callable[[str], str]) -> str | None:
    cleaned = value.strip().strip("`'\"").strip(".,:;!?")
    if not cleaned:
        return None
    if cleaned in {".", "..", "/", "\\"}:
        return None
    return normalize_relative_path(cleaned)


def _extract_write_target_path(content: str) -> str | None:
    for pattern in WRITE_PATH_RE:
        match = re.search(pattern, content)
        if match is not None:
            path = match.group("path").strip()
            if path:
                return path
    return None


def _extract_section_title(user_input: str) -> str | None:
    for pattern in SECTION_TITLE_PATTERNS:
        match = pattern.search(user_input)
        if match is not None:
            title = " ".join(match.group("title").strip().split())
            if title:
                return title
    return None
