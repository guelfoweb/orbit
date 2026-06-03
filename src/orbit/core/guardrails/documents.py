from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
import re
import shlex
import time

from ..events import ToolCallEvent, ToolResultEvent, ToolRouteEvent
from ..intent.router import INTENT_CURRENT_FACTUAL_LOOKUP, INTENT_PDF_ANALYSIS, INTENT_TEXT_DOCUMENT_ANALYSIS, is_static_file_analysis_intent
from ..messages import (
    last_read_file_result,
    merged_read_file_result_in_current_turn,
    normalize_relative_path,
    successful_bash_results_in_current_turn,
    successful_read_results_in_current_turn,
)


SHOW_CONTENT_HINTS = ("show", "mostra", "content", "contenuto", "display")
READ_TEXT_HINTS = ("read", "open", "leggi", "apri")
SUMMARY_HINTS = ("summarize", "summary", "riassumi", "riassunto", "in one line", "in una riga")
ABOUT_DOCUMENT_HINTS = (
    "what this document is about",
    "what is this document about",
    "what the document is about",
    "what is about this document",
    "what is about this doc",
    "what is about this documet",
    "what is this doc about",
    "what is it about",
    "di cosa parla",
    "cosa contiene",
)
SINGLE_LINE_SUMMARY_HINTS = ("in one line", "one short line", "in una riga", "una riga")
PDF_READ_HINTS = ("read", "leggi", "leggimi", "open", "apri", "file", "documento", "document")
PDF_PAGE_COUNT_HINTS = ("page", "pages", "pagina", "pagine")
PDF_PAGE_COUNT_QUERY_HINTS = ("how many", "page count", "page number", "page numbers", "number of pages", "quante pagine", "numero di pagine", "numeri pagina", "numeri di pagina")
TEXT_PATH_RE = re.compile(r"(?P<path>[A-Za-z0-9_./-]+\.(?:md|txt|json|toml|yaml|yml|py|rst))")
QUOTED_PDF_PATH_RE = re.compile(r"(?P<quote>[\"'`])(?P<path>[^\"'`]+\.pdf)(?P=quote)", re.IGNORECASE)
SUMMARY_LINE_COUNT_RE = re.compile(r"\b(?P<count>\d{1,2})\s+(?:short\s+)?(?:lines?|righe?)\b", re.IGNORECASE)
SUMMARY_SENTENCE_COUNT_RE = re.compile(
    r"\b(?P<count>\d{1,2}|one|two|three|four|five|una|uno|due|tre|quattro|cinque)\s+(?:short\s+)?(?:sentences?|frasi?)\b",
    re.IGNORECASE,
)

SUMMARY_READ_MAX_CHUNKS = 4
SUMMARY_READ_LONG_MAX_CHUNKS = 8
SUMMARY_READ_PREFIX_MAX_CHUNKS = 4
SUMMARY_READ_MAX_LINES = 120
SUMMARY_READ_MAX_CHARS = 6000
PDF_TEXT_HEAD_LINES = 240
PDF_TEXT_MAX_CHARS = 12000
SUMMARY_EVIDENCE_TEXT_LIMIT = 1200
SUMMARY_PREFIX_EXCERPT_LIMIT = 6000
TERM_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'-]{2,}")
TERM_STOPWORDS = {
    "about",
    "after",
    "alla",
    "alle",
    "also",
    "anche",
    "ancora",
    "anni",
    "article",
    "avere",
    "away",
    "bene",
    "come",
    "con",
    "contro",
    "cosa",
    "così",
    "dalla",
    "dalle",
    "dallo",
    "della",
    "delle",
    "dello",
    "dentro",
    "detto",
    "disse",
    "dopo",
    "dove",
    "from",
    "into",
    "italia",
    "italy",
    "mentre",
    "nella",
    "nelle",
    "nello",
    "opera",
    "prima",
    "quel",
    "perchè",
    "questo",
    "quella",
    "quello",
    "sotto",
    "sono",
    "storia",
    "tanto",
    "sulla",
    "sulle",
    "their",
    "there",
    "these",
    "they",
    "this",
    "through",
    "where",
}
NARRATIVE_MARKERS = (
    "renzo",
    "lucia",
    "rodrigo",
    "federigo",
    "agnese",
    "abbondio",
    "cristoforo",
    "cerc",
    "rispose",
    "rimase",
    "andò",
    "ando",
    "arriv",
    "fugg",
    "spos",
    "peste",
    "tries",
    "faces",
    "meets",
    "returns",
    "discovers",
    "learns",
)


def seed_explicit_text_read_impl(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    run_guardrail_tool: Callable[..., dict[str, Any]],
) -> None:
    if route.intent not in {INTENT_TEXT_DOCUMENT_ANALYSIS, INTENT_CURRENT_FACTUAL_LOOKUP}:
        return
    path = extract_explicit_text_path(user_input)
    if path is None:
        return
    lowered = user_input.lower()
    hint_text = lowered.replace(path.lower(), " ")
    wants_show = any(hint in hint_text for hint in SHOW_CONTENT_HINTS) or (
        route.intent == INTENT_CURRENT_FACTUAL_LOOKUP and any(hint in hint_text for hint in READ_TEXT_HINTS)
    )
    wants_summary = any(hint in hint_text for hint in SUMMARY_HINTS)
    if not wants_show and not wants_summary:
        return
    if wants_summary:
        seed_document_summary_reads(
            path=path,
            user_input=user_input,
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            run_guardrail_tool=run_guardrail_tool,
        )
        return
    if wants_show:
        if any(item.get("path") == path for item in successful_read_results_in_current_turn(messages)):
            return
        result = run_guardrail_tool(
            name="read_file",
            arguments={"path": path},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=True,
        )
        if route.intent == INTENT_CURRENT_FACTUAL_LOOKUP and result.get("ok"):
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"`{path}` has already been read locally. In the final answer, separate concrete facts from this local file "
                        "from web-search evidence. Do not describe local evidence generically if the file content is available."
                    ),
                }
            )
        return


def seed_explicit_pdf_read_impl(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    run_guardrail_tool: Callable[..., dict[str, Any]],
) -> None:
    if route.intent != INTENT_PDF_ANALYSIS and not is_static_file_analysis_intent(route.intent):
        return
    path = extract_explicit_pdf_path(user_input)
    if path is None:
        return
    lowered = user_input.lower()
    if not _wants_pdf_text_or_metadata(lowered):
        return
    if latest_pdf_text_extract_result(messages, path) is not None:
        return
    pypdf_result = extract_pdf_text_with_pypdf(
        path=path,
        workdir=getattr(registry, "workdir", None),
        max_lines=PDF_TEXT_HEAD_LINES,
        max_chars=PDF_TEXT_MAX_CHARS,
    )
    if pypdf_result is not None:
        append_internal_pdf_extract_result(
            result=pypdf_result,
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
        )
        if result_has_useful_text(pypdf_result):
            return
        if _asks_for_pdf_page_count(lowered) and isinstance(pypdf_result.get("pages"), int):
            return
    pdftotext_result = run_guardrail_tool(
        name="bash",
        arguments={"command": f"pdftotext {shlex.quote(path)} - | head -n {PDF_TEXT_HEAD_LINES}"},
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        emit_route=True,
    )
    if result_has_useful_text(pdftotext_result):
        return
    if latest_pdf_strings_result(messages, path) is not None:
        return
    run_guardrail_tool(
        name="bash",
        arguments={"command": f"strings {shlex.quote(path)} | head -n {PDF_TEXT_HEAD_LINES}"},
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        emit_route=False,
    )


def local_explicit_text_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if intent != INTENT_TEXT_DOCUMENT_ANALYSIS:
        return None
    path = extract_explicit_text_path(user_input)
    if path is None:
        return None
    result = merged_read_file_result_in_current_turn(messages, path) or last_read_file_result(messages, path)
    if result is None:
        return None
    content = result.get("content")
    if not isinstance(content, str) or not content:
        return None
    lowered = user_input.lower()
    if any(hint in lowered for hint in SUMMARY_HINTS):
        single_line = any(hint in lowered for hint in SINGLE_LINE_SUMMARY_HINTS)
        chunks = _read_chunks_for_path(messages, path)
        sampled_read = _is_sampled_summary_read(messages, path)
        if len(chunks) > 1:
            summary = summarize_chunked_text_results(
                chunks,
                single_line=single_line,
                max_lines=extract_requested_summary_lines(user_input),
                sentence_count=extract_requested_summary_sentences(user_input),
            )
        else:
            summary = summarize_text_content(
                content,
                single_line=single_line,
                max_lines=extract_requested_summary_lines(user_input),
                sentence_count=extract_requested_summary_sentences(user_input),
            )
        if summary is not None:
            if result.get("truncated") or result.get("has_more") or sampled_read:
                if single_line:
                    return summary + " [based on retrieved portions of a longer file]"
                return summary + "\n\n[based on retrieved portions of a longer file]"
            return summary
    if any(hint in lowered for hint in SHOW_CONTENT_HINTS):
        if result.get("truncated") or result.get("has_more"):
            return content + "\n\n[truncated: ask for a smaller range if needed]"
        return content
    return None


def condense_explicit_text_summary_messages(
    *,
    user_input: str,
    messages: list[dict[str, Any]],
    summary_text: str | None = None,
) -> None:
    path = extract_explicit_text_path(user_input)
    if path is None:
        return
    lowered = user_input.lower()
    if not any(hint in lowered for hint in SUMMARY_HINTS):
        return
    last_user_index = max((index for index, message in enumerate(messages) if message.get("role") == "user"), default=-1)
    matching_indexes: list[int] = []
    sampled_start_lines: list[int] = []
    total_lines = None
    for index in range(last_user_index + 1, len(messages)):
        message = messages[index]
        if message.get("role") != "tool" or message.get("tool_name") != "read_file":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or normalize_relative_path(str(payload.get("path", ""))) != path:
            continue
        matching_indexes.append(index)
        start_line = payload.get("start_line")
        if isinstance(start_line, int):
            sampled_start_lines.append(start_line)
        if isinstance(payload.get("total_lines"), int):
            total_lines = payload.get("total_lines")
    if len(matching_indexes) <= 1:
        return
    chunks = _read_chunks_for_path(messages, path)
    chunk_notes = build_chunk_evidence_notes(chunks)
    if not summary_text:
        summary_text = build_chunk_evidence_summary(chunk_notes)
    prefix_focused = _prefers_prefix_summary(user_input=user_input, path=path)
    replacement = {
        "ok": True,
        "path": path,
        "summary_read": True,
        "prefix_focused": prefix_focused,
        "sampled_chunks": len(matching_indexes),
        "sampled_start_lines": sampled_start_lines,
        "total_lines": total_lines,
        "chunk_notes": chunk_notes,
        "document_map": build_document_map(chunks, total_lines=total_lines),
        "synthesis_guidance": (
            "Use text_excerpt first when present; otherwise use chunk_notes as sampled evidence. Prefer recurring entities, "
            "themes, conflicts, and progression over isolated quoted fragments."
        ),
        "content": summary_text[:SUMMARY_EVIDENCE_TEXT_LIMIT],
        "notice": "summary sample read",
    }
    if prefix_focused:
        excerpt = _prefix_text_excerpt(chunks)
        if excerpt:
            replacement["text_excerpt"] = excerpt[:SUMMARY_PREFIX_EXCERPT_LIMIT]
    first_index = matching_indexes[0]
    messages[first_index] = {
        "role": "tool",
        "tool_name": "read_file",
        "content": json.dumps(replacement, ensure_ascii=False),
    }
    for index in reversed(matching_indexes[1:]):
        del messages[index]


def should_defer_explicit_text_summary_to_model(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> bool:
    if intent != INTENT_TEXT_DOCUMENT_ANALYSIS:
        return False
    path = extract_explicit_text_path(user_input)
    if path is None:
        return False
    lowered = user_input.lower()
    if not any(hint in lowered for hint in SUMMARY_HINTS):
        return False
    return len(_read_chunks_for_path(messages, path)) > 1


def local_explicit_pdf_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if intent != INTENT_PDF_ANALYSIS and not is_static_file_analysis_intent(intent):
        return None
    path = extract_explicit_pdf_path(user_input)
    if path is None:
        return None
    result = latest_pdf_text_extract_result(messages, path)
    lowered = user_input.lower()
    if result is None and _asks_for_pdf_page_count(lowered):
        result = latest_pypdf_result(messages, path)
    if result is None:
        return None
    if _asks_for_pdf_page_count(lowered):
        pages = result.get("pages")
        if isinstance(pages, int) and pages >= 0:
            return f"`{path}` has {pages} page{'s' if pages != 1 else ''}."
    stdout = result.get("stdout")
    if not isinstance(stdout, str) or not stdout.strip():
        return None
    content = stdout.strip()
    source = pdf_extract_source(result, path)
    if any(hint in lowered for hint in SUMMARY_HINTS + ABOUT_DOCUMENT_HINTS):
        summary = summarize_text_content(content, single_line=any(hint in lowered for hint in SINGLE_LINE_SUMMARY_HINTS))
        if summary is not None:
            return summary
    if any(hint in lowered for hint in SHOW_CONTENT_HINTS + PDF_READ_HINTS):
        return content + f"\n\n[bounded PDF extract via {source}]"
    return None


def _wants_pdf_text_or_metadata(lowered: str) -> bool:
    return any(hint in lowered for hint in SHOW_CONTENT_HINTS + SUMMARY_HINTS + ABOUT_DOCUMENT_HINTS + PDF_READ_HINTS) or _asks_for_pdf_page_count(lowered)


def _asks_for_pdf_page_count(lowered: str) -> bool:
    if any(hint in lowered for hint in PDF_PAGE_COUNT_QUERY_HINTS):
        return True
    return any(hint in lowered for hint in PDF_PAGE_COUNT_HINTS) and any(hint in lowered for hint in ("how", "many", "count", "number", "quante", "quanti", "numero"))


def extract_explicit_text_path(user_input: str) -> str | None:
    match = TEXT_PATH_RE.search(user_input)
    if match is None:
        return None
    return normalize_relative_path(match.group("path"))


def extract_explicit_pdf_path(user_input: str) -> str | None:
    match = QUOTED_PDF_PATH_RE.search(user_input)
    if match is None:
        return extract_pdf_path_from_tokens(user_input)
    return normalize_relative_path(match.group("path"))


def seed_document_summary_reads(
    *,
    path: str,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    run_guardrail_tool: Callable[..., dict[str, Any]],
) -> None:
    lowered = user_input.lower()
    single_line = any(hint in lowered for hint in SINGLE_LINE_SUMMARY_HINTS)
    max_chunks = SUMMARY_READ_MAX_CHUNKS if single_line else SUMMARY_READ_LONG_MAX_CHUNKS
    prefix_focused = _prefers_prefix_summary(user_input=user_input, path=path)
    if prefix_focused:
        max_chunks = min(max_chunks, SUMMARY_READ_PREFIX_MAX_CHUNKS)
    existing = [
        item
        for item in successful_read_results_in_current_turn(messages)
        if normalize_relative_path(str(item.get("path", ""))) == path
    ]
    emit_route = True
    latest = existing[-1] if existing else None
    chunks_read = len(existing)
    if latest is None:
        latest = run_guardrail_tool(
            name="read_file",
            arguments={"path": path, "start_line": 1, "max_lines": SUMMARY_READ_MAX_LINES, "max_chars": SUMMARY_READ_MAX_CHARS},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=emit_route,
        )
        if not latest.get("ok"):
            return
        chunks_read += 1
        emit_route = False
    total_lines = latest.get("total_lines")
    if not isinstance(total_lines, int) or total_lines <= 0:
        total_lines = None
    planned_starts = _summary_read_start_lines(
        total_lines=total_lines,
        max_chunks=max_chunks,
        chunk_lines=SUMMARY_READ_MAX_LINES,
        prefix_focused=prefix_focused,
    )
    existing_starts = {int(item.get("start_line", 1)) for item in existing if isinstance(item.get("start_line"), int)}
    if isinstance(latest.get("start_line"), int):
        existing_starts.add(int(latest["start_line"]))
    for next_start_line in planned_starts:
        if chunks_read >= max_chunks or not latest.get("ok"):
            break
        if next_start_line in existing_starts or next_start_line <= 1:
            continue
        latest = run_guardrail_tool(
            name="read_file",
            arguments={"path": path, "start_line": next_start_line, "max_lines": SUMMARY_READ_MAX_LINES, "max_chars": SUMMARY_READ_MAX_CHARS},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=emit_route,
        )
        if not latest.get("ok"):
            break
        chunks_read += 1
        emit_route = False
        existing_starts.add(next_start_line)


def summarize_text_content(
    content: str,
    *,
    single_line: bool = False,
    max_lines: int | None = None,
    sentence_count: int | None = None,
) -> str | None:
    candidates = _paragraph_candidates(content)
    if not candidates:
        candidates = _line_candidates(content)
    if not candidates:
        return None
    if isinstance(sentence_count, int) and sentence_count > 0:
        sentence_candidates = _sentence_candidates(content) or candidates
        selected = sentence_candidates[: max(1, min(5, sentence_count))]
        return " ".join(_as_sentence(line) for line in selected)
    if single_line or len(candidates) == 1:
        return candidates[0]
    limit = max(2, min(12, max_lines if isinstance(max_lines, int) and max_lines > 0 else 3))
    selected = _spread_candidates(candidates, limit=limit)
    return "\n".join(f"- {line}" for line in selected)


def summarize_chunked_text_results(
    chunks: list[dict[str, Any]],
    *,
    single_line: bool = False,
    max_lines: int | None = None,
    sentence_count: int | None = None,
) -> str | None:
    chunk_candidates: list[list[str]] = []
    for chunk in chunks:
        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        candidates = _chunk_candidate_summaries(content)
        if not candidates:
            continue
        chunk_candidates.append(candidates)
    summaries = _merge_chunk_candidates(chunk_candidates)
    if not summaries:
        return None
    if isinstance(sentence_count, int) and sentence_count > 0:
        selected = _spread_candidates(summaries, limit=max(1, min(5, sentence_count)))
        return " ".join(_as_sentence(line) for line in selected)
    if single_line:
        return summaries[0]
    limit = max(2, min(12, max_lines if isinstance(max_lines, int) and max_lines > 0 else 3))
    selected = _spread_candidates(summaries, limit=limit)
    return "\n".join(f"- {line}" for line in selected)


def build_chunk_evidence_summary(chunk_notes: list[str]) -> str:
    if not chunk_notes:
        return ""
    visible = chunk_notes[:8]
    return "Sampled file evidence:\n" + "\n".join(f"- {note}" for note in visible)


def build_document_map(chunks: list[dict[str, Any]], *, total_lines: int | None) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for chunk in chunks[:12]:
        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        start_line = chunk.get("start_line")
        returned_lines = chunk.get("returned_lines")
        if not isinstance(start_line, int) or start_line <= 0:
            start_line = 1
        if not isinstance(returned_lines, int) or returned_lines <= 0:
            returned_lines = len(content.splitlines())
        end_line = start_line + max(0, returned_lines - 1)
        mapped.append(
            {
                "lines": f"{start_line}-{end_line}",
                "position": _document_position(start_line=start_line, total_lines=total_lines),
                "focus": _chunk_focus_summary(content) or "",
                "key_terms": _content_key_terms(content),
            }
        )
    return mapped


def extract_requested_summary_lines(user_input: str) -> int | None:
    match = SUMMARY_LINE_COUNT_RE.search(user_input)
    if match is None:
        return None
    count = int(match.group("count"))
    return max(1, min(12, count))


def extract_requested_summary_sentences(user_input: str) -> int | None:
    match = SUMMARY_SENTENCE_COUNT_RE.search(user_input)
    if match is None:
        return None
    raw = match.group("count").lower()
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "una": 1,
        "uno": 1,
        "due": 2,
        "tre": 3,
        "quattro": 4,
        "cinque": 5,
    }
    count = words.get(raw, int(raw) if raw.isdigit() else 3)
    return max(1, min(5, count))


def _spread_candidates(candidates: list[str], *, limit: int) -> list[str]:
    if len(candidates) <= limit:
        return candidates
    if limit <= 1:
        return [candidates[0]]
    out: list[str] = []
    last_index = len(candidates) - 1
    for slot in range(limit):
        index = round(slot * last_index / (limit - 1))
        candidate = candidates[index]
        if out and candidate == out[-1]:
            continue
        out.append(candidate)
    if len(out) < limit:
        for candidate in candidates:
            if candidate in out:
                continue
            out.append(candidate)
            if len(out) >= limit:
                break
    return out[:limit]


def _as_sentence(candidate: str) -> str:
    sentence = candidate.strip().lstrip("-* ").strip()
    if not sentence:
        return sentence
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _sentence_candidates(content: str) -> list[str]:
    normalized = re.sub(r"(?<=[.!?])(?=[A-ZÀ-Ö])", " ", content)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", normalized)
    out: list[str] = []
    for chunk in chunks:
        candidate = chunk.strip()
        if len(candidate) < 20:
            continue
        out.append(candidate)
    return out[:80]


def _paragraph_candidates(content: str) -> list[str]:
    if "\n\n" not in content:
        return []
    candidates: list[str] = []
    paragraph_lines: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            _flush_paragraph_candidate(paragraph_lines, candidates)
            paragraph_lines = []
            continue
        paragraph_lines.append(line)
    _flush_paragraph_candidate(paragraph_lines, candidates)
    return candidates[:120]


def _flush_paragraph_candidate(paragraph_lines: list[str], candidates: list[str]) -> None:
    if not paragraph_lines:
        return
    paragraph = " ".join(paragraph_lines)
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    if not paragraph:
        return
    if paragraph.startswith("#"):
        return
    if len(paragraph.split()) <= 6 and paragraph.upper() == paragraph:
        return
    if len(paragraph) > 220:
        paragraph = paragraph[:217].rstrip() + "..."
    if candidates and candidates[-1] == paragraph:
        return
    candidates.append(paragraph)


def _line_candidates(content: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[*-]\s*", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if len(line) > 180:
            line = line[:177].rstrip() + "..."
        if candidates and candidates[-1] == line:
            continue
        candidates.append(line)
        if len(candidates) >= 120:
            break
    return candidates


def build_chunk_evidence_notes(chunks: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    for chunk in chunks:
        note = _chunk_evidence_note(chunk)
        if not note or note in notes:
            continue
        notes.append(note)
        if len(notes) >= 12:
            break
    return notes


def _chunk_candidate_summaries(content: str) -> list[str]:
    raw_candidates = _paragraph_candidates(content) or _line_candidates(content)
    if not raw_candidates:
        return []
    good = [candidate for candidate in raw_candidates if not _is_low_information_candidate(candidate)]
    candidates = good or raw_candidates
    candidates = sorted(candidates, key=_candidate_summary_score, reverse=True)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate in deduped:
            continue
        deduped.append(candidate)
        if len(deduped) >= 3:
            break
    return deduped


def _is_low_information_candidate(candidate: str) -> bool:
    words = candidate.split()
    if len(words) <= 2 and len(candidate) <= 24:
        return True
    if len(words) <= 6 and candidate.upper() == candidate:
        return True
    lowered = candidate.lower()
    if lowered.startswith("[illustrazione:") or lowered.startswith("[illustration:"):
        return True
    if re.match(r"^\[\d+\]\s", candidate):
        return True
    if " pag. " in lowered or lowered.startswith("pag. "):
        return True
    if candidate.count("?") >= 2:
        return True
    return False


def _candidate_summary_score(candidate: str) -> tuple[int, int, int]:
    terms = _content_key_terms(candidate)
    tokens = candidate.split()
    capitalized = sum(1 for token in tokens if token[:1].isupper() and len(token.strip(".,;:!?\"'«»")) > 2)
    length = min(len(candidate), 160)
    penalty = 0
    lowered = candidate.lower()
    if candidate.startswith(("«", "\"", "'")):
        penalty -= 2
    if "[illustrazione:" in lowered or "[illustration:" in lowered:
        penalty -= 4
    if "?" in candidate:
        penalty -= 4
    if re.match(r"^\[\d+\]\s", candidate) or " pag. " in lowered:
        penalty -= 8
    if tokens and tokens[0][:1].islower():
        penalty -= 2
    narrative_score = _narrative_marker_score(lowered)
    if narrative_score:
        penalty += min(6, narrative_score * 2)
    return (len(terms) + capitalized + penalty, length, -candidate.count("..."))


def _narrative_marker_score(lowered: str) -> int:
    return sum(1 for marker in NARRATIVE_MARKERS if marker in lowered)


def _merge_chunk_candidates(chunk_candidates: list[list[str]]) -> list[str]:
    merged: list[str] = []
    depth = 0
    while True:
        added = False
        for candidates in chunk_candidates:
            if depth >= len(candidates):
                continue
            candidate = candidates[depth]
            if candidate in merged:
                continue
            merged.append(candidate)
            added = True
        if not added:
            break
        depth += 1
        if len(merged) >= 120:
            break
    return merged


def _chunk_evidence_note(chunk: dict[str, Any]) -> str | None:
    content = chunk.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    start_line = chunk.get("start_line")
    returned_lines = chunk.get("returned_lines")
    line_span = None
    if isinstance(start_line, int) and isinstance(returned_lines, int) and returned_lines > 0:
        line_span = f"lines {start_line}-{start_line + returned_lines - 1}"
    focus = _chunk_focus_summary(content)
    if not focus:
        return None
    terms = _content_key_terms(content)
    parts = []
    if line_span:
        parts.append(line_span)
    parts.append(f"focus: {focus}")
    if terms:
        parts.append("terms: " + ", ".join(terms))
    return "; ".join(parts)


def _document_position(*, start_line: int, total_lines: int | None) -> str:
    if not isinstance(total_lines, int) or total_lines <= 0:
        return "unknown"
    ratio = start_line / max(1, total_lines)
    if ratio <= 0.2:
        return "beginning"
    if ratio >= 0.8:
        return "ending"
    if ratio <= 0.45:
        return "early-middle"
    if ratio >= 0.6:
        return "late-middle"
    return "middle"


def _chunk_focus_summary(content: str) -> str | None:
    candidates = _chunk_candidate_summaries(content)
    if not candidates:
        return None
    candidate = candidates[0]
    candidate = candidate.strip("`\"' ")
    candidate = re.sub(r"\s+", " ", candidate).strip()
    if not candidate:
        return None
    if len(candidate) > 140:
        candidate = candidate[:137].rstrip() + "..."
    return candidate


def _content_key_terms(content: str) -> list[str]:
    counts: dict[str, tuple[int, str]] = {}
    order: list[str] = []
    for token in TERM_RE.findall(content):
        normalized = token.lower().strip("-'")
        if len(normalized) < 4 or normalized in TERM_STOPWORDS:
            continue
        if normalized not in counts:
            counts[normalized] = (0, token)
            order.append(normalized)
        count, display = counts[normalized]
        preferred = display
        if display.islower() and any(char.isupper() for char in token):
            preferred = token
        counts[normalized] = (count + 1, preferred)
    ranked = sorted(
        order,
        key=lambda item: (-counts[item][0], order.index(item)),
    )
    terms: list[str] = []
    for normalized in ranked:
        display = counts[normalized][1]
        if display in terms:
            continue
        terms.append(display)
        if len(terms) >= 4:
            break
    return terms


def _read_chunks_for_path(messages: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
    normalized_path = normalize_relative_path(path)
    chunks = [
        item
        for item in successful_read_results_in_current_turn(messages)
        if normalize_relative_path(str(item.get("path", ""))) == normalized_path
    ]
    chunks.sort(key=lambda item: int(item.get("start_line", 1)))
    return chunks


def _is_sampled_summary_read(messages: list[dict[str, Any]], path: str) -> bool:
    chunks = [
        item
        for item in successful_read_results_in_current_turn(messages)
        if normalize_relative_path(str(item.get("path", ""))) == path
    ]
    if len(chunks) <= 1:
        return False
    chunks.sort(key=lambda item: int(item.get("start_line", 1)))
    for left, right in zip(chunks, chunks[1:]):
        left_start = left.get("start_line")
        left_lines = left.get("returned_lines")
        right_start = right.get("start_line")
        if not isinstance(left_start, int) or not isinstance(left_lines, int) or not isinstance(right_start, int):
            continue
        expected_next = left_start + max(0, left_lines)
        if right_start != expected_next:
            return True
    return False


def _summary_read_start_lines(
    *,
    total_lines: int | None,
    max_chunks: int,
    chunk_lines: int,
    prefix_focused: bool = False,
) -> list[int]:
    if max_chunks <= 0:
        return []
    if total_lines is None or total_lines <= chunk_lines:
        return [1]
    if prefix_focused:
        return [1 + slot * chunk_lines for slot in range(max_chunks) if 1 + slot * chunk_lines <= total_lines]
    if total_lines <= chunk_lines * max_chunks:
        return [1 + slot * chunk_lines for slot in range(max_chunks) if 1 + slot * chunk_lines <= total_lines]
    last_start = max(1, total_lines - chunk_lines + 1)
    if max_chunks == 1:
        return [1]
    starts: list[int] = []
    span = max(0, total_lines - chunk_lines)
    for slot in range(max_chunks):
        if slot == 0:
            start_line = 1
        elif slot == max_chunks - 1:
            start_line = last_start
        else:
            start_line = 1 + round((slot * span) / (max_chunks - 1))
        if starts and start_line <= starts[-1]:
            continue
        starts.append(start_line)
    return starts


def _prefers_prefix_summary(*, user_input: str, path: str) -> bool:
    lowered = f"{user_input} {path}".lower()
    section_terms = (
        "canto",
        "capitolo",
        "chapter",
        "scene",
        "scena",
        "act ",
        "atto",
        "part ",
        "parte",
        "book ",
        "libro",
    )
    return any(term in lowered for term in section_terms)


def _prefix_text_excerpt(chunks: list[dict[str, Any]]) -> str:
    ordered = sorted(chunks, key=lambda item: int(item.get("start_line", 1)))
    parts: list[str] = []
    for chunk in ordered:
        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        start_line = chunk.get("start_line")
        label = f"[lines {start_line}]" if isinstance(start_line, int) else "[excerpt]"
        parts.append(f"{label}\n{content.strip()}")
        joined = "\n\n".join(parts)
        if len(joined) >= SUMMARY_PREFIX_EXCERPT_LIMIT:
            return joined[:SUMMARY_PREFIX_EXCERPT_LIMIT]
    return "\n\n".join(parts)


def latest_pdftotext_result(messages: list[dict[str, Any]], path: str) -> dict[str, Any] | None:
    normalized_path = normalize_relative_path(path)
    expected_prefixes = (
        f"pdftotext {normalized_path} -",
        f"pdftotext {shlex.quote(normalized_path)} -",
    )
    for result in reversed(successful_bash_results_in_current_turn(messages)):
        command = result.get("command")
        if not isinstance(command, str):
            continue
        if any(command.strip().startswith(prefix) for prefix in expected_prefixes):
            return result
    return None


def latest_pypdf_result(messages: list[dict[str, Any]], path: str) -> dict[str, Any] | None:
    normalized_path = normalize_relative_path(path)
    expected_prefix = f"pypdf_extract {shlex.quote(normalized_path)} "
    for result in reversed(successful_bash_results_in_current_turn(messages)):
        command = result.get("command")
        if not isinstance(command, str):
            continue
        if command.strip().startswith(expected_prefix):
            return result
    return None


def latest_pdf_strings_result(messages: list[dict[str, Any]], path: str) -> dict[str, Any] | None:
    normalized_path = normalize_relative_path(path)
    expected_prefixes = (
        f"strings {normalized_path} |",
        f"strings {shlex.quote(normalized_path)} |",
    )
    for result in reversed(successful_bash_results_in_current_turn(messages)):
        command = result.get("command")
        if not isinstance(command, str):
            continue
        if any(command.strip().startswith(prefix) for prefix in expected_prefixes):
            return result
    return None


def latest_pdf_text_extract_result(messages: list[dict[str, Any]], path: str) -> dict[str, Any] | None:
    result = latest_pypdf_result(messages, path)
    if result is not None and result_has_useful_text(result):
        return result
    result = latest_pdftotext_result(messages, path)
    if result is not None and result_has_useful_text(result):
        return result
    result = latest_pdf_strings_result(messages, path)
    if result is not None and result_has_useful_text(result):
        return result
    return None


def has_pdf_text_extract_in_current_turn(messages: list[dict[str, Any]]) -> bool:
    for result in successful_bash_results_in_current_turn(messages):
        if result_has_useful_text(result) and is_pdf_extract_command(str(result.get("command", ""))):
            return True
    return False


def result_has_useful_text(result: dict[str, Any]) -> bool:
    stdout = result.get("stdout")
    return isinstance(stdout, str) and bool(stdout.strip())


def is_pdf_extract_command(command: str) -> bool:
    stripped = command.strip()
    return stripped.startswith("pypdf_extract ") or stripped.startswith("pdftotext ") or stripped.startswith("strings ")


def is_strings_extract_result(result: dict[str, Any], path: str) -> bool:
    command = result.get("command")
    if not isinstance(command, str):
        return False
    normalized_path = normalize_relative_path(path)
    return command.strip().startswith(f"strings {normalized_path} |") or command.strip().startswith(
        f"strings {shlex.quote(normalized_path)} |"
    )


def pdf_extract_source(result: dict[str, Any], path: str) -> str:
    command = result.get("command")
    if isinstance(command, str) and command.strip().startswith("pypdf_extract "):
        return "pypdf"
    if is_strings_extract_result(result, path):
        return "strings"
    return "pdftotext"


def extract_pdf_text_with_pypdf(*, path: str, workdir: object, max_lines: int, max_chars: int) -> dict[str, Any] | None:
    if not isinstance(workdir, Path):
        return None
    try:
        from pypdf import PdfReader
    except Exception:
        return None
    normalized = normalize_relative_path(path)
    command = f"pypdf_extract {shlex.quote(normalized)} --max-lines {max_lines}"
    root = workdir.resolve()
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return {
            "ok": False,
            "command": command,
            "stdout": "",
            "stderr": f"path escapes workdir: {path}",
            "returncode": 1,
            "extractor": "pypdf",
        }
    if not candidate.exists() or not candidate.is_file():
        return {
            "ok": False,
            "command": command,
            "stdout": "",
            "stderr": f"file not found: {path}",
            "returncode": 1,
            "extractor": "pypdf",
        }
    try:
        reader = PdfReader(str(candidate))
        lines: list[str] = []
        chars = 0
        truncated = False
        for page in reader.pages:
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if len(lines) >= max_lines or chars + len(line) + 1 > max_chars:
                    truncated = True
                    break
                lines.append(line)
                chars += len(line) + 1
            if truncated:
                break
        stdout = "\n".join(lines)
        return {
            "ok": bool(stdout.strip()),
            "command": command,
            "stdout": stdout,
            "stderr": "" if stdout.strip() else "pypdf extracted no useful text",
            "returncode": 0 if stdout.strip() else 1,
            "extractor": "pypdf",
            "pages": len(reader.pages),
            "truncated": truncated,
        }
    except Exception as exc:
        return {
            "ok": False,
            "command": command,
            "stdout": "",
            "stderr": f"pypdf failed: {exc}",
            "returncode": 1,
            "extractor": "pypdf",
        }


def append_internal_pdf_extract_result(
    *,
    result: dict[str, Any],
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
) -> None:
    if on_event is not None:
        on_event(ToolRouteEvent(loop=0, intent=route.intent, categories=route.categories, reason=route.reason))
        on_event(ToolCallEvent(loop=0, name="bash", arguments={"command": str(result.get("command") or "pypdf_extract")}))
    started_at = time.monotonic_ns()
    elapsed_ns = time.monotonic_ns() - started_at
    if elapsed_ns > 0:
        metrics.tool_duration_ns += elapsed_ns
    policy_state.tool_steps += 1
    if on_event is not None:
        on_event(
            ToolResultEvent(
                loop=0,
                name="bash",
                ok=bool(result.get("ok")),
                error=result.get("error"),
                returncode=result.get("returncode"),
                stderr=result.get("stderr"),
                stdout=result.get("stdout"),
                elapsed_ms=elapsed_ns / 1_000_000,
            )
        )
    messages.append(
        {
            "role": "tool",
            "tool_name": "bash",
            "content": registry.encode_tool_result(result),
        }
    )


def extract_pdf_path_from_tokens(user_input: str) -> str | None:
    tokens = [token.strip("()[]{}<>.,:;!?") for token in user_input.split()]
    if not tokens:
        return None
    boundary_words = {
        "summarize", "summary", "show", "content", "read", "open",
        "riassumi", "riassunto", "mostra", "contenuto", "leggi", "apri",
        "the", "this", "that", "these", "those", "it",
        "il", "lo", "la", "i", "gli", "le", "un", "una", "questo", "questa", "quello", "quella",
        "file", "document", "documento",
        "of", "for", "to", "di", "del", "della", "dei", "delle", "su",
    }
    best: str | None = None
    for end_index, token in enumerate(tokens):
        if not token.lower().endswith(".pdf"):
            continue
        for width in range(1, min(4, end_index + 1) + 1):
            parts = tokens[end_index + 1 - width : end_index + 1]
            if any(not part for part in parts):
                continue
            first = parts[0].lower()
            candidate = " ".join(parts).strip()
            if "/" in candidate and "/" not in parts[0]:
                continue
            if "/" not in candidate and first in boundary_words:
                continue
            normalized = normalize_relative_path(candidate)
            if best is None or ("/" in normalized and "/" not in best) or len(normalized) > len(best):
                best = normalized
    return best
