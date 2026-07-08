from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass

from orbit.backend import ChatResult
from orbit.backend.base import Message
from orbit.runtime.completion_budget import resolve_max_tokens


FINAL_FROM_TOOL_MIN_TOKENS = 256
LARGE_FILE_FINAL_MAX_TOKENS = 128
EXHAUSTIVE_LARGE_FILE_FINAL_MAX_TOKENS = 256
WEB_FETCH_FINAL_MAX_TOKENS = 72
PDF_FINAL_MAX_TOKENS = 128
PDF_BRIEF_FINAL_MAX_TOKENS = 72
LIST_FINAL_MAX_TOKENS = 96
OPERATIONAL_STATUS_FINAL_MAX_TOKENS = 96
SYSTEM_INFO_FINAL_MAX_TOKENS = 160
LONG_SHELL_ANALYSIS_FINAL_MAX_TOKENS = 96
BRIEF_SHELL_FINAL_MAX_TOKENS = 96
COMPACT_FINAL_RETRY_MAX_TOKENS = 160
COMPACT_FINAL_RETRY_MIN_TOKENS = 64
FINAL_TOOL_CONTENT_COMPACT_THRESHOLD = 1200
FINAL_TOOL_CONTENT_HEAD_CHARS = 800
FINAL_TOOL_CONTENT_TAIL_CHARS = 300
FINAL_TOOL_PDF_BRIEF_CHARS = 900
FINAL_TOOL_PDF_MAX_CHARS = 1400
FINAL_TOOL_TRUNCATION_MARKER = "[output truncated for model context]"
_COMPACT_LIST_REQUEST_RE = re.compile(
    r"\b(?:only\s+(?:the\s+)?(?:filenames?|names?)|return\s+only|only\s+the\s+listed\s+names|solo\s+i\s+nomi|solo\s+i\s+file|solo\s+nomi)\b",
    re.IGNORECASE,
)
_EXHAUSTIVE_DOCUMENT_RE = re.compile(
    r"\b(?:entire|whole|full|complete|completo|completa|intero|intera|detailed|detail|detagliat\w*|approfond\w*|exhaustive|thorough|critique|critica|strengths|weaknesses|punti\s+forti|punti\s+deboli|cite|cita)\b",
    re.IGNORECASE,
)
_FINAL_MARKERS = (
    "**final answer:**",
    "final answer:",
    "the final answer is:",
    "the final answer:",
)
_REASONING_PREFIXES = (
    "### reasoning",
    "## reasoning",
    "# reasoning",
    "reasoning:",
    "plan:",
)
_REASONING_LEAK_META_PHRASES = (
    "the user likely meant",
    "the user likely means",
    "the user probably meant",
    "the user probably means",
    "the user may have meant",
    "the user may mean",
    "looking at the words",
    "wait, looking at the words",
    "the sentence should",
    "i should",
)


@dataclass(frozen=True)
class FinalToolPolicy:
    messages: list[Message]
    max_tokens: int
    length_retry_allowed: bool
    incomplete_retry_allowed: bool
    web_fetch_result: bool
    web_search_result: bool


@dataclass(frozen=True)
class FinalAnswerCompleteness:
    status: str
    detail: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"


def build_final_tool_policy(
    messages: list[Message],
    *,
    max_tokens: int,
    streamed: bool,
    evidence_kind: str | None = None,
    evidence_chars: int | None = None,
) -> FinalToolPolicy:
    prompt = last_user_text(messages)
    call_messages = prepare_final_tool_messages(messages)
    large_file_excerpt = has_large_file_excerpt(call_messages)
    exhaustive_document_request = is_exhaustive_document_request(prompt)
    web_fetch_result = has_html_cleaned_tool_result(call_messages)
    web_search_result = has_web_search_tool_result(call_messages)
    pdf_text_result = has_pdf_text_tool_result(call_messages)
    list_like_result = has_list_like_tool_result(call_messages) and is_compact_list_request(prompt)
    shell_full_result = has_tool_result(call_messages, "exec_shell_full_command")
    system_info_result = has_tool_result(call_messages, "system_info")
    operational_status_result = shell_full_result and is_operational_status_request(prompt)
    brief_request = is_brief_final_request(prompt)
    brief_shell_result = shell_full_result and brief_request and not (
        large_file_excerpt or web_fetch_result or pdf_text_result or list_like_result or operational_status_result
    )
    compact_shell_analysis_result = (
        shell_full_result
        and _has_long_shell_tool_result(messages)
        and is_shell_review_request(prompt)
        and not (large_file_excerpt or web_fetch_result or pdf_text_result or list_like_result or operational_status_result)
    )
    if large_file_excerpt:
        call_messages = [
            *call_messages,
            {
                "role": "user",
                "content": (
                    "Use only the already inspected chunk(s) or excerpt(s). "
                    + (
                        "Give a concise but fuller synthesis from the available evidence. "
                        "If the inspected chunks still do not cover the whole document, say that briefly. "
                        "Do not quote long passages. "
                        if exhaustive_document_request
                        else "Answer in at most five short bullets, each under twelve words. Do not quote long passages. "
                    )
                    + "Do not request more chunks unless the user explicitly asked for exhaustive analysis."
                ),
            },
        ]
    elif web_fetch_result:
        call_messages = [
            *call_messages,
            {
                "role": "user",
                "content": (
                    "Use only the fetched page text already available. "
                    "Write exactly two concise bullets. "
                    "Use the requested language; if unspecified, use the fetched page language. "
                    "Focus on the central thesis and key messages. "
                    "Each bullet must be under eighteen words. No introduction. Stop after the second bullet. "
                    "Do not request more chunks unless the user explicitly asked for exhaustive analysis."
                ),
            },
        ]
    elif operational_status_result:
        call_messages = [
            *call_messages,
            {
                "role": "user",
                "content": (
                    "Answer the latest operational/status question directly. "
                    "Use only the most recent relevant shell output. "
                    "Ignore older tool results or content analysis unless explicitly requested. "
                    "Do not summarize file or page content. "
                    "If recent evidence is insufficient, say you cannot confirm."
                ),
            },
        ]
    elif pdf_text_result:
        call_messages = [
            *call_messages,
            {
                "role": "user",
                "content": (
                    "A local PDF text extraction already succeeded. "
                    "Treat the PDF file as present and readable. "
                    "Base the answer only on the extracted PDF text already available. "
                    "Do not claim the file is missing or inaccessible unless a tool result explicitly says extraction failed. "
                    "Do not call tools again. "
                    + (
                        "Answer in exactly one concise sentence. No introduction. "
                        if brief_request
                        else ""
                    )
                    + "If the extracted text is only partial, say that briefly and summarize only that evidence."
                ),
            },
        ]
    elif list_like_result:
        call_messages = [
            *call_messages,
            {"role": "user", "content": "Return only the listed names, compactly. No categories or explanations."},
        ]
    elif shell_full_result:
        call_messages = [
            *call_messages,
            {
                "role": "user",
                "content": (
                    (
                        (
                            "Use only the latest relevant shell-full output. "
                            "Answer in at most two short sentences. "
                            "Focus only on the main confirmed issue and the shortest useful fix. "
                            "No headings, bullets, code fences, introduction, or thinking. "
                            "Do not call tools again. "
                            "If the evidence is insufficient, say you cannot confirm."
                            if brief_request
                            else
                            "Use only the latest relevant shell-full output. "
                            "Answer using exactly 4 short bullets. "
                            "Each bullet must be one sentence in this format: '- Finding: ... Fix: ...'. "
                            "No headings, code fences, examples, introduction, or thinking. "
                            "Focus only on the main findings and brief remediation. "
                            "Do not call tools again. "
                            "If the evidence is insufficient, say you cannot confirm."
                        )
                        if compact_shell_analysis_result
                        else
                        (
                            "Use only the available shell-full output. "
                            "Answer the latest user request in one concise sentence. "
                            "Prefer the most recent relevant shell result. "
                            "Do not summarize unrelated older output. "
                            "Do not call tools again. "
                            "If the evidence is insufficient, say you cannot confirm."
                            if brief_request
                            else
                            "Use only the available shell-full output. "
                            "Answer the latest user request directly and concisely from that evidence. "
                            "Prefer the most recent relevant shell result. "
                            "Do not summarize unrelated older output. "
                            "Do not call tools again. "
                            "If the evidence is insufficient, say you cannot confirm."
                        )
                    )
                ),
            },
        ]
    return FinalToolPolicy(
        messages=call_messages,
        max_tokens=final_tool_max_tokens(
            max_tokens,
            large_file_excerpt=large_file_excerpt,
            exhaustive_document_request=exhaustive_document_request,
            web_fetch_result=web_fetch_result,
            web_search_result=web_search_result,
            pdf_text_result=pdf_text_result,
            list_like_result=list_like_result,
            operational_status_result=operational_status_result,
            system_info_result=system_info_result,
            compact_shell_analysis_result=compact_shell_analysis_result,
            brief_shell_result=brief_shell_result,
            brief_request=brief_request,
            evidence_kind=evidence_kind,
            evidence_chars=evidence_chars,
        ),
        length_retry_allowed=(not streamed and (large_file_excerpt or web_fetch_result)),
        incomplete_retry_allowed=shell_full_result and not (web_fetch_result or web_search_result or list_like_result or operational_status_result),
        web_fetch_result=web_fetch_result,
        web_search_result=web_search_result,
    )


def final_tool_max_tokens(
    max_tokens: int,
    *,
    large_file_excerpt: bool,
    exhaustive_document_request: bool,
    web_fetch_result: bool,
    web_search_result: bool,
    pdf_text_result: bool,
    list_like_result: bool,
    operational_status_result: bool = False,
    system_info_result: bool = False,
    compact_shell_analysis_result: bool = False,
    brief_shell_result: bool = False,
    brief_request: bool = False,
    evidence_kind: str | None = None,
    evidence_chars: int | None = None,
) -> int:
    if large_file_excerpt:
        if exhaustive_document_request:
            return resolve_max_tokens("final_from_tool", max_tokens, evidence_kind="read", evidence_chars=evidence_chars)
        return min(
            resolve_max_tokens("final_from_tool", max_tokens, evidence_kind="read", evidence_chars=evidence_chars),
            LARGE_FILE_FINAL_MAX_TOKENS,
        )
    if web_fetch_result and not web_search_result:
        return min(max_tokens, WEB_FETCH_FINAL_MAX_TOKENS)
    if web_search_result:
        return resolve_max_tokens("final_from_tool", max_tokens, evidence_kind="web_search", evidence_chars=evidence_chars)
    if pdf_text_result:
        if brief_request:
            return min(max_tokens, PDF_BRIEF_FINAL_MAX_TOKENS)
        return min(
            resolve_max_tokens("final_from_tool", max_tokens, evidence_kind="read", evidence_chars=evidence_chars),
            PDF_FINAL_MAX_TOKENS,
        )
    if list_like_result:
        return min(max_tokens, LIST_FINAL_MAX_TOKENS)
    if system_info_result:
        return min(max_tokens, SYSTEM_INFO_FINAL_MAX_TOKENS)
    if operational_status_result:
        return min(max_tokens, OPERATIONAL_STATUS_FINAL_MAX_TOKENS)
    if compact_shell_analysis_result:
        return min(
            resolve_max_tokens("final_from_tool", max_tokens, evidence_kind=evidence_kind or "shell", evidence_chars=evidence_chars),
            LONG_SHELL_ANALYSIS_FINAL_MAX_TOKENS,
        )
    if brief_shell_result:
        return min(
            resolve_max_tokens("final_from_tool", max_tokens, evidence_kind=evidence_kind or "shell", evidence_chars=evidence_chars),
            BRIEF_SHELL_FINAL_MAX_TOKENS,
        )
    return resolve_max_tokens("final_from_tool", max_tokens, evidence_kind=evidence_kind, evidence_chars=evidence_chars)


def prepare_final_tool_messages(messages: list[Message]) -> list[Message]:
    prompt = last_user_text(messages)
    prepared = _prune_to_latest_successful_tool_evidence(messages)
    return _compact_latest_tool_evidence(prepared, prompt=prompt)


def _prune_to_latest_successful_tool_evidence(messages: list[Message]) -> list[Message]:
    tool_message, _command = last_successful_shell_result_and_command(messages)
    if tool_message is None:
        return [dict(message) for message in messages]
    selected_tool_call_id = tool_message.get("tool_call_id")
    selected_tool_identity = id(tool_message)
    pruned: list[Message] = []
    for message in messages:
        role = message.get("role")
        if role == "tool" and message.get("name") == "exec_shell_full_command":
            keep_tool = False
            if isinstance(selected_tool_call_id, str):
                keep_tool = message.get("tool_call_id") == selected_tool_call_id
            else:
                keep_tool = id(message) == selected_tool_identity
            if not keep_tool:
                continue
            pruned.append(dict(message))
            continue
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            filtered_calls = list(message["tool_calls"])
            if isinstance(selected_tool_call_id, str):
                filtered_calls = [
                    tool_call for tool_call in filtered_calls if isinstance(tool_call, dict) and tool_call.get("id") == selected_tool_call_id
                ]
            if not filtered_calls and not str(message.get("content") or "").strip():
                continue
            copied = dict(message)
            if filtered_calls:
                copied["tool_calls"] = filtered_calls
            else:
                copied.pop("tool_calls", None)
            pruned.append(copied)
            continue
        pruned.append(dict(message))
    return pruned


def _compact_latest_tool_evidence(messages: list[Message], *, prompt: str | None) -> list[Message]:
    tool_message, command = last_successful_shell_result_and_command(messages)
    if tool_message is None:
        return messages
    if is_list_shell_command(command) and not is_compact_list_request(prompt):
        return messages
    compacted: list[Message] = [dict(message) for message in messages]
    target_index = next(
        (index for index in range(len(compacted) - 1, -1, -1) if compacted[index].get("role") == "tool"),
        None,
    )
    if target_index is None:
        return compacted
    content = compacted[target_index].get("content")
    if not isinstance(content, str):
        return compacted
    replacement = _compact_tool_content(content, prompt=prompt)
    if replacement == content:
        return compacted
    compacted[target_index]["content"] = replacement
    return compacted


def _compact_tool_content(content: str, *, prompt: str | None) -> str:
    if len(content) <= FINAL_TOOL_CONTENT_COMPACT_THRESHOLD:
        return content
    if "shell_output_pdf_text: true" in content:
        return _compact_pdf_tool_content(content, brief_request=is_brief_final_request(prompt))
    return _compact_generic_tool_content(content)


def _compact_pdf_tool_content(content: str, *, brief_request: bool) -> str:
    prefix, body = _split_tool_content_body(content)
    limit = FINAL_TOOL_PDF_BRIEF_CHARS if brief_request else FINAL_TOOL_PDF_MAX_CHARS
    compact_body = _truncate_with_marker(body, head_chars=limit, tail_chars=0)
    return _join_tool_content(prefix, compact_body)


def _compact_generic_tool_content(content: str) -> str:
    prefix, body = _split_tool_content_body(content)
    compact_body = _truncate_with_marker(
        body,
        head_chars=FINAL_TOOL_CONTENT_HEAD_CHARS,
        tail_chars=FINAL_TOOL_CONTENT_TAIL_CHARS,
    )
    return _join_tool_content(prefix, compact_body)


def _split_tool_content_body(content: str) -> tuple[str, str]:
    marker = "content:\n"
    if marker in content:
        prefix, body = content.split(marker, 1)
        return f"{prefix}{marker}", body
    return "", content


def _join_tool_content(prefix: str, body: str) -> str:
    return f"{prefix}{body}" if prefix else body


def _truncate_with_marker(text: str, *, head_chars: int, tail_chars: int) -> str:
    if len(text) <= head_chars + tail_chars + len(FINAL_TOOL_TRUNCATION_MARKER) + 8:
        return text
    head = text[:head_chars].rstrip()
    if tail_chars <= 0:
        return f"{head}\n{FINAL_TOOL_TRUNCATION_MARKER}"
    tail = text[-tail_chars:].lstrip()
    return f"{head}\n{FINAL_TOOL_TRUNCATION_MARKER}\n{tail}"


def final_tool_retry_max_tokens(
    max_tokens: int,
    *,
    web_fetch_result: bool,
    web_search_result: bool = False,
    previous_finish_reason: str | None = None,
) -> int:
    if web_search_result:
        return resolve_max_tokens(
            "final_from_tool_retry",
            max_tokens,
            evidence_kind="web_search",
            previous_finish_reason=previous_finish_reason,
        )
    if web_fetch_result:
        return min(max_tokens, WEB_FETCH_FINAL_MAX_TOKENS)
    return resolve_max_tokens(
        "final_from_tool_retry",
        max_tokens,
        previous_finish_reason=previous_finish_reason,
    )


def final_tool_retry_instruction() -> Message:
    return {
        "role": "user",
        "content": "Do not call tools. Provide a shorter final answer from the available tool result now.",
    }


def final_tool_compact_retry_instruction() -> Message:
    return {
        "role": "user",
        "content": (
            "The previous final answer was too long, repetitive, or ran out of space. "
            "Write one short final answer now using only the existing tool results. "
            "Use plain prose only. No headings. No long bullet lists. No code fences. "
            "No thinking. No repetition. Focus only on the most important findings. "
            "Limit yourself to three to five sentences. Do not call tools."
        ),
    }


def final_tool_compact_retry_max_tokens(max_tokens: int, *, messages: list[Message] | None = None) -> int:
    return resolve_max_tokens("repair", max_tokens)


def final_from_tool_retry_reason(
    result: ChatResult,
    *,
    length_retry_allowed: bool,
    incomplete_retry_allowed: bool = False,
    messages: list[Message] | None = None,
) -> str | None:
    if result.tool_calls:
        return "tool_call_in_final"
    if contains_raw_tool_call(result.content):
        return "raw_tool_call"
    if not result.content and result.finish_reason == "stop":
        return "empty_final"
    if not result.content.strip() and result.finish_reason == "length":
        return "empty_length"
    if length_retry_allowed and result.finish_reason == "length":
        return "length"
    return None


def final_from_tool_compact_retry_reason(result: ChatResult, *, messages: list[Message]) -> str | None:
    if has_web_search_tool_result(messages):
        return None
    if not _has_long_shell_tool_result(messages):
        return None
    if result.finish_reason == "length" and result.content.strip():
        return "length"
    if is_repetitive_final_answer(result.content):
        return "repetition"
    return None


def classify_final_answer_completeness(content: str, *, messages: list[Message] | None = None) -> FinalAnswerCompleteness:
    stripped = content.strip()
    lowered = stripped.lower()
    if not stripped:
        return FinalAnswerCompleteness("incomplete_stub", "empty")
    if (
        _has_open_thought_channel(content)
        or looks_like_reasoning_without_final_answer(lowered)
        or looks_like_reasoning_leakage(content, lowered_content=lowered)
    ):
        return FinalAnswerCompleteness("reasoning_like")
    if "<|channel>thought" in content and "<channel|>" in content:
        tail = content.split("<channel|>", 1)[1].strip()
        if tail:
            return FinalAnswerCompleteness("complete")
        return FinalAnswerCompleteness("reasoning_like")
    if re.search(r"(?:^|\n)\s*#{1,6}\s*$", content.rstrip()):
        return FinalAnswerCompleteness("malformed_markdown", "heading_stub")
    if re.search(r"(?:^|\n)\s*(?:[-*]|\d+\.)\s+\*\*[^*\n]+:\*\*\s*$", content.rstrip()):
        return FinalAnswerCompleteness("incomplete_stub", "list_label_stub")
    if content.count("`") % 2 == 1:
        return FinalAnswerCompleteness("malformed_markdown", "unclosed_backtick")
    stub_text = re.sub(r"[*_~`]+$", "", stripped).rstrip()
    if stub_text.endswith((":", "-", "*")):
        return FinalAnswerCompleteness("incomplete_stub", "trailing_stub")
    if looks_like_incomplete_final(content):
        return FinalAnswerCompleteness("incomplete_stub", "plain_incomplete")
    if messages and _looks_too_short_after_tool(content, messages):
        return FinalAnswerCompleteness("too_short_after_tool")
    return FinalAnswerCompleteness("complete")


def contains_raw_tool_call(content: str) -> bool:
    return "<|tool_call>" in content or "<tool_call|>" in content


def looks_like_incomplete_final(content: str) -> bool:
    text = content.strip()
    if len(text) < 48:
        return False
    if "\n" in text:
        return False
    if text.startswith(("-", "*", "•")):
        return False
    if "/" in text and " " not in text:
        return False
    if re.fullmatch(r"[A-Za-z0-9._/-]+", text):
        return False
    if re.search(r"[.!?][\"')\\]]?$", text):
        return False
    words = text.split()
    if len(words) < 8:
        return False
    return bool(re.search(r"[A-Za-z0-9]$", text))


def looks_like_reasoning_without_final_answer(lowered_content: str) -> bool:
    if any(marker in lowered_content for marker in _FINAL_MARKERS):
        return False
    return lowered_content.startswith(_REASONING_PREFIXES)


def looks_like_reasoning_leakage(content: str, *, lowered_content: str | None = None) -> bool:
    lowered = lowered_content if lowered_content is not None else content.lower()
    if not any(phrase in lowered for phrase in _REASONING_LEAK_META_PHRASES):
        return False
    possibility_matches = re.findall(r"(?:^|\n)\s*(?:[-*]\s+)?(?:\*\*)?possibility\s+[a-z]\b", lowered, flags=re.MULTILINE)
    if len(possibility_matches) >= 2:
        return True
    bullet_lines = [
        line.strip().lower()
        for line in content.splitlines()
        if line.strip()
    ]
    possibility_bullets = sum(
        1
        for line in bullet_lines
        if ("possibility " in line and re.search(r"possibility\s+[a-z]\b", line))
    )
    return possibility_bullets >= 2


def _has_open_thought_channel(content: str) -> bool:
    if "<|channel>thought" not in content:
        return False
    tail = content.split("<|channel>thought", 1)[1]
    return "<channel|>" not in tail


def _looks_too_short_after_tool(content: str, messages: list[Message]) -> bool:
    prompt = last_user_text(messages)
    if is_operational_status_request(prompt) or is_compact_list_request(prompt):
        return False
    tool_message, command = last_successful_shell_result_and_command(messages)
    if tool_message is None or is_list_shell_command(command):
        return False
    tool_content = tool_message.get("content")
    if not isinstance(tool_content, str) or len(tool_content) < 800:
        return False
    text = content.strip()
    if len(text) >= 48:
        return False
    words = text.split()
    return len(words) <= 6 and not re.search(r"[.!?][\"')\]]?$", text)


def is_repetitive_final_answer(content: str) -> bool:
    text = content.strip()
    if len(text) < 100:
        return False
    paragraphs = [_normalize_repetition_unit(part) for part in re.split(r"\n\s*\n", text) if _normalize_repetition_unit(part)]
    if _has_duplicate_units(paragraphs, min_len=48):
        return True
    sentences = [
        _normalize_repetition_unit(part)
        for part in re.split(r"(?<=[.!?])\s+|\n+", text)
        if _normalize_repetition_unit(part)
    ]
    if _has_duplicate_units(sentences, min_len=32):
        return True
    return False


def _has_long_shell_tool_result(messages: list[Message]) -> bool:
    tool_message, command = last_successful_shell_result_and_command(messages)
    if tool_message is None or is_list_shell_command(command):
        return False
    content = tool_message.get("content")
    return isinstance(content, str) and len(content) >= 1200


def _normalize_repetition_unit(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _has_duplicate_units(units: list[str], *, min_len: int) -> bool:
    counts: dict[str, int] = {}
    for unit in units:
        if len(unit) < min_len:
            continue
        counts[unit] = counts.get(unit, 0) + 1
        if counts[unit] >= 2:
            return True
    return False


def has_large_file_excerpt(messages: list[Message]) -> bool:
    for message in reversed(messages):
        if message.get("role") == "tool":
            content = message.get("content")
            return isinstance(content, str) and (
                "large_file_excerpt: true" in content
                or (
                    "shell_output_pdf_text: true" in content
                    and "chunk_index:" in content
                    and "total_chunks:" in content
                )
            )
    return False


def has_html_cleaned_tool_result(messages: list[Message]) -> bool:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        return isinstance(content, str) and "shell_output_html_cleaned: true" in content
    return False


def has_web_search_tool_result(messages: list[Message]) -> bool:
    tool_message, command = last_successful_shell_result_and_command(messages)
    if tool_message is None:
        return False
    content = tool_message.get("content")
    if isinstance(content, str) and "web_search_results: true" in content:
        return True
    if not command:
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    return bool(tokens and tokens[0] == "orbit-web-search")


def has_pdf_text_tool_result(messages: list[Message]) -> bool:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if isinstance(content, str) and "shell_output_pdf_text: true" in content:
            return True
    return False


def has_tool_result(messages: list[Message], name: str) -> bool:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        return message.get("name") == name
    return False


def has_list_like_tool_result(messages: list[Message]) -> bool:
    tool_message, command = last_successful_shell_result_and_command(messages)
    if tool_message is None:
        return False
    return is_list_shell_command(command)
    return False


def last_successful_shell_result_and_command(messages: list[Message]) -> tuple[Message | None, str | None]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "tool" or message.get("name") != "exec_shell_full_command":
            continue
        content = message.get("content")
        if not isinstance(content, str) or _is_error_tool_content(content):
            continue
        tool_call_id = message.get("tool_call_id")
        return message, _shell_command_for_tool_call_id(messages[:index], tool_call_id)
    return None, None


def _shell_command_for_tool_call_id(messages: list[Message], tool_call_id: object) -> str | None:
    if not isinstance(tool_call_id, str):
        return last_shell_full_command(messages)
    for message in reversed(messages):
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for tool_call in reversed(calls):
            if not isinstance(tool_call, dict):
                continue
            if tool_call.get("id") != tool_call_id:
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict) or function.get("name") != "exec_shell_full_command":
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return None
            if not isinstance(arguments, dict):
                return None
            command = arguments.get("command")
            return command if isinstance(command, str) else None
    return None


def last_shell_full_command(messages: list[Message]) -> str | None:
    for message in reversed(messages):
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for tool_call in reversed(calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict) or function.get("name") != "exec_shell_full_command":
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return None
            if not isinstance(arguments, dict):
                return None
            command = arguments.get("command")
            return command if isinstance(command, str) else None
    return None


def is_list_shell_command(command: str | None) -> bool:
    if not command:
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    if tokens[0] == "ls":
        return True
    if tokens[0] != "find":
        return False
    if "-exec" in tokens:
        return False
    return True


def _is_error_tool_content(content: str) -> bool:
    return content.startswith("error:")


def is_compact_list_request(prompt: str | None) -> bool:
    if prompt is None:
        return True
    return bool(_COMPACT_LIST_REQUEST_RE.search(prompt))


_OPERATIONAL_STATUS_RE = re.compile(
    r"\b(?:is|was|were|did|does|do|has|have|what|where|confirm|check|verify|status|exists?|saved?|renamed?|"
    r"deleted?|removed?|created?|written?|updated?|changed?|moved?|copied?|path|name|file)\b",
    re.IGNORECASE,
)
_OPERATIONAL_ACTION_RE = re.compile(
    r"^\s*(?:remove|delete|rename|move|create|save|write|copy|update|change)\b",
    re.IGNORECASE,
)
_CONTENT_REQUEST_RE = re.compile(
    r"\b(?:summari[sz]e|analy[sz]e|explain|review|describe|read|show|print|content|contents|source|html|page)\b",
    re.IGNORECASE,
)
_CONTENT_PHRASE_RE = re.compile(
    r"\b(?:what(?:'s|\s+is)?\s+in|what\s+does\b.*\bcontain|contains?|inside)\b",
    re.IGNORECASE,
)
_BRIEF_FINAL_REQUEST_RE = re.compile(
    r"\b(?:one\s+sentence|single\s+sentence|one\s+concise\s+sentence|concise\s+sentence|brief(?:ly)?|short(?:ly)?|in\s+short|main\s+(?:issue|point|finding)|brief\s+answer|short\s+answer)\b",
    re.IGNORECASE,
)
_SHELL_REVIEW_REQUEST_RE = re.compile(
    r"\b(?:review|inspect|audit|diagnos(?:e|is)|debug|bug|issue|issues|problem|problems|fix|remediat\w*|vulnerab\w*|security|secure|weakness|weaknesses|risk|risks|exploit|injection|misconfig\w*|failure|failures)\b",
    re.IGNORECASE,
)


def last_user_text(messages: list[Message]) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        return content if isinstance(content, str) else None
    return None


def is_brief_final_request(prompt: str | None) -> bool:
    if prompt is None:
        return False
    return bool(_BRIEF_FINAL_REQUEST_RE.search(prompt))


def is_shell_review_request(prompt: str | None) -> bool:
    if not prompt:
        return False
    return bool(_SHELL_REVIEW_REQUEST_RE.search(prompt))


def is_operational_status_request(prompt: str | None) -> bool:
    if not prompt:
        return False
    if (_CONTENT_REQUEST_RE.search(prompt) or _CONTENT_PHRASE_RE.search(prompt)) and not _OPERATIONAL_ACTION_RE.search(prompt):
        return False
    return _OPERATIONAL_STATUS_RE.search(prompt) is not None or _OPERATIONAL_ACTION_RE.search(prompt) is not None


def is_exhaustive_document_request(prompt: str | None) -> bool:
    return bool(prompt and _EXHAUSTIVE_DOCUMENT_RE.search(prompt))
