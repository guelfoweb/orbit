from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass

from orbit.backend import ChatResult
from orbit.backend.base import Message


FINAL_FROM_TOOL_MIN_TOKENS = 256
LARGE_FILE_FINAL_MAX_TOKENS = 128
WEB_FETCH_FINAL_MAX_TOKENS = 72
LIST_FINAL_MAX_TOKENS = 96
OPERATIONAL_STATUS_FINAL_MAX_TOKENS = 96


@dataclass(frozen=True)
class FinalToolPolicy:
    messages: list[Message]
    max_tokens: int
    length_retry_allowed: bool
    incomplete_retry_allowed: bool
    web_fetch_result: bool


def build_final_tool_policy(messages: list[Message], *, max_tokens: int, streamed: bool) -> FinalToolPolicy:
    large_file_excerpt = has_large_file_excerpt(messages)
    web_fetch_result = has_html_cleaned_tool_result(messages)
    list_like_result = has_list_like_tool_result(messages)
    shell_full_result = has_tool_result(messages, "exec_shell_full_command")
    operational_status_result = shell_full_result and is_operational_status_request(last_user_text(messages))
    call_messages = messages
    if large_file_excerpt:
        call_messages = [
            *call_messages,
            {
                "role": "user",
                "content": (
                    "Use the available large-file excerpt only. "
                    "Answer in at most five short bullets, each under twelve words. Do not quote long passages. "
                    "Do not request more chunks unless the user explicitly asked for exhaustive analysis."
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
                    "Use only the available shell-full output. "
                    "Answer the latest user request directly and concisely from that evidence. "
                    "Prefer the most recent relevant shell result. "
                    "Do not summarize unrelated older output. "
                    "Do not call tools again. "
                    "If the evidence is insufficient, say you cannot confirm."
                ),
            },
        ]
    return FinalToolPolicy(
        messages=call_messages,
        max_tokens=final_tool_max_tokens(
            max_tokens,
            large_file_excerpt=large_file_excerpt,
            web_fetch_result=web_fetch_result,
            list_like_result=list_like_result,
            operational_status_result=operational_status_result,
        ),
        length_retry_allowed=(not streamed and (large_file_excerpt or web_fetch_result)),
        incomplete_retry_allowed=shell_full_result and not (web_fetch_result or list_like_result or operational_status_result),
        web_fetch_result=web_fetch_result,
    )


def final_tool_max_tokens(
    max_tokens: int,
    *,
    large_file_excerpt: bool,
    web_fetch_result: bool,
    list_like_result: bool,
    operational_status_result: bool = False,
) -> int:
    if large_file_excerpt:
        return min(max_tokens, LARGE_FILE_FINAL_MAX_TOKENS)
    if web_fetch_result:
        return min(max_tokens, WEB_FETCH_FINAL_MAX_TOKENS)
    if list_like_result:
        return min(max_tokens, LIST_FINAL_MAX_TOKENS)
    if operational_status_result:
        return min(max_tokens, OPERATIONAL_STATUS_FINAL_MAX_TOKENS)
    return max(max_tokens, FINAL_FROM_TOOL_MIN_TOKENS)


def final_tool_retry_max_tokens(max_tokens: int, *, web_fetch_result: bool) -> int:
    return min(
        max(max_tokens, FINAL_FROM_TOOL_MIN_TOKENS),
        WEB_FETCH_FINAL_MAX_TOKENS if web_fetch_result else max(max_tokens, FINAL_FROM_TOOL_MIN_TOKENS),
    )


def final_tool_retry_instruction() -> Message:
    return {
        "role": "user",
        "content": "Do not call tools. Provide a shorter final answer from the available tool result now.",
    }


def final_from_tool_retry_reason(
    result: ChatResult,
    *,
    length_retry_allowed: bool,
    incomplete_retry_allowed: bool = False,
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
    if incomplete_retry_allowed and looks_like_incomplete_final(result.content):
        return "incomplete_final"
    return None


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


def has_large_file_excerpt(messages: list[Message]) -> bool:
    for message in reversed(messages):
        if message.get("role") == "tool":
            content = message.get("content")
            return isinstance(content, str) and "large_file_excerpt: true" in content
    return False


def has_html_cleaned_tool_result(messages: list[Message]) -> bool:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        return isinstance(content, str) and "shell_output_html_cleaned: true" in content
    return False


def has_tool_result(messages: list[Message], name: str) -> bool:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        return message.get("name") == name
    return False


def has_list_like_tool_result(messages: list[Message]) -> bool:
    last_shell_command = last_shell_full_command(messages)
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        name = message.get("name")
        if name == "exec_shell_full_command":
            return is_list_shell_command(last_shell_command)
        return False
    return False


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
                    continue
            if not isinstance(arguments, dict):
                continue
            command = arguments.get("command")
            if isinstance(command, str):
                return command
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
    return tokens[0] in {"ls", "find"}


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


def last_user_text(messages: list[Message]) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        return content if isinstance(content, str) else None
    return None


def is_operational_status_request(prompt: str | None) -> bool:
    if not prompt:
        return False
    if (_CONTENT_REQUEST_RE.search(prompt) or _CONTENT_PHRASE_RE.search(prompt)) and not _OPERATIONAL_ACTION_RE.search(prompt):
        return False
    return _OPERATIONAL_STATUS_RE.search(prompt) is not None or _OPERATIONAL_ACTION_RE.search(prompt) is not None
