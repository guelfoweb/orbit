from __future__ import annotations

import shlex
from dataclasses import dataclass

from orbit.backend import ChatResult
from orbit.backend.base import Message


FINAL_FROM_TOOL_MIN_TOKENS = 256
LARGE_FILE_FINAL_MAX_TOKENS = 128
WEB_FETCH_FINAL_MAX_TOKENS = 72
LIST_FINAL_MAX_TOKENS = 96


@dataclass(frozen=True)
class FinalToolPolicy:
    messages: list[Message]
    max_tokens: int
    length_retry_allowed: bool
    web_fetch_result: bool


def build_final_tool_policy(messages: list[Message], *, max_tokens: int, streamed: bool) -> FinalToolPolicy:
    large_file_excerpt = has_large_file_excerpt(messages)
    web_fetch_result = has_web_fetch_tool_result(messages)
    web_search_result = has_tool_result(messages, "search_web")
    list_like_result = has_list_like_tool_result(messages)
    shell_result = has_tool_result(messages, "exec_shell_command")
    read_file_result = has_tool_result(messages, "read_file")
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
    elif web_search_result:
        call_messages = [
            *call_messages,
            {
                "role": "user",
                "content": (
                    "Use only the search results already available. "
                    "Answer in at most four short bullets. "
                    "Keep the main facts and cite source names only when useful. "
                    "Do not add background beyond the results. "
                    "Expand only if the user asks for more detail."
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
    elif list_like_result:
        call_messages = [
            *call_messages,
            {"role": "user", "content": "Return only the listed names, compactly. No categories or explanations."},
        ]
    elif shell_result:
        call_messages = [
            *call_messages,
            {
                "role": "user",
                "content": (
                    "Use only the command output. "
                    "Return at most six compact findings. "
                    "Preserve important numbers and names. "
                    "Do not explain generic concepts. Expand only if asked."
                ),
            },
        ]
    elif read_file_result:
        call_messages = [
            *call_messages,
            {
                "role": "user",
                "content": (
                    "Use only the file content. "
                    "Respect any requested length. "
                    "If no length is requested, answer concisely. "
                    "Expand only if asked."
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
        ),
        length_retry_allowed=(not streamed and (large_file_excerpt or web_fetch_result)),
        web_fetch_result=web_fetch_result,
    )


def final_tool_max_tokens(
    max_tokens: int,
    *,
    large_file_excerpt: bool,
    web_fetch_result: bool,
    list_like_result: bool,
) -> int:
    if large_file_excerpt:
        return min(max_tokens, LARGE_FILE_FINAL_MAX_TOKENS)
    if web_fetch_result:
        return min(max_tokens, WEB_FETCH_FINAL_MAX_TOKENS)
    if list_like_result:
        return min(max_tokens, LIST_FINAL_MAX_TOKENS)
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


def final_from_tool_retry_reason(result: ChatResult, *, length_retry_allowed: bool) -> str | None:
    if result.tool_calls:
        return "tool_call_in_final"
    if contains_raw_tool_call(result.content):
        return "raw_tool_call"
    if not result.content and result.finish_reason == "stop":
        return "empty_final"
    if length_retry_allowed and result.finish_reason == "length":
        return "length"
    return None


def contains_raw_tool_call(content: str) -> bool:
    return "<|tool_call>" in content or "<tool_call|>" in content


def has_large_file_excerpt(messages: list[Message]) -> bool:
    for message in reversed(messages):
        if message.get("role") == "tool":
            content = message.get("content")
            return isinstance(content, str) and "large_file_excerpt: true" in content
    return False


def has_web_fetch_tool_result(messages: list[Message]) -> bool:
    return has_tool_result(messages, "fetch_url")


def has_tool_result(messages: list[Message], name: str) -> bool:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        return message.get("name") == name
    return False


def has_list_like_tool_result(messages: list[Message]) -> bool:
    last_shell_command = last_exec_shell_command(messages)
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        name = message.get("name")
        if name in {"list_files", "file_glob_search"}:
            return True
        if name == "exec_shell_command":
            return is_list_shell_command(last_shell_command)
        return False
    return False


def last_exec_shell_command(messages: list[Message]) -> str | None:
    for message in reversed(messages):
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for tool_call in reversed(calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict) or function.get("name") != "exec_shell_command":
                continue
            arguments = function.get("arguments")
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
