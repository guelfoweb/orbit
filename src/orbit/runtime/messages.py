from __future__ import annotations

import os
import platform
from pathlib import Path

from orbit.backend.base import Message
from orbit.runtime.media import AudioInput, ImageInput


def _detect_os() -> str:
    name = platform.system().lower()
    if name == "darwin":
        return "macos"
    if name in {"linux", "windows"}:
        return name
    return name or "unknown"


def _detect_shell() -> str:
    if _detect_os() == "windows":
        comspec = os.environ.get("COMSPEC")
        return Path(comspec).name if comspec else "powershell"
    shell = os.environ.get("SHELL")
    return Path(shell).name if shell else "sh"


CHAT_SYSTEM_PROMPT = (
    "Concise local assistant. "
    "Answer normally for conversation, explanation, opinion, writing, and general knowledge. "
    "If a local file task lacks a specific path and no tool is available, ask for the path in one sentence. "
    "Do not claim you lack local filesystem access and do not explain generic file-processing best practices unless asked. "
    "Do not emit route JSON or raw tool-call syntax."
)
MEDIA_SYSTEM_PROMPT = "Answer using the attached image/audio."
_COMMAND_SYSTEM_TEMPLATE = """Answer normally unless shell is needed.
Shell tasks: files/edit/create/append/delete, system, URLs/web/search/fetch, execution, analysis.
Return valid one-line JSON only:

{{"command":"..."}}

Environment: OS={os_name}; shell={shell_name}.

Use native commands in workdir. Use curl for URLs. Quote spaced paths.

Do not claim no access for local/system/web.
Never use <|tool_call>, call:shell, markdown, fences, or prose for shell.

Example:
specs of this computer -> {{"command":"uname -a; lscpu; free -h; df -h"}}

For analysis, prefer content, source, binaries, strings, logs, archives, or fetched data, not metadata."""
ROUTE_SYSTEM_PROMPT = _COMMAND_SYSTEM_TEMPLATE.format(os_name=_detect_os(), shell_name=_detect_shell())
TOOL_CALL_SYSTEM_PROMPT = (
    "Call exec_shell_full_command exactly once and output no prose. "
    "Use one compact one-line shell command. "
    "Use curl for URLs when content is needed. "
    "Quote paths containing spaces. "
    "For analysis, collect direct evidence from content/source/strings/logs/archives/fetched data."
)
TOOL_CALL_JSON_RETRY_PROMPT = (
    "The previous tool call had invalid JSON arguments. "
    "Return exactly one tool call now. "
    "Arguments must be valid compact JSON. "
    "For shell command, use one single-line command string only: no comments, no literal newlines."
)
FINAL_FROM_TOOL_SYSTEM_PROMPT = (
    "Answer concisely from the available tool result. "
    "Do not call tools. "
    "Do not emit raw tool-call syntax. "
    "If a tool result is present, do not claim lack of access. "
    "If the tool result is an error, report the error briefly."
)
DEFAULT_SYSTEM_PROMPT = ROUTE_SYSTEM_PROMPT
TOOL_SYSTEM_PROMPT = TOOL_CALL_SYSTEM_PROMPT


def message_content(
    prompt: str,
    images: list[ImageInput],
    audios: list[AudioInput],
) -> str | list[dict[str, object]]:
    if not images and not audios:
        return prompt
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image.data_url}})
    for audio in audios:
        content.append({"type": "input_audio", "input_audio": {"data": audio.data, "format": audio.format}})
    return content


def with_media_system_prompt(messages: list[Message]) -> list[Message]:
    copied = [dict(message) for message in messages]
    if copied and copied[0].get("role") == "system":
        copied[0]["content"] = MEDIA_SYSTEM_PROMPT
        return copied
    return [{"role": "system", "content": MEDIA_SYSTEM_PROMPT}, *copied]


def with_command_system_prompt(messages: list[Message]) -> list[Message]:
    copied = [dict(message) for message in messages]
    if copied and copied[0].get("role") == "system":
        copied[0]["content"] = ROUTE_SYSTEM_PROMPT
        return copied
    return [{"role": "system", "content": ROUTE_SYSTEM_PROMPT}, *copied]


def with_chat_system_prompt(messages: list[Message]) -> list[Message]:
    copied = [dict(message) for message in messages]
    if copied and copied[0].get("role") == "system":
        copied[0]["content"] = CHAT_SYSTEM_PROMPT
        return copied
    return [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, *copied]


def with_tool_call_system_prompt(messages: list[Message]) -> list[Message]:
    if messages and messages[0].get("role") == "system":
        copied = [dict(message) for message in messages]
        copied[0]["content"] = TOOL_CALL_SYSTEM_PROMPT
        return copied
    return messages


def with_final_tool_system_prompt(messages: list[Message]) -> list[Message]:
    if messages and messages[0].get("role") == "system":
        copied = [dict(message) for message in messages]
        copied[0]["content"] = FINAL_FROM_TOOL_SYSTEM_PROMPT
        return copied
    return [{"role": "system", "content": FINAL_FROM_TOOL_SYSTEM_PROMPT}, *messages]
