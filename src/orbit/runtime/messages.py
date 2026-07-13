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


CHAT_SYSTEM_PROMPT = "Answer normally for conversation, explanation, writing, opinion, and general knowledge."
VISIBLE_CHAT_SYSTEM_PROMPT = (
    "Use visible assistant answers as the source for follow-ups. Answer the latest request directly. "
    "Preserve facts, paths, counts, errors, filenames, and matched values. "
    "If a detail is missing, say only that it is unavailable in the visible conversation. Never infer omitted context."
)
MEDIA_SYSTEM_PROMPT = "Answer using the attached image/audio."
_COMMAND_SYSTEM_TEMPLATE = """Decide compactly whether the user request needs local tools.
Tool tasks: files/read/edit/create/append/delete, system, URLs/web/search/fetch, execution, and analysis that needs local or fetched evidence.
For tool tasks, return a tool decision; do not answer directly or return CHAT.
If the latest request is only a recap, repeat, summary, explanation, comparison, or continuation of information already in this conversation, prefer {{"route":"CHAT"}} when the prior context is sufficient.
Call tools for fresh/current data, verification, changed files/state, new information, or missing/stale/ambiguous/insufficient prior context.
Web/search/latest/current/online and URL fetch/read/open/explain/summarize/analyze requests are tool tasks; return a compact tool decision, not a direct answer.
Specific file read/explain/summarize/analyze requests require file content evidence; return a content-reading command decision, not a directory listing.
If the target is a file path or filename, use a content-reading command; do not inspect it with list_directory.
Use directory listing only when the user asks to list files or inspect directory structure; never use {{"path":"..."}} to answer about a file's contents.
The one-sentence direct-answer exception below is only for requests that are not tool tasks and need no external evidence.
If no shell/tool and no external evidence is needed:
- For a complete answer that fits in one short sentence, write the answer directly and stop.
- For any answer needing explanation, a list, a paragraph, or more than one short sentence, return {{"route":"CHAT"}} only.
Return valid one-line JSON only for route/tool decisions.

For shell:
{{"command":"..."}}

For specific file content read/explain/summarize:
{{"command":"cat README.md"}}

Example file summary request:
summarize README.md -> {{"command":"cat README.md"}}

For generic web search:
{{"command":"orbit-web-search \\"query\\""}}

For URL fetch/read page:
{{"url":"https://example.com"}}

For normal no-tool final answer pass:
{{"route":"CHAT"}}

For compact directory listing only:
{{"path":".","recursive":false}}

For compact local machine specs:
{{"include_cpu":true,"include_memory":true,"include_disks":true,"include_os":true}}

Environment: OS={os_name}; shell={shell_name}.

Use given paths exactly. Use native commands in workdir. For compact directory listings, prefer the list_directory JSON shape over shell commands like ls -R, find, or tree. For local machine specs, prefer the system_info JSON shape over noisy shell commands like lscpu, free, df, uname, or cat /proc/*. Generic web search: orbit-web-search "query". For explicit URL fetch/read/explain/summarize/analyze requests, prefer the fetch_url tool; shell fetch commands such as curl are still allowed when needed. Quote spaced paths.

Do not claim no access for local/system/web.
Never use <|tool_call>, call:shell, markdown, fences, or prose for shell.
Do not write long prose in the route pass.

Example:
specs of this computer -> {{"include_cpu":true,"include_memory":true,"include_disks":true,"include_os":true}}

For analysis, prefer content, source, binaries, strings, logs, archives, or fetched data, not metadata."""
ROUTE_SYSTEM_PROMPT = _COMMAND_SYSTEM_TEMPLATE.format(os_name=_detect_os(), shell_name=_detect_shell())
TOOL_CALL_SYSTEM_PROMPT = (
    "Call exactly one available tool and output no prose. "
    "Operate on the latest user request only. "
    "Ignore older tool results, file/page content, or prior task context unless the latest user request explicitly refers to them. "
    "Prefer list_directory for compact directory listings. "
    "Prefer system_info for compact local machine specs such as OS, CPU, RAM, disk, and Python runtime. "
    "Prefer fetch_url for explicit URL fetch/read/explain/summarize/analyze requests. "
    'Use orbit-web-search "query" for generic web search. '
    "Use exec_shell_full_command for local/system tasks or when another tool is more appropriate. "
    "Quote paths containing spaces in shell commands. "
    "For analysis, collect direct evidence from content/source/strings/logs/archives/fetched data."
)
TOOL_CALL_JSON_RETRY_PROMPT = (
    "The previous tool call had invalid JSON arguments. "
    "Return exactly one tool call now. "
    "Arguments must be valid compact JSON. "
    "For shell command, use one single-line command string only: no comments, no literal newlines."
)
FINAL_FROM_TOOL_SYSTEM_PROMPT = (
    "Answer concisely from the tool result. "
    "Do not call tools or emit raw tool-call syntax. "
    "Never claim lack of access when a result is present. "
    "Report errors briefly."
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


def with_visible_chat_system_prompt(messages: list[Message]) -> list[Message]:
    return [{"role": "system", "content": VISIBLE_CHAT_SYSTEM_PROMPT}, *[dict(message) for message in messages]]


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
