from __future__ import annotations

from orbit.backend.base import Message
from orbit.runtime.media import AudioInput, ImageInput


CHAT_SYSTEM_PROMPT = (
    "Concise local assistant. "
    "Answer normally for conversation, explanation, opinion, writing, and general knowledge. "
    "If a local file task lacks a specific path and no tool is available, ask for the path in one sentence. "
    "Do not claim you lack local filesystem access and do not explain generic file-processing best practices unless asked. "
    "Do not emit route JSON or raw tool-call syntax."
)
MEDIA_SYSTEM_PROMPT = "Answer using the attached image/audio."
ROUTE_SYSTEM_PROMPT = """Classify the latest user request for Orbit. Return only one JSON object.

{"_route":"CHAT"}
{"_route":"FILESYSTEM","tool":"list_files|read_file|grep_search|file_glob_search|exec_shell_command"}
{"_route":"FILE_EDIT","tool":"write_file|edit_file|apply_diff|make_directory|delete_path"}
{"_route":"WEB","tool":"search_web|fetch_url"}
{"_route":"MEDIA"}

Use CHAT for conversation, explanation, opinion, writing, or general knowledge.
Include obvious arguments in the same JSON: path, url, query, pattern, command.
Copy paths and URLs exactly from the user prompt. Never normalize, correct, or rewrite them.
If the user asks to run/execute a shell command, choose FILESYSTEM/exec_shell_command.
Do not convert shell commands into FILE_EDIT tools; execution guardrails decide if they are allowed.
For explicit http/https URLs return {"_route":"WEB","tool":"fetch_url","url":"<url>"}.
Do not say you lack internet.
Do not perform the task. Classify only."""
TOOL_CALL_SYSTEM_PROMPT = (
    "When tools are available, call exactly one needed tool and output no prose. "
    "When tools are not available, answer concisely from the tool result. "
    "No repeated equivalent tool calls. "
    "Use read_file before edit_file when context is needed. "
    "Use edit_file/apply_diff, not exec_shell_command, for edits."
)
FINAL_FROM_TOOL_SYSTEM_PROMPT = TOOL_CALL_SYSTEM_PROMPT
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


def with_route_system_prompt(messages: list[Message]) -> list[Message]:
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
