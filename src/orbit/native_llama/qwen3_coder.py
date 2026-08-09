from __future__ import annotations

import json

from .chat_template import NativeMessage


QWEN3_CODER_ARTIFACT_PROTOCOL_ID = "qwen3-coder-json-string-v1"
QWEN3_CODER_ARTIFACT_OPENING = '"'
QWEN3_CODER_ARTIFACT_SYSTEM_PROMPT = (
    "Generate one complete bounded UTF-8 file body as the value of the pre-opened JSON string. "
    "Emit JSON string characters and escapes only, close the string after the final file character, "
    "and end the assistant turn. The opening and closing quotes are transport syntax, not file content."
)
QWEN3_CODER_ARTIFACT_GRAMMAR = r'''
root ::= string-char* "\""
string-char ::= [^"\\\x00-\x1F] | "\\" escape
escape ::= ["\\/bfnrt] | "u" hex hex hex hex
hex ::= [0-9a-fA-F]
'''.strip()


def qwen3_coder_artifact_messages(messages: list[NativeMessage]) -> list[NativeMessage]:
    if len(messages) != 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise RuntimeError("Qwen3-Coder artifact generation received an unexpected message sequence")
    framed = [dict(message) for message in messages]
    framed[0]["content"] = QWEN3_CODER_ARTIFACT_SYSTEM_PROMPT
    return framed


def qwen3_coder_artifact_prompt(rendered_prompt: str) -> str:
    if not rendered_prompt:
        raise RuntimeError("Qwen3-Coder artifact generation received an empty rendered prompt")
    return rendered_prompt + QWEN3_CODER_ARTIFACT_OPENING


def parse_qwen3_coder_artifact_content(generated: str) -> str:
    try:
        content, end = json.JSONDecoder().raw_decode(QWEN3_CODER_ARTIFACT_OPENING + generated)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("Qwen3-Coder artifact generation produced malformed JSON-string framing") from exc
    if end != len(generated) + len(QWEN3_CODER_ARTIFACT_OPENING) or not isinstance(content, str):
        raise RuntimeError("Qwen3-Coder artifact generation produced data outside JSON-string framing")
    try:
        content.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise RuntimeError("Qwen3-Coder artifact generation produced invalid UTF-8 content") from exc
    return content
