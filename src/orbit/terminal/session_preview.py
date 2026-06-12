from __future__ import annotations


SESSION_PREVIEW_MESSAGES = 4
SESSION_PREVIEW_CHARS = 90


def has_existing_session_context(messages: list[dict[str, object]]) -> bool:
    return any(message.get("role") != "system" for message in messages)


def format_recent_session_messages(messages: list[dict[str, object]]) -> list[str]:
    visible = [message for message in messages if message.get("role") != "system"]
    tail = visible[-SESSION_PREVIEW_MESSAGES:]
    return [f"  {str(message.get('role', 'unknown'))}: {_preview_message_content(message)}" for message in tail]


def _preview_message_content(message: dict[str, object]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text")
        if not text:
            text = "[media prompt]"
    else:
        text = str(content)
    compact = " ".join(text.split())
    if len(compact) <= SESSION_PREVIEW_CHARS:
        return compact
    return f"{compact[:SESSION_PREVIEW_CHARS].rstrip()}..."
