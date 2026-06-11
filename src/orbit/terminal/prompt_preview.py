from __future__ import annotations

import hashlib

LONG_TEXT_PREVIEW_CHARS = 50
LONG_TEXT_THRESHOLD = 800
LONG_TEXT_DIGEST_CHARS = 8


def is_long_text_prompt(prompt: str, *, threshold: int = LONG_TEXT_THRESHOLD) -> bool:
    return len(prompt) > threshold


def compact_prompt_preview(
    prompt: str,
    *,
    threshold: int = LONG_TEXT_THRESHOLD,
    preview_chars: int = LONG_TEXT_PREVIEW_CHARS,
    multiline: bool = False,
) -> str:
    if not is_long_text_prompt(prompt, threshold=threshold):
        return prompt
    prefix = _single_line_prefix(prompt, preview_chars=preview_chars)
    separator = "\n" if multiline else " "
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:LONG_TEXT_DIGEST_CHARS]
    return f"{prefix}...{separator}[text {len(prompt)} chars #{digest}]"


def is_compact_prompt_preview(prompt: str) -> bool:
    stripped = prompt.strip()
    return "[text " in stripped and stripped.endswith("]")


def _single_line_prefix(prompt: str, *, preview_chars: int) -> str:
    return " ".join(prompt[:preview_chars].split())
