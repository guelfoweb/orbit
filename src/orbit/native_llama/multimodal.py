from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any

from .chat_template import NativeMessage


@dataclass(frozen=True)
class PreparedMultimodalInput:
    messages: list[NativeMessage]
    media_payloads: list[bytes]
    has_image: bool
    has_audio: bool


def prepare_multimodal_messages(
    messages: list[NativeMessage],
    *,
    media_marker: str,
) -> PreparedMultimodalInput | None:
    prepared_messages: list[NativeMessage] = []
    payloads: list[bytes] = []
    has_image = False
    has_audio = False

    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, list):
            prepared_messages.append(dict(message))
            continue

        prepared_content: list[dict[str, object]] = []
        saw_media = False
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text")
                if isinstance(text, str):
                    prepared_content.append({"type": "text", "text": text})
                continue
            if item_type == "image_url":
                data_url = _image_data_url(item)
                payloads.append(_decode_data_url(data_url, expected_prefix="data:image/"))
                prepared_content.append({"type": "media_marker", "text": media_marker})
                has_image = True
                saw_media = True
                continue
            if item_type == "input_audio":
                payloads.append(_decode_audio_data(item))
                prepared_content.append({"type": "media_marker", "text": media_marker})
                has_audio = True
                saw_media = True
                continue
            raise ValueError("unsupported content[].type")

        if saw_media:
            updated = dict(message)
            updated["content"] = prepared_content
            prepared_messages.append(updated)
        else:
            prepared_messages.append(dict(message))

    if not payloads:
        return None
    return PreparedMultimodalInput(
        messages=prepared_messages,
        media_payloads=payloads,
        has_image=has_image,
        has_audio=has_audio,
    )


def flatten_message_content(message: NativeMessage) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return " ".join(parts)


def _image_data_url(item: dict[str, Any]) -> str:
    image_url = item.get("image_url")
    if not isinstance(image_url, dict):
        raise ValueError("image_url must be an object")
    url = image_url.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("image_url.url must be a non-empty string")
    return url


def _decode_audio_data(item: dict[str, Any]) -> bytes:
    input_audio = item.get("input_audio")
    if not isinstance(input_audio, dict):
        raise ValueError("input_audio must be an object")
    data = input_audio.get("data")
    audio_format = input_audio.get("format")
    if not isinstance(data, str) or not data:
        raise ValueError("input_audio.data must be a non-empty string")
    if audio_format not in {"wav", "mp3"}:
        raise ValueError("input_audio.format must be either 'wav' or 'mp3'")
    try:
        return base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("input_audio.data must be valid base64") from exc


def _decode_data_url(value: str, *, expected_prefix: str) -> bytes:
    if not value.startswith(expected_prefix):
        raise ValueError(f"invalid media url format: {value[:32]}")
    try:
        header, encoded = value.split(",", 1)
    except ValueError as exc:
        raise ValueError("invalid data URL") from exc
    if ";base64" not in header:
        raise ValueError("media url must be base64 encoded")
    try:
        return base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("media url contains invalid base64") from exc
