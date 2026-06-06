from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path


MAX_IMAGE_BYTES = 8 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@dataclass(frozen=True)
class ImageInput:
    path: Path
    mime_type: str
    data_url: str


@dataclass(frozen=True)
class AudioInput:
    path: Path
    format: str
    data: str


MAX_AUDIO_BYTES = 8 * 1024 * 1024
SUPPORTED_AUDIO_TYPES = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
}


def load_image(path: str) -> ImageInput:
    image_path = Path(path).expanduser().resolve()
    if not image_path.exists():
        raise ValueError(f"image not found: {path}")
    if not image_path.is_file():
        raise ValueError(f"image path is not a file: {path}")
    size = image_path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"image is too large: {path} ({size} bytes, max {MAX_IMAGE_BYTES})")

    mime_type = mimetypes.guess_type(str(image_path))[0]
    if mime_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(f"unsupported image type for {path}: {mime_type or 'unknown'}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return ImageInput(
        path=image_path,
        mime_type=mime_type,
        data_url=f"data:{mime_type};base64,{encoded}",
    )


def load_audio(path: str) -> AudioInput:
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.exists():
        raise ValueError(f"audio not found: {path}")
    if not audio_path.is_file():
        raise ValueError(f"audio path is not a file: {path}")
    size = audio_path.stat().st_size
    if size > MAX_AUDIO_BYTES:
        raise ValueError(f"audio is too large: {path} ({size} bytes, max {MAX_AUDIO_BYTES})")

    mime_type = mimetypes.guess_type(str(audio_path))[0]
    audio_format = SUPPORTED_AUDIO_TYPES.get(mime_type or "")
    if not audio_format:
        suffix = audio_path.suffix.lower().lstrip(".")
        audio_format = suffix if suffix in {"wav", "mp3"} else ""
    if not audio_format:
        raise ValueError(f"unsupported audio type for {path}: {mime_type or 'unknown'}")

    return AudioInput(
        path=audio_path,
        format=audio_format,
        data=base64.b64encode(audio_path.read_bytes()).decode("ascii"),
    )
