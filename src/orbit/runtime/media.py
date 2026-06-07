from __future__ import annotations

import base64
import mimetypes
import shlex
from dataclasses import dataclass
from pathlib import Path


MAX_IMAGE_BYTES = 8 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


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
SUPPORTED_AUDIO_SUFFIXES = {".mp3", ".wav"}


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


def load_referenced_media(prompt: str, *, workdir: Path) -> tuple[list[ImageInput], list[AudioInput]]:
    images: list[ImageInput] = []
    audios: list[AudioInput] = []
    seen: set[Path] = set()
    for candidate in _media_path_candidates(prompt):
        path = _resolve_inside_workdir(candidate, workdir=workdir)
        if path is None or path in seen or not path.is_file():
            continue
        seen.add(path)
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            images.append(load_image(str(path)))
        elif suffix in SUPPORTED_AUDIO_SUFFIXES:
            audios.append(load_audio(str(path)))
    return images, audios


def _media_path_candidates(prompt: str) -> list[str]:
    try:
        tokens = shlex.split(prompt)
    except ValueError:
        tokens = prompt.split()
    candidates: list[str] = []
    for token in tokens:
        cleaned = token.strip(" \t\r\n.,;:()[]{}<>\"'")
        suffix = Path(cleaned).suffix.lower()
        if suffix in SUPPORTED_IMAGE_SUFFIXES or suffix in SUPPORTED_AUDIO_SUFFIXES:
            candidates.append(cleaned)
    return candidates


def _resolve_inside_workdir(path: str, *, workdir: Path) -> Path | None:
    if not path:
        return None
    root = workdir.expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved
