from __future__ import annotations

from dataclasses import dataclass
import base64
from io import BytesIO
from pathlib import Path
import re

from PIL import Image

from ..tooling.common import ToolError, resolve_path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
IMAGE_REQUEST_HINTS = (
    "describe",
    "descrivi",
    "analyze",
    "analyse",
    "analizza",
    "compare",
    "confronta",
    "differences",
    "difference",
    "diff",
    "differenze",
    "inspect",
    "ispeziona",
    "look at",
    "guarda",
    "summarize",
    "riassumi",
    "read",
    "leggi",
    "read text",
    "extract text",
    "ocr",
    "what is in",
    "what's in",
    "what is shown",
    "cosa c'è",
    "cosa mostra",
)
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_EDGE = 1024

QUOTED_IMAGE_PATH_RE = re.compile(
    r"(?P<quote>[\"'`])(?P<path>[^\"'`]+\.(?:png|jpe?g|webp|bmp|gif))(?P=quote)",
    re.IGNORECASE,
)
IMAGE_PATH_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./\\-]+\.(?:png|jpe?g|webp|bmp|gif))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExplicitImageRequest:
    path: str
    full_path: Path


def resolve_explicit_image_requests(*, user_input: str, workdir: Path) -> list[ExplicitImageRequest]:
    lowered = user_input.lower()
    if not any(hint in lowered for hint in IMAGE_REQUEST_HINTS):
        return []
    requests: list[ExplicitImageRequest] = []
    seen_paths: set[str] = set()
    for raw_path in extract_explicit_image_paths(user_input):
        full_path = resolve_path(workdir, raw_path)
        normalized = full_path.relative_to(workdir).as_posix()
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        if not full_path.is_file():
            raise ToolError(f"image not found: {raw_path}")
        if full_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ToolError(f"unsupported image format: {raw_path}")
        requests.append(ExplicitImageRequest(path=normalized, full_path=full_path))
    return requests


def extract_explicit_image_path(user_input: str) -> str | None:
    paths = extract_explicit_image_paths(user_input)
    if not paths:
        return None
    return paths[0]


def extract_explicit_image_paths(user_input: str) -> list[str]:
    matches: list[str] = []
    seen: set[str] = set()
    for quoted in QUOTED_IMAGE_PATH_RE.finditer(user_input):
        candidate = quoted.group("path").strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            matches.append(candidate)
    for match in IMAGE_PATH_RE.finditer(user_input):
        candidate = match.group("path").strip().strip(".,:;!?)]}")
        if candidate.lower().endswith(IMAGE_EXTENSIONS) and candidate not in seen:
            seen.add(candidate)
            matches.append(candidate)
    return matches


def encode_image_base64(full_path: Path) -> str:
    payload = _normalized_image_bytes(full_path)
    if not payload:
        raise ToolError(f"image is empty: {full_path.name}")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ToolError(
            f"image is too large: {full_path.name} "
            f"({len(payload)} bytes > {MAX_IMAGE_BYTES} byte limit)"
        )
    return base64.b64encode(payload).decode("ascii")


def _normalized_image_bytes(full_path: Path) -> bytes:
    try:
        with Image.open(full_path) as image:
            normalized = _normalize_image(image)
            buffer = BytesIO()
            normalized.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()
    except OSError as exc:
        raise ToolError(f"unsupported or unreadable image: {full_path.name}") from exc


def _normalize_image(image: Image.Image) -> Image.Image:
    prepared = image.convert("RGBA")
    background = Image.new("RGBA", prepared.size, (255, 255, 255, 255))
    flattened = Image.alpha_composite(background, prepared).convert("RGB")
    if max(flattened.size) <= MAX_IMAGE_EDGE:
        return flattened
    resized = flattened.copy()
    resized.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    return resized
