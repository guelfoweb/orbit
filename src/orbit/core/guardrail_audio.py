from __future__ import annotations

from dataclasses import dataclass
import base64
import shutil
import subprocess
import tempfile
from pathlib import Path
import re

from ..tooling.common import ToolError, resolve_path


AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
AUDIO_REQUEST_HINTS = (
    "audio",
    "voice",
    "recording",
    "transcribe",
    "transcription",
    "what is said",
    "what does it say",
    "listen",
    "speech",
    "summarize audio",
    "voce",
    "registrazione",
    "trascrivi",
    "trascrizione",
    "cosa dice",
    "ascolta",
    "parlato",
    "riassumi audio",
)
MAX_AUDIO_SOURCE_BYTES = 25 * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = 60.0
AUDIO_CHUNK_SECONDS = 5.0
MAX_AUDIO_CHUNKS = 12

QUOTED_AUDIO_PATH_RE = re.compile(
    r"(?P<quote>[\"'`])(?P<path>[^\"'`]+\.(?:wav|mp3|m4a|flac|ogg))(?P=quote)",
    re.IGNORECASE,
)
AUDIO_PATH_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./\\-]+\.(?:wav|mp3|m4a|flac|ogg))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExplicitAudioRequest:
    path: str
    full_path: Path


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start_seconds: float
    duration_seconds: float
    payload_base64: str


def resolve_explicit_audio_requests(*, user_input: str, workdir: Path) -> list[ExplicitAudioRequest]:
    lowered = user_input.lower()
    if not any(hint in lowered for hint in AUDIO_REQUEST_HINTS):
        return []
    requests: list[ExplicitAudioRequest] = []
    seen_paths: set[str] = set()
    for raw_path in extract_explicit_audio_paths(user_input):
        full_path = resolve_path(workdir, raw_path)
        normalized = full_path.relative_to(workdir).as_posix()
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        if not full_path.is_file():
            raise ToolError(f"audio not found: {raw_path}")
        if full_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ToolError(f"unsupported audio format: {raw_path}")
        if full_path.stat().st_size > MAX_AUDIO_SOURCE_BYTES:
            raise ToolError(
                f"audio is too large: {raw_path} "
                f"({full_path.stat().st_size} bytes > {MAX_AUDIO_SOURCE_BYTES} byte limit)"
            )
        requests.append(ExplicitAudioRequest(path=normalized, full_path=full_path))
    return requests


def extract_explicit_audio_paths(user_input: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for quoted in QUOTED_AUDIO_PATH_RE.finditer(user_input):
        candidate = quoted.group("path").strip()
        if candidate:
            found.append((quoted.start("path"), candidate))
    for match in AUDIO_PATH_RE.finditer(user_input):
        candidate = match.group("path").strip().strip(".,:;!?)]}")
        if candidate.lower().endswith(AUDIO_EXTENSIONS):
            found.append((match.start("path"), candidate))
    matches: list[str] = []
    seen: set[str] = set()
    for _position, candidate in sorted(found, key=lambda item: item[0]):
        if candidate in seen:
            continue
        seen.add(candidate)
        matches.append(candidate)
    return matches


def prepare_audio_chunks(full_path: Path, *, chunk_seconds: float = AUDIO_CHUNK_SECONDS) -> list[AudioChunk]:
    if shutil.which("ffmpeg") is None:
        raise ToolError("audio inspection requires ffmpeg to normalize local audio files")
    if chunk_seconds <= 0:
        raise ToolError("audio chunk duration must be positive")
    with tempfile.TemporaryDirectory(prefix="orbit-audio-") as tmpdir:
        normalized_path = Path(tmpdir) / "audio-16k-mono.wav"
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(full_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                str(normalized_path),
            ]
        )
        duration = _audio_duration_seconds(normalized_path)
        if duration <= 0:
            raise ToolError(f"audio has no duration: {full_path.name}")
        if duration > MAX_AUDIO_DURATION_SECONDS:
            raise ToolError(
                f"audio is too long: {full_path.name} "
                f"({duration:.1f}s > {MAX_AUDIO_DURATION_SECONDS:.0f}s limit)"
            )
        chunks: list[AudioChunk] = []
        start = 0.0
        index = 1
        while start < duration and index <= MAX_AUDIO_CHUNKS:
            current_duration = min(chunk_seconds, duration - start)
            chunk_path = Path(tmpdir) / f"chunk-{index:03d}.wav"
            _run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-ss",
                    f"{start:.3f}",
                    "-t",
                    f"{current_duration:.3f}",
                    "-i",
                    str(normalized_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-sample_fmt",
                    "s16",
                    str(chunk_path),
                ]
            )
            payload = chunk_path.read_bytes()
            if payload:
                chunks.append(
                    AudioChunk(
                        index=index,
                        start_seconds=start,
                        duration_seconds=current_duration,
                        payload_base64=base64.b64encode(payload).decode("ascii"),
                    )
                )
            start += chunk_seconds
            index += 1
        if not chunks:
            raise ToolError(f"audio produced no readable chunks: {full_path.name}")
        return chunks


def _audio_duration_seconds(path: Path) -> float:
    if shutil.which("ffprobe") is None:
        raise ToolError("audio inspection requires ffprobe to inspect local audio files")
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    if proc.returncode != 0:
        raise ToolError(f"failed to inspect audio duration: {proc.stderr.strip() or proc.stdout.strip()}")
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise ToolError("failed to parse audio duration") from exc


def _run_ffmpeg(command: list[str]) -> None:
    proc = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
    if proc.returncode != 0:
        raise ToolError(f"ffmpeg failed: {proc.stderr.strip() or proc.stdout.strip()}")
