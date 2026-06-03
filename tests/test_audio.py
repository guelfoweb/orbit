from __future__ import annotations

from pathlib import Path
import math
import struct
import sys
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.core.guardrails.audio import (
    AUDIO_CHUNK_SECONDS,
    extract_explicit_audio_paths,
    prepare_audio_chunks,
    resolve_explicit_audio_requests,
)


class AudioTests(unittest.TestCase):
    def test_extract_explicit_audio_paths(self) -> None:
        paths = extract_explicit_audio_paths("transcribe audio/voice-sample.wav and `clips/test.mp3`")
        self.assertEqual(paths, ["audio/voice-sample.wav", "clips/test.mp3"])

    def test_resolve_explicit_audio_requests_requires_audio_intent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="orbit-audio-") as tmpdir:
            workdir = Path(tmpdir)
            path = workdir / "sample.wav"
            _write_wav(path, seconds=1.0)

            self.assertEqual(resolve_explicit_audio_requests(user_input="read sample.wav", workdir=workdir), [])
            requests = resolve_explicit_audio_requests(user_input="transcribe sample.wav", workdir=workdir)

            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].path, "sample.wav")

    def test_prepare_audio_chunks_normalizes_and_splits_wav(self) -> None:
        with tempfile.TemporaryDirectory(prefix="orbit-audio-") as tmpdir:
            path = Path(tmpdir) / "sample.wav"
            _write_wav(path, seconds=6.0, rate=44100)

            chunks = prepare_audio_chunks(path)

            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0].index, 1)
            self.assertAlmostEqual(chunks[0].duration_seconds, AUDIO_CHUNK_SECONDS)
            self.assertGreater(len(chunks[0].payload_base64), 100)
            self.assertEqual(chunks[1].index, 2)
            self.assertGreater(chunks[1].duration_seconds, 0.0)


def _write_wav(path: Path, *, seconds: float, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        frames = bytearray()
        for i in range(int(rate * seconds)):
            sample = int(0.2 * 32767 * math.sin(2 * math.pi * 440 * i / rate))
            frames += struct.pack("<h", sample)
        audio.writeframes(frames)


if __name__ == "__main__":
    unittest.main()
