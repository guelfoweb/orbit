from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.media import load_audio, load_image, load_referenced_media


class MediaTests(unittest.TestCase):
    def test_load_image_builds_data_url(self) -> None:
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mP8z8AARQAFFwH+"
            "qH8nNwAAAABJRU5ErkJggg=="
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.png"
            path.write_bytes(png_bytes)

            image = load_image(str(path))

        self.assertEqual(image.mime_type, "image/png")
        self.assertTrue(image.data_url.startswith("data:image/png;base64,"))

    def test_load_image_rejects_missing_file(self) -> None:
        with self.assertRaises(ValueError):
            load_image("/tmp/does-not-exist.png")

    def test_load_audio_builds_base64_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.wav"
            path.write_bytes(b"RIFFxxxxWAVE")

            audio = load_audio(str(path))

        self.assertEqual(audio.format, "wav")
        self.assertTrue(audio.data)

    def test_load_audio_rejects_unsupported_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.flac"
            path.write_bytes(b"not really flac")

            with self.assertRaises(ValueError):
                load_audio(str(path))

    def test_load_referenced_media_finds_workdir_image(self) -> None:
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mP8z8AARQAFFwH+"
            "qH8nNwAAAABJRU5ErkJggg=="
        )
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "images").mkdir()
            (workdir / "images" / "tiny.png").write_bytes(png_bytes)

            images, audios = load_referenced_media("describe images/tiny.png", workdir=workdir)

        self.assertEqual(len(images), 1)
        self.assertEqual(audios, [])

    def test_load_referenced_media_allows_trailing_sentence_punctuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "voice.wav").write_bytes(b"RIFFxxxxWAVE")

            images, audios = load_referenced_media("transcribe voice.wav.", workdir=workdir)

        self.assertEqual(images, [])
        self.assertEqual(len(audios), 1)

    def test_load_referenced_media_ignores_paths_outside_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            outside = Path(tmp) / "outside.png"
            workdir.mkdir()
            outside.write_bytes(b"not a real png")

            images, audios = load_referenced_media(f"describe {outside}", workdir=workdir)

        self.assertEqual(images, [])
        self.assertEqual(audios, [])

    def test_load_referenced_media_ignores_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workdir = root / "work"
            outside = root / "outside.png"
            workdir.mkdir()
            outside.write_bytes(b"not a real png")
            link = workdir / "outside.png"
            try:
                link.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink not available: {exc}")

            images, audios = load_referenced_media("describe outside.png", workdir=workdir)

        self.assertEqual(images, [])
        self.assertEqual(audios, [])


if __name__ == "__main__":
    unittest.main()
