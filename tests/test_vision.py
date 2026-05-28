from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.core.guardrail_vision import MAX_IMAGE_EDGE, _normalized_image_bytes


class VisionTests(unittest.TestCase):
    def test_normalized_image_bytes_flattens_alpha_to_rgb_png(self) -> None:
        with tempfile.TemporaryDirectory(prefix="orbit-vision-") as tmpdir:
            path = Path(tmpdir) / "alpha.png"
            image = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
            image.save(path, format="PNG")

            payload = _normalized_image_bytes(path)
            normalized = Image.open(BytesIO(payload))

            self.assertEqual(normalized.mode, "RGB")
            self.assertEqual(normalized.size, (4, 4))

    def test_normalized_image_bytes_resizes_large_images(self) -> None:
        with tempfile.TemporaryDirectory(prefix="orbit-vision-") as tmpdir:
            path = Path(tmpdir) / "large.png"
            image = Image.new("RGB", (2200, 1400), (0, 0, 255))
            image.save(path, format="PNG")

            payload = _normalized_image_bytes(path)
            normalized = Image.open(BytesIO(payload))

            self.assertLessEqual(max(normalized.size), MAX_IMAGE_EDGE)

