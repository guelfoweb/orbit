from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.model_names import resolve_model_display_name


class ModelNameTests(unittest.TestCase):
    def test_resolve_model_display_name_from_ollama_manifest(self) -> None:
        digest = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "manifests"
            manifest = root / "registry.ollama.ai" / "library" / "gemma4" / "12b"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}"}]}),
                encoding="utf-8",
            )

            name = resolve_model_display_name(f"sha256-{digest}", manifest_roots=[root])

        self.assertEqual(name, "gemma4:12b")

    def test_resolve_model_display_name_prefers_base_manifest_over_derived_from_layer(self) -> None:
        digest = "b" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "manifests"
            derived = root / "registry.ollama.ai" / "library" / "gemma4" / "12b-c4k"
            base = root / "registry.ollama.ai" / "library" / "gemma4" / "12b"
            derived.parent.mkdir(parents=True)
            derived.write_text(json.dumps({"layers": [{"digest": f"sha256:{digest}", "from": "gemma4:12b"}]}), encoding="utf-8")
            base.write_text(json.dumps({"layers": [{"digest": f"sha256:{digest}"}]}), encoding="utf-8")

            name = resolve_model_display_name(f"sha256-{digest}", manifest_roots=[root])

        self.assertEqual(name, "gemma4:12b")


if __name__ == "__main__":
    unittest.main()
