from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from orbit.native_llama.native_artifacts import LINUX_RUNTIME_LIBS, OPTIONAL_RUNTIME_LIBS, SHIM_ARTIFACTS


ROOT = Path(__file__).resolve().parents[1]


class NativePackagingMetadataTests(unittest.TestCase):
    def test_pyproject_includes_native_vendor_artifacts(self) -> None:
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        package_data = data["tool"]["setuptools"]["package-data"]["orbit.native_llama"]

        self.assertIn("vendor/lib/*", package_data)
        self.assertIn("vendor/shim/*", package_data)
        self.assertIn("vendor/source/llama.cpp/**/*", package_data)
        self.assertIn("vendor/THIRD_PARTY_NOTICES.md", package_data)
        self.assertIn("model_registry.json", package_data)

    def test_manifest_includes_native_vendor_tree(self) -> None:
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertIn("recursive-include src/orbit/native_llama/vendor *", manifest)

    def test_vendor_lib_readme_exists(self) -> None:
        readme = ROOT / "src/orbit/native_llama/vendor/lib/README.md"

        self.assertTrue(readme.exists())
        self.assertIn("packaged native runtime libraries", readme.read_text(encoding="utf-8"))

    def test_third_party_notices_exist(self) -> None:
        notices = ROOT / "src/orbit/native_llama/vendor/THIRD_PARTY_NOTICES.md"

        self.assertTrue(notices.exists())
        self.assertIn("llama.cpp", notices.read_text(encoding="utf-8"))

    def test_vendored_mtmd_model_sources_exist(self) -> None:
        mtmd_models = ROOT / "src/orbit/native_llama/vendor/source/llama.cpp/tools/mtmd/models"

        self.assertTrue(mtmd_models.exists())
        self.assertTrue((mtmd_models / "models.h").exists())
        self.assertGreaterEqual(len(list(mtmd_models.glob("*.cpp"))), 10)

    def test_native_artifact_contract_lists_expected_linux_files(self) -> None:
        self.assertIn("libllama.so", LINUX_RUNTIME_LIBS)
        self.assertIn("libggml-cpu.so", LINUX_RUNTIME_LIBS)
        self.assertIn("libmtmd.so", OPTIONAL_RUNTIME_LIBS)
        self.assertIn("liborbit-persistent-mtp.so", SHIM_ARTIFACTS)
        self.assertIn("orbit-mtp-probe", SHIM_ARTIFACTS)


if __name__ == "__main__":
    unittest.main()
