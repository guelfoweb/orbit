from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.model_contract import ReasoningBehavior
from orbit.native_llama.model_profiles import detect_native_model_profile
from orbit.native_llama.research_admission import ResearchAdmission


def gguf(metadata):
    def string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded
    return (struct.pack("<IIQQ", 0x46554747, 3, 0, len(metadata))
            + b"".join(string(k) + struct.pack("<I", 8) + string(v) for k, v in metadata.items()))


class ResearchAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.template = "<|im_start|>assistant\n<think>"
        self.metadata = {"general.name": "Candidate", "general.architecture": "test",
                         "general.file_type": "15"}
        self.path = self.root / "candidate.gguf"
        self.path.write_bytes(gguf(self.metadata | {"tokenizer.chat_template": self.template}))
        self.admission = ResearchAdmission(
            hashlib.sha256(self.path.read_bytes()).hexdigest(), self.path.stat().st_size,
            hashlib.sha256(self.template.encode()).hexdigest(), "Candidate", "test",
            tuple(sorted(self.metadata.items())), ReasoningBehavior.ALWAYS, True,
        )
        self.manifest = self.root / "identity.json"
        self.value = asdict(self.admission)
        self.value.update(version=1, metadata=self.metadata)
        self.manifest.write_text(json.dumps(self.value))

    def test_exact_manifest_attests_and_admits_without_production_enablement(self):
        admission = ResearchAdmission.read(self.manifest)
        attestation = admission.attest(self.path)
        profile = admission.profile(self.path, self.template, attestation)
        self.assertFalse(profile.verified)
        self.assertEqual(profile.profile_id, "research-" + admission.artifact_sha256)
        self.assertTrue(profile.uses_native_chat_bridge)
        self.assertFalse(any(profile.diagnostics(thinking_enabled=False)["capabilities"].values()))
        self.assertEqual(profile.effective_reasoning_behavior, ReasoningBehavior.ALWAYS)
        self.assertFalse(detect_native_model_profile(self.metadata, self.template).verified)

    def test_bad_digest_rejected_even_with_correct_name_and_size(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            replace(self.admission, artifact_sha256="0"*64).attest(self.path)

    def test_bad_size_rejected_before_reading_weights(self):
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            replace(self.admission, artifact_size=self.path.stat().st_size+1).attest(self.path)

    def test_metadata_and_template_must_match_independently_of_file_digest(self):
        for admission in (replace(self.admission, template_sha256="0"*64),
                          replace(self.admission, metadata=tuple(sorted(self.metadata.items())) + (("extra", "missing"),))):
            with self.subTest(admission=admission), self.assertRaisesRegex(ValueError, "mismatch"):
                admission.attest(self.path)

    def test_missing_metadata_does_not_match_the_literal_none(self):
        admission = replace(self.admission, metadata=self.admission.metadata + (("missing", "None"),))
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            admission.attest(self.path)

    def test_change_after_attestation_and_native_template_drift_are_rejected(self):
        attestation = self.admission.attest(self.path)
        with self.assertRaisesRegex(ValueError, "template mismatch"):
            self.admission.profile(self.path, self.template + "changed", attestation)
        self.path.write_bytes(self.path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.admission.profile(self.path, self.template, attestation)

    def test_identity_is_content_bound_not_filename_bound(self):
        renamed = self.root / "another.gguf"
        self.path.rename(renamed)
        attestation = self.admission.attest(renamed)
        self.assertFalse(self.admission.profile(renamed, self.template, attestation).verified)

    def test_split_file_not_admitted_as_single_artifact(self):
        target = self.root / "candidate-00001-of-00002.gguf"
        self.path.rename(target)
        with self.assertRaisesRegex(ValueError, "single-file"):
            self.admission.attest(target)

    def test_manifest_rejects_unknown_fields_types_and_unbound_name(self):
        for update in ({"version": True}, {"automatic_promotion": True},
                       {"artifact_size": True}, {"reasoning_behavior": "guess"},
                       {"separated_reasoning": "true"}, {"model_name": "Impostor"},
                       {"metadata": {"general.name": "Candidate"}},
                       {"metadata": {"general.name": "Candidate", "general.architecture": "test", "count": 40}}):
            with self.subTest(update=update):
                self.manifest.write_text(json.dumps(self.value | update))
                with self.assertRaises(ValueError):
                    ResearchAdmission.read(self.manifest)

    def test_manifest_rejects_duplicate_keys_and_oversized_input(self):
        for text in ('{"version":1,"version":1}', ' ' * 16385):
            self.manifest.write_text(text)
            with self.assertRaises(ValueError):
                ResearchAdmission.read(self.manifest)

    def test_failed_attestation_prevents_native_load(self):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._research_admission = replace(self.admission, artifact_sha256="0"*64)
        client.paths = SimpleNamespace(model=self.path)
        client.lib = Mock()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            client.load()
        client.lib.assert_not_called()
        self.assertEqual(client.lib.mock_calls, [])

    def test_experimental_mtp_and_gpu_cannot_be_enabled(self):
        for config in (NativeClientConfig(gpu_layers=1), NativeClientConfig(use_mtp_experimental=True),
                       NativeClientConfig(mtp_probe_enabled=True), NativeClientConfig(low_memory=True)):
            with self.subTest(config=config), patch("orbit.native_llama.client.LlamaLibrary") as library:
                with self.assertRaisesRegex(ValueError, "CPU-only"):
                    NativeLlamaClient(SimpleNamespace(model=self.path), config, research_admission=self.admission)
                library.assert_not_called()

    def test_unverified_default_still_fails_before_bridge_creation(self):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._model = object()
        client.lib = SimpleNamespace(lib=SimpleNamespace(llama_model_chat_template=lambda *_: self.template.encode()))
        client._read_model_metadata = lambda: self.metadata
        with patch("orbit.native_llama.client.ChatBridgeLibrary") as bridge:
            with self.assertRaisesRegex(RuntimeError, "unsupported or unverified"):
                client._initialize_model_profile()
            bridge.assert_not_called()

    def test_explicit_admission_reuses_the_native_bridge_and_keeps_unverified(self):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._model = object()
        client.paths = SimpleNamespace(model=self.path, build_bin=self.root)
        client.config = NativeClientConfig()
        client._research_admission = self.admission
        client._research_attestation = self.admission.attest(self.path)
        client.lib = SimpleNamespace(lib=SimpleNamespace(llama_model_chat_template=lambda *_: self.template.encode()))
        client._read_model_metadata = lambda: self.metadata
        with patch("orbit.native_llama.client._resolve_chat_bridge_path", return_value=self.root / "bridge"), \
             patch("orbit.native_llama.client.ChatBridgeLibrary") as bridge:
            client._initialize_model_profile()
            bridge.return_value.create.assert_called_once_with(client._model)
        self.assertFalse(client.model_profile.verified)
