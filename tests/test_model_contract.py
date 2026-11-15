from dataclasses import asdict, replace
import unittest

from orbit.backend.llama_server import LlamaServerBackend
from orbit.native_llama.model_contract import ReasoningBehavior, TEXT_CAPABILITIES
from orbit.native_llama.model_profiles import NativeModelProfile
from orbit.qualification.model_result import ModelQualificationResult


def profile(**changes):
    return replace(NativeModelProfile(
        profile_id="test", family="test", model_name="test", architecture="test",
        renderer="llama.cpp-jinja", reasoning_protocol="other-protocol",
        tool_call_protocol="test", history_serialization="preserve", verified=True,
        failure_reason=None, template_source="test", template_sha256="a" * 64,
        thinking_supported=True, mtp_supported=False, gemma_prefix_reuse_supported=False,
        supported_capabilities=frozenset({"chat"}),
        reasoning_behavior=ReasoningBehavior.OPTIONAL, separated_reasoning=True,
    ), **changes)


class ModelContractTests(unittest.TestCase):
    def test_identity_verification_does_not_grant_other_capabilities(self):
        diagnostics = profile().diagnostics(thinking_enabled=False)
        self.assertTrue(diagnostics["verified"])
        self.assertTrue(diagnostics["capabilities"]["chat"])
        self.assertFalse(diagnostics["capabilities"]["tools"])
        self.assertFalse(diagnostics["capabilities"]["full_document_analysis"])
        self.assertEqual(diagnostics["qualification"]["status"], "reported_separately")
        self.assertFalse(diagnostics["qualification"]["automatic_promotion"])

    def test_unverified_profile_grants_no_capability(self):
        diagnostics = profile(verified=False, supported_capabilities=TEXT_CAPABILITIES).diagnostics(thinking_enabled=True)
        self.assertFalse(any(diagnostics["capabilities"].values()))

    def test_always_thinking_reports_request_and_effect_separately(self):
        diagnostics = profile(reasoning_behavior=ReasoningBehavior.ALWAYS).diagnostics(thinking_enabled=False)
        self.assertFalse(diagnostics["thinking_requested"])
        self.assertTrue(diagnostics["thinking_enabled"])

    def test_optional_and_absent_reasoning(self):
        self.assertFalse(profile().diagnostics(thinking_enabled=False)["thinking_enabled"])
        self.assertTrue(profile().diagnostics(thinking_enabled=True)["thinking_enabled"])
        absent = profile(thinking_supported=False, reasoning_behavior=ReasoningBehavior.NONE)
        self.assertFalse(absent.diagnostics(thinking_enabled=True)["thinking_enabled"])

    def test_unspecified_support_does_not_grant_operations(self):
        values = asdict(profile(reasoning_behavior=None))
        values.pop("supported_capabilities")
        restricted = NativeModelProfile(**values)
        self.assertFalse(any(restricted.diagnostics(thinking_enabled=False)["capabilities"][k] for k in TEXT_CAPABILITIES))

    def test_invalid_contract_cannot_silently_become_supported(self):
        for changes in ({"supported_capabilities": frozenset({"invented"})},
                        {"reasoning_behavior": "always"},
                        {"reasoning_behavior": ReasoningBehavior.NONE},
                        {"separated_reasoning": "true"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                profile(**changes)

    def test_separated_reasoning_is_not_bound_to_qwen_name(self):
        backend = LlamaServerBackend(base_url="http://unused", timeout=1)
        backend._props_cache = {"model_compatibility": profile().diagnostics(thinking_enabled=True)}
        self.assertTrue(backend.uses_native_separated_reasoning())
        backend._props_cache["model_compatibility"]["separated_reasoning"] = False
        backend._props_cache["model_compatibility"]["reasoning_protocol"] = "qwen-think"
        self.assertFalse(backend.uses_native_separated_reasoning())

    def test_reasoning_capability_requires_verified_boolean(self):
        backend = LlamaServerBackend(base_url="http://unused", timeout=1)
        for verified, separated in ((False, True), (True, "true"), (True, 1)):
            backend._props_cache = {"model_compatibility": {"verified": verified, "separated_reasoning": separated}}
            self.assertFalse(backend.uses_native_separated_reasoning())


class QualificationRecordTests(unittest.TestCase):
    def record(self, **changes):
        values = dict(artifact_sha256="a"*64, backend_sha256="b"*64,
                      configuration_sha256="c"*64, capability="chat", status="MANUAL_CHECK",
                      scope="Retained responses need semantic review", evidence_sha256="d"*64)
        return ModelQualificationResult(**(values | changes))

    def test_result_binds_artifact_backend_configuration_and_evidence(self):
        result = self.record()
        for field in ("artifact_sha256", "backend_sha256", "configuration_sha256", "evidence_sha256"):
            self.assertNotEqual(result, self.record(**{field: "e"*64}))
        self.assertEqual(result.as_dict()["status"], "MANUAL_CHECK")

    def test_pass_without_evidence_and_empty_scope_are_refused(self):
        for changes in ({"status": "PASS", "evidence_sha256": None}, {"scope": ""},
                        {"capability": "all"}, {"artifact_sha256": "missing"},
                        {"status": "SUCCESS"}, {"status": "NOT_RUN"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.record(**changes)

    def test_not_run_has_no_measured_evidence(self):
        self.assertIsNone(self.record(status="NOT_RUN", evidence_sha256=None).evidence_sha256)
