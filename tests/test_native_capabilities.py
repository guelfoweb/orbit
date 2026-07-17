from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from orbit.native_llama.capabilities import (
    GEMMA4_FINAL_PREFIX_TEXT_HASH,
    GEMMA4_FINAL_PREFIX_TOKEN_HASH,
    GEMMA4_RENDERER_FIXTURE_HASHES,
    GEMMA4_RENDERER_FIXTURE_SUITE_HASH,
    GEMMA4_TOOL_PROTOCOL_TEXT_HASH,
    LlamaCppBuildInfo,
    _hash_token_ids,
    _hash_named_hashes,
    _render_renderer_fixtures,
    _render_tool_protocol_fixture,
    build_gemma4_capability_manifest,
    read_llama_cpp_build_info,
    safe_gemma4_capability_manifest,
)
from orbit.native_llama.chat_template import render_gemma4_route_prompt_segments
from orbit.runtime.messages import FINAL_FROM_TOOL_SYSTEM_PROMPT


FINAL_PREFIX_TOKENS = [
    2, 105, 9731, 107, 7925, 506, 2864, 214219, 699, 5904, 4914, 236761, 23097, 1186, 506, 48037,
    4133, 3890, 236764, 44257, 4453, 4889, 1056, 4354, 236761, 3574, 711, 10149, 10797, 236764,
    2246, 6436, 236764, 29910, 10445, 5904, 236772, 6639, 33413, 236764, 653, 3539, 6220, 529,
    2802, 1056, 4914, 7519, 236761, 98936, 56124, 2561, 532, 2072, 9825, 21485, 236761, 8006,
    1308, 506, 3890, 236761, 106, 107,
]


class _ConformantClient:
    paths = SimpleNamespace(build_bin=Path("/native/lib"))

    def tokenize(self, prompt: str) -> list[int]:
        segments = render_gemma4_route_prompt_segments(
            [
                {"role": "system", "content": FINAL_FROM_TOOL_SYSTEM_PROMPT},
                {"role": "user", "content": "capability request"},
                {"role": "system", "content": "bounded capability evidence"},
            ],
            thinking=False,
        )
        if prompt == segments.stable_prefix_text:
            return list(FINAL_PREFIX_TOKENS)
        if prompt == segments.full_prompt_text:
            return [*FINAL_PREFIX_TOKENS, 105, 999]
        raise AssertionError("unexpected capability prompt")


class NativeCapabilityTests(unittest.TestCase):
    def test_tool_protocol_fixture_hash_is_stable(self) -> None:
        import hashlib

        actual = hashlib.sha256(_render_tool_protocol_fixture().encode("utf-8")).hexdigest()

        self.assertEqual(actual, GEMMA4_TOOL_PROTOCOL_TEXT_HASH)

    def test_renderer_fixture_suite_hashes_are_stable(self) -> None:
        import hashlib

        actual = {
            name: hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            for name, rendered in _render_renderer_fixtures().items()
        }

        self.assertEqual(actual, GEMMA4_RENDERER_FIXTURE_HASHES)
        self.assertEqual(_hash_named_hashes(actual), GEMMA4_RENDERER_FIXTURE_SUITE_HASH)

    def test_final_prefix_token_serialization_matches_production_probe(self) -> None:
        self.assertEqual(len(FINAL_PREFIX_TOKENS), 64)
        self.assertEqual(_hash_token_ids(FINAL_PREFIX_TOKENS), GEMMA4_FINAL_PREFIX_TOKEN_HASH)

    @mock.patch("orbit.native_llama.capabilities.read_llama_cpp_build_info")
    def test_manifest_is_verified_only_when_build_renderer_and_tokenizer_match(self, build_info) -> None:
        build_info.return_value = LlamaCppBuildInfo(
            build_number=278,
            commit="6f79e02",
            target="Linux x86_64",
            compiler="GNU 13",
            library_hash="a" * 64,
            source="runtime_symbols",
        )

        manifest = build_gemma4_capability_manifest(
            _ConformantClient(),
            final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT,
        )

        self.assertEqual(manifest["status"], "verified")
        self.assertEqual(manifest["verification_scope"], "backend_identity_and_prompt_conformance")
        self.assertFalse(manifest["behavior_enforced"])
        self.assertTrue(manifest["renderer"]["conformant"])
        self.assertEqual(manifest["renderer"]["fixture_hashes"], GEMMA4_RENDERER_FIXTURE_HASHES)
        self.assertEqual(manifest["renderer"]["fixture_suite_hash"], GEMMA4_RENDERER_FIXTURE_SUITE_HASH)
        self.assertEqual(manifest["renderer"]["final_prefix_text_hash"], GEMMA4_FINAL_PREFIX_TEXT_HASH)
        self.assertTrue(manifest["tokenizer"]["conformant"])
        self.assertEqual(manifest["tokenizer"]["prefix_tokens"], 64)
        self.assertEqual(manifest["tokenizer"]["next_dynamic_token"], 105)
        self.assertTrue(manifest["backend"]["verified_commit"])

    @mock.patch("orbit.native_llama.capabilities.read_llama_cpp_build_info")
    def test_tokenizer_mismatch_is_reported_without_enforcement(self, build_info) -> None:
        build_info.return_value = LlamaCppBuildInfo(278, "6f79e02", None, None, None, "runtime_symbols")
        client = _ConformantClient()
        client.tokenize = lambda _prompt: [2, 105]  # type: ignore[method-assign]

        manifest = build_gemma4_capability_manifest(client, final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT)

        self.assertEqual(manifest["status"], "tokenizer_mismatch")
        self.assertFalse(manifest["tokenizer"]["conformant"])
        self.assertFalse(manifest["behavior_enforced"])

    @mock.patch("orbit.native_llama.capabilities._render_renderer_fixtures")
    @mock.patch("orbit.native_llama.capabilities.read_llama_cpp_build_info")
    def test_renderer_fixture_drift_is_reported_without_enforcement(self, build_info, render_fixtures) -> None:
        build_info.return_value = LlamaCppBuildInfo(278, "6f79e02", None, None, None, "runtime_symbols")
        render_fixtures.return_value = {
            **_render_renderer_fixtures(),
            "tool_generation": "drifted renderer output",
        }

        manifest = build_gemma4_capability_manifest(
            _ConformantClient(),
            final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT,
        )

        self.assertEqual(manifest["status"], "renderer_mismatch")
        self.assertFalse(manifest["renderer"]["conformant"])
        self.assertFalse(manifest["behavior_enforced"])

    @mock.patch("orbit.native_llama.capabilities.read_llama_cpp_build_info")
    def test_unavailable_tokenizer_cannot_break_manifest_generation(self, build_info) -> None:
        build_info.return_value = LlamaCppBuildInfo(278, "6f79e02", None, None, None, "runtime_symbols")
        client = _ConformantClient()
        client.tokenize = lambda _prompt: (_ for _ in ()).throw(RuntimeError("not loaded"))  # type: ignore[method-assign]

        manifest = build_gemma4_capability_manifest(client, final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT)

        self.assertEqual(manifest["status"], "tokenizer_unavailable")
        self.assertEqual(manifest["tokenizer"]["error"], "tokenizer_probe_unavailable")
        self.assertFalse(manifest["behavior_enforced"])

    @mock.patch("orbit.native_llama.capabilities.read_llama_cpp_build_info")
    def test_unknown_backend_commit_is_observational_not_a_runtime_failure(self, build_info) -> None:
        build_info.return_value = LlamaCppBuildInfo(279, "unknown-new", None, None, None, "runtime_symbols")

        manifest = build_gemma4_capability_manifest(
            _ConformantClient(),
            final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT,
        )

        self.assertEqual(manifest["status"], "backend_unverified")
        self.assertFalse(manifest["backend"]["verified_commit"])
        self.assertFalse(manifest["behavior_enforced"])

    @mock.patch("orbit.native_llama.capabilities._sha256_file", return_value="b" * 64)
    @mock.patch("orbit.native_llama.capabilities._read_text_symbol")
    @mock.patch("orbit.native_llama.capabilities._read_int_symbol", return_value=278)
    @mock.patch("orbit.native_llama.capabilities.load_native_cdll", return_value=object())
    def test_reads_bounded_runtime_build_symbols(self, _load, _number, text_symbol, _file_hash) -> None:
        text_symbol.side_effect = ["6f79e02", "Linux x86_64", "GNU 13"]
        with mock.patch.object(Path, "is_file", return_value=True):
            info = read_llama_cpp_build_info(Path("/native/lib"))

        self.assertEqual(info.build_number, 278)
        self.assertEqual(info.commit, "6f79e02")
        self.assertEqual(info.target, "Linux x86_64")
        self.assertEqual(info.compiler, "GNU 13")
        self.assertEqual(info.library_hash, "b" * 64)
        self.assertEqual(info.source, "runtime_symbols")

    def test_missing_runtime_library_reports_bounded_unavailable_state(self) -> None:
        with mock.patch.object(Path, "is_file", return_value=False):
            info = read_llama_cpp_build_info(Path("/missing"))

        self.assertEqual(info.source, "unavailable")
        self.assertEqual(info.error, "llama_common_unavailable")
        self.assertIsNone(info.commit)

    @mock.patch("orbit.native_llama.capabilities.build_gemma4_capability_manifest", side_effect=RuntimeError("diagnostic failure"))
    def test_safe_manifest_cannot_break_server_startup(self, _build_manifest) -> None:
        manifest = safe_gemma4_capability_manifest(
            _ConformantClient(),
            final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT,
        )

        self.assertEqual(manifest["status"], "manifest_unavailable")
        self.assertEqual(manifest["error"], "manifest_failed:RuntimeError")
        self.assertFalse(manifest["behavior_enforced"])


if __name__ == "__main__":
    unittest.main()
