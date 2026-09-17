"""Qwen3.8 Flash Next UD-IQ1_M production enablement (QWEN38-ORBIT-PRODUCTION-ENABLEMENT-19).

The contract these pin: exactly one (machine, model) pair -- the Dell Pro 5 14
P514260 running the registry model `qwen38-flash-next-ud-iq1-m`, CONFIRMED by
the GGUF's own verified identity -- resolves the research-qualified startup
profile (ctx 4096, threads 10/10, batch 256/128, one slot, no GPU layers,
repack OFF, MTP OFF, mmap + lazy tensor reads, no NextN tensors). Nothing else
inherits it, an explicit operator value always wins, and every existing model
keeps its Mission-18 behaviour.
"""
from __future__ import annotations

import hashlib
import io
import os
import pathlib
import sys
import tempfile
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_llama.bindings import (  # noqa: E402
    LLAMA_LAZY_MODE_OFF,
    LLAMA_LAZY_MODE_ON,
    LLAMA_LOAD_MODE_MMAP,
)
from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient  # noqa: E402
from orbit.native_llama.model_discovery import ModelDiscoveryRow, _is_secondary_split_shard  # noqa: E402
from orbit.native_llama.model_download import download_model  # noqa: E402
from orbit.native_llama.model_profiles import (  # noqa: E402
    GEMMA4_PROFILE_ID,
    ORNITH15_PROFILE_ID,
    QWEN36_PROFILE_ID,
    QWEN38_FLASH_NEXT_PROFILE_ID,
    QWEN38_OFFICIAL_TEMPLATE_SHA256,
    QWEN38_PROFILE_ID,
    QWEN3_CODER_PROFILE_ID,
    detect_native_model_profile,
    verified_native_model_identity,
)
from orbit.native_llama.model_registry import get_manifest, load_registry  # noqa: E402
from orbit.native_llama.paths import NativeLlamaPaths  # noqa: E402
from orbit.native_server import app as app_module  # noqa: E402
from tests.test_native_server_bootstrap import _FakeHTTPServer, _FakeNativeClient  # noqa: E402
from orbit.native_server.server_profile import (  # noqa: E402
    ENV_CPU_REPACK,
    ENV_THREADS,
    HostTopology,
    QUALIFIED_NOREPACK_MACHINE,
    QUALIFIED_NOREPACK_MODEL_ID,
    QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID,
    QualifiedStartupProfile,
    qualified_startup_profile,
    resolve_cpu_repack,
    resolve_profile,
)

GIB = 1024
DELL = HostTopology(
    cpu_model="Intel(R) Core(TM) Ultra 7 366H",
    physical_cores=16, logical_cpus=16,
    total_ram_mib=30 * GIB, available_ram_mib=26 * GIB, swap_total_mib=8 * GIB,
    machine_model=QUALIFIED_NOREPACK_MACHINE,
)
NUC = HostTopology(
    cpu_model="Intel(R) Core(TM) i7-10710U",
    physical_cores=6, logical_cpus=12,
    total_ram_mib=64 * GIB, available_ram_mib=40 * GIB, swap_total_mib=8 * GIB,
    machine_model="Intel(R) NUC Kit",
)
NO_CLI = {"threads": None, "threads_batch": None, "batch": None, "ubatch": None, "cache_ram_mib": None}
OTHER_MACHINE = "Intel(R) NUC Kit"

# The real metadata of the qualified artifact (GGUF header of shard 1).
FLASH_NEXT_METADATA = {
    "general.architecture": "qwen4exp",
    "general.name": "Qwen3.8 Flash Next",
    "general.file_type": "31",
    "general.quantization_version": "2",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
    "tokenizer.ggml.bos_token_id": "248044",
    "tokenizer.ggml.eos_token_id": "248046",
    "qwen4exp.context_length": "262144",
    "qwen4exp.block_count": "48",
    "qwen4exp.expert_count": "512",
    "qwen4exp.expert_used_count": "10",
}
QWEN38_27B_METADATA = {
    "general.architecture": "qwen35",
    "general.name": "Qwen3.8-27B",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
    "tokenizer.ggml.bos_token_id": "248044",
    "tokenizer.ggml.eos_token_id": "248046",
    "qwen35.context_length": "262144",
    "qwen35.block_count": "65",
}
QWEN36_METADATA = {
    "general.architecture": "qwen35moe",
    "general.name": "Qwen3.6-35B-A3B",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
}
TEMPLATE = "official-qwen38-template"


def _detect(metadata, template=TEMPLATE):
    digest = hashlib.sha256(template.encode()).hexdigest()
    with mock.patch("orbit.native_llama.model_profiles.QWEN38_OFFICIAL_TEMPLATE_SHA256", digest):
        return detect_native_model_profile(metadata, template)


def _resolve(*, topology=DELL, qualified=None, cli=None, environ=None, user_profile=None, tmp=None):
    env = {"XDG_CACHE_HOME": tmp or tempfile.mkdtemp(prefix="orbit_qwen38_profile_")}
    env.update(environ or {})
    return resolve_profile(
        cli=cli or NO_CLI,
        topology=topology,
        environ=env,
        user_profile=user_profile,
        qualified_profile=qualified.tuning_fields() if qualified is not None else None,
        model_bytes=74_538_755_776,
        model_sha256="flash-next",
        backend_id="41abbfd599fb",
        ctx_tokens=4096,
        allow_calibration=False,
    )


def _args(**overrides):
    base = dict(model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID, model=None, ctx=None, threads=None,
                threads_batch=None, batch=None, ubatch=None, low_memory=False,
                enable_mtp_experimental=False, moe_expert_usage=False, recalibrate=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_inspector(profile):
    """A NativeProfileInspector stand-in returning one detected profile."""
    instance = mock.Mock()
    instance.return_value = profile
    factory = mock.Mock(return_value=instance)
    return factory, instance


class FlashNextIdentityTests(unittest.TestCase):
    """The GGUF identity: one qualified quant, nothing else verified."""

    def test_exact_artifact_metadata_detects_the_verified_profile(self) -> None:
        profile = _detect(FLASH_NEXT_METADATA)
        self.assertEqual(profile.profile_id, QWEN38_FLASH_NEXT_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.architecture, "qwen4exp")
        self.assertEqual(profile.verified_quantization, "UD-IQ1_M")
        self.assertEqual(profile.renderer, "llama.cpp-jinja")
        # Same template bytes as the verified 27B pin -> same wire protocol.
        self.assertEqual(profile.tool_call_protocol, "qwen3.6-xml")
        self.assertEqual(profile.history_serialization, "qwen-leading-system-only")
        self.assertTrue(profile.thinking_supported)
        self.assertFalse(profile.mtp_supported)
        self.assertFalse(profile.gemma_prefix_reuse_supported)
        self.assertTrue(profile.route_prefix_reuse_supported)

    def test_the_template_pin_is_the_official_qwen38_template(self) -> None:
        # The real artifact embeds the template whose sha256 is already pinned
        # for Qwen3.8 27B; a different template must not verify.
        self.assertEqual(len(QWEN38_OFFICIAL_TEMPLATE_SHA256), 64)
        pinned = hashlib.sha256(TEMPLATE.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.QWEN38_OFFICIAL_TEMPLATE_SHA256", pinned):
            profile = detect_native_model_profile(FLASH_NEXT_METADATA, "some other template")
        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen38_flash_next_template_identity_mismatch")

    def test_D_another_qwen38_flash_next_quant_is_not_qualified(self) -> None:
        for file_type in ("24", "15", "23", ""):  # IQ1_S, Q4_K_M, IQ3_XXS, missing (LLAMA_FTYPE_MOSTLY_*)
            profile = _detect({**FLASH_NEXT_METADATA, "general.file_type": file_type})
            self.assertFalse(profile.verified, file_type)
            self.assertEqual(profile.profile_id, "unsupported")
            self.assertEqual(profile.failure_reason, "qwen38_flash_next_quantization_identity_mismatch")
            self.assertFalse(profile.diagnostics(thinking_enabled=False)["capabilities"]["chat"])

    def test_a_different_qwen4exp_model_or_geometry_is_unsupported(self) -> None:
        renamed = _detect({**FLASH_NEXT_METADATA, "general.name": "Qwen4 Something"})
        self.assertEqual(renamed.failure_reason, "qwen38_flash_next_model_identity_mismatch")
        geometry = _detect({**FLASH_NEXT_METADATA, "qwen4exp.expert_count": "256"})
        self.assertEqual(geometry.failure_reason, "qwen38_flash_next_metadata_identity_mismatch")
        tokenizer = _detect({**FLASH_NEXT_METADATA, "tokenizer.ggml.pre": "qwen2"})
        self.assertEqual(tokenizer.failure_reason, "qwen38_flash_next_tokenizer_identity_mismatch")

    def test_registry_entry_binds_the_first_shard_to_the_verified_identity(self) -> None:
        manifest = get_manifest(QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID)
        self.assertEqual(manifest.profile_id, QWEN38_FLASH_NEXT_PROFILE_ID)
        self.assertEqual(manifest.architecture, "qwen4exp")
        self.assertEqual(manifest.target.repo, "unsloth/Qwen3.8-Flash-Next-GGUF")
        self.assertEqual(manifest.target.file, "Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf")
        self.assertIsNone(manifest.mmproj)
        self.assertIsNone(manifest.mtp)
        identity = verified_native_model_identity(manifest.profile_id)
        self.assertEqual((identity.model_name, identity.architecture), ("Qwen3.8 Flash Next", "qwen4exp"))

    def test_only_the_first_split_shard_names_a_model(self) -> None:
        self.assertFalse(_is_secondary_split_shard("Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf"))
        self.assertTrue(_is_secondary_split_shard("Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf"))
        self.assertTrue(_is_secondary_split_shard("Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf"))
        self.assertFalse(_is_secondary_split_shard("Ornith-1.5-35B-Q4_K_M.gguf"))


class QualifiedProfileResolutionTests(unittest.TestCase):
    def test_A_exact_dell_and_qwen38_flash_next_resolve_the_qualified_profile(self) -> None:
        qualified = qualified_startup_profile(
            machine_model=QUALIFIED_NOREPACK_MACHINE, model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID
        )
        self.assertIsInstance(qualified, QualifiedStartupProfile)
        self.assertEqual(
            (qualified.ctx, qualified.threads, qualified.threads_batch, qualified.batch, qualified.ubatch),
            (4096, 10, 10, 256, 128),
        )
        self.assertEqual(qualified.profile_id, QWEN38_FLASH_NEXT_PROFILE_ID)
        self.assertEqual(qualified.source, "qualified-dell-qwen38-flash-next")
        self.assertEqual(qualified.load_mode, LLAMA_LOAD_MODE_MMAP)
        self.assertEqual(qualified.lazy_mode, LLAMA_LAZY_MODE_ON)
        self.assertIs(qualified.load_mtp, False)

        resolution = _resolve(qualified=qualified)
        profile = resolution.profile
        self.assertEqual((profile.threads, profile.threads_batch, profile.batch, profile.ubatch), (10, 10, 256, 128))
        self.assertEqual(profile.source, "qualified")
        for field in ("threads", "threads_batch", "batch", "ubatch"):
            self.assertEqual(profile.field_sources[field], "qualified")
        self.assertFalse(resolution.calibrated)

        value, source = resolve_cpu_repack(
            machine_model=QUALIFIED_NOREPACK_MACHINE, model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID,
            cli=None, environ={},
        )
        self.assertIs(value, False)
        self.assertEqual(source, "qualified-dell-qwen38-flash-next")
        # One slot and no GPU layers are fixed properties of the native server.
        self.assertEqual(NativeClientConfig().gpu_layers, 0)
        self.assertFalse(NativeClientConfig().use_mtp_experimental)

    def test_A_server_applies_the_tier_only_after_confirming_the_gguf_identity(self) -> None:
        verified = _detect(FLASH_NEXT_METADATA)
        factory, instance = _fake_inspector(verified)
        with mock.patch.object(app_module, "detect_topology", return_value=DELL), \
             mock.patch.object(app_module, "resolve_bootstrap_paths",
                               return_value=SimpleNamespace(build_bin=Path("/bin"), model=Path("/m.gguf"))), \
             mock.patch.object(app_module, "NativeProfileInspector", factory):
            qualified = app_module._qualified_startup_profile(_args())
        self.assertIsNotNone(qualified)
        self.assertEqual(qualified.model_id, QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID)
        instance.assert_called_once_with(Path("/m.gguf"))
        instance.close.assert_called_once()
        self.assertEqual(app_module._resolve_ctx(_args(), qualified), (4096, "qualified"))

    def test_B_qwen38_flash_next_on_another_machine_inherits_nothing(self) -> None:
        self.assertIsNone(qualified_startup_profile(machine_model=OTHER_MACHINE,
                                                    model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID))
        value, source = resolve_cpu_repack(machine_model=OTHER_MACHINE,
                                           model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID, cli=None, environ={})
        self.assertIsNone(value)
        self.assertEqual(source, "backend-default")
        factory, instance = _fake_inspector(_detect(FLASH_NEXT_METADATA))
        with mock.patch.object(app_module, "detect_topology", return_value=NUC), \
             mock.patch.object(app_module, "NativeProfileInspector", factory):
            self.assertIsNone(app_module._qualified_startup_profile(_args()))
        factory.assert_not_called()  # no inspection is even attempted off the qualified machine
        resolution = _resolve(topology=NUC, qualified=None)
        self.assertNotEqual(resolution.profile.field_sources["threads"], "qualified")
        self.assertEqual(app_module._resolve_ctx(_args(), None), (app_module.DEFAULT_CTX_TOKENS, "default"))

    def test_C_an_unrelated_model_on_the_dell_inherits_nothing(self) -> None:
        for model_id in ("gemma4-26b-a4b-it-q40", "qwen38-27b-q4-k-m", "qwen3-coder-30b-a3b-instruct-q4-k-m"):
            self.assertIsNone(qualified_startup_profile(machine_model=QUALIFIED_NOREPACK_MACHINE, model_id=model_id))
            value, source = resolve_cpu_repack(machine_model=QUALIFIED_NOREPACK_MACHINE, model_id=model_id,
                                               cli=None, environ={})
            self.assertIsNone(value, model_id)
            self.assertEqual(source, "backend-default")
        with mock.patch.object(app_module, "detect_topology", return_value=DELL):
            self.assertIsNone(app_module._qualified_startup_profile(_args(model_id="qwen38-27b-q4-k-m")))
            self.assertIsNone(app_module._qualified_startup_profile(_args(model_id=None, model="/x.gguf")))

    def test_D_a_file_that_is_not_the_qualified_artifact_gets_no_tier(self) -> None:
        # Right registry id, but the GGUF at that path is another quant, another
        # model, or is not verified at all.
        for detected in (
            _detect({**FLASH_NEXT_METADATA, "general.file_type": "24"}),   # IQ1_S
            _detect(QWEN38_27B_METADATA),                                 # verified, different profile
            SimpleNamespace(verified=False, profile_id=QWEN38_FLASH_NEXT_PROFILE_ID),
        ):
            factory, _instance = _fake_inspector(detected)
            with mock.patch.object(app_module, "detect_topology", return_value=DELL), \
                 mock.patch.object(app_module, "resolve_bootstrap_paths",
                                   return_value=SimpleNamespace(build_bin=Path("/bin"), model=Path("/m.gguf"))), \
                 mock.patch.object(app_module, "NativeProfileInspector", factory):
                self.assertIsNone(app_module._qualified_startup_profile(_args()))
        # An inspection failure is a cache miss, never a crash and never a tier.
        with mock.patch.object(app_module, "detect_topology", return_value=DELL), \
             mock.patch.object(app_module, "resolve_bootstrap_paths", side_effect=FileNotFoundError("missing")):
            self.assertIsNone(app_module._qualified_startup_profile(_args()))

    def test_E_cli_env_and_user_profile_beat_the_qualified_profile(self) -> None:
        qualified = qualified_startup_profile(machine_model=QUALIFIED_NOREPACK_MACHINE,
                                              model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID)
        cli = dict(NO_CLI, threads=4)
        resolution = _resolve(qualified=qualified, cli=cli)
        self.assertEqual(resolution.profile.threads, 4)
        self.assertEqual(resolution.profile.field_sources["threads"], "cli")
        self.assertEqual(resolution.profile.threads_batch, 10)  # untouched fields still qualified
        self.assertEqual(resolution.profile.field_sources["threads_batch"], "qualified")

        resolution = _resolve(qualified=qualified, environ={ENV_THREADS: "12"})
        self.assertEqual(resolution.profile.threads, 12)
        self.assertEqual(resolution.profile.field_sources["threads"], "env")

        resolution = _resolve(qualified=qualified, user_profile={"batch": 512})
        self.assertEqual(resolution.profile.batch, 512)
        self.assertEqual(resolution.profile.field_sources["batch"], "user-profile")
        self.assertEqual(resolution.profile.ubatch, 128)

        self.assertEqual(app_module._resolve_ctx(_args(ctx=2048), qualified), (2048, "cli"))
        self.assertEqual(
            resolve_cpu_repack(machine_model=QUALIFIED_NOREPACK_MACHINE,
                               model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID, cli=True, environ={}),
            (True, "cli"),
        )
        self.assertEqual(
            resolve_cpu_repack(machine_model=QUALIFIED_NOREPACK_MACHINE,
                               model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID, cli=None,
                               environ={ENV_CPU_REPACK: "on"}),
            (True, "env"),
        )

    def test_F_the_ornith_dell_no_repack_profile_is_unchanged(self) -> None:
        self.assertEqual(QUALIFIED_NOREPACK_MODEL_ID, "ornith15-35b-a3b-q4-k-m")
        self.assertEqual(
            resolve_cpu_repack(machine_model=QUALIFIED_NOREPACK_MACHINE, model_id=QUALIFIED_NOREPACK_MODEL_ID,
                               cli=None, environ={}),
            (False, "qualified-dell-ornith"),
        )
        # Ornith has no qualified tuning tier: threads/ctx keep resolving as before.
        self.assertIsNone(qualified_startup_profile(machine_model=QUALIFIED_NOREPACK_MACHINE,
                                                    model_id=QUALIFIED_NOREPACK_MODEL_ID))
        resolution = _resolve(qualified=None)
        self.assertNotIn("qualified", set(resolution.profile.field_sources.values()))
        self.assertEqual(app_module._resolve_ctx(_args(model_id=QUALIFIED_NOREPACK_MODEL_ID), None),
                         (8192, "default"))

    def test_G_existing_gemma_and_qwen_profiles_are_unchanged(self) -> None:
        qwen38 = _detect(QWEN38_27B_METADATA)
        self.assertEqual((qwen38.profile_id, qwen38.verified, qwen38.verified_quantization),
                         (QWEN38_PROFILE_ID, True, "Q4_K_M"))
        self.assertTrue(qwen38.route_prefix_reuse_supported)
        digest = hashlib.sha256(b"official-qwen36-template").hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.QWEN36_OFFICIAL_TEMPLATE_SHA256", digest):
            qwen36 = detect_native_model_profile(QWEN36_METADATA, "official-qwen36-template")
        self.assertEqual((qwen36.profile_id, qwen36.verified), (QWEN36_PROFILE_ID, True))
        gemma = detect_native_model_profile(
            {"general.architecture": "gemma4", "general.name": "gemma-4-26B-A4B-it", "tokenizer.ggml.model": "gemma4"},
            "gemma-template",
        )
        self.assertEqual((gemma.profile_id, gemma.verified, gemma.renderer), (GEMMA4_PROFILE_ID, True, "orbit-gemma4"))
        # Registry: the existing entries keep their ids, order and profiles; the
        # new one is appended.
        ids = [item.id for item in load_registry()]
        self.assertEqual(ids[:5], [
            "gemma4-26b-a4b-it-q40", "qwen36-35b-a3b-q4-k-m", "qwen38-27b-q4-k-m",
            "ornith15-35b-a3b-q4-k-m", "qwen3-coder-30b-a3b-instruct-q4-k-m",
        ])
        self.assertEqual(ids[5], QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID)
        for profile_id in (ORNITH15_PROFILE_ID, QWEN3_CODER_PROFILE_ID, QWEN38_PROFILE_ID):
            self.assertIsNotNone(verified_native_model_identity(profile_id))


class LoadSemanticsTests(unittest.TestCase):
    """H: load_mode / lazy_mode / load_mtp reach llama_model_params as intended."""

    def _client(self, config: NativeClientConfig) -> NativeLlamaClient:
        paths = NativeLlamaPaths(
            llama_root=Path("/llama"), build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"), model=Path("/models/m.gguf"),
            model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID,
        )
        binding = mock.Mock()
        binding.lib.llama_model_default_params.return_value = SimpleNamespace(
            use_extra_bufts=True, progress_callback=None, progress_callback_user_data=None, n_gpu_layers=0,
        )
        with mock.patch("orbit.native_llama.client.LlamaLibrary", return_value=binding):
            return NativeLlamaClient(paths, config)

    def test_H_qualified_values_reach_the_backend_params(self) -> None:
        qualified = qualified_startup_profile(machine_model=QUALIFIED_NOREPACK_MACHINE,
                                              model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID)
        client = self._client(NativeClientConfig(
            context_tokens=qualified.ctx, threads=qualified.threads, threads_batch=qualified.threads_batch,
            batch_size=qualified.batch, ubatch_size=qualified.ubatch, use_extra_bufts=False,
            load_mode=qualified.load_mode, lazy_mode=qualified.lazy_mode, load_mtp=qualified.load_mtp,
        ))
        with mock.patch("orbit.native_llama.client.inspect_native_model_profile"):
            params = client._model_load_params(None)
        self.assertEqual(params.load_mode, LLAMA_LOAD_MODE_MMAP)
        self.assertEqual(params.lazy_mode, LLAMA_LAZY_MODE_ON)
        self.assertIs(params.load_mtp, False)
        self.assertIs(params.use_extra_bufts, False)
        status = client.model_load_status()
        self.assertEqual((status["load_mode"], status["lazy_mode"], status["load_mtp"], status["cpu_repack"]),
                         (LLAMA_LOAD_MODE_MMAP, LLAMA_LAZY_MODE_ON, False, False))

    def test_H_existing_models_keep_the_mission_18_pins_by_default(self) -> None:
        client = self._client(NativeClientConfig())
        with mock.patch("orbit.native_llama.client.inspect_native_model_profile"):
            params = client._model_load_params(None)
        self.assertEqual(params.load_mode, LLAMA_LOAD_MODE_MMAP)
        self.assertEqual(params.lazy_mode, LLAMA_LAZY_MODE_OFF)
        self.assertIs(params.load_mtp, True)
        self.assertIs(params.use_extra_bufts, True)  # backend default, repack on


class ServerWiringTests(unittest.TestCase):
    """The tier reaches the real start: ctx, load semantics and the preview."""

    def test_run_server_passes_ctx_and_load_semantics_from_the_qualified_tier(self) -> None:
        qualified = qualified_startup_profile(machine_model=QUALIFIED_NOREPACK_MACHINE,
                                              model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID)
        resolution = SimpleNamespace(
            profile=SimpleNamespace(source="qualified", threads=10, threads_batch=10, batch=256, ubatch=128),
            calibrated=False, fingerprint="fp", cache_path=None, calibration_error=None, measurements=None,
        )
        seen = {}

        def fake_resolve(args, calibrator=None, qualified=None):
            seen.setdefault("ctx", args.ctx); seen.setdefault("qualified", qualified)
            return resolution

        _FakeNativeClient.instances.clear(); _FakeHTTPServer.instances.clear()
        stderr = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, {"ORBIT_KV_PREFIX_PREWARM": "off"}, clear=False))
            stack.enter_context(mock.patch.object(app_module, "_qualified_startup_profile", return_value=qualified))
            stack.enter_context(mock.patch.object(app_module, "_resolve_startup_profile", side_effect=fake_resolve))
            stack.enter_context(mock.patch.object(app_module, "resolve_bootstrap_paths",
                                                  return_value=SimpleNamespace(model=Path("/m/first.gguf"),
                                                                               model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID)))
            stack.enter_context(mock.patch.object(app_module, "NativeLlamaClient", _FakeNativeClient))
            stack.enter_context(mock.patch.object(app_module, "ThreadingHTTPServer", _FakeHTTPServer))
            stack.enter_context(mock.patch.object(app_module, "resolve_model_alias", return_value="Qwen 3.8 Flash Next"))
            stack.enter_context(mock.patch.object(app_module, "render_profile_lines", return_value=["profile: qualified"]))
            stack.enter_context(mock.patch.object(app_module, "_log_native_threads", return_value=None))
            stack.enter_context(mock.patch.object(app_module, "detect_topology", return_value=DELL))
            stack.enter_context(redirect_stderr(stderr)); stack.enter_context(redirect_stdout(io.StringIO()))
            code = app_module.run_server(["--model-id", QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID])
        self.assertEqual(code, 0)
        config = _FakeNativeClient.instances[0].config
        self.assertEqual(config.context_tokens, 4096)
        self.assertEqual((config.threads, config.threads_batch, config.batch_size, config.ubatch_size), (10, 10, 256, 128))
        self.assertEqual((config.load_mode, config.lazy_mode, config.load_mtp),
                         (LLAMA_LOAD_MODE_MMAP, LLAMA_LAZY_MODE_ON, False))
        self.assertIs(config.use_extra_bufts, False)      # qualified-dell-qwen38-flash-next
        self.assertFalse(config.use_mtp_experimental)
        self.assertEqual(seen["ctx"], 4096)               # the resolver saw the resolved ctx
        self.assertIs(seen["qualified"], qualified)       # and the same tier object
        self.assertIn("qualified profile: qualified-dell-qwen38-flash-next", stderr.getvalue())
        self.assertIn("ctx: 4096 (qualified)", stderr.getvalue())

    def test_run_server_without_a_tier_keeps_8192_and_the_mission_18_pins(self) -> None:
        resolution = SimpleNamespace(
            profile=SimpleNamespace(source="heuristic", threads=6, threads_batch=6, batch=256, ubatch=128),
            calibrated=False, fingerprint="fp", cache_path=None, calibration_error=None, measurements=None,
        )
        _FakeNativeClient.instances.clear(); _FakeHTTPServer.instances.clear()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, {"ORBIT_KV_PREFIX_PREWARM": "off"}, clear=False))
            stack.enter_context(mock.patch.object(app_module, "_qualified_startup_profile", return_value=None))
            stack.enter_context(mock.patch.object(app_module, "_resolve_startup_profile", return_value=resolution))
            stack.enter_context(mock.patch.object(app_module, "resolve_bootstrap_paths",
                                                  return_value=SimpleNamespace(model=Path("/m/v.gguf"), model_id="gemma4-26b-a4b-it-q40")))
            stack.enter_context(mock.patch.object(app_module, "NativeLlamaClient", _FakeNativeClient))
            stack.enter_context(mock.patch.object(app_module, "ThreadingHTTPServer", _FakeHTTPServer))
            stack.enter_context(mock.patch.object(app_module, "resolve_model_alias", return_value="Test"))
            stack.enter_context(mock.patch.object(app_module, "render_profile_lines", return_value=["profile: heuristic"]))
            stack.enter_context(mock.patch.object(app_module, "_log_native_threads", return_value=None))
            stack.enter_context(mock.patch.object(app_module, "detect_topology", return_value=DELL))
            stack.enter_context(redirect_stderr(io.StringIO())); stack.enter_context(redirect_stdout(io.StringIO()))
            code = app_module.run_server(["--model", "/m/v.gguf"])
        self.assertEqual(code, 0)
        config = _FakeNativeClient.instances[0].config
        self.assertEqual(config.context_tokens, 8192)
        self.assertEqual((config.load_mode, config.lazy_mode, config.load_mtp), (None, None, None))
        self.assertIsNone(config.use_extra_bufts)

    def test_show_profile_preview_of_an_interactive_selection_carries_the_registry_id(self) -> None:
        row = ModelDiscoveryRow(model="Qwen 3.8 Flash Next", local="AVAILABLE", support="VERIFIED",
                                path_or_action="/models/x/first.gguf", model_id=QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID)
        args = _args(model_id=None, model=None)
        with mock.patch.object(app_module, "_interactive_model_selection_requested", return_value=True), \
             mock.patch.object(app_module, "_choose_verified_model", return_value=(row, Path("/bin"))), \
             mock.patch.object(app_module, "_select_memory_mode", return_value=None), \
             mock.patch.object(app_module, "_model_identity_for_profile", return_value=("id", 1)):
            target = app_module._resolve_preview_target(args)
        self.assertNotIsInstance(target, int)
        self.assertEqual(args.model, Path("/models/x/first.gguf"))
        self.assertEqual(args.model_id, QUALIFIED_QWEN38_FLASH_NEXT_MODEL_ID)  # what a real start sets too


class SplitDownloadTests(unittest.TestCase):
    def test_orbit_download_expands_the_split_set_and_never_uses_the_single_file_path(self) -> None:
        # GGUF-MULTISHARD-DOWNLOAD-21: the first shard names the whole set; the
        # single-file `retrieve` path is never used for it. Full coverage lives in
        # tests/test_gguf_multishard.py; here only the expansion seam is pinned.
        retrieve = mock.Mock()
        seen: list[str] = []

        def opener(request, timeout=None):
            seen.append(request.full_url)
            raise OSError("offline")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OSError):
                download_model("unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf",
                               models_dir=Path(tmp), retrieve=retrieve, opener=opener)
        retrieve.assert_not_called()
        self.assertEqual(seen, ["https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf"])


if __name__ == "__main__":
    unittest.main()
