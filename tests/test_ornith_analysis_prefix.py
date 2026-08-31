"""Exact-prefix ANALYSIS prewarm for verified Ornith, through the client seam.

An analysis step opens with a fixed contract -- the ANALYSIS system prompt and
the `execute_analysis` schema -- before it names the artifact or the analyst's
instruction. These tests drive the real planner and per-lineage registry, not
the derivation helper alone, because the properties that matter live in the
wiring: that the captured prefix stops before anything session-specific, that
an ANALYSIS checkpoint and a CHAT one can never stand in for each other, and
that a rolling ANALYSIS checkpoint still outranks the prewarm.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.model_profiles import (
    NativeModelProfile,
    ORNITH15_PROFILE_ID,
    QWEN3_CODER_PROFILE_ID,
)
from orbit.native_llama.ornith_analysis_prefix import (
    ORNITH_ANALYSIS_LINEAGE_ID,
    ORNITH_ANALYSIS_PREFIX_ENV,
    ORNITH_ANALYSIS_PREFIX_FORMAT_VERSION,
    ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT,
    ORNITH_ANALYSIS_TOKENIZER_IDENTITY,
    derive_ornith_analysis_prefix_spec,
    resolve_ornith_analysis_prefix_reuse,
)
from orbit.native_llama.ornith_route_prefix import (
    ORNITH_ROUTE_PREFIX_FORMAT_VERSION,
    ORNITH_ROUTE_PREFIX_TOKEN_COUNT,
)
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.prefix_anchor import PrefixAnchorState
from orbit.runtime.analysis_runtime import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_TOOL_SCHEMA

# Pinned literal: the prewarm captures this contract verbatim.
ANALYSIS_SYSTEM_PROMPT_SHA256 = "871cbcaaac7ff2ce6d113358377064dcef9d9649716ec33722c29b26252b101c"


class AnalysisPrefixConfigTests(unittest.TestCase):
    def test_default_on_explicit_on_and_kill_switch(self) -> None:
        self.assertTrue(resolve_ornith_analysis_prefix_reuse({}).enabled)
        self.assertTrue(
            resolve_ornith_analysis_prefix_reuse({ORNITH_ANALYSIS_PREFIX_ENV: "1"}).enabled
        )
        self.assertFalse(
            resolve_ornith_analysis_prefix_reuse({ORNITH_ANALYSIS_PREFIX_ENV: "0"}).enabled
        )

    def test_invalid_value_disables_safely(self) -> None:
        config = resolve_ornith_analysis_prefix_reuse({ORNITH_ANALYSIS_PREFIX_ENV: "maybe"})
        self.assertFalse(config.enabled)
        self.assertEqual(config.validation_error, "invalid_ornith_analysis_prefix_reuse_value")

    def test_env_switch_is_distinct_from_the_chat_one(self) -> None:
        from orbit.native_llama.ornith_route_prefix import ORNITH_ROUTE_PREFIX_ENV

        self.assertNotEqual(ORNITH_ANALYSIS_PREFIX_ENV, ORNITH_ROUTE_PREFIX_ENV)

    def test_identity_constants_differ_from_the_chat_lineage(self) -> None:
        # Same model and tokenizer family, entirely different opening tokens,
        # so the format version is what keeps the two identities apart.
        self.assertNotEqual(
            ORNITH_ANALYSIS_PREFIX_FORMAT_VERSION, ORNITH_ROUTE_PREFIX_FORMAT_VERSION
        )
        self.assertEqual(ORNITH_ANALYSIS_TOKENIZER_IDENTITY, "gpt2:qwen35")
        self.assertNotEqual(ORNITH_ANALYSIS_LINEAGE_ID, ORNITH15_PROFILE_ID)

    def test_count_is_smaller_than_the_route_count(self) -> None:
        # A whole first ANALYSIS request is ~540 tokens against ~840 for a
        # route request, so the route's count cannot fit here.
        self.assertLess(ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT, ORNITH_ROUTE_PREFIX_TOKEN_COUNT)


class AnalysisPrefixDerivationTests(unittest.TestCase):
    """The derivation must refuse anything it cannot prove invariant."""

    def _render(self, system: str):
        def render(user: str) -> str:
            return system + "\n<user>" + user + "</user>"

        return render

    def test_derives_the_fixed_prefix_before_dynamic_content(self) -> None:
        system = "s" * 500
        render = self._render(system)
        full = render("Artifact under analysis: /workspace/input (21 bytes, sha256 abc).")

        spec, reason = derive_ornith_analysis_prefix_spec(
            system_prompt=system,
            full_prompt=full,
            full_tokens=[ord(c) for c in full],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )

        self.assertIsNone(reason)
        assert spec is not None
        self.assertEqual(len(spec.prefix_tokens), ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT)
        self.assertGreater(spec.invariant_token_count, ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT)

    def test_no_volatile_source_identity_survives_into_the_prefix(self) -> None:
        system = "s" * 500
        render = self._render(system)
        full = render("Artifact under analysis: /workspace/input (4096 bytes, sha256 deadbeef).")
        spec, _reason = derive_ornith_analysis_prefix_spec(
            system_prompt=system,
            full_prompt=full,
            full_tokens=[ord(c) for c in full],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )
        assert spec is not None
        text = "".join(chr(t) for t in spec.prefix_tokens)

        for volatile in ("deadbeef", "4096 bytes", "orbit-qwen-route-boundary", "<user>"):
            self.assertNotIn(volatile, text, "no session-specific token may enter the prewarm")

    def test_short_prompt_is_refused(self) -> None:
        system = "s" * 10
        render = self._render(system)
        full = render("x")
        spec, reason = derive_ornith_analysis_prefix_spec(
            system_prompt=system,
            full_prompt=full,
            full_tokens=[ord(c) for c in full],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )
        self.assertIsNone(spec)
        self.assertEqual(reason, "route_prompt_too_short")

    def test_unstable_boundary_is_refused_not_trimmed(self) -> None:
        # User text lands before the fixed count: refuse, never shorten.
        head = "s" * 100

        def render(user: str) -> str:
            return head + user + "t" * 500

        full = render("actual analyst message")
        spec, reason = derive_ornith_analysis_prefix_spec(
            system_prompt=head,
            full_prompt=full,
            full_tokens=[ord(c) for c in full],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )
        self.assertIsNone(spec)
        self.assertIsNotNone(reason)

    def test_production_prompt_diverging_from_the_reference_is_refused(self) -> None:
        system = "s" * 500
        render = self._render(system)
        foreign = "DIFFERENT" + "s" * 500 + "\n<user>x</user>"
        spec, reason = derive_ornith_analysis_prefix_spec(
            system_prompt=system,
            full_prompt=foreign,
            full_tokens=[ord(c) for c in foreign],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )
        self.assertIsNone(spec)
        self.assertEqual(reason, "production_prefix_mismatch")


class _FakeBridge:
    """ChatML-shaped render whose output depends on the tools it is given."""

    def __init__(self) -> None:
        self.tool_calls: list[int] = []

    def render(self, _context, messages, tools, *, thinking: bool):
        assert not thinking
        self.tool_calls.append(len(tools or []))
        schema = f"<tools:{len(tools or [])}>"
        # Schema first, as the real ChatML template renders it inside the
        # system turn ahead of the rest of the contract.
        body = schema + str(messages[0]["content"])
        if len(messages) > 1:
            body += "\n<user>" + str(messages[1]["content"]) + "</user>"
        return {"prompt": body, "generation_prompt": ""}

    def free(self, _context) -> None:
        return None


def _profile(**overrides) -> NativeModelProfile:
    base = dict(
        profile_id=ORNITH15_PROFILE_ID,
        family="ornith1.5",
        model_name="Ornith-1.5-35B",
        architecture="qwen35moe",
        renderer="llama.cpp-jinja",
        reasoning_protocol="qwen-think",
        tool_call_protocol="qwen3.6-xml",
        history_serialization="qwen-leading-system-only",
        verified=True,
        failure_reason=None,
        template_source="gguf-embedded-official",
        template_sha256="f" * 64,
        thinking_supported=True,
        mtp_supported=False,
        gemma_prefix_reuse_supported=False,
        route_prefix_reuse_supported=True,
        verified_quantization="Q4_K_M",
    )
    base.update(overrides)
    return NativeModelProfile(**base)


class AnalysisPrefixClientTests(unittest.TestCase):
    def _client(self, *, profile=None, enabled=True, file_type="15") -> NativeLlamaClient:
        paths = NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/ornith.gguf"),
            model_id="legacy-path",
        )
        config = NativeClientConfig(
            qwen_route_prefix_reuse_enabled=False,
            qwen3_coder_route_prefix_reuse_enabled=False,
            ornith_route_prefix_reuse_enabled=False,
            ornith_analysis_prefix_reuse_enabled=enabled,
        )
        with mock.patch("orbit.native_llama.client.LlamaLibrary"):
            client = NativeLlamaClient(paths, config)
        client.model_profile = profile or _profile()
        client._model_metadata_identity = {
            "general.architecture": "qwen35moe",
            "general.name": "Ornith-1.5-35B",
            "general.file_type": file_type,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "qwen35",
        }
        client.chat_bridge = _FakeBridge()  # type: ignore[assignment]
        client._chat_bridge_context = 1  # type: ignore[assignment]
        client._model = 1  # type: ignore[assignment]
        client._vocab = 1  # type: ignore[assignment]
        client.tokenize = lambda text: [ord(c) for c in text]  # type: ignore[method-assign]
        client._qwen_backend_build_identity = lambda: "build"  # type: ignore[method-assign]
        return client

    def _messages(self, artifact: str = "(21 bytes, sha256 abc)"):
        return [
            {"role": "system", "content": "s" * 500},
            {"role": "user", "content": f"Artifact under analysis: /workspace/input {artifact}."},
        ]

    def _plan(self, client, artifact: str = "(21 bytes, sha256 abc)", *, lineage=True, tools=None):
        messages = self._messages(artifact)
        tools = [{"name": "execute_analysis"}] if tools is None else tools
        prompt = client.apply_chat_template(messages, tools=tools, thinking=False)
        return client._qwen_route_anchor_plan_for_prompt(
            messages, tools=tools, thinking=False, prompt=prompt, analysis_lineage=lineage
        )

    # --- exact prefix ---------------------------------------------------
    def test_plan_is_produced_on_the_analysis_lineage(self) -> None:
        plan = self._plan(self._client())

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.profile_id, ORNITH_ANALYSIS_LINEAGE_ID)
        self.assertEqual(len(plan.prefix_tokens), ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT)

    def test_prefix_is_exact_across_different_sources_and_messages(self) -> None:
        client = self._client()
        reference = self._plan(client)
        assert reference is not None

        for artifact in (
            "(21 bytes, sha256 abc)",
            "(4096 bytes, sha256 deadbeefcafe)",
            "(1 bytes, sha256 0)",
            "(5000000 bytes, sha256 " + "f" * 64 + ")",
        ):
            with self.subTest(artifact=artifact[:24]):
                messages = self._messages(artifact)
                tools = [{"name": "execute_analysis"}]
                prompt = client.apply_chat_template(messages, tools=tools, thinking=False)
                tokens = client.tokenize(prompt)
                self.assertEqual(
                    tuple(tokens[: len(reference.prefix_tokens)]),
                    tuple(reference.prefix_tokens),
                    "the prewarm must be an exact token prefix of every real request",
                )

    def test_prefix_stops_before_the_artifact_identity(self) -> None:
        plan = self._plan(self._client(), "(4096 bytes, sha256 deadbeef)")
        assert plan is not None
        text = "".join(chr(t) for t in plan.prefix_tokens)

        for volatile in ("deadbeef", "4096", "/workspace/input", "<user>"):
            self.assertNotIn(volatile, text)

    def test_the_tool_schema_is_inside_the_prefix(self) -> None:
        # The schema is part of the fixed contract, so it must be captured --
        # otherwise a changed tool surface would reuse the wrong prefix.
        plan = self._plan(self._client())
        assert plan is not None
        self.assertIn("<tools:1>", "".join(chr(t) for t in plan.prefix_tokens))

    # --- gating ---------------------------------------------------------
    def test_disabled_config_produces_no_plan(self) -> None:
        self.assertIsNone(self._plan(self._client(enabled=False)))

    def test_thinking_produces_no_plan(self) -> None:
        client = self._client()
        messages = self._messages()
        tools = [{"name": "execute_analysis"}]
        prompt = client.apply_chat_template(messages, tools=tools, thinking=False)
        self.assertIsNone(
            client._qwen_route_anchor_plan_for_prompt(
                messages, tools=tools, thinking=True, prompt=prompt, analysis_lineage=True
            )
        )

    def test_a_different_tool_surface_produces_no_plan(self) -> None:
        for tools in ([], [{"a": 1}, {"b": 2}]):
            with self.subTest(tools=len(tools)):
                self.assertIsNone(self._plan(self._client(), tools=tools))

    def test_unverified_profile_or_quantization_produces_no_plan(self) -> None:
        self.assertIsNone(self._plan(self._client(profile=_profile(verified=False))))
        self.assertIsNone(self._plan(self._client(file_type="7")))

    def test_foreign_profile_produces_no_plan(self) -> None:
        client = self._client(profile=_profile(profile_id=QWEN3_CODER_PROFILE_ID))
        self.assertIsNone(self._plan(client))


class AnalysisPrefixIsolationTests(unittest.TestCase):
    """CHAT and ANALYSIS prefixes must never stand in for each other."""

    def _client(self):
        return AnalysisPrefixClientTests._client(AnalysisPrefixClientTests("run"))

    def test_each_lineage_keeps_its_own_slot(self) -> None:
        client = self._client()
        analysis = PrefixAnchorState(prefix_hash="analysis", token_count=384, valid=True)
        chat = PrefixAnchorState(prefix_hash="chat", token_count=768, valid=True)

        client._set_qwen_route_prefix_state(ORNITH_ANALYSIS_LINEAGE_ID, analysis)
        client._set_qwen_route_prefix_state(ORNITH15_PROFILE_ID, chat)

        self.assertIs(
            client._qwen_route_prefix_state_for_profile(ORNITH_ANALYSIS_LINEAGE_ID), analysis
        )
        self.assertIs(client._qwen_route_prefix_state_for_profile(ORNITH15_PROFILE_ID), chat)

    def test_spec_and_status_slots_are_per_lineage(self) -> None:
        client = self._client()
        client._set_qwen_route_prefix_spec(ORNITH_ANALYSIS_LINEAGE_ID, "analysis-spec")  # type: ignore[arg-type]
        client._set_qwen_route_prefix_spec(ORNITH15_PROFILE_ID, "chat-spec")  # type: ignore[arg-type]

        self.assertEqual(
            client._qwen_route_prefix_spec_for_profile(ORNITH_ANALYSIS_LINEAGE_ID), "analysis-spec"
        )
        self.assertEqual(
            client._qwen_route_prefix_spec_for_profile(ORNITH15_PROFILE_ID), "chat-spec"
        )
        self.assertIsNot(
            client._qwen_route_prefix_status_for_profile(ORNITH_ANALYSIS_LINEAGE_ID),
            client._qwen_route_prefix_status_for_profile(ORNITH15_PROFILE_ID),
        )

    def test_identities_differ_so_neither_checkpoint_validates_the_other(self) -> None:
        client = self._client()
        spec = SimpleNamespace(prefix_token_hash="h", invariant_text_hash="i", system_prompt_hash="s")

        analysis = client._qwen_route_prefix_state_kwargs(spec, profile_id=ORNITH_ANALYSIS_LINEAGE_ID)
        chat = client._qwen_route_prefix_state_kwargs(spec, profile_id=ORNITH15_PROFILE_ID)

        self.assertNotEqual(analysis["template_id"], chat["template_id"])
        self.assertNotEqual(analysis["tools_mode"], chat["tools_mode"])

    def test_analysis_lineage_never_returns_a_rolling_slot(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src/orbit/native_llama/client.py"
        text = source.read_text(encoding="utf-8")
        start = text.index("def _qwen_route_prefix_state_for_profile")
        registry = text[start : text.index("    def ", start + 50)]

        self.assertNotIn("_rolling_analysis_anchor_state", registry)
        self.assertNotIn("_rolling_route_anchor_state", registry)

    def test_chat_lineage_refuses_a_prompt_carrying_the_analysis_tool(self) -> None:
        client = self._client()
        messages = AnalysisPrefixClientTests._messages(AnalysisPrefixClientTests("run"))
        tools = [{"name": "execute_analysis"}]
        prompt = client.apply_chat_template(messages, tools=tools, thinking=False)

        self.assertIsNone(
            client._qwen_route_anchor_plan_for_prompt(
                messages, tools=tools, thinking=False, prompt=prompt, analysis_lineage=False
            ),
            "a tools-bearing prompt must not take the CHAT route-prefix lineage",
        )


class AnalysisPrefixInvalidationTests(unittest.TestCase):
    def _client(self):
        return AnalysisPrefixClientTests._client(AnalysisPrefixClientTests("run"))

    def test_reset_invalidates_the_analysis_prefix(self) -> None:
        client = self._client()
        client._set_qwen_route_prefix_state(
            ORNITH_ANALYSIS_LINEAGE_ID,
            PrefixAnchorState(prefix_hash="p", token_count=384, valid=True),
        )

        client._invalidate_qwen_route_prefix("session_reset")

        self.assertFalse(
            client._qwen_route_prefix_state_for_profile(ORNITH_ANALYSIS_LINEAGE_ID).valid
        )

    def test_blanket_invalidation_covers_every_lineage(self) -> None:
        client = self._client()
        for lineage in (ORNITH_ANALYSIS_LINEAGE_ID, ORNITH15_PROFILE_ID, QWEN3_CODER_PROFILE_ID):
            client._set_qwen_route_prefix_state(
                lineage, PrefixAnchorState(prefix_hash="p", token_count=384, valid=True)
            )

        client._invalidate_qwen_route_prefix("model_reload")

        for lineage in (ORNITH_ANALYSIS_LINEAGE_ID, ORNITH15_PROFILE_ID, QWEN3_CODER_PROFILE_ID):
            self.assertFalse(client._qwen_route_prefix_state_for_profile(lineage).valid, lineage)

    def test_targeted_invalidation_leaves_the_chat_lineage_alone(self) -> None:
        client = self._client()
        for lineage in (ORNITH_ANALYSIS_LINEAGE_ID, ORNITH15_PROFILE_ID):
            client._set_qwen_route_prefix_state(
                lineage, PrefixAnchorState(prefix_hash="p", token_count=384, valid=True)
            )

        client._invalidate_qwen_route_prefix("x", profile_id=ORNITH_ANALYSIS_LINEAGE_ID)

        self.assertFalse(
            client._qwen_route_prefix_state_for_profile(ORNITH_ANALYSIS_LINEAGE_ID).valid
        )
        self.assertTrue(client._qwen_route_prefix_state_for_profile(ORNITH15_PROFILE_ID).valid)

    def test_session_reset_path_names_the_analysis_lineage(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src/orbit/native_llama/client.py"
        text = source.read_text(encoding="utf-8")
        start = text.index("def reset_session_state")
        block = text[start : text.index("    def ", start + 50)]

        self.assertIn("ORNITH_ANALYSIS_LINEAGE_ID", block)


class RollingAnalysisOutranksPrewarmTests(unittest.TestCase):
    """Required priority: rolling ANALYSIS > ANALYSIS prewarm > cold."""

    def _client(self):
        return AnalysisPrefixClientTests._client(AnalysisPrefixClientTests("run"))

    def _identity(self):
        from orbit.native_llama.rolling_route_anchor import (
            ROLLING_ANALYSIS_STRATEGY_ID,
            RollingRouteIdentity,
        )

        return RollingRouteIdentity(
            strategy_id=ROLLING_ANALYSIS_STRATEGY_ID,
            session_id="default",
            profile_id=ORNITH15_PROFILE_ID,
            model_id="/models/ornith.gguf",
            template_id="tpl",
            tool_schema_hash="analysis-tools",
            capability_summary_hash="caps",
            runtime_policy_hash="policy",
            native_version="libllama.so",
            tools_mode="on",
            reset_generation=0,
        )

    def test_a_usable_rolling_analysis_checkpoint_outranks_the_prewarm(self) -> None:
        from orbit.native_llama.rolling_route_anchor import RollingRouteAnchorState

        client = self._client()
        identity = self._identity()
        saved = [1, 2, 3, 4, 5]
        client._rolling_analysis_anchor_state = RollingRouteAnchorState(
            identity=identity, tokens=list(saved), checkpoint_data=b"c", created_at_monotonic=1.0
        )

        self.assertTrue(
            client._rolling_outranks_route_prefix(
                saved + [6], rolling_route_eligible=True, rolling_route_identity=identity
            ),
            "rolling ANALYSIS must win when it can serve this prompt",
        )

    def test_the_guard_reads_the_analysis_rolling_slot_not_the_chat_one(self) -> None:
        from orbit.native_llama.rolling_route_anchor import RollingRouteAnchorState

        client = self._client()
        identity = self._identity()
        # Only the CHAT rolling slot holds anything; the analysis identity must
        # not be served from it.
        client._rolling_route_anchor_state = RollingRouteAnchorState(
            identity=identity, tokens=[1, 2, 3], checkpoint_data=b"c", created_at_monotonic=1.0
        )

        self.assertFalse(
            client._rolling_outranks_route_prefix(
                [1, 2, 3, 4], rolling_route_eligible=True, rolling_route_identity=identity
            )
        )

    def test_prewarm_still_runs_when_rolling_cannot_serve_the_prompt(self) -> None:
        client = self._client()
        identity = self._identity()

        self.assertFalse(
            client._rolling_outranks_route_prefix(
                [9, 9, 9], rolling_route_eligible=True, rolling_route_identity=identity
            ),
            "a cold analysis session must still get its prewarm",
        )


class AnalysisPrewarmAddsNoModelCallTests(unittest.TestCase):
    def _capture_block(self) -> str:
        source = Path(__file__).resolve().parents[1] / "src/orbit/native_llama/client.py"
        text = source.read_text(encoding="utf-8")
        start = text.index("def capture_qwen3_coder_route_prefix_prefill_only")
        return text[start : text.index("    def ", start + 50)]

    def test_capture_never_generates(self) -> None:
        block = self._capture_block()
        for forbidden in ("_generate_from_current_context", "llama_sampler_sample"):
            self.assertNotIn(forbidden, block)

    def test_analysis_step_makes_exactly_one_model_call(self) -> None:
        import shutil
        import tempfile

        from orbit.backend.base import ChatResult
        from orbit.runtime.analysis_runtime import (
            AnalysisRuntime,
            AnalysisWorkspace,
            acquire_analysis_source,
        )
        from orbit.runtime.evidence import EvidenceStore

        tmp = Path(tempfile.mkdtemp(prefix="orbit-aprewarm-test-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        artifact = tmp / "note.txt"
        artifact.write_text("fixture\n", encoding="utf-8")
        workspace = AnalysisWorkspace.create()
        source = acquire_analysis_source(artifact, workspace.source_root)

        class Backend:
            def __init__(self) -> None:
                self.calls = 0

            def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                            on_delta=None, on_progress=None):
                self.calls += 1
                return ChatResult("ok", "scripted", "stop", [], 1, 1, 0, None, None)

            def chat(self, *args, **kwargs):
                return self.chat_stream(*args, **kwargs)

        backend = Backend()
        runtime = AnalysisRuntime(
            backend=backend,
            source=source,
            evidence_store=EvidenceStore(root=tmp / "ev"),
            workspace=workspace,
        )
        self.addCleanup(runtime.close)

        for expected in (1, 2, 3):
            result = runtime.step("continue")
            self.assertEqual(backend.calls, expected)
            self.assertEqual(result.model_calls, 1)
            self.assertTrue(result.control_returned)


class AnalysisCaptureWiringTests(unittest.TestCase):
    """Without a capture path the prewarm would be inert: nothing populates it."""

    def _capture_block(self) -> str:
        source = Path(__file__).resolve().parents[1] / "src/orbit/native_llama/client.py"
        text = source.read_text(encoding="utf-8")
        start = text.index("def capture_qwen3_coder_route_prefix_prefill_only")
        return text[start : text.index("    def ", start + 50)]

    def test_capture_accepts_the_analysis_lineage(self) -> None:
        import inspect

        sig = inspect.signature(NativeLlamaClient.capture_qwen3_coder_route_prefix_prefill_only)

        self.assertIn("analysis_lineage", sig.parameters)
        self.assertIn("tools", sig.parameters)

    def test_capture_books_the_result_under_the_analysis_lineage(self) -> None:
        block = self._capture_block()

        self.assertIn("ORNITH_ANALYSIS_LINEAGE_ID", block)
        self.assertIn("ornith_analysis_prefix_reuse_enabled", block)

    def test_capture_passes_the_tools_through_to_the_planner(self) -> None:
        block = self._capture_block()

        # Rendering the capture without the schema would produce a different
        # prompt from production and derivation would refuse it.
        self.assertIn("capture_tools", block)
        self.assertIn("analysis_lineage=analysis_lineage", block)

    def test_capture_refuses_the_analysis_lineage_on_a_foreign_profile(self) -> None:
        client = AnalysisPrefixClientTests._client(
            AnalysisPrefixClientTests("run"), profile=_profile(profile_id=QWEN3_CODER_PROFILE_ID)
        )
        result = client.capture_qwen3_coder_route_prefix_prefill_only(
            system_prompt="s" * 500, tools_mode="on", tools=[{"n": 1}], analysis_lineage=True
        )
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "model_profile_ineligible")


class AnalysisFallbackAttributionTests(unittest.TestCase):
    def test_serve_time_fallback_is_recorded_against_the_plan_lineage(self) -> None:
        client = AnalysisPrefixClientTests._client(AnalysisPrefixClientTests("run"))
        plan = SimpleNamespace(prefix_tokens=[1, 2, 3], profile_id=ORNITH_ANALYSIS_LINEAGE_ID)
        client.tokenize = lambda text: [9, 9, 9, 9]  # type: ignore[method-assign]
        client._route_anchor_plan = lambda *a, **k: None  # type: ignore[method-assign]
        client._session.ctx_tgt = object()
        client._session.sampler = object()

        with self.assertRaises(Exception):
            client._complete_prompt_standard("p", qwen_route_anchor_plan=plan)

        analysis_status = client._qwen_route_prefix_status_for_profile(ORNITH_ANALYSIS_LINEAGE_ID)
        chat_status = client._qwen_route_prefix_status_for_profile(ORNITH15_PROFILE_ID)
        self.assertEqual(analysis_status.failure_reason, "production_prefix_changed")
        self.assertGreaterEqual(analysis_status.fallback_count, 1)
        self.assertIsNone(chat_status.failure_reason, "the CHAT lineage must not absorb it")

    def test_probe_tools_does_not_alias_the_module_level_schema(self) -> None:
        import copy as _copy

        original = _copy.deepcopy(ANALYSIS_TOOL_SCHEMA)
        client = AnalysisPrefixClientTests._client(AnalysisPrefixClientTests("run"))
        messages = AnalysisPrefixClientTests._messages(AnalysisPrefixClientTests("run"))
        tools = [ANALYSIS_TOOL_SCHEMA]
        prompt = client.apply_chat_template(messages, tools=tools, thinking=False)
        client._qwen_route_anchor_plan_for_prompt(
            messages, tools=tools, thinking=False, prompt=prompt, analysis_lineage=True
        )

        self.assertEqual(ANALYSIS_TOOL_SCHEMA, original, "the shared schema must not be mutated")


class AnalysisPromptUnchangedTests(unittest.TestCase):
    def test_analysis_system_prompt_matches_the_qualified_pin(self) -> None:
        import hashlib

        # Pinned: the prewarm captures this contract, so a silent edit would
        # change the prefix without changing the identity that guards it. The
        # pin moved once, with the input-contract fix that told the model
        # /workspace/input is the artifact file rather than a directory, and
        # only alongside the prefix requalification that change forced.
        #
        # It moved a second time, deliberately and with user authorization, for
        # evidence-aware compaction: a compacted turn shows the model a
        # reference instead of the observation, so the prompt has to say what a
        # reference is and how to read the exact bytes back. Without that the
        # compaction hides evidence, which is worse than the context ceiling it
        # removes. `test_evidence_authority` pins the exact clause.
        #
        # It moved a third time, again deliberately and with user
        # authorization, for progress control: a live run spent all nine of its
        # actions re-reading one file and never ran the decoder it had already
        # identified. The runtime half of that fix stops re-executing an
        # observation it has already made; this clause is the other half,
        # telling the model that an identified deterministic transformation is
        # worth more than another read of source it already holds. It names no
        # technique -- no XOR, no base64, no format -- because choosing one is
        # the model's job. `test_evidence_authority` pins the exact clause and
        # its position.
        self.assertEqual(
            hashlib.sha256(ANALYSIS_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            ANALYSIS_SYSTEM_PROMPT_SHA256,
        )

    def test_analysis_tool_surface_is_still_a_single_tool(self) -> None:
        self.assertEqual(ANALYSIS_TOOL_SCHEMA["function"]["name"], "execute_analysis")


if __name__ == "__main__":
    unittest.main()
