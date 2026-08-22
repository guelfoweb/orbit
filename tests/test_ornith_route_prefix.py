"""Exact-prefix CHAT prewarm for verified Ornith, through the client seam.

The startup prewarm never applied to Ornith: it rendered Gemma markup while
Ornith renders ChatML, so the two disagreed at the very first token. These
tests drive the real planner, capture and restore paths -- not the derivation
helper alone -- because what has to hold lives in the wiring: that a prewarm
is only ever restored when its tokens are an exact prefix of the request, that
each profile keeps its own checkpoint, and that a valid rolling checkpoint is
never displaced by a prewarm.
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
from orbit.native_llama.ornith_route_prefix import (
    ORNITH_ROUTE_PREFIX_ENV,
    ORNITH_ROUTE_PREFIX_FORMAT_VERSION,
    ORNITH_ROUTE_PREFIX_TOKEN_COUNT,
    ORNITH_ROUTE_TOKENIZER_IDENTITY,
    derive_ornith_route_prefix_spec,
    resolve_ornith_route_prefix_reuse,
)
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.prefix_anchor import PrefixAnchorState
from orbit.runtime.messages import ROUTE_SYSTEM_PROMPT


class OrnithRoutePrefixConfigTests(unittest.TestCase):
    def test_default_on_explicit_on_and_kill_switch(self) -> None:
        self.assertTrue(resolve_ornith_route_prefix_reuse({}).enabled)
        self.assertTrue(resolve_ornith_route_prefix_reuse({ORNITH_ROUTE_PREFIX_ENV: "1"}).enabled)
        self.assertFalse(resolve_ornith_route_prefix_reuse({ORNITH_ROUTE_PREFIX_ENV: "0"}).enabled)

    def test_invalid_value_disables_safely(self) -> None:
        config = resolve_ornith_route_prefix_reuse({ORNITH_ROUTE_PREFIX_ENV: "yes"})
        self.assertFalse(config.enabled)
        self.assertEqual(config.validation_error, "invalid_ornith_route_prefix_reuse_value")

    def test_identity_constants_are_distinct_from_other_profiles(self) -> None:
        from orbit.native_llama.qwen3_coder_route_prefix import (
            QWEN3_CODER_ROUTE_PREFIX_FORMAT_VERSION,
            QWEN3_CODER_ROUTE_TOKENIZER_IDENTITY,
        )

        self.assertNotEqual(
            ORNITH_ROUTE_PREFIX_FORMAT_VERSION, QWEN3_CODER_ROUTE_PREFIX_FORMAT_VERSION
        )
        # Ornith's tokenizer family is its own; a shared identity string would
        # let one profile's checkpoint look valid for another.
        self.assertEqual(ORNITH_ROUTE_TOKENIZER_IDENTITY, "gpt2:qwen35")
        self.assertNotEqual(ORNITH_ROUTE_TOKENIZER_IDENTITY, QWEN3_CODER_ROUTE_TOKENIZER_IDENTITY)


class OrnithAnchorRequestPredicateTests(unittest.TestCase):
    """The backend must ask for the anchor when, and only when, Ornith reuse is on."""

    def _requested(self, env: dict) -> bool:
        import os as _os

        from orbit.backend.llama_server import _qwen_route_prefix_anchor_requested
        from orbit.runtime.kv_diag import model_call_context

        with mock.patch.dict(_os.environ, env, clear=True):
            with model_call_context(phase="route", tools_mode="on"):
                return _qwen_route_prefix_anchor_requested(native_backend=True)

    def test_ornith_reuse_alone_requests_the_anchor(self) -> None:
        # The other two profiles are switched off, so only the Ornith switch
        # can be what enables the request.
        self.assertTrue(
            self._requested(
                {
                    "ORBIT_QWEN_ROUTE_PREFIX_REUSE": "0",
                    "ORBIT_QWEN3_CODER_ROUTE_PREFIX_REUSE": "0",
                    "ORBIT_ORNITH_ROUTE_PREFIX_REUSE": "1",
                }
            )
        )

    def test_all_switches_off_requests_nothing(self) -> None:
        self.assertFalse(
            self._requested(
                {
                    "ORBIT_QWEN_ROUTE_PREFIX_REUSE": "0",
                    "ORBIT_QWEN3_CODER_ROUTE_PREFIX_REUSE": "0",
                    "ORBIT_ORNITH_ROUTE_PREFIX_REUSE": "0",
                }
            )
        )

    def test_ornith_kill_switch_is_honoured_on_its_own(self) -> None:
        self.assertFalse(
            self._requested(
                {
                    "ORBIT_QWEN_ROUTE_PREFIX_REUSE": "0",
                    "ORBIT_QWEN3_CODER_ROUTE_PREFIX_REUSE": "0",
                    "ORBIT_ORNITH_ROUTE_PREFIX_REUSE": "invalid",
                }
            )
        )


class OrnithRoutePrefixBoundaryTests(unittest.TestCase):
    """The derivation must reject anything it cannot prove invariant."""

    def _render(self, system: str):
        def render(user: str) -> str:
            return system + "\n<user>" + user + "</user>"

        return render

    def test_derives_the_fixed_prefix_before_dynamic_user_content(self) -> None:
        system = "s" * 900
        render = self._render(system)
        full = render("actual request")

        spec, reason = derive_ornith_route_prefix_spec(
            system_prompt=system,
            full_prompt=full,
            full_tokens=[ord(c) for c in full],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )

        self.assertIsNone(reason)
        assert spec is not None
        self.assertEqual(len(spec.prefix_tokens), ORNITH_ROUTE_PREFIX_TOKEN_COUNT)
        self.assertGreater(spec.invariant_token_count, ORNITH_ROUTE_PREFIX_TOKEN_COUNT)

    def test_short_prompt_is_refused(self) -> None:
        system = "s" * 10
        render = self._render(system)
        full = render("x")
        spec, reason = derive_ornith_route_prefix_spec(
            system_prompt=system,
            full_prompt=full,
            full_tokens=[ord(c) for c in full],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )
        self.assertIsNone(spec)
        self.assertEqual(reason, "route_prompt_too_short")

    def test_unstable_boundary_is_refused(self) -> None:
        # The prompt is long enough to clear the length gate, but user text
        # begins at token 700 -- before the fixed count. The derivation must
        # refuse rather than shorten the prefix to whatever happens to match.
        head = "s" * 700
        tail = "t" * 400

        def render(user: str) -> str:
            return head + user + tail

        full = render("actual request")
        spec, reason = derive_ornith_route_prefix_spec(
            system_prompt=head,
            full_prompt=full,
            full_tokens=[ord(c) for c in full],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )
        self.assertIsNone(spec)
        self.assertEqual(reason, "stable_token_boundary_unavailable")

    def test_production_prompt_that_diverges_from_the_reference_is_refused(self) -> None:
        # The two reference renders agree, but the prompt actually being served
        # does not start with them. Accepting here would restore a KV sequence
        # that is not this prompt's prefix.
        system = "s" * 900
        render = self._render(system)
        foreign = "DIFFERENT" + "s" * 900 + "\n<user>x</user>"

        spec, reason = derive_ornith_route_prefix_spec(
            system_prompt=system,
            full_prompt=foreign,
            full_tokens=[ord(c) for c in foreign],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )

        self.assertIsNone(spec, "a prompt that is not an extension of the prefix must be refused")
        self.assertEqual(reason, "production_prefix_mismatch")

    def test_references_disagreeing_inside_the_prefix_are_refused(self) -> None:
        # Here the "invariant" region is not invariant: the user text lands
        # inside the fixed count, so the two references differ there. Accepting
        # would bake one request's user tokens into the prewarm.
        def render(user: str) -> str:
            return "s" * 100 + user + "s" * 900

        full = render("actual request")
        spec, reason = derive_ornith_route_prefix_spec(
            system_prompt="s" * 100,
            full_prompt=full,
            full_tokens=[ord(c) for c in full],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )

        self.assertIsNone(spec, "user text inside the prefix must be refused")
        self.assertIn(reason, {"reference_prefix_mismatch", "stable_token_boundary_unavailable"})

    def test_no_reference_user_text_survives_into_the_derived_prefix(self) -> None:
        system = "s" * 900
        render = self._render(system)
        full = render("actual request")
        spec, _reason = derive_ornith_route_prefix_spec(
            system_prompt=system,
            full_prompt=full,
            full_tokens=[ord(c) for c in full],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )
        assert spec is not None
        text = "".join(chr(t) for t in spec.prefix_tokens)

        for marker in ("orbit-qwen-route-boundary", "A-orbit", "Z-orbit", "actual request", "<user>"):
            self.assertNotIn(marker, text, "no reference or user token may enter the prewarm")

    def test_missing_system_prompt_is_refused(self) -> None:
        spec, reason = derive_ornith_route_prefix_spec(
            system_prompt="",
            full_prompt="x" * 900,
            full_tokens=[1] * 900,
            render_reference=lambda u: "x" * 900,
            tokenize=lambda t: [1] * len(t),
        )
        self.assertIsNone(spec)
        self.assertEqual(reason, "missing_route_system_prompt")


class OrnithPrefixHeadroomTests(unittest.TestCase):
    """The fixed count must sit inside the invariant region for every prompt
    the production builder can actually produce.

    `ROUTE_SYSTEM_PROMPT` interpolates the detected OS and shell, so the prompt
    is not one fixed string. Measured on the real Ornith tokenizer the
    invariant region is 925-927 tokens across every variant, leaving at least
    157 tokens of headroom over the fixed 768. This test pins the property that
    matters -- the variation is far from the boundary -- without needing the
    model, by checking the character spread the interpolation can introduce.
    """

    def test_os_and_shell_interpolation_barely_moves_the_prompt(self) -> None:
        from orbit.runtime.messages import _COMMAND_SYSTEM_TEMPLATE

        lengths = [
            len(_COMMAND_SYSTEM_TEMPLATE.format(os_name=os_name, shell_name=shell))
            for os_name, shell in (
                ("linux", "bash"), ("linux", "zsh"), ("linux", "sh"), ("linux", "fish"),
                ("macos", "zsh"), ("windows", "powershell.exe"), ("windows", "cmd.exe"),
                ("unknown", "unknown"),
            )
        ]

        # A handful of characters, against ~60 tokens of measured headroom.
        self.assertLess(max(lengths) - min(lengths), 64)

    def test_the_interpolation_sits_late_in_the_template(self) -> None:
        from orbit.runtime.messages import _COMMAND_SYSTEM_TEMPLATE

        position = _COMMAND_SYSTEM_TEMPLATE.index("Environment: OS=")

        # Well past the 768-token prefix region, so a longer shell name cannot
        # push user content forward into the captured prefix.
        self.assertGreater(position / len(_COMMAND_SYSTEM_TEMPLATE), 0.5)

    def test_a_shrunken_invariant_region_is_refused_not_trimmed(self) -> None:
        # The guarantee that makes the fixed count safe: if the invariant
        # region ever fell below it, derivation returns None instead of
        # silently capturing fewer tokens.
        def render(user: str) -> str:
            return "s" * 100 + user + "t" * 900

        full = render("actual request")
        spec, reason = derive_ornith_route_prefix_spec(
            system_prompt="s" * 100,
            full_prompt=full,
            full_tokens=[ord(c) for c in full],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )

        self.assertIsNone(spec)
        self.assertIsNotNone(reason)


class _FakeBridge:
    """Renders ChatML-shaped output: system turn, then the user turn."""

    def render(self, _context, messages, _tools, *, thinking: bool):
        assert not thinking
        return {
            "prompt": str(messages[0]["content"]) + "\n<user>" + str(messages[1]["content"]) + "</user>",
            "generation_prompt": "",
        }

    def free(self, _context) -> None:
        return None


def _profile(profile_id: str = ORNITH15_PROFILE_ID, **overrides) -> NativeModelProfile:
    base = dict(
        profile_id=profile_id,
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


class OrnithRoutePrefixClientTests(unittest.TestCase):
    """The planner and the per-profile registry, on the real client."""

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
            ornith_route_prefix_reuse_enabled=enabled,
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

    def _messages(self, user: str = "hello"):
        return [
            {"role": "system", "content": "s" * 900},
            {"role": "user", "content": user},
        ]

    def _plan(self, client, user: str = "hello"):
        messages = self._messages(user)
        prompt = client.apply_chat_template(messages, tools=None, thinking=False)
        return client._qwen_route_anchor_plan_for_prompt(
            messages, tools=None, thinking=False, prompt=prompt
        )

    # --- exact prefix ---------------------------------------------------
    def test_plan_is_produced_for_verified_ornith(self) -> None:
        client = self._client()
        plan = self._plan(client)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.profile_id, ORNITH15_PROFILE_ID)
        self.assertEqual(len(plan.prefix_tokens), ORNITH_ROUTE_PREFIX_TOKEN_COUNT)

    def test_prefix_is_an_exact_prefix_of_several_different_first_messages(self) -> None:
        client = self._client()
        reference = self._plan(client, "hello")
        assert reference is not None

        for user in ("hello", "what is a packer?", "ciao", "a", "x" * 200, "explain XOR"):
            with self.subTest(user=user[:20]):
                messages = self._messages(user)
                prompt = client.apply_chat_template(messages, tools=None, thinking=False)
                tokens = client.tokenize(prompt)
                self.assertEqual(
                    tuple(tokens[: len(reference.prefix_tokens)]),
                    tuple(reference.prefix_tokens),
                    "prewarm must be an exact token prefix of the real request",
                )

    def test_prefix_stops_before_any_user_token(self) -> None:
        client = self._client()
        plan = self._plan(client, "UNIQUEUSERMARKER")
        assert plan is not None
        prefix_text = "".join(chr(t) for t in plan.prefix_tokens)

        self.assertNotIn("UNIQUEUSERMARKER", prefix_text)
        self.assertNotIn("<user>", prefix_text)

    # --- invalidation ---------------------------------------------------
    def test_disabled_config_produces_no_plan(self) -> None:
        self.assertIsNone(self._plan(self._client(enabled=False)))

    def test_unverified_profile_produces_no_plan(self) -> None:
        self.assertIsNone(self._plan(self._client(profile=_profile(verified=False))))

    def test_profile_without_route_prefix_support_produces_no_plan(self) -> None:
        self.assertIsNone(
            self._plan(self._client(profile=_profile(route_prefix_reuse_supported=False)))
        )

    def test_foreign_profile_does_not_take_the_ornith_branch(self) -> None:
        client = self._client(profile=_profile(profile_id="some-other-profile"))
        self.assertIsNone(self._plan(client))

    def test_unverified_quantization_produces_no_plan(self) -> None:
        self.assertIsNone(self._plan(self._client(file_type="7")))

    def test_thinking_and_tools_produce_no_plan(self) -> None:
        client = self._client()
        messages = self._messages()
        prompt = client.apply_chat_template(messages, tools=None, thinking=False)

        self.assertIsNone(
            client._qwen_route_anchor_plan_for_prompt(
                messages, tools=None, thinking=True, prompt=prompt
            )
        )
        self.assertIsNone(
            client._qwen_route_anchor_plan_for_prompt(
                messages, tools=[{"a": 1}], thinking=False, prompt=prompt
            )
        )

    # --- per-profile isolation -----------------------------------------
    def test_each_profile_keeps_its_own_checkpoint_slot(self) -> None:
        client = self._client()
        ornith_state = PrefixAnchorState(prefix_hash="ornith", token_count=768, valid=True)
        coder_state = PrefixAnchorState(prefix_hash="coder", token_count=768, valid=True)

        client._set_qwen_route_prefix_state(ORNITH15_PROFILE_ID, ornith_state)
        client._set_qwen_route_prefix_state(QWEN3_CODER_PROFILE_ID, coder_state)

        self.assertIs(client._qwen_route_prefix_state_for_profile(ORNITH15_PROFILE_ID), ornith_state)
        self.assertIs(
            client._qwen_route_prefix_state_for_profile(QWEN3_CODER_PROFILE_ID), coder_state
        )
        self.assertIsNot(
            client._qwen_route_prefix_state_for_profile(ORNITH15_PROFILE_ID),
            client._qwen_route_prefix_state_for_profile(QWEN3_CODER_PROFILE_ID),
        )

    def test_spec_and_status_slots_are_also_per_profile(self) -> None:
        client = self._client()
        client._set_qwen_route_prefix_spec(ORNITH15_PROFILE_ID, "ornith-spec")  # type: ignore[arg-type]
        client._set_qwen_route_prefix_spec(QWEN3_CODER_PROFILE_ID, "coder-spec")  # type: ignore[arg-type]

        self.assertEqual(client._qwen_route_prefix_spec_for_profile(ORNITH15_PROFILE_ID), "ornith-spec")
        self.assertEqual(
            client._qwen_route_prefix_spec_for_profile(QWEN3_CODER_PROFILE_ID), "coder-spec"
        )
        self.assertIsNot(
            client._qwen_route_prefix_status_for_profile(ORNITH15_PROFILE_ID),
            client._qwen_route_prefix_status_for_profile(QWEN3_CODER_PROFILE_ID),
        )

    def test_reset_invalidates_the_ornith_checkpoint(self) -> None:
        client = self._client()
        client._set_qwen_route_prefix_state(
            ORNITH15_PROFILE_ID, PrefixAnchorState(prefix_hash="p", token_count=768, valid=True)
        )

        client._invalidate_qwen_route_prefix("session_reset")

        self.assertFalse(client._qwen_route_prefix_state_for_profile(ORNITH15_PROFILE_ID).valid)

    def test_blanket_invalidation_covers_every_profile(self) -> None:
        client = self._client()
        for profile_id in (ORNITH15_PROFILE_ID, QWEN3_CODER_PROFILE_ID):
            client._set_qwen_route_prefix_state(
                profile_id, PrefixAnchorState(prefix_hash="p", token_count=768, valid=True)
            )

        client._invalidate_qwen_route_prefix("model_reload")

        for profile_id in (ORNITH15_PROFILE_ID, QWEN3_CODER_PROFILE_ID):
            self.assertFalse(client._qwen_route_prefix_state_for_profile(profile_id).valid)

    def test_targeted_invalidation_leaves_other_profiles_alone(self) -> None:
        client = self._client()
        for profile_id in (ORNITH15_PROFILE_ID, QWEN3_CODER_PROFILE_ID):
            client._set_qwen_route_prefix_state(
                profile_id, PrefixAnchorState(prefix_hash="p", token_count=768, valid=True)
            )

        client._invalidate_qwen_route_prefix("x", profile_id=QWEN3_CODER_PROFILE_ID)

        self.assertTrue(client._qwen_route_prefix_state_for_profile(ORNITH15_PROFILE_ID).valid)
        self.assertFalse(client._qwen_route_prefix_state_for_profile(QWEN3_CODER_PROFILE_ID).valid)


class OrnithPrewarmDoesNotDisplaceRollingTests(unittest.TestCase):
    """A rolling checkpoint that can serve this prompt outranks the prewarm.

    The prewarm is a fixed 768-token head; a rolling checkpoint holds the
    conversation's own tokens and is never shorter. Letting the prewarm win
    would make later CHAT turns slower, which is the opposite of the point.
    """

    def _client(self):
        return OrnithRoutePrefixClientTests._client(OrnithRoutePrefixClientTests("run"))

    def test_a_usable_rolling_checkpoint_suppresses_the_prewarm_plan(self) -> None:
        from orbit.native_llama.rolling_route_anchor import (
            ROLLING_ROUTE_STRATEGY_ID,
            RollingRouteAnchorState,
            RollingRouteIdentity,
            rolling_route_reuse_start,
        )

        identity = RollingRouteIdentity(
            strategy_id=ROLLING_ROUTE_STRATEGY_ID,
            session_id="default",
            profile_id=ORNITH15_PROFILE_ID,
            model_id="/models/ornith.gguf",
            template_id="tpl",
            tool_schema_hash="tools",
            capability_summary_hash="caps",
            runtime_policy_hash="policy",
            native_version="libllama.so",
            tools_mode="on",
            reset_generation=0,
        )
        saved = [1, 2, 3, 4]
        state = RollingRouteAnchorState(
            identity=identity, tokens=list(saved), checkpoint_data=b"c", created_at_monotonic=1.0
        )

        # The condition the production chain uses to step over the prewarm.
        self.assertIsNotNone(rolling_route_reuse_start(state, saved + [5], identity))
        # And it declines when the rolling state cannot serve the prompt, so
        # the prewarm still gets its turn on a genuinely cold first request.
        self.assertIsNone(rolling_route_reuse_start(state, [9, 9, 9], identity))

    def _identity(self):
        from orbit.native_llama.rolling_route_anchor import (
            ROLLING_ROUTE_STRATEGY_ID,
            RollingRouteIdentity,
        )

        return RollingRouteIdentity(
            strategy_id=ROLLING_ROUTE_STRATEGY_ID,
            session_id="default",
            profile_id=ORNITH15_PROFILE_ID,
            model_id="/models/ornith.gguf",
            template_id="tpl",
            tool_schema_hash="tools",
            capability_summary_hash="caps",
            runtime_policy_hash="policy",
            native_version="libllama.so",
            tools_mode="on",
            reset_generation=0,
        )

    def test_a_usable_rolling_checkpoint_wins_over_the_prewarm(self) -> None:
        from orbit.native_llama.rolling_route_anchor import RollingRouteAnchorState

        client = self._client()
        identity = self._identity()
        saved = [1, 2, 3, 4]
        client._rolling_route_anchor_state = RollingRouteAnchorState(
            identity=identity, tokens=list(saved), checkpoint_data=b"c", created_at_monotonic=1.0
        )

        self.assertTrue(
            client._rolling_outranks_route_prefix(
                saved + [5], rolling_route_eligible=True, rolling_route_identity=identity
            ),
            "a rolling checkpoint that serves this prompt must outrank the prewarm",
        )

    def test_the_guard_is_consulted_before_the_prewarm_plan_is_used(self) -> None:
        """Production must ask the guard, not merely have it available."""
        client = self._client()
        asked: list[tuple] = []

        def guard(tokens, **kwargs):
            asked.append((tuple(tokens), kwargs))
            return False

        client._rolling_outranks_route_prefix = guard  # type: ignore[method-assign]
        client.tokenize = lambda text: [1, 2, 3]  # type: ignore[method-assign]
        client._route_anchor_plan = lambda *a, **k: None  # type: ignore[method-assign]
        # Past the "native client not loaded" precondition, so execution
        # actually reaches the guard before failing on the fake library.
        client._session.ctx_tgt = object()
        client._session.sampler = object()

        plan = SimpleNamespace(prefix_tokens=[1, 2], profile_id=ORNITH15_PROFILE_ID)
        with self.assertRaises(Exception):
            client._complete_prompt_standard("p", qwen_route_anchor_plan=plan)

        self.assertEqual(len(asked), 1, "the priority guard must be consulted")
        self.assertEqual(asked[0][0], (1, 2, 3))
        self.assertIn("rolling_route_eligible", asked[0][1])

    def test_prewarm_still_runs_when_rolling_cannot_serve_the_prompt(self) -> None:
        from orbit.native_llama.rolling_route_anchor import (
            ROLLING_ROUTE_STRATEGY_ID,
            RollingRouteAnchorState,
            RollingRouteIdentity,
            rolling_route_reuse_start,
        )

        identity = RollingRouteIdentity(
            strategy_id=ROLLING_ROUTE_STRATEGY_ID,
            session_id="default",
            profile_id=ORNITH15_PROFILE_ID,
            model_id="/models/ornith.gguf",
            template_id="tpl",
            tool_schema_hash="tools",
            capability_summary_hash="caps",
            runtime_policy_hash="policy",
            native_version="libllama.so",
            tools_mode="on",
            reset_generation=0,
        )
        cold = RollingRouteAnchorState()

        # Nothing saved yet: the guard must not fire, leaving the prewarm to
        # do its job on the genuinely cold first turn.
        self.assertIsNone(rolling_route_reuse_start(cold, [1, 2, 3], identity))


class OrnithAnalysisIsolationTests(unittest.TestCase):
    def test_analysis_rolling_state_is_a_separate_attribute(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src/orbit/native_llama/client.py"
        text = source.read_text(encoding="utf-8")

        self.assertIn("_rolling_analysis_anchor_state", text)
        self.assertIn("_ornith_route_prefix_anchor_state", text)
        # The prewarm slot must never be read by the analysis rolling path.
        analysis_fn = text[text.index("def _rolling_anchor_state_for") :][:600]
        self.assertNotIn("_ornith_route_prefix_anchor_state", analysis_fn)

    def test_route_prefix_registry_never_returns_the_analysis_slot(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src/orbit/native_llama/client.py"
        text = source.read_text(encoding="utf-8")
        registry = text[text.index("def _qwen_route_prefix_state_for_profile") :][:700]

        self.assertNotIn("_rolling_analysis_anchor_state", registry)
        self.assertNotIn("_rolling_route_anchor_state", registry)


class OrnithPrewarmAddsNoModelCallTests(unittest.TestCase):
    """Capture is prefill-only: it must never sample or generate."""

    def _block(self) -> str:
        source = Path(__file__).resolve().parents[1] / "src/orbit/native_llama/client.py"
        text = source.read_text(encoding="utf-8")
        start = text.index("def capture_qwen3_coder_route_prefix_prefill_only")
        return text[start : text.index("    def ", start + 50)]

    def test_capture_never_generates(self) -> None:
        block = self._block()

        for forbidden in ("_generate_from_current_context", "llama_sampler_sample", "on_token("):
            self.assertNotIn(forbidden, block, "prewarm capture must not generate a token")

    def test_capture_requires_a_clean_prefill_boundary(self) -> None:
        block = self._block()

        self.assertIn("processed != prefix_token_count or reused != 0", block)
        self.assertIn("active_context_present", block)

    def test_capture_reports_prefill_only_result(self) -> None:
        block = self._block()

        self.assertIn("restore_ready=True", block)
        self.assertIn("decode_calls=", block)


class OrnithStartupPrewarmWiringTests(unittest.TestCase):
    def test_startup_prewarm_reaches_the_capture_for_ornith(self) -> None:
        import orbit.native_server.app as app

        source = Path(app.__file__).read_text(encoding="utf-8")
        block = source[source.index("def prewarm_startup_route_prefix") :][:2600]

        self.assertIn("ORNITH15_PROFILE_ID", block)
        # Ornith must be handled by the ChatML capture branch, before the
        # Gemma-only check that would otherwise skip it as ineligible.
        self.assertLess(
            block.index("ORNITH15_PROFILE_ID"),
            block.index("gemma_prefix_reuse_supported"),
            "Ornith must not fall through to the Gemma ineligibility branch",
        )

    def test_route_prompt_is_unchanged_by_this_mission(self) -> None:
        import hashlib

        self.assertEqual(
            hashlib.sha256(ROUTE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "d38e293a1d8fc0efb5371cff08bb5870ffc4faa6b96b889ff2af54ba2b66a38d",
        )


if __name__ == "__main__":
    unittest.main()
