"""Eager ANALYSIS prewarm: startup captures the ANALYSIS prefix beside CHAT.

Without this the first ANALYSIS step of a server's life is always cold -- the
checkpoint it would restore is the one that step captures. These tests drive
the real startup entry point and the real derivation, because the properties
that matter live in the wiring: that the captured prefix is derived from the
stable production contract and nothing else, that it is token-identical to the
prefix the lazy path derives from a real request, that the two lineages cannot
overwrite or corrupt one another, and that a rolling ANALYSIS checkpoint still
outranks the prewarm.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from orbit.native_llama.client import NativeLlamaClient, NativeRoutePrefixPrefillResult
from orbit.native_llama.model_profiles import ORNITH15_PROFILE_ID, QWEN3_CODER_PROFILE_ID
from orbit.native_llama.ornith_analysis_prefix import (
    ORNITH_ANALYSIS_LINEAGE_ID,
    ORNITH_ANALYSIS_PREFIX_ENV,
    ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT,
    ORNITH_ANALYSIS_PREWARM_ENV,
    derive_ornith_analysis_prefix_spec,
    resolve_ornith_analysis_prefix_prewarm,
    resolve_ornith_analysis_prefix_reuse,
)
from orbit.native_llama.prefix_anchor import PrefixAnchorState
from orbit.native_server import app as app_module
from orbit.native_server.app import (
    prewarm_startup_analysis_prefix,
    prewarm_startup_route_prefix,
)
from orbit.runtime.analysis_runtime import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_TOOL_SCHEMA

from tests.test_native_server_bootstrap import _FakeNativeClient

# Pinned literals. The prewarm captures this contract verbatim; if either
# changes the captured prefix is a different token sequence and the identity
# these tests assert no longer describes what ships.
ANALYSIS_SYSTEM_PROMPT_SHA256 = "88f234df8f26e55b0a7d380e1df4672d87a4a8e719c7342424a5ff4b819c122c"
ANALYSIS_TOOL_SCHEMA_SHA256 = "57710e9ee2c19683cb74b854d5b6f0714fb4802ad1a51971e43cd7f6d080f2a4"

# Startup prewarm enabled, ANALYSIS eager explicitly requested. Eager capture is
# opt-in, so every test that expects one must ask for it exactly as an operator
# would; a bare startup env is the normal-production case and is covered below.
STARTUP = {
    "ORBIT_KV_PREFIX_PREWARM": "startup",
    ORNITH_ANALYSIS_PREWARM_ENV: "1",
}

# Normal production: startup prewarm on, nothing else set.
DEFAULT_STARTUP = {"ORBIT_KV_PREFIX_PREWARM": "startup"}


def _ornith_client() -> _FakeNativeClient:
    client = _FakeNativeClient()
    client.config = SimpleNamespace(
        qwen3_coder_route_prefix_reuse_enabled=True,
        ornith_route_prefix_reuse_enabled=True,
        ornith_analysis_prefix_reuse_enabled=True,
    )
    client.model_profile = SimpleNamespace(
        profile_id=ORNITH15_PROFILE_ID,
        gemma_prefix_reuse_supported=False,
        verified=True,
        route_prefix_reuse_supported=True,
    )
    return client


class StartupCapturesTheStableContractTests(unittest.TestCase):
    """What startup captures comes from production constants, nothing else."""

    def test_contract_literals_are_unchanged(self) -> None:
        self.assertEqual(
            hashlib.sha256(ANALYSIS_SYSTEM_PROMPT.encode()).hexdigest(),
            ANALYSIS_SYSTEM_PROMPT_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(json.dumps(ANALYSIS_TOOL_SCHEMA, sort_keys=True).encode()).hexdigest(),
            ANALYSIS_TOOL_SCHEMA_SHA256,
        )

    @mock.patch.dict("os.environ", STARTUP, clear=True)
    def test_startup_captures_the_analysis_contract_and_the_real_schema(self) -> None:
        client = _ornith_client()

        result = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(result.succeeded)
        self.assertTrue(result.restore_ready)
        self.assertEqual(result.prefix_token_count, ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT)
        self.assertEqual(client.analysis_capture_calls, 1)
        # The system prompt is the production contract verbatim, and the tool
        # surface is the real execute_analysis schema -- not a stand-in.
        self.assertEqual(client.captured_system_prompts, [ANALYSIS_SYSTEM_PROMPT])
        self.assertEqual(client.captured_tools, [[ANALYSIS_TOOL_SCHEMA]])
        self.assertEqual(client.captured_lineages, ["analysis"])

    @mock.patch.dict("os.environ", STARTUP, clear=True)
    def test_no_volatile_or_synthetic_content_is_passed_to_the_capture(self) -> None:
        client = _ornith_client()

        prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        payload = json.dumps(
            {"system": client.captured_system_prompts, "tools": client.captured_tools}
        )
        for marker in (
            "sha256 ",
            "Artifact under analysis",
            "A-orbit",
            "Z-orbit",
            "boundary",
            "/workspace/work/",
        ):
            self.assertNotIn(marker, payload, f"volatile/synthetic marker {marker!r} reached capture")

    @mock.patch.dict("os.environ", STARTUP, clear=True)
    def test_capture_is_requested_on_the_analysis_lineage(self) -> None:
        client = _ornith_client()

        prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        self.assertEqual(client.captured_lineages, ["analysis"])
        self.assertEqual(client.qwen3_coder_capture_calls, 0)


class StartupAndLazyFeedTheDerivationTheSameContractTests(unittest.TestCase):
    """Both entry points must hand the derivation the same stable contract.

    Driven through the real derivation with a deterministic surrogate template
    and tokenizer. This is deliberately narrower than its subject: the surrogate
    renders a tools block, so a tools divergence is detectable here, but only the
    canary proves the identity on the real template with the real tokenizer.
    What is guarded here is that startup and lazy cannot silently diverge in what
    they feed the derivation.
    """

    def _render(self, system: str, tools=None):
        # The tool schema renders ahead of the user turn, exactly as the real
        # template places it, so a divergence in the tool surface moves the
        # captured prefix instead of hiding behind a constant system string.
        block = json.dumps(tools or [], sort_keys=True)

        def render(user: str) -> str:
            return system + "\n<tools>" + block + "</tools>\n<user>" + user + "</user>"

        return render

    def _derive(self, system: str, full: str, tools=None):
        render = self._render(system, tools)
        return derive_ornith_analysis_prefix_spec(
            system_prompt=system,
            full_prompt=full,
            full_tokens=[ord(c) for c in full],
            render_reference=render,
            tokenize=lambda t: [ord(c) for c in t],
        )

    def test_startup_fixture_and_a_real_request_derive_the_same_prefix(self) -> None:
        system = "s" * 200
        tools = [ANALYSIS_TOOL_SCHEMA]
        startup_spec, startup_reason = self._derive(
            system, self._render(system, tools)("A-orbit-qwen-route-boundary"), tools
        )
        lazy_spec, lazy_reason = self._derive(
            system,
            self._render(system, tools)(
                "Artifact under analysis: the file /workspace/input "
                "(4096 bytes, sha256 deadbeef). decode the payload"
            ),
            tools,
        )

        self.assertIsNone(startup_reason)
        self.assertIsNone(lazy_reason)
        assert startup_spec is not None and lazy_spec is not None
        self.assertEqual(startup_spec.prefix_tokens, lazy_spec.prefix_tokens)
        self.assertEqual(startup_spec.prefix_token_hash, lazy_spec.prefix_token_hash)
        self.assertEqual(startup_spec.system_prompt_hash, lazy_spec.system_prompt_hash)
        self.assertEqual(startup_spec.invariant_text_hash, lazy_spec.invariant_text_hash)
        self.assertEqual(startup_spec.next_boundary_token, lazy_spec.next_boundary_token)

    def test_a_different_contract_derives_a_different_prefix(self) -> None:
        """Identity is content-derived, not asserted: change it and it moves."""
        a, _ = self._derive("s" * 500, self._render("s" * 500)("x"))
        b, _ = self._derive("t" * 500, self._render("t" * 500)("x"))
        assert a is not None and b is not None
        self.assertNotEqual(a.prefix_token_hash, b.prefix_token_hash)

    def test_a_different_tool_surface_derives_a_different_prefix(self) -> None:
        """The schema is inside the captured region, so changing it moves it.

        This is the divergence a constant-system-string surrogate cannot see:
        if startup and the lazy path ever disagreed about the tool surface, the
        captured prefix would not be the one being served.
        """
        system = "s" * 200
        real = [ANALYSIS_TOOL_SCHEMA]
        altered = [json.loads(json.dumps(ANALYSIS_TOOL_SCHEMA))]
        altered[0]["function"]["description"] = "a different description entirely"

        a, a_reason = self._derive(system, self._render(system, real)("x"), real)
        b, b_reason = self._derive(system, self._render(system, altered)("x"), altered)

        self.assertIsNone(a_reason)
        self.assertIsNone(b_reason)
        assert a is not None and b is not None
        self.assertNotEqual(a.prefix_token_hash, b.prefix_token_hash)

    def test_a_startup_prefix_is_refused_against_a_foreign_tool_surface(self) -> None:
        """The exact-prefix gate, exercised: mismatched tools must not derive.

        A prefix derived from reference renders carrying one tool surface, checked
        against a prompt rendered with another, must be refused. Which gate refuses
        it depends on how the surfaces differ -- a same-length change trips the
        token check, a length-changing one trips the text-boundary check first --
        so both cases are exercised and the refusal, not the reason, is the
        property being asserted. In this surrogate the text-boundary check sees
        the divergence first; on the real template the token check can be the one
        to fire. Either is a refusal, which is what must hold.
        """
        system = "s" * 200
        real = [ANALYSIS_TOOL_SCHEMA]

        renamed = [json.loads(json.dumps(ANALYSIS_TOOL_SCHEMA))]
        renamed[0]["function"]["name"] = "execute_something_else"

        same_length = [json.loads(json.dumps(ANALYSIS_TOOL_SCHEMA))]
        original_name = same_length[0]["function"]["name"]
        same_length[0]["function"]["name"] = "X" * len(original_name)

        refusals = set()
        for altered in (renamed, same_length):
            served = self._render(system, altered)("x")
            spec, reason = derive_ornith_analysis_prefix_spec(
                system_prompt=system,
                full_prompt=served,
                full_tokens=[ord(c) for c in served],
                render_reference=self._render(system, real),
                tokenize=lambda t: [ord(c) for c in t],
            )
            self.assertIsNone(spec, "a foreign tool surface must never derive a prefix")
            refusals.add(reason)

        # Both refusals come from the exact-prefix family, never from success.
        self.assertTrue(
            refusals
            <= {"production_prefix_mismatch", "invariant_text_boundary_mismatch"},
            f"unexpected refusal reasons: {refusals}",
        )


class BothSlotsCoexistTests(unittest.TestCase):
    """CHAT and ANALYSIS are captured independently and cannot collide."""

    @mock.patch.dict("os.environ", STARTUP, clear=True)
    def test_startup_captures_chat_and_analysis(self) -> None:
        client = _ornith_client()

        chat = prewarm_startup_route_prefix(client)  # type: ignore[arg-type]
        analysis = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(chat.succeeded)
        self.assertTrue(analysis.succeeded)
        self.assertEqual(client.qwen3_coder_capture_calls, 1)
        self.assertEqual(client.analysis_capture_calls, 1)
        self.assertEqual(client.captured_lineages, ["chat", "analysis"])
        # Different prefixes, different sizes: one did not stand in for the other.
        self.assertNotEqual(chat.prefix_hash, analysis.prefix_hash)
        self.assertNotEqual(chat.prefix_token_count, analysis.prefix_token_count)

    @mock.patch.dict("os.environ", STARTUP, clear=True)
    def test_analysis_capture_failure_cannot_corrupt_the_chat_result(self) -> None:
        client = _ornith_client()
        chat = prewarm_startup_route_prefix(client)  # type: ignore[arg-type]
        client.raise_on_analysis_capture = True

        analysis = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(chat.succeeded)
        self.assertTrue(chat.restore_ready)
        self.assertFalse(analysis.succeeded)
        self.assertFalse(analysis.restore_ready)
        self.assertEqual(analysis.failed_reason, "startup_prewarm_failed:RuntimeError")

    @mock.patch.dict("os.environ", STARTUP, clear=True)
    def test_chat_capture_failure_does_not_prevent_the_analysis_capture(self) -> None:
        client = _ornith_client()
        client.raise_on_capture = True
        chat = prewarm_startup_route_prefix(client)  # type: ignore[arg-type]
        client.raise_on_capture = False

        analysis = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        self.assertFalse(chat.succeeded)
        self.assertTrue(analysis.succeeded)
        self.assertTrue(analysis.restore_ready)

    def test_the_two_lineages_address_different_client_slots(self) -> None:
        """A real client keeps the ANALYSIS state apart from the CHAT one."""
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._ornith_route_prefix_anchor_state = PrefixAnchorState(
            prefix_hash="chat", token_count=768, valid=True
        )
        client._ornith_analysis_prefix_anchor_state = PrefixAnchorState(
            prefix_hash="analysis", token_count=384, valid=True
        )

        chat_state = NativeLlamaClient._qwen_route_prefix_state_for_profile(
            client, ORNITH15_PROFILE_ID
        )
        analysis_state = NativeLlamaClient._qwen_route_prefix_state_for_profile(
            client, ORNITH_ANALYSIS_LINEAGE_ID
        )

        self.assertEqual(chat_state.prefix_hash, "chat")
        self.assertEqual(analysis_state.prefix_hash, "analysis")

        NativeLlamaClient._set_qwen_route_prefix_state(
            client, ORNITH_ANALYSIS_LINEAGE_ID, PrefixAnchorState(prefix_hash="analysis-2", valid=True)
        )
        self.assertEqual(client._ornith_route_prefix_anchor_state.prefix_hash, "chat")
        self.assertEqual(client._ornith_analysis_prefix_anchor_state.prefix_hash, "analysis-2")


class EligibilityTests(unittest.TestCase):
    """The ANALYSIS prewarm honours every gate the CHAT one does."""

    @mock.patch.dict("os.environ", {"ORBIT_KV_PREFIX_PREWARM": "off"}, clear=True)
    def test_prewarm_off_skips_without_capture(self) -> None:
        client = _ornith_client()
        result = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "disabled")
        self.assertEqual(client.analysis_capture_calls, 0)

    @mock.patch.dict("os.environ", {**STARTUP, "ORBIT_TOOLS": "off"}, clear=True)
    def test_tools_off_skips_without_capture(self) -> None:
        client = _ornith_client()
        result = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "tools_disabled")
        self.assertEqual(client.analysis_capture_calls, 0)

    @mock.patch.dict("os.environ", {**STARTUP, "ORBIT_KV_PREFIX_ANCHOR": "off"}, clear=True)
    def test_anchor_off_skips_without_capture(self) -> None:
        client = _ornith_client()
        result = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "anchor_disabled")
        self.assertEqual(client.analysis_capture_calls, 0)

    @mock.patch.dict("os.environ", STARTUP, clear=True)
    def test_a_foreign_profile_keeps_the_lazy_capture(self) -> None:
        client = _ornith_client()
        client.model_profile = SimpleNamespace(
            profile_id=QWEN3_CODER_PROFILE_ID,
            gemma_prefix_reuse_supported=False,
            verified=True,
            route_prefix_reuse_supported=True,
        )

        result = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "model_profile_ineligible")
        self.assertEqual(client.analysis_capture_calls, 0)


class EagerCaptureIsOptInTests(unittest.TestCase):
    """Normal startup must not pay the eager ANALYSIS cost.

    The capture is ~11s of startup prefill and ~74MB of resident checkpoint,
    measured on the qualified Ornith profile. A server that only ever chats
    would pay both for nothing, so an operator asks for it rather than opting
    out -- and with it off, the lazy first-use capture is untouched.
    """

    def test_the_switch_defaults_to_off(self) -> None:
        config = resolve_ornith_analysis_prefix_prewarm({})
        self.assertFalse(config.enabled)
        self.assertEqual(config.source, "default")

    def test_explicit_opt_in_and_explicit_disable(self) -> None:
        self.assertTrue(
            resolve_ornith_analysis_prefix_prewarm({ORNITH_ANALYSIS_PREWARM_ENV: "1"}).enabled
        )
        self.assertFalse(
            resolve_ornith_analysis_prefix_prewarm({ORNITH_ANALYSIS_PREWARM_ENV: "0"}).enabled
        )

    def test_an_invalid_value_fails_closed_with_a_validation_error(self) -> None:
        config = resolve_ornith_analysis_prefix_prewarm({ORNITH_ANALYSIS_PREWARM_ENV: "yes please"})
        self.assertFalse(config.enabled)
        self.assertEqual(config.validation_error, "invalid_ornith_analysis_prefix_prewarm_value")
        self.assertEqual(config.source, "stable")

    def test_the_prewarm_switch_is_distinct_from_the_reuse_switch(self) -> None:
        """Two questions, two switches: may it be reused, and who pays to capture."""
        self.assertNotEqual(ORNITH_ANALYSIS_PREWARM_ENV, ORNITH_ANALYSIS_PREFIX_ENV)
        # Reuse stays on by default; only the eager capture is opt-in.
        self.assertTrue(resolve_ornith_analysis_prefix_reuse({}).enabled)
        self.assertFalse(resolve_ornith_analysis_prefix_prewarm({}).enabled)

    @mock.patch.dict("os.environ", DEFAULT_STARTUP, clear=True)
    def test_default_startup_does_not_capture_analysis(self) -> None:
        client = _ornith_client()

        result = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "analysis_prewarm_not_requested")
        self.assertEqual(client.analysis_capture_calls, 0)
        self.assertFalse(result.attempted)

    @mock.patch.dict("os.environ", DEFAULT_STARTUP, clear=True)
    def test_default_startup_still_captures_chat(self) -> None:
        """The ANALYSIS gate must not touch CHAT."""
        client = _ornith_client()

        chat = prewarm_startup_route_prefix(client)  # type: ignore[arg-type]
        analysis = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(chat.succeeded)
        self.assertTrue(chat.restore_ready)
        self.assertEqual(client.qwen3_coder_capture_calls, 1)
        self.assertTrue(analysis.skipped)
        self.assertEqual(client.captured_lineages, ["chat"])

    @mock.patch.dict(
        "os.environ",
        {**DEFAULT_STARTUP, ORNITH_ANALYSIS_PREWARM_ENV: "0"},
        clear=True,
    )
    def test_explicit_disable_keeps_analysis_lazy(self) -> None:
        client = _ornith_client()

        result = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "analysis_prewarm_not_requested")
        self.assertEqual(client.analysis_capture_calls, 0)

    @mock.patch.dict(
        "os.environ",
        {**DEFAULT_STARTUP, ORNITH_ANALYSIS_PREWARM_ENV: "not-a-value"},
        clear=True,
    )
    def test_an_invalid_value_does_not_capture(self) -> None:
        client = _ornith_client()

        result = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(result.skipped)
        self.assertEqual(client.analysis_capture_calls, 0)

    @mock.patch.dict("os.environ", STARTUP, clear=True)
    def test_opt_in_captures_analysis(self) -> None:
        client = _ornith_client()

        result = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        self.assertTrue(result.succeeded)
        self.assertEqual(result.prefix_token_count, ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT)
        self.assertEqual(client.analysis_capture_calls, 1)

    @mock.patch.dict(
        "os.environ",
        {**DEFAULT_STARTUP, ORNITH_ANALYSIS_PREFIX_ENV: "0", ORNITH_ANALYSIS_PREWARM_ENV: "1"},
        clear=True,
    )
    def test_reuse_disabled_still_refuses_the_eager_capture(self) -> None:
        """Asking for eager capture cannot override the reuse kill switch."""
        client = _ornith_client()
        client.config.ornith_analysis_prefix_reuse_enabled = False

        result = prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        self.assertFalse(result.succeeded)
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "route_prefix_reuse_disabled")

    @mock.patch.dict("os.environ", DEFAULT_STARTUP, clear=True)
    def test_the_lazy_capture_path_still_plans_when_eager_is_off(self) -> None:
        """With eager off, the first analysis step still derives its prefix.

        Behavioural, through the real planner: gating startup must not disable
        the lazy capture, or the first analysis would be cold *and* no later
        session would ever restore. Asserting only that the client's source does
        not mention the startup switch would not establish this -- a planner
        broken by any other means would still pass that check.
        """
        from tests.test_ornith_analysis_prefix import AnalysisPrefixClientTests

        client = AnalysisPrefixClientTests._client(AnalysisPrefixClientTests("run"))
        messages = [
            {"role": "system", "content": "s" * 500},
            {
                "role": "user",
                "content": "Artifact under analysis: /workspace/input (21 bytes, sha256 abc).",
            },
        ]
        tools = [{"name": "execute_analysis"}]
        prompt = client.apply_chat_template(messages, tools=tools, thinking=False)

        plan = client._qwen_route_anchor_plan_for_prompt(
            messages, tools=tools, thinking=False, prompt=prompt, analysis_lineage=True
        )

        self.assertIsNotNone(plan, "lazy analysis planning must survive the eager gate")
        assert plan is not None
        self.assertEqual(plan.profile_id, ORNITH_ANALYSIS_LINEAGE_ID)
        self.assertEqual(len(plan.prefix_tokens), ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT)

    def test_the_client_never_consults_the_startup_switch(self) -> None:
        """A tripwire against a future edit wiring the gate into the lazy path.

        Deliberately a source-level check, and deliberately not the only cover
        for the lazy path -- the test above is what proves the planner works.
        """
        for member in (
            NativeLlamaClient._qwen_route_anchor_plan_for_prompt,
            NativeLlamaClient.capture_qwen3_coder_route_prefix_prefill_only,
        ):
            source = inspect.getsource(member)
            self.assertNotIn(ORNITH_ANALYSIS_PREWARM_ENV, source)
            self.assertNotIn("resolve_ornith_analysis_prefix_prewarm", source)


class WiringTests(unittest.TestCase):
    """Startup actually calls the ANALYSIS prewarm, and calls it correctly."""

    def _run(self, *, interrupt_during_chat: bool = False):
        """Drive the real run_server through a fake Ornith client.

        The fake reports the Ornith profile from load(), because that is what
        selects the cancellable startup branch this change extends -- a fake
        without a profile takes the other branch entirely and would leave the
        new call unexercised while still passing.
        """
        import io

        from tests.test_native_server_bootstrap import _FakeHTTPServer

        class _FakeOrnithClient(_FakeNativeClient):
            def load(self) -> None:
                super().load()
                self.model_profile = SimpleNamespace(
                    profile_id=ORNITH15_PROFILE_ID,
                    gemma_prefix_reuse_supported=False,
                    verified=True,
                    route_prefix_reuse_supported=True,
                )

        _FakeNativeClient.instances.clear()
        _FakeHTTPServer.instances.clear()
        order: list[str] = []

        real_chat = app_module.prewarm_startup_route_prefix
        real_analysis = app_module.prewarm_startup_analysis_prefix

        def chat(client):
            order.append("chat")
            if interrupt_during_chat:
                # Deliver SIGINT to the handler run_server installed, the way the
                # signal module would, rather than to the test process itself.
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return real_chat(client)

        def analysis(client):
            order.append("analysis")
            return real_analysis(client)

        with (
            mock.patch(
                "orbit.native_server.app.resolve_bootstrap_paths",
                return_value=SimpleNamespace(model=Path("/models/valid.gguf")),
            ),
            mock.patch("orbit.native_server.app.NativeLlamaClient", _FakeOrnithClient),
            mock.patch("orbit.native_server.app.ThreadingHTTPServer", _FakeHTTPServer),
            mock.patch("orbit.native_server.app.prewarm_startup_route_prefix", chat),
            mock.patch("orbit.native_server.app.prewarm_startup_analysis_prefix", analysis),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            code = app_module.run_server(["--model", "/models/valid.gguf"])
        return code, order, _FakeNativeClient.instances[-1]

    @mock.patch.dict("os.environ", DEFAULT_STARTUP, clear=True)
    def test_run_server_default_captures_chat_only(self) -> None:
        """Normal production startup: CHAT eager, ANALYSIS lazy."""
        code, order, client = self._run()

        self.assertEqual(code, 0)
        # The ANALYSIS prewarm is still called -- it is the function that
        # decides -- but it must decline, so no capture is performed.
        self.assertEqual(order, ["chat", "analysis"])
        self.assertEqual(client.captured_lineages, ["chat"])
        self.assertEqual(client.analysis_capture_calls, 0)
        self.assertEqual(client.qwen3_coder_capture_calls, 1)

    @mock.patch.dict("os.environ", STARTUP, clear=True)
    def test_run_server_captures_chat_then_analysis(self) -> None:
        code, order, client = self._run()

        self.assertEqual(code, 0)
        # Behavioural, not textual: startup really performed both captures, and
        # the CHAT one was not displaced.
        self.assertEqual(order, ["chat", "analysis"])
        self.assertEqual(client.captured_lineages, ["chat", "analysis"])
        self.assertEqual(client.analysis_capture_calls, 1)

    @mock.patch.dict("os.environ", STARTUP, clear=True)
    def test_run_server_skips_the_analysis_prewarm_when_startup_is_interrupted(self) -> None:
        """A Ctrl-C during the CHAT prewarm must not be followed by more prefill."""
        code, order, client = self._run(interrupt_during_chat=True)

        self.assertEqual(code, 130)
        self.assertEqual(order, ["chat"])
        self.assertEqual(client.analysis_capture_calls, 0)
        self.assertTrue(client.closed)

    def test_the_capture_call_names_the_analysis_lineage_and_the_real_schema(self) -> None:
        source = inspect.getsource(app_module.prewarm_startup_analysis_prefix)
        tree = ast.parse(source.strip())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "capture_qwen3_coder_route_prefix_prefill_only"
        ]
        self.assertEqual(len(calls), 1)
        kwargs = {kw.arg: kw for kw in calls[0].keywords}
        self.assertIn("analysis_lineage", kwargs)
        self.assertIs(kwargs["analysis_lineage"].value.value, True)
        self.assertEqual(kwargs["system_prompt"].value.id, "ANALYSIS_SYSTEM_PROMPT")
        self.assertIn("tools", kwargs)
        self.assertEqual(kwargs["tools"].value.elts[0].id, "ANALYSIS_TOOL_SCHEMA")

    def test_startup_does_not_build_a_user_message(self) -> None:
        """No synthetic request is constructed here; the client owns the fixture."""
        source = inspect.getsource(app_module.prewarm_startup_analysis_prefix)
        self.assertNotIn('"role"', source)
        self.assertNotIn("'role'", source)


class DiagnosticsTests(unittest.TestCase):
    """Two prewarms emit to one event stream, so the event says which is which."""

    def _emitted(self, run) -> list[dict]:
        """Capture what actually reaches the diagnostics sink.

        Patching `emit_route_prefix_prewarm_event` would prove nothing: that
        function builds a fixed-allowlist event and drops unknown keys, so a
        lineage the builder does not carry would still appear in a captured
        call and vanish from the real stream. This patches the sink instead.
        """
        lines: list[dict] = []
        with mock.patch("orbit.native_llama.kv_diag._emit", lines.append):
            run()
        return lines

    @mock.patch.dict(
        "os.environ", {**STARTUP, "ORBIT_KV_DIAG": "1"}, clear=True
    )
    def test_each_prewarm_event_names_its_lineage(self) -> None:
        client = _ornith_client()

        def run() -> None:
            prewarm_startup_route_prefix(client)  # type: ignore[arg-type]
            prewarm_startup_analysis_prefix(client)  # type: ignore[arg-type]

        events = [
            e for e in self._emitted(run) if e.get("event") == "kv_diag_route_prefix_prewarm"
        ]

        self.assertEqual(len(events), 2)
        self.assertEqual([e["prewarm_lineage"] for e in events], ["chat", "analysis"])
        self.assertEqual(events[1]["prewarm_prefix_token_count"], ORNITH_ANALYSIS_PREFIX_TOKEN_COUNT)

    @mock.patch.dict(
        "os.environ", {**STARTUP, "ORBIT_KV_DIAG": "1"}, clear=True
    )
    def test_a_caller_that_names_no_lineage_still_reads_as_chat(self) -> None:
        """The field defaults rather than appearing empty for existing callers."""
        from orbit.native_llama.kv_diag import emit_route_prefix_prewarm_event

        events = self._emitted(lambda: emit_route_prefix_prewarm_event({}))

        self.assertEqual(events[0]["prewarm_lineage"], "chat")


class RollingStillOutranksTests(unittest.TestCase):
    """A rolling ANALYSIS checkpoint remains the better restore."""

    def test_rolling_outranks_the_prewarm_when_it_can_serve_the_prompt(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        identity = object()
        with mock.patch(
            "orbit.native_llama.client.rolling_route_reuse_start", return_value=7
        ), mock.patch.object(
            NativeLlamaClient, "_rolling_anchor_state_for", lambda self, i: object()
        ):
            self.assertTrue(
                NativeLlamaClient._rolling_outranks_route_prefix(
                    client,
                    [1, 2, 3],
                    rolling_route_eligible=True,
                    rolling_route_identity=identity,  # type: ignore[arg-type]
                )
            )

    def test_the_prewarm_still_runs_when_rolling_cannot_serve_the_prompt(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        with mock.patch(
            "orbit.native_llama.client.rolling_route_reuse_start", return_value=None
        ), mock.patch.object(
            NativeLlamaClient, "_rolling_anchor_state_for", lambda self, i: object()
        ):
            self.assertFalse(
                NativeLlamaClient._rolling_outranks_route_prefix(
                    client,
                    [1, 2, 3],
                    rolling_route_eligible=True,
                    rolling_route_identity=object(),  # type: ignore[arg-type]
                )
            )

    def test_an_ineligible_rolling_state_never_outranks(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        self.assertFalse(
            NativeLlamaClient._rolling_outranks_route_prefix(
                client, [1, 2, 3], rolling_route_eligible=False, rolling_route_identity=object()  # type: ignore[arg-type]
            )
        )


if __name__ == "__main__":
    unittest.main()
