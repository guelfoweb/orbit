"""Rolling ANALYSIS KV for consecutive STEP turns, through the real wiring.

Stage A (#Stage-A merge 99d63bf) recovered the rolling ANALYSIS anchor for
the control turns and their repairs. The STEP turn stayed cold for two
measured reasons: the controller renders every STEP as the committed history
followed by ONE transient user turn -- the per-question guidance, replaced on
the next STEP -- so a checkpoint taken after it is never a prefix of the next
STEP; and a FINISH control turn runs between any two STEPs, evicting the STEP
checkpoint from the single analysis slot before the next STEP could use it.

Stage B answers both with the smallest extension of the existing machinery: a
STEP checkpoint is taken BEFORE the last user turn, and it lives in a slot of
its own, addressed -- like the other two -- by the identity's `strategy_id`.

These tests drive the production methods rather than the anchor module in
isolation: the boundary block inside `_complete_prompt_standard` is lifted
from its source and executed against recorded inputs, the restore goes
through `_prepare_memory_with_ornith_rolling_route_anchor`, the head is
rendered by `_step_boundary_head` over a recording renderer, `complete_chat`
is driven to the prefill, and the gate is the backend's own. Every safety
test asserts on the STEP slot's own state and on WHICH bytes were restored,
never only on a call count.
"""

from __future__ import annotations

import gc
import inspect
import json
import sys
import textwrap
import threading
import unittest
import weakref
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import orbit.backend.llama_server as llama_server
import orbit.native_llama.client as client_module
from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.model_profiles import ORNITH15_PROFILE_ID
from orbit.native_llama.rolling_anchor_store import RollingAnchorStore
from orbit.native_llama.rolling_route_anchor import (
    ROLLING_ANALYSIS_STRATEGY_ID,
    ROLLING_ROUTE_STRATEGY_ID,
    ROLLING_STEP_STRATEGY_ID,
    RollingRouteAnchorState,
    RollingRouteIdentity,
    rolling_capture_boundary,
    rolling_route_reuse_start,
    rolling_step_boundary,
)
from orbit.runtime.analysis_runtime import (
    ANALYSIS_COVER_PHASE,
    ANALYSIS_FINISH_PHASE,
    ANALYSIS_PLAN_PHASE,
    ANALYSIS_REPORT_PHASE,
    ANALYSIS_STEP_PHASE,
)
from orbit.runtime.kv_diag import model_call_context

from tests.test_analysis_rolling_kv import identity, strategy_client
from tests.test_analysis_rolling_kv_control_repair import _boundary_block_source

# A STEP prompt as tokens: the committed history, the transient guidance, the
# generation prompt. The next STEP repeats only the history -- grown by the
# action in between -- under a different guidance.
GEN = [900, 901]
HISTORY1 = [10, 11, 12, 13, 14]
GUIDE1 = [30, 31, 32]
STEP1 = HISTORY1 + GUIDE1 + GEN
HISTORY2 = HISTORY1 + [15, 16, 17]          # assistant call, tool result, analyst turn
GUIDE2 = [30, 31, 33]                       # "remaining" went down
STEP2 = HISTORY2 + GUIDE2 + GEN
GUIDE_Q2 = [40, 41, 42, 43]                 # a different question
STEP2_Q2 = HISTORY2 + GUIDE_Q2 + GEN
STEP_BYTES = b"step-checkpoint"
CONTROL_BYTES = b"control-checkpoint-other-bytes"


def step_identity(**overrides) -> RollingRouteIdentity:
    fields = {"tool_schema_hash": "step-tools", **overrides}
    return identity(ROLLING_STEP_STRATEGY_ID, **fields)


def control_identity(**overrides) -> RollingRouteIdentity:
    fields = {"tool_schema_hash": "finish-tools", **overrides}
    return identity(ROLLING_ANALYSIS_STRATEGY_ID, **fields)


def step_state(tokens=HISTORY1, ident=None, data=STEP_BYTES) -> RollingRouteAnchorState:
    return RollingRouteAnchorState(
        identity=ident or step_identity(), tokens=list(tokens),
        checkpoint_data=data, created_at_monotonic=1.0,
    )


def tokens_as_text(tokens) -> str:
    return " ".join(str(t) for t in tokens)


def int_tokenize(text: str) -> list[int]:
    return [int(w) for w in text.split()]


# --------------------------------------------------------------------------
class StepBoundaryHelperTest(unittest.TestCase):
    """`rolling_step_boundary`: the head is a boundary only when exact."""

    def test_the_boundary_is_the_head_when_it_opens_the_prompt(self) -> None:
        prompt, head = "sys hist1 hist2 guide gen", "sys hist1 hist2"
        tokens = [1, 2, 3, 4, 5]
        n = rolling_step_boundary(prompt, tokens, head=head + " ", tokenize=lambda t: [1, 2, 3][: len(t.split())])
        self.assertEqual(n, 3)

    def test_no_head_means_no_boundary(self) -> None:
        for head in (None, ""):
            self.assertIsNone(rolling_step_boundary("a b c", [1, 2, 3], head=head, tokenize=int_tokenize))

    def test_a_head_that_does_not_open_the_prompt_is_refused(self) -> None:
        """A renderer that treats its last turn specially would produce this."""
        self.assertIsNone(rolling_step_boundary("10 11 12 30 31", [10, 11, 12, 30, 31], head="10 11 99 ", tokenize=int_tokenize))

    def test_a_head_equal_to_the_prompt_leaves_nothing_to_extend(self) -> None:
        self.assertIsNone(rolling_step_boundary("10 11 12", [10, 11, 12], head="10 11 12", tokenize=int_tokenize))

    def test_a_head_longer_than_the_prompt_is_refused(self) -> None:
        self.assertIsNone(rolling_step_boundary("10 11", [10, 11], head="10 11 12", tokenize=int_tokenize))

    def test_a_boundary_inside_a_token_is_refused(self) -> None:
        """The head's tokens must be a strict prefix of the prompt's tokens."""
        prompt = "10 11 12 30 31"
        merged = [10, 11, 1230, 31]           # the renderer's tokenizer fused across the boundary

        def tokenizer(text):
            return merged if text == prompt else int_tokenize(text)

        self.assertIsNone(rolling_step_boundary(prompt, merged, head="10 11 12 ", tokenize=tokenizer))

    def test_a_head_that_differs_at_any_single_position_is_refused(self) -> None:
        prompt_tokens = [10, 11, 12, 13, 14, 30, 31]
        prompt = tokens_as_text(prompt_tokens)
        head_tokens = prompt_tokens[:5]
        head = tokens_as_text(head_tokens) + " "
        self.assertEqual(rolling_step_boundary(prompt, prompt_tokens, head=head, tokenize=int_tokenize), 5)
        for position in range(5):
            mutated = list(head_tokens)
            mutated[position] = 99

            def strict_tokenizer(text, expected=head):
                if text != expected:
                    raise AssertionError(f"unexpected tokenize input {text!r}")
                return mutated

            self.assertIsNone(
                rolling_step_boundary(prompt, prompt_tokens, head=head, tokenize=strict_tokenizer),
                f"a one-token difference at position {position} must refuse",
            )

    def test_a_realistic_length_boundary_is_exact_at_every_position(self) -> None:
        """The live shape: a 1339-token head inside a 1415-token prompt."""
        head_tokens = [1000 + (i * 7919) % 5000 for i in range(1339)]
        tail = [30 + i for i in range(76)]
        prompt_tokens = head_tokens + tail
        prompt = tokens_as_text(prompt_tokens)
        head = tokens_as_text(head_tokens) + " "
        self.assertEqual(rolling_step_boundary(prompt, prompt_tokens, head=head, tokenize=int_tokenize), 1339)
        for position in (0, 1, 63, 64, 65, 700, 1337, 1338):
            mutated = list(head_tokens)
            mutated[position] = 1

            def strict_tokenizer(text, expected=head):
                if text != expected:
                    raise AssertionError("unexpected tokenize input")
                return mutated

            self.assertIsNone(
                rolling_step_boundary(prompt, prompt_tokens, head=head, tokenize=strict_tokenizer),
                f"a one-token difference at position {position} must refuse",
            )

    def test_an_empty_tokenization_is_refused(self) -> None:
        self.assertIsNone(rolling_step_boundary("10 11 12", [10, 11, 12], head="10 ", tokenize=lambda t: []))


# --------------------------------------------------------------------------
class StepGateTest(unittest.TestCase):
    """The backend names the STEP turn; it is a refinement of the lineage gate."""

    def _requested(self, phase, *, native: bool = True, enabled: bool = True) -> tuple[bool, bool]:
        with mock.patch.object(llama_server, "prefix_anchor_enabled", lambda: enabled):
            if phase is None:
                return (
                    llama_server._analysis_rolling_anchor_requested(native_backend=native),
                    llama_server._analysis_step_anchor_requested(native_backend=native),
                )
            with model_call_context(phase=phase, tools_mode="on"):
                return (
                    llama_server._analysis_rolling_anchor_requested(native_backend=native),
                    llama_server._analysis_step_anchor_requested(native_backend=native),
                )

    def test_the_step_phase_is_the_step_turn(self) -> None:
        self.assertEqual(self._requested(ANALYSIS_STEP_PHASE), (True, True))
        self.assertEqual(self._requested(f"{ANALYSIS_STEP_PHASE}:Q1"), (True, True))

    def test_control_phases_stay_in_the_lineage_but_are_not_the_step(self) -> None:
        for phase in (ANALYSIS_PLAN_PHASE, ANALYSIS_FINISH_PHASE, f"{ANALYSIS_FINISH_PHASE}:Q2"):
            self.assertEqual(self._requested(phase), (True, False), phase)

    def test_phases_outside_the_lineage_are_never_the_step(self) -> None:
        for phase in (ANALYSIS_REPORT_PHASE, ANALYSIS_COVER_PHASE, "route", None, f"{ANALYSIS_STEP_PHASE}_extra"):
            self.assertEqual(self._requested(phase), (False, False), str(phase))

    def test_the_step_flag_never_widens_the_lineage(self) -> None:
        """Anchor disabled, or a non-native backend: both flags off together."""
        self.assertEqual(self._requested(ANALYSIS_STEP_PHASE, native=False), (False, False))
        self.assertEqual(self._requested(ANALYSIS_STEP_PHASE, enabled=False), (False, False))

    def test_the_backend_sends_both_flags_for_a_step(self) -> None:
        """Through `chat_stream`, with the payload builder observed."""
        from orbit.backend.payloads import ChatPayloadOptions

        seen: list[ChatPayloadOptions] = []

        def fake_build(options):
            seen.append(options)
            return {"messages": []}

        backend = llama_server.LlamaServerBackend.__new__(llama_server.LlamaServerBackend)
        backend.thinking = False
        backend._is_orbit_native_backend = lambda: True  # type: ignore[method-assign]
        backend.request_model_name = lambda: "m"  # type: ignore[method-assign]
        backend._serialize_for_profile = lambda messages: messages  # type: ignore[method-assign]
        backend._observe_call = lambda fn: fn()  # type: ignore[method-assign]
        backend._post_native_stream = lambda *a, **k: mock.Mock(tool_calls=[], content="")  # type: ignore[method-assign]
        with mock.patch.object(llama_server, "build_chat_payload", fake_build), \
                mock.patch.object(llama_server, "_attach_native_kv_diag_payload", lambda *a, **k: None), \
                mock.patch.object(llama_server, "prefix_anchor_enabled", lambda: True):
            with model_call_context(phase=ANALYSIS_STEP_PHASE, tools_mode="on"):
                backend.chat_stream([{"role": "user", "content": "x"}], temperature=0.0,
                                    max_tokens=1, tools=[{"a": 1}], on_delta=lambda t: None)
            with model_call_context(phase=f"{ANALYSIS_FINISH_PHASE}:Q1", tools_mode="on"):
                backend.chat_stream([{"role": "user", "content": "x"}], temperature=0.0,
                                    max_tokens=1, tools=[{"a": 1}], on_delta=lambda t: None)
        self.assertEqual([(o.analysis_rolling_anchor, o.analysis_step_anchor) for o in seen],
                         [(True, True), (True, False)])


class StepFlagPlumbingTest(unittest.TestCase):
    """The flag crosses the HTTP boundary exactly like its sibling."""

    def test_payload_carries_the_flag_only_when_asked(self) -> None:
        from orbit.backend.payloads import ChatPayloadOptions, build_chat_payload

        on = build_chat_payload(ChatPayloadOptions(
            model="m", messages=[], temperature=0.0, max_tokens=1,
            analysis_rolling_anchor=True, analysis_step_anchor=True))
        off = build_chat_payload(ChatPayloadOptions(
            model="m", messages=[], temperature=0.0, max_tokens=1, analysis_rolling_anchor=True))
        self.assertTrue(on["analysis_step_anchor"])
        self.assertNotIn("analysis_step_anchor", off)

    def test_request_parsing_round_trips_the_flag(self) -> None:
        from orbit.native_server.protocol import parse_chat_request

        messages = [{"role": "user", "content": "hi"}]
        self.assertTrue(parse_chat_request(
            {"model": "m", "messages": messages, "analysis_step_anchor": True}).analysis_step_anchor)
        self.assertFalse(parse_chat_request({"model": "m", "messages": messages}).analysis_step_anchor)
        self.assertFalse(parse_chat_request(
            {"model": "m", "messages": messages, "analysis_step_anchor": "yes"}).analysis_step_anchor,
            "only the literal True is a request")

    def test_the_server_hands_the_flag_to_the_client(self) -> None:
        from orbit.native_server import app as app_module

        source = inspect.getsource(app_module)
        self.assertIn("analysis_step_anchor=request.analysis_step_anchor", source)


# --------------------------------------------------------------------------
class _Session:
    def __init__(self) -> None:
        self.ctx_tgt = object()
        self.session_id = "default"
        self.cached_prompt_tokens: list[int] = []
        self.committed_sequence_tokens: list[int] = []
        self.mtp_enabled = False


GEN_TEXT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _render_messages(messages, *, tools=None, thinking=None) -> str:
    """A ChatML-shaped renderer: one block per turn, then the opener.

    Sensitive to `tools` and `thinking` the way a real template is (the tool
    schema lands in the leading system block, thinking changes the opener):
    a head rendered under different arguments than the prompt is NOT a
    textual prefix of it, which is exactly what production would then refuse.
    """
    parts = [f"<|tools:{json.dumps(tools, sort_keys=True)}|think:{bool(thinking)}|>"]
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n")
    return "".join(parts) + GEN_TEXT


class StepHeadTest(unittest.TestCase):
    """`_step_boundary_head`: the prompt up to the last user turn, or None."""

    MESSAGES = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "artifact"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "continue"},
        {"role": "user", "content": "Work on this question and nothing else"},
    ]

    def _client(self, *, bridge: bool = True, render=None, generation_prompt=GEN_TEXT):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client.model_profile = mock.Mock(uses_native_chat_bridge=bridge, profile_id=ORNITH15_PROFILE_ID)
        rendered: list[list] = []

        def default_render(messages, *, tools=None, thinking=None):
            rendered.append(list(messages))
            prompt = _render_messages(messages, tools=tools, thinking=thinking)
            client._active_profile_render = {"prompt": prompt, "generation_prompt": generation_prompt}
            return prompt

        client.apply_chat_template = render or default_render  # type: ignore[method-assign]
        return client, rendered

    def test_the_head_is_everything_before_the_last_user_turn(self) -> None:
        client, rendered = self._client()
        head = client._step_boundary_head(self.MESSAGES, tools=[{"a": 1}], thinking=False)
        self.assertEqual(rendered, [self.MESSAGES[:-1]], "rendered exactly the messages before the guidance")
        self.assertEqual(head, _render_messages(self.MESSAGES[:-1], tools=[{"a": 1}], thinking=False)[: -len(GEN_TEXT)])
        self.assertTrue(head.endswith("continue<|im_end|>\n"))
        self.assertNotIn("Work on this question", head, "the transient guidance is never in the head")
        full = _render_messages(self.MESSAGES, tools=[{"a": 1}], thinking=False)
        self.assertTrue(full.startswith(head), "the head opens the full prompt")
        for other in (
            _render_messages(self.MESSAGES, tools=None, thinking=False),
            _render_messages(self.MESSAGES, tools=[{"a": 1}], thinking=True),
        ):
            self.assertFalse(other.startswith(head),
                             "a head rendered under other tools/thinking would not open the prompt")

    def test_a_trailing_system_block_after_the_guidance_does_not_move_the_boundary(self) -> None:
        """`_admit` may append a rehydration system turn after the last user turn."""
        client, rendered = self._client()
        messages = [*self.MESSAGES, {"role": "system", "content": "evidence block"}]
        client._step_boundary_head(messages, tools=None, thinking=False)
        self.assertEqual(rendered, [self.MESSAGES[:-1]])

    def test_a_non_bridge_profile_reports_no_head(self) -> None:
        client, rendered = self._client(bridge=False)
        self.assertIsNone(client._step_boundary_head(self.MESSAGES, tools=None, thinking=False))
        self.assertEqual(rendered, [], "nothing is rendered when the renderer cannot report its opener")

    def test_no_user_turn_or_nothing_before_it_means_no_head(self) -> None:
        client, _ = self._client()
        self.assertIsNone(client._step_boundary_head([{"role": "system", "content": "S"}], tools=None, thinking=False))
        self.assertIsNone(client._step_boundary_head([{"role": "user", "content": "only"}], tools=None, thinking=False))
        self.assertIsNone(client._step_boundary_head([], tools=None, thinking=False))

    def test_a_renderer_without_a_generation_prompt_means_no_head(self) -> None:
        client, _ = self._client(generation_prompt=None)
        self.assertIsNone(client._step_boundary_head(self.MESSAGES, tools=None, thinking=False))

    def test_a_render_that_does_not_end_with_its_opener_means_no_head(self) -> None:
        client, _ = self._client(generation_prompt="<|different-opener|>")
        self.assertIsNone(client._step_boundary_head(self.MESSAGES, tools=None, thinking=False))

    def test_a_failing_render_means_no_head(self) -> None:
        def boom(messages, *, tools=None, thinking=None):
            raise RuntimeError("bridge unavailable")

        client, _ = self._client(render=boom)
        self.assertIsNone(client._step_boundary_head(self.MESSAGES, tools=None, thinking=False))


# --------------------------------------------------------------------------
class CompleteChatStepWiringTest(unittest.TestCase):
    """The STEP identity and its head travel from `complete_chat` to the prefill."""

    def _client(self, *, bridge: bool = True):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client.model_profile = mock.Mock(
            profile_id=ORNITH15_PROFILE_ID, verified=True,
            uses_native_chat_bridge=bridge, template_sha256="tpl",
        )
        client.config = mock.Mock(
            use_mtp_experimental=False, context_tokens=8192, thinking=False,
            progress_step=64, batch_size=256,
        )
        client.paths = mock.Mock(model="/models/ornith.gguf")
        client._model_metadata_identity = {}
        client._reset_generation = 0
        client._session = _Session()
        client._persistent_mtp_runtime = None
        client._media_marker = None
        client._final_prefix_store = lambda: mock.Mock()  # type: ignore[method-assign]
        client._thinking_enabled = lambda thinking: False  # type: ignore[method-assign]
        client._ensure_prompt_cache_mode = lambda mode: None  # type: ignore[method-assign]
        client._qwen_route_anchor_plan_for_prompt = lambda *a, **k: None  # type: ignore[method-assign]
        client._final_prefix_experiment_eligible = lambda flag: False  # type: ignore[method-assign]
        client._ornith_rolling_route_eligible = lambda **k: False  # type: ignore[method-assign]
        renders: list[list] = []

        def render(messages, *, tools=None, thinking=None):
            renders.append(list(messages))
            prompt = _render_messages(messages, tools=tools, thinking=thinking)
            client._active_profile_render = {"prompt": prompt, "generation_prompt": GEN_TEXT}
            return prompt

        client.apply_chat_template = render  # type: ignore[method-assign]
        seen: list[dict] = []

        def fake_complete_prompt(prompt, **kwargs):
            seen.append(dict(kwargs, prompt=prompt))
            return mock.Mock(prompt_tokens=1, output_tokens=1,
                             reused_prompt_tokens=0, evaluated_prompt_tokens=1)

        client.complete_prompt = fake_complete_prompt  # type: ignore[method-assign]
        return client, seen, renders

    MESSAGES = StepHeadTest.MESSAGES

    def _call(self, client, **flags):
        with mock.patch.object(client_module, "prepare_multimodal_messages", lambda *a, **k: None):
            client.complete_chat(self.MESSAGES, tools=[{"a": 1}], **flags)

    def test_a_step_turn_takes_the_step_strategy_and_hands_over_its_head(self) -> None:
        client, seen, renders = self._client()
        self._call(client, analysis_rolling_anchor=True, analysis_step_anchor=True)
        call = seen[0]
        self.assertTrue(call["rolling_route_eligible"])
        self.assertEqual(call["rolling_route_identity"].strategy_id, ROLLING_STEP_STRATEGY_ID)
        head = call["rolling_boundary_head"]
        self.assertEqual(head, _render_messages(self.MESSAGES[:-1], tools=[{"a": 1}], thinking=False)[: -len(GEN_TEXT)])
        self.assertTrue(call["prompt"].startswith(head),
                        "the head must be rendered under the prompt's own tools and thinking")
        self.assertNotIn("Work on this question", head)

    def test_the_production_render_is_the_last_render(self) -> None:
        """The bridge parser follows the most recent render; the head goes first."""
        client, seen, renders = self._client()
        self._call(client, analysis_rolling_anchor=True, analysis_step_anchor=True)
        self.assertEqual(renders, [self.MESSAGES[:-1], self.MESSAGES])
        self.assertEqual(client._active_profile_render["prompt"], seen[0]["prompt"])

    def test_a_control_turn_keeps_the_stage_a_strategy_and_no_head(self) -> None:
        client, seen, renders = self._client()
        self._call(client, analysis_rolling_anchor=True)
        call = seen[0]
        self.assertEqual(call["rolling_route_identity"].strategy_id, ROLLING_ANALYSIS_STRATEGY_ID)
        self.assertIsNone(call["rolling_boundary_head"])
        self.assertEqual(call["rolling_boundary_suffix"], GEN_TEXT, "Stage A's boundary still travels")
        self.assertEqual(renders, [self.MESSAGES], "no head render for a control turn")

    def test_the_step_flag_alone_requests_nothing(self) -> None:
        """It refines the lineage; it cannot open it."""
        client, seen, renders = self._client()
        self._call(client, analysis_step_anchor=True)
        self.assertFalse(seen[0]["rolling_route_eligible"])
        self.assertIsNone(seen[0]["rolling_route_identity"])
        self.assertIsNone(seen[0]["rolling_boundary_head"])
        self.assertEqual(renders, [self.MESSAGES])

    def test_a_non_bridge_profile_takes_the_step_strategy_with_no_head(self) -> None:
        """Fail closed: the slot is addressed, nothing is captured for it."""
        client, seen, _ = self._client(bridge=False)
        self._call(client, analysis_rolling_anchor=True, analysis_step_anchor=True)
        self.assertEqual(seen[0]["rolling_route_identity"].strategy_id, ROLLING_STEP_STRATEGY_ID)
        self.assertIsNone(seen[0]["rolling_boundary_head"])

    def test_the_step_identity_carries_every_safety_dimension(self) -> None:
        client, seen, _ = self._client()
        self._call(client, analysis_rolling_anchor=True, analysis_step_anchor=True)
        ident = seen[0]["rolling_route_identity"]
        self.assertEqual(ident.session_id, "default")
        self.assertEqual(ident.profile_id, ORNITH15_PROFILE_ID)
        self.assertEqual(ident.model_id, "/models/ornith.gguf")
        self.assertEqual(ident.template_id, "tpl")
        self.assertEqual(ident.reset_generation, 0)
        self.assertEqual(ident.tools_mode, "on")
        self.assertTrue(ident.tool_schema_hash and ident.runtime_policy_hash and ident.native_version)
        # and the control identity for the same tools differs in strategy only
        seen.clear()
        self._call(client, analysis_rolling_anchor=True)
        control = seen[0]["rolling_route_identity"]
        self.assertNotEqual(ident, control)
        self.assertEqual(ident.tool_schema_hash, control.tool_schema_hash)

    def test_the_step_identity_changes_with_schema_policy_and_reset(self) -> None:
        """M8 / M9 / M10: each dimension is derived, not a constant."""
        client, seen, _ = self._client()
        with mock.patch.object(client_module, "prepare_multimodal_messages", lambda *a, **k: None):
            client.complete_chat(self.MESSAGES, tools=[{"a": 1}],
                                 analysis_rolling_anchor=True, analysis_step_anchor=True)
            base = seen[-1]["rolling_route_identity"]
            client.complete_chat(self.MESSAGES, tools=[{"b": 2}],
                                 analysis_rolling_anchor=True, analysis_step_anchor=True)
            other_tools = seen[-1]["rolling_route_identity"]
            client.config.context_tokens = 4096
            client.complete_chat(self.MESSAGES, tools=[{"a": 1}],
                                 analysis_rolling_anchor=True, analysis_step_anchor=True)
            other_ctx = seen[-1]["rolling_route_identity"]
            client.config.context_tokens = 8192
            client._reset_generation += 1
            client.complete_chat(self.MESSAGES, tools=[{"a": 1}],
                                 analysis_rolling_anchor=True, analysis_step_anchor=True)
            after_reset = seen[-1]["rolling_route_identity"]
        self.assertNotEqual(base.tool_schema_hash, other_tools.tool_schema_hash,
                            "a different tool schema must change the identity")
        self.assertNotEqual(base.runtime_policy_hash, other_ctx.runtime_policy_hash,
                            "a different context budget must change the identity")
        self.assertNotEqual(base.reset_generation, after_reset.reset_generation,
                            "a reset must change the identity")
        stored = step_state(HISTORY1, ident=base)
        for stale in (other_tools, other_ctx, after_reset):
            self.assertIsNone(rolling_route_reuse_start(stored, STEP2, stale))


# --------------------------------------------------------------------------
class StepCaptureBlockTest(unittest.TestCase):
    """What the STEP lineage snapshots, where, and when it does not."""

    def _run(self, *, ident, head, prompt_tokens, processed, existing=None,
             cancelled=False, capture_result=None, eligible=True, suffix=None):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        store = client._rolling_anchor_store()
        if existing is not None:
            store.store(existing.identity, existing)
        client.cancel_event = threading.Event()
        if cancelled:
            client.cancel_event.set()
        client._session = _Session()
        prompt = tokens_as_text(prompt_tokens)
        client.tokenize = int_tokenize  # type: ignore[method-assign]
        decoded: list[tuple[int, int]] = []

        def fake_decode(token_array, *, processed, end, step, total, **_kw):
            decoded.append((processed, end))
            return end

        client._decode_prompt_range = fake_decode  # type: ignore[method-assign]
        captured: list[list[int]] = []

        def fake_capture(lib, ctx, *, prompt_tokens, identity):
            captured.append(list(prompt_tokens))
            return (capture_result or step_state(prompt_tokens, ident)), {}

        g = {
            "rolling_route_eligible": eligible,
            "rolling_route_identity": ident,
            "ROLLING_ANALYSIS_STRATEGY_ID": ROLLING_ANALYSIS_STRATEGY_ID,
            "ROLLING_STEP_STRATEGY_ID": ROLLING_STEP_STRATEGY_ID,
            "rolling_capture_boundary": rolling_capture_boundary,
            "rolling_step_boundary": rolling_step_boundary,
            "rolling_route_should_replace": client_module.rolling_route_should_replace,
            "capture_rolling_route_anchor": fake_capture,
            "prompt": prompt,
            "prompt_tokens": list(prompt_tokens),
            "rolling_boundary_suffix": suffix,
            "rolling_boundary_head": head,
            "processed": processed, "reused": processed, "token_array": object(), "step": 64,
            "n_prompt": len(prompt_tokens), "on_progress": None, "should_cancel": None,
            "pf_start": 0, "lib": object(), "self": client,
        }
        exec(_boundary_block_source(), g)
        return client, g, decoded, captured

    HEAD1 = tokens_as_text(HISTORY1) + " "
    HEAD2 = tokens_as_text(HISTORY2) + " "

    def test_a_step_is_snapshotted_before_its_guidance(self) -> None:
        """The M4 witness: the checkpoint is the history, never the guidance."""
        client, g, decoded, captured = self._run(
            ident=step_identity(), head=self.HEAD1, prompt_tokens=STEP1, processed=0)
        self.assertTrue(g["step_lineage"])
        self.assertEqual(g["capture_at"], len(HISTORY1))
        self.assertEqual(decoded, [(0, len(HISTORY1))], "prefill stops at the boundary first")
        self.assertEqual(captured, [HISTORY1])
        state = client._rolling_step_anchor_state
        self.assertEqual(state.tokens, HISTORY1)
        self.assertFalse(set(GUIDE1) & set(state.tokens), "no guidance token in the checkpoint")
        self.assertFalse(set(GEN) & set(state.tokens), "no generation-prompt token in the checkpoint")
        self.assertFalse(client._rolling_analysis_anchor_state.valid, "the control slot is untouched")

    def test_the_step_checkpoint_serves_the_next_step(self) -> None:
        client, *_ = self._run(ident=step_identity(), head=self.HEAD1, prompt_tokens=STEP1, processed=0)
        state = client._rolling_step_anchor_state
        self.assertEqual(rolling_route_reuse_start(state, STEP2, step_identity()), len(HISTORY1))
        self.assertEqual(rolling_route_reuse_start(state, STEP2_Q2, step_identity()), len(HISTORY1),
                         "a question transition changes only the guidance")
        self.assertIsNone(rolling_route_reuse_start(state, HISTORY1, step_identity()),
                          "the boundary itself leaves nothing to evaluate")
        self.assertIsNone(rolling_route_reuse_start(state, STEP2, control_identity()),
                          "a control identity can never claim the STEP checkpoint")

    def test_the_slot_advances_to_the_new_boundary_after_a_restore(self) -> None:
        """STEP 2 restored to |HISTORY1|; its own boundary is |HISTORY2|."""
        client, g, decoded, captured = self._run(
            ident=step_identity(), head=self.HEAD2, prompt_tokens=STEP2,
            processed=len(HISTORY1), existing=step_state(HISTORY1))
        self.assertEqual(g["capture_at"], len(HISTORY2))
        self.assertEqual(decoded, [(len(HISTORY1), len(HISTORY2))])
        self.assertEqual(captured, [HISTORY2])
        self.assertEqual(client._rolling_step_anchor_state.tokens, HISTORY2)

    def test_a_rewritten_history_replaces_the_stale_checkpoint(self) -> None:
        """Newest boundary wins for STEP: a non-extension means history was rewritten."""
        rewritten = [10, 11, 99, 13, 14, 15, 16, 17]
        client, g, _d, captured = self._run(
            ident=step_identity(), head=tokens_as_text(rewritten) + " ",
            prompt_tokens=rewritten + GUIDE2 + GEN, processed=0, existing=step_state(HISTORY1))
        self.assertEqual(captured, [rewritten])
        self.assertEqual(client._rolling_step_anchor_state.tokens, rewritten)

    def test_a_boundary_already_resident_is_not_recaptured(self) -> None:
        client, g, decoded, captured = self._run(
            ident=step_identity(), head=self.HEAD1, prompt_tokens=STEP1,
            processed=len(HISTORY1), existing=step_state(HISTORY1))
        self.assertIsNone(g["capture_at"])
        self.assertEqual(captured, [])
        self.assertEqual(client._rolling_step_anchor_state.tokens, HISTORY1, "the checkpoint stands")

    def test_no_head_means_no_step_capture_and_no_whole_prompt_capture(self) -> None:
        """Fail closed: nothing is captured, and the existing checkpoint stands."""
        client, g, decoded, captured = self._run(
            ident=step_identity(), head=None, prompt_tokens=STEP1, processed=0,
            existing=step_state(HISTORY1))
        self.assertIsNone(g["capture_at"])
        self.assertEqual(captured, [])
        self.assertEqual(client._rolling_step_anchor_state.tokens, HISTORY1)
        # the whole-prompt guard below the block is skipped for the STEP lineage
        source = inspect.getsource(NativeLlamaClient._complete_prompt_standard)
        start = source.index("        if (\n            capture_at is None")
        end = source.index("        self.last_committed_generated_tokens = []", start)
        guard = textwrap.dedent(source[start:end])
        whole: list[list[int]] = []

        def fake_capture(lib, ctx, *, prompt_tokens, identity):
            whole.append(list(prompt_tokens))
            return step_state(prompt_tokens), {}

        exec(guard, {
            "capture_at": None, "step_lineage": True, "rolling_route_eligible": True,
            "rolling_route_identity": step_identity(), "processed": len(STEP1),
            "n_prompt": len(STEP1), "prompt_tokens": STEP1, "self": client, "lib": object(),
            "rolling_route_should_replace": client_module.rolling_route_should_replace,
            "capture_rolling_route_anchor": fake_capture,
        })
        self.assertEqual(whole, [], "a whole STEP prompt is never a checkpoint: it ends in the guidance")
        self.assertEqual(client._rolling_step_anchor_state.tokens, HISTORY1)

    def test_the_whole_prompt_guard_stays_skipped_for_an_empty_step_slot(self) -> None:
        """No checkpoint yet is not a reason to store the whole STEP prompt."""
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client.cancel_event = threading.Event()
        client._session = _Session()
        self.assertFalse(client._rolling_step_anchor_state.valid)
        source = inspect.getsource(NativeLlamaClient._complete_prompt_standard)
        start = source.index("        if (\n            capture_at is None")
        end = source.index("        self.last_committed_generated_tokens = []", start)
        guard = textwrap.dedent(source[start:end])
        whole: list[list[int]] = []

        def fake_capture(lib, ctx, *, prompt_tokens, identity):
            whole.append(list(prompt_tokens))
            return step_state(prompt_tokens), {}

        exec(guard, {
            "capture_at": None, "step_lineage": True, "rolling_route_eligible": True,
            "rolling_route_identity": step_identity(), "processed": len(STEP1),
            "n_prompt": len(STEP1), "prompt_tokens": STEP1, "self": client, "lib": object(),
            "rolling_route_should_replace": client_module.rolling_route_should_replace,
            "capture_rolling_route_anchor": fake_capture,
        })
        self.assertEqual(whole, [])
        self.assertFalse(client._rolling_step_anchor_state.valid)

    def test_a_head_that_does_not_open_the_prompt_captures_nothing(self) -> None:
        client, g, _d, captured = self._run(
            ident=step_identity(), head="10 11 99 ", prompt_tokens=STEP1, processed=0)
        self.assertIsNone(g["capture_at"])
        self.assertEqual(captured, [])
        self.assertFalse(client._rolling_step_anchor_state.valid)

    def test_a_cancelled_prefill_never_captures(self) -> None:
        client, g, _d, captured = self._run(
            ident=step_identity(), head=self.HEAD1, prompt_tokens=STEP1, processed=0, cancelled=True)
        self.assertEqual(captured, [])
        self.assertFalse(client._rolling_step_anchor_state.valid)

    def test_a_cancellation_during_the_boundary_prefill_never_captures(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client.cancel_event = threading.Event()
        client._session = _Session()
        client.tokenize = int_tokenize  # type: ignore[method-assign]

        def cancelling_decode(token_array, *, processed, end, step, total, **_kw):
            client.cancel_event.set()
            return end

        client._decode_prompt_range = cancelling_decode  # type: ignore[method-assign]
        captured: list[list[int]] = []

        def fake_capture(lib, ctx, *, prompt_tokens, identity):
            captured.append(list(prompt_tokens))
            return step_state(prompt_tokens), {}

        g = {
            "rolling_route_eligible": True, "rolling_route_identity": step_identity(),
            "ROLLING_ANALYSIS_STRATEGY_ID": ROLLING_ANALYSIS_STRATEGY_ID,
            "ROLLING_STEP_STRATEGY_ID": ROLLING_STEP_STRATEGY_ID,
            "rolling_capture_boundary": rolling_capture_boundary,
            "rolling_step_boundary": rolling_step_boundary,
            "rolling_route_should_replace": client_module.rolling_route_should_replace,
            "capture_rolling_route_anchor": fake_capture,
            "prompt": tokens_as_text(STEP1), "prompt_tokens": list(STEP1),
            "rolling_boundary_suffix": None, "rolling_boundary_head": self.HEAD1,
            "processed": 0, "reused": 0, "token_array": object(), "step": 64,
            "n_prompt": len(STEP1), "on_progress": None, "should_cancel": None,
            "pf_start": 0, "lib": object(), "self": client,
        }
        exec(_boundary_block_source(), g)
        self.assertEqual(g["processed"], len(HISTORY1))
        self.assertEqual(captured, [])
        self.assertFalse(client._rolling_step_anchor_state.valid)

    def test_a_failed_capture_preserves_the_previous_checkpoint(self) -> None:
        failed = RollingRouteAnchorState(invalidation_reason="checkpoint_capture_failed")
        client, _g, _d, captured = self._run(
            ident=step_identity(), head=self.HEAD2, prompt_tokens=STEP2,
            processed=len(HISTORY1), existing=step_state(HISTORY1), capture_result=failed)
        self.assertEqual(captured, [HISTORY2], "capture was attempted")
        self.assertEqual(client._rolling_step_anchor_state.tokens, HISTORY1)

    def test_a_control_turn_ignores_the_head_and_keeps_its_stage_a_boundary(self) -> None:
        """Stage A unchanged: a control identity captures before the opener."""
        control_prompt = HISTORY1 + GUIDE1 + GEN
        client, g, decoded, captured = self._run(
            ident=control_identity(), head=self.HEAD1, prompt_tokens=control_prompt,
            processed=0, suffix=tokens_as_text(GEN))
        self.assertFalse(g["step_lineage"])
        self.assertEqual(g["capture_at"], len(HISTORY1 + GUIDE1))
        self.assertEqual(captured, [HISTORY1 + GUIDE1])
        self.assertEqual(client._rolling_analysis_anchor_state.tokens, HISTORY1 + GUIDE1)
        self.assertFalse(client._rolling_step_anchor_state.valid, "the STEP slot is untouched")

    def test_a_later_control_turn_replaces_a_stale_control_checkpoint(self) -> None:
        """The Stage A repair reuse depends on this once STEPs stop evicting.

        Live (KVB_20260907T204928Z): FINISH:Q1 -> repair -> STEP -> FINISH:Q1
        on a grown history. The second FINISH does not extend the first's
        repair checkpoint; under keep-older it was never captured and its own
        repair prefilled cold (3 of 4 repairs in that run). The control slot
        must advance to the newest boundary exactly as the STEP slot does.
        """
        stale = RollingRouteAnchorState(
            identity=control_identity(), tokens=HISTORY1 + GUIDE1 + [50, 51],   # FINISH-1 + its repair
            checkpoint_data=CONTROL_BYTES, created_at_monotonic=1.0)
        finish2 = HISTORY2 + [60, 61, 62] + GEN                                   # FINISH-2 on the grown history
        client, g, decoded, captured = self._run(
            ident=control_identity(), head=None, prompt_tokens=finish2, processed=0,
            existing=stale, suffix=tokens_as_text(GEN))
        self.assertTrue(g["control_lineage"])
        self.assertEqual(g["capture_at"], len(finish2) - len(GEN))
        self.assertEqual(captured, [finish2[: -len(GEN)]], "the newest control boundary is captured")
        state = client._rolling_analysis_anchor_state
        self.assertEqual(state.tokens, finish2[: -len(GEN)])
        repair2 = finish2[: -len(GEN)] + [70, 71] + GEN
        self.assertEqual(rolling_route_reuse_start(state, repair2, control_identity()), len(finish2) - len(GEN),
                         "and its repair restores it")
        self.assertFalse(client._rolling_step_anchor_state.valid, "the STEP slot is untouched")

    def test_the_route_lineage_keeps_keep_older(self) -> None:
        """The CHAT route's policy is not changed by either analysis lineage."""
        route_ident = identity(ROLLING_ROUTE_STRATEGY_ID)
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        older = RollingRouteAnchorState(identity=route_ident, tokens=[1, 2, 3], checkpoint_data=b"route")
        client._store_rolling_anchor_state(route_ident, older)
        self.assertFalse(client_module.rolling_route_should_replace(older, [9, 9, 9, 9], route_ident))
        self.assertIs(client._rolling_anchor_state_for(route_ident), older)

    def test_the_route_lineage_is_untouched(self) -> None:
        client, g, _d, captured = self._run(
            ident=identity(ROLLING_ROUTE_STRATEGY_ID), head=self.HEAD1, prompt_tokens=STEP1, processed=0)
        self.assertFalse(g["step_lineage"])
        self.assertIsNone(g["capture_at"])
        self.assertEqual(captured, [])


# --------------------------------------------------------------------------
class _RecordingLib:
    def __init__(self) -> None:
        self.restored: list[bytes] = []
        self.cleared = 0

    def llama_state_seq_set_data(self, ctx, buffer, size, seq_id):
        self.restored.append(bytes(buffer))
        return size

    def llama_get_memory(self, ctx):
        return object()

    def llama_memory_clear(self, mem, flag):
        self.cleared += 1


class StepRestoreTest(unittest.TestCase):
    """The restore, through the production strategy method, on the STEP slot."""

    def _client(self, state: RollingRouteAnchorState, ident: RollingRouteIdentity):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        lib = _RecordingLib()
        client.lib = mock.Mock(lib=lib)
        client._session = _Session()
        store = client._rolling_anchor_store()
        store.store(state.identity, state)
        # a control checkpoint with different bytes sits beside it
        store.store(control_identity(), RollingRouteAnchorState(
            identity=control_identity(), tokens=list(HISTORY1 + GUIDE1),
            checkpoint_data=CONTROL_BYTES, created_at_monotonic=1.0))
        client._rolling_route_identity_cache = ident
        prepared: list[list[int]] = []
        client._prepare_memory_for_prompt = lambda tokens: prepared.append(list(tokens)) or 0  # type: ignore[method-assign]
        client._invalidate_committed_sequence = lambda: client._session.committed_sequence_tokens.clear()  # type: ignore[method-assign]
        return client, lib, prepared

    def _restore(self, prompt, *, state=None, ident=None):
        client, lib, prepared = self._client(state or step_state(), ident or step_identity())
        client._prepare_memory_with_ornith_rolling_route_anchor(list(prompt))
        return client, lib, prepared

    def test_an_exact_extension_restores_the_step_bytes(self) -> None:
        client, lib, prepared = self._restore(STEP2)
        self.assertEqual(lib.restored, [STEP_BYTES], "exactly the STEP checkpoint's bytes")
        self.assertNotIn(CONTROL_BYTES, lib.restored)
        self.assertEqual(client._session.committed_sequence_tokens, HISTORY1)
        self.assertEqual(client._session.cached_prompt_tokens, HISTORY1)
        self.assertEqual(prepared, [STEP2], "strict append still judges the whole prompt")

    def test_only_the_guidance_changed(self) -> None:
        """§6.3: the stable prefix restores; the new guidance is evaluated."""
        for prompt in (STEP2, STEP2_Q2, HISTORY2 + [77] + GEN):
            client, lib, _ = self._restore(prompt)
            self.assertEqual(lib.restored, [STEP_BYTES], prompt)
            self.assertEqual(client._session.committed_sequence_tokens, HISTORY1)

    def test_a_one_token_mutation_before_the_boundary_is_refused(self) -> None:
        for position in (0, 2, len(HISTORY1) - 1):
            mutated = list(STEP2)
            mutated[position] = 99
            client, lib, prepared = self._restore(mutated)
            self.assertEqual(lib.restored, [], f"position {position}")
            self.assertEqual(client._session.committed_sequence_tokens, [])
            self.assertEqual(prepared, [mutated], "falls to strict append cold")

    def test_a_rewritten_history_before_the_boundary_is_refused(self) -> None:
        """§6.4 / §6.6 / §6.11: compaction, an incompatible FINISH, a rewrite."""
        rewritten = [10, 11, 12, 13] + [55, 56] + HISTORY2[5:] + GUIDE2 + GEN
        client, lib, _ = self._restore(rewritten)
        self.assertEqual(lib.restored, [])

    def test_different_tokens_and_no_extension_are_refused(self) -> None:
        for prompt in ([1, 2, 3, 4, 5, 6, 7, 8], HISTORY1, HISTORY1[:-1] + GUIDE1):
            client, lib, _ = self._restore(prompt)
            self.assertEqual(lib.restored, [], prompt)

    def test_a_stale_anchor_is_refused_on_every_identity_dimension(self) -> None:
        """§6.7 / §6.9 / §6.10: session, reset generation, schema, policy, model, template, native."""
        for field, value in (
            ("session_id", "other-session"), ("reset_generation", 1),
            ("tool_schema_hash", "other-tools"), ("runtime_policy_hash", "ctx-4096"),
            ("model_id", "/models/other.gguf"), ("template_id", "tpl-2"),
            ("native_version", "libllama-new.so"), ("profile_id", "other-profile"),
            ("capability_summary_hash", "other-caps"), ("tools_mode", "off"),
        ):
            client, lib, _ = self._restore(STEP2, ident=step_identity(**{field: value}))
            self.assertEqual(lib.restored, [], f"{field}={value!r} must refuse the stored checkpoint")
            self.assertEqual(client._session.committed_sequence_tokens, [])

    def test_a_control_identity_never_reads_the_step_checkpoint(self) -> None:
        """M2 / §6 phase isolation: the FINISH prompt extends the history too."""
        finish_prompt = HISTORY1 + [60, 61, 62] + GEN
        client, lib, _ = self._restore(finish_prompt, ident=control_identity())
        self.assertNotIn(STEP_BYTES, lib.restored, "the STEP bytes must never serve a control turn")

    def test_a_step_identity_never_reads_the_control_checkpoint(self) -> None:
        client, lib, _ = self._restore(HISTORY1 + GUIDE1 + [5] + GEN, state=RollingRouteAnchorState())
        self.assertEqual(lib.restored, [], "an empty STEP slot falls cold even though the control slot would match")


# --------------------------------------------------------------------------
class SlotIsolationTest(unittest.TestCase):
    """Three slots, addressed by strategy; a FINISH cannot evict a STEP."""

    def test_the_step_identity_addresses_its_own_slot(self) -> None:
        self.assertEqual(RollingAnchorStore.slot_for(step_identity()), "step")
        self.assertEqual(RollingAnchorStore.slot_for(control_identity()), "analysis")
        self.assertEqual(RollingAnchorStore.slot_for(identity(ROLLING_ROUTE_STRATEGY_ID)), "route")
        self.assertEqual(RollingAnchorStore.slot_for(None), "route")

    def test_a_finish_capture_leaves_a_valid_step_checkpoint_standing(self) -> None:
        """M5: the control slot is written; the STEP slot is not touched."""
        store = RollingAnchorStore()
        step = step_state(HISTORY1)
        store.store(step.identity, step)
        store.store(control_identity(), RollingRouteAnchorState(
            identity=control_identity(), tokens=HISTORY1 + GUIDE1, checkpoint_data=CONTROL_BYTES))
        store.store(control_identity(), RollingRouteAnchorState(
            identity=control_identity(), tokens=HISTORY1 + GUIDE1 + [1], checkpoint_data=CONTROL_BYTES))
        self.assertIs(store.step_state, step)
        self.assertEqual(rolling_route_reuse_start(store.state_for(step_identity()), STEP2, step_identity()),
                         len(HISTORY1))

    def test_a_step_capture_leaves_the_control_checkpoint_standing(self) -> None:
        store = RollingAnchorStore()
        control = RollingRouteAnchorState(identity=control_identity(), tokens=HISTORY1 + GUIDE1,
                                          checkpoint_data=CONTROL_BYTES)
        store.store(control.identity, control)
        store.store(step_identity(), step_state(HISTORY2))
        self.assertIs(store.analysis_state, control)

    def test_neither_analysis_slot_can_read_the_other(self) -> None:
        store = RollingAnchorStore()
        store.store(step_identity(), step_state(HISTORY1))
        self.assertFalse(store.state_for(control_identity()).valid)
        self.assertFalse(store.state_for(identity(ROLLING_ROUTE_STRATEGY_ID)).valid)
        self.assertIsNone(rolling_route_reuse_start(store.step_state, STEP2, control_identity()),
                          "identities compare whole: the strategy alone refuses")

    def test_invalidation_clears_all_three_and_keeps_the_first_reason(self) -> None:
        store = RollingAnchorStore()
        store.store(identity(ROLLING_ROUTE_STRATEGY_ID), RollingRouteAnchorState(
            identity=identity(ROLLING_ROUTE_STRATEGY_ID), tokens=[1], checkpoint_data=b"r"))
        store.store(control_identity(), RollingRouteAnchorState(
            identity=control_identity(), tokens=[2], checkpoint_data=b"c"))
        store.store(step_identity(), step_state([3]))
        store.invalidate("session_reset")
        store.invalidate("later")
        for ident in (identity(ROLLING_ROUTE_STRATEGY_ID), control_identity(), step_identity()):
            state = store.state_for(ident)
            self.assertFalse(state.valid)
            self.assertEqual(state.tokens, [])
            self.assertEqual(state.invalidation_reason, "session_reset")

    def test_an_untouched_step_slot_acquires_no_reason(self) -> None:
        store = RollingAnchorStore()
        store.store(control_identity(), RollingRouteAnchorState(
            identity=control_identity(), tokens=[2], checkpoint_data=b"c"))
        store.invalidate("control_only")
        self.assertIsNone(store.step_state.invalidation_reason)

    def test_the_client_property_reads_the_store(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        state = step_state(HISTORY1)
        client._store_rolling_anchor_state(state.identity, state)
        self.assertIs(client._rolling_step_anchor_state, state)
        self.assertIs(client._rolling_anchor_state_for(step_identity()), state)
        self.assertFalse(client._rolling_analysis_anchor_state.valid)

    def test_a_bare_client_reads_as_no_step_checkpoint(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        self.assertFalse(client._rolling_step_anchor_state.valid)


# --------------------------------------------------------------------------
class LifecycleTest(unittest.TestCase):
    """A STEP checkpoint cannot outlive the session, and does not accumulate."""

    def test_invalidation_clears_the_step_slot(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._store_rolling_anchor_state(step_identity(), step_state(HISTORY1))
        client._invalidate_rolling_route_anchor("session_close")
        state = client._rolling_step_anchor_state
        self.assertFalse(state.valid)
        self.assertEqual(state.invalidation_reason, "session_close")
        self.assertIsNone(rolling_route_reuse_start(state, STEP2, step_identity()))

    def test_reset_session_state_destroys_the_step_checkpoint(self) -> None:
        """§6.8 / §6.9 / M7: behavioural, through the real reset."""
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._session = _Session()
        client._session.cancel_requested = False
        client._session.prompt_cache_mode = "tools:thinking=off"
        client._session.continuation_ready = True
        client._session.last_metrics = object()
        client.cancel_event = threading.Event()
        client.lib = mock.Mock(lib=mock.Mock(llama_get_memory=lambda ctx: object(),
                                             llama_memory_clear=lambda mem, flag: None))
        client.config = mock.Mock(use_mtp_experimental=False)
        client._reset_generation = 0
        client._persistent_mtp_runtime = None
        client._store_rolling_anchor_state(step_identity(), step_state(HISTORY1))
        client._store_rolling_anchor_state(control_identity(), RollingRouteAnchorState(
            identity=control_identity(), tokens=HISTORY1 + GUIDE1, checkpoint_data=CONTROL_BYTES))
        client._invalidate_committed_sequence = lambda: None  # type: ignore[method-assign]
        client._invalidate_final_prefix = lambda reason: None  # type: ignore[method-assign]
        client._invalidate_qwen_route_prefix = lambda reason, profile_id=None: None  # type: ignore[method-assign]
        client._invalidate_qwen36_shell_tool_prefix = lambda reason: None  # type: ignore[method-assign]

        client.reset_session_state()

        state = client._rolling_step_anchor_state
        self.assertFalse(state.valid, "a reset must destroy the STEP checkpoint")
        self.assertEqual(state.invalidation_reason, "session_reset")
        self.assertFalse(client._rolling_analysis_anchor_state.valid)
        self.assertEqual(client._reset_generation, 1)
        self.assertIsNone(rolling_route_reuse_start(state, STEP2, step_identity(reset_generation=1)))

    def test_a_replaced_step_checkpoint_is_released(self) -> None:
        """Repeated STEP updates hold one checkpoint, never a growing set."""
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        first = step_state(HISTORY1, data=bytes(64))
        ref = weakref.ref(first)
        client._store_rolling_anchor_state(first.identity, first)
        del first
        second = step_state(HISTORY2, data=bytes(64))
        client._store_rolling_anchor_state(second.identity, second)
        gc.collect()
        self.assertIsNone(ref(), "the replaced checkpoint must not be retained anywhere")
        self.assertIs(client._rolling_step_anchor_state, second)

    def test_the_store_holds_exactly_three_slots(self) -> None:
        """No fourth slot without a measured need (mission §7)."""
        self.assertEqual(RollingAnchorStore.__slots__, ("_route", "_analysis", "_step"))


if __name__ == "__main__":
    unittest.main()
