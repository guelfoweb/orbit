"""Who owns chat templating on the MTP path.

`complete_prompt` receives a prompt that `apply_chat_template` has ALREADY
rendered using the profile's own renderer. For a profile whose template lives in
the GGUF (renderer "llama.cpp-jinja"), that is the model's official template.

The defect these cover: `_prepare_mtp_prompt` decided which family a prompt
belonged to by sniffing for Gemma4 markers in the prompt TEXT. An Ornith prompt
contains none of them, so it fell through to `render_gemma4_chat`, which wrapped
the already-rendered prompt as the *content of a new user message*. The model was
handed a Gemma4 envelope containing Ornith markup as literal text, and generated
scaffolding instead of an answer.

Prompt text is data. It is not evidence of which model produced it, and a user
can type any of those markers. Family dispatch must come from the verified
profile, so these tests pin behaviour by profile rather than by prompt content.
"""

from __future__ import annotations

import unittest

from orbit.native_llama.chat_template import render_gemma4_chat
from orbit.native_llama.client import _prepare_mtp_prompt, _strip_thinking_prompt

# An officially rendered Ornith prompt: qwen-style markers from the GGUF template.
ORNITH_RENDERED = "<|im_start|>user\nReply exactly: ORBIT_D1_OK<|im_end|>\n<|im_start|>assistant\n"

FOREIGN_MARKERS = ("<bos>", "<|turn>", "<|channel>", "<turn|>", "<channel|>")


class _Profile:
    def __init__(self, renderer: str) -> None:
        self.renderer = renderer
        self.verified = True
        self.mtp_supported = False

    @property
    def uses_native_chat_bridge(self) -> bool:
        return self.renderer == "llama.cpp-jinja"


GEMMA4 = _Profile("orbit-gemma4")
JINJA = _Profile("llama.cpp-jinja")


class HistoricalDefectTests(unittest.TestCase):
    """Reproduces the exact Ornith corruption. No model required."""

    def test_ornith_rendered_prompt_is_not_re_rendered(self) -> None:
        out = _prepare_mtp_prompt(ORNITH_RENDERED, thinking=False, profile=JINJA)
        self.assertEqual(
            out,
            ORNITH_RENDERED,
            "an already-rendered profile prompt must reach MTP unchanged",
        )

    def test_no_foreign_template_markers_are_introduced(self) -> None:
        out = _prepare_mtp_prompt(ORNITH_RENDERED, thinking=False, profile=JINJA)
        for marker in FOREIGN_MARKERS:
            self.assertNotIn(
                marker, out, f"Orbit introduced a foreign template marker: {marker}"
            )

    def test_ornith_prompt_is_not_wrapped_as_user_content(self) -> None:
        """The failure mode was nesting the whole prompt inside a new user turn."""
        out = _prepare_mtp_prompt(ORNITH_RENDERED, thinking=False, profile=JINJA)
        self.assertFalse(out.startswith("<bos>"))
        self.assertEqual(out.count("<|im_start|>user"), 1)


class Gemma4ByteEquivalenceTests(unittest.TestCase):
    """Captured from the pre-fix implementation; must not drift."""

    def test_thinking_off_strips_the_thought_suffix(self) -> None:
        rendered = render_gemma4_chat([{"role": "user", "content": "HELLO"}], thinking=False)
        self.assertEqual(
            _prepare_mtp_prompt(rendered, thinking=False, profile=GEMMA4),
            "<bos><|turn>user\nHELLO<turn|>\n",
        )

    def test_thinking_on_passes_through(self) -> None:
        rendered = render_gemma4_chat([{"role": "user", "content": "HELLO"}], thinking=True)
        self.assertEqual(
            _prepare_mtp_prompt(rendered, thinking=True, profile=GEMMA4), rendered
        )

    def test_multiturn_matches_prefix_characterization(self) -> None:
        rendered = render_gemma4_chat(
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
            ],
            thinking=False,
        )
        self.assertEqual(
            _prepare_mtp_prompt(rendered, thinking=False, profile=GEMMA4),
            "<bos><|turn>user\nA<turn|>\n<|turn>model\nB<turn|>\n<|turn>user\nC<turn|>\n",
        )


class MarkerSpoofingTests(unittest.TestCase):
    """Prompt text must never decide which family's contract applies."""

    def test_gemma_markers_inside_a_jinja_prompt_do_not_switch_family(self) -> None:
        spoofed = (
            "<|im_start|>user\nExplain what <bos><|turn>model means"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        self.assertEqual(
            _prepare_mtp_prompt(spoofed, thinking=False, profile=JINJA),
            spoofed,
            "a user quoting Gemma4 markers must not be re-rendered as Gemma4",
        )

    def test_gemma_thought_suffix_quoted_by_a_user_is_not_stripped(self) -> None:
        suffix = "<|turn>model\n<|channel>thought\n<channel|>"
        spoofed = f"<|im_start|>user\nliteral: {suffix}<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(
            _prepare_mtp_prompt(spoofed, thinking=False, profile=JINJA), spoofed
        )

    def test_absent_profile_still_does_not_re_render_a_rendered_prompt(self) -> None:
        """An unloaded client must not wrap an already-rendered prompt.

        `model_profile` is None only before `load()` resolves it; in production
        it is always set before the MTP session exists. Even so, a prompt that
        already carries an envelope must not gain a second one.
        """
        gemma_rendered = render_gemma4_chat(
            [{"role": "user", "content": "HELLO"}], thinking=True
        )
        self.assertEqual(
            _prepare_mtp_prompt(gemma_rendered, thinking=True, profile=None),
            gemma_rendered,
        )


class BridgeProfileNeverStripsTests(unittest.TestCase):
    """A bridge-rendered prompt must survive even if it *ends* like Gemma4.

    Without this, applying `_strip_thinking_prompt` to bridge profiles is
    undetectable: ordinary Ornith prompts do not end with the Gemma4 thought
    suffix, so the strip is a silent no-op on them.
    """

    GEMMA_SUFFIX = "<|turn>model\n<|channel>thought\n<channel|>"

    def test_bridge_prompt_ending_in_gemma_suffix_is_untouched(self) -> None:
        prompt = f"<|im_start|>user\nquote: {self.GEMMA_SUFFIX}"
        self.assertTrue(prompt.endswith(self.GEMMA_SUFFIX))
        self.assertEqual(
            _prepare_mtp_prompt(prompt, thinking=False, profile=JINJA),
            prompt,
            "a bridge-owned prompt must never have the Gemma4 suffix stripped",
        )

    def test_bridge_prompt_ending_in_gemma_suffix_untouched_with_thinking(self) -> None:
        prompt = f"<|im_start|>assistant\n{self.GEMMA_SUFFIX}"
        self.assertEqual(
            _prepare_mtp_prompt(prompt, thinking=True, profile=JINJA), prompt
        )


class CallSitePassesProfileTests(unittest.TestCase):
    """The real call site must hand the resolved profile to the helper.

    Without this, dropping `profile=` at client.py's call site is invisible:
    every helper-level test supplies the profile explicitly.
    """

    def test_try_complete_passes_the_resolved_profile(self) -> None:
        from ctypes import c_void_p
        from unittest.mock import patch
        from orbit.native_llama import client as client_mod
        from orbit.native_llama.client import NativeLlamaClient
        from orbit.native_llama.session_state import NativeSessionState

        seen: dict[str, object] = {}
        real = client_mod._prepare_mtp_prompt

        def spy(prompt, *, thinking=False, profile=None):
            seen["profile"] = profile
            seen["out"] = real(prompt, thinking=thinking, profile=profile)
            raise _Stop()

        c = NativeLlamaClient.__new__(NativeLlamaClient)
        c.model_profile = JINJA
        c._session = NativeSessionState(session_id="cs")
        c._session.ctx_tgt = c_void_p(0x1)
        c._session.mtp_enabled = True
        # A self-MTP runtime, as Ornith has: the completion gate's metadata
        # check is bypassed for it, so execution reaches the prompt helper.
        c._persistent_mtp_runtime = type("R", (), {"self_mtp": True})()
        c.config = type("C", (), {"use_mtp_experimental": True})()
        c.paths = type("P", (), {"mtp_available": False, "fallback_reason": None})()
        c.cancel_event = type("E", (), {"is_set": lambda self: False})()
        c.mtp_fallback_reason = None
        c.last_mtp_completion = None

        with patch.object(client_mod, "_prepare_mtp_prompt", spy), \
             patch.object(NativeLlamaClient, "_thinking_enabled", lambda self, v: False), \
             patch.object(NativeLlamaClient, "_invalidate_committed_sequence", lambda self: None):
            try:
                c._try_complete_with_mtp_experimental(ORNITH_RENDERED, max_tokens=8)
            except _Stop:
                pass

        self.assertIs(
            seen.get("profile"), JINJA,
            "the call site must pass the resolved profile, not None",
        )
        self.assertEqual(seen.get("out"), ORNITH_RENDERED)


class _Stop(Exception):
    pass


class StripThinkingPromptTests(unittest.TestCase):
    """Direct coverage for the previously untested helper."""

    def test_strips_only_the_exact_gemma4_suffix(self) -> None:
        rendered = render_gemma4_chat([{"role": "user", "content": "X"}], thinking=False)
        self.assertTrue(rendered.endswith("<|turn>model\n<|channel>thought\n<channel|>"))
        self.assertEqual(
            _strip_thinking_prompt(rendered),
            rendered[: -len("<|turn>model\n<|channel>thought\n<channel|>")],
        )

    def test_leaves_a_jinja_prompt_untouched(self) -> None:
        self.assertEqual(_strip_thinking_prompt(ORNITH_RENDERED), ORNITH_RENDERED)

    def test_leaves_unrelated_text_untouched(self) -> None:
        self.assertEqual(_strip_thinking_prompt("no markers here"), "no markers here")


class RenderOwnershipTests(unittest.TestCase):
    def test_prepare_is_idempotent_for_jinja_profiles(self) -> None:
        once = _prepare_mtp_prompt(ORNITH_RENDERED, thinking=False, profile=JINJA)
        twice = _prepare_mtp_prompt(once, thinking=False, profile=JINJA)
        self.assertEqual(once, twice)

    def test_prepare_is_idempotent_for_gemma4(self) -> None:
        rendered = render_gemma4_chat([{"role": "user", "content": "HELLO"}], thinking=False)
        once = _prepare_mtp_prompt(rendered, thinking=False, profile=GEMMA4)
        twice = _prepare_mtp_prompt(once, thinking=False, profile=GEMMA4)
        self.assertEqual(once, twice)

    def test_jinja_prompt_never_grows(self) -> None:
        """Re-rendering can only add envelope; passthrough cannot."""
        out = _prepare_mtp_prompt(ORNITH_RENDERED, thinking=False, profile=JINJA)
        self.assertLessEqual(len(out), len(ORNITH_RENDERED))


if __name__ == "__main__":
    unittest.main()
