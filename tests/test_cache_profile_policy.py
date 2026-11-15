from dataclasses import asdict
from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from orbit.native_llama.cache_profile_policy import (
    CacheProfilePolicy, cache_policy_for_profile,
    ROLLING_ROUTE_PROFILE_IDS, POST_FINAL_ROUTE_SHADOW_PROFILE_IDS,
)
from orbit.native_llama.client import NativeLlamaClient


CODER = "orbit-qwen3-coder-native-v1"
ORNITH = "orbit-ornith15-native-v1"
FLASH = "orbit-qwen38-flash-next-native-v1"
OTHER = ("orbit-gemma4-native-v1", "orbit-qwen36-native-v1", "orbit-qwen38-native-v1",
         "research-exact-model", "orbit-ornith15-native-v2", None)
PROFILES = (CODER, ORNITH, FLASH, *OTHER)


def client(profile_id, *, verified=True, thinking=False, mtp=False):
    value = NativeLlamaClient.__new__(NativeLlamaClient)
    value.model_profile = SimpleNamespace(profile_id=profile_id, verified=verified)
    value.config = SimpleNamespace(thinking=thinking, use_mtp_experimental=mtp)
    value._session = SimpleNamespace(mtp_enabled=False)
    return value


class CacheProfilePolicyTests(unittest.TestCase):
    def test_frozen_memberships_match_the_existing_qualification(self):
        expected = {
            CODER: {"jinja_route_prefill", "preserve_fixed_on_mode_switch",
                    "preserve_fixed_on_session_reset", "attempt_analysis_prewarm_after_route"},
            ORNITH: {"jinja_route_prefill", "rolling_route", "analysis_prefix",
                     "rolling_analysis", "attempt_analysis_prewarm_after_route"},
            FLASH: {"jinja_route_prefill", "rolling_route", "post_final_shadow"},
        }
        for profile_id in PROFILES:
            actual = {name for name, enabled in asdict(cache_policy_for_profile(profile_id)).items() if enabled}
            self.assertEqual(actual, expected.get(profile_id, set()), profile_id)
        self.assertEqual(ROLLING_ROUTE_PROFILE_IDS, {ORNITH, FLASH})
        self.assertEqual(POST_FINAL_ROUTE_SHADOW_PROFILE_IDS, {FLASH})

    def test_unknown_and_malformed_ids_never_inherit_a_family_policy(self):
        for profile_id in ([], {}, 1, "ornith", "qwen4exp", "orbit-qwen38-flash-next-native-v2"):
            self.assertFalse(any(asdict(cache_policy_for_profile(profile_id)).values()))

    def test_policies_are_immutable(self):
        with self.assertRaises(AttributeError):
            cache_policy_for_profile(FLASH).rolling_route = False

    def test_inconsistent_policies_are_rejected(self):
        for kwargs in ({"post_final_shadow": True}, {"preserve_fixed_on_mode_switch": True},
                       {"preserve_fixed_on_session_reset": True},
                       {"attempt_analysis_prewarm_after_route": True}, {"rolling_route": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CacheProfilePolicy(**kwargs)

    def test_real_client_gates_preserve_verified_thinking_and_mtp_guards(self):
        for profile_id, verified, thinking, mtp in product(PROFILES, (True, False), (True, False), (True, False)):
            value = client(profile_id, verified=verified, thinking=thinking, mtp=mtp)
            allowed = verified and not thinking and not mtp
            with self.subTest(profile_id=profile_id, verified=verified, thinking=thinking, mtp=mtp):
                self.assertEqual(value._ornith_rolling_route_eligible(tools=None, route_prefix_anchor=True, thinking=thinking),
                                 allowed and profile_id in {ORNITH, FLASH})
                self.assertEqual(value._ornith_rolling_analysis_eligible(analysis_rolling_anchor=True, thinking=thinking),
                                 allowed and profile_id == ORNITH)
                self.assertEqual(value.post_final_route_shadow_eligible(), allowed and profile_id == FLASH)
                self.assertFalse(value._ornith_rolling_route_eligible(tools=None, route_prefix_anchor=False, thinking=thinking))
                self.assertFalse(value._ornith_rolling_analysis_eligible(analysis_rolling_anchor=False, thinking=thinking))

    def test_active_mtp_session_still_refuses_reuse(self):
        for profile_id in (ORNITH, FLASH):
            value = client(profile_id)
            value._session.mtp_enabled = True
            self.assertFalse(value._ornith_rolling_route_eligible(tools=None, route_prefix_anchor=True, thinking=False))
            self.assertFalse(value._ornith_rolling_analysis_eligible(analysis_rolling_anchor=True, thinking=False))
            self.assertFalse(value.post_final_route_shadow_eligible())

    def test_mode_switch_preservation_matches_the_old_profile_matrix(self):
        for profile_id, verified, transition in product(PROFILES, (True, False), (
            ("chat:thinking=off", "tools:thinking=off"),
            ("tools:thinking=off", "chat:thinking=off"),
            ("chat:thinking=on", "tools:thinking=off"),
        )):
            value = client(profile_id, verified=verified)
            value._session.prompt_cache_mode = transition[0]
            value.reset_session_state = Mock()
            value._ensure_prompt_cache_mode(transition[1])
            qualified = verified and "thinking=on" not in transition[0]
            value.reset_session_state.assert_called_once_with(
                preserve_qwen3_coder_route_checkpoint=qualified and profile_id == CODER,
                preserve_ornith_rolling_route_checkpoint=qualified and profile_id in {ORNITH, FLASH},
            )
            self.assertEqual(value._session.prompt_cache_mode, transition[1])
