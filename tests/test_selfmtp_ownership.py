"""Ownership and ABI compatibility for the single-GGUF self-MTP session.

Two properties matter here and neither is observable from a passing generation:

  1. A self-MTP session BORROWS the model and the target context. It owns only
     the MTP context and its speculative state. If destroy ever freed a
     borrowed handle, the client's own teardown would double-free moments
     later -- and the crash would land far from the cause.

  2. The self-MTP symbols are required ONLY when a self-MTP session is about to
     be built. A shim predating them is still a valid shim for normal decoding
     and for the external-draft path, and must not be rejected or rebuilt.

Both are proved against the real compiled shim where one is available, and
against the module's own contracts otherwise. No 21 GiB model is loaded.
"""

from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_llama import persistent_mtp as pm  # noqa: E402

BUILT_SHIM = Path("/tmp/selfmtp-build/liborbit-persistent-mtp.so")


class SymbolRequirementTests(unittest.TestCase):
    """Self-MTP symbols must not become mandatory for every shim user."""

    def test_base_requirements_do_not_include_self_mtp(self) -> None:
        for symbol in pm._SELF_MTP_REQUIRED_SHIM_SYMBOLS:
            self.assertNotIn(
                symbol,
                pm._REQUIRED_SHIM_SYMBOLS,
                "self-MTP symbol leaked into the base ABI requirement",
            )

    def test_base_requirements_are_exactly_the_pre_mission_set(self) -> None:
        """Pinned literally: widening this silently breaks older shims."""
        self.assertEqual(
            pm._REQUIRED_SHIM_SYMBOLS,
            (
                "orbit_mtp_session_complete",
                "orbit_mtp_session_set_followup_suffix_tokens",
                "orbit_mtp_session_last_first_sample_trace_json",
                "orbit_mtp_session_request_boundary_refill_marker",
            ),
        )

    def test_self_mtp_requirements_cover_create_and_destroy(self) -> None:
        self.assertIn("orbit_selfmtp_session_create", pm._SELF_MTP_REQUIRED_SHIM_SYMBOLS)
        self.assertIn("orbit_selfmtp_session_destroy", pm._SELF_MTP_REQUIRED_SHIM_SYMBOLS)


class LegacyShimAcceptanceTests(unittest.TestCase):
    """A shim without self-MTP symbols stays usable for everything else."""

    class _LegacyLib:
        """Exports the base symbols only -- a pre-self-MTP shim."""

        def __init__(self) -> None:
            for name in pm._REQUIRED_SHIM_SYMBOLS:
                setattr(self, name, object())

    def _check(self, lib, *, require_self_mtp: bool) -> bool:
        symbols = pm._REQUIRED_SHIM_SYMBOLS
        if require_self_mtp:
            symbols = symbols + pm._SELF_MTP_REQUIRED_SHIM_SYMBOLS
        return all(hasattr(lib, s) for s in symbols)

    def test_legacy_shim_satisfies_the_base_abi(self) -> None:
        self.assertTrue(self._check(self._LegacyLib(), require_self_mtp=False))

    def test_legacy_shim_is_rejected_only_for_self_mtp(self) -> None:
        self.assertFalse(self._check(self._LegacyLib(), require_self_mtp=True))

    def test_packaged_shim_is_reused_when_self_mtp_is_not_requested(self) -> None:
        """The rebuild must not fire for normal or external-draft use."""
        packaged = Path("/packaged/liborbit-persistent-mtp.so")
        with mock.patch.object(pm, "packaged_shim_path", return_value=packaged), \
             mock.patch.object(pm, "_shim_exports_required_symbols", return_value=True) as check, \
             mock.patch.object(pm, "compile_cpp_helper") as compiled:
            result = pm.build_persistent_mtp_shim(llama_root=None, build_bin=None)
        self.assertEqual(result, packaged)
        compiled.assert_not_called()
        self.assertEqual(check.call_args.args[2], pm._REQUIRED_SHIM_SYMBOLS)

    def test_self_mtp_request_asks_for_the_wider_symbol_set(self) -> None:
        packaged = Path("/packaged/liborbit-persistent-mtp.so")
        with mock.patch.object(pm, "packaged_shim_path", return_value=packaged), \
             mock.patch.object(pm, "_shim_exports_required_symbols", return_value=True) as check, \
             mock.patch.object(pm, "compile_cpp_helper"):
            pm.build_persistent_mtp_shim(
                llama_root=None, build_bin=None, require_self_mtp=True
            )
        requested = check.call_args.args[2]
        for symbol in pm._SELF_MTP_REQUIRED_SHIM_SYMBOLS:
            self.assertIn(symbol, requested)
        for symbol in pm._REQUIRED_SHIM_SYMBOLS:
            self.assertIn(symbol, requested)

    def test_a_shim_lacking_self_mtp_triggers_a_rebuild_only_then(self) -> None:
        packaged = Path("/packaged/liborbit-persistent-mtp.so")
        built = Path("/built/liborbit-persistent-mtp.so")
        with mock.patch.object(pm, "packaged_shim_path", return_value=packaged), \
             mock.patch.object(pm, "_shim_exports_required_symbols", return_value=False), \
             mock.patch.object(pm, "require_legacy_llama_root", return_value=Path("/llama")), \
             mock.patch.object(pm, "compile_cpp_helper", return_value=built) as compiled:
            result = pm.build_persistent_mtp_shim(
                llama_root=Path("/llama"), build_bin=None, require_self_mtp=True
            )
        self.assertEqual(result, built)
        compiled.assert_called_once()


@unittest.skipUnless(BUILT_SHIM.is_file(), "compiled self-MTP shim not present")
class CompiledShimTests(unittest.TestCase):
    """Against the real .so: symbols exist and ownership is declared."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = ctypes.CDLL(str(BUILT_SHIM), mode=ctypes.RTLD_GLOBAL)

    def test_self_mtp_entry_points_are_exported(self) -> None:
        for symbol in pm._SELF_MTP_REQUIRED_SHIM_SYMBOLS:
            with self.subTest(symbol):
                self.assertTrue(hasattr(self.lib, symbol))

    def test_external_draft_entry_point_is_still_exported(self) -> None:
        """The two-GGUF path must remain available and untouched."""
        self.assertTrue(hasattr(self.lib, "orbit_mtp_session_create"))
        self.assertTrue(hasattr(self.lib, "orbit_mtp_session_free"))

    def test_create_rejects_a_null_model(self) -> None:
        self.lib.orbit_selfmtp_session_create.restype = ctypes.c_void_p
        self.lib.orbit_selfmtp_session_create.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32,
        ]
        handle = self.lib.orbit_selfmtp_session_create(
            None, ctypes.c_void_p(1), 8192, 256, 128, 6, 6
        )
        self.assertFalse(handle)

    def test_create_rejects_a_null_target_context(self) -> None:
        self.lib.orbit_selfmtp_session_create.restype = ctypes.c_void_p
        handle = self.lib.orbit_selfmtp_session_create(
            ctypes.c_void_p(1), None, 8192, 256, 128, 6, 6
        )
        self.assertFalse(handle)

    def test_destroy_of_null_is_safe(self) -> None:
        """Idempotence starts here: destroying nothing must not fault."""
        self.lib.orbit_selfmtp_session_destroy.argtypes = [ctypes.c_void_p]
        self.lib.orbit_selfmtp_session_destroy(None)

    def test_accessors_of_null_return_null(self) -> None:
        for name in (
            "orbit_mtp_session_borrowed_model",
            "orbit_mtp_session_borrowed_ctx_tgt",
            "orbit_mtp_session_ctx_dft",
        ):
            fn = getattr(self.lib, name)
            fn.restype = ctypes.c_void_p
            fn.argtypes = [ctypes.c_void_p]
            with self.subTest(name):
                self.assertFalse(fn(None))

    def test_owns_model_of_null_is_false(self) -> None:
        self.lib.orbit_mtp_session_owns_model.restype = ctypes.c_bool
        self.lib.orbit_mtp_session_owns_model.argtypes = [ctypes.c_void_p]
        self.assertFalse(self.lib.orbit_mtp_session_owns_model(None))


class SourceOwnershipContractTests(unittest.TestCase):
    """The teardown contract, asserted against the shim source.

    Behavioural proof needs a loaded model; these pin the invariants that make
    that safe to attempt, so a regression is caught before the 21 GiB run.
    """

    SOURCE = ROOT / "src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp"

    def setUp(self) -> None:
        self.text = self.SOURCE.read_text()
        start = self.text.index("orbit_selfmtp_session_destroy")
        self.destroy = self.text[start : self.text.index("\n}", start)]
        start_c = self.text.index("orbit_selfmtp_session_create")
        self.create = self.text[start_c : self.text.index("\n}", start_c)]

    def test_destroy_never_frees_a_model(self) -> None:
        self.assertNotIn("llama_model_free", self.destroy)

    def test_destroy_frees_the_mtp_context_it_created(self) -> None:
        """Guarded on the pointer, not on a constant.

        Asserting only that the free CALL appears is too weak: wrapping it in
        `if (false)` leaves the text intact and leaks the context on every
        teardown. The guard has to be the handle itself.
        """
        self.assertIn("llama_free(session->ctx_dft)", self.destroy)
        free_at = self.destroy.index("llama_free(session->ctx_dft)")
        guard = self.destroy[:free_at]
        self.assertIn("if (session->ctx_dft)", guard)
        self.assertNotIn("if (false)", self.destroy)
        self.assertNotIn("if (0)", self.destroy)

    def test_destroy_frees_the_speculative_state_it_created(self) -> None:
        self.assertIn("common_speculative_free(session->spec)", self.destroy)
        free_at = self.destroy.index("common_speculative_free(session->spec)")
        self.assertIn("if (session->spec)", self.destroy[:free_at])

    def test_destroy_never_frees_the_target_context(self) -> None:
        self.assertNotIn("llama_free(session->ctx_tgt_borrowed)", self.destroy)
        self.assertNotIn("llama_free(ctx_tgt", self.destroy)

    def test_destroy_nulls_every_handle_it_releases(self) -> None:
        """What makes a repeated destroy a no-op rather than a double free."""
        for cleared in (
            "session->spec = nullptr",
            "session->ctx_dft = nullptr",
            "session->model_dft = nullptr",
            "session->ctx_tgt_borrowed = nullptr",
        ):
            self.assertIn(cleared, self.destroy)

    def test_self_mtp_create_never_loads_a_model(self) -> None:
        """One model load total: the caller's. Never a second mmap."""
        self.assertNotIn("llama_model_load_from_file", self.create)

    def test_self_mtp_create_marks_the_model_as_borrowed(self) -> None:
        self.assertIn("session->owns_model_dft = false", self.create)

    def test_self_mtp_create_builds_exactly_one_context(self) -> None:
        self.assertEqual(self.create.count("llama_init_from_model"), 1)

    def test_self_mtp_create_uses_the_mtp_context_type(self) -> None:
        self.assertIn("LLAMA_CONTEXT_TYPE_MTP", self.create)
        self.assertIn("ctx_params.ctx_other", self.create)

    def test_partial_init_failure_cleans_up(self) -> None:
        """Both failure exits must run cleanup, not leak the MTP context."""
        self.assertEqual(self.create.count("cleanup_session(session.get())"), 2)

    def test_cleanup_respects_borrowed_ownership(self) -> None:
        start = self.text.index("static void cleanup_session")
        cleanup = self.text[start : self.text.index("\n}", start)]
        self.assertIn("if (session->owns_model_dft)", cleanup)

    def test_external_draft_create_still_owns_its_model(self) -> None:
        start = self.text.index("void * orbit_mtp_session_create")
        create = self.text[start : self.text.index("\n}", start)]
        self.assertIn("session->owns_model_dft = true", create)
        self.assertIn("llama_model_load_from_file", create)


if __name__ == "__main__":
    unittest.main()
