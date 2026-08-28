"""Self-MTP wiring, driven through the real client method.

These call `NativeLlamaClient._initialize_persistent_mtp_session` -- the
production branch -- rather than the helpers underneath it. A helper can be
perfectly correct while the client never reaches it, and that failure mode is
invisible to helper-only tests.

No model is loaded: the client is built with `__new__` and given exactly the
attributes the method reads, so the branching is exercised without 21 GiB.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_llama import client as client_mod  # noqa: E402
from orbit.native_llama.client import NativeLlamaClient  # noqa: E402
from orbit.native_llama.model_profiles import (  # noqa: E402
    ORNITH15_PROFILE_ID,
    QWEN3_CODER_PROFILE_ID,
    SELF_MTP_CAPABILITY,
)

CURRENT = Path("/models/current/Ornith-1.5-35B-Q4_K_M.gguf")
LEGACY = Path("/models/legacy/Ornith-1.5-35B-Q4_K_M.gguf")


def _client(
    *,
    mtp_requested: bool,
    profile_id: str = ORNITH15_PROFILE_ID,
    verified: bool = True,
    mtp_supported: bool = False,
    model_path: Path = CURRENT,
    draft_mtp_model: Path | None = None,
    ctx_tgt: object = "ctx-tgt-handle",
) -> NativeLlamaClient:
    c = NativeLlamaClient.__new__(NativeLlamaClient)
    c.config = SimpleNamespace(
        use_mtp_experimental=mtp_requested,
        context_tokens=8192, batch_size=256, ubatch_size=128,
        threads=6, threads_batch=6,
    )
    c.paths = SimpleNamespace(
        model=model_path, llama_root=Path("/llama"),
        mtp_available=draft_mtp_model is not None,
        draft_mtp_model=draft_mtp_model,
        fallback_reason=None if draft_mtp_model else "mtp-not-declared",
    )
    c.model_profile = SimpleNamespace(
        profile_id=profile_id, verified=verified, mtp_supported=mtp_supported
    )
    c._model = "model-handle"
    c._persistent_mtp_runtime = None
    c._session = SimpleNamespace(
        ctx_tgt=ctx_tgt, ctx_dft=None, spec=None,
        mtp_enabled=False, mtp_failed=False, mtp_failure_reason=None,
    )
    return c


class NormalStartupTests(unittest.TestCase):
    """No MTP request means no digest and no self-MTP logic at all."""

    def test_normal_startup_never_hashes_the_artifact(self) -> None:
        c = _client(mtp_requested=False)
        with mock.patch.object(client_mod, "verified_artifact_supports") as supports, \
             mock.patch("orbit.native_llama.artifact_capabilities.artifact_sha256") as hashed:
            c._initialize_persistent_mtp_session()
        supports.assert_not_called()
        hashed.assert_not_called()

    def test_normal_startup_builds_no_session(self) -> None:
        c = _client(mtp_requested=False)
        with mock.patch.object(client_mod, "create_self_mtp_session") as self_mtp, \
             mock.patch.object(client_mod, "create_persistent_mtp_session") as external:
            c._initialize_persistent_mtp_session()
        self_mtp.assert_not_called()
        external.assert_not_called()
        self.assertFalse(c._session.mtp_enabled)


class SelfMtpSelectionTests(unittest.TestCase):
    """The qualified artifact reaches the self-MTP constructor, and only it."""

    def test_current_artifact_selects_the_self_mtp_constructor(self) -> None:
        c = _client(mtp_requested=True)
        runtime = SimpleNamespace(ctx_dft="ctx-dft", spec="spec", self_mtp=True)
        with mock.patch.object(client_mod, "verified_artifact_supports", return_value=True), \
             mock.patch.object(client_mod, "create_self_mtp_session", return_value=runtime) as self_mtp, \
             mock.patch.object(client_mod, "create_persistent_mtp_session") as external:
            c._initialize_persistent_mtp_session()
        self_mtp.assert_called_once()
        external.assert_not_called()
        self.assertTrue(c._session.mtp_enabled)
        self.assertEqual(c._session.ctx_dft, "ctx-dft")

    def test_self_mtp_borrows_the_loaded_model_and_context(self) -> None:
        """No path is passed: the already-loaded handles are reused."""
        c = _client(mtp_requested=True)
        runtime = SimpleNamespace(ctx_dft="ctx-dft", spec="spec", self_mtp=True)
        with mock.patch.object(client_mod, "verified_artifact_supports", return_value=True), \
             mock.patch.object(client_mod, "create_self_mtp_session", return_value=runtime) as self_mtp:
            c._initialize_persistent_mtp_session()
        kwargs = self_mtp.call_args.kwargs
        self.assertEqual(kwargs["model"], "model-handle")
        self.assertEqual(kwargs["ctx_tgt"], "ctx-tgt-handle")
        self.assertNotIn("draft_model_path", kwargs)

    def test_legacy_artifact_does_not_select_self_mtp(self) -> None:
        c = _client(mtp_requested=True, model_path=LEGACY)
        with mock.patch.object(client_mod, "verified_artifact_supports", return_value=False), \
             mock.patch.object(client_mod, "create_self_mtp_session") as self_mtp:
            c._initialize_persistent_mtp_session()
        self_mtp.assert_not_called()
        self.assertFalse(c._session.mtp_enabled)
        self.assertIsNotNone(c._session.mtp_failure_reason)

    def test_legacy_artifact_makes_no_mtp_active_claim(self) -> None:
        c = _client(mtp_requested=True, model_path=LEGACY)
        with mock.patch.object(client_mod, "verified_artifact_supports", return_value=False):
            c._initialize_persistent_mtp_session()
        self.assertFalse(c._session.mtp_enabled)
        self.assertIsNone(c._session.ctx_dft)
        self.assertIsNone(c._session.spec)


class DigestBoundaryTests(unittest.TestCase):
    """Digest calls happen once, at the self-MTP boundary, and nowhere else."""

    def test_explicit_request_resolves_capability_exactly_once(self) -> None:
        c = _client(mtp_requested=True)
        runtime = SimpleNamespace(ctx_dft="d", spec="s", self_mtp=True)
        with mock.patch.object(client_mod, "verified_artifact_supports", return_value=True) as supports, \
             mock.patch.object(client_mod, "create_self_mtp_session", return_value=runtime):
            c._initialize_persistent_mtp_session()
        supports.assert_called_once()

    def test_a_profile_declaring_no_artifacts_is_ruled_out_without_hashing(self) -> None:
        """Gemma / external-draft models must not pay Ornith's 46 s digest."""
        c = _client(
            mtp_requested=True,
            profile_id=QWEN3_CODER_PROFILE_ID,
            mtp_supported=True,
            draft_mtp_model=Path("/models/draft.gguf"),
        )
        runtime = SimpleNamespace(ctx_dft="d", spec="s")
        with mock.patch("orbit.native_llama.artifact_capabilities.artifact_sha256") as hashed, \
             mock.patch.object(client_mod, "create_persistent_mtp_session", return_value=runtime):
            c._initialize_persistent_mtp_session()
        hashed.assert_not_called()

    def test_unverified_profile_is_ruled_out_without_hashing(self) -> None:
        c = _client(mtp_requested=True, verified=False)
        with mock.patch("orbit.native_llama.artifact_capabilities.artifact_sha256") as hashed:
            c._initialize_persistent_mtp_session()
        hashed.assert_not_called()


class ExternalDraftPreservedTests(unittest.TestCase):
    """The existing architecture keeps working, unchanged."""

    def test_external_draft_model_uses_the_existing_constructor(self) -> None:
        c = _client(
            mtp_requested=True,
            profile_id=QWEN3_CODER_PROFILE_ID,
            mtp_supported=True,
            draft_mtp_model=Path("/models/draft.gguf"),
        )
        runtime = SimpleNamespace(ctx_dft="ext-dft", spec="ext-spec")
        with mock.patch.object(client_mod, "verified_artifact_supports", return_value=False), \
             mock.patch.object(client_mod, "create_persistent_mtp_session", return_value=runtime) as external, \
             mock.patch.object(client_mod, "create_self_mtp_session") as self_mtp:
            c._initialize_persistent_mtp_session()
        external.assert_called_once()
        self_mtp.assert_not_called()
        self.assertTrue(c._session.mtp_enabled)
        self.assertEqual(c._session.ctx_dft, "ext-dft")

    def test_external_draft_still_reports_its_own_unsupported_reason(self) -> None:
        c = _client(mtp_requested=True, profile_id=QWEN3_CODER_PROFILE_ID, mtp_supported=False)
        with mock.patch.object(client_mod, "verified_artifact_supports", return_value=False):
            c._initialize_persistent_mtp_session()
        self.assertEqual(c._session.mtp_failure_reason, "model_profile_mtp_unsupported")


class FailurePathTests(unittest.TestCase):
    """Failures must fail closed and leave borrowed objects intact."""

    def test_capability_hash_error_fails_closed(self) -> None:
        c = _client(mtp_requested=True)
        with mock.patch.object(
            client_mod, "verified_artifact_supports", side_effect=OSError("disk")
        ), mock.patch.object(client_mod, "create_self_mtp_session") as self_mtp, \
           mock.patch.object(client_mod, "create_persistent_mtp_session") as external:
            c._initialize_persistent_mtp_session()
        self_mtp.assert_not_called()
        external.assert_not_called()
        self.assertFalse(c._session.mtp_enabled)
        self.assertTrue(c._session.mtp_failed)
        self.assertIn("capability-error", c._session.mtp_failure_reason)

    def test_self_mtp_create_failure_leaves_ownership_untouched(self) -> None:
        c = _client(mtp_requested=True)
        with mock.patch.object(client_mod, "verified_artifact_supports", return_value=True), \
             mock.patch.object(
                 client_mod, "create_self_mtp_session", side_effect=RuntimeError("no ctx")
             ):
            c._initialize_persistent_mtp_session()
        self.assertTrue(c._session.mtp_failed)
        self.assertFalse(c._session.mtp_enabled)
        self.assertIsNone(c._persistent_mtp_runtime)
        # Borrowed handles are exactly as they were: the client still owns them.
        self.assertEqual(c._model, "model-handle")
        self.assertEqual(c._session.ctx_tgt, "ctx-tgt-handle")

    def test_missing_target_context_fails_closed(self) -> None:
        c = _client(mtp_requested=True, ctx_tgt=None)
        with mock.patch.object(client_mod, "verified_artifact_supports", return_value=True), \
             mock.patch.object(client_mod, "create_self_mtp_session") as self_mtp:
            c._initialize_persistent_mtp_session()
        self_mtp.assert_not_called()
        self.assertEqual(c._session.mtp_failure_reason, "target-context-missing")

    def test_a_failed_self_mtp_does_not_fall_through_to_external_draft(self) -> None:
        """A qualified artifact that fails must not silently try the other path."""
        c = _client(mtp_requested=True, draft_mtp_model=Path("/models/draft.gguf"))
        with mock.patch.object(client_mod, "verified_artifact_supports", return_value=True), \
             mock.patch.object(
                 client_mod, "create_self_mtp_session", side_effect=RuntimeError("boom")
             ), mock.patch.object(client_mod, "create_persistent_mtp_session") as external:
            c._initialize_persistent_mtp_session()
        external.assert_not_called()


class SelfMtpConstructorTests(unittest.TestCase):
    """`create_self_mtp_session` itself: ABI demand and returned ownership.

    The client tests mock this function out, so nothing above exercises what it
    actually asks for or hands back. Both matter: a session that forgets to
    demand the self-MTP ABI can call a missing symbol, and one that forgets to
    mark itself `self_mtp` is torn down through the wrong destroy -- which frees
    a borrowed model.
    """

    def _create(self, lib):
        from orbit.native_llama.persistent_mtp import create_self_mtp_session

        with mock.patch(
            "orbit.native_llama.persistent_mtp.build_persistent_mtp_shim",
            return_value=Path("/shim.so"),
        ) as build:
            runtime = create_self_mtp_session(
                llama_root=Path("/llama"),
                paths=SimpleNamespace(
                    build_bin=Path("/bin"), model=CURRENT, llama_root=Path("/llama")
                ),
                model="model-handle",
                ctx_tgt="ctx-tgt-handle",
                context_tokens=8192, batch_size=256, ubatch_size=128,
                threads=6, threads_batch=6,
                library_factory=lambda *_a, **_k: SimpleNamespace(lib=lib),
            )
        return runtime, build

    @staticmethod
    def _lib(handle=0x1234):
        lib = mock.Mock()
        lib.orbit_selfmtp_session_create.return_value = handle
        lib.orbit_mtp_session_ctx_dft.return_value = "ctx-dft"
        lib.orbit_mtp_session_spec.return_value = "spec"
        for name in ("rss_before_kb", "rss_after_init_kb", "rss_peak_kb"):
            getattr(lib, f"orbit_mtp_session_{name}").return_value = 1
        return lib

    def test_construction_demands_the_self_mtp_abi(self) -> None:
        _, build = self._create(self._lib())
        self.assertIs(build.call_args.kwargs["require_self_mtp"], True)

    def test_construction_uses_the_self_mtp_entry_point(self) -> None:
        lib = self._lib()
        self._create(lib)
        lib.orbit_selfmtp_session_create.assert_called_once()
        lib.orbit_mtp_session_create.assert_not_called()
        args = lib.orbit_selfmtp_session_create.call_args.args
        self.assertEqual(args[0], "model-handle")
        self.assertEqual(args[1], "ctx-tgt-handle")

    def test_runtime_is_marked_self_mtp(self) -> None:
        """What routes teardown to destroy instead of the owning free."""
        runtime, _ = self._create(self._lib())
        self.assertTrue(runtime.self_mtp)

    def test_a_null_handle_raises_rather_than_returning_a_broken_runtime(self) -> None:
        lib = self._lib(handle=0)
        lib.orbit_mtp_last_error.return_value = b"boom"
        with self.assertRaises(RuntimeError):
            self._create(lib)

    def test_missing_model_or_context_is_refused_before_any_shim_work(self) -> None:
        from orbit.native_llama.persistent_mtp import create_self_mtp_session

        with mock.patch(
            "orbit.native_llama.persistent_mtp.build_persistent_mtp_shim"
        ) as build:
            for bad in ({"model": None}, {"ctx_tgt": None}):
                kwargs = dict(
                    llama_root=Path("/llama"),
                    paths=SimpleNamespace(build_bin=Path("/bin")),
                    model="m", ctx_tgt="c",
                    context_tokens=8192, batch_size=256, ubatch_size=128,
                    threads=6, threads_batch=6,
                )
                kwargs.update(bad)
                with self.subTest(**bad):
                    with self.assertRaises(RuntimeError):
                        create_self_mtp_session(**kwargs)
        build.assert_not_called()


class EligibilityShortCircuitTests(unittest.TestCase):
    """`_self_mtp_eligible` must rule out cheaply before it hashes."""

    def test_identity_without_declared_artifacts_short_circuits(self) -> None:
        c = _client(mtp_requested=True, profile_id=QWEN3_CODER_PROFILE_ID)
        with mock.patch.object(client_mod, "verified_artifact_supports") as supports:
            self.assertFalse(c._self_mtp_eligible())
        supports.assert_not_called()

    def test_ornith_identity_proceeds_to_the_content_check(self) -> None:
        c = _client(mtp_requested=True)
        with mock.patch.object(
            client_mod, "verified_artifact_supports", return_value=True
        ) as supports:
            self.assertTrue(c._self_mtp_eligible())
        supports.assert_called_once()
        self.assertEqual(supports.call_args.args[1], SELF_MTP_CAPABILITY)


class FailureDoesNotFallThroughTests(unittest.TestCase):
    """A qualified artifact that fails must not silently try external-draft.

    Asserted without a draft model configured too, so the guard is the return
    value rather than the absence of a draft path.
    """

    def test_create_failure_stops_the_chain(self) -> None:
        c = _client(mtp_requested=True, mtp_supported=True,
                    draft_mtp_model=Path("/models/draft.gguf"))
        with mock.patch.object(client_mod, "verified_artifact_supports", return_value=True),              mock.patch.object(
                 client_mod, "create_self_mtp_session", side_effect=RuntimeError("x")
             ), mock.patch.object(client_mod, "create_persistent_mtp_session") as external:
            c._initialize_persistent_mtp_session()
        external.assert_not_called()
        self.assertTrue(c._session.mtp_failed)

    def test_hash_error_stops_the_chain(self) -> None:
        c = _client(mtp_requested=True, mtp_supported=True,
                    draft_mtp_model=Path("/models/draft.gguf"))
        with mock.patch.object(
            client_mod, "verified_artifact_supports", side_effect=OSError("io")
        ), mock.patch.object(client_mod, "create_persistent_mtp_session") as external:
            c._initialize_persistent_mtp_session()
        external.assert_not_called()
        self.assertIn("capability-error", c._session.mtp_failure_reason)


class ResetPreservesOwnershipTests(unittest.TestCase):
    """Reset returns a NEW runtime around the SAME native handle.

    `_try_complete_with_mtp_experimental` resets before every completion. If
    the rebuilt runtime lost `self_mtp`, a borrowing session would become an
    owning one after the first request -- and teardown would free the model the
    client still owns. The flag has to be carried, not re-derived.
    """

    def _reset(self, self_mtp: bool):
        from orbit.native_llama.persistent_mtp import (
            PersistentMtpSessionRuntime,
            reset_persistent_mtp_session,
        )

        lib = mock.Mock()
        lib.orbit_mtp_session_reset.return_value = True
        lib.orbit_mtp_session_ctx_dft.return_value = "ctx-dft"
        lib.orbit_mtp_session_spec.return_value = "spec"
        before = PersistentMtpSessionRuntime(
            handle="native-handle", ctx_dft="d", spec="s",
            library=SimpleNamespace(lib=lib), self_mtp=self_mtp,
        )
        return reset_persistent_mtp_session(
            llama_root=Path("/llama"),
            paths=SimpleNamespace(build_bin=Path("/bin")),
            runtime=before,
            ctx_tgt="ctx-tgt",
        )

    def test_reset_keeps_a_self_mtp_session_borrowing(self) -> None:
        self.assertTrue(self._reset(self_mtp=True).self_mtp)

    def test_reset_keeps_an_external_draft_session_owning(self) -> None:
        self.assertFalse(self._reset(self_mtp=False).self_mtp)

    def test_reset_reuses_the_same_native_handle(self) -> None:
        self.assertEqual(self._reset(self_mtp=True).handle, "native-handle")


class TeardownOrderTests(unittest.TestCase):
    """Self-MTP destroy runs before the client frees what it lent."""

    def test_free_routes_self_mtp_sessions_to_destroy(self) -> None:
        from orbit.native_llama.persistent_mtp import (
            PersistentMtpSessionRuntime,
            free_persistent_mtp_session,
        )

        lib = mock.Mock()
        runtime = PersistentMtpSessionRuntime(
            handle="h", ctx_dft="d", spec="s",
            library=SimpleNamespace(lib=lib), self_mtp=True,
        )
        free_persistent_mtp_session(
            llama_root=Path("/llama"),
            paths=SimpleNamespace(build_bin=Path("/bin")),
            runtime=runtime,
        )
        lib.orbit_selfmtp_session_destroy.assert_called_once_with("h")
        lib.orbit_mtp_session_free.assert_not_called()

    def test_free_routes_external_draft_sessions_to_the_old_free(self) -> None:
        from orbit.native_llama.persistent_mtp import (
            PersistentMtpSessionRuntime,
            free_persistent_mtp_session,
        )

        lib = mock.Mock()
        runtime = PersistentMtpSessionRuntime(
            handle="h", ctx_dft="d", spec="s",
            library=SimpleNamespace(lib=lib), self_mtp=False,
        )
        free_persistent_mtp_session(
            llama_root=Path("/llama"),
            paths=SimpleNamespace(build_bin=Path("/bin")),
            runtime=runtime,
        )
        lib.orbit_mtp_session_free.assert_called_once_with("h")
        lib.orbit_selfmtp_session_destroy.assert_not_called()

    def test_repeated_free_destroys_only_once(self) -> None:
        c = _client(mtp_requested=True)
        lib = mock.Mock()
        from orbit.native_llama.persistent_mtp import PersistentMtpSessionRuntime

        c._persistent_mtp_runtime = PersistentMtpSessionRuntime(
            handle="h", ctx_dft="d", spec="s",
            library=SimpleNamespace(lib=lib), self_mtp=True,
        )
        c._free_persistent_mtp_session()
        c._free_persistent_mtp_session()
        self.assertEqual(lib.orbit_selfmtp_session_destroy.call_count, 1)
        self.assertIsNone(c._persistent_mtp_runtime)

    def test_client_close_frees_the_session_before_context_and_model(self) -> None:
        """Ordering is the whole safety argument; assert it, don't assume it."""
        import inspect

        source = inspect.getsource(NativeLlamaClient.close)
        free_at = source.index("_free_persistent_mtp_session")
        ctx_at = source.index("llama_free(self._session.ctx_tgt)")
        model_at = source.index("llama_model_free(self._model)")
        self.assertLess(free_at, ctx_at, "MTP session must be freed before ctx_tgt")
        self.assertLess(ctx_at, model_at, "ctx_tgt must be freed before the model")


if __name__ == "__main__":
    unittest.main()
