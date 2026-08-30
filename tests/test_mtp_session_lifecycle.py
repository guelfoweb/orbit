"""The MTP session lifecycle: one owner for the runtime and its derived state.

An MTP session is a native runtime plus four session fields that project it
(`ctx_dft`, `spec`, `mtp_enabled`, and the `mtp_failed`/`mtp_failure_reason`
pair). Those transitions were written out by hand at six publish sites and five
failure sites before the extraction, which is how they drift apart.

These characterize the transitions behaviourally -- constructing a real
`MtpSessionLifecycle` over a real `NativeSessionState` and asserting the state
it leaves behind -- so a change to the mechanics changes the result here.

Ownership is the load-bearing part: `free` tears down what this object owns,
`discard` drops a runtime the native side already tore down. Confusing the two
is a double free or a leak, so both are pinned.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.native_llama.mtp_session_lifecycle import MtpSessionLifecycle
from orbit.native_llama.session_state import NativeSessionState


class _Runtime:
    def __init__(self, name="rt"):
        self.name = name
        self.ctx_dft = f"{name}-ctx"
        self.spec = f"{name}-spec"


def _lifecycle():
    """A lifecycle over a real session, recording every teardown."""
    session = NativeSessionState(session_id="lifecycle")
    freed: list[object] = []
    owner = MtpSessionLifecycle(lambda: session, free_runtime=freed.append)
    return owner, session, freed


class PublishTests(unittest.TestCase):
    """A constructed runtime becomes the live session, consistently."""

    def test_publish_records_runtime_and_derived_state(self) -> None:
        owner, session, _ = _lifecycle()
        runtime = _Runtime()

        owner.publish(runtime)

        self.assertIs(owner.runtime, runtime, "the runtime must be retained")
        self.assertEqual(session.ctx_dft, "rt-ctx")
        self.assertEqual(session.spec, "rt-spec")
        self.assertTrue(session.mtp_enabled, "a published session must be enabled")
        self.assertFalse(session.mtp_failed)

    def test_publish_leaves_a_previous_reason_unless_asked(self) -> None:
        """Initialization keeps a recorded reason; reset clears it.

        The two call-site shapes are preserved rather than unified: collapsing
        them would change what `/props` reports after a recovered session.
        """
        owner, session, _ = _lifecycle()
        session.mtp_failure_reason = "earlier-reason"

        owner.publish(_Runtime())
        self.assertEqual(session.mtp_failure_reason, "earlier-reason")

        owner.publish(_Runtime(), clear_failure_reason=True)
        self.assertIsNone(session.mtp_failure_reason)


class FailureTests(unittest.TestCase):
    def test_record_failure_marks_failed_without_disabling(self) -> None:
        owner, session, _ = _lifecycle()
        session.mtp_enabled = True

        owner.record_failure("boom")

        self.assertTrue(session.mtp_failed)
        self.assertEqual(session.mtp_failure_reason, "boom")
        self.assertTrue(session.mtp_enabled, "init failure must not disable")

    def test_record_failure_can_disable_a_live_session(self) -> None:
        owner, session, _ = _lifecycle()
        session.mtp_enabled = True

        owner.record_failure("fatal", disable=True)

        self.assertFalse(session.mtp_enabled)
        self.assertTrue(session.mtp_failed)

    def test_clear_state_resets_the_whole_projection(self) -> None:
        """Initialization starts from a clean projection, reason included.

        A surviving `mtp_failure_reason` would be reported for a session that
        has not failed -- the stale-diagnostic class of bug.
        """
        owner, session, _ = _lifecycle()
        owner.publish(_Runtime())
        session.mtp_failed = True
        session.mtp_failure_reason = "stale"

        owner.clear_state()

        self.assertIsNone(session.ctx_dft)
        self.assertIsNone(session.spec)
        self.assertFalse(session.mtp_enabled)
        self.assertFalse(session.mtp_failed)
        self.assertIsNone(
            session.mtp_failure_reason,
            "a stale reason would be reported for a session that never failed",
        )


class OwnershipTests(unittest.TestCase):
    """Who frees what. Getting this wrong is a double free or a leak."""

    def test_free_tears_down_exactly_once_and_clears_state(self) -> None:
        owner, session, freed = _lifecycle()
        runtime = _Runtime()
        owner.publish(runtime)

        owner.free()

        self.assertEqual(freed, [runtime], "the owned runtime must be freed once")
        self.assertIsNone(owner.runtime)
        self.assertIsNone(session.ctx_dft)
        self.assertIsNone(session.spec)
        self.assertFalse(
            session.mtp_enabled,
            "a torn-down session must not still report itself enabled",
        )

    def test_repeated_free_is_safe(self) -> None:
        """Teardown is idempotent: a second free must not re-free."""
        owner, _, freed = _lifecycle()
        owner.publish(_Runtime())

        owner.free()
        owner.free()
        owner.free()

        self.assertEqual(len(freed), 1, "repeated free must not free again")

    def test_free_without_a_session_is_a_no_op(self) -> None:
        owner, _, freed = _lifecycle()
        owner.free()
        self.assertEqual(freed, [])

    def test_discard_does_not_free(self) -> None:
        """A runtime the native side already tore down must NOT be freed.

        `discard` is the failed-reset path: the native session destroyed itself,
        so calling free here would be a double free.
        """
        owner, session, freed = _lifecycle()
        owner.publish(_Runtime())

        owner.discard("reset-failed")

        self.assertEqual(freed, [], "discard must never free; that is a double free")
        self.assertIsNone(owner.runtime)
        self.assertFalse(session.mtp_enabled)
        self.assertTrue(session.mtp_failed)
        self.assertEqual(session.mtp_failure_reason, "reset-failed")

    def test_state_is_cleared_even_if_teardown_raises(self) -> None:
        """A raising teardown must not leave the session naming freed handles."""
        session = NativeSessionState(session_id="raises")

        def boom(_runtime):
            raise RuntimeError("native teardown failed")

        owner = MtpSessionLifecycle(lambda: session, free_runtime=boom)
        owner.publish(_Runtime())

        with self.assertRaises(RuntimeError):
            owner.free()

        self.assertIsNone(owner.runtime)
        self.assertIsNone(session.ctx_dft)
        self.assertFalse(session.mtp_enabled)


class SingleOwnerTests(unittest.TestCase):
    def test_the_session_projects_the_owner_not_a_copy(self) -> None:
        """There is one runtime, and the session fields describe it."""
        owner, session, _ = _lifecycle()
        first, second = _Runtime("a"), _Runtime("b")

        owner.publish(first)
        self.assertEqual(session.ctx_dft, "a-ctx")
        owner.publish(second)
        self.assertEqual(session.ctx_dft, "b-ctx")
        self.assertIs(owner.runtime, second, "no stale runtime may survive")

    def test_the_lifecycle_never_imports_the_client(self) -> None:
        """The collaborator must not reach back into the client."""
        source = (SRC / "orbit/native_llama/mtp_session_lifecycle.py").read_text()
        self.assertNotIn("from .client", source)
        self.assertNotIn("import client", source)


if __name__ == "__main__":
    unittest.main()


class ClientFailedResetTests(unittest.TestCase):
    """The failed-reset path must DISCARD the runtime, never free it.

    When `reset_persistent_mtp_session` raises, the native session has already
    torn itself down. Freeing it again is a double free -- a real crash, not a
    lost optimisation. The collaborator's own `discard` is pinned above; this
    pins that the CLIENT reaches for it at that call site, which is where the
    choice is actually made.
    """

    def test_a_failed_reset_discards_without_freeing(self) -> None:
        """Extracted from `reset_session_state`'s MTP block, run verbatim.

        Driving the whole method needs a dozen unrelated collaborators, and
        stubbing them all would test the stubs. The MTP block is lifted from
        the shipped source and executed against a real lifecycle, so the choice
        of `discard` over `free` at that call site is what is under test.
        """
        import ast
        import inspect
        import types

        from orbit.native_llama import client as client_module
        from orbit.native_llama.client import NativeLlamaClient

        source = inspect.getsource(NativeLlamaClient.reset_session_state)
        tree = ast.parse(source.lstrip())
        block = [
            node for node in tree.body[0].body
            if isinstance(node, ast.Try)
            and "reset_persistent_mtp_session" in ast.unparse(node)
        ]
        self.assertEqual(
            len(block), 1,
            "the failed-reset block moved; update this extraction rather than "
            "letting it bind to a stale shape",
        )

        session = NativeSessionState(session_id="failed-reset")
        freed: list[object] = []
        lifecycle = MtpSessionLifecycle(lambda: session, free_runtime=freed.append)
        lifecycle.publish(_Runtime())

        class _Client:
            _session = session
            paths = types.SimpleNamespace(llama_root=None)
            _persistent_mtp_runtime = lifecycle.runtime

            def _mtp_lifecycle_owner(self):
                return lifecycle

        def exploding_reset(**_kwargs):
            raise RuntimeError("reset exploded")

        namespace = {
            "self": _Client(),
            "reset_persistent_mtp_session": exploding_reset,
        }
        # The block contains a bare `return`, so it is wrapped in a function
        # body rather than executed at module level. The statements themselves
        # are the shipped ones, unmodified.
        wrapper = ast.parse("def _run(self, reset_persistent_mtp_session):\n    pass\n")
        wrapper.body[0].body = block
        ast.fix_missing_locations(wrapper)
        scope = dict(vars(client_module))
        exec(  # noqa: S102 - the shipped block, executed verbatim
            compile(wrapper, "<reset-block>", "exec"), scope, scope
        )
        scope["_run"](namespace["self"], exploding_reset)

        self.assertEqual(
            freed, [],
            "a failed reset must not free the runtime; the native session has "
            "already torn itself down, so freeing again is a double free",
        )
        self.assertIsNone(lifecycle.runtime)
        self.assertFalse(session.mtp_enabled)
        self.assertTrue(session.mtp_failed)
        self.assertIn("reset exploded", session.mtp_failure_reason or "")
