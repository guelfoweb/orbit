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


class AttachAndReadyTests(unittest.TestCase):
    """`attach` and `mark_ready` split what `publish` does in one step.

    The completion path needs each half separately: it points the session at a
    runtime BEFORE knowing whether the turn succeeds, and declares readiness
    only afterwards. Using `publish` in either place would flip the verdict at
    the wrong moment.
    """

    def test_attach_leaves_the_verdict_untouched(self) -> None:
        """The decisive asymmetry: attach must NOT declare the session ready.

        At this point the completion has not run. A failed session flipped to
        enabled here would report success before anything succeeded.
        """
        owner, session, _ = _lifecycle()
        session.mtp_enabled = False
        session.mtp_failed = True
        session.mtp_failure_reason = "previous failure"
        runtime = _Runtime()

        owner.attach(runtime)

        self.assertIs(owner.runtime, runtime)
        self.assertEqual(session.ctx_dft, "rt-ctx")
        self.assertEqual(session.spec, "rt-spec")
        self.assertFalse(session.mtp_enabled, "attach must not enable")
        self.assertTrue(session.mtp_failed, "attach must not clear the failure")
        self.assertEqual(session.mtp_failure_reason, "previous failure")

    def test_mark_ready_declares_health_without_a_runtime(self) -> None:
        """The counterpart: a verdict with no handle changing hands."""
        owner, session, _ = _lifecycle()
        runtime = _Runtime()
        owner.attach(runtime)
        session.mtp_failed = True
        session.mtp_failure_reason = "stale"

        owner.mark_ready()

        self.assertIs(owner.runtime, runtime, "mark_ready must not drop the runtime")
        self.assertTrue(session.mtp_enabled)
        self.assertFalse(session.mtp_failed)
        self.assertIsNone(session.mtp_failure_reason)

    def test_attach_then_mark_ready_equals_publish(self) -> None:
        """Together they reach the same state `publish` produces.

        Which is exactly why they must stay separate: the completion path needs
        the gap between them.
        """
        a_owner, a_session, _ = _lifecycle()
        b_owner, b_session, _ = _lifecycle()
        runtime = _Runtime()

        a_owner.attach(runtime)
        a_owner.mark_ready()
        b_owner.publish(runtime, clear_failure_reason=True)

        for field in ("ctx_dft", "spec", "mtp_enabled", "mtp_failed", "mtp_failure_reason"):
            self.assertEqual(
                getattr(a_session, field), getattr(b_session, field), field
            )


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


class HotPathDelegationTests(unittest.TestCase):
    """The completion path must reach the RIGHT lifecycle operation.

    The collaborator's own tests pin what each method does; these pin which one
    the client calls. That gap is real: a mutant swapping `attach` for
    `publish` in the completion path passes every collaborator test while
    declaring a failed session ready before the turn has run.
    """

    def _calls(self):
        """Every lifecycle method invoked by the completion path, in order."""
        import ast
        import inspect

        from orbit.native_llama.client import NativeLlamaClient

        tree = ast.parse(
            inspect.getsource(
                NativeLlamaClient._try_complete_with_mtp_experimental
            ).lstrip()
        ).body[0]
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                rendered = ast.unparse(node)
                marker = "_mtp_lifecycle_owner()."
                if marker in rendered and rendered.count("(") > 1:
                    found.append(rendered.split(marker, 1)[1])
        return found

    def test_the_completion_path_owns_no_lifecycle_state(self) -> None:
        """No direct writes to lifecycle-owned fields may remain."""
        import inspect

        from orbit.native_llama.client import NativeLlamaClient

        body = inspect.getsource(
            NativeLlamaClient._try_complete_with_mtp_experimental
        )
        for field in ("mtp_enabled", "mtp_failed", "mtp_failure_reason",
                      "ctx_dft", "spec"):
            self.assertNotIn(
                f"_session.{field} =", body,
                f"the completion path still writes {field} directly; the "
                f"lifecycle owner must be the only writer",
            )

    def test_the_runtime_is_attached_not_published(self) -> None:
        """Before the completion runs, readiness is not yet known.

        `publish` here would enable a session that has failed, and clear its
        failure, before anything has succeeded.

        Secondary pin. The primary evidence is behavioural: the executed
        hot-path tests below drive the shipped block and assert resulting
        state. This one only names the call site, which is cheap to keep and
        localises a failure, but it must never be the only thing standing
        between a mutation and a green suite.
        """
        calls = self._calls()
        self.assertTrue(
            any(c.startswith("attach(") for c in calls),
            "the completion path must attach the runtime, not publish it",
        )
        self.assertFalse(
            any(c.startswith("publish(runtime)") for c in calls),
            "an unqualified publish would declare readiness too early",
        )

    def _run_failure_block(self, *, cancelled: bool, lifecycle, session,
                           reset_raises=False):
        """Execute the shipped `if not result.success:` block verbatim.

        Extracted from the completion path and run against a real lifecycle, so
        the assertions below are about behaviour rather than source shape. The
        block is 14 lines with 7 stubbable collaborators, which is why it can be
        driven directly where the whole 196-line method cannot.
        """
        import ast
        import inspect
        import types

        from orbit.native_llama import client as client_module
        from orbit.native_llama.client import NativeLlamaClient

        tree = ast.parse(
            inspect.getsource(
                NativeLlamaClient._try_complete_with_mtp_experimental
            ).lstrip()
        ).body[0]
        blocks = [
            n for n in tree.body
            if isinstance(n, ast.If) and "result.success" in ast.unparse(n.test)
            and len(ast.unparse(n).splitlines()) > 5
        ]
        self.assertEqual(len(blocks), 1, "the failure block moved; update this")

        wrapper = ast.parse(
            "def _run(self, result, reset_persistent_mtp_session, MtpCompletionResult):\n    pass\n"
        )
        wrapper.body[0].body = blocks
        ast.fix_missing_locations(wrapper)

        class _Client:
            def __init__(self):
                self._session = session
                self.paths = types.SimpleNamespace(llama_root=None)
                self.cancel_event = types.SimpleNamespace(
                    is_set=lambda: cancelled
                )
                self.last_mtp_completion = None
                self.mtp_fallback_reason = None
                self._persistent_mtp_runtime = lifecycle.runtime

            def _mtp_lifecycle_owner(self):
                return lifecycle

        def reset(**_kwargs):
            if reset_raises:
                raise RuntimeError("reset exploded")
            return _Runtime("rebuilt")

        scope = dict(vars(client_module))
        exec(  # noqa: S102 - the shipped block, executed verbatim
            compile(wrapper, "<failure-block>", "exec"), scope, scope
        )
        client = _Client()
        scope["_run"](
            client,
            types.SimpleNamespace(success=False, error="it broke"),
            reset,
            lambda **kw: types.SimpleNamespace(**kw),
        )
        return client

    def test_a_failed_completion_disables_the_session(self) -> None:
        """A broken runtime must not be offered again on the next turn."""
        owner, session, _ = _lifecycle()
        owner.publish(_Runtime())

        self._run_failure_block(cancelled=False, lifecycle=owner, session=session)

        self.assertFalse(
            session.mtp_enabled,
            "a failed completion must disable MTP; leaving it enabled offers "
            "the broken runtime again",
        )
        self.assertTrue(session.mtp_failed)
        self.assertEqual(session.mtp_failure_reason, "it broke")

    def test_a_cancelled_turn_rebuilds_and_clears_the_reason(self) -> None:
        """A recovered session must not keep reporting the cancelled turn."""
        owner, session, _ = _lifecycle()
        owner.publish(_Runtime())
        session.mtp_failure_reason = "stale"

        self._run_failure_block(cancelled=True, lifecycle=owner, session=session)

        self.assertTrue(session.mtp_enabled, "a rebuilt session is usable again")
        self.assertFalse(session.mtp_failed)
        self.assertIsNone(
            session.mtp_failure_reason,
            "a surviving reason would be reported for a recovered session",
        )
        self.assertEqual(owner.runtime.name, "rebuilt")

    def test_a_failed_rebuild_after_cancel_disables_without_freeing(self) -> None:
        """The reset raised, so the native session is gone: drop, never free."""
        owner, session, freed = _lifecycle()
        owner.publish(_Runtime())

        self._run_failure_block(
            cancelled=True, lifecycle=owner, session=session, reset_raises=True
        )

        self.assertFalse(session.mtp_enabled)
        self.assertTrue(session.mtp_failed)
        self.assertIn("reset exploded", session.mtp_failure_reason or "")
        self.assertEqual(
            freed, [],
            "a failed rebuild must not free; the native session already tore "
            "itself down and freeing again is a double free",
        )

    def test_a_successful_completion_marks_the_session_ready(self) -> None:
        """The success path must declare health, not merely skip the failure.

        Executed rather than inspected: the statements that run after the
        failure block are lifted from the shipped method and driven against a
        real lifecycle, so a no-op there is visible as a session that never
        becomes enabled.
        """
        import ast
        import inspect
        import types

        from orbit.native_llama import client as client_module
        from orbit.native_llama.client import NativeLlamaClient

        tree = ast.parse(
            inspect.getsource(
                NativeLlamaClient._try_complete_with_mtp_experimental
            ).lstrip()
        ).body[0]
        # The two statements immediately following the failure block: clearing
        # the fallback reason and declaring the session ready.
        idx = [
            i for i, n in enumerate(tree.body)
            if isinstance(n, ast.If) and "result.success" in ast.unparse(n.test)
            and len(ast.unparse(n).splitlines()) > 5
        ]
        self.assertEqual(len(idx), 1, "the failure block moved; update this")
        success_stmts = tree.body[idx[0] + 1: idx[0] + 3]
        self.assertTrue(success_stmts, "no statements follow the failure block")

        owner, session, _ = _lifecycle()
        owner.publish(_Runtime())
        session.mtp_enabled = False
        session.mtp_failed = True
        session.mtp_failure_reason = "earlier turn failed"

        wrapper = ast.parse("def _run(self):\n    pass\n")
        wrapper.body[0].body = list(success_stmts)
        ast.fix_missing_locations(wrapper)

        class _Client:
            mtp_fallback_reason = "stale"
            _session = session

            def _mtp_lifecycle_owner(self):
                return owner

        scope = dict(vars(client_module))
        exec(  # noqa: S102 - the shipped statements, executed verbatim
            compile(wrapper, "<success-stmts>", "exec"), scope, scope
        )
        scope["_run"](_Client())

        self.assertTrue(
            session.mtp_enabled,
            "a successful completion must declare the session ready; leaving "
            "it disabled would refuse MTP on the next turn for no reason",
        )
        self.assertFalse(session.mtp_failed)
        self.assertIsNone(session.mtp_failure_reason)

    def test_the_post_cancel_rebuild_clears_the_failure_reason(self) -> None:
        """A rebuilt session must not keep reporting the cancelled turn.

        Secondary pin; `test_a_cancelled_turn_rebuilds_and_clears_the_reason`
        is the executed one that carries the weight.
        """
        calls = self._calls()
        publishes = [c for c in calls if c.startswith("publish(")]
        self.assertTrue(publishes)
        for call in publishes:
            self.assertIn(
                "clear_failure_reason=True", call,
                "the rebuild publishes a healthy session; a surviving reason "
                "would be reported for a session that recovered",
            )
