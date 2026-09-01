"""Grounded finalization must not quote a value it has already corrected.

The failure this guards against was measured: an analysis wrote an IoC report
artifact, later rewrote it at the same handle with a corrected domain, and
explicitly revalidated the correction against the source -- and the final
grounded report still cited the superseded spelling. Both versions were in the
store, both re-attested, and nothing said which one was current.

Standing is decided from provenance alone: same durable handle, newer digest.
No content is compared, so a record that merely disagrees in prose supersedes
nothing, and a record that quotes an old value to correct it keeps its place.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.analysis_runtime import ANALYSIS_TOOL_NAME
from orbit.runtime.evidence_authority import (
    ACTIVE,
    SUPERSEDED,
    active_records,
    evaluate_standing,
)


class _Record:
    """The shape `evaluate_standing` reads: an id and artifact provenance."""

    def __init__(self, evidence_id: str, artifacts=(), sequence: int | None = None):
        self.evidence_id = evidence_id
        self.evidence_sequence = sequence
        self.metadata = {
            "artifacts": [
                {"handle": handle, "sha256": digest} for handle, digest in artifacts
            ]
        }


HANDLE = "/workspace/work/ioc_report.txt"


class SupersessionTests(unittest.TestCase):
    """The measured class of failure, reproduced generically."""

    def _abc(self):
        # A: report v1 carries value X.  B: same handle rewritten as v2 with Y.
        # C: a validation record that confirms Y and quotes X to contradict it.
        a = _Record("ev_a", [(HANDLE, "sha_v1")], 1)
        b = _Record("ev_b", [(HANDLE, "sha_v2")], 2)
        c = _Record("ev_c", [], 3)
        return a, b, c

    def test_the_earlier_version_of_a_handle_is_superseded(self) -> None:
        a, b, c = self._abc()
        standing = evaluate_standing([a, b, c])

        self.assertEqual(standing["ev_a"].status, SUPERSEDED)
        self.assertEqual(standing["ev_a"].superseded_by, "ev_b")
        self.assertEqual(standing["ev_a"].handle, HANDLE)
        self.assertEqual(standing["ev_b"].status, ACTIVE)

    def test_the_correction_record_keeps_its_standing(self) -> None:
        """C quotes the wrong value to contradict it; it is not the wrong value.

        This is why standing is decided on handles rather than on content: a
        rule that demoted whatever mentions a stale string would delete the
        very record that explains the correction.
        """
        a, b, c = self._abc()
        standing = evaluate_standing([a, b, c])
        self.assertEqual(standing["ev_c"].status, ACTIVE)

    def test_the_superseded_record_is_excluded_from_the_report(self) -> None:
        a, b, c = self._abc()
        active = active_records([a, b, c])
        self.assertEqual([r.evidence_id for r in active], ["ev_b", "ev_c"])

    def test_nothing_is_deleted(self) -> None:
        """Standing decides what may be cited, never what exists."""
        a, b, c = self._abc()
        given = [a, b, c]
        active_records(given)
        self.assertEqual([r.evidence_id for r in given], ["ev_a", "ev_b", "ev_c"])
        self.assertIn("ev_a", evaluate_standing(given))

    def test_an_unrelated_artifact_is_never_superseded(self) -> None:
        """Only versions of the same handle compete.

        Several distinct handles are written between the two versions of
        HANDLE, so a lookup that consulted the wrong handle -- the most recent
        one, say -- would demote a record that nothing replaced. Two handles
        are not enough to catch that: the wrong answer can coincide with the
        right one.
        """
        a = _Record("ev_a", [(HANDLE, "sha_v1")], 1)
        others = [
            _Record(f"ev_other{i}", [(f"/workspace/work/other{i}.bin", f"s{i}")], i + 2)
            for i in range(4)
        ]
        b = _Record("ev_b", [(HANDLE, "sha_v2")], 9)

        standing = evaluate_standing([a, *others, b])

        for other in others:
            self.assertEqual(
                standing[other.evidence_id].status,
                ACTIVE,
                f"{other.evidence_id} was replaced by nothing",
            )
        self.assertEqual(standing["ev_a"].status, SUPERSEDED)
        self.assertEqual(standing["ev_a"].superseded_by, "ev_b")
        self.assertEqual(standing["ev_b"].status, ACTIVE)

    def test_supersession_names_the_handle_that_replaced_it(self) -> None:
        """The recorded reason must identify the actual handle, not any handle."""
        a = _Record("ev_a", [(HANDLE, "v1")], 1)
        noise = _Record("ev_noise", [("/workspace/work/unrelated.bin", "n1")], 2)
        b = _Record("ev_b", [(HANDLE, "v2")], 3)

        standing = evaluate_standing([a, noise, b])

        self.assertEqual(standing["ev_a"].handle, HANDLE)
        self.assertEqual(standing["ev_a"].superseded_by, "ev_b")

    def test_a_record_with_no_artifact_is_always_active(self) -> None:
        observation = _Record("ev_obs", [], 1)
        later = _Record("ev_later", [(HANDLE, "sha_v1")], 2)
        standing = evaluate_standing([observation, later])
        self.assertEqual(standing["ev_obs"].status, ACTIVE)

    def test_rewriting_a_handle_with_identical_bytes_supersedes_nothing(self) -> None:
        """A rewrite that changed nothing did not produce a new version."""
        a = _Record("ev_a", [(HANDLE, "same")], 1)
        b = _Record("ev_b", [(HANDLE, "same")], 2)
        standing = evaluate_standing([a, b])
        self.assertEqual(standing["ev_a"].status, ACTIVE)
        self.assertEqual(standing["ev_b"].status, ACTIVE)

    def test_a_record_keeping_one_current_artifact_stays_active(self) -> None:
        """Partial replacement must not discard what is still current."""
        a = _Record("ev_a", [(HANDLE, "v1"), ("/workspace/work/keep.bin", "k1")], 1)
        b = _Record("ev_b", [(HANDLE, "v2")], 2)

        standing = evaluate_standing([a, b])

        self.assertEqual(standing["ev_a"].status, ACTIVE)

    def test_three_versions_leave_only_the_newest_active(self) -> None:
        v1 = _Record("ev_1", [(HANDLE, "a")], 1)
        v2 = _Record("ev_2", [(HANDLE, "b")], 2)
        v3 = _Record("ev_3", [(HANDLE, "c")], 3)

        standing = evaluate_standing([v1, v2, v3])

        self.assertEqual(standing["ev_1"].status, SUPERSEDED)
        self.assertEqual(standing["ev_2"].status, SUPERSEDED)
        self.assertEqual(standing["ev_3"].status, ACTIVE)

    def test_standing_does_not_depend_on_input_order(self) -> None:
        """Which version is current is the store's answer, not the caller's.

        The records arrive as `records.values()`, insertion-ordered today. A
        reversed or re-serialized index would otherwise invert every verdict and
        mark the *corrected* version superseded -- reintroducing, silently and
        with confidence, the exact failure this module exists to prevent.
        """
        import itertools

        v1 = _Record("ev_v1", [(HANDLE, "digest_v1")], 10)
        v2 = _Record("ev_v2", [(HANDLE, "digest_v2")], 14)

        for order in itertools.permutations([v1, v2]):
            with self.subTest(order=[r.evidence_id for r in order]):
                standing = evaluate_standing(list(order))
                self.assertEqual(standing["ev_v1"].status, SUPERSEDED)
                self.assertEqual(standing["ev_v2"].status, ACTIVE)

    def test_order_independence_holds_for_three_versions(self) -> None:
        import itertools

        versions = [
            _Record("ev_1", [(HANDLE, "a")], 1),
            _Record("ev_2", [(HANDLE, "b")], 2),
            _Record("ev_3", [(HANDLE, "c")], 3),
        ]
        for order in itertools.permutations(versions):
            with self.subTest(order=[r.evidence_id for r in order]):
                standing = evaluate_standing(list(order))
                self.assertEqual(standing["ev_3"].status, ACTIVE)
                self.assertEqual(standing["ev_1"].status, SUPERSEDED)
                self.assertEqual(standing["ev_2"].status, SUPERSEDED)

    def test_an_unsequenced_record_cannot_supersede_a_sequenced_one(self) -> None:
        """A record the store never numbered is not evidence of being newer.

        Treating it as newest inverts the module's own purpose: a stale version
        from a legacy or damaged index would supersede the correction that
        replaced it. It sorts first instead -- it may be superseded, but it
        supersedes nothing that carries a sequence.
        """
        stale_unsequenced = _Record("ev_none", [(HANDLE, "stale")], None)
        corrected = _Record("ev_seq", [(HANDLE, "current")], 5)

        for order in ([stale_unsequenced, corrected], [corrected, stale_unsequenced]):
            with self.subTest(order=[r.evidence_id for r in order]):
                standing = evaluate_standing(order)
                self.assertEqual(standing["ev_none"].status, SUPERSEDED)
                self.assertEqual(standing["ev_seq"].status, ACTIVE)

    def test_sequence_zero_is_not_treated_as_missing(self) -> None:
        """`0` is a real sequence; a falsy check would confuse it with None.

        The discriminating case needs an unsequenced record in the same chain:
        if `0` were read as missing, it would tie with the unsequenced record
        and the tie could resolve either way. Ordered correctly, `0` is newer
        than unsequenced and older than everything else.
        """
        unsequenced = _Record("ev_none", [(HANDLE, "u")], None)
        zero = _Record("ev_zero", [(HANDLE, "a")], 0)
        later = _Record("ev_five", [(HANDLE, "b")], 5)

        # Every arrival order, because a key that conflated 0 with None would
        # leave their relative order to sort stability -- correct in some
        # inputs and wrong in others.
        import itertools

        for order in itertools.permutations([later, zero, unsequenced]):
            with self.subTest(order=[r.evidence_id for r in order]):
                standing = evaluate_standing(list(order))
                self.assertEqual(standing["ev_five"].status, ACTIVE)
                self.assertEqual(standing["ev_zero"].status, SUPERSEDED)
                self.assertEqual(standing["ev_none"].status, SUPERSEDED)
                self.assertEqual(standing["ev_none"].superseded_by, "ev_five")

        # And with no later record, 0 must outrank unsequenced in any order.
        for order in itertools.permutations([zero, unsequenced]):
            with self.subTest(pair=[r.evidence_id for r in order]):
                standing = evaluate_standing(list(order))
                self.assertEqual(standing["ev_zero"].status, ACTIVE)
                self.assertEqual(standing["ev_none"].status, SUPERSEDED)

    def test_a_record_without_an_id_does_not_raise(self) -> None:
        """`evaluate_standing` skips it; `active_records` must agree."""
        class _Anonymous:
            evidence_id = ""
            evidence_sequence = 1
            metadata: dict = {}

        good = _Record("ev_good", [(HANDLE, "d")], 2)
        kept = active_records([_Anonymous(), good])

        self.assertIn(good, kept)

    def test_tied_sort_keys_do_not_resolve_by_arrival_order(self) -> None:
        """Order-independence must hold at the margin too, or it is not a rule.

        Two records can share a sort key: both unsequenced, or holding the same
        number after a discard freed it, since the store numbers by max+1 over
        what it currently holds. `sorted` is stable, so without a tiebreak those
        resolve by arrival order and the verdict flips when the same records are
        read back differently -- which is the property this ordering exists to
        remove.
        """
        import itertools

        cases = {
            "both unsequenced": (
                _Record("ev_a", [(HANDLE, "d1")], None),
                _Record("ev_b", [(HANDLE, "d2")], None),
            ),
            "same sequence": (
                _Record("ev_c", [(HANDLE, "d1")], 7),
                _Record("ev_d", [(HANDLE, "d2")], 7),
            ),
        }
        for label, pair in cases.items():
            with self.subTest(case=label):
                verdicts = set()
                for order in itertools.permutations(pair):
                    standing = evaluate_standing(list(order))
                    verdicts.add(
                        tuple(sorted((k, v.status) for k, v in standing.items()))
                    )
                self.assertEqual(
                    len(verdicts), 1, f"{label}: verdict depends on arrival order"
                )

    def test_an_unreadable_sequence_is_treated_as_unsequenced(self) -> None:
        """A read-only audit must not raise on a record it cannot number.

        The store normalises a junk sequence away on load, but this module
        accepts anything shaped like a record, and demoting is the conservative
        answer: it may be superseded, and supersedes nothing numbered.
        """
        junk = _Record("ev_junk", [(HANDLE, "d1")], "not-a-number")
        numbered = _Record("ev_ok", [(HANDLE, "d2")], 5)

        standing = evaluate_standing([junk, numbered])

        self.assertEqual(standing["ev_junk"].status, SUPERSEDED)
        self.assertEqual(standing["ev_ok"].status, ACTIVE)

    def test_standing_never_inspects_content(self) -> None:
        """It reads handles and digests; it cannot know what a value means."""
        import inspect

        from orbit.runtime import evidence_authority

        source = inspect.getsource(evidence_authority.evaluate_standing)
        for banned in ("re.", "startswith", ".lower()", "in body", "raw_sha256"):
            self.assertNotIn(banned, source, f"{banned!r} suggests content matching")


class NoModelFacingTextChangedTests(unittest.TestCase):
    """This fix is mechanical. It must not touch what the model is told.

    Standing is decided from provenance the store already records, so the
    finalizer needs no new instruction to honour it: a superseded version
    simply never reaches the context. Pinning that here keeps a later
    "just one sentence" from turning an evidence-selection fix into a prompt
    change, which is a different kind of change with a different risk.
    """

    BASELINE = "0c9d8ba330dc67c7500cbd75c80940c222a3a573"
    NAMES = (
        "ANALYSIS_SYSTEM_PROMPT",
        "ANALYSIS_REPORT_INSTRUCTION",
        "NO_EVIDENCE_REPORT",
        "AUTONOMOUS_CONTINUATION_MESSAGE",
        "AUTONOMOUS_REPLAN_MESSAGE",
    )

    # One intentional, user-authorized revision to the analysis contract, not
    # accidental drift. Evidence-aware compaction replaces a large observation
    # in history with its canonical reference, so the model has to be told what
    # a reference is and how to read the exact bytes back -- without that
    # sentence the compaction silently hides evidence, which is worse than the
    # context ceiling it removes. The clause is pinned verbatim here so the
    # protection still fires on any OTHER prompt change, including a later edit
    # to this same clause.
    # Each authorized revision is pinned as (anchor, addition): the exact text
    # added, and the exact line it was added after. Position is part of the
    # pin, so relocating a sanctioned clause fails just as an unsanctioned one
    # does. Appending to this tuple is the only way to sanction a change, and
    # it is a reviewed, user-authorized act each time.
    AUTHORIZED_ADDITIONS = {"ANALYSIS_SYSTEM_PROMPT": (
        # ANALYSIS-COMPACTION-1: evidence references and exact rehydration.
        (
            '    "Perform at most one execute_analysis action per turn, '
            'then stop and report what it produced.\\n"\n',
            '    "Earlier results may appear as an evidence reference '
            '(`tool_evidence_ref: true`) "\n'
            '    "instead of their full text; that is the exact output, archived, '
            'not a summary. "\n'
            '    "When you need those exact bytes again, name its id as '
            '`evidence:<evidence_id>` "\n'
            '    "and they are restored verbatim. Never infer content from a '
            'reference alone.\\n"\n',
        ),
        # ANALYSIS-PROGRESS-1: prefer executing an identified deterministic
        # transformation over re-reading source already collected.
        (
            '    "and they are restored verbatim. Never infer content from a '
            'reference alone.\\n"\n',
            '    "When you have identified a deterministic transformation -- a '
            'decoder, "\n'
            '    "decompressor or decryption whose algorithm and concrete inputs '
            'you "\n'
            '    "already hold -- execute it and store its output before '
            're-reading source "\n'
            '    "you have already collected. Reading the same bytes again '
            'cannot resolve "\n'
            '    "what only running the transformation can.\\n"\n',
        ),
    ),
    # ANALYSIS-PROGRESS-1, authorized under the mission's clause 22. The live
    # run proved the runtime side worked -- 12 actions, 12 distinct programs,
    # nothing to suppress -- while the model still spent its last action on a
    # further read of source whose decoder inputs it had already extracted.
    # This is the instruction it reads before every autonomous step, so it is
    # where a priority between two legitimate next steps has to be stated. It
    # names no technique, and the generic-language test still enforces that.
    # ANALYSIS-IOC-1. A live report on a PowerShell downloader called a
    # single fetch "beaconing", called a written-and-run staging file
    # "persistence", read dead time arithmetic as evasion, and proposed
    # retrieving the remote payload as the next step of an offline analysis.
    # None of those were in the evidence. The clause states when a label is
    # earned and what "next step" means in an isolated session; it names no
    # artifact and no technique, and the generic-language test still applies.
    "ANALYSIS_REPORT_INSTRUCTION": (
        (
            '    "remains unresolved; and the single next step most worth taking."\n',
            '    "\\nName a behaviour only when the evidence shows it: repeated or "\n'
            '    "call-back contact before calling something beaconing, and a mechanism "\n'
            '    "for future or recurrent execution before calling something persistence "\n'
            '    "-- writing or copying a file is staging until something makes it run "\n'
            '    "again, unless where it is written is itself what runs it. Describe "\n'
            '    "timing and file deletion as what they do; call them "\n'
            '    "evasion or anti-forensic only where a purpose is evidenced. Prefer a "\n'
            '    "plain description to a technique label when intent is not established.\\n"\n'
            '    "This analysis is offline and isolated: the next step must be one that "\n'
            '    "can be taken here, on the artifact and the evidence. Retrieving a "\n'
            '    "remote resource is not that step, though it may be named as separately "\n'
            '    "authorised follow-up."\n',
        ),
    ),
    "AUTONOMOUS_CONTINUATION_MESSAGE": (
        (
            '    "findings."\n',
            '    " If you have already identified a deterministic transformation "\n'
            '    "and hold its concrete inputs, run it now rather than inspecting "\n'
            '    "the source further."\n',
        ),
    ),
    }

    # Model-facing constants introduced AFTER the baseline revision, which the
    # diff above cannot protect: there is nothing to compare them against. They
    # are pinned by digest instead, so the protection is the same in substance
    # -- the text cannot drift without a reviewed, user-authorized edit here.
    #
    # ANALYSIS-REPAIR-1: sent once after an execution that ran and raised. The
    # live run proved the model attempts the transformation, receives its own
    # source and a full traceback, and then abandons the attempt to resume
    # reading source. This is the one instruction that declines to change the
    # subject for a single call. It names no error class and no correction.
    POST_BASELINE_DIGESTS = {
        "AUTONOMOUS_REPAIR_MESSAGE": (
            "5358c64e548cf4ad2c22fc8ca2ffa252ee490ae21e33de616dc8be00a46b5c81"
        ),
    }

    def test_post_baseline_constants_are_pinned(self) -> None:
        """A constant younger than the baseline still cannot drift silently."""
        import hashlib

        from orbit.runtime import analysis_runtime

        for name, digest in self.POST_BASELINE_DIGESTS.items():
            with self.subTest(constant=name):
                value = getattr(analysis_runtime, name)
                self.assertEqual(
                    hashlib.sha256(value.encode()).hexdigest(),
                    digest,
                    f"{name} changed; model-facing text must not drift",
                )

    def _constant(self, text: str, name: str) -> str | None:
        import re

        block = re.search(rf"^{name} = \(.*?^\)", text, re.S | re.M)
        if block:
            return block.group(0)
        line = re.search(rf"^{name} = .*?$", text, re.M)
        return line.group(0) if line else None

    def test_no_model_facing_constant_changed(self) -> None:
        import subprocess

        result = subprocess.run(
            ["git", "show", f"{self.BASELINE}:src/orbit/runtime/analysis_runtime.py"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode != 0:
            self.skipTest("baseline revision unavailable")
        baseline = result.stdout
        current = (ROOT / "src/orbit/runtime/analysis_runtime.py").read_text()

        for name in self.NAMES:
            with self.subTest(constant=name):
                before = self._constant(baseline, name)
                self.assertIsNotNone(before, f"{name} not found in baseline")
                expected = before
                if name in self.AUTHORIZED_ADDITIONS:
                    # Baseline plus exactly the authorized clauses, each at its
                    # own anchor. Anything else -- a reworded clause, a second
                    # sentence, a sanctioned clause moved elsewhere, an
                    # unrelated edit -- still fails.
                    expected = before
                    for anchor, addition in self.AUTHORIZED_ADDITIONS[name]:
                        self.assertNotIn(
                            addition, expected,
                            "the authorized clause must not already be present",
                        )
                        self.assertIn(anchor, expected, "prompt anchor moved")
                        expected = expected.replace(anchor, anchor + addition, 1)
                self.assertEqual(
                    expected,
                    self._constant(current, name),
                    f"{name} changed beyond the authorized revision; model-facing "
                    "text must not drift",
                )

    def test_the_tool_schema_is_unchanged(self) -> None:
        """The schema is inside the prewarmed prefix; changing it moves it."""
        import hashlib
        import json

        from orbit.runtime.analysis_runtime import ANALYSIS_TOOL_SCHEMA

        self.assertEqual(
            hashlib.sha256(
                json.dumps(ANALYSIS_TOOL_SCHEMA, sort_keys=True).encode()
            ).hexdigest(),
            "57710e9ee2c19683cb74b854d5b6f0714fb4802ad1a51971e43cd7f6d080f2a4",
        )


class FinalizationIntegrationTests(unittest.TestCase):
    """The real runtime selector, through a real store, with real artifacts."""

    def _runtime(self, backend):
        import tempfile

        from orbit.runtime.analysis_runtime import (
            AnalysisRuntime, AnalysisWorkspace, acquire_analysis_source,
        )
        from orbit.runtime.evidence import EvidenceStore

        tmp = Path(tempfile.mkdtemp(prefix="orbit-authority-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        artifact = tmp / "input.txt"
        artifact.write_text("alpha\n", encoding="utf-8")
        ws = AnalysisWorkspace.create()
        self.addCleanup(ws.close)
        source = acquire_analysis_source(artifact, ws.source_root)
        store = EvidenceStore(root=tmp / "ev")
        built = AnalysisRuntime(
            backend=backend, source=source, evidence_store=store, workspace=ws
        )
        self.addCleanup(built.close)
        return built, store

    def _write(self, name: str, body: str) -> str:
        return (
            "import pathlib\n"
            f"pathlib.Path('/workspace/work/{name}').write_text({body!r})\n"
            f"print('wrote {name}:', {body!r}, end='')"
        )

    def test_a_rewritten_artifact_supersedes_its_earlier_version(self) -> None:
        from tests.test_analysis_runtime import ScriptedBackend, tool_response

        backend = ScriptedBackend(
            tool_response(self._write("report.txt", "value=WRONG")),
            tool_response(self._write("report.txt", "value=RIGHT")),
        )
        runtime, store = self._runtime(backend)
        runtime.step("write it")
        runtime.step("correct it")

        cited = runtime._reportable_records()
        bodies = " ".join(store.reattest_exact(r.evidence_id) or "" for r in cited)

        self.assertIn("RIGHT", bodies)
        self.assertNotIn("WRONG", bodies)

    def test_the_superseded_record_is_retained_and_reattestable(self) -> None:
        from tests.test_analysis_runtime import ScriptedBackend, tool_response

        backend = ScriptedBackend(
            tool_response(self._write("report.txt", "value=WRONG")),
            tool_response(self._write("report.txt", "value=RIGHT")),
        )
        runtime, store = self._runtime(backend)
        runtime.step("write it")
        runtime.step("correct it")

        superseded = runtime.superseded_records()

        self.assertEqual(len(superseded), 1)
        body = store.reattest_exact(superseded[0].evidence_id)
        self.assertIsNotNone(body, "history must stay verifiable")
        self.assertIn("WRONG", body)

    def test_finalization_stays_bounded(self) -> None:
        """Dropping superseded records must not raise the record ceiling."""
        from orbit.runtime.analysis_runtime import MAX_REPORT_EVIDENCE_RECORDS
        from tests.test_analysis_runtime import ScriptedBackend, tool_response

        n = MAX_REPORT_EVIDENCE_RECORDS + 4
        backend = ScriptedBackend(
            *[tool_response(f"print('step {i}', end='')") for i in range(n)]
        )
        runtime, _ = self._runtime(backend)
        for i in range(n):
            runtime.step(f"step {i}")

        self.assertLessEqual(len(runtime._reportable_records()), MAX_REPORT_EVIDENCE_RECORDS)

    def test_superseded_records_are_dropped_before_the_bound(self) -> None:
        """Order matters: filtering after the bound wastes places on history.

        With more actions than the ceiling, a superseded record inside the
        last-N window would push a current one out if the bound were applied
        first. Filtering first spends every place on evidence a report may
        actually cite.
        """
        from orbit.runtime.analysis_runtime import MAX_REPORT_EVIDENCE_RECORDS
        from tests.test_analysis_runtime import ScriptedBackend, tool_response

        # The rewrite happens LATE, so the superseded version sits inside the
        # last-N window. That is the only arrangement where the two orders
        # differ: filtering first keeps N citable records, filtering afterwards
        # spends one of the N places on history and returns N-1.
        script = [
            tool_response(f"print('step {i}', end='')")
            for i in range(MAX_REPORT_EVIDENCE_RECORDS - 1)
        ]
        script += [
            tool_response(self._write("report.txt", "v1")),
            tool_response(self._write("report.txt", "v2")),
        ]
        runtime, store = self._runtime(ScriptedBackend(*script))
        for i in range(len(script)):
            runtime.step(f"step {i}")

        cited = runtime._reportable_records()
        bodies = " ".join(store.reattest_exact(r.evidence_id) or "" for r in cited)

        self.assertLessEqual(len(cited), MAX_REPORT_EVIDENCE_RECORDS)
        # Filtering first fills every available place with citable evidence.
        self.assertEqual(len(cited), MAX_REPORT_EVIDENCE_RECORDS)
        self.assertEqual(len(runtime.superseded_records()), 1)
        # The superseded version is gone from the citable set...
        self.assertNotIn("report.txt: 'v1'", bodies)
        # ...and no superseded record occupies one of the bounded places.
        superseded_ids = {r.evidence_id for r in runtime.superseded_records()}
        self.assertFalse(superseded_ids & {r.evidence_id for r in cited})

    def test_superseded_records_tolerates_a_record_without_an_id(self) -> None:
        """The two views must agree about how defensive to be.

        `active_records` skips a record the evaluator never scored; if
        `superseded_records` indexed the same map directly it would raise on
        exactly the record its sibling accepts.
        """
        class _Anonymous:
            evidence_id = ""
            evidence_sequence = 1
            tool_name = ANALYSIS_TOOL_NAME
            metadata: dict = {}

        from tests.test_analysis_runtime import ScriptedBackend, tool_response

        runtime, store = self._runtime(
            ScriptedBackend(tool_response("print('a', end='')"))
        )
        runtime.step("look")
        store.records["anon"] = _Anonymous()

        self.assertIsInstance(runtime.superseded_records(), list)
        self.assertIsInstance(runtime._reportable_records(), list)

    def test_the_report_makes_exactly_one_model_call(self) -> None:
        from tests.test_analysis_runtime import ScriptedBackend, prose_response, tool_response

        backend = ScriptedBackend(
            tool_response("print('a', end='')"), prose_response("REPORT")
        )
        runtime, _ = self._runtime(backend)
        runtime.step("look")
        report = runtime.report()

        self.assertEqual(report.model_calls, 1)
        self.assertEqual(backend.calls, 2)


if __name__ == "__main__":
    unittest.main()
