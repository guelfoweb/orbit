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

from orbit.runtime.analysis_runtime import (
    ANALYSIS_TOOL_NAME,
    DETERMINISTIC_AUTHORITY_PREAMBLE,
    UNSUPPORTED_INDICATOR_FOOTER,
    UNSUPPORTED_INDICATOR_NOTICE,
    UNSUPPORTED_INLINE_MARK,
)
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
            # ANALYSIS-IOC-4. The operational list is the runtime's: a
            # report that wrote its own listed a corrupted host while the
            # canonical section two below carried the right one, and a reader
            # had no way to tell which was authoritative.
            # ANALYSIS-IOC-3. Two live PowerShell runs wrote a corrupted host
            # under Confirmed findings despite being given the exact value
            # nine times in the prompt. Detection cannot fix a transcription
            # failure, so the report no longer requires transcription: the
            # model cites a token and the runtime substitutes the value.
            '    "Never retype a network address: write the IOC- token the indicator list "\n'
            '    "gives you and the exact value is substituted. An address you type "\n'
            '    "yourself is unsupported even when you meant the right one.\\n"\n'
            # ANALYSIS-IOC-4. The operational list is the runtime's: a report
            # that wrote its own listed a corrupted host while the canonical
            # section below carried the right one, and a reader had no way to
            # tell which was authoritative.
            '    "Do not write an indicators list of your own: the runtime publishes that "\n'
            '    "section from the canonical records, and a second list can only "\n'
            '    "contradict it.\\n"\n'
            # ANALYSIS-IOC-5. A live report titled a section "Self-persisting
            # dropper behaviour" while its own body described write-run-delete
            # with no re-execution mechanism -- an intent label in a heading,
            # the same class as "Beaconing" for a single fetch. A heading is a
            # claim and the instruction now says so.
            '    "A section heading is a claim too: do not title a section beaconing, "\n'
            '    "persistence, evasion or anti-forensic unless the evidence shows the "\n'
            '    "mechanism, exactly as for a sentence. A heading its own body walks back "\n'
            '    "is worse than no heading.\\n"\n'
            # ANALYSIS-IOC-2. A live PowerShell report again headed a section
            # "Beaconing" for a single fetch-and-execute, having correctly
            # refused "persistence" and "anti-forensics" in the same report.
            # The bar was stated once, mid-sentence, beside two others; this
            # states the beaconing case as concretely as the staging one.
            '    "A payload fetched once and run once is retrieval and execution, not "\n'
            '    "beaconing.\\n"\n'
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

    # ANALYSIS-EVIDENCE-FIRST-1. Model-facing text that lives in a function
    # rather than a constant, so `_constant()` cannot see it and the diff
    # guard above would never fire on a reworded clause. Pinned by rendering
    # with fixed placeholders: the digest covers the wording, not the ids.
    #
    # The sentence that matters most is the last one -- it is what allows a
    # run to end at one model call and no actions -- and without this pin it
    # could be deleted with the whole suite still green.
    EVIDENCE_FIRST_DIGEST = (
        "e65ae0bc02b73ee1a532a1268c81c8f19fb03ecbf3a305b5f7895a27508ac85a"
    )

    def test_the_evidence_first_instruction_is_pinned(self) -> None:
        import hashlib

        from orbit.runtime.analysis_runtime import _evidence_first_instruction

        rendered = _evidence_first_instruction("<ANALYST>", ["<ID1>", "<ID2>"])
        self.assertEqual(
            hashlib.sha256(rendered.encode()).hexdigest(),
            self.EVIDENCE_FIRST_DIGEST,
            "_evidence_first_instruction changed; model-facing text must not drift",
        )

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


class _StubBackend:
    """Never called: these tests inspect the prompt, not a reply."""

    def chat_stream(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("no model call expected")


class _ReportingBackend:
    """Returns one fixed narrative, so the test controls what the report says."""

    def __init__(self, text: str) -> None:
        self.text = text

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                    on_delta, on_progress=None):
        from orbit.backend.base import ChatResult

        if on_delta is not None:
            on_delta(self.text)
        return ChatResult(
            content=self.text, model="stub", finish_reason="stop", tool_calls=[],
            prompt_tokens=1, completion_tokens=1, cached_tokens=0,
            prompt_tokens_per_second=None, generation_tokens_per_second=None,
        )


class DeterministicFactsReachTheReportTests(unittest.TestCase):
    """A value the runtime decoded exactly must reach the model that writes about it.

    Measured on a live run: the runtime had decoded the artifact's real C2 and
    three WMI strings, every one of them was excluded from the report prompt
    because the appendix renders them -- and the appendix is concatenated
    AFTER generation. The model wrote its narrative having never seen them,
    supplied an endpoint of its own, and cited the evidence ids whose exact
    contents said otherwise.
    """

    #: One PowerShell numeric-XOR stage, in the shape the deobfuscator
    #: recognises, carrying a URL chosen by the test. Nothing here is taken
    #: from any real sample: the point is that the runtime decodes it and the
    #: report has to agree with what came out.
    DECODED_URL = "http://example-c2.test/a.php?s=SYNTH"

    def _artifact_text(self, url: str) -> str:
        key = 34
        blob = ",".join(str(ord(c) ^ key) for c in url)
        return (
            "powershell -noprofile -WindowStyle hidden -C "
            f"\"$a='{blob}';$k={key};$out='';"
            "$parts=$a -split ',';foreach($p in $parts)"
            "{$out=$out+[char]([int]$p -bxor $k)};iex $out\"\n"
        )

    def _runtime(self, backend, url: str | None = None):
        import tempfile

        from orbit.runtime.analysis_runtime import (
            AnalysisRuntime, AnalysisWorkspace, acquire_analysis_source,
        )
        from orbit.runtime.evidence import EvidenceStore

        tmp = Path(tempfile.mkdtemp(prefix="orbit-grounding-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        artifact = tmp / "sample.js"
        artifact.write_text(self._artifact_text(url or self.DECODED_URL))
        ws = AnalysisWorkspace.create()
        self.addCleanup(ws.close)
        built = AnalysisRuntime(
            backend=backend,
            source=acquire_analysis_source(artifact, ws.source_root),
            evidence_store=EvidenceStore(root=tmp / "ev"),
            workspace=ws,
        )
        self.addCleanup(built.close)
        built._run_transform_preflight()
        return built

    def _with_one_action(self, backend, url: str | None = None):
        """A runtime that also holds one ACTION record, so report() takes the
        evidence-grounded path rather than the zero-call "no evidence" one."""
        runtime = self._runtime(backend, url)
        runtime.evidence_store.add(
            ANALYSIS_TOOL_NAME,
            "status: ok",
            metadata={
                "analysis_source_sha256": runtime.source.sha256,
                "tool_call_id": "call_1",
                "user_turn_id": "turn_1",
                "produced_by_phase": "analysis_action",
            },
        )
        return runtime

    def test_a_decoded_value_is_in_the_prompt_the_model_answers(self) -> None:
        """The failing boundary: what report() puts in front of the model."""
        runtime = self._runtime(_StubBackend())

        messages = runtime._report_messages(
            "What does this establish?", runtime._reportable_records()
        )
        prompt = " ".join(str(m.get("content", "")) for m in messages)

        self.assertIn(self.DECODED_URL, prompt)

    def test_the_decoded_value_is_marked_as_outranking_other_text(self) -> None:
        """It must be usable as authority, not read as one more opinion."""
        runtime = self._runtime(_StubBackend())

        messages = runtime._report_messages("", runtime._reportable_records())
        prompt = " ".join(str(m.get("content", "")) for m in messages)

        self.assertIn(DETERMINISTIC_AUTHORITY_PREAMBLE, prompt)
        self.assertLess(
            prompt.index(DETERMINISTIC_AUTHORITY_PREAMBLE),
            prompt.index("Evidence collected so far"),
            "the authoritative values must precede the citable cards",
        )

    def test_an_endpoint_the_artifact_never_contained_is_marked_unsupported(
        self,
    ) -> None:
        """The exact live failure: an invented address, cited as a finding."""
        invented = "http://198.51.100.7:4444/payload.ps1"
        runtime = self._with_one_action(
            _ReportingBackend(f"The dropper downloads from {invented} and runs it.")
        )

        report = runtime.report("what is it")

        self.assertIn(UNSUPPORTED_INDICATOR_NOTICE, report.text)
        self.assertIn(invented, report.text.split(UNSUPPORTED_INDICATOR_NOTICE)[1])

    def test_a_near_miss_is_answered_with_provenance_not_a_replacement(
        self,
    ) -> None:
        """Measured live: the model wrote `gibuyuy37v2v` for `gibuzuy37v2v`.

        The answer is what the records DO contain, not a guess at what was
        meant. Equal length and one differing character does not prove two
        hostnames are the same endpoint.
        """
        near = self.DECODED_URL.replace("example-c2", "examp1e-c2")
        self.assertEqual(len(near), len(self.DECODED_URL))
        runtime = self._with_one_action(
            _ReportingBackend(f"The script contacts {near} and runs the reply.")
        )

        report = runtime.report("what is it")

        self.assertIn(UNSUPPORTED_INDICATOR_NOTICE, report.text)
        self.assertIn("UNSUPPORTED: no evidence record contains this value",
                      report.text)
        # The recovered value is named, WITH the record that holds it...
        self.assertIn(self.DECODED_URL, report.text)
        # ...and the runtime does not decide which one was meant.
        self.assertIn("NOT established", report.text)
        self.assertNotIn("differs by ONE character", report.text)

    def test_the_warning_precedes_the_narrative_it_rejects(self) -> None:
        """A trailing notice cannot repair a confirmed claim already read."""
        near = self.DECODED_URL.replace("example-c2", "examp1e-c2")
        runtime = self._with_one_action(
            _ReportingBackend(
                "## Confirmed findings\n\n" + f"The C2 is {near}. " * 5
            )
        )

        report = runtime.report("what is it")

        self.assertTrue(
            report.text.startswith(UNSUPPORTED_INDICATOR_NOTICE),
            "the rejection must lead the report, not trail it",
        )
        # The model's own words are still quoted, because repudiating a claim
        # requires showing it.
        self.assertIn(near, report.text)
        self.assertIn(near, report.model_text)
        self.assertFalse(report.model_text.startswith(UNSUPPORTED_INDICATOR_NOTICE))

    def test_two_distinct_recovered_endpoints_are_both_named(self) -> None:
        """Two real endpoints one character apart are not each other."""
        from orbit.runtime.analysis_runtime import _unsupported_line

        provenance = {
            "http://a1.example.test/x": ["ev_aaa"],
            "http://a2.example.test/x": ["ev_bbb"],
        }
        line = _unsupported_line("http://a3.example.test/x", provenance)

        self.assertIn("ev_aaa", line)
        self.assertIn("ev_bbb", line)
        self.assertIn("NOT established", line)

    def test_an_unsupported_endpoint_is_marked_where_it_is_used(self) -> None:
        """A leading warning does not stop a later sentence confirming it.

        Measured on two independent live PowerShell runs: given the correct
        host nine times in its prompt, the model wrote a one-token-shorter
        form under "Confirmed findings" -- "The URI is confirmed" beside a
        value no record holds, citing a real evidence id.
        """
        wrong = self.DECODED_URL.replace("example-c2", "examp1e-c2")
        runtime = self._with_one_action(_ReportingBackend(
            f"## Confirmed findings\n\n**The URI is confirmed.** {wrong} appears "
            f"in the artifact.\n\n## Artifacts produced\n\n- writes {wrong}"
        ))

        report = runtime.report("what is it")

        body = report.text.split(UNSUPPORTED_INDICATOR_FOOTER, 1)[1]
        # No occurrence is left standing unqualified.
        for piece in body.split(wrong)[1:]:
            self.assertTrue(
                piece.startswith(UNSUPPORTED_INLINE_MARK),
                "an unsupported endpoint was left unmarked at the point of use",
            )

    def test_marking_never_rewrites_the_model_text(self) -> None:
        """Removing only the marks must return the model's words exactly."""
        wrong = self.DECODED_URL.replace("example-c2", "examp1e-c2")
        narrative = (
            f"## Confirmed findings\n\nIt contacts {wrong} and also names "
            f"the bare host {wrong.split('//')[1].split('/')[0]} later."
        )
        runtime = self._with_one_action(_ReportingBackend(narrative))

        report = runtime.report("what is it")

        body = report.text.split(UNSUPPORTED_INDICATOR_FOOTER, 1)[1].lstrip("\n")
        body = body.split("\n\n## Verified indicators")[0]
        self.assertEqual(body.replace(UNSUPPORTED_INLINE_MARK, ""), narrative)
        # And the untouched original is kept for audit.
        self.assertEqual(report.model_text, narrative)

    def test_a_bare_authority_beside_the_uri_is_marked_too(self) -> None:
        """A narrative names an endpoint both ways; both must be qualified.

        KNOWN LIMIT, stated rather than implied: detection is by URI, so a
        report that named ONLY a bare hostname and never a URI would raise
        nothing to mark. That case is not covered here and is not claimed to
        be -- on both live PowerShell runs the bare host appeared only
        alongside the wrong URI, which is what this marks.
        """
        wrong = self.DECODED_URL.replace("example-c2", "examp1e-c2")
        host = wrong.split("//")[1].split("/")[0]
        runtime = self._with_one_action(_ReportingBackend(
            f"## Confirmed findings\n\nIt contacts {wrong}.\n"
            f"The host is {host}, seen again."
        ))

        report = runtime.report("what is it")

        body = report.text.split(UNSUPPORTED_INDICATOR_FOOTER, 1)[1]
        # The URI keeps its mark, and the bare host gets one of its own.
        self.assertIn(wrong + UNSUPPORTED_INLINE_MARK, body)
        self.assertIn(host + UNSUPPORTED_INLINE_MARK + ", seen again", body)

    def test_a_canonical_reference_resolves_to_the_exact_value(self) -> None:
        """The model cites a token; the runtime writes the address.

        This is the half that removes the failure instead of detecting it.
        Two live PowerShell runs wrote a corrupted host under Confirmed
        findings while the prompt held the exact value nine times, so the
        report no longer asks the model to transcribe an address at all.
        """
        from orbit.runtime.analysis_runtime import _indicator_token

        runtime = self._with_one_action(_StubBackend())
        indicators = runtime.canonical_indicators()
        self.assertTrue(indicators, "the fixture must recover one indicator")
        token = _indicator_token(indicators[0])

        runtime = self._with_one_action(_ReportingBackend(
            f"## Confirmed findings\n\nIt contacts {token} once."
        ))
        report = runtime.report("what is it")

        body = report.text.split("## Verified indicators")[0]
        self.assertIn(self.DECODED_URL, body)
        self.assertNotIn(token, body, "the token must not survive into the report")
        # Nothing is flagged: a resolved reference is a recovered value.
        self.assertNotIn(UNSUPPORTED_INDICATOR_NOTICE, report.text)
        # And the model's own words are kept for audit.
        self.assertIn(token, report.model_text)

    def test_an_unknown_reference_resolves_to_nothing(self) -> None:
        """No nearest match, no guess, no false confirmation."""
        from orbit.runtime.analysis_runtime import UNRESOLVED_REFERENCE_MARK

        runtime = self._with_one_action(_ReportingBackend(
            "## Confirmed findings\n\nIt contacts IOC-deadbeef once."
        ))

        report = runtime.report("what is it")

        self.assertIn(UNRESOLVED_REFERENCE_MARK, report.text)
        self.assertNotIn(self.DECODED_URL,
                         report.text.split("## Verified indicators")[0])

    def test_the_reference_table_reaches_the_prompt(self) -> None:
        """A token the model never saw cannot be cited."""
        from orbit.runtime.analysis_runtime import _indicator_token

        runtime = self._with_one_action(_StubBackend())
        token = _indicator_token(runtime.canonical_indicators()[0])
        runtime = self._with_one_action(_ReportingBackend("ok"))

        messages = runtime._report_messages("x", runtime._reportable_records())
        prompt = " ".join(str(m.get("content", "")) for m in messages)

        self.assertIn(token, prompt)
        self.assertIn(self.DECODED_URL, prompt)

    def test_a_token_identifies_one_indicator_not_a_record(self) -> None:
        """Digest-derived, so several indicators in one record stay distinct."""
        from orbit.runtime.analysis_runtime import _indicator_token
        from orbit.runtime.analysis_indicators import Indicator

        one = Indicator(kind="uri", value="http://a.test/x", evidence_id="ev_1",
                        source="s", line=1, sha256="a" * 64)
        two = Indicator(kind="uri", value="http://b.test/x", evidence_id="ev_1",
                        source="s", line=1, sha256="b" * 64)

        self.assertNotEqual(_indicator_token(one), _indicator_token(two))

    def test_the_model_cannot_publish_its_own_indicator_list(self) -> None:
        """One operational section, and the runtime owns it.

        Measured live: a report listed a corrupted host under its own
        "Indicators" while the canonical section two below carried the right
        one. A reader has no way to tell which is authoritative, so the
        model's list does not reach the document -- for every run, whether
        the value it would have written was right or wrong.
        """
        wrong = self.DECODED_URL.replace("example-c2", "examp1e-c2")
        runtime = self._with_one_action(_ReportingBackend(
            f"## Confirmed findings\n\nIt runs a payload.\n\n"
            f"## Indicators\n\n- URI: {wrong}\n\n"
            f"## Behaviour established\n\nIt fetches once."
        ))

        report = runtime.report("what is it")

        # The model's list is gone from the BODY; the canonical section
        # remains. The fabricated value still appears in the notice, because
        # deleting the section must not delete the evidence that the report
        # invented an address -- a reader shown a clean document would never
        # learn it happened.
        body = report.text.split(UNSUPPORTED_INDICATOR_FOOTER, 1)[1]
        self.assertNotIn(wrong, body)
        self.assertIn(wrong, report.text.split(UNSUPPORTED_INDICATOR_FOOTER, 1)[0])
        self.assertIn(self.DECODED_URL, report.text)
        self.assertIn("Verified indicators", report.text)
        # The surrounding analysis is untouched.
        self.assertIn("It runs a payload.", report.text)
        self.assertIn("It fetches once.", report.text)
        # And the original is kept for audit.
        self.assertIn(wrong, report.model_text)

    def test_a_correct_indicator_list_is_dropped_too(self) -> None:
        """The rule is the contract, not a per-run judgement of correctness."""
        runtime = self._with_one_action(_ReportingBackend(
            f"## Confirmed findings\n\nIt runs a payload.\n\n"
            f"## Indicators\n\n- URI: {self.DECODED_URL}\n"
        ))

        report = runtime.report("what is it")

        published = report.text.split("## Indicators", 1)[1].split("##", 1)[0]
        self.assertNotIn("- URI:", published,
                         "the model's list must not survive, right or wrong")
        self.assertIn("Verified indicators", report.text)

    def test_dropping_the_list_does_not_excuse_prose(self) -> None:
        """Moving a false claim into a sentence must not escape the check."""
        wrong = self.DECODED_URL.replace("example-c2", "examp1e-c2")
        runtime = self._with_one_action(_ReportingBackend(
            f"## Confirmed findings\n\nThe downloader contacts {wrong}."
        ))

        report = runtime.report("what is it")

        self.assertTrue(report.text.startswith(UNSUPPORTED_INDICATOR_NOTICE))
        self.assertIn(wrong + UNSUPPORTED_INLINE_MARK, report.text)

    def test_a_fabricated_query_on_a_real_host_is_flagged(self) -> None:
        """A superset of a recovered URI is a different endpoint.

        Containment ran both ways, so appending invented exfiltration
        parameters to a genuine host raised nothing: no notice, no inline
        mark, and the scorers used the same rule so they missed it too.
        """
        from orbit.runtime.analysis_runtime import _unsupported_indicators

        known = {self.DECODED_URL}
        invented = f"{self.DECODED_URL}&cmd=exfil&target=192.168.1.1"

        self.assertEqual(
            _unsupported_indicators(f"It posts to {invented}.", known),
            [invented],
        )
        # Quoting LESS than the record holds is still the recovered endpoint.
        shorter = self.DECODED_URL.split("?")[0]
        self.assertEqual(_unsupported_indicators(f"It hits {shorter}.", known), [])

    def test_a_heading_inside_a_code_fence_is_not_a_section(self) -> None:
        """Dropping one ate the quoted source and the analysis after it."""
        from orbit.runtime.analysis_runtime import _drop_runtime_owned_sections

        text = (
            "## Confirmed findings\n\nThe script contains:\n\n```powershell\n"
            "# Indicators\n$u = 'http://quoted.test/x'\n```\n\n"
            "That is retrieval, not beaconing.\n\n## Behaviour\n\nfetches once.\n"
        )

        out, dropped = _drop_runtime_owned_sections(text)

        self.assertEqual(dropped, [])
        self.assertIn("That is retrieval, not beaconing.", out)
        self.assertIn("http://quoted.test/x", out)
        self.assertEqual(out.count("```"), 2, "the closing fence must survive")

    def test_a_fabrication_only_in_the_dropped_list_is_still_reported(
        self,
    ) -> None:
        """Removing the section must not remove the evidence it was wrong.

        A fabricated endpoint that appears ONLY under the model's own
        indicators heading was deleted before the consistency check ran: no
        notice, no mark, nothing. The reader was shown a clean document about
        a report that had invented an address.
        """
        wrong = self.DECODED_URL.replace("example-c2", "examp1e-c2")
        runtime = self._with_one_action(_ReportingBackend(
            f"## Confirmed findings\n\nIt runs a payload.\n\n"
            f"## Indicators\n\n- URI: {wrong}\n"
        ))

        report = runtime.report("what is it")

        self.assertTrue(report.text.startswith(UNSUPPORTED_INDICATOR_NOTICE))
        notice = report.text.split(UNSUPPORTED_INDICATOR_FOOTER, 1)[0]
        self.assertIn(wrong, notice)

    def _covered_runtime(self, backend):
        """A runtime on the covered path: source supplied, no action evidence.

        `report()` routes here when there is nothing to cite but the model
        was given the whole artifact. The path was entirely untested, so
        removing any of its three layers passed the suite.
        """
        runtime = self._runtime(backend)
        runtime.messages.append(
            {"role": "user", "content": "artifact", "source_covered": True}
        )
        runtime.messages.append({"role": "assistant", "content": "seen"})
        self.assertTrue(runtime.source_covered)
        self.assertEqual(runtime._reportable_records(), [])
        return runtime

    def test_the_covered_path_drops_the_model_list_too(self) -> None:
        wrong = self.DECODED_URL.replace("example-c2", "examp1e-c2")
        runtime = self._covered_runtime(_ReportingBackend(
            f"## Confirmed findings\n\nx\n\n## Indicators\n\n- URI: {wrong}\n"
        ))

        report = runtime.report("what is it")

        body = report.text.split(UNSUPPORTED_INDICATOR_FOOTER, 1)[1] \
            if UNSUPPORTED_INDICATOR_FOOTER in report.text else report.text
        self.assertNotIn(wrong, body)

    def test_the_covered_path_resolves_references(self) -> None:
        from orbit.runtime.analysis_runtime import _indicator_token

        probe = self._covered_runtime(_StubBackend())
        token = _indicator_token(probe.canonical_indicators()[0])
        runtime = self._covered_runtime(_ReportingBackend(
            f"## Confirmed findings\n\nIt contacts {token} once."
        ))

        report = runtime.report("what is it")

        self.assertIn(self.DECODED_URL, report.text)
        self.assertNotIn(token, report.text)

    def test_the_covered_path_keeps_the_original_for_audit(self) -> None:
        narrative = "## Confirmed findings\n\nIt runs a payload.\n"
        runtime = self._covered_runtime(_ReportingBackend(narrative))

        report = runtime.report("what is it")

        self.assertEqual(report.model_text.rstrip(), narrative.rstrip())

    def test_a_fence_line_must_be_only_a_fence(self) -> None:
        """An inline span at line start is not an opening fence.

        Matching any line beginning with the marker counted ```short``` as a
        fence, inverted the parity of everything after it, and put a real
        fenced block outside a fence -- deleting the analysis the guard exists
        to protect.
        """
        from orbit.runtime.analysis_runtime import _drop_runtime_owned_sections

        text = (
            "```short``` is the marker.\n\n## Confirmed findings\n\nSource:\n\n"
            "```powershell\n## Indicators\n$u='http://quoted.test/x'\n```\n\n"
            "Critical analysis that must survive.\n"
        )

        out, dropped = _drop_runtime_owned_sections(text)

        self.assertEqual(dropped, [])
        self.assertIn("Critical analysis that must survive.", out)
        self.assertIn("http://quoted.test/x", out)

    def test_an_unterminated_fence_runs_to_the_end(self) -> None:
        """A block the model never closed is still quoted text, not sections."""
        from orbit.runtime.analysis_runtime import _drop_runtime_owned_sections

        text = (
            "## Confirmed findings\n\nSource:\n\n```powershell\n"
            "## Indicators\n$u='http://quoted.test/x'\n"
        )

        out, dropped = _drop_runtime_owned_sections(text)

        self.assertEqual(dropped, [])
        self.assertIn("http://quoted.test/x", out)

    def test_an_indented_fence_is_still_a_fence(self) -> None:
        """Up to three spaces of indent, per CommonMark -- lists nest code."""
        from orbit.runtime.analysis_runtime import _drop_runtime_owned_sections

        # The fence is indented; the heading inside it is not, so only fence
        # recognition decides whether the block is quoted text or sections.
        text = (
            "## Confirmed findings\n\n- the script contains:\n\n   ```\n"
            "## Indicators\n$u='http://quoted.test/x'\n   ```\n\nkeep me.\n"
        )

        out, dropped = _drop_runtime_owned_sections(text)

        self.assertEqual(dropped, [])
        self.assertIn("keep me.", out)
        self.assertIn("http://quoted.test/x", out)

    def test_a_backtick_in_a_code_info_string_still_opens_a_fence(self) -> None:
        """```python` must protect its block, not delete it.

        Rejecting any backtick in the info string dropped the quoted section
        and inverted the parity of everything after it -- the exact class the
        fence guard exists to prevent, back one backtick away.
        """
        from orbit.runtime.analysis_runtime import _drop_runtime_owned_sections

        text = (
            "## Confirmed findings\n\n```python`\n## Indicators\n"
            "$u='http://quoted.test/x'\n```\n\n## Conclusion\n\nkeep me.\n"
        )

        out, dropped = _drop_runtime_owned_sections(text)

        self.assertEqual(dropped, [])
        self.assertIn("keep me.", out)
        self.assertIn("http://quoted.test/x", out)

    def test_a_longer_fence_is_not_closed_by_a_shorter_run(self) -> None:
        """A ``` inside a ```` block is content, not a terminator."""
        from orbit.runtime.analysis_runtime import _drop_runtime_owned_sections

        text = (
            "## Confirmed findings\n\n````\n## Indicators\n```\n"
            "data='http://quoted.test/x'\n````\n\nkeep me.\n"
        )

        out, dropped = _drop_runtime_owned_sections(text)

        self.assertEqual(dropped, [])
        self.assertIn("keep me.", out)
        self.assertIn("http://quoted.test/x", out)

    def test_a_tilde_inline_span_is_not_a_fence(self) -> None:
        """~~~x~~~ at line start opens and closes on one line."""
        from orbit.runtime.analysis_runtime import _drop_runtime_owned_sections

        text = (
            "~~~x~~~ is the marker.\n\n## Confirmed findings\n\n```\n"
            "## Indicators\nu='http://quoted.test/x'\n```\n\nkeep me.\n"
        )

        out, dropped = _drop_runtime_owned_sections(text)

        self.assertEqual(dropped, [])
        self.assertIn("keep me.", out)
        self.assertIn("http://quoted.test/x", out)

    def test_a_fence_closes_only_on_its_own_marker(self) -> None:
        """``` does not close ~~~, so the block stays a block."""
        from orbit.runtime.analysis_runtime import _drop_runtime_owned_sections

        text = (
            "## Confirmed findings\n\n~~~\n## Indicators\nx\n~~~\n\nkeep me.\n"
        )

        out, dropped = _drop_runtime_owned_sections(text)

        self.assertEqual(dropped, [])
        self.assertIn("keep me.", out)

    def test_only_headings_that_are_the_list_are_dropped(self) -> None:
        """The heading must READ as the operational list, not merely mention it.

        An exact-title list let "Indicators of compromise" through. Matching
        on words alone then went too far and removed "No indicators found" --
        a statement, not a list -- and "Indicators and next steps", which
        carries analysis this contract has no business deleting.
        """
        from orbit.runtime.analysis_runtime import _drop_runtime_owned_sections

        # A NAMED list must not bypass the contract: "Key indicators",
        # "Network IOCs", "Verified indicators" (the runtime's own section
        # name, when the model writes one of its own) all read as the list.
        owned = ("Indicators", "Indicator", "IOC", "IOCs",
                 "Indicators of compromise", "Network indicators",
                 "Indicators (network)", "Key indicators", "Network IOCs",
                 "Verified indicators", "Malicious indicators",
                 "C2 indicators", "Command-and-control indicators",
                 "Observed IOCs", "Primary indicators", "Host indicators")
        # A heading that MENTIONS the word but is analysis keeps its body.
        kept = ("Behaviour indicators", "No indicators found",
                "Indicators and next steps", "Confirmed findings",
                "Summary", "What remains unresolved",
                "How the indicators were derived",
                "Why no indicator was recovered",
                "Limitations of indicator extraction", "Indicator analysis")

        for heading in owned:
            with self.subTest(heading=heading, expect="dropped"):
                text = (f"## Confirmed findings\n\nx\n\n## {heading}\n\n"
                        "- URI: http://bad.test/y\n\n## Behaviour\n\ny\n")
                out, dropped = _drop_runtime_owned_sections(text)
                self.assertEqual(dropped, [heading])
                self.assertNotIn("http://bad.test/y", out)
                self.assertIn("## Behaviour", out)

        for heading in kept:
            with self.subTest(heading=heading, expect="kept"):
                text = (f"## Confirmed findings\n\nx\n\n## {heading}\n\n"
                        "- the analysis says this\n\n## Behaviour\n\ny\n")
                out, dropped = _drop_runtime_owned_sections(text)
                self.assertEqual(dropped, [])
                self.assertIn("the analysis says this", out)

    def test_a_setext_indicator_heading_is_dropped(self) -> None:
        """A setext heading is a heading; the model must not bypass by syntax.

        `Indicators` underlined by `===` or `---` is the operational list in a
        different Markdown form, and it was not recognised at all.
        """
        from orbit.runtime.analysis_runtime import _drop_runtime_owned_sections

        for rule in ("=" * 10, "-" * 10):
            with self.subTest(rule=rule[0]):
                text = (
                    f"Confirmed findings\n{'=' * 18}\n\nx\n\n"
                    f"Indicators\n{rule}\n\n- URI: http://bad.test/y\n\n"
                    f"Behaviour\n{'=' * 9}\n\ny\n"
                )
                out, dropped = _drop_runtime_owned_sections(text)
                self.assertEqual(dropped, ["Indicators"])
                self.assertNotIn("http://bad.test/y", out)
                self.assertIn("Behaviour", out)

    def test_a_thematic_break_is_not_a_heading(self) -> None:
        """A bare `---` rule with no title over it is not a section."""
        from orbit.runtime.analysis_runtime import _drop_runtime_owned_sections

        text = ("## Confirmed findings\n\nsome analysis\n\n---\n\n"
                "more analysis mentioning http://kept.test/y\n")

        out, dropped = _drop_runtime_owned_sections(text)

        self.assertEqual(dropped, [])
        self.assertIn("http://kept.test/y", out)

    def test_a_fabrication_only_in_a_named_list_is_still_reported(self) -> None:
        """Detection runs before the drop, for a bypassing heading too."""
        wrong = self.DECODED_URL.replace("example-c2", "examp1e-c2")
        runtime = self._with_one_action(_ReportingBackend(
            f"## Confirmed findings\n\nIt runs.\n\n## Key indicators\n\n"
            f"- URI: {wrong}\n"
        ))

        report = runtime.report("what is it")

        self.assertTrue(report.text.startswith(UNSUPPORTED_INDICATOR_NOTICE))
        notice = report.text.split(UNSUPPORTED_INDICATOR_FOOTER, 1)[0]
        self.assertIn(wrong, notice)  # kept in the error record
        body = report.text.split(UNSUPPORTED_INDICATOR_FOOTER, 1)[1]
        self.assertNotIn(wrong, body)  # dropped from the published body
        self.assertIn(wrong, report.model_text)  # original preserved

    def test_the_original_text_is_persisted_for_audit(self) -> None:
        """The notice says the original is kept; it has to actually be kept.

        A dropped section exists nowhere else, so without this the audit
        trail the document promises did not exist on disk.
        """
        import json
        import os
        import pathlib
        import tempfile

        retain = pathlib.Path(tempfile.mkdtemp(prefix="orbit-retain-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(retain, ignore_errors=True))
        previous = os.environ.get("ORBIT_ANALYSIS_RETAIN_DIR")
        os.environ["ORBIT_ANALYSIS_RETAIN_DIR"] = str(retain)
        self.addCleanup(
            lambda: os.environ.__setitem__("ORBIT_ANALYSIS_RETAIN_DIR", previous)
            if previous is not None
            else os.environ.pop("ORBIT_ANALYSIS_RETAIN_DIR", None)
        )

        runtime = self._with_one_action(_ReportingBackend(
            "## Confirmed findings\n\nx\n\n## Indicators\n\n- URI: http://bad.test/y\n"
        ))
        report = runtime.report("what is it")

        rows = [
            json.loads(line)
            for line in (retain / "report_admission.jsonl").read_text().splitlines()
            if line.strip()
        ]
        persisted = [row for row in rows if row.get("model_text")]
        self.assertTrue(persisted, "the model's text reached no diagnostic file")
        self.assertIn("http://bad.test/y", persisted[-1]["model_text"])
        # The published BODY is clean; the notice still reports what the
        # model claimed, so the removal is visible rather than silent.
        body = report.text.split(UNSUPPORTED_INDICATOR_FOOTER, 1)[1]
        self.assertNotIn("http://bad.test/y", body)

    def test_the_real_decoded_endpoint_is_not_flagged(self) -> None:
        """The check must not cry wolf over the value the runtime itself decoded."""
        runtime = self._with_one_action(
            _ReportingBackend(f"It contacts {self.DECODED_URL} once.")
        )

        report = runtime.report("what is it")

        self.assertNotIn(UNSUPPORTED_INDICATOR_NOTICE, report.text)

    def test_many_quotable_records_still_fit_the_report_prompt(self) -> None:
        """Each record under the per-record bound, the sum over it.

        Measured live: five action records of ~3,092 chars are each under
        MAX_REPORT_EVIDENCE_QUOTE_CHARS and together rendered a 19,225-char
        user message that admission refused outright -- so a run that had
        done its work produced an appendix and no narrative.
        """
        # A tokenising backend, because that is the production path and the
        # one the live failure exercised; the char fallback is deliberately
        # looser and would not demote at this size.
        class _Tokenising:
            def count_text_tokens(self, text: str):
                from types import SimpleNamespace

                # ~2.0 chars/token. Measured against the real tokeniser, a
                # card is a mix: obfuscated payload runs at 1.17 but the
                # surrounding metadata is structured text and tokenises far
                # better, so a whole `final_card` of 1,654 chars costs 812
                # tokens rather than the 1,413 a flat payload density predicts.
                return SimpleNamespace(
                    tokens=max(1, int(len(text) / 2.0)), context_tokens=8192
                )

            def chat_stream(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("no model call expected")

        runtime = self._runtime(_Tokenising())
        for index, size in enumerate((3070, 3092, 1804, 3092, 3092), start=1):
            runtime.evidence_store.add(
                ANALYSIS_TOOL_NAME, "x" * size,
                metadata={
                    "analysis_source_sha256": runtime.source.sha256,
                    "tool_call_id": f"call_{index}",
                    "user_turn_id": f"turn_{index}",
                    "produced_by_phase": "analysis_action",
                },
            )

        records = runtime._reportable_records()
        messages = runtime._report_messages("report it", records)
        user = messages[1]["content"]

        self.assertEqual(len(records), 5, "every record is still carried")
        # The stub backend cannot tokenise, so the char fallback applies.
        # 19,225 was the live failure; a demoted record still costs a citation
        # with size, digest and raw_ref, so the floor is the number of records
        # rather than zero.
        # 19,225 was the live failure. The char fallback is looser than the
        # token path by design -- without a tokeniser there is nothing to be
        # precise with -- so this asserts the bound binds, not that it is tight.
        self.assertLess(
            len(user), 19000,
            f"report prompt is {len(user)} chars; the aggregate budget did not bind",
        )
        cards = runtime._evidence_cards(records)
        self.assertGreaterEqual(
            sum(1 for card in cards if "raw_ref" in card), 1,
            "the budget must demote older records, not quote them all",
        )
        # How MANY survive depends on the window, and this fixture's is
        # narrower than the live one -- the replayed failures carried all five.
        # What must hold at any size is that the newest is never the one
        # dropped, since a report cites what it just learned.
        self.assertGreaterEqual(len(cards), 1)
        self.assertIn(
            records[-1].evidence_id, "\n".join(cards),
            "the newest record must always be carried",
        )
        # Demotion must not cost the facts the report has to state exactly.
        self.assertIn(self.DECODED_URL, user)

    def test_the_budget_is_capped_by_what_the_context_actually_leaves(
        self,
    ) -> None:
        """A fixed evidence budget bounds the evidence, not the prompt.

        Measured live: five records came to 6,364 tokens -- inside an 8,192
        context -- and admission refused anyway, because it reserves
        generation space the evidence budget never saw. Over by 220 tokens,
        which no constant chosen in advance would catch for every artifact.
        """

        class _Tokenising:
            """A backend that tokenises and reports a small window."""

            context = 2000

            def count_text_tokens(self, text: str):
                from types import SimpleNamespace

                return SimpleNamespace(
                    tokens=max(1, len(text) // 4), context_tokens=self.context
                )

            def chat_stream(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("no model call expected")

        runtime = self._runtime(_Tokenising())
        for index in range(1, 6):
            runtime.evidence_store.add(
                ANALYSIS_TOOL_NAME, "z" * 3000,
                metadata={
                    "analysis_source_sha256": runtime.source.sha256,
                    "tool_call_id": f"call_{index}",
                    "user_turn_id": f"turn_{index}",
                    "produced_by_phase": "analysis_action",
                },
            )

        records = runtime._reportable_records()
        cards = runtime._evidence_cards(records)

        # The window leaves almost nothing, so nothing is quoted whole and
        # the prompt is bounded by dropping the oldest rather than overflowing.
        self.assertTrue(
            all("raw_ref" in card for card in cards),
            "a context this small must demote every record it carries",
        )
        self.assertLess(
            len(cards), len(records),
            "a context this small must also drop what it cannot even cite",
        )
        # What IS carried is the newest, and it is carried by id.
        joined = "\n".join(cards)
        for card in cards:
            self.assertIn("raw_ref", card)
        self.assertIn(records[-1].evidence_id, joined,
                      "the newest record must survive")

    def test_a_demoted_record_is_cited_not_dropped(self) -> None:
        """Over budget means carried as a citation, never silently removed."""
        runtime = self._runtime(_StubBackend())
        for index in range(1, 7):
            runtime.evidence_store.add(
                ANALYSIS_TOOL_NAME, f"RECORD{index}-" + "y" * 3000,
                metadata={
                    "analysis_source_sha256": runtime.source.sha256,
                    "tool_call_id": f"call_{index}",
                    "user_turn_id": f"turn_{index}",
                    "produced_by_phase": "analysis_action",
                },
            )

        records = runtime._reportable_records()
        cards = "\n".join(runtime._evidence_cards(records))

        # Every record appears by id, whether quoted whole or cited: the
        # budget is spent on citations first, so nothing is dropped while the
        # floor still fits.
        for record in records:
            self.assertIn(record.evidence_id, cards, record.evidence_id)
        # The newest keeps its text; older ones are demoted to citations.
        self.assertIn("RECORD6", cards)

    def test_a_model_summary_is_labelled_as_unverified_not_as_evidence(self) -> None:
        """A FINISH summary is interpretation; the dossier must not promote it."""
        from orbit.runtime.analysis_controller import AnalysisController

        controller = AnalysisController()
        controller.adopt_plan([
            {"question": "what does it contact?", "missing_fact": "needs execution"}
        ])
        controller.activate_next()
        controller.close_active(
            "resolved",
            summary="It contacts http://wrong.invalid/x.",
            evidence_ids=("ev_abc123",),
        )

        dossier = controller.dossier()

        self.assertIn("unverified", dossier.lower())
        self.assertIn("cited by that claim", dossier)


if __name__ == "__main__":
    unittest.main()
