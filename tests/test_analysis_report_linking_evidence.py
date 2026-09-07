"""A report card preserves a link an action established past the head cut.

The bug: an action whose output exceeds `MAX_EVIDENCE_CHARS` is recorded as a
head-only observation for the step that saw it, plus a full untruncated raw
sibling in the store. `_reportable_records` carries only the observation, and
`_evidence_card` quoted it whole -- so a report built from it saw only the head.
When an artifact does its defining work past the head (a downloader's
fetch-and-run, a launcher's `Win32_Process.Create(...)` spawn), the call sites
that link its decoded strings into that action were absent from the report
prompt entirely, and the model could not state a link the store held.

The fix quotes a head-AND-tail window of the FULL raw output for a record whose
observation was cut, so the closing behaviour survives. These tests pin
delivery and the invariants around it; report SYNTHESIS correctness is verified
on real report generations, not here.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime.analysis_runtime import (  # noqa: E402
    MAX_EVIDENCE_CHARS,
    MAX_REPORT_EVIDENCE_QUOTE_CHARS,
    REPORT_LINKING_EXCERPT_HEAD_CHARS,
    REPORT_LINKING_EXCERPT_TAIL_CHARS,
    AnalysisRuntime,
    _head_and_tail_excerpt,
    acquire_analysis_source,
)
from orbit.runtime.analysis_sandbox import AnalysisResult  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

# The "link" is not one token but a SEQUENCE of call sites: the setup that
# makes an object, the configuration applied to it, and the call that finally
# uses it. Reproducing that shape matters, because a link is only recovered if
# the whole sequence is in the window -- a tail sized for the last call alone
# loses the setup that gives it meaning.
#
# Spaced to the live Fattura measurements, as a fraction of the full output:
# .Get(<class>) at 77%, SpawnInstance_() at 80%, ShowWindow at 80.5%, and
# .Create(<decoded command>, null, <startup>) at 89%. The earliest of them sits
# ~1,750 chars from the end of a 7.7 KB body, which is why a 300-char tail --
# the `_bounded_text` cap -- recovers the final call and still loses the link.
LINK_SETUP = "LINK_SETUP_site.Get(config_class)"
LINK_SPAWN = "LINK_SPAWN_site.SpawnInstance_()"
LINK_CONFIG = "LINK_CONFIG_site.ShowWindow = 0"
LINK_CALL = "LINK_CALL_site.Create(decoded_command, null, startup)"
LINK_PARTS = (LINK_SETUP, LINK_SPAWN, LINK_CONFIG, LINK_CALL)
# Kept for the assertions that only need one marker past the cut.
LINK = LINK_CALL
HEAD_MARKER = "HEAD_ONLY_STRUCTURE"


def _large_output_with_trailing_link() -> str:
    """An output over the observation cut whose link sits near the end.

    The link parts are spread across the closing region at the live spacing,
    not bunched at the last few characters: a window that keeps only the final
    call would pass a test built the easy way and still lose the link on a real
    artifact.
    """
    head = HEAD_MARKER + "\n" + ("h" * (MAX_EVIDENCE_CHARS + 500))
    middle = "m" * 1500
    # ~1,750 chars from the end to the earliest link part, matching the live
    # shape; the parts are separated by filler the way real code is.
    tail = "\n".join([
        LINK_SETUP,
        "s" * 150,
        LINK_SPAWN,
        LINK_CONFIG,
        "c" * 600,
        LINK_CALL,
        "t" * 900,
    ])
    return "\n".join([head, middle, tail])


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory(prefix="orbit-linkev-")
        self.addCleanup(self._dir.cleanup)
        tmp = pathlib.Path(self._dir.name)
        artifact = tmp / "artifact.js"
        artifact.write_text("var x = 1;", encoding="utf-8")
        self.source = acquire_analysis_source(artifact, tmp / "owned")
        self.store = EvidenceStore(root=tmp / "evidence")
        self.runtime = AnalysisRuntime(
            backend=object(), source=self.source, evidence_store=self.store
        )
        self.addCleanup(self.runtime.close)

    def _record_action(self, stdout: str):
        """Record one action through the shipped path, returning both records.

        Mirrors `_run_action`: bound the observation, then persist the truncated
        observation and its full raw sibling together, so the store holds
        exactly what a real run leaves.
        """
        from orbit.runtime.analysis_runtime import _bounded_observation

        result = AnalysisResult(
            status="ok", code_sha256="c" * 64, input_sha256="i" * 64,
            stdout=stdout, stderr="", exit_status=0, duration_seconds=0.0,
        )
        observation, truncated, full_chars = _bounded_observation(result)
        call = {"id": "call_1",
                "function": {"name": "execute_analysis", "arguments": "{}"}}
        self.runtime.messages = [{"role": "system", "content": "s"}]
        record, raw_record = self.runtime._record_action_evidence(
            call, result, observation, truncated=truncated, full_chars=full_chars,
        )
        return record, raw_record


class DeliveryTests(_Base):
    def test_the_report_card_carries_a_link_past_the_head_cut(self) -> None:
        record, raw = self._record_action(_large_output_with_trailing_link())
        # Precondition: the observation the step saw is head-only and dropped it.
        self.assertTrue(record.metadata["observation_truncated"])
        observation = self.store.load_raw(record.evidence_id)
        full = self.store.load_raw(raw.evidence_id)
        for part in LINK_PARTS:
            self.assertNotIn(part, observation)  # lost to the head-only cut
            self.assertIn(part, full)            # held by the full sibling
        # The card the report sees now carries the WHOLE sequence: the setup,
        # the configuration and the call that uses them. A window that kept
        # only the last of these would leave the link unexplainable.
        card = self.runtime._evidence_card(record)
        for part in LINK_PARTS:
            self.assertIn(part, card)
        self.assertIn(HEAD_MARKER, card)

    def test_the_card_names_the_full_size_and_the_raw_ref(self) -> None:
        record, raw = self._record_action(_large_output_with_trailing_link())
        card = self.runtime._evidence_card(record)
        # The excerpt must not pass as the whole record.
        self.assertIn("full output", card)
        self.assertIn(f"evidence:{raw.evidence_id}", card)
        self.assertIn("head-and-tail excerpt", card)

    def test_the_card_stays_within_the_per_record_quote_budget(self) -> None:
        """Anchored on the shipped bound, never on the window constants.

        Asserting against `REPORT_LINKING_EXCERPT_*` would be a tautology:
        raising the window raises the assertion with it, so a card that grew
        past the per-record limit would still pass. `MAX_REPORT_EVIDENCE_QUOTE_
        CHARS` is the independent anchor -- it is the bound `_evidence_card`
        checks a record against before quoting it, so the card it then returns
        must honour it too. Sized against the constants alone, the card came
        out ~260 chars over.
        """
        record, _ = self._record_action(_large_output_with_trailing_link())
        card = self.runtime._evidence_card(record)
        self.assertLessEqual(len(card), MAX_REPORT_EVIDENCE_QUOTE_CHARS)

    def test_an_untruncated_record_is_still_quoted_whole(self) -> None:
        """The ordinary case does not change: a small observation is verbatim."""
        body = "status: ok\nstdout:\nsmall complete output " + LINK
        record, _ = self._record_action(body)
        self.assertFalse(record.metadata["observation_truncated"])
        card = self.runtime._evidence_card(record)
        # Quoted whole -- not routed through the head-and-tail path.
        self.assertNotIn("head-and-tail excerpt", card)
        self.assertIn(LINK, card)


class InvariantTests(_Base):
    def test_removing_the_link_from_the_source_does_not_fabricate_it(self) -> None:
        """A card reflects the store, never a fixed expectation.

        If the artifact's output never contained the link, the card must not
        contain it either -- the delivery mechanism must not synthesise a
        conclusion the evidence does not carry.
        """
        no_link = _large_output_with_trailing_link().replace(LINK, "nn" * 20)
        record, raw = self._record_action(no_link)
        self.assertNotIn(LINK, self.store.load_raw(raw.evidence_id))
        card = self.runtime._evidence_card(record)
        self.assertNotIn(LINK, card)

    def test_the_full_raw_body_stays_re_attestable(self) -> None:
        record, raw = self._record_action(_large_output_with_trailing_link())
        # The card excerpts; the store still holds the complete bytes exactly.
        exact = self.store.reattest_exact(raw.evidence_id)
        self.assertIsNotNone(exact)
        self.assertIn(LINK, exact)
        self.assertIn(HEAD_MARKER, exact)

    def test_without_a_raw_sibling_the_card_never_claims_full_output(self) -> None:
        """No raw id: fall back, never quote the observation AS the full body.

        A fallback that reached for the record's own id would quote the
        head-truncated observation under a header announcing the full output
        and the size the step did not see -- the card would then be lying about
        its own provenance, which is worse than the excerpt it replaced.
        """
        record, _ = self._record_action(_large_output_with_trailing_link())
        record.metadata.pop("raw_output_evidence_id", None)
        card = self.runtime._evidence_card(record)
        # Not the linking card's header, which announces the full output and
        # the smaller excerpt the step saw. (The observation's own truncation
        # notice says "full output stored in evidence" -- an honest statement
        # about the store, not a claim about what this card holds.)
        self.assertNotIn("full output; the step saw", card)
        self.assertNotIn("head-and-tail excerpt", card)
        self.assertIn(HEAD_MARKER, card)

    def test_a_raw_body_that_fails_re_attestation_is_not_quoted(self) -> None:
        """Present on disk is not enough: the provenance gate has to pass.

        `reattest_exact` checks identity, provenance and the raw ref -- not
        merely that a sidecar exists. Reading the bytes directly would quote
        content the store declines to attest, which is exactly what the
        re-attestation gate exists to prevent.
        """
        record, raw = self._record_action(_large_output_with_trailing_link())
        # A record the store must refuse to re-attest, with its bytes intact.
        self.store.records[raw.evidence_id].metadata["sidecar_status"] = "rewritten"
        self.assertIsNone(self.store.reattest_exact(raw.evidence_id))
        self.assertIn(LINK_CALL, self.store.load_raw(raw.evidence_id))
        card = self.runtime._evidence_card(record)
        self.assertNotIn(LINK_CALL, card)
        self.assertNotIn("head-and-tail excerpt", card)

    def test_falls_back_to_quote_whole_when_the_raw_body_is_gone(self) -> None:
        """No raw sibling to re-attest: the observation is quoted, not dropped."""
        record, raw = self._record_action(_large_output_with_trailing_link())
        # Simulate an unavailable raw record (retention off / sidecar removed).
        (self.store.root / f"{raw.evidence_id}.txt").unlink()
        self.store.raw_cache.pop(raw.evidence_id, None)
        card = self.runtime._evidence_card(record)
        # Head-and-tail path declined; the head-only observation is still quoted
        # (never the crash, never a silent drop).
        self.assertIn(HEAD_MARKER, card)


class ExcerptHelperTests(unittest.TestCase):
    def test_a_trailing_region_survives_head_and_tail(self) -> None:
        # The whole closing sequence, spread the way real code spreads it --
        # not one marker glued to the final character.
        tail = LINK_SETUP + "s" * 400 + LINK_SPAWN + LINK_CONFIG + "c" * 600 + LINK_CALL
        text = "H" * 900 + "M" * 6000 + tail + "T" * 500
        out = _head_and_tail_excerpt(text, head_chars=900, tail_chars=2200)
        for part in LINK_PARTS:
            self.assertIn(part, out)
        self.assertIn("elided", out)

    def test_short_text_is_returned_whole(self) -> None:
        text = "short " + LINK
        out = _head_and_tail_excerpt(text, head_chars=900, tail_chars=2200)
        self.assertEqual(out, text)
        self.assertNotIn("elided", out)

    def test_text_that_fits_is_whole_even_when_longer_than_the_head(self) -> None:
        """The fits-whole branch returns everything, not just the head.

        A body between `head_chars` and `head_chars + tail_chars` needs no
        elision at all, and returning only its head there would drop a trailing
        link from exactly the records small enough that nothing warned it had
        been shortened.
        """
        text = "H" * 1200 + LINK_CALL
        self.assertGreater(len(text), 900)
        self.assertLessEqual(len(text), 900 + 2200)
        out = _head_and_tail_excerpt(text, head_chars=900, tail_chars=2200)
        self.assertEqual(out, text)
        self.assertIn(LINK_CALL, out)
        self.assertNotIn("elided", out)

    def test_the_marker_states_how_much_was_dropped(self) -> None:
        text = "A" * 10000
        out = _head_and_tail_excerpt(text, head_chars=900, tail_chars=2200)
        dropped = 10000 - 900 - 2200
        self.assertIn(f"{dropped} chars elided", out)

    def test_the_head_comes_before_the_tail(self) -> None:
        """Order is meaning: an excerpt that reads end-then-start misleads.

        The reader is being shown where an artifact starts and where it ends.
        Emitting them out of order would present the closing behaviour as the
        opening structure, which is a worse misreading than an elision.
        """
        text = "OPENING_TOKEN" + "m" * 8000 + "CLOSING_TOKEN"
        out = _head_and_tail_excerpt(text, head_chars=900, tail_chars=2200)
        self.assertLess(out.index("OPENING_TOKEN"), out.index("CLOSING_TOKEN"))


if __name__ == "__main__":
    unittest.main()
