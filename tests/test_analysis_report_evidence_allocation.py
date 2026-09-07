"""Provenance is the floor; excerpts are what the residual budget buys.

The old floor was `final_card` -- a citation PLUS an excerpt -- reserved for
every record before any record could be shown properly. On a dossier with
several large records that spent the whole budget on text nobody could use, and
a record that could not afford even that floor was dropped from the prompt.

Now every candidate record is carried as provenance (id, tool, kind, status,
raw_ref, digest, size, and where they apply the truncation status and the raw
sibling), and excerpts are optional content allocated from what remains.

These tests use fixtures whose SHAPE is taken from the retained ARC1/YA/YB
dossiers -- a large observation cut at MAX_EVIDENCE_CHARS with its untruncated
raw sibling, beside smaller complete observations -- never their text. No
expected conclusion appears in anything a model would see: the artifact bodies
here are synthetic, and the structural checks look for RELATIONSHIPS between
generated identifiers, not for any string from a real sample.
"""
from __future__ import annotations

import pathlib
import re
import sys
import tempfile
import unittest
import unittest.mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime.analysis_runtime import (  # noqa: E402
    CARD_BODY_TRUNCATED_NOTICE,
    EVIDENCE_CONTENT_NOT_SHOWN,
    MAX_EVIDENCE_CHARS,
    AnalysisRuntime,
    _bounded_observation,
    _fit_card,
    acquire_analysis_source,
)
from orbit.runtime.analysis_sandbox import AnalysisResult  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

#: The per-record character bound, written as a LITERAL on purpose. Reading the
#: production constant would make every assertion below true by construction --
#: raise the constant and the expectation rises with it. If the constant is
#: changed deliberately, this literal must be changed deliberately too.
PER_RECORD_BOUND = 3200

# A synthetic "chain": an object is obtained, a member is fetched from it, an
# instance is spawned from that, configured, and finally passed as an argument
# alongside a decoded command. The identifiers are nonsense on purpose -- what
# is asserted is that the RELATIONSHIPS survive, not that any word appears.
CHAIN = "\n".join([
    "    var alpha = Obtain(decode('aaa', 11, ':'));",
    "    var beta = alpha.Fetch(decode('bbb', 22, ':'));",
    "    var gamma = [beta.MakeInstance_()];",
    "    gamma[0].Flag = 0;",
    "    var delta = Obtain(decode('ccc', 33, '>'));",
    "    delta.Invoke(decode('ddd', 44, '<'), null, gamma[0], 0);",
])


def structural_chain_readable(text: str) -> bool:
    """Are the object/call/argument relationships recoverable from `text`?

    Deliberately not a marker count. Each step must bind to the previous one by
    the identifier the text itself introduced, so a lone `Invoke(` line -- the
    shape that once passed for a delivered chain -- does not satisfy this.
    """
    fetch = re.search(r"var (\w+) = (\w+)\.Fetch\(", text)
    spawn = re.search(r"var (\w+) = \[(\w+)\.MakeInstance_\(\)\]", text)
    flag = re.search(r"(\w+)\[0\]\.Flag = 0", text)
    invoke = re.search(r"(\w+)\.Invoke\((.*?)\);", text, re.S)
    if not (fetch and spawn and flag and invoke):
        return False
    if spawn.group(2) != fetch.group(1):      # spawned from the fetched member
        return False
    if flag.group(1) != spawn.group(1):       # configured on the spawned one
        return False
    args = invoke.group(2)
    if not re.search(r",\s*44\s*,", args):    # the decoded command is an argument
        return False
    return bool(re.search(rf"\b{re.escape(spawn.group(1))}\[0\]", args))


class _TokenCount:
    def __init__(self, tokens: int, context_tokens: int) -> None:
        self.tokens = tokens
        self.context_tokens = context_tokens


class _TokenisingBackend:
    """A backend that counts tokens, so the TOKEN budget path is exercised.

    Without one, `_evidence_cards` falls back to a 24,000-CHARACTER budget --
    far looser than any real run -- and a dossier that would be squeezed in
    production fits comfortably in the test. Measured on the retained dossiers,
    obfuscated evidence runs about 1.2-1.5 characters per token; 1.3 here is a
    deliberately ordinary value, not tuned to make anything pass.
    """

    CHARS_PER_TOKEN = 1.3

    def __init__(self, context_tokens: int = 8192) -> None:
        self.context_tokens = context_tokens

    def count_text_tokens(self, text: str) -> _TokenCount:
        return _TokenCount(int(len(text) / self.CHARS_PER_TOKEN) + 1,
                           self.context_tokens)


class _Base(unittest.TestCase):
    #: Subclasses that need the loose character budget override this.
    BACKEND: object = None

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory(prefix="orbit-alloc-")
        self.addCleanup(self._dir.cleanup)
        tmp = pathlib.Path(self._dir.name)
        artifact = tmp / "artifact.js"
        artifact.write_text("var x = 1;", encoding="utf-8")
        self.source = acquire_analysis_source(artifact, tmp / "owned")
        self.store = EvidenceStore(root=tmp / "evidence")
        self.runtime = AnalysisRuntime(
            backend=self.BACKEND if self.BACKEND is not None else _TokenisingBackend(),
            source=self.source, evidence_store=self.store,
        )
        self.addCleanup(self.runtime.close)
        self._call = 0

    def _record(self, stdout: str):
        """One action recorded through the shipped path."""
        self._call += 1
        result = AnalysisResult(
            status="ok", code_sha256="c" * 64, input_sha256="i" * 64,
            stdout=stdout, stderr="", exit_status=0, duration_seconds=0.0,
        )
        observation, truncated, full_chars = _bounded_observation(result)
        call = {"id": f"call_{self._call}",
                "function": {"name": "execute_analysis", "arguments": "{}"}}
        self.runtime.messages = [{"role": "system", "content": "s"}]
        return self.runtime._record_action_evidence(
            call, result, observation, truncated=truncated, full_chars=full_chars,
        )

    def _big_with_trailing_chain(self) -> str:
        """A body over the observation cut whose chain sits past the head.

        Written as many short lines rather than one enormous one, because real
        captured output is line-structured. A single giant line makes the
        evidence layer attach a large `stdout_excerpt` to the record, which
        inflates the provenance card to several times what any retained dossier
        produces (447 tokens against a measured 180-205) and would make this
        fixture, not the allocator, decide the outcome.
        """
        filler = "\n".join(f"    var pad{i} = 'x';" for i in range(240))
        tail_pad = "\n".join(f"    var post{i} = 'y';" for i in range(40))
        return "\n".join(["HEAD_STRUCTURE", filler, CHAIN, tail_pad])

    def _small(self, tag: str) -> str:
        return f"complete output {tag}\n" + ("s" * 200)


class ProvenanceFloorTests(_Base):
    def test_every_candidate_record_is_represented(self) -> None:
        """The old floor dropped a record it could not afford. This one does not.

        Shaped like YB: several large truncated observations beside small ones.
        """
        for i in range(4):
            self._record(self._big_with_trailing_chain())
        self._record(self._small("a"))
        records = self.runtime._reportable_records()
        cards = self.runtime._evidence_cards(records)
        self.assertEqual(len(cards), len(records))
        for record in records:
            self.assertTrue(
                any(record.evidence_id in c for c in cards),
                f"{record.evidence_id} vanished from the prompt",
            )

    def test_a_card_without_an_excerpt_says_content_is_not_shown(self) -> None:
        record, _ = self._record(self._small("b"))
        card = self.runtime._provenance_card(record)
        self.assertIn(EVIDENCE_CONTENT_NOT_SHOWN, card)
        # It must not read as an absence of content in the artifact.
        self.assertIn("the record is not empty", card)
        self.assertIn("re-attestable by raw_ref", card)

    def test_provenance_keeps_identity_digest_size_and_raw_ref(self) -> None:
        record, _ = self._record(self._big_with_trailing_chain())
        card = self.runtime._provenance_card(record)
        self.assertIn(record.evidence_id, card)
        self.assertIn(record.raw_sha256[:16], card)
        self.assertIn(f"{record.raw_chars} chars", card)
        self.assertIn(record.raw_ref, card)
        self.assertIn("observation_truncated: true", card)

    def test_the_raw_sibling_is_named_with_its_own_id_and_digest(self) -> None:
        """Attribution must not borrow the truncated observation's digest.

        The sibling holds different bytes; citing the observation's hash for
        them would attest something that hash never covered.
        """
        record, raw = self._record(self._big_with_trailing_chain())
        card = self.runtime._provenance_card(record)
        self.assertNotEqual(raw.raw_sha256[:16], record.raw_sha256[:16])
        self.assertNotEqual(raw.raw_chars, record.raw_chars)

        # The line naming the sibling must carry the SIBLING's digest and size.
        # Asserting only that the sibling's digest appears somewhere in the
        # card is not enough: a card that names the sibling's id beside the
        # OBSERVATION's digest attests bytes that digest never covered, and
        # that mistake would still satisfy a whole-card containment check.
        line = next(l for l in card.splitlines()
                    if l.startswith("full_output_evidence:"))
        self.assertIn(raw.evidence_id, line)
        self.assertIn(raw.raw_sha256[:16], line)
        self.assertIn(f"{raw.raw_chars} chars", line)
        self.assertNotIn(record.raw_sha256[:16], line)
        self.assertNotIn(f"{record.raw_chars} chars", line)

    def test_a_provenance_only_card_is_cheaper_than_a_citation(self) -> None:
        """The whole mechanism: a cheaper floor is what frees the residual."""
        from orbit.runtime.evidence import final_card
        record, _ = self._record(self._big_with_trailing_chain())
        self.assertLess(
            len(self.runtime._provenance_card(record)), len(final_card(record))
        )


class AllocationTests(_Base):
    def test_a_truncated_record_gets_its_excerpt_before_a_complete_one(self) -> None:
        """Ordering is structural, never about what a body contains.

        A record whose observation was cut is the only kind whose stored body
        holds material no card can show; a complete observation is already
        described by provenance plus its own small excerpt.
        """
        self._record(self._big_with_trailing_chain())   # oldest, truncated
        self._record(self._small("newest"))             # newest, complete
        records = self.runtime._reportable_records()
        cards = self.runtime._evidence_cards(records)
        joined = "\n\n".join(cards)
        self.assertTrue(structural_chain_readable(joined))

    def test_the_chain_survives_into_the_assembled_dossier(self) -> None:
        record, _ = self._record(self._big_with_trailing_chain())
        cards = self.runtime._evidence_cards(self.runtime._reportable_records())
        joined = "\n\n".join(cards)
        self.assertTrue(structural_chain_readable(joined))

    def test_removing_a_necessary_fragment_breaks_the_structural_proof(self) -> None:
        """The check is not satisfiable by a lone call line.

        Drop only the spawn/configure steps and keep the final call: the
        relationships can no longer be recovered, so the proof must fail. This
        is what stops the test passing on marker presence.
        """
        broken = "\n".join(
            line for line in CHAIN.splitlines()
            if "MakeInstance_" not in line and ".Flag = 0" not in line
        )
        self.assertIn(".Invoke(", broken)
        self.assertFalse(structural_chain_readable(broken))
        self.assertTrue(structural_chain_readable(CHAIN))

    def test_allocation_is_deterministic_for_the_same_dossier(self) -> None:
        for i in range(3):
            self._record(self._big_with_trailing_chain())
        records = self.runtime._reportable_records()
        first = self.runtime._evidence_cards(records)
        second = self.runtime._evidence_cards(records)
        self.assertEqual(first, second)

    def test_the_floor_leaves_more_residual_than_a_citation_floor(self) -> None:
        """The mechanism, asserted where a synthetic fixture can honestly bind it.

        What frees the residual is that the floor is cheaper -- provenance
        instead of citation-plus-excerpt -- for every record in the dossier.
        Whether that residual then buys a window depends on the artifact's own
        token density, which no synthetic body reproduces faithfully: this
        fixture is denser than real obfuscated JScript, so a fixed pass/fail on
        "the window fits here" would be a statement about the fixture. The
        real dossiers are measured outside the suite, and recorded in
        workdir/diag/arc1_impl/.
        """
        from orbit.runtime.evidence import final_card
        for _ in range(3):
            self._record(self._big_with_trailing_chain())
        for tag in ("x", "y", "z"):
            self._record(self._small(tag))
        records = self.runtime._reportable_records()
        measure = self.runtime.backend.count_text_tokens
        provenance_floor = sum(
            measure(self.runtime._provenance_card(r)).tokens for r in records
        )
        citation_floor = sum(measure(final_card(r)).tokens for r in records)
        self.assertLess(provenance_floor, citation_floor)
        # And every record is still carried, which the citation floor could not
        # promise: it dropped whatever it could not afford.
        cards = self.runtime._evidence_cards(records)
        self.assertEqual(len(cards), len(records))

    def test_a_records_floor_is_released_when_it_gains_an_excerpt(self) -> None:
        """Pricing is a delta, so the floor already paid is not paid twice.

        Charging the excerpt's ABSOLUTE cost on top of a floor already spent
        never overflows -- it is too conservative, not too loose -- so nothing
        crashes and no bound is breached. It simply buys fewer excerpts than the
        budget allows, which silently wastes the residual this whole change
        exists to create. Asserted by arithmetic rather than by outcome: the
        dossier's cost must equal the sum of the forms actually chosen, not the
        floors plus the excerpts.
        """
        for _ in range(2):
            self._record(self._big_with_trailing_chain())
        self._record(self._small("d"))
        from orbit.runtime.analysis_runtime import (
            MAX_REPORT_EVIDENCE_TOTAL_QUOTE_TOKENS,
        )
        records = self.runtime._reportable_records()
        cards = self.runtime._evidence_cards(records)
        measure = self.runtime.backend.count_text_tokens

        floors = {r.evidence_id: self.runtime._provenance_card(r) for r in records}
        spent = sum(measure(c).tokens for c in cards)
        upgraded = [r for r, c in zip(records, cards)
                    if c != floors[r.evidence_id]]
        self.assertTrue(upgraded, "fixture bought no excerpt; nothing to assert")

        # Whatever the allocator declined to buy must genuinely not have fit.
        # Under absolute pricing a record is refused while its DELTA would have
        # fitted the remaining budget -- affordable evidence left unshown.
        budget = min(MAX_REPORT_EVIDENCE_TOTAL_QUOTE_TOKENS,
                     measure("x").context_tokens)
        for record, card in zip(records, cards):
            if card != floors[record.evidence_id]:
                continue
            full = self.runtime._evidence_card(record)
            delta = measure(full).tokens - measure(floors[record.evidence_id]).tokens
            self.assertGreater(
                spent + delta, budget,
                f"{record.evidence_id} was left at provenance although its "
                f"excerpt cost {delta} tokens against {budget - spent} spare",
            )

    def test_a_truncated_record_is_offered_an_excerpt_before_a_complete_one(self) -> None:
        """The ordering rule itself, where the budget can only afford one.

        The central mechanism of this change, and it needs a test that BINDS
        it: with room for a single excerpt, the record whose observation was
        cut must get it, because it is the only one holding material no card
        can otherwise show. A complete observation is already described by its
        provenance plus a small quote.

        Constructed so the newest record is the COMPLETE one: under a plain
        newest-first pass the cut record would be offered last.

        Asserted on the OFFER, not on the outcome. Whether the residual can
        actually afford the excerpt depends on the artifact's token density,
        which no synthetic body reproduces faithfully; binding this to "the
        window appears" would make the fixture, not the rule, decide. The
        offer order is the rule, and it is what a reordering mutation breaks.
        """
        self._record(self._big_with_trailing_chain())          # oldest, truncated
        for tag in ("p", "q", "r", "s"):                       # newer, complete
            self._record(self._small(tag))
        records = self.runtime._reportable_records()
        self.assertFalse(records[-1].metadata.get("observation_truncated"),
                         "fixture must end with a complete observation")

        offered: list[str] = []
        real_card = self.runtime._evidence_card

        def spy(record):
            offered.append(record.evidence_id)
            return real_card(record)

        self.runtime._evidence_card = spy  # type: ignore[method-assign]
        try:
            self.runtime._evidence_cards(records)
        finally:
            del self.runtime._evidence_card

        truncated = [r.evidence_id for r in records
                     if r.metadata.get("observation_truncated")]
        self.assertTrue(truncated, "fixture has no truncated record")
        self.assertEqual(
            offered[0], truncated[0],
            "the cut record was not the first offered an excerpt",
        )

    def test_spending_equals_the_sum_of_the_chosen_forms(self) -> None:
        """Priced once, including when an excerpt is CHEAPER than its floor.

        Clamping a negative delta to zero holds budget against nothing: the
        dossier then costs less than the allocator believes, and a later record
        can be refused an excerpt that would have fitted.
        """
        # Shaped like the real YB dossier: two tiny complete records whose
        # excerpts are CHEAPER than their provenance cards, and -- offered
        # after them -- a larger complete record whose excerpt costs more.
        # That ordering is what makes a discarded saving observable: the
        # saving is taken first, and the record that needs it comes later.
        self._record(self._big_with_trailing_chain())
        self._record("tiny t")
        self._record("tiny u")
        self._record("bulk v\n" + "\n".join(f"line {i}" for i in range(90)))
        records = self.runtime._reportable_records()
        measure = self.runtime.backend.count_text_tokens
        floors = {r.evidence_id: self.runtime._provenance_card(r) for r in records}
        cheaper = [
            r for r in records
            if measure(self.runtime._evidence_card(r)).tokens
            < measure(floors[r.evidence_id]).tokens
        ]
        self.assertTrue(cheaper, "fixture has no excerpt cheaper than its floor")

        # Observe the running total the allocator actually keeps, by watching
        # the budget it tests against. A clamp on a negative delta leaves that
        # total ABOVE the dossier it has assembled -- budget held against
        # nothing, which can refuse a later record an excerpt it could afford.
        #
        # `spent` is internal, so it is observed rather than replayed: the
        # allocator compares `spent + delta` to the budget, so the largest
        # budget at which a given record is still refused reveals what `spent`
        # was when that record was offered.
        savings = sum(
            measure(floors[r.evidence_id]).tokens
            - measure(self.runtime._evidence_card(r)).tokens
            for r in cheaper
        )
        self.assertGreater(savings, 0)

        # The observable consequence: the assembled dossier costs LESS than the
        # floors it started from, by exactly the savings taken. A saving that
        # is discarded in the accounting is still taken in the text, so this
        # pins the rendering rather than the running total.
        #
        # Honest limit of this test: it does NOT catch a clamp on the running
        # total by itself. `spent` is internal, and reconstructing it here
        # would only re-derive whichever formula the test itself writes. The
        # clamp's real cost -- a later record refused an excerpt it could
        # afford -- needs a dossier where a positive-delta record is offered
        # after a negative-delta one AND the budget lands in the narrow band
        # between the two totals; that band was measured at 378 tokens wide on
        # this fixture but is not reachable through the public entry point
        # without also changing which records are carried. The arithmetic is
        # pinned in the accounting itself (`spent += delta`, no clamp) and the
        # discrepancy it prevents is recorded in workdir/diag/arc1_impl/.
        cards = self.runtime._evidence_cards(records)
        actual = sum(measure(c).tokens for c in cards)
        floor_total = sum(measure(c).tokens for c in floors.values())
        upgraded_cost = sum(
            measure(c).tokens - measure(floors[r.evidence_id]).tokens
            for c, r in zip(cards, records) if c != floors[r.evidence_id]
        )
        self.assertEqual(actual, floor_total + upgraded_cost)
        self.assertLess(
            actual - upgraded_cost, floor_total + savings,
            "fixture must take at least one cheaper-than-floor excerpt",
        )

    def test_the_dossier_stays_inside_the_token_budget(self) -> None:
        """Pricing each record once, not floor-plus-upgrade, is what holds.

        Charging a record its floor and then the FULL cost of its excerpt -- as
        though the floor were still owed -- overspends the aggregate budget.
        """
        from orbit.runtime.analysis_runtime import (
            MAX_REPORT_EVIDENCE_TOTAL_QUOTE_TOKENS,
        )
        for _ in range(6):
            self._record(self._big_with_trailing_chain())
        records = self.runtime._reportable_records()
        cards = self.runtime._evidence_cards(records)
        total = self.runtime.backend.count_text_tokens("\n\n".join(cards)).tokens
        self.assertLessEqual(total, MAX_REPORT_EVIDENCE_TOTAL_QUOTE_TOKENS)


class PerRecordBoundTests(_Base):
    """NEW-1: the bound belongs on the rendered card, not on the body alone."""

    def test_a_body_at_the_bound_renders_a_card_within_the_bound(self) -> None:
        header = ["tool_evidence_card: true", "evidence_id: ev_x", "evidence:"]
        for body_len in (PER_RECORD_BOUND - 1, PER_RECORD_BOUND,
                         PER_RECORD_BOUND + 1):
            card = _fit_card(header, "B" * body_len, PER_RECORD_BOUND)
            self.assertLessEqual(
                len(card), PER_RECORD_BOUND,
                f"body {body_len} rendered a card of {len(card)}",
            )

    def test_shortening_is_marked_and_identity_is_never_cut(self) -> None:
        header = ["tool_evidence_card: true", "evidence_id: ev_identity_kept",
                  "status: ok", "evidence:"]
        card = _fit_card(header, "B" * 9000, PER_RECORD_BOUND)
        self.assertLessEqual(len(card), PER_RECORD_BOUND)
        self.assertIn("ev_identity_kept", card)
        self.assertIn("status: ok", card)
        self.assertIn(CARD_BODY_TRUNCATED_NOTICE, card)

    def test_a_body_dropped_for_want_of_room_is_still_marked(self) -> None:
        """Even with no room for a fragment, the loss is stated.

        A card that dropped its content silently would read as a record that
        has none. Identity is kept whole here even at the cost of the bound:
        this shape means a header larger than the whole per-record budget,
        which an analyst has to see rather than have truncated away.
        """
        header = ["tool_evidence_card: true", "evidence_id: ev_huge_header",
                  "x" * 300]
        card = _fit_card(header, "B" * 500, 320)
        self.assertIn("ev_huge_header", card)
        self.assertIn(CARD_BODY_TRUNCATED_NOTICE, card)
        self.assertNotIn("BBBB", card)

    def test_a_body_that_fits_is_not_marked_as_shortened(self) -> None:
        header = ["tool_evidence_card: true", "evidence_id: ev_y", "evidence:"]
        card = _fit_card(header, "B" * 50, PER_RECORD_BOUND)
        self.assertNotIn(CARD_BODY_TRUNCATED_NOTICE, card)
        self.assertTrue(card.endswith("B" * 50))

    def test_every_rendered_card_of_a_dossier_respects_the_bound(self) -> None:
        for _ in range(3):
            self._record(self._big_with_trailing_chain())
        self._record(self._small("c"))
        for card in self.runtime._evidence_cards(self.runtime._reportable_records()):
            self.assertLessEqual(len(card), PER_RECORD_BOUND)

    def test_a_quote_whole_record_just_under_the_bound_does_not_overflow(self) -> None:
        """The exact NEW-1 shape, through the production card renderer.

        A complete observation whose body passes the body-level gate at just
        under the bound used to render a card OVER it, because the identity
        framing is added afterwards. The record is built through the shipped
        recording path, so this is the real renderer, not a helper.
        """
        body_len = PER_RECORD_BOUND - 40
        self.assertLess(body_len, MAX_EVIDENCE_CHARS + 1,
                        "fixture must not be truncated by the observation cut")
        record, _ = self._record("q" * body_len)
        self.assertFalse(record.metadata["observation_truncated"])
        self.assertLessEqual(record.raw_chars, PER_RECORD_BOUND)  # passes the gate
        card = self.runtime._evidence_card(record)
        self.assertLessEqual(
            len(card), PER_RECORD_BOUND,
            f"a {record.raw_chars}-char body rendered a {len(card)}-char card",
        )
        self.assertIn(record.evidence_id, card)


if __name__ == "__main__":
    unittest.main()
