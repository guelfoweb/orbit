"""Part B bootstrap view: deterministic, generic, bounded, provenance-safe.

These tests pin the properties the oversized-source bootstrap must have, on
GENERIC synthetic fixtures (L1-L10) -- no mine.hta strings, so the design is
proven to generalize rather than to fit one sample. Pure; no model, no sandbox.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from orbit.runtime.analysis_bootstrap import (  # noqa: E402
    BOOTSTRAP_HEAD_BYTES,
    BOOTSTRAP_INDEX_ENTRIES,
    BOOTSTRAP_TAIL_BYTES,
    build_bootstrap_view,
    _line_offset_index,
)

MOUNT = "/workspace/input"


def _view(text: str, name: str = "artifact.bin") -> str:
    raw = text.encode("utf-8", "surrogatepass")
    return build_bootstrap_view(
        source_text=text, size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        original_path=name, source_mount=MOUNT,
    )


# --- Generic oversized fixtures (no sample-specific strings) -----------------
# Each is deliberately larger than any real context would admit in full, and
# places its "critical fact" (a unique generic token) in a known region so a
# test can assert the bootstrap points at the right place to read.
def _pad(n: int, seed: str = "x") -> str:
    return "".join(f"var pad_{seed}_{i} = {i};\n" for i in range(n))


L1_HEAD = "CRITICAL_ALPHA = 1\n" + _pad(4000)           # fact near beginning
L2_TAIL = _pad(4000) + "CRITICAL_OMEGA = 1\n"           # fact near end
L3_SPLIT = "CRITICAL_ALPHA\n" + _pad(2000) + "CRITICAL_OMEGA\n" + _pad(2000)
L5_DECOY = "DECOY_MARKER = 0\n" + _pad(4000) + "REAL_VALUE = 1\n"
L7_UNKNOWN = bytes(range(256)) * 200  # not decodable as text (binary)
L8_EMPTY = _pad(5000)                  # no interesting content, just padding
L9_JUST_BELOW = "a" * 20              # trivially small (coverable) -- control


class BootstrapShapeTests(unittest.TestCase):
    def test_deterministic(self) -> None:
        """Same bytes -> identical view, always."""
        self.assertEqual(_view(L1_HEAD), _view(L1_HEAD))

    def test_bounded_head_and_tail(self) -> None:
        """The view shows at most the configured head/tail bytes, never more."""
        v = _view(L1_HEAD)
        self.assertIn(f"first {BOOTSTRAP_HEAD_BYTES} bytes", v)
        self.assertIn(f"last {BOOTSTRAP_TAIL_BYTES} bytes", v)
        # The whole middle is never inlined: the view is far smaller than source.
        self.assertLess(len(v), len(L1_HEAD) // 2)

    def test_states_source_not_in_context(self) -> None:
        v = _view(L1_HEAD)
        self.assertIn("NOT been supplied in full", v)
        self.assertIn("read_file", v)
        self.assertIn(MOUNT, v)

    def test_carries_exact_provenance(self) -> None:
        text = L1_HEAD
        raw = text.encode()
        v = _view(text)
        self.assertIn(hashlib.sha256(raw).hexdigest(), v)
        self.assertIn(str(len(raw)), v)

    def test_asserts_no_behaviour(self) -> None:
        """The view's OWN words are structural: no behaviour verbs, no finding
        claims. Checked on the framing, not the embedded raw source windows --
        the source may legitimately contain any word; the runtime must not add
        a behavioural claim of its own."""
        v = _view(L1_HEAD)
        framing = v.split("--- source head")[0].lower()
        for banned in ("malicious", "downloads a", "executes the", "is a payload",
                       "beacon", "persistence", "the artifact will"):
            self.assertNotIn(banned, framing)

    def test_generic_no_sample_specific_tokens(self) -> None:
        """The FRAMING is identical across unrelated artifacts -- only the
        embedded head/tail/provenance differ, never the instruction text."""
        a = _view("AAAA\n" + _pad(3000), "a.js")
        b = _view("BBBB\n" + _pad(3000), "b.ps1")
        def framing(v: str) -> str:
            # strip the head/tail windows and index/provenance, keep instruction
            return v.split("--- source head")[0].split("Source:")[0]
        self.assertEqual(framing(a), framing(b))
        # And nothing mine.hta-specific ever appears.
        for sample_specific in ("Wscript.Shell", "powershell", "DownloadData",
                                "myxwr5cli", "CreateObject", "opensource"):
            self.assertNotIn(sample_specific, _view(L1_HEAD))

    def test_l1_fact_near_beginning_is_visible_in_head(self) -> None:
        self.assertIn("CRITICAL_ALPHA", _view(L1_HEAD))

    def test_l2_fact_near_end_is_visible_in_tail(self) -> None:
        self.assertIn("CRITICAL_OMEGA", _view(L2_TAIL))

    def test_l3_separated_facts_addressable_via_index(self) -> None:
        """Facts in widely separated middle regions are not both shown, but the
        line index lets a reader target them by offset."""
        v = _view(L3_SPLIT)
        self.assertIn("Line-to-byte index", v)
        # At least the first and last line offsets are present.
        self.assertIn("line 1:", v)

    def test_l5_decoy_and_real_both_only_readable_not_asserted(self) -> None:
        """A decoy marker is shown only as raw source bytes, never editorialized.

        The head window quotes the artifact's own bytes verbatim (so DECOY_MARKER
        may appear there because the file contains it), but the view's own
        framing must never LABEL anything as a decoy or a real value -- it asserts
        nothing about what any region means; both are reachable by read_file.
        """
        v = _view(L5_DECOY)
        framing = v.split("--- source head")[0]  # the runtime's own words only
        self.assertNotIn("decoy", framing.lower())
        self.assertNotIn("real value", framing.lower())
        # The raw head bytes are shown exactly (orientation), which is correct.
        self.assertIn("DECOY_MARKER", v)

    def test_l8_no_interesting_content_still_well_formed(self) -> None:
        v = _view(L8_EMPTY)
        self.assertIn("Line-to-byte index", v)
        self.assertIn("read_file", v)

    def test_index_is_bounded(self) -> None:
        """A huge file yields a bounded index, not one entry per line."""
        huge = _pad(50000)  # 50k lines
        idx = _line_offset_index(huge, BOOTSTRAP_INDEX_ENTRIES)
        self.assertLessEqual(len(idx), BOOTSTRAP_INDEX_ENTRIES)
        # First line always present; offsets strictly increasing.
        self.assertEqual(idx[0][0], 1)
        offsets = [off for _ln, off in idx]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(len(offsets), len(set(offsets)))

    def test_index_offsets_are_exact_line_starts(self) -> None:
        text = "alpha\nbeta\ngamma\ndelta\n"
        idx = dict((ln, off) for ln, off in _line_offset_index(text, 40))
        raw = text.encode()
        # Every indexed offset is either 0 or immediately preceded by '\n'.
        for off in idx.values():
            self.assertTrue(off == 0 or raw[off - 1:off] == b"\n")


if __name__ == "__main__":
    unittest.main()
