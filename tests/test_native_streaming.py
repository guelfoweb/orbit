from __future__ import annotations

import unittest

from orbit.native_llama.client import _ControlChannelStreamFilter, _LeadingThoughtLabelFilter, _StopSequenceStreamFilter


class NativeStreamingTests(unittest.TestCase):
    def test_control_channel_filter_removes_empty_thought_block(self) -> None:
        stream = _ControlChannelStreamFilter()

        deltas = stream.write("<|channel>thought\n<channel|>alpha") + stream.finish()

        self.assertEqual("".join(deltas), "alpha")

    def test_control_channel_filter_handles_split_markers(self) -> None:
        stream = _ControlChannelStreamFilter()

        deltas: list[str] = []
        for chunk in ("<|chan", "nel>thought\n", "<chan", "nel|>ciao"):
            deltas.extend(stream.write(chunk))
        deltas.extend(stream.finish())

        self.assertEqual("".join(deltas), "ciao")

    def test_control_channel_filter_discards_incomplete_channel_at_finish(self) -> None:
        stream = _ControlChannelStreamFilter()

        deltas = stream.write("visible<|channel>thought\nhidden") + stream.finish()

        self.assertEqual("".join(deltas), "visible")

    def test_leading_thought_label_filter_suppresses_plain_label_at_start(self) -> None:
        stream = _LeadingThoughtLabelFilter()

        deltas: list[str] = []
        deltas.extend(stream.write("thought\nI was devel"))
        deltas.extend(stream.write("oped by Google DeepMind."))
        deltas.extend(stream.finish())

        self.assertEqual("".join(deltas), "I was developed by Google DeepMind.")

    def test_leading_thought_label_filter_suppresses_thought_preview_label(self) -> None:
        stream = _LeadingThoughtLabelFilter()

        deltas: list[str] = []
        deltas.extend(stream.write("thought preview\nDante Alighieri"))
        deltas.extend(stream.finish())

        self.assertEqual("".join(deltas), "Dante Alighieri")

    def test_leading_thought_label_filter_suppresses_single_letter_reasoning_artifact(self) -> None:
        stream = _LeadingThoughtLabelFilter()

        deltas: list[str] = []
        deltas.extend(stream.write("s\nDante Alighieri was an Italian poet."))
        deltas.extend(stream.finish())

        self.assertEqual("".join(deltas), "Dante Alighieri was an Italian poet.")

    def test_stop_filter_does_not_emit_stop_sequence(self) -> None:
        emitted: list[str] = []
        stream = _StopSequenceStreamFilter(("STOP",), emit=emitted.append)

        deltas: list[str] = []
        deltas.extend(stream.write("alpha ST"))
        deltas.extend(stream.write("OP beta"))
        deltas.extend(stream.finish())

        self.assertEqual("".join(deltas), "alpha ")
        self.assertEqual("".join(emitted), "alpha ")
        self.assertTrue(stream.stopped)

    def test_stop_filter_handles_stop_split_across_small_chunks(self) -> None:
        emitted: list[str] = []
        stream = _StopSequenceStreamFilter(("END",), emit=emitted.append)

        deltas: list[str] = []
        for chunk in ("he", "llo E", "ND ignored"):
            deltas.extend(stream.write(chunk))
        deltas.extend(stream.finish())

        self.assertEqual("".join(deltas), "hello ")
        self.assertEqual("".join(emitted), "hello ")
        self.assertTrue(stream.stopped)

    def test_stop_filter_flushes_when_no_stop_is_found(self) -> None:
        emitted: list[str] = []
        stream = _StopSequenceStreamFilter(("STOP",), emit=emitted.append)

        deltas = stream.write("hello") + stream.write(" world") + stream.finish()

        self.assertEqual("".join(deltas), "hello world")
        self.assertEqual("".join(emitted), "hello world")
        self.assertFalse(stream.stopped)


if __name__ == "__main__":
    unittest.main()
