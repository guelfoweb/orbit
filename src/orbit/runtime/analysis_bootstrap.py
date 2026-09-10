"""A deterministic, generic structural view of a source too large to admit.

When an artifact's source does not fit the model context, COVER refuses it (it
will not present part of a source and call it coverage), and PLAN would
otherwise run with no view of the source at all -- which is how an oversized
artifact produced an empty plan and, before the closure fix, a silent run. This
builds the smallest honest thing that lets PLAN form useful questions and tells
the model the source is there to be read in bounded pieces.

What it is, precisely:

- size, line count, and a container/language hint taken ONLY from the file
  extension and a cheap first-bytes check -- never from interpreting the code;
- a bounded head window and a bounded tail window of the exact source bytes;
- a coarse line index: for a sample of lines, the byte offset where each begins,
  so the model can ask for a specific region by offset;
- an explicit statement that the full source is NOT in context and is readable
  with `read_file(SOURCE_PATH, offset, limit)`, with the artifact's own digest
  as the provenance every requested region is measured against.

What it is NOT, and must never become:

- it names no API, object, language construct, URL or indicator that a specific
  sample might contain -- the text is identical in shape for any oversized
  artifact, so no sample is ever privileged;
- it asserts no behaviour -- a head/tail excerpt is bytes, not a finding;
- it runs nothing and decodes nothing.

Everything here is pure and deterministic: the same bytes always produce the
same view, and the view is bounded by construction so it cannot itself overflow
the PLAN budget.
"""
from __future__ import annotations

# Bounds on what the view shows. Generous enough that a real program's opening
# structure and closing behaviour are both visible, small enough that the whole
# view is a fraction of the PLAN budget even on dense obfuscated content (~1.1
# chars/token measured on this corpus). The head and tail are byte windows of
# the exact source; the middle is never shown here -- it is read on request.
BOOTSTRAP_HEAD_BYTES = 1200
BOOTSTRAP_TAIL_BYTES = 1200
#: How many line-offset entries the coarse index carries at most. The file's
#: lines are sampled evenly, so a 50k-line artifact still yields a bounded index
#: a model can use to name a region, without one entry per line.
BOOTSTRAP_INDEX_ENTRIES = 40

#: Marks the elided middle. The bytes are not here; they are re-readable from the
#: source by offset, which the view says explicitly.
_ELISION = "... [middle not shown here; read any region with read_file] ..."


def _container_hint(filename: str, head: str) -> str:
    """A generic container/language guess from the name and first bytes only.

    Deliberately shallow: it reports what the extension and a cheap structural
    marker suggest, never what the code does. Unknown is an honest answer and
    the common one -- the model decides what the artifact is, from the bytes.
    """
    name = (filename or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    lowered = head.lstrip().lower()
    markers = []
    if ext:
        markers.append(f"extension .{ext}")
    # Structural, not semantic: the presence of a tag or shebang is a fact about
    # the bytes. It says nothing about what any script within does.
    if lowered.startswith("<"):
        markers.append("begins with a markup/tag character")
    if "<script" in head.lower():
        markers.append("contains a <script> element")
    if lowered.startswith("#!"):
        markers.append("begins with a shebang")
    return "; ".join(markers) if markers else "no container markers recognized"


def _line_offset_index(text: str, max_entries: int) -> list[tuple[int, int]]:
    """`(line_number, byte_offset)` for an evenly sampled set of lines.

    Byte offsets, because `read_file` ranges by byte. Line starts are found
    exactly; the sample is evenly spaced so the index is bounded regardless of
    how many lines the file has. Line 1 and the last line are always included so
    the span is honest.
    """
    # Offsets of every line start, computed exactly from the bytes.
    starts = [0]
    encoded = text.encode("utf-8", "surrogatepass")
    for i, byte in enumerate(encoded):
        if byte == 0x0A:  # '\n'
            starts.append(i + 1)
    if starts and starts[-1] == len(encoded):
        starts.pop()  # a trailing newline does not begin a real line
    total = len(starts)
    if total <= max_entries:
        chosen = list(range(total))
    else:
        # Even sample that always keeps the first and last line start.
        step = (total - 1) / (max_entries - 1)
        chosen = sorted({0, total - 1} | {round(k * step) for k in range(max_entries)})
        chosen = chosen[:max_entries]
    return [(idx + 1, starts[idx]) for idx in chosen]


def build_bootstrap_view(
    *,
    source_text: str,
    size_bytes: int,
    sha256: str,
    original_path: str,
    source_mount: str,
) -> str:
    """The structural view for an oversized source. Pure and deterministic.

    `source_text` is the exact decoded artifact (the caller has already decided
    it is text). The view never presents the whole of it: only bounded head and
    tail windows and a coarse line-offset index, plus the instruction to read
    the rest by range.
    """
    encoded = source_text.encode("utf-8", "surrogatepass")
    head_bytes = encoded[:BOOTSTRAP_HEAD_BYTES]
    tail_bytes = encoded[-BOOTSTRAP_TAIL_BYTES:] if len(encoded) > BOOTSTRAP_HEAD_BYTES else b""
    # Decode the windows tolerantly: a byte window can split a multibyte
    # codepoint at its edge, and a replacement char at a boundary is honest
    # about where the window was cut. The windows are for orientation, not for
    # exact quotation -- exact bytes come from read_file.
    head = head_bytes.decode("utf-8", "replace")
    tail = tail_bytes.decode("utf-8", "replace")
    line_count = source_text.count("\n") + (0 if source_text.endswith("\n") else 1) if source_text else 0
    hint = _container_hint(original_path, head)
    index = _line_offset_index(source_text, BOOTSTRAP_INDEX_ENTRIES)
    index_lines = "\n".join(f"  line {ln}: byte offset {off}" for ln, off in index)

    overlap_note = (
        f"\n\n--- source tail (last {len(tail_bytes)} bytes) ---\n{tail}"
        if tail_bytes
        else ""
    )
    return (
        "The artifact source is larger than this context and has NOT been "
        "supplied in full. What follows is a deterministic structural view of "
        "it, not the source itself and not an analysis of it.\n"
        f"Source: {size_bytes} bytes, {line_count} lines, sha256 {sha256}. "
        f"Container markers: {hint}.\n"
        "You can read any region of the exact source on demand by running "
        f"`orbit_tools.read_file({source_mount!r}, offset=<byte>, limit=<bytes>)` "
        "inside execute_analysis; every read is the exact bytes of this same "
        f"artifact (sha256 {sha256}). Reading a region you have not yet seen is "
        "itself a legitimate step -- prefer it to assuming what the unseen "
        "middle contains.\n"
        "Line-to-byte index (evenly sampled), so you can target a region:\n"
        f"{index_lines}\n\n"
        f"--- source head (first {len(head_bytes)} bytes) ---\n{head}\n"
        f"{_ELISION}{overlap_note}"
    )
