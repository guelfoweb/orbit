"""Render a finished report for a terminal, from the whole text at once.

A report is Markdown because that is what it is stored and shared as. On a
terminal the markers are noise -- `##` and `**` are instructions to a reader
that a terminal can carry out directly -- so this reads the complete text and
emits the same content with the structure shown rather than spelled.

Two properties matter more than coverage.

The first is that this changes nothing but appearance. `report.text` is the
canonical artifact: APIs, saved sessions, files and pipes all read it, and it
must be byte-identical whatever a terminal happens to do. So nothing here
writes back, and every path returns the input unchanged when styling is not
allowed.

The second is that report text is not trusted. It carries decoded artifact
bytes -- an attacker's choice of characters -- so the text is sanitised before
any escape of ours is added, and never after: adding structure to sanitised
text is safe, sanitising text that already contains our escapes would either
destroy them or, worse, leave a crafted sequence indistinguishable from one.

This is deliberately not a Markdown implementation. It handles the constructs
a report actually contains, line by line, and leaves anything else exactly as
written -- a line this does not recognise is still correct, just unstyled.
"""

from __future__ import annotations

import re

from orbit.terminal.theme import (
    BOLD,
    CYAN,
    DIM,
    RESET,
    YELLOW,
    sanitize_terminal_text,
    supports_ansi,
)

# `## Verified indicators` and `## Deterministic transformations` are written
# by the runtime from evidence, not by the model. They are styled differently
# from narrative headings so a reader can see which half of the report they
# are in without either section's text changing.
_RUNTIME_SECTIONS = ("Verified indicators", "Deterministic transformations")

# The separator after a marker is captured, not merely matched: re-emitting a
# single hardcoded space would collapse aligned lists (`1.  ` in a numbered
# list past nine) and, worse, rewrite an indented code block, where the
# indentation is the content. Rendering must change appearance and nothing
# else, so every run of whitespace comes back exactly as written.
_HEADING = re.compile(r"^(#{1,6})([ \t]+)(.*)$")
_BULLET = re.compile(r"^(\s*)([-*+])([ \t]+)(.*)$")
_NUMBERED = re.compile(r"^(\s*)(\d+[.)])([ \t]+)(.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")
# `key: value` provenance lines the runtime emits inside its own sections.
_FIELD = re.compile(r"^(\s+)([a-z_][a-z0-9_ ]*):(\s.*)$")

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")
# An evidence id as the runtime writes it: `ev_` plus two hex groups. Matched
# so it can be tinted; the id itself is never rewritten.
_EVIDENCE_ID = re.compile(r"\bev_[0-9a-f]{12}_[0-9a-f]{16}\b")


def render_report(text: str, *, force_style: bool | None = None) -> str:
    """The report as a terminal should show it.

    Returns the text sanitised but otherwise unchanged whenever styling is not
    permitted -- NO_COLOR, a dumb terminal, a pipe, a redirect -- so a captured
    or piped report is raw Markdown with no escape in it at all.

    `force_style` exists for tests, which cannot make a StringIO a terminal.
    """
    styled = supports_ansi() if force_style is None else force_style
    safe = sanitize_terminal_text(text, allow_newlines=True)
    if not styled or not safe:
        return safe

    out: list[str] = []
    in_fence = False
    for line in safe.split("\n"):
        if _FENCE.match(line):
            # The fence markers stay: they are how a reader knows where the
            # verbatim block begins and ends, and a terminal has no border.
            in_fence = not in_fence
            out.append(f"{DIM}{line}{RESET}")
            continue
        if in_fence:
            # Verbatim means verbatim. Inline markers inside a code block are
            # code, not formatting.
            out.append(f"{CYAN}{line}{RESET}")
            continue
        out.append(_render_line(line))
    return "\n".join(out)


def _render_line(line: str) -> str:
    heading = _HEADING.match(line)
    if heading:
        hashes, gap, title = heading.groups()
        # Runtime-authored sections are tinted differently from the model's
        # own headings: same text, different voice.
        colour = YELLOW if title.strip() in _RUNTIME_SECTIONS else CYAN
        # The inline spans re-open the heading's own attributes after each
        # reset, so a bold or code span mid-title does not silently end the
        # heading's colour for the rest of the line.
        return f"{colour}{BOLD}{hashes}{gap}{_inline(title, reopen=colour + BOLD)}{RESET}"

    bullet = _BULLET.match(line)
    if bullet:
        indent, marker, gap, rest = bullet.groups()
        return f"{indent}{CYAN}{marker}{RESET}{gap}{_inline(rest)}"

    numbered = _NUMBERED.match(line)
    if numbered:
        indent, marker, gap, rest = numbered.groups()
        return f"{indent}{CYAN}{marker}{RESET}{gap}{_inline(rest)}"

    field = _FIELD.match(line)
    if field:
        indent, name, rest = field.groups()
        return f"{indent}{DIM}{name}:{RESET}{_inline(rest)}"

    return _inline(line)


def _inline(text: str, *, reopen: str = "") -> str:
    """Inline spans, innermost meaning first.

    Code is applied before bold so a marker inside backticks stays literal,
    and evidence ids are tinted last so an id already inside a code span is
    not styled twice.

    `reopen` is what the caller had open around this text. A span closes with
    a full reset, which would otherwise end the caller's attribute too and
    leave the rest of a heading unstyled; re-opening after each span keeps the
    line looking like one line.
    """
    # The markers are kept, not consumed. Removing them would make the
    # rendered text differ from the report by more than colour, and the
    # property worth having is that stripping the escapes gives the canonical
    # text back exactly -- which is what makes "rendering changed nothing but
    # appearance" checkable rather than asserted.
    close = f"{RESET}{reopen}"
    rendered = _CODE.sub(lambda m: f"{CYAN}`{m.group(1)}`{close}", text)
    rendered = _BOLD.sub(lambda m: f"{BOLD}**{m.group(1)}**{close}", rendered)
    # The id is reproduced exactly; only colour is added around it. An
    # indicator or evidence reference that came back altered would be a
    # different fact, which is the one thing rendering must never do.
    return _EVIDENCE_ID.sub(lambda m: f"{DIM}{m.group(0)}{close}", rendered)
